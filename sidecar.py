"""Per-file configuration via sidecar files.

A sidecar file lets you override selected pipeline settings for a single
audio file. It lives next to the audio file, is named after the audio
file's basename with a ``.swhisper`` extension (e.g. ``interview1.swhisper``
for ``interview1.wav``), and uses the same ``KEY=value`` syntax and the same
``SWHISPER_*`` variable names as the global ``.swhisper.env`` file::

    # interview1.swhisper
    SWHISPER_NUM_SPEAKERS=3            # shorthand: sets min AND max speakers
    SWHISPER_INITIAL_PROMPT="Intervju om arbetet"
    SWHISPER_BACKFILL_IGNORE_WORDS=Balans,Tack

Only whitelisted parameters are honored (see ``SIDECAR_PARAMS`` below).
Settings that are global by nature (paths, tokens, devices, model ids,
pipeline selection, caching) are rejected with an explanatory warning.

Reprocessing semantics:
- ``SWHISPER_FORCE_REPROCESS=true`` in a sidecar re-runs transcription for
  that file only; ``SWHISPER_DIARIZE_FORCE_REPROCESS=true`` re-runs
  diarization for that file only.
- If a sidecar file is newer than the file's existing output (the whisper
  JSON for transcription, the ``.ok`` marker for diarization), the file is
  automatically reprocessed. Set ``SWHISPER_SIDECAR_AUTOREPROCESS=false``
  globally to disable this.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SIDECAR_SUFFIX = ".swhisper"
LOG_PREFIX = "📎 [sidecar]"

# Targets that a parameter can be applied to
TARGET_TRANSCRIPTION = "transcription"   # TranscriptionConfig
TARGET_WHISPER = "whisper"               # WhisperSettings
TARGET_DIARIZATION = "diarization"       # DiarizationConfig

# Pipeline stages (used for staleness detection)
STAGE_TRANSCRIBE = "transcribe"
STAGE_DIARIZE = "diarize"


# ---------------------------------------------------------------------------
# Value parsers (strict: raise ValueError on garbage so we can warn loudly
# instead of silently falling back like the global env parsing does)
# ---------------------------------------------------------------------------

_TRUE_SET = {"true", "1", "yes", "on"}
_FALSE_SET = {"false", "0", "no", "off"}


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_SET:
        return True
    if lowered in _FALSE_SET:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_float(value: str) -> float:
    return float(value.strip())


def _parse_str(value: str) -> str:
    return value


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_word_list(value: str) -> List[str]:
    """Comma-separated words, normalized to lowercase (backfill ignore list)."""
    return [w.strip().lower() for w in value.split(",") if w.strip()]


# ---------------------------------------------------------------------------
# Parameter whitelist
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SidecarParam:
    env_key: str
    target: str                       # TARGET_* constant
    attr: str                         # attribute name on the target object
    parse: Callable[[str], Any]
    stage: str                        # STAGE_* constant (which stage it affects)
    note: str = ""


def _params() -> List[SidecarParam]:
    P = SidecarParam
    return [
        # --- Convenience shorthand (applied to DiarizationConfig min+max) ---
        P("SWHISPER_NUM_SPEAKERS", TARGET_DIARIZATION, "__num_speakers__", _parse_int,
          STAGE_DIARIZE, "shorthand: sets both min_speakers and max_speakers"),

        # --- Diarization: speaker counts ---
        P("SWHISPER_MIN_SPEAKERS", TARGET_DIARIZATION, "min_speakers", _parse_int, STAGE_DIARIZE),
        P("SWHISPER_MAX_SPEAKERS", TARGET_DIARIZATION, "max_speakers", _parse_int, STAGE_DIARIZE),

        # --- Per-file reprocessing ---
        P("SWHISPER_FORCE_REPROCESS", TARGET_TRANSCRIPTION, "force_reprocess", _parse_bool,
          STAGE_TRANSCRIBE, "re-run transcription for this file"),
        P("SWHISPER_DIARIZE_FORCE_REPROCESS", TARGET_DIARIZATION, "force_reprocess", _parse_bool,
          STAGE_DIARIZE, "re-run diarization for this file"),

        # --- Transcription: chunking ---
        P("SWHISPER_CHUNK_DURATION", TARGET_TRANSCRIPTION, "target_chunk_duration", _parse_int, STAGE_TRANSCRIBE),
        P("SWHISPER_MIN_SILENCE", TARGET_TRANSCRIPTION, "min_silence_duration", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_OVERLAP_DURATION", TARGET_TRANSCRIPTION, "overlap_duration", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_MAX_CHUNK_DURATION", TARGET_TRANSCRIPTION, "max_chunk_duration", _parse_int, STAGE_TRANSCRIBE),
        P("SWHISPER_MIN_CHUNK_DURATION", TARGET_TRANSCRIPTION, "min_chunk_duration", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_VAD_GAP_MERGE_THRESHOLD", TARGET_TRANSCRIPTION, "vad_gap_merge_threshold", _parse_float, STAGE_TRANSCRIBE),

        # --- Whisper decoding ---
        P("SWHISPER_LANGUAGE", TARGET_WHISPER, "language", _parse_str, STAGE_TRANSCRIBE),
        P("SWHISPER_TASK", TARGET_WHISPER, "task", _parse_str, STAGE_TRANSCRIBE),
        P("SWHISPER_INITIAL_PROMPT", TARGET_WHISPER, "initial_prompt", _parse_str, STAGE_TRANSCRIBE),
        P("SWHISPER_VAD", TARGET_WHISPER, "vad", _parse_str, STAGE_TRANSCRIBE),
        P("SWHISPER_BEAM_SIZE", TARGET_WHISPER, "beam_size", _parse_int, STAGE_TRANSCRIBE),
        P("SWHISPER_BEST_OF", TARGET_WHISPER, "best_of", _parse_int, STAGE_TRANSCRIBE),
        P("SWHISPER_TEMPERATURE", TARGET_WHISPER, "temperature", _parse_float_list, STAGE_TRANSCRIBE),
        P("SWHISPER_COMPRESSION_THRESHOLD", TARGET_WHISPER, "compression_ratio_threshold", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_LOGPROB_THRESHOLD", TARGET_WHISPER, "logprob_threshold", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_NO_SPEECH_THRESHOLD", TARGET_WHISPER, "no_speech_threshold", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_CONDITION_ON_PREVIOUS", TARGET_WHISPER, "condition_on_previous_text", _parse_bool, STAGE_TRANSCRIBE),
        P("SWHISPER_DETECT_DISFLUENCIES", TARGET_WHISPER, "detect_disfluencies", _parse_bool, STAGE_TRANSCRIBE),
        P("SWHISPER_SEED", TARGET_WHISPER, "seed", _parse_int, STAGE_TRANSCRIBE),
        P("SWHISPER_REFINE_PRECISION", TARGET_WHISPER, "refine_whisper_precision", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_MIN_WORD_DURATION", TARGET_WHISPER, "min_word_duration", _parse_float, STAGE_TRANSCRIBE),
        P("SWHISPER_TRUST_TIMESTAMPS", TARGET_WHISPER, "trust_whisper_timestamps", _parse_bool, STAGE_TRANSCRIBE),
        P("SWHISPER_SUPPRESS_TOKENS", TARGET_WHISPER, "suppress_tokens", _parse_str, STAGE_TRANSCRIBE),
        P("SWHISPER_COMPUTE_CONFIDENCE", TARGET_WHISPER, "compute_word_confidence", _parse_bool, STAGE_TRANSCRIBE),

        # --- Diarization: clustering / pipeline parameters ---
        P("SWHISPER_VBX_THRESHOLD", TARGET_DIARIZATION, "vbx_clustering_threshold", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_VBX_FA", TARGET_DIARIZATION, "vbx_fa", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_VBX_FB", TARGET_DIARIZATION, "vbx_fb", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_COMMUNITY_MIN_DURATION_OFF", TARGET_DIARIZATION, "community_min_duration_off", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_SEGMENTATION_THRESHOLD", TARGET_DIARIZATION, "segmentation_threshold", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_MIN_DURATION_ON", TARGET_DIARIZATION, "min_duration_on", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_MIN_DURATION_OFF", TARGET_DIARIZATION, "min_duration_off", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_CLUSTERING_METHOD", TARGET_DIARIZATION, "clustering_method", _parse_str, STAGE_DIARIZE),
        P("SWHISPER_CLUSTERING_MIN_SIZE", TARGET_DIARIZATION, "clustering_min_cluster_size", _parse_int, STAGE_DIARIZE),
        P("SWHISPER_CLUSTERING_THRESHOLD", TARGET_DIARIZATION, "clustering_threshold", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_EMBEDDING_THRESHOLD", TARGET_DIARIZATION, "embedding_distance_threshold", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_SPEAKER_CHANGE_PENALTY", TARGET_DIARIZATION, "speaker_change_penalty", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_TEMPORAL_CONSTRAINTS", TARGET_DIARIZATION, "use_temporal_constraints", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_USE_EXCLUSIVE_DIARIZATION", TARGET_DIARIZATION, "use_exclusive_speaker_diarization", _parse_bool, STAGE_DIARIZE),

        # --- Diarization: alignment / post-processing ---
        P("SWHISPER_WHISPER_TIMESTAMP_OFFSET", TARGET_DIARIZATION, "whisper_timestamp_offset", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_SMOOTHING_ENABLED", TARGET_DIARIZATION, "smoothing_enabled", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_MIN_SPEAKER_WORDS", TARGET_DIARIZATION, "min_speaker_words", _parse_int, STAGE_DIARIZE),
        P("SWHISPER_PRESERVE_MARKERS", TARGET_DIARIZATION, "preserve_markers", _parse_bool, STAGE_DIARIZE),

        # --- Diarization: silence formatting ---
        P("SWHISPER_MIN_SILENCE_DURATION", TARGET_DIARIZATION, "min_silence_duration", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_INCLUDE_SILENCE", TARGET_DIARIZATION, "include_silence_markers", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_LOG_SILENCE", TARGET_DIARIZATION, "log_silence_gaps", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_SILENCE_LINEBREAK", TARGET_DIARIZATION, "silence_gap_linebreak_threshold", _parse_float, STAGE_DIARIZE),

        # --- Backfill ---
        P("SWHISPER_BACKFILL_ENABLED", TARGET_DIARIZATION, "backfill_missing_turns", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_BACKFILL_OVERLAP", TARGET_DIARIZATION, "backfill_overlap", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_BACKFILL_MIN_DURATION", TARGET_DIARIZATION, "backfill_min_duration", _parse_float, STAGE_DIARIZE),
        P("SWHISPER_BACKFILL_IGNORE_WORDS", TARGET_DIARIZATION, "backfill_ignore_words", _parse_word_list, STAGE_DIARIZE),
        P("SWHISPER_BACKFILL_ENABLE_SENSEVOICE", TARGET_DIARIZATION, "backfill_enable_sensevoice", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_BACKFILL_SENSEVOICE_LANGUAGE", TARGET_DIARIZATION, "backfill_sensevoice_language", _parse_str, STAGE_DIARIZE),

        # --- Output formatting ---
        P("SWHISPER_TSV_WORD_PER_LINE", TARGET_DIARIZATION, "tsv_word_per_line", _parse_bool, STAGE_DIARIZE),
        P("SWHISPER_OUTPUT_PREAMBLE", TARGET_DIARIZATION, "output_preamble", _parse_str, STAGE_DIARIZE),
    ]


SIDECAR_PARAMS: Dict[str, SidecarParam] = {p.env_key: p for p in _params()}

# Keys that exist in .swhisper.env but are deliberately NOT allowed per file,
# with the reason (used for a helpful warning instead of "unknown key").
BLOCKED_KEYS: Dict[str, str] = {
    "HUGGING_FACE_TOKEN": "authentication is global",
    "HUGGINGFACE_ACCESS_TOKEN": "authentication is global",
    "PYANNOTEAI_API_KEY": "authentication is global",
    "PYANNOTE_API_KEY": "authentication is global",
    "SWHISPER_AUDIO_DIR": "directory layout is global (would break file discovery)",
    "SWHISPER_TEMP_DIR": "directory layout is global",
    "SWHISPER_JSON_DIR": "directory layout is global",
    "SWHISPER_OUTPUT_DIR": "directory layout is global",
    "SWHISPER_LOG_DIR": "directory layout is global",
    "SWHISPER_AUDIO_SUBDIR": "directory layout is global",
    "SWHISPER_DIARIZE_AUDIO_DIR": "directory layout is global",
    "SWHISPER_BACKFILL_SNIPPET_DIR": "directory layout is global",
    "SWHISPER_BACKFILL_CACHE_DIR": "directory layout is global",
    "SWHISPER_DIARIZATION_CACHE_DIR": "directory layout is global",
    "SWHISPER_STRICT_MODEL_PATH": "model paths are global",
    "SWHISPER_MODEL": "model is loaded/verified once per run",
    "SWHISPER_MODEL_REVISION": "model is loaded/verified once per run",
    "SWHISPER_BACKFILL_MODEL": "model is loaded/verified once per run",
    "SWHISPER_PIPELINE_MODEL": "diarization pipeline selection is global",
    "SWHISPER_PRECISION_MODEL": "diarization pipeline selection is global",
    "SWHISPER_USE_PRECISION": "diarization backend selection is resolved once per run",
    "SWHISPER_DEVICE": "device selection is global",
    "SWHISPER_DIARIZE_DEVICE": "device selection is global",
    "SWHISPER_BACKFILL_DEVICE": "device selection is global",
    "SWHISPER_BACKFILL_SENSEVOICE_DEVICE": "device selection is global",
    "SWHISPER_BACKFILL_SENSEVOICE_MODEL": "model is loaded once per run",
    "SWHISPER_FILE_LIMIT": "applies to the batch, not a file",
    "SWHISPER_DIARIZATION_FIRST": "pipeline mode is chosen by entry point",
    "SWHISPER_SPEAKER_VAD": "pipeline mode setting, global",
    "SWHISPER_SPEAKER_CHUNKING": "pipeline mode setting, global",
    "SWHISPER_PRESERVE_CHECKPOINTS": "workspace behavior is global",
    "SWHISPER_CHECK_REVISION": "environment check, global",
    "SWHISPER_CHECK_ENVIRONMENT": "environment check, global",
    "SWHISPER_PARAMETER_TESTING": "parameter testing is a global mode",
    "SWHISPER_TEST_OUTPUT_DIR": "parameter testing is a global mode",
    "SWHISPER_TEST_MAX_COMBINATIONS": "parameter testing is a global mode",
    "SWHISPER_TEST_SINGLE_FILE": "parameter testing is a global mode",
    "SWHISPER_DIARIZATION_CACHE_ENABLED": "caching behavior is global",
    "SWHISPER_BACKFILL_CACHE_ENABLED": "caching behavior is global",
    "SWHISPER_BACKFILL_SAVE_AUDIO": "debug output behavior is global",
    "SWHISPER_FP16": "device-dependent, global",
    "SWHISPER_ALIGNMENT_LAYERS": "model internals, global",
    "SWHISPER_VERBOSE": "logging behavior is global",
    "SWHISPER_SIDECAR_AUTOREPROCESS": "controls sidecar behavior itself, global only",
}


# ---------------------------------------------------------------------------
# Sidecar discovery and parsing
# ---------------------------------------------------------------------------

def find_sidecar(directory: str, basename: str) -> Optional[str]:
    """Return path to ``<directory>/<basename>.swhisper`` if it exists."""
    if not directory or not basename:
        return None
    candidate = os.path.join(directory, basename + SIDECAR_SUFFIX)
    return candidate if os.path.isfile(candidate) else None


def parse_sidecar_file(path: str) -> Dict[str, str]:
    """Parse a sidecar file into raw string values.

    Uses the same conventions as ``.swhisper.env``: ``KEY=value`` lines,
    ``#`` comments (full-line, and inline for unquoted values), surrounding
    quotes stripped, and ``\\n``/``\\t``/``\\r`` escapes decoded.
    """
    values: Dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if '#' in value and ('"' not in value and "'" not in value):
            value = value.split('#')[0].strip()
        value = value.strip().strip('"').strip("'")
        value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        if key and value:
            values[key] = value
    return values


# ---------------------------------------------------------------------------
# Applying overrides
# ---------------------------------------------------------------------------

def _classify_and_parse(
    overrides: Dict[str, str],
    sidecar_label: str,
) -> List[Tuple[SidecarParam, Any]]:
    """Validate raw overrides against the whitelist; warn on problems."""
    parsed: List[Tuple[SidecarParam, Any]] = []
    for key, raw in overrides.items():
        param = SIDECAR_PARAMS.get(key)
        if param is None:
            if key in BLOCKED_KEYS:
                print(f"{LOG_PREFIX} ⚠️ {sidecar_label}: '{key}' is not allowed per file ({BLOCKED_KEYS[key]}) — ignored")
            else:
                print(f"{LOG_PREFIX} ⚠️ {sidecar_label}: unknown key '{key}' — ignored")
            continue
        try:
            parsed.append((param, param.parse(raw)))
        except (ValueError, TypeError) as exc:
            print(f"{LOG_PREFIX} ⚠️ {sidecar_label}: could not parse {key}={raw!r} ({exc}) — using global value")

    # Deterministic application order (whitelist order), so that the
    # NUM_SPEAKERS shorthand is always applied before explicit MIN/MAX.
    order = {key: idx for idx, key in enumerate(SIDECAR_PARAMS)}
    parsed.sort(key=lambda pv: order[pv[0].env_key])
    keys_present = {p.env_key for p, _ in parsed}
    if "SWHISPER_NUM_SPEAKERS" in keys_present and keys_present & {"SWHISPER_MIN_SPEAKERS", "SWHISPER_MAX_SPEAKERS"}:
        print(f"{LOG_PREFIX} ⚠️ {sidecar_label}: SWHISPER_NUM_SPEAKERS combined with explicit "
              f"MIN/MAX_SPEAKERS — the explicit values take precedence")
    return parsed


def _expand_special(param: SidecarParam, value: Any) -> List[Tuple[str, str, Any]]:
    """Expand pseudo-attributes into concrete (target, attr, value) triples."""
    if param.attr == "__num_speakers__":
        return [
            (TARGET_DIARIZATION, "min_speakers", value),
            (TARGET_DIARIZATION, "max_speakers", value),
        ]
    return [(param.target, param.attr, value)]


def _apply_parsed(
    parsed: List[Tuple[SidecarParam, Any]],
    targets: Dict[str, Any],
    sidecar_label: str,
) -> List[Tuple[Any, str, Any]]:
    """Apply parsed overrides in place. Returns undo info (obj, attr, old)."""
    undo: List[Tuple[Any, str, Any]] = []
    for param, value in parsed:
        for target_name, attr, val in _expand_special(param, value):
            obj = targets.get(target_name)
            if obj is None:
                # The object for this stage isn't part of this call
                # (e.g. diarization keys during the transcription stage);
                # it will be picked up when that stage runs.
                continue
            if not hasattr(obj, attr):
                print(f"{LOG_PREFIX} ⚠️ {sidecar_label}: {param.env_key} maps to missing attribute '{attr}' — ignored")
                continue
            old = getattr(obj, attr)
            undo.append((obj, attr, copy.deepcopy(old) if isinstance(old, (list, dict)) else old))
            setattr(obj, attr, val)
            print(f"{LOG_PREFIX} {sidecar_label}: {param.env_key} = {val!r} (was {old!r})")
    return undo


@contextmanager
def sidecar_overrides(
    directory: str,
    basename: str,
    *,
    transcription_config: Any = None,
    whisper_settings: Any = None,
    diarization_config: Any = None,
):
    """Context manager: apply sidecar overrides in place, restore on exit.

    Used by the transcription pipelines, where config objects are shared by
    reference across components — mutation propagates everywhere, and the
    restore guarantees the next file starts from the global configuration.
    Yields the sidecar path (or None if there is no sidecar).
    """
    path = find_sidecar(directory, basename)
    if path is None:
        yield None
        return

    label = os.path.basename(path)
    print(f"{LOG_PREFIX} Applying per-file settings from {label}")
    overrides = parse_sidecar_file(path)
    parsed = _classify_and_parse(overrides, label)
    targets = {
        TARGET_TRANSCRIPTION: transcription_config,
        TARGET_WHISPER: whisper_settings,
        TARGET_DIARIZATION: diarization_config,
    }
    undo = _apply_parsed(parsed, targets, label)
    try:
        yield path
    finally:
        for obj, attr, old in reversed(undo):
            setattr(obj, attr, old)
        if undo:
            print(f"{LOG_PREFIX} Restored global settings after {label}")


def apply_sidecar_to_diarization_config(config: Any, directory: str, basename: str) -> Tuple[Any, Optional[str]]:
    """Return a per-file clone of a DiarizationConfig with sidecar overrides.

    Used by the diarization stage, where the config is passed per call and a
    clone is cleaner than mutate-and-restore. Returns ``(config, None)``
    unchanged if there is no sidecar.
    """
    path = find_sidecar(directory, basename)
    if path is None:
        return config, None

    label = os.path.basename(path)
    overrides = parse_sidecar_file(path)
    parsed = _classify_and_parse(overrides, label)
    diar_parsed = [(p, v) for (p, v) in parsed if p.target == TARGET_DIARIZATION]
    if not diar_parsed:
        return config, path

    print(f"{LOG_PREFIX} Applying per-file settings from {label}")
    clone = copy.copy(config)
    # Detach mutable list fields so overrides can't leak into the shared config
    for list_attr in ("backfill_ignore_words", "preserved_markers"):
        current = getattr(clone, list_attr, None)
        if isinstance(current, list):
            setattr(clone, list_attr, list(current))
    _apply_parsed(diar_parsed, {TARGET_DIARIZATION: clone}, label)
    return clone, path


# ---------------------------------------------------------------------------
# Reprocessing decisions (skip-logic integration)
# ---------------------------------------------------------------------------

def _autoreprocess_enabled() -> bool:
    val = os.getenv("SWHISPER_SIDECAR_AUTOREPROCESS")
    if val is None:
        return True
    return val.strip().lower() in _TRUE_SET


def sidecar_requests_reprocess(
    directory: str,
    basename: str,
    artifact_path: str,
    stage: str,
) -> Tuple[bool, str]:
    """Decide whether an already-processed file should be redone.

    Returns ``(True, reason)`` if the sidecar sets the stage's force flag, or
    if the sidecar was modified after the given output artifact was written
    and contains settings relevant to that stage (auto-reprocess; can be
    disabled globally with SWHISPER_SIDECAR_AUTOREPROCESS=false).

    For the diarize stage, transcription keys also count as relevant: if they
    changed, the whisper JSON was regenerated and diarization must follow.
    """
    path = find_sidecar(directory, basename)
    if path is None:
        return False, ""

    try:
        overrides = parse_sidecar_file(path)
    except OSError as exc:
        print(f"{LOG_PREFIX} ⚠️ could not read {path}: {exc}")
        return False, ""

    force_key = "SWHISPER_FORCE_REPROCESS" if stage == STAGE_TRANSCRIBE else "SWHISPER_DIARIZE_FORCE_REPROCESS"
    raw = overrides.get(force_key)
    if raw is not None:
        try:
            if _parse_bool(raw):
                return True, f"{force_key}=true in sidecar"
        except ValueError:
            pass

    if not _autoreprocess_enabled():
        return False, ""

    def _relevant(key: str) -> bool:
        param = SIDECAR_PARAMS.get(key)
        if param is None:
            return False
        if stage == STAGE_DIARIZE:
            return True  # any allowed key implies this stage must rerun
        return param.stage == stage

    if not any(_relevant(k) for k in overrides):
        return False, ""

    try:
        if os.path.exists(artifact_path) and os.path.getmtime(path) > os.path.getmtime(artifact_path):
            return True, "sidecar modified after last processing"
    except OSError:
        return False, ""
    return False, ""


__all__ = [
    "SIDECAR_SUFFIX",
    "SIDECAR_PARAMS",
    "BLOCKED_KEYS",
    "SidecarParam",
    "find_sidecar",
    "parse_sidecar_file",
    "sidecar_overrides",
    "apply_sidecar_to_diarization_config",
    "sidecar_requests_reprocess",
    "STAGE_TRANSCRIBE",
    "STAGE_DIARIZE",
]
