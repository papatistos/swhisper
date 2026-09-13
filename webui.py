#!/usr/bin/env python3
"""swhisper web UI — per-file settings editor and pipeline run control.

Usage::

    conda activate swhisper        # or your environment
    python webui.py [--port 8722] [--audio-dir /path/to/audio] [--no-browser]

Opens http://127.0.0.1:8722 in your default browser (unless --no-browser).

The UI lists the audio files in the configured audio directory with their
processing status, lets you edit per-file settings (stored as ``.swhisper``
sidecar files, see SIDECAR_CONFIG.md) and free-text comments, starts/stops
pipeline runs with live log output, and keeps a history of runs with the
files each run produced (``swhisper_run_history.jsonl`` in the audio dir).

The server binds to localhost only and is meant for local use.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import math
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from path_settings import (
    get_commented_audio_dirs,
    get_path_settings,
    normalize_path_value,
    remember_audio_dir,
)
from sidecar import (
    BLOCKED_KEYS,
    SIDECAR_PARAMS,
    SIDECAR_SUFFIX,
    STAGE_DIARIZE,
    STAGE_TRANSCRIBE,
    find_sidecar,
    parse_sidecar_file,
    sidecar_requests_reprocess,
)
from webui_run import preview_run, inspect_progress
from env_settings import env_snapshot, parse_env, update_env
import transcript_stats

from transcript_render import (
    RenderOptions,
    RTFFormatter,
    TXTFormatter,
    derive_segments,
    DEFAULT_MIN_SILENCE,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:  # pragma: no cover
    sys.exit("❌ The web UI needs fastapi and uvicorn:  conda install -c conda-forge fastapi uvicorn")


# ---------------------------------------------------------------------------
# Config access without importing torch-heavy packages
# ---------------------------------------------------------------------------

def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# transcribe/__init__.py and diarize/__init__.py import torch; load the config
# modules directly instead so the UI starts instantly and works without ML deps.
_tconfig = _load_module("swhisper_webui_tconfig", "transcribe/config.py")
_dconfig = _load_module("swhisper_webui_dconfig", "diarize/config.py")
GLOBAL_ENV_PATH = PROJECT_ROOT / '.swhisper.env'

# Extra defaults that cannot be overridden by individual files.
GLOBAL_EXTRAS = {
    'SWHISPER_MODEL': ('transcription', 'model_str', 'str', 'Whisper model'),
    'SWHISPER_MODEL_REVISION': ('transcription', 'revision', 'str', 'Model revision'),
    'SWHISPER_DEVICE': ('transcription', 'device', 'str', 'Transcription device'),
    'SWHISPER_DIARIZE_DEVICE': ('diarization', 'device', 'str', 'Diarization device'),
    'SWHISPER_FILE_LIMIT': ('transcription', 'file_limit', 'int', 'Files per run'),
    'SWHISPER_JSON_DIR': ('transcription', 'json_dir', 'str', 'Transcription JSON folder'),
    'JSON_INPUT_DIR': ('diarization', 'json_input_dir', 'str', 'Diarization JSON input folder'),
    'SWHISPER_OUTPUT_DIR': ('diarization', 'output_dir', 'str', 'Transcript output folder'),
    'SWHISPER_PRESERVE_CHECKPOINTS': ('transcription', 'preserve_checkpoints', 'bool', 'Preserve checkpoints'),
    'SWHISPER_SPEAKER_VAD': ('transcription', 'use_speaker_segments_as_vad', 'bool', 'Use speaker segments for voice detection'),
    'SWHISPER_SPEAKER_CHUNKING': ('transcription', 'speaker_aware_chunking', 'bool', 'Split audio at speaker boundaries'),
    'SWHISPER_USE_PRECISION': ('diarization', 'use_precision_service', 'bool', 'Use Precision diarization service'),
    'SWHISPER_PRECISION_MODEL': ('diarization', 'precision_pipeline_model', 'str', 'Precision model'),
    'SWHISPER_BACKFILL_MODEL': ('diarization', 'backfill_model', 'str', 'Backfill model'),
    'SWHISPER_BACKFILL_DEVICE': ('diarization', 'backfill_device', 'str', 'Backfill device'),
    'SWHISPER_FP16': ('whisper', 'fp16', 'bool', 'Half precision'),
    'SWHISPER_ALIGNMENT_LAYERS': ('whisper', 'word_alignment_most_top_layers', 'int', 'Alignment layers'),
    'SWHISPER_VERBOSE': ('whisper', 'verbose', 'bool', 'Detailed transcription logging'),
}


def parameter_ui(key):
    labels = {'NUM_SPEAKERS': 'Number of speakers', 'MIN_SPEAKERS': 'Minimum speakers',
              'MAX_SPEAKERS': 'Maximum speakers', 'FORCE_REPROCESS': 'Reprocess transcription',
              'DIARIZE_FORCE_REPROCESS': 'Reprocess diarization', 'VAD': 'Voice activity detection',
              'INITIAL_PROMPT': 'Transcription context', 'CONDITION_ON_PREVIOUS': 'Use previous text as context',
              'OUTPUT_PREAMBLE': 'Transcript introduction'}
    short = key.removeprefix('SWHISPER_')
    result = {'label': labels.get(short, short.replace('_', ' ').capitalize())}
    if short in ('NUM_SPEAKERS', 'MIN_SPEAKERS', 'MAX_SPEAKERS', 'BEAM_SIZE', 'BEST_OF', 'FILE_LIMIT'):
        result['min'] = 1
    if short in ('NO_SPEECH_THRESHOLD',):
        result.update(min=0, max=1)
    if any(part in short for part in ('DURATION', 'OVERLAP_DURATION')):
        result['min'] = 0
    if short == 'TASK':
        result['choices'] = ['transcribe', 'translate']
    return result


def validate_settings(values, specs):
    errors = {}
    for key, raw in values.items():
        spec = specs.get(key)
        if spec is None:
            errors[key] = 'Not an editable setting'
            continue
        try:
            if key in SIDECAR_PARAMS:
                value = SIDECAR_PARAMS[key].parse(raw)
            elif spec['type'] == 'bool':
                value = SIDECAR_PARAMS['SWHISPER_FORCE_REPROCESS'].parse(raw)
            elif spec['type'] == 'int':
                value = int(raw)
            elif spec['type'] == 'float':
                value = float(raw)
            else:
                value = raw
            numeric = value if isinstance(value, list) else [value]
            if any(isinstance(v, float) and not math.isfinite(v) for v in numeric):
                raise ValueError('Must be a finite number')
            if 'min' in spec and value < spec['min']:
                raise ValueError(f'Must be at least {spec["min"]}')
            if 'max' in spec and value > spec['max']:
                raise ValueError(f'Must be at most {spec["max"]}')
            if 'choices' in spec and value not in spec['choices']:
                raise ValueError('Choose ' + ' or '.join(spec['choices']))
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
    low, high = values.get('SWHISPER_MIN_SPEAKERS'), values.get('SWHISPER_MAX_SPEAKERS')
    if low and high and not errors and int(low) > int(high):
        errors['SWHISPER_MAX_SPEAKERS'] = 'Must be at least the minimum speaker count'
    return errors

AUDIO_EXTENSIONS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}

COMMENT_PREFIX = "#;"  # user comment lines inside sidecar files (ignored by parsers)
HISTORY_FILENAME = "swhisper_run_history.jsonl"
UI_RUN_LOG_DIR = "logs"  # persisted console output of runs started from this UI
ACTIVE_RUN_FILENAME = "ui-run-active.json"

PIPELINES: Dict[str, Dict[str, str]] = {
    "transcribe4": {
        "script": "transcribe4.py",
        "label": "Standard pipeline (transcribe → diarize, pyannote 4)",
    },
    "transcribe5": {
        "script": "transcribe5.py",
        "label": "Diarization-first pipeline",
    },
    "diarize4": {
        "script": "diarize4.py",
        "label": "Diarization only (existing whisper JSON)",
    },
}

# Display grouping for the sidecar form (unlisted keys land in "Other")
PARAM_GROUPS: List[Dict[str, Any]] = [
    {"name": "Speakers", "keys": [
        "SWHISPER_NUM_SPEAKERS", "SWHISPER_MIN_SPEAKERS", "SWHISPER_MAX_SPEAKERS"]},
    {"name": "Reprocess", "keys": [
        "SWHISPER_FORCE_REPROCESS", "SWHISPER_DIARIZE_FORCE_REPROCESS"]},
    {"name": "Whisper", "keys": [
        "SWHISPER_LANGUAGE", "SWHISPER_TASK", "SWHISPER_INITIAL_PROMPT", "SWHISPER_VAD",
        "SWHISPER_BEAM_SIZE", "SWHISPER_BEST_OF", "SWHISPER_TEMPERATURE",
        "SWHISPER_COMPRESSION_THRESHOLD", "SWHISPER_LOGPROB_THRESHOLD",
        "SWHISPER_NO_SPEECH_THRESHOLD", "SWHISPER_CONDITION_ON_PREVIOUS",
        "SWHISPER_DETECT_DISFLUENCIES", "SWHISPER_SEED", "SWHISPER_REFINE_PRECISION",
        "SWHISPER_MIN_WORD_DURATION", "SWHISPER_TRUST_TIMESTAMPS",
        "SWHISPER_SUPPRESS_TOKENS", "SWHISPER_COMPUTE_CONFIDENCE"]},
    {"name": "Chunking", "keys": [
        "SWHISPER_CHUNK_DURATION", "SWHISPER_MIN_SILENCE", "SWHISPER_OVERLAP_DURATION",
        "SWHISPER_MAX_CHUNK_DURATION", "SWHISPER_MIN_CHUNK_DURATION",
        "SWHISPER_VAD_GAP_MERGE_THRESHOLD"]},
    {"name": "Diarization tuning", "keys": [
        "SWHISPER_VBX_THRESHOLD", "SWHISPER_VBX_FA", "SWHISPER_VBX_FB",
        "SWHISPER_COMMUNITY_MIN_DURATION_OFF", "SWHISPER_SEGMENTATION_THRESHOLD",
        "SWHISPER_MIN_DURATION_ON", "SWHISPER_MIN_DURATION_OFF",
        "SWHISPER_CLUSTERING_METHOD", "SWHISPER_CLUSTERING_MIN_SIZE",
        "SWHISPER_CLUSTERING_THRESHOLD", "SWHISPER_EMBEDDING_THRESHOLD",
        "SWHISPER_SPEAKER_CHANGE_PENALTY", "SWHISPER_TEMPORAL_CONSTRAINTS",
        "SWHISPER_USE_EXCLUSIVE_DIARIZATION"]},
    {"name": "Alignment & silence", "keys": [
        "SWHISPER_WHISPER_TIMESTAMP_OFFSET", "SWHISPER_SMOOTHING_ENABLED",
        "SWHISPER_MIN_SPEAKER_WORDS", "SWHISPER_PRESERVE_MARKERS",
        "SWHISPER_MIN_SILENCE_DURATION", "SWHISPER_INCLUDE_SILENCE",
        "SWHISPER_LOG_SILENCE", "SWHISPER_SILENCE_LINEBREAK"]},
    {"name": "Backfill", "keys": [
        "SWHISPER_BACKFILL_ENABLED", "SWHISPER_BACKFILL_OVERLAP",
        "SWHISPER_BACKFILL_MIN_DURATION", "SWHISPER_BACKFILL_IGNORE_WORDS",
        "SWHISPER_BACKFILL_ENABLE_SENSEVOICE", "SWHISPER_BACKFILL_SENSEVOICE_LANGUAGE"]},
    {"name": "Output", "keys": [
        "SWHISPER_TSV_WORD_PER_LINE", "SWHISPER_OUTPUT_PREAMBLE"]},
]

# Mouseover help texts for the sidecar form
PARAM_HELP: Dict[str, str] = {
    "SWHISPER_NUM_SPEAKERS": "Exact number of speakers for this file — sets both min and max. Tip from the README: specifying 1–2 more speakers than actually present often works better.",
    "SWHISPER_MIN_SPEAKERS": "Lower bound for the number of speakers pyannote may detect (default 1).",
    "SWHISPER_MAX_SPEAKERS": "Upper bound for the number of speakers. The README recommends 1–2 more than the real number; post-processing often removes the excess.",
    "SWHISPER_FORCE_REPROCESS": "Re-run transcription for this file even though whisper JSON output already exists.",
    "SWHISPER_DIARIZE_FORCE_REPROCESS": "Re-run diarization for this file even though its .ok completion marker exists.",
    "SWHISPER_LANGUAGE": "Language code passed to Whisper (global default: sv). Use e.g. 'en' for an English recording.",
    "SWHISPER_TASK": "'transcribe' keeps the source language; 'translate' translates to English.",
    "SWHISPER_INITIAL_PROMPT": "Text used to prime Whisper — e.g. names, places, or terminology that occur in this recording. Improves spelling of rare words.",
    "SWHISPER_VAD": "Voice activity detection applied before transcription: silero:v3.1, silero:v4.0, auditok, or None (no VAD).",
    "SWHISPER_BEAM_SIZE": "Beam search width (default 5). Higher = potentially better output, slower.",
    "SWHISPER_BEST_OF": "Number of candidates when sampling at non-zero temperature (default 5).",
    "SWHISPER_TEMPERATURE": "Comma-separated fallback temperatures, e.g. 0.0,0.2,0.4. Whisper retries with the next value when decoding fails quality checks.",
    "SWHISPER_COMPRESSION_THRESHOLD": "Decoding counts as failed if the text's gzip compression ratio exceeds this (default 2.4) — a guard against repetitive hallucination.",
    "SWHISPER_LOGPROB_THRESHOLD": "Segments with average log-probability below this count as failed. -1 effectively disables the check.",
    "SWHISPER_NO_SPEECH_THRESHOLD": "Probability above which a segment is dismissed as silence (default 0.6).",
    "SWHISPER_CONDITION_ON_PREVIOUS": "Feed the previous transcript window as context. More consistent style, but errors and hallucinations can propagate.",
    "SWHISPER_DETECT_DISFLUENCIES": "Mark sounds that could not be transcribed as [*] disfluency markers.",
    "SWHISPER_SEED": "Random seed for reproducibility (default 1234).",
    "SWHISPER_REFINE_PRECISION": "Precision (in seconds) for refining word timestamps (default 0.5).",
    "SWHISPER_MIN_WORD_DURATION": "Minimum duration a word may get in the timestamps (default 0.02 s).",
    "SWHISPER_TRUST_TIMESTAMPS": "Trust Whisper's own segment timestamps instead of recomputing them (default false).",
    "SWHISPER_SUPPRESS_TOKENS": "Comma-separated token ids Whisper must not produce (default 50364).",
    "SWHISPER_COMPUTE_CONFIDENCE": "Compute word-level confidence scores. Note: may not work with the KBLab model.",
    "SWHISPER_CHUNK_DURATION": "Target chunk length in seconds for long files (default 180). Boundaries are placed at silences.",
    "SWHISPER_MIN_SILENCE": "Minimum silence duration (s) where a chunk boundary may be placed (default 0.5).",
    "SWHISPER_OVERLAP_DURATION": "Overlap between consecutive chunks in seconds, used to stitch transcripts together (default 1.0).",
    "SWHISPER_MAX_CHUNK_DURATION": "Diarization-first mode: maximum chunk duration before a split is forced (default 200 s).",
    "SWHISPER_MIN_CHUNK_DURATION": "Diarization-first mode: minimum chunk duration, to avoid tiny chunks (default 30 s).",
    "SWHISPER_VAD_GAP_MERGE_THRESHOLD": "Diarization-first mode: merge speaker segments separated by gaps shorter than this (default 0.5 s).",
    "SWHISPER_VBX_THRESHOLD": "VBx clustering threshold for the community-1 model (default 0.6). Lower values create more speakers.",
    "SWHISPER_VBX_FA": "False-alarm probability for speaker activity (default 0.07). Lower = more conservative speech detection.",
    "SWHISPER_VBX_FB": "Probability of not detecting a speaker (default 0.8). Lower = more aggressive speaker separation.",
    "SWHISPER_COMMUNITY_MIN_DURATION_OFF": "Minimum duration of non-speech segments for the community-1 model (default 0.0).",
    "SWHISPER_SEGMENTATION_THRESHOLD": "Legacy 3.1 pipeline: sensitivity to speaker changes in segmentation (default 0.01, lower = more sensitive).",
    "SWHISPER_MIN_DURATION_ON": "Legacy 3.1 pipeline: minimum duration of speech segments (s).",
    "SWHISPER_MIN_DURATION_OFF": "Legacy 3.1 pipeline: minimum duration of non-speech segments (s).",
    "SWHISPER_CLUSTERING_METHOD": "Legacy 3.1 pipeline: clustering linkage method (default centroid).",
    "SWHISPER_CLUSTERING_MIN_SIZE": "Legacy 3.1 pipeline: minimum number of frames to form a speaker cluster (default 15).",
    "SWHISPER_CLUSTERING_THRESHOLD": "Legacy 3.1 pipeline: clustering distance threshold (default 0.15). Lower = stricter clustering.",
    "SWHISPER_EMBEDDING_THRESHOLD": "Speaker embedding distance threshold (default 0.6). Lower = voices must be more similar to be merged into one speaker.",
    "SWHISPER_SPEAKER_CHANGE_PENALTY": "Penalty for frequent speaker changes (default 0.1). Ignored by newer pyannote models.",
    "SWHISPER_TEMPORAL_CONSTRAINTS": "Use temporal constraints during diarization. Ignored by newer pyannote models.",
    "SWHISPER_USE_EXCLUSIVE_DIARIZATION": "Use pyannote 4's exclusive (non-overlapping) diarization stream when available.",
    "SWHISPER_WHISPER_TIMESTAMP_OFFSET": "Shift all Whisper timestamps by this many seconds before alignment (default 0.1 — the README notes a systematic ~0.1 s shift).",
    "SWHISPER_SMOOTHING_ENABLED": "Smooth word-level speaker assignment to reduce spurious single-word speaker flips.",
    "SWHISPER_MIN_SPEAKER_WORDS": "Minimum consecutive words required to accept a speaker change during smoothing (default 3).",
    "SWHISPER_PRESERVE_MARKERS": "Keep [*] disfluency markers in the diarized transcript.",
    "SWHISPER_MIN_SILENCE_DURATION": "Ignore silences shorter than this when writing silence markers (default 0.2 s).",
    "SWHISPER_INCLUDE_SILENCE": "Write silence durations into the transcript in CA notation, e.g. (.3).",
    "SWHISPER_LOG_SILENCE": "Write a separate log file with between-word gap durations.",
    "SWHISPER_SILENCE_LINEBREAK": "Surround silences of at least this many seconds with blank lines (default 1.0; 0 disables). Affects rtf/txt output only.",
    "SWHISPER_BACKFILL_ENABLED": "Re-transcribe speaker turns that got no words. Recovers missed speech but can introduce hallucinations.",
    "SWHISPER_BACKFILL_OVERLAP": "Extra seconds of audio included around each backfilled segment (default 0.5).",
    "SWHISPER_BACKFILL_MIN_DURATION": "Skip backfilling segments shorter than this (default 0.1 s).",
    "SWHISPER_BACKFILL_IGNORE_WORDS": "Comma-separated words to treat as backfill hallucinations (replaced by markers), e.g. Balans.",
    "SWHISPER_BACKFILL_ENABLE_SENSEVOICE": "Run SenseVoice emotion and audio-event detection (laughter, crying, …) on backfilled segments.",
    "SWHISPER_BACKFILL_SENSEVOICE_LANGUAGE": "SenseVoice language: auto, zh, en, yue, ja, or ko.",
    "SWHISPER_TSV_WORD_PER_LINE": "TSV output: one word per line (true) or one segment per line (false).",
    "SWHISPER_OUTPUT_PREAMBLE": "Custom preamble text at the top of transcript files. Use \\n for line breaks.",
}

_TYPE_BY_PARSER = {
    "_parse_bool": "bool",
    "_parse_int": "int",
    "_parse_float": "float",
    "_parse_str": "str",
    "_parse_float_list": "float_list",
    "_parse_word_list": "word_list",
}


class AppState:
    def __init__(self, audio_dir: Optional[str] = None):
        self.audio_dir_override = audio_dir
        self.audio_dir_pinned = audio_dir is not None
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.pipeline: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.log_lines: List[str] = []
        self.stop_requested = False
        self.run_audio_dir: Optional[str] = None
        self.run_log_path: Optional[str] = None
        self.run_id: Optional[str] = None
        self.pid: Optional[int] = None

    # -- configuration ------------------------------------------------------

    def audio_dir(self) -> str:
        if self.audio_dir_override:
            return self.audio_dir_override
        settings = get_path_settings()
        if settings.audio_dir:
            return str(settings.audio_dir)
        raise HTTPException(500, "No audio directory configured (SWHISPER_AUDIO_DIR or --audio-dir).")

    def dirs(self) -> Dict[str, str]:
        audio = self.audio_dir()
        json_dir = os.getenv("SWHISPER_JSON_DIR", "whisper-json-output")
        output_dir = os.getenv("SWHISPER_OUTPUT_DIR", "transcripts")
        return {
            "audio": audio,
            "json": os.path.join(audio, json_dir),
            "output": os.path.join(audio, output_dir),
        }

    # -- run control ---------------------------------------------------------

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        """Return whether *pid* is still alive without owning its process."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _active_run_path(self, audio_dir: str) -> str:
        return os.path.join(audio_dir, UI_RUN_LOG_DIR, ACTIVE_RUN_FILENAME)

    def _audio_dir_candidates(self) -> List[str]:
        candidates: List[str] = []
        try:
            candidates.append(self.audio_dir())
        except HTTPException:
            pass
        if not self.audio_dir_pinned:
            candidates.extend(get_commented_audio_dirs())
        unique: List[str] = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.realpath(candidate)
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _read_active_metadata(self, audio_dir: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._active_run_path(audio_dir), encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        return metadata if isinstance(metadata, dict) else None

    def _write_active_metadata_locked(self) -> None:
        """Atomically publish enough state for a later UI to reattach."""
        if not self.run_audio_dir or not self.run_id or self.pid is None:
            return
        run_dir = os.path.join(self.run_audio_dir, UI_RUN_LOG_DIR)
        os.makedirs(run_dir, exist_ok=True)
        metadata = {
            "run_id": self.run_id,
            "pid": self.pid,
            "pipeline": self.pipeline,
            "script": PIPELINES.get(self.pipeline, {}).get("script"),
            "audio_dir": self.run_audio_dir,
            "started_at": self.started_at,
            "log_path": os.path.relpath(self.run_log_path, self.run_audio_dir)
            if self.run_log_path else None,
            "stop_requested": self.stop_requested,
        }
        path = self._active_run_path(self.run_audio_dir)
        temporary = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        Path(temporary).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _remove_active_metadata_locked(self) -> None:
        if not self.run_audio_dir:
            return
        path = self._active_run_path(self.run_audio_dir)
        try:
            metadata = self._read_active_metadata(self.run_audio_dir)
            if metadata and self.run_id and metadata.get("run_id") != self.run_id:
                return
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def _read_complete_lines(path: Optional[str]) -> List[str]:
        if not path:
            return []
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        lines = content.splitlines()
        # Do not return a line while the child is still in the middle of
        # writing it.  Otherwise the next poll would append that line twice.
        if content and not content.endswith(("\n", "\r")) and lines:
            lines.pop()
        return lines

    def _append_log_line_locked(self, line: str) -> None:
        self.log_lines.append(line)
        if len(self.log_lines) > 20000:
            del self.log_lines[:5000]
        if self.run_log_path:
            try:
                with open(self.run_log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass

    def _adopt_metadata_locked(self, metadata: Dict[str, Any]) -> bool:
        try:
            pid = int(metadata["pid"])
            started_at = float(metadata["started_at"])
        except (KeyError, TypeError, ValueError):
            return False

        audio_dir = str(metadata.get("audio_dir") or self.audio_dir())
        log_path = metadata.get("log_path")
        if not isinstance(log_path, str) or not log_path:
            return False
        candidate = os.path.realpath(os.path.join(audio_dir, log_path))
        try:
            if os.path.commonpath([candidate, os.path.realpath(audio_dir)]) != os.path.realpath(audio_dir):
                return False
        except ValueError:
            return False

        self.process = None
        self.pid = pid
        self.pipeline = metadata.get("pipeline")
        self.started_at = started_at
        self.finished_at = None
        self.returncode = metadata.get("returncode")
        self.stop_requested = bool(metadata.get("stop_requested"))
        self.run_audio_dir = audio_dir
        self.run_log_path = candidate
        self.run_id = str(metadata.get("run_id") or "") or None
        self.log_lines = self._read_complete_lines(self.run_log_path)
        return bool(self.run_id and self.pipeline)

    def running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return self.pid is not None and self._pid_exists(self.pid)

    def log_since(self, offset: int = 0) -> Dict[str, Any]:
        with self.lock:
            path = self.run_log_path
            fallback = list(self.log_lines)
        lines = self._read_complete_lines(path) if path else fallback
        if not lines and fallback and not path:
            lines = fallback
        offset = max(0, min(offset, len(lines)))
        return {"lines": lines[offset:], "next": len(lines)}

    def attach_existing_run(self) -> None:
        """Adopt the active child recorded by a previous web UI instance."""
        finish: Optional[Tuple[str, Optional[int]]] = None
        watcher: Optional[Tuple[int, str]] = None
        with self.lock:
            if self.running():
                return
            active_runs: List[Tuple[float, str, Dict[str, Any]]] = []
            stale_runs: List[Tuple[float, str, Dict[str, Any]]] = []
            for audio_dir in self._audio_dir_candidates():
                metadata = self._read_active_metadata(audio_dir)
                if not metadata:
                    continue
                try:
                    active_pid = int(metadata.get("pid", 0))
                except (TypeError, ValueError):
                    active_pid = 0
                if active_pid and self._pid_exists(active_pid):
                    try:
                        started = float(metadata.get("started_at", 0))
                    except (TypeError, ValueError):
                        started = 0
                    active_runs.append((started, audio_dir, metadata))
                else:
                    try:
                        started = float(metadata.get("started_at", 0))
                    except (TypeError, ValueError):
                        started = 0
                    stale_runs.append((started, audio_dir, metadata))
            selected_runs = active_runs or stale_runs
            if not selected_runs:
                return
            _, audio_dir, metadata = max(selected_runs, key=lambda item: item[0])
            if not self.audio_dir_pinned:
                self.audio_dir_override = audio_dir
            if not self._adopt_metadata_locked(metadata):
                return
            if self.pid is not None and self._pid_exists(self.pid):
                watcher = (self.pid, self.run_id)  # type: ignore[arg-type]
            else:
                finish = (self.run_id, self.returncode)  # type: ignore[arg-type]

        if watcher:
            threading.Thread(
                target=self._watch_attached_process,
                args=watcher,
                daemon=True,
            ).start()
        elif finish:
            self._finish_run(finish[0], finish[1], offline=True)

    def start(self, pipeline: str) -> None:
        if pipeline not in PIPELINES:
            raise HTTPException(400, f"Unknown pipeline '{pipeline}'")
        with self.lock:
            if self.running():
                raise HTTPException(409, "A run is already in progress")
            script = PROJECT_ROOT / PIPELINES[pipeline]["script"]
            audio_dir = self.audio_dir()
            active = self._read_active_metadata(audio_dir)
            if active:
                try:
                    active_pid = int(active.get("pid", 0))
                except (TypeError, ValueError):
                    active_pid = 0
                if active_pid and self._pid_exists(active_pid):
                    raise HTTPException(409, "A run is already in progress")
                try:
                    os.remove(self._active_run_path(audio_dir))
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

            started_at = time.time()
            run_id = uuid.uuid4().hex
            ts = datetime.fromtimestamp(started_at).strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(audio_dir, UI_RUN_LOG_DIR)
            os.makedirs(run_dir, exist_ok=True)
            run_log_path = os.path.join(
                run_dir, f"ui-run-{ts}-{pipeline}-{run_id[:6]}.log")
            header = f"▶ Started {PIPELINES[pipeline]['script']} — audio dir: {audio_dir}"
            env = dict(os.environ, PYTHONUNBUFFERED="1", SWHISPER_AUDIO_DIR=audio_dir)
            with open(run_log_path, "w", encoding="utf-8", buffering=1) as log_file:
                log_file.write(header + "\n")
                log_file.flush()
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script)],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,  # own process group → clean SIGINT
                )

            self.process = process
            self.pid = process.pid
            self.pipeline = pipeline
            self.started_at = started_at
            self.finished_at = None
            self.returncode = None
            self.stop_requested = False
            self.run_audio_dir = audio_dir
            self.run_log_path = run_log_path
            self.run_id = run_id
            self.log_lines = [header]
            self._write_active_metadata_locked()
            threading.Thread(
                target=self._watch_process,
                args=(process, run_id),
                daemon=True,
            ).start()

    def _watch_process(self, process: subprocess.Popen, run_id: str) -> None:
        try:
            returncode = process.wait()
        except OSError:
            returncode = None
        self._finish_run(run_id, returncode)

    def _watch_attached_process(self, pid: int, run_id: str) -> None:
        while self._pid_exists(pid):
            time.sleep(1.0)
        self._finish_run(run_id, None, offline=True)

    def _finish_run(
        self, run_id: str, returncode: Optional[int], offline: bool = False
    ) -> None:
        with self.lock:
            if self.run_id != run_id or self.finished_at is not None:
                return
            self.returncode = returncode
            self.finished_at = time.time()
            if offline and returncode is None:
                status = "— Run ended while the web UI was offline (exit code unavailable) —"
            else:
                status = "⏹ stopped by user" if self.stop_requested else (
                    "✅ finished" if returncode == 0 else f"❌ exited with code {returncode}")
                status = f"— Run {status} —"
            self._append_log_line_locked(status)
            self._remove_active_metadata_locked()
        try:
            self._write_history_entry()
        except Exception as exc:  # history must never break the UI
            with self.lock:
                self._append_log_line_locked(f"⚠️ Could not write run history: {exc}")

    def stop(self) -> None:
        with self.lock:
            if not self.running():
                raise HTTPException(409, "No run in progress")
            self.stop_requested = True
            self._write_active_metadata_locked()
            pid = self.pid
            if pid is None:
                raise HTTPException(409, "The active run has no process ID")
            # SIGINT → pipelines run their KeyboardInterrupt cleanup paths
            try:
                os.killpg(os.getpgid(pid), signal.SIGINT)
            except ProcessLookupError:
                raise HTTPException(409, "The run has already finished") from None

    # -- run history ----------------------------------------------------------

    def _collect_new_files(self, audio_dir: str, since: float) -> List[str]:
        """Files under the audio dir created/modified by the run (mtime-based)."""
        produced: List[str] = []
        for root, dirnames, filenames in os.walk(audio_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename == HISTORY_FILENAME or filename.startswith("."):
                    continue
                path = os.path.join(root, filename)
                try:
                    if os.path.getmtime(path) >= since - 2:
                        produced.append(os.path.relpath(path, audio_dir))
                except OSError:
                    continue
                if len(produced) >= 1000:
                    return sorted(produced)
        return sorted(produced)

    def _write_history_entry(self) -> None:
        audio_dir = self.run_audio_dir
        if not audio_dir or not os.path.isdir(audio_dir):
            return
        with self.lock:
            entry = {
                "pipeline": self.pipeline,
                "script": PIPELINES.get(self.pipeline, {}).get("script"),
                "started": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
                "finished": datetime.fromtimestamp(self.finished_at).isoformat(timespec="seconds"),
                "duration_s": round(self.finished_at - self.started_at, 1),
                "returncode": self.returncode,
                "stopped_by_user": self.stop_requested,
                "success": self.returncode == 0 and not self.stop_requested,
                "audio_dir": audio_dir,
            }
            started_at = self.started_at
        entry["files_produced"] = self._collect_new_files(audio_dir, started_at)
        entry["file_count"] = len(entry["files_produced"])
        history_path = os.path.join(audio_dir, HISTORY_FILENAME)
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with self.lock:
            self._append_log_line_locked(
                f"🗂 Run recorded in {HISTORY_FILENAME} ({entry['file_count']} file(s) produced/updated)")


state = AppState()
app = FastAPI(title="swhisper web UI")


@app.on_event("startup")
async def _attach_active_run() -> None:
    """Reconnect to a transcription that outlived the previous UI process."""
    state.attach_existing_run()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _global_defaults() -> Dict[str, Any]:
    """Current effective global values for every sidecar param."""
    tc = _tconfig.TranscriptionConfig()
    ws = _tconfig.WhisperSettings()
    dc = _dconfig.DiarizationConfig()
    objs = {"transcription": tc, "whisper": ws, "diarization": dc}
    out: Dict[str, Any] = {}
    for key, param in SIDECAR_PARAMS.items():
        if param.attr.startswith("__"):
            out[key] = None
            continue
        obj = objs[param.target]
        value = getattr(obj, param.attr, None)
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        out[key] = value
    return out


def _basenames(audio_dir: str) -> List[str]:
    names: List[str] = []
    try:
        entries = sorted(os.listdir(audio_dir))
    except FileNotFoundError:
        raise HTTPException(500, f"Audio directory not found: {audio_dir}")
    for name in entries:
        base, ext = os.path.splitext(name)
        if ext.lower() in AUDIO_EXTENSIONS and base not in names:
            names.append(base)
    return names


def _safe_base(audio_dir: str, basename: str) -> str:
    """Reject path tricks; the basename must resolve inside the audio dir."""
    if not basename or basename != os.path.basename(basename) or basename.startswith("."):
        raise HTTPException(400, "Invalid file name")
    resolved = os.path.realpath(os.path.join(audio_dir, basename + SIDECAR_SUFFIX))
    if os.path.commonpath([resolved, os.path.realpath(audio_dir)]) != os.path.realpath(audio_dir):
        raise HTTPException(400, "Invalid file name")
    return basename


def _read_comment(path: str) -> str:
    """Extract user comment lines (prefixed ``#;``) from a sidecar file."""
    lines: List[str] = []
    try:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if raw.startswith(COMMENT_PREFIX):
                lines.append(raw[len(COMMENT_PREFIX):].lstrip())
    except OSError:
        pass
    return "\n".join(lines)


_TRANSCRIPT_TS_RE = re.compile(r"^_(\d{8}-\d{6})$")
# Older runs used a different log timestamp format, e.g. "_2025-06-11 145137"
_LOG_TS_RES = [
    re.compile(r"^_(\d{8}-\d{6})$"),
    re.compile(r"^_(\d{4}-\d{2}-\d{2} \d{6})$"),
]


def _transcript_dirs(dirs: Dict[str, str]) -> List[str]:
    """Directories that may contain TXT transcripts for the current audio dir."""
    return [
        os.path.join(dirs["output"], "txt"),               # diarize pipeline (transcribe4/diarize4)
        os.path.join(dirs["json"], "transcripts", "txt"),  # transcribe5 diarization-first pipeline
    ]


def _transcript_versions(dirs: Dict[str, str], base: str) -> List[Dict[str, Any]]:
    """All TXT transcript versions for *base*, newest (by mtime) first.

    Pipelines write ``<base>_<YYYYMMDD-HHMMSS>.txt``; a plain ``<base>.txt``
    is accepted as an untimestamped fallback.
    """
    versions: List[Dict[str, Any]] = []
    for txt_dir in _transcript_dirs(dirs):
        try:
            names = os.listdir(txt_dir)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".txt"):
                continue
            stem = name[:-4]
            if stem == base:
                ts = None
            elif stem.startswith(base + "_"):
                m = _TRANSCRIPT_TS_RE.match(stem[len(base):])
                if not m:
                    continue
                ts = m.group(1)
            else:
                continue
            path = os.path.join(txt_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            versions.append({
                # id doubles as the ?version= parameter and is always an
                # exact match against this list, so no path traversal.
                "id": os.path.relpath(path, dirs["audio"]),
                "filename": name,
                "timestamp": ts,
                "mtime": st.st_mtime,
                "size": st.st_size,
            })
    versions.sort(key=lambda v: v["mtime"], reverse=True)
    return versions


def _log_dirs(dirs: Dict[str, str]) -> List[str]:
    """Directories that may contain .log files for the current audio dir."""
    return [
        os.path.join(dirs["output"], "logs"),               # diarize pipeline
        os.path.join(dirs["json"], "transcripts", "logs"),  # transcribe5 location
    ]


def _log_versions(dirs: Dict[str, str], base: str) -> List[Dict[str, Any]]:
    """Per-file .log files for *base* (excluding _silence_gaps), newest first."""
    versions: List[Dict[str, Any]] = []
    for log_dir in _log_dirs(dirs):
        try:
            names = os.listdir(log_dir)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".log") or "_silence_gaps" in name:
                continue
            stem = name[:-4]
            if stem == base:
                ts = None
            elif stem.startswith(base + "_"):
                suffix = stem[len(base):]
                m = next((rx.match(suffix) for rx in _LOG_TS_RES if rx.match(suffix)), None)
                if not m:
                    continue
                ts = m.group(1)
            else:
                continue
            path = os.path.join(log_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            versions.append({
                # id doubles as the ?file= parameter and is always an exact
                # match against this list, so no path traversal.
                "id": os.path.relpath(path, dirs["audio"]),
                "filename": name,
                "timestamp": ts,
                "mtime": st.st_mtime,
                "size": st.st_size,
            })
    versions.sort(key=lambda v: v["mtime"], reverse=True)
    return versions


def _write_sidecar(path: str, values: Dict[str, str], comment: str) -> None:
    lines = ["# Generated by swhisper web UI — per-file overrides", ""]
    if comment.strip():
        lines += [f"{COMMENT_PREFIX} {line}".rstrip() for line in comment.strip().splitlines()]
        lines.append("")
    for key, value in values.items():
        rendered = value.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        if '#' in rendered or rendered != rendered.strip():
            rendered = f'"{rendered}"'
        lines.append(f"{key}={rendered}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (PROJECT_ROOT / "webui.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def api_meta() -> Dict[str, Any]:
    dirs = state.dirs()
    groups = []
    grouped_keys = set()
    for group in PARAM_GROUPS:
        keys = [k for k in group["keys"] if k in SIDECAR_PARAMS]
        grouped_keys.update(keys)
        groups.append({"name": group["name"], "keys": keys})
    leftovers = [k for k in SIDECAR_PARAMS if k not in grouped_keys]
    if leftovers:
        groups.append({"name": "Other", "keys": leftovers})
    defaults = _global_defaults()
    params = {
        key: {
            "stage": p.stage,
            "help": PARAM_HELP.get(key, p.note),
            "type": _TYPE_BY_PARSER.get(p.parse.__name__, "str"),
            "global": defaults.get(key),
            **parameter_ui(key),
        }
        for key, p in SIDECAR_PARAMS.items()
    }
    return {
        "audio_dir": dirs["audio"],
        "audio_paths": get_commented_audio_dirs(),
        "pipelines": [{"id": k, "label": v["label"]} for k, v in PIPELINES.items()],
        "groups": groups,
        "params": params,
    }


def _global_form():
    meta = api_meta()
    params = {k: dict(v) for k, v in meta['params'].items() if k != 'SWHISPER_NUM_SPEAKERS'}
    objs = {'transcription': _tconfig.TranscriptionConfig(), 'whisper': _tconfig.WhisperSettings(),
            'diarization': _dconfig.DiarizationConfig()}
    for key, (target, attr, kind, label) in GLOBAL_EXTRAS.items():
        params[key] = dict(type=kind, stage='diarize' if target == 'diarization' else 'transcribe',
                           help='Used by future runs. Per-file overrides take precedence where supported.',
                           **{**parameter_ui(key), 'label': label, 'global': getattr(objs[target], attr)})
    text, revision = env_snapshot(GLOBAL_ENV_PATH)
    raw = parse_env(text)
    values = {key: os.environ.get(key, raw.get(key)) for key in params if key in raw or key in os.environ}
    groups = [{'name': g['name'], 'keys': [k for k in g['keys'] if k in params]} for g in meta['groups']]
    groups.append({'name': 'Models and execution', 'keys': list(GLOBAL_EXTRAS)})
    return dict(params=params, groups=groups, values=values, revision=revision, path=str(GLOBAL_ENV_PATH))


@app.get('/api/settings/global')
def api_global_settings():
    return _global_form()


@app.put('/api/settings/global')
def api_save_global_settings(body: Dict[str, Any]):
    global _tconfig, _dconfig
    values = body.get('values')
    if not isinstance(values, dict) or not all(isinstance(v, str) for v in values.values()):
        raise HTTPException(422, 'Settings must be text values')
    with state.lock:
        if state.running():
            raise HTTPException(409, 'Global settings cannot be changed during a run.')
        form = _global_form()
        errors = validate_settings(values, form['params'])
        if errors:
            return JSONResponse(status_code=422, content={'errors': errors})
        try:
            update_env(GLOBAL_ENV_PATH, values, set(form['params']), body.get('revision'))
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        for key in form['params']:
            if key in values:
                os.environ[key] = values[key]
            else:
                os.environ.pop(key, None)
        # Config dataclasses capture environment values at import time.
        _tconfig, _dconfig = (_load_module('swhisper_webui_tconfig', 'transcribe/config.py'),
                              _load_module('swhisper_webui_dconfig', 'diarize/config.py'))
        return _global_form()


@app.post("/api/audio-dir")
def api_set_audio_dir(body: Dict[str, str]) -> Dict[str, Any]:
    raw_path = body.get("path") or ""
    try:
        path = str(normalize_path_value(raw_path))
    except ValueError as exc:
        raise HTTPException(400, f"Invalid path: {exc}")
    if not os.path.isdir(path):
        raise HTTPException(400, f"Not a directory: {path}")
    if state.running():
        raise HTTPException(409, "Cannot change the audio directory while a run is in progress")
    try:
        remember_audio_dir(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"Could not update .swhisper.env: {exc}")
    state.audio_dir_override = path
    return {"audio_dir": path, "audio_paths": get_commented_audio_dirs()}


@app.get("/api/files")
def api_files() -> Dict[str, Any]:
    dirs = state.dirs()
    files = []
    for base in _basenames(dirs["audio"]):
        json_path = os.path.join(dirs["json"], f"{base}.json")
        ok_path = os.path.join(dirs["output"], f"{base}.ok")
        sidecar_path = find_sidecar(dirs["audio"], base)
        transcribed = os.path.exists(json_path)
        diarized = os.path.exists(ok_path)
        redo_t = redo_d = False
        has_comment = False
        if sidecar_path:
            has_comment = bool(_read_comment(sidecar_path).strip())
            if transcribed:
                redo_t, _ = sidecar_requests_reprocess(dirs["audio"], base, json_path, STAGE_TRANSCRIBE)
            if diarized:
                redo_d, _ = sidecar_requests_reprocess(dirs["audio"], base, ok_path, STAGE_DIARIZE)
        files.append({
            "base": base,
            "transcribed": transcribed,
            "diarized": diarized,
            "sidecar": bool(sidecar_path) and bool(parse_sidecar_file(sidecar_path)) if sidecar_path else False,
            "comment": has_comment,
            "will_reprocess": redo_t or redo_d,
            "has_transcript": bool(_transcript_versions(dirs, base)),
            "has_logs": bool(_log_versions(dirs, base)),
        })
    return {"files": files}


@app.get("/api/sidecar/{basename}")
def api_get_sidecar(basename: str) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = find_sidecar(dirs["audio"], base)
    if path is None:
        return {"exists": False, "values": {}, "comment": "", "unknown": {}}
    raw = parse_sidecar_file(path)
    known = {k: v for k, v in raw.items() if k in SIDECAR_PARAMS}
    other = {k: v for k, v in raw.items() if k not in SIDECAR_PARAMS}
    return {"exists": True, "values": known, "comment": _read_comment(path), "unknown": other}


@app.put("/api/sidecar/{basename}")
def api_put_sidecar(basename: str, body: Dict[str, str]) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = os.path.join(dirs["audio"], base + SIDECAR_SUFFIX)
    comment = body.pop("_comment", "") or ""

    clean: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for key, raw in body.items():
        raw = (raw or "").strip()
        if not raw:
            continue
        param = SIDECAR_PARAMS.get(key)
        if param is None:
            reason = BLOCKED_KEYS.get(key, "unknown setting")
            errors[key] = f"not allowed per file ({reason})"
            continue
        try:
            param.parse(raw)
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
            continue
        clean[key] = raw

    if errors:
        return JSONResponse(status_code=422, content={"errors": errors})

    errors = validate_settings(clean, api_meta()['params'])
    if errors:
        return JSONResponse(status_code=422, content={"errors": errors})

    # Preserve keys the form cannot edit, including settings from newer versions.
    # Read them from disk, never trust unknown keys supplied by the client.
    existing = parse_sidecar_file(path) if os.path.isfile(path) else {}
    clean = {**{k: v for k, v in existing.items() if k not in SIDECAR_PARAMS}, **clean}

    if not clean and not comment.strip():
        # Nothing left — remove the sidecar entirely
        if os.path.isfile(path):
            os.remove(path)
        return {"exists": False, "values": {}, "comment": ""}

    # If only the comment changed, keep the old mtime so the edit does not
    # count as a settings change (and thus never triggers reprocessing).
    old_values: Optional[Dict[str, str]] = None
    old_times: Optional[Tuple[float, float]] = None
    if os.path.isfile(path):
        try:
            old_values = parse_sidecar_file(path)
            stat = os.stat(path)
            old_times = (stat.st_atime, stat.st_mtime)
        except OSError:
            pass

    _write_sidecar(path, clean, comment)

    if old_values is not None and old_times is not None and old_values == clean:
        os.utime(path, old_times)

    return {"exists": True, "values": clean, "comment": comment}


@app.delete("/api/sidecar/{basename}")
def api_delete_sidecar(basename: str) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = os.path.join(dirs["audio"], base + SIDECAR_SUFFIX)
    if os.path.isfile(path):
        os.remove(path)
    return {"exists": False}


@app.get("/api/history")
def api_history(limit: int = 50) -> Dict[str, Any]:
    dirs = state.dirs()
    history_path = os.path.join(dirs["audio"], HISTORY_FILENAME)
    entries: List[Dict[str, Any]] = []
    if os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    entries.reverse()  # newest first
    return {"path": history_path, "entries": entries[:max(1, min(limit, 500))]}


@app.get("/api/transcript/{basename}")
def api_transcript(basename: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Serve a TXT transcript for a file — the newest version by default."""
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    versions = _transcript_versions(dirs, base)
    if not versions:
        raise HTTPException(404, "No transcript found for this file")
    if version:
        selected = next((v for v in versions if v["id"] == version), None)
        if selected is None:
            raise HTTPException(400, "Unknown transcript version")
    else:
        selected = versions[0]
    path = os.path.join(dirs["audio"], selected["id"])
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(500, f"Could not read transcript: {exc}")
    return {
        "base": base,
        "versions": versions,
        "selected": selected["id"],
        "content": content,
        "has_stats": transcript_stats.has_stats_header(content),
        "stats_available": bool(transcript_stats.find_analysis_json(path)),
        # Only transcripts with a diarized JSON beside them can be re-rendered.
        "can_render": _diarized_json_for(dirs, selected["id"]) is not None,
    }


@app.post("/api/transcript/{basename}/stats")
def api_transcript_stats(basename: str, body: Dict[str, str]) -> Dict[str, Any]:
    """Add or remove the preamble+stats header on an existing TXT transcript,
    in place, without re-running transcription or diarization."""
    action = body.get("action")
    if action not in ("add", "remove"):
        raise HTTPException(400, "action must be 'add' or 'remove'")
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    versions = _transcript_versions(dirs, base)
    if not versions:
        raise HTTPException(404, "No transcript found for this file")
    version = body.get("version")
    if version:
        selected = next((v for v in versions if v["id"] == version), None)
        if selected is None:
            raise HTTPException(400, "Unknown transcript version")
    else:
        selected = versions[0]
    path = os.path.join(dirs["audio"], selected["id"])
    if action == "add":
        transcript_stats.add_stats(path, dirs["audio"], base)
    else:
        transcript_stats.remove_stats(path)
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(500, f"Could not read transcript: {exc}")
    return {
        "base": base,
        "versions": versions,
        "selected": selected["id"],
        "content": content,
        "has_stats": transcript_stats.has_stats_header(content),
        "stats_available": bool(transcript_stats.find_analysis_json(path)),
        # Only transcripts with a diarized JSON beside them can be re-rendered.
        "can_render": _diarized_json_for(dirs, selected["id"]) is not None,
    }


# --- on-demand re-rendering -------------------------------------------------
#
# Which markers and pauses a transcript shows is a presentation choice, not a
# property of the run: every one of them is recoverable from the diarized
# JSON. So the viewer offers them as toggles and re-renders in milliseconds,
# instead of the pipeline writing one file per combination.

def _render_config(threshold: Optional[float], preamble: bool):
    """The handful of attributes the formatters read off a diarization config."""
    defaults = _dconfig.DiarizationConfig()

    class _Config:
        silence_gap_linebreak_threshold = threshold
        output_preamble = defaults.output_preamble if preamble else ""
        get_preamble_with_transcript_id = defaults.get_preamble_with_transcript_id

    return _Config()


def _diarized_json_for(dirs: Dict[str, str], version_id: str) -> Optional[str]:
    """The diarized JSON a TXT transcript version was rendered from.

    Both pipelines write ``<root>/txt/<stem>.txt`` beside
    ``<root>/json/<stem>_diarized.json``.
    """
    txt_path = os.path.join(dirs["audio"], version_id)
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    root = os.path.dirname(os.path.dirname(txt_path))
    candidate = os.path.join(root, "json", f"{stem}_diarized.json")
    return candidate if os.path.isfile(candidate) else None


def _selected_version(dirs: Dict[str, str], base: str, version: Optional[str]) -> Dict[str, Any]:
    versions = _transcript_versions(dirs, base)
    if not versions:
        raise HTTPException(404, "No transcript found for this file")
    if not version:
        return versions[0]
    selected = next((v for v in versions if v["id"] == version), None)
    if selected is None:
        raise HTTPException(400, "Unknown transcript version")
    return selected


def _render(basename: str, body: Dict[str, Any]):
    """Shared by preview and save: locate the JSON, derive, format."""
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    selected = _selected_version(dirs, base, body.get("version"))
    json_path = _diarized_json_for(dirs, selected["id"])
    if json_path is None:
        raise HTTPException(
            404,
            "This transcript has no diarized JSON beside it, so it cannot be "
            "re-rendered. Re-run diarization to produce one.")
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not read diarized JSON: {exc}")

    options = RenderOptions.from_dict(body.get("options"))
    metadata = data.get("metadata") or {}
    floor = metadata.get("min_silence_duration")
    if not isinstance(floor, (int, float)):
        floor = DEFAULT_MIN_SILENCE

    segments = derive_segments(data.get("segments") or [], options,
                               pipeline_min_silence=float(floor))
    return dirs, base, selected, segments, options, metadata


@app.post("/api/render/{basename}")
def api_render(basename: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Re-render a transcript under the requested options, without saving."""
    dirs, base, selected, segments, options, metadata = _render(basename, body)
    config = _render_config(options.linebreak_threshold, options.preamble)
    stats = metadata.get("speaker_stats")
    transcript_id = selected.get("timestamp")

    content = TXTFormatter().render(segments, config=config,
                                    transcript_id=transcript_id, speaker_stats=stats,
                                    stats=options.stats)

    return {
        "base": base,
        "selected": selected["id"],
        "versions": _transcript_versions(dirs, base),
        "content": content,
        "options": asdict(options),
        "suffix": options.suffix(),
    }


@app.post("/api/render/{basename}/save")
def api_render_save(basename: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Write the current view to disk, next to the transcript it came from."""
    formats = body.get("formats") or ["txt"]
    unknown = [f for f in formats if f not in ("txt", "rtf")]
    if unknown:
        raise HTTPException(400, f"Unknown format(s): {', '.join(unknown)}")

    dirs, base, selected, segments, options, metadata = _render(basename, body)
    config = _render_config(options.linebreak_threshold, options.preamble)
    stats = metadata.get("speaker_stats")
    transcript_id = selected.get("timestamp")

    stem = os.path.splitext(os.path.basename(selected["id"]))[0] + options.suffix()
    root = os.path.dirname(os.path.dirname(os.path.join(dirs["audio"], selected["id"])))

    written: List[str] = []
    for fmt in formats:
        target_dir = os.path.join(root, fmt)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{stem}.{fmt}")
        formatter = TXTFormatter() if fmt == "txt" else RTFFormatter()
        formatter.format(segments, path, config=config,
                         transcript_id=transcript_id, speaker_stats=stats,
                         stats=options.stats)
        written.append(os.path.relpath(path, dirs["audio"]))

    return {"base": base, "written": written, "suffix": options.suffix()}


@app.get("/api/json-transcript/{basename}")
def api_json_transcript(basename: str) -> Dict[str, Any]:
    """Serve the raw whisper JSON transcript (pre-diarization) for a file."""
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = os.path.join(dirs["json"], f"{base}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "No JSON transcript yet — run transcription first.")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not read JSON transcript: {exc}")
    return {
        "base": base,
        "path": os.path.relpath(path, dirs["audio"]),
        "content": json.dumps(data, indent=2, ensure_ascii=False),
    }


@app.get("/api/logs")
def api_logs(base: Optional[str] = None, file: Optional[str] = None,
             offset: int = 0) -> Dict[str, Any]:
    """Logs for the Logs tab.

    - With ``base``: per-file ``.log`` files written by the pipelines. If
      ``file`` (a version id) is given, return that log's content; otherwise
      return just the version list.
    - Without ``base``: UI-run logs — the special ``live`` entry (incremental
      console via ``offset``) plus persisted logs of runs started from this UI.
    """
    dirs = state.dirs()
    audio = dirs["audio"]
    if base is not None:
        base = _safe_base(audio, base)
        versions = _log_versions(dirs, base)
        if file is None:
            return {"base": base, "versions": versions}
        selected = next((v for v in versions if v["id"] == file), None)
        if selected is None:
            raise HTTPException(400, "Unknown log file")
        try:
            content = Path(os.path.join(audio, selected["id"])).read_text(
                encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(500, f"Could not read log: {exc}")
        return {"base": base, "versions": versions, "selected": selected["id"],
                "content": content}

    versions: List[Dict[str, Any]] = [{
        "id": "live", "filename": "Live console", "timestamp": None,
        "mtime": time.time(), "size": 0,
    }]
    run_dir = os.path.join(audio, UI_RUN_LOG_DIR)
    try:
        names = os.listdir(run_dir)
    except OSError:
        names = []
    for name in names:
        if not (name.startswith("ui-run-") and name.endswith(".log")):
            continue
        path = os.path.join(run_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        m = re.match(r"^ui-run-(\d{8}-\d{6})-(.*)-[0-9a-f]{6}\.log$", name)
        versions.append({
            "id": f"{UI_RUN_LOG_DIR}/{name}",
            "filename": name,
            "timestamp": m.group(1) if m else None,
            "pipeline": m.group(2) if m else None,
            "mtime": st.st_mtime,
            "size": st.st_size,
        })
    if file == "live":
        live = state.log_since(offset)
        with state.lock:
            return {
                "scope": "run",
                "versions": versions,
                "selected": "live",
                "lines": live["lines"],
                "next": live["next"],
                "running": state.running(),
                "pipeline": state.pipeline,
                "returncode": state.returncode,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
            }
    if file:
        selected = next((v for v in versions if v["id"] == file), None)
        if selected is None:
            raise HTTPException(400, "Unknown run log")
        try:
            content = Path(os.path.join(audio, selected["id"])).read_text(
                encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(500, f"Could not read log: {exc}")
        return {"scope": "run", "versions": versions, "selected": selected["id"],
                "content": content}
    return {"scope": "run", "versions": versions}


@app.get("/api/run/preview")
def api_run_preview(pipeline: str) -> Dict[str, Any]:
    if pipeline not in PIPELINES:
        raise HTTPException(400, "Unknown pipeline")
    preview = preview_run(state.audio_dir(), pipeline,
                          _tconfig.TranscriptionConfig(), _dconfig.DiarizationConfig())
    _, revision = env_snapshot(GLOBAL_ENV_PATH)
    preview['token'] = hashlib.sha256((preview['token'] + revision).encode()).hexdigest()
    return preview


@app.post("/api/run")
def api_run(body: Dict[str, str]) -> Dict[str, Any]:
    pipeline = body.get("pipeline", "")
    if body.get("preview_token"):
        preview = api_run_preview(pipeline)
        if preview["token"] != body["preview_token"]:
            raise HTTPException(409, "Folder or settings changed. Review a new run preview before starting.")
    state.start(pipeline)
    return api_run_status()


@app.post("/api/run/stop")
def api_run_stop() -> Dict[str, Any]:
    state.stop()
    return api_run_status()


@app.get("/api/run/status")
def api_run_status() -> Dict[str, Any]:
    with state.lock:
        status = {
            "running": state.running(),
            "pipeline": state.pipeline,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "returncode": state.returncode,
            "stop_requested": state.stop_requested,
            "run_id": state.run_id,
            "audio_dir": state.run_audio_dir,
        }
        lines = state._read_complete_lines(state.run_log_path) if state.run_log_path else list(state.log_lines)
    status["log_length"] = len(lines)
    status["progress"] = inspect_progress(lines, status["pipeline"])
    return status


@app.get("/api/run/log")
def api_run_log(offset: int = 0) -> Dict[str, Any]:
    live = state.log_since(offset)
    with state.lock:
        return {
            "offset": offset,
            "next": live["next"],
            "lines": live["lines"],
            "running": state.running(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="swhisper web UI")
    parser.add_argument("--port", type=int, default=8722)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--audio-dir", help="Override SWHISPER_AUDIO_DIR for this session")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()
    if args.audio_dir:
        state.audio_dir_override = os.path.abspath(args.audio_dir)
        state.audio_dir_pinned = True
    url = f"http://{args.host}:{args.port}"
    print(f"🌐 swhisper web UI: {url}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
