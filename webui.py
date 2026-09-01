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
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from path_settings import get_path_settings
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

AUDIO_EXTENSIONS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}

COMMENT_PREFIX = "#;"  # user comment lines inside sidecar files (ignored by parsers)
HISTORY_FILENAME = "swhisper_run_history.jsonl"

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
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.pipeline: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.log_lines: List[str] = []
        self.stop_requested = False
        self.run_audio_dir: Optional[str] = None

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

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, pipeline: str) -> None:
        if pipeline not in PIPELINES:
            raise HTTPException(400, f"Unknown pipeline '{pipeline}'")
        with self.lock:
            if self.running():
                raise HTTPException(409, "A run is already in progress")
            script = PROJECT_ROOT / PIPELINES[pipeline]["script"]
            audio_dir = self.audio_dir()
            env = dict(os.environ, PYTHONUNBUFFERED="1", SWHISPER_AUDIO_DIR=audio_dir)
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,  # own process group → clean SIGINT
            )
            self.pipeline = pipeline
            self.started_at = time.time()
            self.finished_at = None
            self.returncode = None
            self.stop_requested = False
            self.run_audio_dir = audio_dir
            self.log_lines = [f"▶ Started {PIPELINES[pipeline]['script']} (pid {self.process.pid}) — audio dir: {audio_dir}"]
            threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                with self.lock:
                    self.log_lines.append(line.rstrip("\n"))
                    if len(self.log_lines) > 20000:  # ring buffer
                        del self.log_lines[:5000]
        finally:
            proc.wait()
            with self.lock:
                self.returncode = proc.returncode
                self.finished_at = time.time()
                status = "⏹ stopped by user" if self.stop_requested else (
                    "✅ finished" if proc.returncode == 0 else f"❌ exited with code {proc.returncode}")
                self.log_lines.append(f"— Run {status} —")
            try:
                self._write_history_entry()
            except Exception as exc:  # history must never break the UI
                with self.lock:
                    self.log_lines.append(f"⚠️ Could not write run history: {exc}")

    def stop(self) -> None:
        with self.lock:
            if not self.running():
                raise HTTPException(409, "No run in progress")
            self.stop_requested = True
            # SIGINT → pipelines run their KeyboardInterrupt cleanup paths
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

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
            self.log_lines.append(
                f"🗂 Run recorded in {HISTORY_FILENAME} ({entry['file_count']} file(s) produced/updated)")


state = AppState()
app = FastAPI(title="swhisper web UI")


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
        }
        for key, p in SIDECAR_PARAMS.items()
    }
    return {
        "audio_dir": dirs["audio"],
        "pipelines": [{"id": k, "label": v["label"]} for k, v in PIPELINES.items()],
        "groups": groups,
        "params": params,
    }


@app.post("/api/audio-dir")
def api_set_audio_dir(body: Dict[str, str]) -> Dict[str, Any]:
    path = os.path.expanduser((body.get("path") or "").strip())
    if not path:
        raise HTTPException(400, "No path given")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise HTTPException(400, f"Not a directory: {path}")
    if state.running():
        raise HTTPException(409, "Cannot change the audio directory while a run is in progress")
    state.audio_dir_override = path
    return {"audio_dir": path}


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


@app.post("/api/run")
def api_run(body: Dict[str, str]) -> Dict[str, Any]:
    state.start(body.get("pipeline", ""))
    return api_run_status()


@app.post("/api/run/stop")
def api_run_stop() -> Dict[str, Any]:
    state.stop()
    return api_run_status()


@app.get("/api/run/status")
def api_run_status() -> Dict[str, Any]:
    with state.lock:
        return {
            "running": state.running(),
            "pipeline": state.pipeline,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "returncode": state.returncode,
            "log_length": len(state.log_lines),
        }


@app.get("/api/run/log")
def api_run_log(offset: int = 0) -> Dict[str, Any]:
    with state.lock:
        offset = max(0, min(offset, len(state.log_lines)))
        return {
            "offset": offset,
            "next": len(state.log_lines),
            "lines": state.log_lines[offset:],
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
    url = f"http://{args.host}:{args.port}"
    print(f"🌐 swhisper web UI: {url}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
