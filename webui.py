#!/usr/bin/env python3
"""swhisper web UI — per-file settings editor and pipeline run control.

Usage::

    pip install fastapi uvicorn
    python webui.py [--port 8722] [--audio-dir /path/to/audio]

Then open http://127.0.0.1:8722 in a browser.

The UI lists the audio files in the configured audio directory with their
processing status, lets you edit per-file settings (stored as ``.swhisper``
sidecar files, see SIDECAR_CONFIG.md), and starts/stops pipeline runs while
streaming their log output.

The server binds to localhost only and is meant for local use.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    sys.exit("❌ The web UI needs fastapi and uvicorn:  pip install fastapi uvicorn")


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
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if self.audio_dir_override:
                env["SWHISPER_AUDIO_DIR"] = self.audio_dir_override
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
            self.log_lines = [f"▶ Started {PIPELINES[pipeline]['script']} (pid {self.process.pid})"]
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

    def stop(self) -> None:
        with self.lock:
            if not self.running():
                raise HTTPException(409, "No run in progress")
            self.stop_requested = True
            # SIGINT → pipelines run their KeyboardInterrupt cleanup paths
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)


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
            "note": p.note,
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
        if sidecar_path:
            if transcribed:
                redo_t, _ = sidecar_requests_reprocess(dirs["audio"], base, json_path, STAGE_TRANSCRIBE)
            if diarized:
                redo_d, _ = sidecar_requests_reprocess(dirs["audio"], base, ok_path, STAGE_DIARIZE)
        files.append({
            "base": base,
            "transcribed": transcribed,
            "diarized": diarized,
            "sidecar": bool(sidecar_path),
            "will_reprocess": redo_t or redo_d,
        })
    return {"files": files}


@app.get("/api/sidecar/{basename}")
def api_get_sidecar(basename: str) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = find_sidecar(dirs["audio"], base)
    if path is None:
        return {"exists": False, "values": {}}
    raw = parse_sidecar_file(path)
    known = {k: v for k, v in raw.items() if k in SIDECAR_PARAMS}
    other = {k: v for k, v in raw.items() if k not in SIDECAR_PARAMS}
    return {"exists": True, "values": known, "unknown": other}


@app.put("/api/sidecar/{basename}")
def api_put_sidecar(basename: str, values: Dict[str, str]) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = os.path.join(dirs["audio"], base + SIDECAR_SUFFIX)

    clean: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for key, raw in values.items():
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

    if not clean:
        # Saving an empty form removes the sidecar
        if os.path.isfile(path):
            os.remove(path)
        return {"exists": False, "values": {}}

    lines = ["# Generated by swhisper web UI — per-file overrides", ""]
    for key, value in clean.items():
        rendered = value.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        if '#' in rendered or rendered != rendered.strip():
            rendered = f'"{rendered}"'
        lines.append(f"{key}={rendered}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"exists": True, "values": clean}


@app.delete("/api/sidecar/{basename}")
def api_delete_sidecar(basename: str) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    path = os.path.join(dirs["audio"], base + SIDECAR_SUFFIX)
    if os.path.isfile(path):
        os.remove(path)
    return {"exists": False}


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
    args = parser.parse_args()
    if args.audio_dir:
        state.audio_dir_override = os.path.abspath(args.audio_dir)
    print(f"🌐 swhisper web UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
