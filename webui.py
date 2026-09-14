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
import tempfile
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
from webui_run import preview_run, ProgressTracker
from run_scope import ENV_VAR as ONLY_FILES_ENV
import webui_durations
import webui_speakers
from env_settings import env_snapshot, parse_env, update_env
from webui_schema import (
    GLOBAL_EXTRAS,
    PARAM_GROUPS,
    PARAM_HELP,
    _TYPE_BY_PARSER,
    parameter_ui,
    validate_settings,
)
from webui_files import (
    AUDIO_EXTENSIONS,
    _basenames,
    _diarized_json_for,
    _log_versions,
    _read_comment,
    log_bases,
    _safe_base,
    _transcript_versions,
    _write_sidecar,
    transcript_bases,
)
import transcript_stats

from transcript_render import (
    RenderOptions,
    RTFFormatter,
    TXTFormatter,
    derive_segments,
    DEFAULT_MIN_SILENCE,
    create_paragraph_text_from_words,
)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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


HISTORY_FILENAME = "swhisper_run_history.jsonl"
UI_RUN_LOG_DIR = "logs"  # persisted console output of runs started from this UI
ACTIVE_RUN_FILENAME = "ui-run-active.json"
MAX_RUN_LOGS = 50           # persisted console logs kept per audio folder
MAX_HISTORY_ENTRIES = 500   # run history lines kept per audio folder

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
    "scribe": {
        "script": "webui_scribe.py",
        "label": "Scribe remote upload + local diarization",
    },
}

# Display grouping for the sidecar form (unlisted keys land in "Other")

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
        self.progress = ProgressTracker()
        self.progress_offset = 0
        self.scribe_plan = None

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
        return AppState._read_lines_from(path, 0)[0]

    @staticmethod
    def _read_lines_from(path: Optional[str], offset: int) -> Tuple[List[str], int]:
        """Complete lines written after byte *offset*, and the next byte offset.

        Reading by byte offset keeps a poll proportional to what the run wrote
        since the previous poll, not to everything it has written so far.
        """
        if not path:
            return [], 0
        try:
            size = os.path.getsize(path)
        except OSError:
            return [], offset
        # A new run writes a new file; a truncated one restarts from the top.
        offset = 0 if offset > size or offset < 0 else offset
        if offset == size:
            return [], offset
        try:
            with open(path, "rb") as handle:
                handle.seek(max(0, offset - 1))
                previous = handle.read(1) if offset else b""
                chunk = handle.read()
        except OSError:
            return [], offset
        # A CRLF split across two reads must not produce a blank line.
        if previous == b"\r" and chunk.startswith(b"\n"):
            chunk = chunk[1:]
            offset += 1
        # Do not return a line while the child is still in the middle of
        # writing it.  Otherwise the next poll would append that line twice.
        cut = max(chunk.rfind(b"\n"), chunk.rfind(b"\r")) + 1
        if not cut:
            return [], offset
        text = chunk[:cut].decode("utf-8", errors="replace")
        return text.splitlines(), offset + cut

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
        self.progress = ProgressTracker(self.pipeline)
        self.progress_offset = 0
        return bool(self.run_id and self.pipeline)

    def running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return self.pid is not None and self._pid_exists(self.pid)

    def log_since(self, offset: int = 0) -> Dict[str, Any]:
        """Console lines after byte *offset*, with the offset to send back."""
        with self.lock:
            path = self.run_log_path
        if not path:
            return {"lines": [], "next": 0}
        lines, next_offset = self._read_lines_from(path, offset)
        return {"lines": lines, "next": next_offset}

    def progress_snapshot(self) -> Tuple[Dict[str, Any], int]:
        """Advance the tracker over newly written lines and report its state."""
        with self.lock:
            path, offset = self.run_log_path, self.progress_offset
            tracker = self.progress
            if path:
                lines, self.progress_offset = self._read_lines_from(path, offset)
                if self.progress_offset < offset:   # log replaced under us
                    tracker = self.progress = ProgressTracker(self.pipeline)
                tracker.feed(lines)
            return tracker.snapshot(), tracker.line_count

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

    def start(self, pipeline: str, only: Optional[List[str]] = None, plan=None) -> None:
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

            if pipeline == 'scribe' and not plan:
                raise HTTPException(400, 'Scribe requires a reviewed plan')
            self.scribe_plan = plan if pipeline == 'scribe' else None
            started_at = time.time()
            run_id = uuid.uuid4().hex
            ts = datetime.fromtimestamp(started_at).strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(audio_dir, UI_RUN_LOG_DIR)
            os.makedirs(run_dir, exist_ok=True)
            run_log_path = os.path.join(
                run_dir, f"ui-run-{ts}-{pipeline}-{run_id[:6]}.log")
            header = f"▶ Started {PIPELINES[pipeline]['script']} — audio dir: {audio_dir}"
            env = dict(os.environ, PYTHONUNBUFFERED="1", SWHISPER_AUDIO_DIR=audio_dir)
            if only:
                env[ONLY_FILES_ENV] = "\n".join(only)
                header += f" — {len(only)} selected file(s)"
            else:
                env.pop(ONLY_FILES_ENV, None)
            with open(run_log_path, "w", encoding="utf-8", buffering=1) as log_file:
                log_file.write(header + "\n")
                log_file.flush()
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script)] + ([json.dumps(plan)] if pipeline == "scribe" else []),
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
            self.progress = ProgressTracker(pipeline)
            self.progress_offset = 0
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
        if self.scribe_plan:
            entry['scribe'] = dict(language=self.scribe_plan['language'],
                                   files=[r['file'] for r in self.scribe_plan['files']],
                                   work_dirs=[r['work_dir'] for r in self.scribe_plan['files']])
        entry["files_produced"] = self._collect_new_files(audio_dir, started_at)
        entry["file_count"] = len(entry["files_produced"])
        history_path = os.path.join(audio_dir, HISTORY_FILENAME)
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with self.lock:
            self._append_log_line_locked(
                f"🗂 Run recorded in {HISTORY_FILENAME} ({entry['file_count']} file(s) produced/updated)")
        self._prune_old_runs(audio_dir)

    def _prune_old_runs(self, audio_dir: str) -> None:
        """Keep the run logs and history bounded; neither had any limit before."""
        keep = os.path.realpath(self.run_log_path) if self.run_log_path else None
        run_dir = os.path.join(audio_dir, UI_RUN_LOG_DIR)
        try:
            logs = [os.path.join(run_dir, name) for name in os.listdir(run_dir)
                    if name.startswith("ui-run-") and name.endswith(".log")]
            logs.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            for path in logs[MAX_RUN_LOGS:]:
                if os.path.realpath(path) != keep:
                    os.remove(path)
        except OSError:
            pass
        history_path = os.path.join(audio_dir, HISTORY_FILENAME)
        try:
            with open(history_path, encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) > MAX_HISTORY_ENTRIES:
                temporary = f"{history_path}.{os.getpid()}.tmp"
                Path(temporary).write_text("".join(lines[-MAX_HISTORY_ENTRIES:]), encoding="utf-8")
                os.replace(temporary, history_path)
        except OSError:
            pass


state = AppState()
app = FastAPI(title="swhisper web UI")

# The server has no authentication because it binds to the loopback address.
# That only holds if requests really come from this machine, so refuse a Host
# header pointing somewhere else (DNS rebinding) and refuse a cross-origin
# request (a page you have open firing bodyless POSTs at the API).
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"}


def _hostname(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):                 # bracketed IPv6 literal
        return value.partition("]")[0] + "]"
    return value.rpartition(":")[0] if value.count(":") == 1 else value


@app.middleware("http")
async def restrict_to_local_requests(request, call_next):
    host = _hostname(request.headers.get("host", ""))
    if host and host not in LOCAL_HOSTNAMES:
        return JSONResponse(status_code=403, content={"detail": f"Refusing request for host {host}"})
    origin = request.headers.get("origin")
    if origin:
        scheme, _, rest = origin.partition("://")
        if scheme not in ("http", "https") or _hostname(rest) not in LOCAL_HOSTNAMES:
            return JSONResponse(status_code=403, content={"detail": "Refusing cross-origin request"})
    return await call_next(request)


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









# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (PROJECT_ROOT / "webui.html").read_text(encoding="utf-8")


# The stylesheet and script are served as separate files so the browser can
# cache them and so each can be linted on its own.
@app.get("/webui.css")
def webui_css() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "webui.css", media_type="text/css")


@app.get("/webui.js")
def webui_js() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "webui.js", media_type="text/javascript")


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


@app.post("/api/audio-dir/pick")
def api_pick_audio_dir() -> Dict[str, Any]:
    """Open a native macOS folder dialog in the server process and return the choice.

    The dialog appears on the machine running this server; the browser cannot
    provide absolute filesystem paths. The current audio directory is unchanged.
    """
    if sys.platform != "darwin":
        raise HTTPException(501, "The folder dialog is only available on macOS")
    if state.running():
        raise HTTPException(409, "Cannot pick a directory while a run is in progress")
    script = ('tell application "System Events" to activate\n'
              'set chosen to choose folder with prompt "Choose an audio directory"\n'
              'return POSIX path of chosen')
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "The folder dialog timed out") from None
    if result.returncode != 0:
        message = (result.stderr or "").strip()
        if "-128" in message or "User canceled" in message:
            raise HTTPException(400, "Folder selection canceled")
        raise HTTPException(500, f"Folder dialog failed: {message or 'unknown osascript error'}")
    path = str(normalize_path_value(result.stdout.strip()))
    if not os.path.isdir(path):
        raise HTTPException(400, f"Not a directory: {path}")
    return {"path": path}


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
    audio_mtimes = {}
    audio_paths: Dict[str, str] = {}
    for name in sorted(os.listdir(dirs['audio'])):
        base, ext = os.path.splitext(name)
        if ext.lower() in AUDIO_EXTENSIONS:
            path = os.path.join(dirs['audio'], name)
            # Prefer the WAV, matching the file the audio endpoint serves.
            if base not in audio_paths or ext.lower() == '.wav':
                audio_paths[base] = path
            try:
                audio_mtimes[base] = max(audio_mtimes.get(base, 0), os.path.getmtime(path))
            except FileNotFoundError:
                pass
    has_transcript = transcript_bases(dirs)
    has_logs = log_bases(dirs)
    # Durations come from a cache so listing a folder never waits on ffprobe.
    cache_dir = os.path.join(dirs["audio"], UI_RUN_LOG_DIR)
    durations = webui_durations.known(cache_dir, audio_paths)
    pending = webui_durations.warm(cache_dir, audio_paths)
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
            "has_transcript": base in has_transcript,
            "has_logs": base in has_logs,
            "mtime": audio_mtimes.get(base, 0),
            "duration": durations.get(base),
        })
    # Transcription writes SWHISPER_JSON_DIR but diarization reads JSON_INPUT_DIR.
    # When they differ, a file stays at "Awaiting diarization" with no clue why.
    diar_input = _dconfig.DiarizationConfig().json_input_dir
    json_output = os.getenv("SWHISPER_JSON_DIR", "whisper-json-output")
    note = (f"Transcription writes {json_output}, diarization reads {diar_input}. "
            "New transcription output is not picked up automatically."
            if diar_input != json_output else None)
    return {"files": files, "durations_pending": pending, "note": note}


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
    config = _dconfig.DiarizationConfig()
    introduction = config.get_preamble_with_transcript_id(selected['timestamp']) if selected.get('timestamp') else config.output_preamble
    return {
        "base": base,
        "versions": versions,
        "selected": selected["id"],
        "content": content,
        "has_stats": transcript_stats.has_stats_header(content),
        "has_preamble": bool(introduction and content.lstrip().startswith(introduction.strip())),
        "stats_available": bool(transcript_stats.find_analysis_json(path)),
        # Only transcripts with a diarized JSON beside them can be re-rendered.
        "can_render": _diarized_json_for(dirs, selected["id"]) is not None,
        "speaker_names": webui_speakers.names_for(dirs["audio"], base),
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
    # Speaker names are a display mapping over the stored run, applied here so
    # the view, the review list and every export agree on who is speaking.
    detected = webui_speakers.detected_in(data.get("segments") or [])
    names = webui_speakers.names_for(dirs["audio"], base)
    metadata = dict(metadata)
    segments, metadata["speaker_stats"] = webui_speakers.apply(
        segments, names, metadata.get("speaker_stats"))
    metadata["detected_speakers"] = detected
    metadata["speaker_names"] = names
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
        "timeline": _timeline_segments(segments, options.linebreak_threshold),
        "speakers": {"detected": metadata.get("detected_speakers") or [],
                     "names": metadata.get("speaker_names") or {}},
    }


def _export_plan(basename, body):
    dirs, base, selected, segments, options, metadata = _render(basename, body)
    formats = body.get('formats') or ['txt']
    if not isinstance(formats, list) or not formats or any(f not in ('txt', 'rtf') for f in formats):
        raise HTTPException(400, 'Choose TXT and/or RTF')
    formats = list(dict.fromkeys(formats))
    name = body.get('name') or (Path(selected['id']).stem + options.suffix() + '_view')
    if not isinstance(name, str) or not name.strip() or name in ('.', '..') or any(c in name for c in '/\\\x00'):
        raise HTTPException(400, 'Export name must be a filename without folders')
    root = Path(dirs['audio']).resolve()
    destination = (root / (body.get('destination') or 'exports')).resolve()
    if destination != root and root not in destination.parents:
        raise HTTPException(400, 'Choose an export folder inside the audio folder')
    targets = [destination / (name + '.' + fmt) for fmt in formats]
    protected = set()
    for version in _transcript_versions(dirs, base):
        original = Path(dirs['audio']) / version['id']
        protected.add(original.resolve())
        protected.add((original.parent.parent / 'rtf' / (original.stem + '.rtf')).resolve())
    if any(target.resolve() in protected for target in targets):
        raise HTTPException(409, 'Choose a new name or folder. Pipeline transcript versions are preserved.')
    if any(target.is_symlink() for target in targets):
        raise HTTPException(409, 'Export targets cannot be symbolic links')
    evidence = [(str(t), t.stat().st_mtime_ns, t.stat().st_size) if t.exists() else (str(t), None) for t in targets]
    token = hashlib.sha256(json.dumps([selected['id'], body.get('options'), evidence], sort_keys=True).encode()).hexdigest()
    plan = dict(name=name, destination=str(destination), paths=[str(t) for t in targets],
                existing=[str(t) for t in targets if t.exists()], token=token)
    return plan, targets, formats, selected, segments, options, metadata


@app.post('/api/render/{basename}/export-preview')
def api_export_preview(basename: str, body: Dict[str, Any]):
    return _export_plan(basename, body)[0]


@app.post('/api/render/{basename}/save')
def api_render_save(basename: str, body: Dict[str, Any]) -> Dict[str, Any]:
    plan, targets, formats, selected, segments, options, metadata = _export_plan(basename, body)
    if body.get('token') != plan['token']:
        raise HTTPException(409, 'Review the export destination again before saving')
    if plan['existing'] and not body.get('overwrite'):
        raise HTTPException(409, 'Export files already exist. Choose a new name or explicitly replace them.')
    config = _render_config(options.linebreak_threshold, options.preamble)
    rendered = [(TXTFormatter() if fmt == 'txt' else RTFFormatter()).render(
        segments, config=config, transcript_id=selected.get('timestamp'),
        speaker_stats=metadata.get('speaker_stats'), stats=options.stats) for fmt in formats]
    # Generate before touching any destination; pipeline originals are never targets.
    for target, content in zip(targets, rendered):
        target.parent.mkdir(parents=True, exist_ok=True)
        if body.get('overwrite'):
            fd, temporary = tempfile.mkstemp(prefix='.swhisper-export-', dir=target.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(content)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
            try:
                with target.open('x', encoding='utf-8') as handle:
                    handle.write(content)
            except FileExistsError:
                raise HTTPException(409, 'An export file appeared after preview. Review the destination again.')
    return {'written': plan['paths']}


def _audio_file(base):
    audio = Path(state.audio_dir())
    _safe_base(str(audio), base)
    candidates = [p for p in audio.iterdir() if p.stem == base and p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    candidates.sort(key=lambda p: (p.suffix.lower() != '.wav', p.name))
    if not candidates:
        raise HTTPException(404, 'No audio file found')
    return candidates[0]


def _timeline_segments(segments, threshold=1.0):
    rows = []
    for segment in segments:
        start, end = segment.get('start'), segment.get('end')
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        text = create_paragraph_text_from_words(segment, gap_threshold=threshold)
        rows.append(dict(start=max(0, start), end=max(start, end), speaker=segment.get('speaker', ''), text=text))
    return rows


@app.get('/api/audio/{basename}')
def api_audio(basename: str):
    return FileResponse(_audio_file(basename))


@app.get('/api/timeline/{basename}')
def api_timeline(basename: str, version: Optional[str] = None):
    dirs = state.dirs()
    base = _safe_base(dirs['audio'], basename)
    selected = _selected_version(dirs, base, version)
    path = _diarized_json_for(dirs, selected['id'])
    if path is None:
        return {'segments': [], 'audio': str(_audio_file(base).name)}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    raw = data.get('segments', [])
    named, _ = webui_speakers.apply(raw, webui_speakers.names_for(dirs['audio'], base))
    return {'segments': _timeline_segments(named), 'audio': _audio_file(base).name,
            'speakers': {'detected': webui_speakers.detected_in(raw),
                         'names': webui_speakers.names_for(dirs['audio'], base)}}


def _detected_speakers(dirs: Dict[str, str], base: str, version: Optional[str]) -> List[str]:
    selected = _selected_version(dirs, base, version)
    json_path = _diarized_json_for(dirs, selected["id"])
    if json_path is None:
        raise HTTPException(404, "This transcript has no diarized JSON, so speakers cannot be named.")
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not read diarized JSON: {exc}")
    return webui_speakers.detected_in(data.get("segments") or [])


@app.get("/api/speakers/{basename}")
def api_get_speakers(basename: str, version: Optional[str] = None) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    return {"base": base, "detected": _detected_speakers(dirs, base, version),
            "names": webui_speakers.names_for(dirs["audio"], base)}


@app.put("/api/speakers/{basename}")
def api_put_speakers(basename: str, body: Dict[str, Any]) -> Dict[str, Any]:
    dirs = state.dirs()
    base = _safe_base(dirs["audio"], basename)
    detected = _detected_speakers(dirs, base, body.get("version"))
    try:
        cleaned = webui_speakers.validate(body.get("names"), detected)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    try:
        saved = webui_speakers.save_for(dirs["audio"], base, cleaned)
    except OSError as exc:
        raise HTTPException(500, f"Could not save speaker names: {exc}")
    return {"base": base, "detected": detected, "names": saved}


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


@app.post("/api/reveal")
def api_reveal(body: Dict[str, str]) -> Dict[str, Any]:
    """Show a produced file in the desktop file manager.

    The path must resolve inside the audio folder, and the command is a fixed
    argument vector, so nothing the browser sends reaches a shell.
    """
    dirs = state.dirs()
    root = os.path.realpath(dirs["audio"])
    target = os.path.realpath(os.path.join(root, body.get("path") or ""))
    if target != root and os.path.commonpath([target, root]) != root:
        raise HTTPException(400, "That path is outside the audio folder")
    if not os.path.exists(target):
        raise HTTPException(404, "That file no longer exists")
    if sys.platform == "darwin":
        command = ["open", "-R", target]
    elif sys.platform.startswith("win"):
        command = ["explorer", f"/select,{target}"]
    else:
        command = ["xdg-open", os.path.dirname(target)]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise HTTPException(500, f"Could not open the file manager: {exc}")
    return {"revealed": os.path.relpath(target, root)}


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


def _validated_selection(audio_dir: str, only) -> Optional[set]:
    """Selected basenames, each checked against the audio folder.

    An empty list means nothing was selected, which is not the same as no
    restriction at all, so it is rejected rather than silently running everything.
    """
    if only is None:
        return None
    if not only:
        raise HTTPException(400, "Select at least one file, or run the whole folder")
    known = set(_basenames(audio_dir))
    chosen = {_safe_base(audio_dir, str(name)) for name in only}
    missing = sorted(chosen - known)
    if missing:
        raise HTTPException(400, f"Not in this folder: {', '.join(missing[:5])}")
    return chosen


@app.get('/api/scribe/artifacts')
def api_scribe_artifacts():
    audio = Path(state.audio_dir()).resolve()
    root = audio / 'scribe-webui'
    files = []
    if root.is_dir() and root.resolve() == root:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if (path.is_file() and path.resolve() == path
                    and not any(part.startswith('.') for part in relative.parts)
                    and path.suffix.lower() in {'.json', '.txt', '.rtf', '.tsv', '.srt', '.vtt', '.log', '.csv'}):
                files.append({'path': str(path.relative_to(audio)), 'size': path.stat().st_size})
    return {'files': files}


@app.get('/api/scribe/artifact')
def api_scribe_artifact(path: str):
    allowed = {item['path'] for item in api_scribe_artifacts()['files']}
    if path not in allowed:
        raise HTTPException(404, 'Scribe artifact not found')
    return FileResponse(Path(state.audio_dir()).resolve() / path)


@app.get("/api/run/preview")
def api_run_preview(pipeline: str, only: Optional[List[str]] = Query(None),
                    selected: str = '', language: str = 'Swedish (verbatim)') -> Dict[str, Any]:
    if pipeline not in PIPELINES:
        raise HTTPException(400, "Unknown pipeline")
    # Scribe carries its own single-recording selection and language; the local
    # pipelines take a set of basenames from the ticked files in the list.
    if pipeline == 'scribe':
        from webui_scribe import preview_scribe
        try:
            preview = preview_scribe(state.audio_dir(), selected, language)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from None
    else:
        audio_dir = state.audio_dir()
        preview = preview_run(audio_dir, pipeline, _tconfig.TranscriptionConfig(),
                              _dconfig.DiarizationConfig(),
                              _validated_selection(audio_dir, only))
    _, revision = env_snapshot(GLOBAL_ENV_PATH)
    preview['token'] = hashlib.sha256((preview['token'] + revision).encode()).hexdigest()
    return preview


@app.post("/api/run")
def api_run(body: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = body.get("pipeline", "")
    if pipeline == 'scribe':
        if body.get('remote_upload') is not True or not body.get('preview_token'):
            raise HTTPException(400, 'Scribe requires a reviewed preview and explicit remote upload consent.')
        selected, language = body.get('selected', ''), body.get('language', 'Swedish (verbatim)')
        if not isinstance(selected, str) or not isinstance(language, str):
            raise HTTPException(400, 'Invalid Scribe selection or language')
        preview = api_run_preview(pipeline, selected=selected, language=language)
        if preview['token'] != body['preview_token']:
            raise HTTPException(409, 'Folder or settings changed. Review a new run preview before starting.')
        if not preview['files']:
            raise HTTPException(400, 'No recordings selected')
        state.start(pipeline, plan=preview)
        return api_run_status()
    only = body.get("only")
    if only is not None and not isinstance(only, list):
        raise HTTPException(422, "Selected files must be a list of names")
    # An empty list is an empty selection, never "run everything".
    if isinstance(only, list) and not only:
        raise HTTPException(400, "Select at least one file, or run the whole folder")
    if body.get("preview_token"):
        preview = api_run_preview(pipeline, only)
        if preview["token"] != body["preview_token"]:
            raise HTTPException(409, "Folder, selection or settings changed. Review a new run preview before starting.")
    state.start(pipeline, sorted(_validated_selection(state.audio_dir(), only) or []) or None)
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
    status["progress"], status["log_length"] = state.progress_snapshot()
    # The audio folder is server-wide state. Report it on every poll so a tab
    # notices when another tab (or another client) moved the server.
    try:
        status["folder"] = state.audio_dir()
    except HTTPException:
        status["folder"] = None
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
