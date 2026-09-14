"""Recording durations for the file list, cached on disk and filled in the background.

Reading a duration costs milliseconds for WAV (libsndfile reads the header) but
hundreds of milliseconds for a compressed container (ffprobe starts a process).
Neither belongs on a request that lists a whole folder, so the listing answers
from a cache and a worker fills the gaps.

The cache is keyed by size and modification time, so replacing a recording with
a different one of the same name re-measures it.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

CACHE_FILENAME = "durations.json"
# Formats libsndfile reads directly; everything else needs ffprobe.
_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

_lock = threading.Lock()
_warming: set = set()


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CACHE_FILENAME)


def _stamp(path: str):
    try:
        info = os.stat(path)
    except OSError:
        return None
    return [info.st_size, int(info.st_mtime)]


def load(cache_dir: str) -> Dict[str, dict]:
    try:
        with open(_cache_path(cache_dir), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(cache_dir: str, data: Dict[str, dict]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir)
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        Path(temporary).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def measure(path: str) -> Optional[float]:
    """Duration of one recording in seconds, or None when it cannot be read."""
    if Path(path).suffix.lower() in _NATIVE_SUFFIXES:
        try:
            import soundfile
            return float(soundfile.info(path).duration)
        except Exception:
            pass          # fall through to ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30, check=False)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def known(cache_dir: str, files: Dict[str, str]) -> Dict[str, Optional[float]]:
    """Cached durations for *files* (basename to path); missing ones are absent."""
    cached = load(cache_dir)
    out: Dict[str, Optional[float]] = {}
    for base, path in files.items():
        entry = cached.get(base)
        if isinstance(entry, dict) and entry.get("stamp") == _stamp(path):
            out[base] = entry.get("duration")
    return out


def warm(cache_dir: str, files: Dict[str, str]) -> bool:
    """Start filling the cache for any file missing from it. True if work started."""
    have = known(cache_dir, files)
    missing = {base: path for base, path in files.items() if base not in have}
    if not missing:
        return False
    key = os.path.realpath(cache_dir)
    with _lock:
        if key in _warming:
            return True
        _warming.add(key)

    def run() -> None:
        try:
            measured = {base: (measure(path), _stamp(path)) for base, path in missing.items()}
            with _lock:
                data = load(cache_dir)
                for base, (duration, stamp) in measured.items():
                    if stamp is not None:
                        data[base] = {"duration": duration, "stamp": stamp}
                _save(cache_dir, data)
        finally:
            with _lock:
                _warming.discard(key)

    threading.Thread(target=run, daemon=True).start()
    return True
