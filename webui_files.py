"""Locating a file's inputs and outputs on disk: audio, transcripts, logs, sidecars.

Every function takes the resolved directory map from ``AppState.dirs`` so the
audio folder is always explicit and never read from module state.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from sidecar import SIDECAR_SUFFIX


AUDIO_EXTENSIONS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}
COMMENT_PREFIX = "#;"  # user comment lines inside sidecar files (ignored by parsers)

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


def _scribe_txt_dirs(dirs: Dict[str, str]) -> List[tuple]:
    """Every staged Scribe export as ``(txt_dir, source_basename)``.

    A Scribe run writes ``audio.txt`` inside its own work folder, so the
    recording it belongs to is only recoverable from the identity it recorded.
    """
    audio = Path(dirs["audio"]).resolve()
    root = audio / "scribe-webui"
    if not root.is_dir() or root.resolve() != root:
        return []
    found: List[tuple] = []
    for work in sorted(root.iterdir()):
        state_path = work / "scribe-state.json"
        txt_dir = work / "transcripts" / "txt"
        if (not work.is_dir() or work.resolve() != work
                or state_path.resolve() != state_path or txt_dir.resolve() != txt_dir):
            continue
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            source = Path(saved["identity"]["source"])
            if source.parent != audio:
                continue
            found.append((str(txt_dir), source.stem))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def _scribe_transcript_dirs(dirs: Dict[str, str], base: str) -> List[tuple]:
    """Associate staged audio exports with their recorded source identity."""
    return [(txt_dir, "audio", "Scribe") for txt_dir, stem in _scribe_txt_dirs(dirs)
            if stem == base]


def _transcript_versions(dirs: Dict[str, str], base: str) -> List[Dict[str, Any]]:
    """All TXT transcript versions for *base*, newest (by mtime) first.

    Pipelines write ``<base>_<YYYYMMDD-HHMMSS>.txt``; a plain ``<base>.txt``
    is accepted as an untimestamped fallback.
    """
    versions: List[Dict[str, Any]] = []
    sources = [(directory, base, None) for directory in _transcript_dirs(dirs)]
    sources.extend(_scribe_transcript_dirs(dirs, base))
    for txt_dir, export_base, pipeline in sources:
        try:
            names = os.listdir(txt_dir)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".txt"):
                continue
            stem = name[:-4]
            if stem == export_base:
                ts = None
            elif stem.startswith(export_base + "_"):
                m = _TRANSCRIPT_TS_RE.match(stem[len(export_base):])
                if not m:
                    continue
                ts = m.group(1)
            else:
                continue
            path = os.path.join(txt_dir, name)
            if pipeline and Path(path).resolve() != Path(path):
                continue
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
                "pipeline": pipeline,
            })
    versions.sort(key=lambda v: v["mtime"], reverse=True)
    return versions


# Inverses of the per-file version scans above. Listing each directory once
# and testing set membership replaces two directory listings per audio file.
_TRANSCRIPT_STEM_RE = re.compile(r"^(.*)_(\d{8}-\d{6})$")
_LOG_STEM_RES = [
    re.compile(r"^(.*)_(\d{8}-\d{6})$"),
    re.compile(r"^(.*)_(\d{4}-\d{2}-\d{2} \d{6})$"),
]


def _stems_with_versions(directories, suffix, patterns, skip=None):
    found = set()
    for directory in directories:
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if not name.endswith(suffix) or (skip and skip in name):
                continue
            stem = name[:-len(suffix)]
            found.add(stem)                      # an untimestamped <base>
            for pattern in patterns:
                match = pattern.match(stem)
                if match:
                    found.add(match.group(1))    # <base>_<timestamp>
    return found


def transcript_bases(dirs: Dict[str, str]) -> set:
    """Basenames that have at least one TXT transcript version."""
    found = _stems_with_versions(_transcript_dirs(dirs), ".txt", [_TRANSCRIPT_STEM_RE])
    # A Scribe export is named after its work folder, so the recording it
    # belongs to comes from the identity, not from the filename.
    for txt_dir, stem in _scribe_txt_dirs(dirs):
        if "audio" in _stems_with_versions([txt_dir], ".txt", [_TRANSCRIPT_STEM_RE]):
            found.add(stem)
    return found


def log_bases(dirs: Dict[str, str]) -> set:
    """Basenames that have at least one per-file pipeline log."""
    return _stems_with_versions(_log_dirs(dirs), ".log", _LOG_STEM_RES, skip="_silence_gaps")


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
