"""Per-file speaker names, applied when a transcript is rendered.

Diarization names speakers ``SPEAKER_00`` and so on. A recording is far easier
to read and to quote once those labels carry the names of the people talking.

The names are a display mapping, never a change to a stored transcript: they are
applied to the segments derived from the diarized JSON, so they reach the
re-rendered view, the timestamped review list and exports, and leave every
transcript the pipeline produced untouched.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

NAMES_FILENAME = "speaker-names.json"
MAX_NAME_LENGTH = 60
RESERVED = {"SILENCE"}


def _path(audio_dir: str) -> str:
    return os.path.join(audio_dir, NAMES_FILENAME)


def load_all(audio_dir: str) -> Dict[str, Dict[str, str]]:
    try:
        with open(_path(audio_dir), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {base: names for base, names in data.items()
            if isinstance(names, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in names.items())}


def names_for(audio_dir: str, base: str) -> Dict[str, str]:
    return dict(load_all(audio_dir).get(base) or {})


def save_for(audio_dir: str, base: str, names: Dict[str, str]) -> Dict[str, str]:
    """Store the mapping for *base*, dropping labels mapped to a blank name."""
    cleaned = {label: name.strip() for label, name in names.items() if name and name.strip()}
    data = load_all(audio_dir)
    if cleaned:
        data[base] = cleaned
    else:
        data.pop(base, None)
    target = _path(audio_dir)
    temporary = f"{target}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    Path(temporary).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    os.replace(temporary, target)
    return cleaned


def validate(names: Any, detected: List[str]) -> Dict[str, str]:
    """Check a mapping supplied by the browser before it is stored."""
    if not isinstance(names, dict):
        raise ValueError("Speaker names must be a mapping of label to name")
    known = set(detected)
    cleaned: Dict[str, str] = {}
    for label, name in names.items():
        if not isinstance(label, str) or not isinstance(name, str):
            raise ValueError("Speaker names must be a mapping of label to name")
        if label not in known:
            raise ValueError(f"{label} is not a speaker in this transcript")
        name = name.strip()
        if not name:
            continue
        if len(name) > MAX_NAME_LENGTH:
            raise ValueError(f"Keep names under {MAX_NAME_LENGTH} characters")
        if "\n" in name or "\t" in name:
            raise ValueError("Speaker names cannot contain line breaks or tabs")
        cleaned[label] = name
    duplicates = {n for n in cleaned.values() if list(cleaned.values()).count(n) > 1}
    if duplicates:
        raise ValueError(f"Two speakers cannot share the name {sorted(duplicates)[0]}")
    return cleaned


def detected_in(segments: List[Dict[str, Any]]) -> List[str]:
    """Speaker labels present in the diarized segments, in first-seen order."""
    order: List[str] = []
    for segment in segments or []:
        label = segment.get("speaker")
        if isinstance(label, str) and label and label not in RESERVED and label not in order:
            order.append(label)
    return order


def apply(segments: List[Dict[str, Any]], names: Dict[str, str],
          speaker_stats: Optional[Dict[str, Any]] = None):
    """Return *segments* and *speaker_stats* with labels replaced by names."""
    if not names:
        return segments, speaker_stats
    renamed = []
    for segment in segments:
        copy = dict(segment)
        copy["speaker"] = names.get(copy.get("speaker"), copy.get("speaker"))
        if isinstance(copy.get("words"), list):
            copy["words"] = [
                {**word, "speaker": names.get(word.get("speaker"), word.get("speaker"))}
                if isinstance(word, dict) else word
                for word in copy["words"]
            ]
        renamed.append(copy)
    if isinstance(speaker_stats, dict):
        speaker_stats = {names.get(label, label): value for label, value in speaker_stats.items()}
    return renamed, speaker_stats
