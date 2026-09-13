"""Utilities for managing user-specific filesystem paths.

This module centralizes local path configuration so that working copies
can override values without modifying the source code. Values can be set via
environment variables or by creating a ``.swhisper.env`` file in the project root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit

_ENV_FILE_NAME = ".swhisper.env"
_AUDIO_DIR_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<comment>#\s*)?SWHISPER_AUDIO_DIR\s*=(?P<value>.*)$"
)


def _strip_outer_quotes(value: str) -> str:
    """Remove one pair of matching quotes around a whole path value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _unescape_shell_path(value: str) -> str:
    """Turn common shell-escaped path syntax into a filesystem path.

    Paths copied from a terminal or a shell command can contain escaped
    spaces (``\\ ``), quotes, and tildes.  In particular, zsh-style copied
    paths may contain several backslashes before a tilde.  Those backslashes
    are quoting syntax, not part of the directory name.
    """
    result = []
    quote = None
    index = 0
    while index < len(value):
        char = value[index]

        if quote is not None and char == quote:
            quote = None
            index += 1
            continue

        if quote == "'":
            result.append(char)
            index += 1
            continue

        if quote is None and char in {"'", '"'}:
            quote = char
            index += 1
            continue

        if char != "\\":
            result.append(char)
            index += 1
            continue

        # Handle a run of backslashes together so that both ordinary shell
        # escapes (e.g. ``\\ ``) and pasted ``\\\\\\~`` forms work.
        run_end = index
        while run_end < len(value) and value[run_end] == "\\":
            run_end += 1
        count = run_end - index
        if run_end == len(value):
            result.extend("\\" * ((count + 1) // 2))
            break

        next_char = value[run_end]
        if next_char == "~":
            # A tilde inside a path component is a literal tilde; all
            # preceding backslashes are shell quoting in the copied value.
            result.append("~")
            index = run_end + 1
            continue

        result.extend("\\" * (count // 2))
        if count % 2:
            result.append(next_char)
            index = run_end + 1
        else:
            index = run_end

    return "".join(result)


def normalize_path_value(raw_value: str, base_dir: Optional[Path] = None) -> Path:
    """Parse a user-entered path in plain, quoted, or shell-escaped form.

    Supported input includes paths such as ``/tmp/My\\ Folder``, quoted
    paths, ``~/...``, environment-variable references, and local ``file://``
    URLs.  The returned path is absolute and normalized, but it is not
    required to exist.
    """
    if not isinstance(raw_value, str):
        raise ValueError("Path must be text")
    value = _strip_outer_quotes(raw_value.strip())
    if not value:
        raise ValueError("Path is empty")

    value = _unescape_shell_path(value)
    if value.startswith("file://"):
        parsed = urlsplit(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("Only local file URLs are supported")
        value = unquote(parsed.path)

    value = os.path.expandvars(os.path.expanduser(value)).strip()
    if not value:
        raise ValueError("Path is empty")

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _path_key(value: str) -> Optional[str]:
    """Return a normalized comparison key for an env-file path value."""
    try:
        return str(normalize_path_value(value))
    except (TypeError, ValueError, OSError):
        return None


def _audio_dir_assignments(lines):
    """Yield ``(index, is_commented, raw_value)`` for audio-dir settings."""
    for index, line in enumerate(lines):
        match = _AUDIO_DIR_ASSIGNMENT_RE.match(line)
        if match:
            yield index, bool(match.group("comment")), match.group("value").strip()


def get_commented_audio_dirs() -> list[str]:
    """Return unique ``#SWHISPER_AUDIO_DIR=...`` paths in file order."""
    env_path = Path(__file__).resolve().parent / _ENV_FILE_NAME
    if not env_path.exists():
        return []

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assignments = list(_audio_dir_assignments(lines))
    active_keys = {
        parsed_key
        for _, is_commented, raw_value in assignments
        if not is_commented
        for parsed_key in [_path_key(raw_value)]
        if parsed_key is not None
    }
    paths = []
    seen = set(active_keys)
    for _, is_commented, raw_value in assignments:
        if not is_commented:
            continue
        key = _path_key(raw_value)
        if key is None or key in seen:
            continue
        seen.add(key)
        paths.append(key)
    return paths


def remember_audio_dir(path_value: str) -> str:
    """Remember an audio directory as a unique commented env-file entry.

    Existing duplicate commented entries are removed at the same time.  An
    active ``SWHISPER_AUDIO_DIR`` entry wins over its commented equivalent,
    so the file does not accumulate two copies of the current path.
    """
    normalized = normalize_path_value(path_value)
    key = str(normalized)
    env_path = Path(__file__).resolve().parent / _ENV_FILE_NAME
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    assignments = list(_audio_dir_assignments(lines))
    active_keys = {
        parsed_key
        for _, is_commented, raw_value in assignments
        if not is_commented
        for parsed_key in [_path_key(raw_value)]
        if parsed_key is not None
    }
    seen_comments = set()
    cleaned = []
    last_audio_index = -1
    for index, line in enumerate(lines):
        assignment = next((item for item in assignments if item[0] == index), None)
        if assignment is None:
            cleaned.append(line)
            continue

        _, is_commented, raw_value = assignment
        last_audio_index = len(cleaned)
        if not is_commented:
            cleaned.append(line)
            continue

        parsed_key = _path_key(raw_value)
        if parsed_key is None or parsed_key in active_keys or parsed_key in seen_comments:
            continue
        seen_comments.add(parsed_key)
        cleaned.append(line)

    if key not in active_keys and key not in seen_comments:
        new_line = f'#SWHISPER_AUDIO_DIR="{key}"'
        if last_audio_index >= 0:
            cleaned.insert(last_audio_index + 1, new_line)
        else:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            cleaned.append(new_line)

    env_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return key


def _load_env_file() -> None:
    """Load key/value pairs from the optional ``.swhisper.env`` file."""
    env_path = Path(__file__).resolve().parent / _ENV_FILE_NAME
    if not env_path.exists():
        return

    from env_settings import parse_env
    for key, value in parse_env(env_path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value


def _optional_path(env_var: str) -> Optional[Path]:
    """Return an optional path from an environment variable."""
    raw_value = os.getenv(env_var)
    if not raw_value:
        return None

    raw_value = raw_value.strip().strip('"').strip("'")
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        project_root = Path(__file__).resolve().parent
        candidate = (project_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    return candidate


@dataclass(frozen=True)
class PathSettings:
    """Container for repository-specific path overrides."""

    audio_dir: Optional[Path]
    temp_dir: Optional[Path]
    diarize_default_audio_dir: Optional[Path]
    strict_model_path: Optional[Path]
    sample_audio_file: Optional[Path]

    def as_strings(self) -> Dict[str, Optional[str]]:
        """Return a dictionary with stringified path values."""
        return {
            key: str(value) if value is not None else None
            for key, value in self.__dict__.items()
        }


@lru_cache(maxsize=1)
def get_path_settings() -> PathSettings:
    """Get path overrides from environment or ``.swhisper.env``."""
    _load_env_file()
    return PathSettings(
        audio_dir=_optional_path("SWHISPER_AUDIO_DIR"),
        temp_dir=_optional_path("SWHISPER_TEMP_DIR"),
        diarize_default_audio_dir=_optional_path("SWHISPER_DIARIZE_AUDIO_DIR"),
        strict_model_path=_optional_path("SWHISPER_STRICT_MODEL_PATH"),
        sample_audio_file=_optional_path("SWHISPER_SAMPLE_AUDIO_FILE"),
    )


__all__ = [
    "PathSettings",
    "get_commented_audio_dirs",
    "get_path_settings",
    "normalize_path_value",
    "remember_audio_dir",
]
