"""Utilities for managing user-specific filesystem paths.

This module centralizes local path configuration so that working copies
can override values without modifying the source code. Values can be set via
environment variables or by creating a ``.swhisper.env`` file in the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

_ENV_FILE_NAME = ".swhisper.env"


def _load_env_file() -> None:
    """Load key/value pairs from the optional ``.swhisper.env`` file."""
    env_path = Path(__file__).resolve().parent / _ENV_FILE_NAME
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Decode common escape sequences
        value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        if key and value and key not in os.environ:
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


__all__ = ["PathSettings", "get_path_settings"]
