#!/usr/bin/env python3
"""
swhisper - Swedish Whisper Transcription with pyannote.audio 4 support

This entry point mirrors the legacy transcription pipeline while switching the
post-processing diarization stage to the pyannote/speaker-diarization-community-1
model available in pyannote.audio 4.
"""

import importlib.util
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure local packages resolve exactly like in the legacy script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_BASE_TRANSCRIBE_PATH = Path(__file__).with_name("transcribe.py")

_spec = importlib.util.spec_from_file_location(
    "swhisper_transcribe_legacy",
    _BASE_TRANSCRIBE_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load legacy transcribe module for reuse.")
_transcribe_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transcribe_module)

TranscriptionConfig = _transcribe_module.TranscriptionConfig
WhisperSettings = _transcribe_module.WhisperSettings
DEFAULT_CONFIG = _transcribe_module.DEFAULT_CONFIG
DEFAULT_WHISPER_SETTINGS = _transcribe_module.DEFAULT_WHISPER_SETTINGS
resource_manager = _transcribe_module.resource_manager

from diarize4 import run as diarize4_run


class TranscriptionAppV4(_transcribe_module.TranscriptionApp):
    """Transcription pipeline that targets pyannote.audio 4 community model."""

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        whisper_settings: Optional[WhisperSettings] = None,
    ) -> None:
        super().__init__(config, whisper_settings)

    def _run_diarization_pipeline(self) -> None:  # noqa: D401 - keep legacy wording
        print("\n" + "=" * 60)
        print("🎯 Launching diarization pipeline (pyannote.audio 4)...")

        try:
            diarize4_run(show_header=False)
        finally:
            resource_manager.clear_device_memory()


def main() -> None:
    """Main entry point for the pyannote.audio 4 transcription workflow."""
    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)

    app = TranscriptionAppV4()
    app.run()


if __name__ == "__main__":
    main()
