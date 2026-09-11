"""Regression test: converting a non-WAV source file to WAV must never delete
the original. A prior version of prepare_staged_audio_file deleted the
source after conversion, on the assumption it was a disposable staged copy
in a scratch directory -- true back when files were copied into a temp
workspace before conversion, no longer true once conversion started running
directly in the source directory.
"""

import shutil

import numpy as np
import pytest
import soundfile as sf

from transcribe.config import TranscriptionConfig
from transcribe.file_utils import FileManager

FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
def test_prepare_staged_audio_file_keeps_the_original(tmp_path):
    # Write real WAV bytes, then hand them to FileManager under a non-.wav
    # extension so prepare_staged_audio_file takes the ffmpeg-conversion path.
    wav_bytes_path = tmp_path / "_source.wav"
    sf.write(str(wav_bytes_path), np.zeros(4410, dtype="float32"), 44100)

    original_path = tmp_path / "recording.m4a"
    shutil.copyfile(wav_bytes_path, original_path)
    wav_bytes_path.unlink()

    manager = FileManager(TranscriptionConfig())
    result = manager.prepare_staged_audio_file(str(tmp_path), "recording.m4a")

    assert result == "recording.wav"
    assert original_path.exists(), "the original source file must never be deleted"
    assert (tmp_path / "recording.wav").exists()
