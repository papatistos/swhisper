"""Discover source recordings without treating converted WAVs as new inputs."""
import os

AUDIO_EXTENSIONS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}


def source_audio_files(audio_dir):
    """Prefer non-WAV sources over same-basename WAVs, preserving file order.

    The pipelines save converted WAVs beside their source recordings. A WAV
    without a corresponding source remains an input in its own right. Multiple
    non-WAV sources with the same basename remain visible as an ambiguity.
    """
    files = [name for name in os.listdir(audio_dir)
             if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS
             and os.path.isfile(os.path.join(audio_dir, name))]
    source_bases = {os.path.splitext(name)[0] for name in files
                    if os.path.splitext(name)[1].lower() != '.wav'}
    return [name for name in files
            if os.path.splitext(name)[1].lower() != '.wav'
            or os.path.splitext(name)[0] not in source_bases]
