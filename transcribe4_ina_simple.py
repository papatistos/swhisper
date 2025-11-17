#!/usr/bin/env python3
"""
Transcription app using inaSpeechSegmenter VAD instead of Silero.

This extends the base TranscriptionApp to use inaSpeechSegmenter (ranked #1 in benchmarks)
for Voice Activity Detection instead of the default Silero VAD.

Simple implementation - just replaces the vad parameter in WhisperSettings.
No monkey-patching needed!
"""

import os
import sys
from pathlib import Path
from typing import Optional
import importlib.util

# Import TranscriptionApp from transcribe.py file
_transcribe_path = Path(__file__).with_name("transcribe.py")
_spec = importlib.util.spec_from_file_location("swhisper_transcribe", _transcribe_path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load transcribe module.")
_transcribe_module = importlib.util.module_from_spec(_spec)
sys.modules['swhisper_transcribe'] = _transcribe_module
_spec.loader.exec_module(_transcribe_module)

TranscriptionApp = _transcribe_module.TranscriptionApp

from transcribe.ina_vad import create_ina_vad_provider


class TranscriptionAppIna(TranscriptionApp):
    """Transcription app with inaSpeechSegmenter VAD.
    
    Inherits all functionality from TranscriptionApp, but replaces
    the Silero VAD with inaSpeechSegmenter for potentially better
    speech detection.
    """
    
    def __init__(self, ina_conda_env: str = 'ina_vad', **kwargs):
        """Initialize transcription app with inaSpeechSegmenter VAD.
        
        Args:
            ina_conda_env: Name of conda environment with inaSpeechSegmenter installed
            **kwargs: Additional arguments passed to TranscriptionAppV4
        """
        super().__init__(**kwargs)
        
        # Create INA VAD provider using subprocess wrapper
        print(f"🎤 Initializing inaSpeechSegmenter VAD (env: {ina_conda_env})...")
        self.ina_vad = create_ina_vad_provider(conda_env=ina_conda_env)
        print("    ✅ inaSpeechSegmenter VAD ready")
    
    def _process_single_file(
        self,
        wav_file: str,
        audio_dir: str,
        output_dir: str,
        file_index: int,
        total_files: int,
        original_label: Optional[str] = None
    ):
        """Process a single audio file with inaSpeechSegmenter VAD.
        
        Simple implementation:
        1. Extract VAD segments using inaSpeechSegmenter (via subprocess)
        2. Set whisper_settings.vad to the extracted segments
        3. Call parent's _process_single_file which will use those segments
        
        Args:
            wav_file: Name of the audio file
            audio_dir: Directory containing the audio file
            output_dir: Directory where JSON output should be written
            file_index: Index of this file in processing queue
            total_files: Total number of files to process
            original_label: Optional original filename (before conversion)
        """
        # Build full path to audio file
        audiofile_path = os.path.join(audio_dir, wav_file)
        
        # Extract speech segments using inaSpeechSegmenter
        print(f"🔍 Running inaSpeechSegmenter VAD on full file...")
        speech_segments = self.ina_vad.get_speech_segments(audiofile_path)
        
        # Calculate coverage stats
        total_speech = sum(end - start for start, end in speech_segments)
        import soundfile as sf
        with sf.SoundFile(audiofile_path) as f:
            total_duration = len(f) / f.samplerate
        coverage_pct = (total_speech / total_duration * 100) if total_duration > 0 else 0
        
        print(f"   ✅ Found {len(speech_segments)} speech segments")
        print(f"   📊 Speech coverage: {total_speech:.1f}s / {total_duration:.1f}s ({coverage_pct:.1f}%)")
        
        # Simply replace the vad parameter in whisper settings!
        # whisper-timestamped accepts either:
        #   - A string like "silero:v3.1" (built-in VAD)
        #   - A list of (start, end) tuples (custom VAD segments)
        original_vad = self.whisper_settings.vad
        self.whisper_settings.vad = speech_segments
        
        try:
            # Call parent's implementation - it will use our VAD segments
            super()._process_single_file(
                wav_file, audio_dir, output_dir,
                file_index, total_files, original_label
            )
        finally:
            # Restore original VAD setting (in case we process multiple files)
            self.whisper_settings.vad = original_vad


def main():
    """Main entry point for transcription with inaSpeechSegmenter VAD."""
    # Create transcription app with INA VAD
    app = TranscriptionAppIna()
    
    # Run transcription
    app.run()


if __name__ == "__main__":
    main()
