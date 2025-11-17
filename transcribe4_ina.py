#!/usr/bin/env python3
"""
Transcription app using inaSpeechSegmenter VAD instead of Silero.

This extends the base TranscriptionApp to use inaSpeechSegmenter (ranked #1 in benchmarks)
for Voice Activity Detection instead of the default Silero VAD.

Simple implementation - sets VAD before parent processes the file.
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

from transcribe.ina_vad_direct import create_ina_vad_direct
from transcribe.config import TranscriptionConfig, WhisperSettings


class TranscriptionAppIna(TranscriptionApp):
    """Transcription app with inaSpeechSegmenter VAD.
    
    Inherits all functionality from TranscriptionApp, but replaces
    the Silero VAD with inaSpeechSegmenter for potentially better
    speech detection.
    
    Uses direct import (no subprocess) - requires inaSpeechSegmenter 0.8.0+
    installed in the swhisper environment.
    """
    
    def __init__(self, 
                 config: Optional[TranscriptionConfig] = None,
                 whisper_settings: Optional[WhisperSettings] = None):
        """Initialize transcription app with inaSpeechSegmenter VAD.
        
        Args:
            config: Transcription configuration
            whisper_settings: Whisper model settings
        """
        super().__init__(config=config, whisper_settings=whisper_settings)
        
        # Create INA VAD provider (direct import)
        print(f"🎤 Initializing inaSpeechSegmenter VAD (direct import)...")
        self.ina_vad = create_ina_vad_direct(
            detect_gender=False,
            vad_engine='smn'
        )
        print("    ✅ inaSpeechSegmenter VAD ready")
    
    def process_files(self):
        """Override to inject VAD before each file is processed."""
        # Get files to process
        files_to_process = self._get_files_to_process()
        
        if not files_to_process:
            print("✅ No files to process")
            return
        
        print(f"\n🎯 Found {len(files_to_process)} file(s) to process")
        
        # Process each file
        for idx, (wav_file, original_label) in enumerate(files_to_process, 1):
            audiofile_path = os.path.join(self.config.audio_dir, wav_file)
            
            # Extract VAD segments BEFORE calling parent's processing
            print(f"\n{'=' * 60}")
            print(f"📄 Processing file {idx}/{len(files_to_process)}: {wav_file}")
            print(f"{'=' * 60}")
            print(f"🔍 Running inaSpeechSegmenter VAD on full file...")
            
            speech_segments = self.ina_vad.get_speech_segments(audiofile_path)
            
            # Calculate coverage stats
            if speech_segments:
                total_speech = sum(end - start for start, end in speech_segments)
                import soundfile as sf
                with sf.SoundFile(audiofile_path) as f:
                    total_duration = len(f) / f.samplerate
                coverage_pct = (total_speech / total_duration * 100) if total_duration > 0 else 0
                
                print(f"   ✅ Found {len(speech_segments)} speech segments")
                print(f"   📊 Speech coverage: {total_speech:.1f}s / {total_duration:.1f}s ({coverage_pct:.1f}%)")
            else:
                print(f"   ⚠️  No speech segments detected - will transcribe full file")
            
            # Set VAD segments in whisper settings
            # whisper-timestamped will use these instead of its own VAD
            original_vad = self.whisper_settings.vad
            self.whisper_settings.vad = speech_segments if speech_segments else None
            
            try:
                # Now call parent's single file processing
                self._process_single_file(
                    wav_file=wav_file,
                    audio_dir=self.config.audio_dir,
                    output_dir=self.config.output_dir,
                    file_index=idx,
                    total_files=len(files_to_process),
                    original_label=original_label
                )
            finally:
                # Restore original VAD setting
                self.whisper_settings.vad = original_vad


def main():
    """Main entry point for transcription with inaSpeechSegmenter VAD."""
    # Create transcription app with INA VAD
    app = TranscriptionAppIna()
    
    # Run transcription
    app.run()


if __name__ == "__main__":
    main()
