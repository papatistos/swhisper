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
    
    def _process_single_file(self, wav_file: str, audio_dir: str, output_dir: str,
                              file_index: int, total_files: int,
                              original_label: Optional[str] = None):
        """Override to use INA VAD segments as chunk boundaries."""
        audiofile_path = os.path.join(audio_dir, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        file_label = original_label or wav_file
        
        print(f"\n🎵 Processing ({file_index}/{total_files}): {file_label}")
        print("-" * 40)
        
        # Extract VAD segments BEFORE calling parent's processing
        print(f"🔍 Running inaSpeechSegmenter VAD analysis...")
        
        # Get ALL segments (speech, music, noise, etc.)
        all_segments = self.ina_vad.get_all_segments(audiofile_path)
        
        # Save ALL segments to TSV file
        tsv_path = os.path.join(output_dir, f"{base_name}_ina_segments.tsv")
        self._save_segments_to_tsv(all_segments, tsv_path)
        print(f"💾 Saved full segmentation to: {tsv_path}")
        
        # Get speech-only segments for transcription
        speech_segments = [(start, stop) for label, start, stop in all_segments 
                          if label in {'speech', 'male', 'female'}]
        
        # Calculate coverage stats
        if speech_segments:
            total_speech = sum(stop - start for start, stop in speech_segments)
            import soundfile as sf
            with sf.SoundFile(audiofile_path) as f:
                total_duration = len(f) / f.samplerate
            coverage_pct = (total_speech / total_duration * 100) if total_duration > 0 else 0
            
            # Count segment types
            segment_counts = {}
            for label, start, stop in all_segments:
                segment_counts[label] = segment_counts.get(label, 0) + 1
            
            print(f"✅ Found {len(all_segments)} total segments:")
            for label in sorted(segment_counts.keys()):
                print(f"   - {label}: {segment_counts[label]} segments")
            print(f"📊 Speech coverage: {total_speech:.1f}s / {total_duration:.1f}s ({coverage_pct:.1f}%)")
        else:
            print(f"⚠️  No speech segments detected - will transcribe full file")
        
        # Use INA speech segments as chunk boundaries
        if speech_segments:
            # Convert INA speech segments to boundaries for chunking
            # Each speech segment becomes a separate chunk
            boundaries = [0.0]  # Start with 0
            for start, stop in speech_segments:
                if start > boundaries[-1]:
                    boundaries.append(start)
                boundaries.append(stop)
            
            # Add final boundary if needed
            if boundaries[-1] < total_duration:
                boundaries.append(total_duration)
            
            print(f"📏 Using {len(speech_segments)} INA speech segments as chunks")
            
            # Disable VAD in whisper settings since INA already did VAD
            original_vad = self.whisper_settings.vad
            self.whisper_settings.vad = None
            
            try:
                # Process using INA-based boundaries directly
                output_path = os.path.join(output_dir, f"{base_name}.json")
                file_label = original_label or wav_file
                
                result = self.transcription_pipeline.process_audio_file(
                    audiofile_path,
                    boundaries,
                    output_path,
                    file_label=file_label,
                    file_index=file_index,
                    total_files=total_files
                )
                
                # Save final result
                self._save_transcription_result(result, output_path)
                
                # Persist VAD segments
                vad_output_path = os.path.join(output_dir, f"{base_name}_vad.tsv")
                self._save_vad_segments(result, vad_output_path)
                
            finally:
                self.whisper_settings.vad = original_vad
        else:
            # No speech detected - fall back to parent's processing
            super()._process_single_file(
                wav_file=wav_file,
                audio_dir=audio_dir,
                output_dir=output_dir,
                file_index=file_index,
                total_files=total_files,
                original_label=original_label
            )
    
    def _save_segments_to_tsv(self, segments: list, tsv_path: str):
        """Save inaSpeechSegmenter segments to TSV file.
        
        Format matches inaSpeechSegmenter's own CSV export format:
        - Column names: 'labels', 'start', 'stop' (matching INA's seg2csv)
        - 'stop' is the END time of the segment
        
        Args:
            segments: List of (label, start, stop) tuples from INA
            tsv_path: Path to output TSV file
        """
        with open(tsv_path, 'w', encoding='utf-8') as f:
            # Write header - matching inaSpeechSegmenter's CSV format
            f.write("labels\tstart\tstop\n")
            
            # Write segments
            for label, start, stop in segments:
                f.write(f"{label}\t{start:.3f}\t{stop:.3f}\n")


def main():
    """Main entry point for transcription with inaSpeechSegmenter VAD."""
    # Create transcription app with INA VAD
    app = TranscriptionAppIna()
    
    # Run transcription
    app.run()


if __name__ == "__main__":
    main()
