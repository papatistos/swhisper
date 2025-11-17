#!/usr/bin/env python3
"""
swhisper - Swedish Whisper Transcription with inaSpeechSegmenter VAD

This entry point uses inaSpeechSegmenter for Voice Activity Detection instead
of Silero. inaSpeechSegmenter was ranked #1 against 6 open-source VAD systems
(including Silero and Pyannote) on a French TV and radio benchmark.

Key features:
- Superior speech/music/noise classification
- CNN-based VAD optimized for broadcast content
- Optional gender detection during VAD stage
- Still uses pyannote.audio 4 for post-transcription diarization
"""

import importlib.util
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Ensure local packages resolve exactly like in transcribe4.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import transcribe4 module to inherit pyannote 4 framework
_BASE_TRANSCRIBE4_PATH = Path(__file__).with_name("transcribe4.py")

_spec = importlib.util.spec_from_file_location(
    "swhisper_transcribe4",
    _BASE_TRANSCRIBE4_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load transcribe4 module.")
_transcribe4_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transcribe4_module)

TranscriptionAppV4 = _transcribe4_module.TranscriptionAppV4
TranscriptionConfig = _transcribe4_module.TranscriptionConfig
WhisperSettings = _transcribe4_module.WhisperSettings
DEFAULT_CONFIG = _transcribe4_module.DEFAULT_CONFIG
DEFAULT_WHISPER_SETTINGS = _transcribe4_module.DEFAULT_WHISPER_SETTINGS

from transcribe.ina_vad import create_ina_vad_provider


class TranscriptionAppV4Ina(TranscriptionAppV4):
    """
    Transcription pipeline using inaSpeechSegmenter for VAD.
    
    This variant replaces Silero VAD with inaSpeechSegmenter, which provides
    superior speech detection especially for broadcast content with music/noise.
    """

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        whisper_settings: Optional[WhisperSettings] = None,
    ) -> None:
        super().__init__(config, whisper_settings)
        
        # Initialize inaSpeechSegmenter VAD (subprocess-based)
        print("🔧 Initializing inaSpeechSegmenter VAD (subprocess mode)...")
        self.ina_vad = create_ina_vad_provider(
            conda_env='ina_vad',     # Separate conda environment
            detect_gender=False,      # We do gender detection with pyannote later
            vad_engine='smn'          # speech/music/noise classification
        )
        print("   ✅ inaSpeechSegmenter VAD ready")
        
        # Store full-file VAD segments (will be set per file)
        self.full_file_vad_segments = None
    
    def _get_vad_segments_for_chunk(
        self,
        chunk_start: float,
        chunk_end: float
    ) -> List[Tuple[float, float]]:
        """
        Get VAD segments for a specific chunk, with chunk-relative timestamps.
        
        This extracts the relevant portion of the full-file VAD segments
        and converts timestamps to be relative to the chunk start.
        
        Args:
            chunk_start: Start time of chunk in seconds (absolute)
            chunk_end: End time of chunk in seconds (absolute)
            
        Returns:
            List of (start, end) tuples relative to chunk start (0.0 = chunk start)
        """
        if self.full_file_vad_segments is None:
            return []
        
        # Extract segments that overlap with this chunk
        chunk_relative_segments = []
        for seg_start, seg_end in self.full_file_vad_segments:
            # Check if segment overlaps with chunk
            if seg_end <= chunk_start or seg_start >= chunk_end:
                continue  # No overlap
            
            # Clip segment to chunk boundaries and convert to chunk-relative
            rel_start = max(0.0, seg_start - chunk_start)
            rel_end = min(chunk_end - chunk_start, seg_end - chunk_start)
            
            # Only include if there's actual duration
            if rel_end > rel_start:
                chunk_relative_segments.append((rel_start, rel_end))
        
        return chunk_relative_segments
    
    def _process_single_file(
        self,
        wav_file: str,
        audio_dir: str,
        output_dir: str,
        file_index: int,
        total_files: int,
        original_label: Optional[str] = None
    ):
        """
        Process a single audio file with inaSpeechSegmenter VAD.
        
        Overrides the parent method to inject VAD segments before transcription.
        """
        print(f"\n{'=' * 60}")
        print(f"📄 Processing file {file_index}/{total_files}")
        print(f"{'=' * 60}\n")
        
        # Get full path
        audiofile_path = os.path.join(audio_dir, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        
        # Check if already processed
        output_path = os.path.join(output_dir, f"{base_name}.json")
        if os.path.exists(output_path) and not self.config.force_reprocess:
            print(f"⏭️  Skipping {wav_file} (already processed)")
            return
        
        # Get file duration for chunking
        import soundfile as sf
        with sf.SoundFile(audiofile_path) as f:
            duration = len(f) / f.samplerate
        
        print(f"🎵 Audio duration: {duration:.1f}s")
        print(f"🔍 Running inaSpeechSegmenter VAD on full file...")
        
                    # Run full-file VAD analysis
            print(f"🔍 Running inaSpeechSegmenter VAD on full file...")
            speech_segments = self.ina_vad.get_speech_segments(audiofile_path)
            
            # Store VAD segments for chunk-wise injection
            self._full_file_vad_segments = speech_segments
            
            # Log statistics
            if speech_segments:
                total_speech = sum(end - start for start, end in speech_segments)
                file_duration = self._get_audio_duration(audiofile_path)
                coverage_pct = (total_speech / file_duration * 100) if file_duration > 0 else 0
                print(f"   ✅ Found {len(speech_segments)} speech segments")
                print(f"   📊 Speech coverage: {total_speech:.1f}s / {file_duration:.1f}s ({coverage_pct:.1f}%)")
            else:
                print(f"   ⚠️  No speech segments detected")
            
            # Call parent's processing - we'll inject VAD per-chunk
            super()._process_single_file(
                audiofile_path, wav_file, audio_duration,
                output_name, checkpoint_state, file_idx, total_files
            )
        
        try:
            # Now we need to inject chunk-relative VAD segments during transcription
            # We'll monkey-patch the TranscriptionWorker to inject our VAD per chunk
            
            # Save original VAD setting
            original_vad = self.whisper_settings.vad
            
            # Set VAD to None - we'll inject it per chunk
            self.whisper_settings.vad = None
            
            # Import the chunk processing we need
            from transcribe.transcription import TranscriptionWorker
            
            # Monkey-patch the chunk_worker to inject our VAD
            original_worker = TranscriptionWorker.chunk_worker
            vad_app_instance = self  # Capture self for closure
            
            @staticmethod
            def custom_chunk_worker(audiofile_path, start_time, end_time, model_path, settings, output_queue,
                                   device="mps", overlap_duration=2.0, file_label=None):
                """Custom worker that injects chunk-relative VAD from inaSpeechSegmenter."""
                # Get chunk-relative VAD segments
                chunk_vad = vad_app_instance._get_vad_segments_for_chunk(start_time, end_time)
                
                # Log VAD info
                if chunk_vad:
                    total_vad_duration = sum(end - start for start, end in chunk_vad)
                    chunk_duration = end_time - start_time
                    vad_coverage = (total_vad_duration / chunk_duration * 100) if chunk_duration > 0 else 0
                    print(f"    🔊 Using {len(chunk_vad)} inaSpeech VAD segments ({vad_coverage:.1f}% coverage)")
                else:
                    print(f"    ⚠️  No speech detected by inaSpeech VAD in this chunk")
                
                # Inject into settings
                settings_with_vad = settings.copy()
                settings_with_vad['vad'] = chunk_vad if chunk_vad else None
                
                # Call original worker with modified settings
                original_worker(audiofile_path, start_time, end_time, model_path, settings_with_vad,
                              output_queue, device, overlap_duration, file_label)
            
            # Apply monkey patch
            TranscriptionWorker.chunk_worker = custom_chunk_worker
            
            try:
                # Call parent's processing method
                super()._process_single_file(
                    wav_file,
                    audio_dir,
                    output_dir,
                    file_index,
                    total_files,
                    original_label
                )
            finally:
                # Restore original worker
                TranscriptionWorker.transcribe_chunk_worker = original_worker
                # Restore original VAD setting
                self.whisper_settings.vad = original_vad
        finally:
            # Clear stored VAD segments
            self.full_file_vad_segments = None
            self.whisper_settings.vad = original_vad


def main() -> None:
    """Main entry point for the inaSpeechSegmenter VAD workflow."""
    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)

    app = TranscriptionAppV4Ina()
    app.run()


if __name__ == "__main__":
    main()
