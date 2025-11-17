#!/usr/bin/env python3
"""
Trans Transcription with inaSpeechSegmenter VAD integration.

Extends TranscriptionAppV4 to use inaSpeechSegmenter (#1 ranked VAD) instead of Silero.
"""

import os
import sys
from typing import List, Tuple, Optional
from transcribe4 import TranscriptionAppV4
from transcribe.ina_vad import InaSpeechVAD


class TranscriptionAppV4Ina(TranscriptionAppV4):
    """
    Transcription pipeline using inaSpeechSegmenter for Voice Activity Detection.
    
    Runs full-file VAD analysis upfront, then injects chunk-relative VAD segments
    into each chunk's transcription.
    """
    
    def __init__(self):
        """Initialize with inaSpeechSegmenter VAD."""
        super().__init__()
        
        print("🔧 Initializing inaSpeechSegmenter VAD (subprocess mode)...")
        self.ina_vad = InaSpeechVAD(conda_env='ina_vad', verify_env=False)
        print("   ✅ inaSpeechSegmenter VAD ready")
        
        # Will store full-file VAD segments for chunk-wise injection
        self._full_file_vad_segments: List[Tuple[float, float]] = []
    
    def _get_vad_segments_for_chunk(self, chunk_start: float, chunk_end: float) -> List[Tuple[float, float]]:
        """
        Extract VAD segments for a specific chunk and convert to chunk-relative times.
        
        Args:
            chunk_start: Chunk start time in full file (seconds)
            chunk_end: Chunk end time in full file (seconds)
            
        Returns:
            List of (start, end) tuples relative to chunk start
        """
        chunk_relative_segments = []
        
        for seg_start, seg_end in self._full_file_vad_segments:
            # Skip segments that don't overlap with this chunk
            if seg_end <= chunk_start or seg_start >= chunk_end:
                continue
            
            # Clip segment to chunk boundaries and convert to chunk-relative
            rel_start = max(0.0, seg_start - chunk_start)
            rel_end = min(chunk_end - chunk_start, seg_end - chunk_start)
            
            # Only include if there's actual duration
            if rel_end > rel_start:
                chunk_relative_segments.append((rel_start, rel_end))
        
        return chunk_relative_segments
    
    def _process_single_file(self, wav_file: str, audio_dir: str, output_dir: str,
                            file_index: int, total_files: int,
                            original_label: Optional[str] = None):
        """
        Process a single audio file with inaSpeechSegmenter VAD.
        
        Runs full-file VAD upfront, stores segments, then lets parent do the chunked transcription.
        We'll override the transcription pipeline to inject VAD per chunk.
        """
        # Get full path
        audiofile_path = os.path.join(audio_dir, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        
        # Check if already processed
        output_path = os.path.join(output_dir, f"{base_name}.json")
        if os.path.exists(output_path) and not self.config.force_reprocess:
            print(f"⏭️  Skipping {wav_file} (already processed)")
            return
        
        # Get file duration for logging
        import soundfile as sf
        with sf.SoundFile(audiofile_path) as f:
            duration = len(f) / f.samplerate
        
        print(f"🎵 Audio duration: {duration:.1f}s")
        
        # Run full-file VAD analysis
        print(f"🔍 Running inaSpeechSegmenter VAD on full file...")
        speech_segments = self.ina_vad.get_speech_segments(audiofile_path)
        
        # Store for chunk-wise injection
        self._full_file_vad_segments = speech_segments
        
        # Log statistics
        if speech_segments:
            total_speech = sum(end - start for start, end in speech_segments)
            coverage_pct = (total_speech / duration * 100) if duration > 0 else 0
            print(f"   ✅ Found {len(speech_segments)} speech segments")
            print(f"   📊 Speech coverage: {total_speech:.1f}s / {duration:.1f}s ({coverage_pct:.1f}%)")
        else:
            print(f"   ⚠️  No speech segments detected")
        
        # Now we need to override the chunk processor to inject VAD per chunk
        # We'll temporarily modify the whisper_settings before each chunk
        original_pipeline_method = self.transcription_pipeline.process_audio_file
        vad_app = self  # Capture for closure
        
        def process_audio_file_with_vad(audiofile_path, boundaries, output_path,
                                       start_chunk=0, existing_chunk_results=None,
                                       file_label=None, file_index=None, total_files=None):
            """Wrapper that injects VAD per chunk."""
            # Intercept chunk processing
            original_process_chunk = vad_app.transcription_pipeline.chunk_processor.process_chunk_in_subprocess
            
            def process_chunk_with_vad(audiofile_path, start_time, end_time, chunk_id, file_label=None):
                """Inject chunk-relative VAD before processing."""
                # Get VAD segments for this chunk
                chunk_vad = vad_app._get_vad_segments_for_chunk(start_time, end_time)
                
                # Log VAD injection
                if chunk_vad:
                    total_vad = sum(e - s for s, e in chunk_vad)
                    chunk_dur = end_time - start_time
                    coverage = (total_vad / chunk_dur * 100) if chunk_dur > 0 else 0
                    print(f"    🔊 Using {len(chunk_vad)} inaSpeech VAD segments ({coverage:.1f}% coverage)")
                else:
                    print(f"    ⚠️  No speech detected by inaSpeech VAD in this chunk")
                
                # Temporarily inject VAD into whisper_settings
                original_vad = vad_app.transcription_pipeline.chunk_processor.whisper_settings._settings.get('vad')
                vad_app.transcription_pipeline.chunk_processor.whisper_settings._settings['vad'] = chunk_vad
                
                try:
                    # Call original process_chunk
                    return original_process_chunk(audiofile_path, start_time, end_time, chunk_id, file_label)
                finally:
                    # Restore original VAD setting
                    if original_vad is None:
                        vad_app.transcription_pipeline.chunk_processor.whisper_settings._settings.pop('vad', None)
                    else:
                        vad_app.transcription_pipeline.chunk_processor.whisper_settings._settings['vad'] = original_vad
            
            # Monkey-patch the chunk processor
            vad_app.transcription_pipeline.chunk_processor.process_chunk_in_subprocess = process_chunk_with_vad
            
            try:
                # Call original process_audio_file
                return original_pipeline_method(audiofile_path, boundaries, output_path,
                                               start_chunk, existing_chunk_results,
                                               file_label, file_index, total_files)
            finally:
                # Restore original method
                vad_app.transcription_pipeline.chunk_processor.process_chunk_in_subprocess = original_process_chunk
        
        # Temporarily replace the pipeline method
        self.transcription_pipeline.process_audio_file = process_audio_file_with_vad
        
        try:
            # Call parent's processing
            super()._process_single_file(wav_file, audio_dir, output_dir,
                                        file_index, total_files, original_label)
        finally:
            # Restore original method
            self.transcription_pipeline.process_audio_file = original_pipeline_method


def main():
    """Entry point."""
    app = TranscriptionAppV4Ina()
    app.run()


if __name__ == "__main__":
    main()
