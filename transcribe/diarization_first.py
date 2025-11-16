"""
Diarization-first transcription pipeline.

This module implements a transcription pipeline that runs speaker diarization
BEFORE transcription, then uses the detected speaker segments to guide the
transcription process. This approach:

1. Provides more accurate speaker attribution (diarization sees full context)
2. Uses superior pyannote VAD instead of Silero
3. Creates natural chunk boundaries at speaker changes
4. Still maintains memory safety for long files
"""

import os
import time
import json
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import whisper_timestamped as whisper

from .config import TranscriptionConfig, WhisperSettings
from .audio_utils import AudioProcessor
from .memory_utils import ResourceManager
from .transcription import fix_zero_duration_words
from .speaker_chunking import (
    extract_speaker_segments,
    create_speaker_aware_chunks,
    convert_segments_to_vad_format,
    analyze_chunk_distribution,
    get_speaker_for_timestamp
)

# Disfluency marker pattern for post-processing
DISFLUENCY_MARKER_PATTERN = re.compile(r"^\[\*+\]$")


class DiarizationFirstPipeline:
    """
    Transcription pipeline that performs diarization first, then uses
    speaker segments to guide transcription.
    """
    
    def __init__(
        self,
        config: TranscriptionConfig,
        whisper_settings: WhisperSettings,
        checkpoint_manager=None
    ):
        self.config = config
        self.whisper_settings = whisper_settings
        self.checkpoint_manager = checkpoint_manager
        self.resource_manager = ResourceManager()
        self.audio_processor = AudioProcessor(config)
    
    def process_audio_file(
        self,
        audiofile_path: str,
        output_path: str,
        diarization_config=None,
        file_label: Optional[str] = None,
        file_index: Optional[int] = None,
        total_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file using diarization-first approach.
        
        Args:
            audiofile_path: Path to the audio file
            output_path: Path for final output
            diarization_config: Configuration for diarization (from diarize.config)
            file_label: Label for progress messages
            file_index: Current file index (for multi-file processing)
            total_files: Total number of files being processed
            
        Returns:
            Merged transcription result with speaker labels
        """
        label = file_label or os.path.basename(audiofile_path)
        
        print(f"\n🎯 Diarization-First Pipeline for: {label}")
        print("=" * 60)
        
        # Step 1: Run diarization
        print("\n📊 Step 1: Running speaker diarization...")
        diarization_result = self._run_diarization(audiofile_path, diarization_config)
        
        # Step 2: Extract speaker segments
        print("📋 Step 2: Extracting speaker segments...")
        speaker_segments = extract_speaker_segments(diarization_result, use_exclusive=True)
        print(f"   Found {len(speaker_segments)} speaker segments (exclusive/non-overlapping)")
        
        # Also extract overlapping segments to detect crosstalk
        overlapping_segments = extract_speaker_segments(diarization_result, use_exclusive=False)
        print(f"   Found {len(overlapping_segments)} segments in overlapping timeline")
        
        # Get unique speakers
        unique_speakers = set(seg['speaker'] for seg in speaker_segments)
        print(f"   Detected {len(unique_speakers)} speakers: {', '.join(sorted(unique_speakers))}")
        
        # Step 3: Create speaker-aware chunks (avoiding crosstalk)
        print("\n✂️  Step 3: Creating speaker-aware chunks (avoiding crosstalk at boundaries)...")
        chunks = create_speaker_aware_chunks(
            speaker_segments,
            target_duration=self.config.target_chunk_duration,
            max_duration=self.config.target_chunk_duration + 20,  # 20s buffer
            min_chunk_duration=30.0,
            overlapping_segments=overlapping_segments,
            avoid_crosstalk=True
        )
        
        # Analyze and display chunk distribution
        stats = analyze_chunk_distribution(chunks)
        print(f"   Created {stats['total_chunks']} chunks:")
        print(f"   - Average duration: {stats['avg_chunk_duration']:.1f}s")
        print(f"   - Range: {stats['min_chunk_duration']:.1f}s to {stats['max_chunk_duration']:.1f}s")
        
        # If file is short, process as single chunk
        if len(chunks) == 0:
            print("   ⚠️  No speaker segments found - falling back to single-chunk processing")
            return self._process_single_chunk_no_diarization(audiofile_path, label)
        elif len(chunks) == 1:
            print("   File is short enough - processing as single chunk with speaker segments")
        
        # Step 4: Process each chunk with transcription
        print("\n🎤 Step 4: Transcribing chunks with speaker-guided VAD...")
        chunk_results = []
        
        for chunk in chunks:
            chunk_idx = chunk['chunk_index']
            chunk_start = chunk['chunk_start']
            chunk_end = chunk['chunk_end']
            chunk_segments = chunk['segments']
            
            print(f"\n📝 Chunk {chunk_idx + 1}/{len(chunks)}: "
                  f"{chunk_start:.1f}s to {chunk_end:.1f}s "
                  f"({chunk_end - chunk_start:.1f}s, {len(chunk_segments)} speaker segments)")
            
            chunk_start_time = time.time()
            
            try:
                result = self._process_chunk_with_speakers(
                    audiofile_path,
                    chunk_start,
                    chunk_end,
                    chunk_segments,
                    label
                )
                
                chunk_results.append((result, chunk_start))
                
                # Monitor progress
                elapsed = time.time() - chunk_start_time
                print(f"   ✅ Completed in {elapsed:.1f}s")
                
                if chunk_idx + 1 < len(chunks):
                    remaining = len(chunks) - chunk_idx - 1
                    est_remaining = elapsed * remaining
                    print(f"   ⏱️  Estimated time remaining: {est_remaining / 60:.1f}min")
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx + 1}: {e}")
                raise
        
        # Step 5: Merge results
        print("\n🔀 Step 5: Merging chunk results...")
        merged_result = self._merge_chunk_results(chunk_results, speaker_segments)
        
        print("✅ Diarization-first processing complete!")
        
        return merged_result
    
    def _run_diarization(self, audio_path: str, diarization_config=None) -> Any:
        """
        Run pyannote diarization on the audio file.
        
        Args:
            audio_path: Path to audio file
            diarization_config: DiarizationConfig object (optional)
            
        Returns:
            pyannote Annotation object with diarization results
        """
        # Import diarization components
        try:
            from diarize.pipeline import DiarizationPipeline
            from diarize.config import DiarizationConfig
        except ImportError as e:
            raise ImportError(
                f"Could not import diarization components: {e}\n"
                "Make sure the diarize module is available."
            )
        
        # Use provided config or create default
        if diarization_config is None:
            diarization_config = DiarizationConfig()
        
        # Create and run diarization pipeline
        diarization_pipeline = DiarizationPipeline(diarization_config)
        
        try:
            result = diarization_pipeline.diarize(audio_path)
            return result
        finally:
            # Clean up
            self.resource_manager.clear_device_memory()
    
    def _process_chunk_with_speakers(
        self,
        audio_path: str,
        chunk_start: float,
        chunk_end: float,
        speaker_segments: List[Dict[str, Any]],
        label: str
    ) -> Dict[str, Any]:
        """
        Process a single chunk using speaker segments as VAD.
        
        Args:
            audio_path: Path to the audio file
            chunk_start: Start time of chunk in original audio
            chunk_end: End time of chunk in original audio
            speaker_segments: Speaker segments for this chunk (absolute timestamps)
            label: Label for progress messages
            
        Returns:
            Whisper transcription result with speaker information
        """
        # Convert speaker segments to VAD format (chunk-relative timestamps)
        vad_segments = convert_segments_to_vad_format(speaker_segments, chunk_start)
        
        # Debug: print first few VAD segments
        print(f"   📊 VAD segments sample (first 3): {vad_segments[:3] if len(vad_segments) > 0 else 'none'}")
        
        # Load audio chunk
        audio_data, sample_rate = self.audio_processor.load_audio_chunk(
            audio_path, chunk_start, chunk_end
        )
        
        # Create modified whisper settings with VAD segments
        whisper_params = self.whisper_settings.to_dict()
        whisper_params['vad'] = vad_segments  # Use speaker segments as VAD
        
        # Transcribe with Whisper model
        with self._safe_whisper_model() as model:
            print(f"   🔄 Transcribing with {len(vad_segments)} speaker segments as VAD...")
            result = whisper.transcribe(model, audio_data, **whisper_params)
        
        # Fix zero-duration words
        result = fix_zero_duration_words(result, min_duration=self.whisper_settings.min_word_duration)
        
        # Add speaker labels to words and segments
        result = self._add_speaker_labels(result, speaker_segments, chunk_start)
        
        # Store the original speaker segments for reference
        result['speaker_segments'] = speaker_segments
        
        return result
    
    def _add_speaker_labels(
        self,
        result: Dict[str, Any],
        speaker_segments: List[Dict[str, Any]],
        chunk_start: float
    ) -> Dict[str, Any]:
        """
        Add speaker labels to transcription result based on timestamp overlap.
        
        Args:
            result: Whisper transcription result
            speaker_segments: Speaker segments with absolute timestamps
            chunk_start: Start time of chunk (for converting relative to absolute)
            
        Returns:
            Modified result with speaker labels added
        """
        # Add speaker labels to segments
        if 'segments' in result:
            for segment in result['segments']:
                # Convert segment time to absolute time
                abs_start = segment['start'] + chunk_start
                abs_mid = abs_start + (segment['end'] - segment['start']) / 2
                
                # Find speaker at midpoint of segment
                speaker = get_speaker_for_timestamp(abs_mid, speaker_segments)
                segment['speaker'] = speaker or 'UNKNOWN'
                
                # Add speaker to words in segment
                if 'words' in segment:
                    for word in segment['words']:
                        abs_word_start = word['start'] + chunk_start
                        word_speaker = get_speaker_for_timestamp(abs_word_start, speaker_segments)
                        word['speaker'] = word_speaker or 'UNKNOWN'
        
        return result
    
    def _merge_chunk_results(
        self,
        chunk_results: List[Tuple[Dict[str, Any], float]],
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge results from multiple chunks.
        
        Args:
            chunk_results: List of (result_dict, chunk_start_time) tuples
            speaker_segments: Original speaker segments for the entire file
            
        Returns:
            Merged transcription result
        """
        if not chunk_results:
            return {'text': '', 'segments': [], 'language': self.whisper_settings.language}
        
        if len(chunk_results) == 1:
            # Single chunk - just return it (already has correct timestamps)
            return chunk_results[0][0]
        
        # Multiple chunks - merge them
        all_segments = []
        all_text_parts = []
        
        for result, chunk_start in chunk_results:
            if 'segments' in result:
                for segment in result['segments']:
                    # Adjust timestamps to absolute time
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] = segment['start'] + chunk_start
                    adjusted_segment['end'] = segment['end'] + chunk_start
                    
                    # Adjust word timestamps
                    if 'words' in adjusted_segment:
                        adjusted_words = []
                        for word in adjusted_segment['words']:
                            adjusted_word = word.copy()
                            adjusted_word['start'] = word['start'] + chunk_start
                            adjusted_word['end'] = word['end'] + chunk_start
                            adjusted_words.append(adjusted_word)
                        adjusted_segment['words'] = adjusted_words
                    
                    all_segments.append(adjusted_segment)
                    all_text_parts.append(segment.get('text', ''))
        
        # Create merged result
        merged = {
            'text': ' '.join(all_text_parts),
            'segments': all_segments,
            'language': self.whisper_settings.language,
            'speaker_segments': speaker_segments  # Include original diarization
        }
        
        return merged
    
    def _process_single_chunk_no_diarization(
        self,
        audio_path: str,
        label: str
    ) -> Dict[str, Any]:
        """
        Fallback: process entire file without diarization (when diarization fails).
        """
        print("   ⚠️  Processing without speaker segmentation")
        
        audio_data, _ = self.audio_processor.load_audio_chunk(
            audio_path, 0, float('inf')
        )
        
        with self._safe_whisper_model() as model:
            result = whisper.transcribe(model, audio_data, **self.whisper_settings.to_dict())
        
        result = fix_zero_duration_words(result, min_duration=self.whisper_settings.min_word_duration)
        
        return result
    
    @contextmanager
    def _safe_whisper_model(self):
        """Context manager for safe whisper model handling."""
        model = None
        try:
            model = whisper.load_model(self.config.model_str, device=self.config.device)
            self.resource_manager.current_whisper_model = model
            yield model
        finally:
            if model is not None:
                del model
                self.resource_manager.current_whisper_model = None
            self.resource_manager.clear_device_memory()
    
    def _add_silence_markers(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add silence markers between words where gaps exceed minimum duration.
        
        This creates silence markers like (0.5), (1.2) between words to match
        the output format of the original pipeline.
        """
        segments = result.get('segments', [])
        if not segments:
            return result
        
        min_silence = self.config.min_silence_duration
        enhanced_segments = []
        
        for seg_idx, segment in enumerate(segments):
            words = segment.get('words', [])
            if not words:
                enhanced_segments.append(segment)
                continue
            
            # Process words and add silence markers
            enhanced_words = []
            
            for i, word in enumerate(words):
                enhanced_words.append(word)
                
                # Check for gap to next word within segment
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    current_end = word.get('end', word.get('start', 0))
                    next_start = next_word.get('start', next_word.get('end', current_end))
                    gap_duration = next_start - current_end
                    
                    if gap_duration >= min_silence:
                        rounded_gap = round(gap_duration, 1)
                        silence_text = f"({rounded_gap:.1f})".replace("(0.", "(.")
                        
                        silence_word = {
                            'start': current_end,
                            'end': next_start,
                            'word': silence_text,
                            'text': silence_text,
                            'speaker': 'SILENCE',
                            'confidence': 1.0,
                            'is_silence_marker': True
                        }
                        enhanced_words.append(silence_word)
            
            # Update segment with enhanced words
            new_segment = segment.copy()
            new_segment['words'] = enhanced_words
            enhanced_segments.append(new_segment)
            
            # Check for cross-segment gap
            if seg_idx + 1 < len(segments):
                next_segment = segments[seg_idx + 1]
                next_words = next_segment.get('words', [])
                
                if enhanced_words and next_words:
                    last_word = enhanced_words[-1]
                    first_next_word = next_words[0]
                    
                    current_end = last_word.get('end', last_word.get('start', 0))
                    next_start = first_next_word.get('start', first_next_word.get('end', current_end))
                    gap_duration = next_start - current_end
                    
                    if gap_duration >= min_silence:
                        rounded_gap = round(gap_duration, 1)
                        silence_text = f"({rounded_gap:.1f})".replace("(0.", "(.")
                        
                        # Create standalone silence segment
                        silence_segment = {
                            'start': current_end,
                            'end': next_start,
                            'text': silence_text,
                            'speaker': 'SILENCE',
                            'is_silence_marker': True,
                            'words': [{
                                'start': current_end,
                                'end': next_start,
                                'word': silence_text,
                                'text': silence_text,
                                'speaker': 'SILENCE',
                                'confidence': 1.0,
                                'is_silence_marker': True
                            }]
                        }
                        enhanced_segments.append(silence_segment)
        
        result['segments'] = enhanced_segments
        return result
    
    def _scale_disfluency_markers(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust disfluency markers to reflect duration via repeated asterisks.
        
        Converts [*] markers based on duration:
        - 0.1s = [*]
        - 0.2s = [**]
        - 0.3s = [***]
        etc.
        """
        segments = result.get('segments', [])
        
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            for word in words:
                marker_text = word.get('word', word.get('text', ''))
                cleaned = marker_text.strip() if isinstance(marker_text, str) else ''
                
                if not cleaned or not DISFLUENCY_MARKER_PATTERN.match(cleaned):
                    continue
                
                start = word.get('start')
                end = word.get('end')
                if start is None or end is None:
                    continue
                
                duration = max(0.0, end - start)
                if duration <= 0:
                    asterisk_count = 1
                else:
                    asterisk_count = max(1, min(50, math.ceil(duration / 0.1)))
                
                new_marker = f"[{'*' * asterisk_count}]"
                word['word'] = new_marker
                if 'text' in word:
                    word['text'] = new_marker
        
        return result
