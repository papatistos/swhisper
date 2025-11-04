"""Utility classes for diarization processing."""

import gc
import sys
import os
import re
import math
import torch
import signal
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

import numpy as np
import soundfile as sf
from scipy.signal import resample
import json
import whisper_timestamped as whisper
from transcribe.transcription import TranscriptionWorker


class DeviceManager:
    """Manages device memory and cleanup operations."""
    
    @staticmethod
    def clear_device_memory() -> None:
        """Frees up GPU/MPS memory and forces garbage collection."""
        print("Freeing up memory...")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear PyTorch caches
        if hasattr(torch, "mps") and torch.backends.mps.is_built():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force another garbage collection
        gc.collect()


class Logger:
    """Custom logger that writes to both terminal and file."""
    
    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self) -> None:
        self.log_file.close()


class LoggerManager:
    """Manages logger lifecycle and cleanup."""
    
    def __init__(self):
        self.current_logger: Optional[Logger] = None
        self.original_stdout = None
        self._setup_signal_handlers()
        self._setup_cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful cleanup."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, self._signal_handler)
    
    def _setup_cleanup(self) -> None:
        """Register cleanup function to run at exit."""
        atexit.register(self.cleanup_resources)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Received signal {signum}. Cleaning up...")
        self.cleanup_resources()
        sys.exit(0)
    
    def cleanup_resources(self) -> None:
        """Clean up all resources before exit."""
        print("🧹 Cleaning up resources...")
        
        # Restore stdout
        if self.current_logger and self.original_stdout:
            sys.stdout = self.original_stdout
            self.current_logger.close()
        
        # Clear device memory
        DeviceManager.clear_device_memory()
        
        print("✅ Cleanup completed")
    
    @contextmanager
    def safe_logger(self, log_file_path: str):
        """Context manager for safe logger handling."""
        logger = None
        
        try:
            logger = Logger(log_file_path)
            self.current_logger = logger
            self.original_stdout = sys.stdout
            sys.stdout = logger
            yield logger
        finally:
            if logger:
                sys.stdout = self.original_stdout
                logger.close()
                self.current_logger = None


class WordProcessor:
    """Processes words and segments for speaker assignment."""
    
    @staticmethod
    def _get_word_text(word: Dict[str, Any]) -> str:
        """Fetch the primary textual representation for a word token."""
        value = word.get('word')
        if isinstance(value, str) and value:
            return value
        value = word.get('text')
        return value if isinstance(value, str) else ""

    @staticmethod
    def _set_word_text(word: Dict[str, Any], new_text: str) -> None:
        """Synchronize word/text fields after updating marker contents."""
        word['word'] = new_text
        word['text'] = new_text

    @staticmethod
    def align_segment_boundaries_to_words(segments: List[Dict]) -> List[Dict]:
        """
        Adjust segment start/end timestamps to match first/last word timestamps.
        
        This fixes drift that occurs during chunk merging in transcription,
        ensuring segment boundaries perfectly align with their actual word content.
        Silence markers are excluded from boundary calculation.
        """
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            # Filter out silence markers for boundary calculation
            real_words = [w for w in words if not w.get('is_silence_marker', False)]
            
            if real_words:
                # Update segment boundaries to match actual word timestamps
                first_word_start = real_words[0].get('start')
                last_word_end = real_words[-1].get('end')
                
                if first_word_start is not None:
                    segment['start'] = first_word_start
                if last_word_end is not None:
                    segment['end'] = last_word_end
        
        return segments
    
    @staticmethod
    def create_paragraph_text_from_words(
        segment: Dict[str, Any],
        gap_threshold: Optional[float] = None
    ) -> str:
        """Reconstruct segment text from words while optionally adding blank lines around long silences."""
        words = segment.get('words', [])
        if not words:
            return segment.get('text', '').strip()

        # Behaviour when no threshold is provided
        if not gap_threshold or gap_threshold <= 0:
            word_texts = []
            for word in words:
                word_text = word.get('word', word.get('text', ''))
                if word_text:  # Preserve markers and silence tokens
                    word_texts.append(word_text)
            return ' '.join(word_texts).strip()

        lines: List[str] = []
        current_tokens: List[str] = []

        def flush_current_tokens() -> None:
            if current_tokens:
                lines.append(' '.join(current_tokens).strip())
                current_tokens.clear()

        for word in words:
            word_text = word.get('word', word.get('text', ''))
            if not word_text:
                continue

            add_line_breaks = False
            if word.get('is_silence_marker', False):
                duration = WordProcessor._parse_silence_duration(word_text)
                if duration is not None and duration >= gap_threshold:
                    add_line_breaks = True

            if add_line_breaks:
                flush_current_tokens()
                if lines and lines[-1] != '':
                    lines.append('')
                lines.append(word_text.strip())
                lines.append('')
            else:
                current_tokens.append(word_text.strip())

        flush_current_tokens()

        if not lines:
            return ' '.join(current_tokens).strip()

        while lines and lines[0] == '':
            lines.pop(0)
        while lines and lines[-1] == '':
            lines.pop()

        cleaned_lines: List[str] = []
        previous_blank = False
        for line in lines:
            if line == '':
                if previous_blank:
                    continue
                previous_blank = True
            else:
                previous_blank = False
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def _parse_silence_duration(token: str) -> Optional[float]:
        """Extract numeric silence duration from a token like '(1.3)'."""
        if not token:
            return None

        stripped = token.strip()
        if stripped.startswith('(') and stripped.endswith(')'):
            stripped = stripped[1:-1]

        stripped = stripped.replace(',', '.')
        if not stripped:
            return None

        try:
            return float(stripped)
        except ValueError:
            return None
    
    @staticmethod
    def should_smooth_word(current_word: Dict, prev_word: Dict, next_word: Dict) -> bool:
        """
        Determine if a word should be smoothed, with special handling for markers.
        """
        # Don't smooth discontinuity markers - they might legitimately have different speakers
        word_text = current_word.get('word', current_word.get('text', ''))
        if is_marker_token(word_text):
            return False
        
        # Standard smoothing logic
        return (current_word['speaker'] != prev_word['speaker'] and 
                current_word['speaker'] != next_word['speaker'] and
                prev_word['speaker'] == next_word['speaker'] and
                prev_word['speaker'] != "UNKNOWN")

    @staticmethod
    def remove_disfluency_markers_from_segments(segments: List[Dict]) -> List[Dict]:
        """Remove disfluency markers like [*], [**], etc. from segment words and text."""
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue

            filtered_words = []
            removed_any = False
            for word in words:
                marker_text = WordProcessor._get_word_text(word)
                cleaned = strip_marker_hint(marker_text)
                if cleaned and DISFLUENCY_MARKER_PATTERN.match(cleaned):
                    removed_any = True
                    continue
                filtered_words.append(word)

            if removed_any:
                segment['words'] = filtered_words
                segment['text'] = WordProcessor.create_paragraph_text_from_words(segment)

        return segments


class SpeakerAssigner:
    """Handles speaker assignment logic."""
    
    @staticmethod
    def find_speaker_for_word(word_start: float, word_end: float, diarization_result) -> str:
        """
        Find speaker for a single word using its timestamps.
        """
        # Check multiple points in the word for robustness
        segments = list(diarization_result.itertracks(yield_label=True))
        word_mid = word_start + (word_end - word_start) / 2
        check_points = [word_start + 0.01, word_mid, word_end - 0.01]
        
        speaker_votes = {}
        
        for check_time in check_points:
            for turn, _, speaker_label in segments:
                if turn.start <= check_time < turn.end:
                    speaker_votes[speaker_label] = speaker_votes.get(speaker_label, 0) + 1
                    break
        
        if speaker_votes:
            return max(speaker_votes.items(), key=lambda x: x[1])[0]

        # Fall back to nearest-neighbour assignment when no overlap is found
        max_gap = 0.5
        closest_before = (None, float('inf'))
        closest_after = (None, float('inf'))

        for turn, _, speaker_label in segments:
            if turn.end <= word_start:
                distance = word_start - turn.end
                if distance < closest_before[1]:
                    closest_before = (speaker_label, distance)
            elif turn.start >= word_end:
                distance = turn.start - word_end
                if distance < closest_after[1]:
                    closest_after = (speaker_label, distance)

        candidate_label = None
        candidate_distance = float('inf')

        if closest_before[0] is not None and closest_before[1] < candidate_distance:
            candidate_label, candidate_distance = closest_before
        if closest_after[0] is not None and closest_after[1] < candidate_distance:
            candidate_label, candidate_distance = closest_after

        if candidate_label is not None and candidate_distance <= max_gap:
            return candidate_label

        return "UNKNOWN"
    
    @staticmethod
    def assign_segment_speaker_from_words(segment: Dict[str, Any]) -> tuple[str, float]:
        """
        Assign segment speaker based on majority vote of its words.
        """
        try:
            word_speakers = [word.get('speaker', 'UNKNOWN') for word in segment.get('words', [])]
            
            if not word_speakers:
                return "UNKNOWN", 0.0
            
            # Count speaker votes (excluding UNKNOWN)
            speaker_counts = {}
            for speaker in word_speakers:
                if speaker != "UNKNOWN":
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            if speaker_counts:
                # Return speaker with most words
                segment_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                
                # Calculate confidence
                total_known_words = sum(speaker_counts.values())
                confidence = speaker_counts[segment_speaker] / total_known_words
                
                return segment_speaker, confidence
            
            return "UNKNOWN", 0.0
        
        except Exception as e:
            print(f"  Warning: Error in assign_segment_speaker_from_words: {e}")
            print(f"  Segment: {segment.get('start', '?'):.1f}s-{segment.get('end', '?'):.1f}s")
            return "UNKNOWN", 0.0
    
    @staticmethod
    def find_segments_without_word_coverage(segments: List[Dict], diarization_result) -> List[Dict[str, Any]]:
        """Return diarization turns that contain no aligned words."""
        word_intervals: List[tuple[float, float]] = []

        for segment in segments:
            for word in segment.get('words', []):
                if not isinstance(word, dict):
                    continue
                if _is_silence_token(word):
                    continue

                word_start = word.get('start')
                word_end = word.get('end')

                if word_start is None or word_end is None:
                    continue

                word_intervals.append((float(word_start), float(word_end)))

        unmatched_turns: List[Dict[str, Any]] = []

        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            has_overlap = False
            turn_start = float(turn.start)
            turn_end = float(turn.end)

            for word_start, word_end in word_intervals:
                if word_end <= turn_start:
                    continue
                if word_start >= turn_end:
                    continue
                has_overlap = True
                break

            if not has_overlap:
                unmatched_turns.append({
                    'speaker': speaker_label,
                    'start': turn_start,
                    'end': turn_end,
                    'duration': float(turn.duration)
                })

        return unmatched_turns

    @staticmethod
    def smooth_word_level_transitions(segments: List[Dict], min_speaker_words: int = 3) -> List[Dict]:
        """
        Smooth out very short speaker changes at word level.
        """
        total_smoothed = 0
        markers_preserved = 0
        
        for segment in segments:
            words = segment.get('words', [])
            if len(words) < min_speaker_words * 2:
                continue
            
            # Look for isolated speaker words
            for i in range(1, len(words) - 1):
                current_word = words[i]
                prev_word = words[i-1]
                next_word = words[i+1]
                
                word_text = current_word.get('word', current_word.get('text', '[no_text]'))
                
                # Check if this is a discontinuity marker
                if is_marker_token(word_text):
                    markers_preserved += 1
                    continue  # Don't smooth markers - they might legitimately change speakers
                
                # Apply standard smoothing logic
                if WordProcessor.should_smooth_word(current_word, prev_word, next_word):
                    word_time = current_word.get('start', 0)
                    print(f"  Smoothing word '{word_text}' at {word_time:.1f}s: {current_word['speaker']} -> {prev_word['speaker']}")
                    words[i]['speaker'] = prev_word['speaker']
                    total_smoothed += 1
            
            # Reassign segment speaker after smoothing
            segment_speaker, confidence = SpeakerAssigner.assign_segment_speaker_from_words(segment)
            segment['speaker'] = segment_speaker
            segment['speaker_confidence'] = confidence
        
        print(f"Smoothed {total_smoothed} isolated word assignments")
        if markers_preserved > 0:
            print(f"Preserved {markers_preserved} discontinuity markers without smoothing")
        return segments


class BackfillTranscriber:
    """Transcribe diarization turns that lack aligned words."""

    def __init__(
        self,
        audio_path: str,
        model_name: str,
        device: Optional[str],
        whisper_settings: Dict[str, Any],
        overlap_duration: float = 0.5,
        snippet_output_dir: Optional[str] = None,
        snippet_prefix: Optional[str] = None,
        cache: Optional['BackfillCache'] = None,
    ) -> None:
        self.audio_path = audio_path
        self.model_name = model_name
        self.device = device or "cpu"
        self.settings = dict(whisper_settings or {})
        self.settings.setdefault('verbose', False)
        self.overlap_duration = max(0.0, overlap_duration)
        self.sample_rate = 16000
        self._model = None
        self.snippet_output_dir = Path(snippet_output_dir).expanduser() if snippet_output_dir else None
        if self.snippet_output_dir:
            self.snippet_output_dir.mkdir(parents=True, exist_ok=True)
        self.snippet_prefix = snippet_prefix or "backfill"
        self._snippet_counter = 0
        self.cache = cache

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._model = whisper.load_model(self.model_name, device=self.device)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            DeviceManager.clear_device_memory()

    def _maybe_save_audio_snippet(
        self,
        audio: Optional[np.ndarray],
        speaker: str,
        start: float,
        end: float,
        start_sample: int
    ) -> None:
        if self.snippet_output_dir is None:
            return
        if audio is None or len(audio) == 0:
            return

        try:
            padded_start = start_sample / self.sample_rate
            rel_start = max(0, int(round((start - padded_start) * self.sample_rate)))
            rel_end = max(rel_start + 1, int(round((end - padded_start) * self.sample_rate)))
            rel_end = min(rel_end, len(audio))
            snippet_audio = audio[rel_start:rel_end]
            if snippet_audio.size == 0:
                snippet_audio = audio

            safe_speaker = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker or "UNKNOWN")
            start_ms = int(round(start * 1000))
            end_ms = int(round(end * 1000))
            filename = f"{self.snippet_prefix}_{safe_speaker}_{start_ms:07d}-{end_ms:07d}_{self._snippet_counter:04d}.wav"
            file_path = self.snippet_output_dir / filename
            sf.write(file_path, snippet_audio, self.sample_rate, subtype='PCM_16')
            self._snippet_counter += 1
            print(f"      🎧 Saved backfill snippet to {file_path}")
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"      ⚠️ Could not save backfill audio snippet: {exc}")

    def transcribe_turns(self, turns: List[Dict[str, Any]], transcript_key: Optional[str] = None) -> Dict[str, Any]:
        recovered_segments: List[Dict[str, Any]] = []
        recovered_turns: List[Dict[str, Any]] = []
        failed_turns: List[Dict[str, Any]] = []
        total_words = 0
        placeholder_segments = 0

        # Load cached segments if available
        cached_segments = {}
        cache_hits = 0
        if self.cache and transcript_key:
            cached_segments = self.cache.load(transcript_key)
            if cached_segments:
                print(f"  -> Loaded {len(cached_segments)} cached backfill segments")

        # Segments to save to cache (only newly transcribed ones)
        newly_transcribed = {}

        for turn in turns:
            speaker = turn.get('speaker', 'UNKNOWN')
            start = float(turn.get('start', 0.0))
            end = float(turn.get('end', start))
            duration = max(0.0, end - start)

            # Create a cache key for this turn
            turn_key = f"{speaker}_{start:.3f}_{end:.3f}"

            # Check cache first
            if turn_key in cached_segments:
                segment = cached_segments[turn_key]
                word_count = len(segment.get('words', []))
                cache_hits += 1
                print(f"----> Using cached result for {speaker}: {start:.2f}s - {end:.2f}s (duration {duration:.2f}s)")
            else:
                print(f"----> Transcribing {speaker}: {start:.2f}s - {end:.2f}s (duration {duration:.2f}s)")
                try:
                    segment, word_count = self._transcribe_single_turn(turn)
                    # Store newly transcribed segment for caching
                    if segment:
                        newly_transcribed[turn_key] = segment
                except Exception as exc:  # pragma: no cover - logging path
                    print(f"   -> Backfill error for {speaker} {start:.2f}-{end:.2f}s: {exc}")
                    segment, word_count = None, 0

            is_placeholder = bool(segment and segment.get('is_placeholder'))

            if segment and (word_count or is_placeholder):
                recovered_segments.append(segment)
                enriched_turn = dict(turn)
                enriched_turn['word_count'] = word_count
                if is_placeholder:
                    enriched_turn['is_placeholder'] = True
                recovered_turns.append(enriched_turn)
                total_words += word_count

                if is_placeholder:
                    placeholder_segments += 1
                    marker_text = segment.get('text', '').strip() or '[ * ]'
                    print(f"      Words: {marker_text} (placeholder)")
                else:
                    word_tokens = [
                        (w.get('word') if isinstance(w.get('word'), str) and w.get('word').strip()
                         else w.get('text', '')).strip()
                        for w in segment.get('words', [])
                        if isinstance(w, dict)
                    ]
                    filtered_tokens = [token for token in word_tokens if token]
                    if filtered_tokens:
                        print(f"      Words: {' '.join(filtered_tokens)}")
                    else:
                        print("      Words: [none detected]")
            else:
                failed_turns.append(turn)
                print("      Words: [none detected]")

        # Save newly transcribed segments to cache
        if self.cache and transcript_key and newly_transcribed:
            # Merge with existing cache
            merged_cache = {**cached_segments, **newly_transcribed}
            self.cache.save(transcript_key, merged_cache)
            print(f"  -> Saved {len(newly_transcribed)} new segments to cache")

        if cache_hits > 0:
            print(f"  -> Cache hits: {cache_hits}/{len(turns)} turns")

        return {
            'segments': recovered_segments,
            'recovered_turns': recovered_turns,
            'failed_turns': failed_turns,
            'word_count': total_words,
            'placeholders': placeholder_segments
        }

    def _transcribe_single_turn(self, turn: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], int]:
        start = float(turn.get('start', 0.0))
        end = float(turn.get('end', start))
        if end <= start:
            return None, 0

        speaker_label = turn.get('speaker', 'UNKNOWN')
        self._ensure_model_loaded()

        chunk_audio, start_sample = self._load_audio_chunk(start, end)
        if chunk_audio is None or len(chunk_audio) == 0:
            return None, 0

        self._maybe_save_audio_snippet(
            chunk_audio,
            speaker_label,
            start,
            end,
            start_sample
        )

        result = whisper.transcribe(self._model, chunk_audio, **self.settings)

        time_offset = start_sample / self.sample_rate

        segments = result.get('segments', [])
        for segment in segments:
            if 'start' in segment:
                segment['start'] += time_offset
            if 'end' in segment:
                segment['end'] += time_offset
            for word in segment.get('words', []):
                if 'start' in word:
                    word['start'] += time_offset
                if 'end' in word:
                    word['end'] += time_offset

        TranscriptionWorker._scale_disfluency_markers(result)

        collected_words: List[Dict[str, Any]] = []

        for segment in segments:
            for word in segment.get('words', []):
                word_start = word.get('start')
                word_end = word.get('end')
                if word_start is None or word_end is None:
                    continue
                if word_end <= start or word_start >= end:
                    continue

                word_text = word.get('word', word.get('text', '')).strip()
                if not word_text:
                    continue

                clipped_word = dict(word)
                clipped_word['start'] = max(start, float(word_start))
                clipped_word['end'] = min(end, float(word_end))
                clipped_word['speaker'] = speaker_label
                clipped_word['is_backfill'] = True
                collected_words.append(clipped_word)

        if not collected_words:
            placeholder_segment = self._build_placeholder_segment(turn, speaker_label)
            return placeholder_segment, 0

        collected_words.sort(key=lambda item: item.get('start', 0.0))

        text_tokens = []
        for word in collected_words:
            token = word.get('word')
            if not isinstance(token, str):
                token = word.get('text', '')
            token = token.strip() if isinstance(token, str) else ''
            if token:
                text_tokens.append(token)

        segment_data = {
            'start': collected_words[0]['start'],
            'end': collected_words[-1]['end'],
            'text': ' '.join(text_tokens),
            'speaker': speaker_label,
            'speaker_confidence': 1.0,
            'words': collected_words,
            'is_backfill': True
        }

        # Rescale any disfluency markers based on the clipped durations.
        temp_result = {'segments': [segment_data]}
        TranscriptionWorker._scale_disfluency_markers(temp_result)

        return temp_result['segments'][0], len(collected_words)

    def _build_placeholder_segment(self, turn: Dict[str, Any], speaker_label: str) -> Dict[str, Any]:
        start = float(turn.get('start', 0.0))
        end = float(turn.get('end', start))
        duration = max(0.0, end - start)
        if duration <= 0.0:
            duration = 0.1

        asterisk_count = max(1, min(50, math.ceil(duration / 0.1)))
        marker_token = f"[ {'*' * asterisk_count} ]"

        placeholder_word = {
            'start': start,
            'end': end,
            'word': marker_token,
            'speaker': speaker_label,
            'confidence': 0.0,
            'is_backfill': True,
            'is_placeholder': True
        }

        return {
            'start': start,
            'end': end,
            'text': marker_token,
            'speaker': speaker_label,
            'speaker_confidence': 0.0,
            'words': [placeholder_word],
            'is_backfill': True,
            'is_placeholder': True
        }

    def _load_audio_chunk(self, start: float, end: float) -> tuple[Optional[np.ndarray], int]:
        margin = self.overlap_duration
        padded_start = max(0.0, start - margin)
        padded_end = max(padded_start + 0.05, end + margin)

        start_sample = int(round(padded_start * self.sample_rate))
        end_sample = int(round(padded_end * self.sample_rate))

        if end_sample <= start_sample:
            end_sample = start_sample + int(self.sample_rate * 0.1)

        with sf.SoundFile(self.audio_path) as audio_file:
            orig_sr = audio_file.samplerate
            orig_start = int(round(start_sample * orig_sr / self.sample_rate))
            orig_end = int(round(end_sample * orig_sr / self.sample_rate))
            if orig_end <= orig_start:
                return None, start_sample

            audio_file.seek(orig_start)
            chunk_audio = audio_file.read(orig_end - orig_start, dtype='float32')

        if chunk_audio is None or len(chunk_audio) == 0:
            return None, start_sample

        if len(chunk_audio.shape) > 1:
            chunk_audio = np.mean(chunk_audio, axis=1)

        if orig_sr != self.sample_rate:
            target_length = int(len(chunk_audio) * self.sample_rate / orig_sr)
            if target_length <= 0:
                return None, start_sample
            chunk_audio = resample(chunk_audio, target_length)

        return chunk_audio.astype(np.float32), start_sample


class SilenceMarkerProcessor:
    """Handles silence marker detection and insertion."""
    
    @staticmethod
    def add_word_level_silence_markers(
        segments: List[Dict],
        min_silence_duration: float = 0.2,
        gap_log_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Add silence markers between words where gaps are longer than min_silence_duration.
        Detects gaps both within segments and between consecutive segments.
        Cross-segment silences are created as standalone segments.
        
        Args:
            segments: List of transcript segments
            min_silence_duration: Minimum silence duration in seconds to mark
            gap_log_path: Optional path to log all gap durations
        
        Returns:
            List of segments with silence markers inserted (within-segment as words, 
            cross-segment as separate segments)
        """
        if not segments:
            return segments
        
        enhanced_segments = []
        gap_log_entries: List[str] = []
        
        for segment_index, segment in enumerate(segments):
            words = segment.get('words', [])
            if not words:
                # If no words, just add the segment as-is
                enhanced_segments.append(segment)
                continue
            
            # Create a new segment with word-level silence detection
            new_segment = segment.copy()
            enhanced_words = []
            
            for i, word in enumerate(words):
                # Add the current word
                enhanced_words.append(word)
                
                # Check if current word is a [*] marker
                word_text = word.get('word', word.get('text', ''))
                is_disfluency_marker = is_marker_token(word_text)
                
                # Check for gap to next word within same segment
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    
                    # Calculate gap between current word end and next word start
                    current_end = word.get('end', word.get('start', 0))
                    next_start = next_word.get('start', next_word.get('end', current_end))
                    
                    gap_duration = next_start - current_end
                    
                    if gap_log_path is not None:
                        gap_log_entries.append(f"{gap_duration:.6f}")
                    
                    # Only add silence marker if gap meets minimum threshold
                    if gap_duration >= min_silence_duration:
                        # Round to nearest 0.1 seconds
                        rounded_gap = round(gap_duration, 1)
                        
                        # Format the silence duration (omit leading zero for values < 1.0)
                        silence_text = f"({rounded_gap:.1f})"
                        silence_text = silence_text.replace("(0.", "(.")
                        
                        # Create silence "word" for within-segment gap
                        silence_word = {
                            'start': current_end,
                            'end': next_start,
                            'word': silence_text,
                            'speaker': 'SILENCE',
                            'confidence': 1.0,
                            'is_silence_marker': True,
                            'after_disfluency': is_disfluency_marker  # Flag to track origin
                        }
                        
                        enhanced_words.append(silence_word)
            
            # Update the segment with enhanced words
            new_segment['words'] = enhanced_words
            enhanced_segments.append(new_segment)
            
            # Check for cross-segment gap (after adding current segment)
            if segment_index + 1 < len(segments):
                next_segment = segments[segment_index + 1]
                next_segment_words = next_segment.get('words', [])
                
                if enhanced_words and next_segment_words:
                    # Get last word of current segment and first word of next segment
                    last_word = enhanced_words[-1]
                    first_next_word = next_segment_words[0]
                    
                    current_end = last_word.get('end', last_word.get('start', 0))
                    next_start = first_next_word.get('start', first_next_word.get('end', current_end))
                    
                    gap_duration = next_start - current_end
                    
                    if gap_log_path is not None:
                        gap_log_entries.append(f"{gap_duration:.6f}")
                    
                    # Create standalone silence segment if gap meets threshold
                    if gap_duration >= min_silence_duration:
                        rounded_gap = round(gap_duration, 1)
                        silence_text = f"({rounded_gap:.1f})"
                        silence_text = silence_text.replace("(0.", "(.")
                        
                        # Create a standalone silence segment
                        silence_segment = {
                            'start': current_end,
                            'end': next_start,
                            'text': silence_text,
                            'speaker': 'SILENCE',
                            'speaker_confidence': 1.0,
                            'is_silence_marker': True,
                            'words': [{
                                'start': current_end,
                                'end': next_start,
                                'word': silence_text,
                                'speaker': 'SILENCE',
                                'confidence': 1.0,
                                'is_silence_marker': True
                            }]
                        }
                        
                        enhanced_segments.append(silence_segment)
        
        if gap_log_path is not None and gap_log_entries:
            SilenceMarkerProcessor._write_gap_log(gap_log_path, gap_log_entries)
        
        return enhanced_segments

    @staticmethod
    def _write_gap_log(log_path: str, entries: List[str]) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(entries))
            log_file.write("\n")
            log_file.write("\n")


# Global logger manager instance
STATIC_MARKERS = {"[DISCONTINUITY]", "[SILENCE]", "[OVERLAP]"}
DISFLUENCY_MARKER_PATTERN = re.compile(r"^\[\*+\]$")
SPACED_MARKER_PATTERN = re.compile(r"^\[\s\*{1,50}\s\]$")
MARKER_INLINE_HINT_PATTERN = re.compile(r"^\[(\s*)(S\d{2}(?:[+/]S\d{2})*)(\*+)(\s*)\]$")
MARKER_PREFIX_HINT_PATTERN = re.compile(r"^(S\d{2}(?:[+/]S\d{2})*)\s+")


def strip_marker_hint(token: str) -> str:
    """Remove leading speaker hint prefixes from a marker token."""
    if not isinstance(token, str):
        return ""
    cleaned = token.strip()
    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        return f"[{leading_spaces}{stars}{trailing_spaces}]"
    return MARKER_PREFIX_HINT_PATTERN.sub("", cleaned)


def split_marker_hint_and_body(token: str) -> Tuple[str, str]:
    """Return the hint prefix (if present) and the base marker body."""
    if not isinstance(token, str):
        return "", ""
    cleaned = token.strip()
    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        hint = inline.group(2)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        base_marker = f"[{leading_spaces}{stars}{trailing_spaces}]"
        return hint, base_marker
    match = MARKER_PREFIX_HINT_PATTERN.match(cleaned)
    if match:
        hint = match.group(1)
        remainder = cleaned[match.end():].lstrip()
        return hint, remainder
    return "", cleaned


def inject_hint_into_marker(marker_body: str, hint: str) -> str:
    """Place the speaker hint immediately inside the marker brackets."""
    if not marker_body or not hint:
        return marker_body

    cleaned = marker_body.strip()

    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        return f"[{leading_spaces}{hint}{stars}{trailing_spaces}]"

    if cleaned.startswith('[') and cleaned.endswith(']'):
        inner = cleaned[1:-1]
        if inner:
            first_star = inner.find('*')
            if first_star != -1:
                return f"[{inner[:first_star]}{hint}{inner[first_star:]}]"
            if set(inner.replace(' ', '')) <= {'*'}:
                return f"[{hint}{inner}]"

    return f"{hint} {marker_body}".strip()


def format_speaker_hint(label: Optional[str]) -> str:
    """Convert diarization speaker labels into the compact Sxx hint form."""
    if not label or not isinstance(label, str):
        return ""

    if label.upper() in {"SILENCE", "UNKNOWN"}:
        return ""

    match = re.search(r"(\d+)", label)
    if match:
        try:
            index = int(match.group(1))
            return f"S{index + 1:02d}"
        except ValueError:
            pass

    alnum = re.sub(r"[^A-Za-z0-9]", "", label).upper()
    if not alnum:
        return ""
    if len(alnum) <= 3:
        return alnum
    return alnum[:3]


def is_marker_token(token: str) -> bool:
    """Return True for preserved markers, including variable-length asterisk forms."""
    if not token:
        return False
    cleaned = strip_marker_hint(token)
    return (
        cleaned in STATIC_MARKERS
        or bool(DISFLUENCY_MARKER_PATTERN.match(cleaned))
        or bool(SPACED_MARKER_PATTERN.match(cleaned))
    )


def _is_silence_token(word: Dict[str, Any]) -> bool:
    """Identify silence placeholders regardless of explicit flags."""
    if not isinstance(word, dict):
        return False

    if word.get('is_silence_marker', False):
        return True

    if word.get('speaker') == 'SILENCE':
        return True

    token = word.get('word', word.get('text', ''))
    if not isinstance(token, str):
        return False

    cleaned = token.strip()
    if not cleaned:
        return False

    if cleaned.startswith('(') and cleaned.endswith(')'):
        if WordProcessor._parse_silence_duration(cleaned) is not None:
            return True

    return False


logger_manager = LoggerManager()


class BackfillCache:
    """Persist and retrieve backfill transcription results for reuse."""

    INDEX_FILENAME = "cache_index.json"

    def __init__(self, cache_dir: str, audio_path: str, model: str, device: str, overlap: float) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = Path(audio_path)
        self.model = model
        self.device = device
        self.overlap = overlap
        self.index_path = self.cache_dir / self.INDEX_FILENAME
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            with self.index_path.open('r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _write_index(self) -> None:
        tmp_path = self.index_path.with_suffix('.tmp')
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(self._index, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.index_path)

    def _audio_signature(self) -> Dict[str, Any]:
        try:
            stat = self.audio_path.stat()
            return {
                'path': str(self.audio_path.resolve()),
                'size': stat.st_size,
                'mtime': int(stat.st_mtime)
            }
        except OSError:
            return {
                'path': str(self.audio_path),
                'size': None,
                'mtime': None
            }

    def _config_signature(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'device': self.device,
            'overlap': float(self.overlap)
        }

    def _cache_file_path(self, transcript_key: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", transcript_key)
        return self.cache_dir / f"{safe_key}.json"

    def load(self, transcript_key: str) -> Dict[str, Any]:
        cache_path = self._cache_file_path(transcript_key)
        if not cache_path.exists():
            return {}

        try:
            with cache_path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return {}

        audio_sig = payload.get('audio')
        config_sig = payload.get('config')
        if audio_sig != self._audio_signature() or config_sig != self._config_signature():
            return {}

        segments = payload.get('segments', {})
        if not isinstance(segments, dict):
            return {}
        return segments

    def save(self, transcript_key: str, segments: Dict[str, Dict[str, Any]]) -> None:
        if not segments:
            return
        cache_path = self._cache_file_path(transcript_key)
        payload = {
            'audio': self._audio_signature(),
            'config': self._config_signature(),
            'segments': segments
        }
        tmp_path = cache_path.with_suffix('.tmp')
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(cache_path)

