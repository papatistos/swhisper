"""Utility classes for diarization processing."""

import gc
import sys
import os
import re
import torch
import signal
import atexit
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


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
                marker_text = word.get('word', word.get('text', ''))
                cleaned = marker_text.strip() if isinstance(marker_text, str) else ''
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
        word_mid = word_start + (word_end - word_start) / 2
        check_points = [word_start + 0.01, word_mid, word_end - 0.01]
        
        speaker_votes = {}
        
        for check_time in check_points:
            for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                if turn.start <= check_time < turn.end:
                    speaker_votes[speaker_label] = speaker_votes.get(speaker_label, 0) + 1
                    break
        
        if speaker_votes:
            return max(speaker_votes.items(), key=lambda x: x[1])[0]
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


def is_marker_token(token: str) -> bool:
    """Return True for preserved markers, including variable-length asterisk forms."""
    if not token:
        return False
    cleaned = token.strip()
    return cleaned in STATIC_MARKERS or bool(DISFLUENCY_MARKER_PATTERN.match(cleaned))


logger_manager = LoggerManager()
