"""Transcription processing with chunking and subprocess management."""

import os
import time
import math
import copy
import difflib
import string
import multiprocessing as mp
import threading
import gc
import re
import whisper_timestamped as whisper
import soundfile as sf
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.signal import resample

from .config import TranscriptionConfig, WhisperSettings
from .memory_utils import ResourceManager, MemoryMonitor
from .audio_utils import AudioProcessor


DISFLUENCY_MARKER_PATTERN = re.compile(r"^\[\*+\]$")


class TranscriptionWorker:
    """Handle transcription in subprocess."""
    
    @staticmethod
    def chunk_worker(audiofile_path: str, start_time: float, end_time: float, 
                    model_path: str, settings: Dict[str, Any], output_queue: mp.Queue, 
                    device: str = "mps", overlap_duration: float = 2.0):
        """Worker function for processing audio chunks in subprocess."""
        try:
            # Import everything needed in the subprocess
            import whisper_timestamped as whisper
            import soundfile as sf
            import numpy as np
            import gc
            from scipy.signal import resample
            
            print(f"    🔧 Loading model {model_path} on {device}...")
            
            # Use the same device as main process
            model = whisper.load_model(model_path, device=device)
            
            print(f"    🔧 Loading audio chunk {start_time:.1f}s-{end_time:.1f}s...")
            
            # Load and process chunk
            sample_rate = 16000
            effective_overlap = max(0.0, overlap_duration or 0.0)
            overlap_samples = int(effective_overlap * sample_rate)
            start_sample = max(0, int(start_time * sample_rate) - overlap_samples)
            end_sample = int(end_time * sample_rate) + overlap_samples
            
            # Load chunk using soundfile
            with sf.SoundFile(audiofile_path) as f:
                orig_sr = f.samplerate
                orig_start = int(start_sample * orig_sr / sample_rate)
                orig_end = int(end_sample * orig_sr / sample_rate)
                
                f.seek(orig_start)
                chunk_audio = f.read(orig_end - orig_start, dtype='float32')
                
                # Convert to mono if stereo
                if len(chunk_audio.shape) > 1:
                    chunk_audio = np.mean(chunk_audio, axis=1)
                
                # Resample if needed
                target_length = int(len(chunk_audio) * 16000 / orig_sr)
                chunk_audio = resample(chunk_audio, target_length)
            
            chunk_audio = chunk_audio.astype(np.float32)
            
            print(f"    🔧 Transcribing...")
            
            # Transcribe chunk
            result = whisper.transcribe(model, chunk_audio, **settings)
            
            print(f"    🔧 Adjusting timestamps...")
            
            # Adjust timestamps to account for the actual start position
            time_offset = start_sample / sample_rate
            if 'segments' in result:
                for segment in result['segments']:
                    segment['start'] += time_offset
                    segment['end'] += time_offset
                    if 'words' in segment:
                        for word in segment['words']:
                            word['start'] += time_offset
                            word['end'] += time_offset

            if 'speech_activity' in result:
                adjusted_vad = []
                for span in result['speech_activity']:
                    start = span.get('start')
                    end = span.get('end')
                    if start is None or end is None:
                        continue
                    adjusted_vad.append({'start': start + time_offset, 'end': end + time_offset})
                result['speech_activity'] = adjusted_vad

            TranscriptionWorker._scale_disfluency_markers(result)
            
            # Clean up before returning
            del chunk_audio
            del model
            gc.collect()
            
            print(f"    ✅ Transcription of this chunk completed successfully")
            output_queue.put((result, time_offset))
            
        except Exception as e:
            import traceback
            error_msg = f"Subprocess error: {str(e)}\n{traceback.format_exc()}"
            print(f"    ❌ Subprocess error: {str(e)}")
            output_queue.put((e, error_msg))

    @staticmethod
    def _scale_disfluency_markers(result: Dict[str, Any]) -> None:
        """Adjust disfluency markers to reflect duration via repeated asterisks."""
        if not result:
            return

        segments = result.get('segments', [])
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue

            segment_updated = False
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
                if word.get('word') != new_marker:
                    word['word'] = new_marker
                    segment_updated = True
                if 'text' in word and word.get('text') != new_marker:
                    word['text'] = new_marker

            if segment_updated:
                rebuilt: List[str] = []
                for w in words:
                    text_value = w.get('word', w.get('text', ''))
                    if not isinstance(text_value, str):
                        continue
                    stripped = text_value.strip()
                    if stripped:
                        rebuilt.append(stripped)

                segment['text'] = ' '.join(rebuilt) if rebuilt else ''


class ChunkProcessor:
    """Process audio chunks using subprocesses."""
    
    def __init__(self, config: TranscriptionConfig, whisper_settings: WhisperSettings, 
                 resource_manager: ResourceManager):
        self.config = config
        self.whisper_settings = whisper_settings
        self.resource_manager = resource_manager
    
    def process_chunk_in_subprocess(self, audiofile_path: str, start_time: float, 
                                  end_time: float, chunk_id: int) -> Tuple[Dict[str, Any], float]:
        """Process a single chunk in a subprocess."""
        print(f"    🚀 Starting subprocess for chunk {chunk_id+1}...")
        
        # Create subprocess
        output_queue = mp.Queue()
        process = mp.Process(
            target=TranscriptionWorker.chunk_worker, 
            args=(
                audiofile_path,
                start_time,
                end_time,
                self.config.model_str,
                self.whisper_settings.to_dict(),
                output_queue,
                self.config.device,
                self.config.overlap_duration,
            )
        )
        
        process.start()
        self.resource_manager.current_subprocess = process
        
        # Monitor memory
        memory_snapshots = []
        
        def monitor_worker():
            time.sleep(5)  # Let subprocess start up
            while process.is_alive():
                memory_info = self._get_subprocess_memory_info(process)
                memory_snapshots.append((time.time(), memory_info))
                time.sleep(10)  # Check every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_worker)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for completion
        process.join(timeout=600)
        
        # Show peak memory usage
        if memory_snapshots:
            peak_mem = self._get_max_memory(memory_snapshots)
            print(f"    📊 Memory usage during transcription ≈ {peak_mem}")
        
        if process.is_alive():
            print(f"    ⚠️ Subprocess timeout, terminating...")
            process.terminate()
            process.join()
            raise Exception("Subprocess timed out after 10 minutes")
        
        # Check if process completed successfully
        if process.exitcode != 0:
            raise Exception(f"Subprocess failed with exit code {process.exitcode}")
        
        # Get result
        if output_queue.empty():
            raise Exception("Subprocess completed but returned no result")
        
        result = output_queue.get()
        
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Exception):
            print(f"    ❌ Worker error: {result[1]}")
            raise result[0]
        
        self.resource_manager.current_subprocess = None
        return result
    
    def _get_subprocess_memory_info(self, process: mp.Process) -> str:
        """Get memory info as a string without printing."""
        try:
            if process and process.is_alive():
                import psutil
                subprocess_psutil = psutil.Process(process.pid)
                rss_mb = subprocess_psutil.memory_info().rss / 1024 / 1024
                return f"RSS: {rss_mb:.0f}MB"
            return "No process"
        except Exception:
            return "Process ended"
    
    def _get_max_memory(self, memory_snapshots: List[Tuple[float, str]]) -> str:
        """Find the snapshot with highest memory usage."""
        if not memory_snapshots:
            return "No data"
        
        max_item = max(memory_snapshots, key=lambda snap: MemoryMonitor.parse_dirty_memory(snap[1]))
        return max_item[1]


class ResultMerger:
    """Merge transcription results from multiple chunks."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
    
    def merge_chunk_results(self, chunk_results: List[Tuple[Dict[str, Any], float]], 
                          boundaries: List[float]) -> Dict[str, Any]:
        """Merge results from multiple chunks, handling overlaps intelligently."""
        if not chunk_results:
            return {}
        
        base_result = copy.deepcopy(chunk_results[0][0])
        merged_segments = base_result.get('segments', []) or []

        # Ensure base segments have consistent metadata
        for segment in merged_segments:
            self._update_segment_metadata(segment)

        all_vad_segments: List[Dict[str, Any]] = list(base_result.get('speech_activity', []))

        for chunk_index in range(1, len(chunk_results)):
            chunk_result_copy = copy.deepcopy(chunk_results[chunk_index][0])
            all_vad_segments.extend(chunk_result_copy.get('speech_activity', []))

            new_segments = chunk_result_copy.get('segments', []) or []
            if not new_segments:
                continue

            boundary = boundaries[chunk_index]
            overlap = getattr(self.config, 'overlap_duration', 0.0) or 0.0

            self._merge_segments_with_overlap(
                merged_segments,
                new_segments,
                boundary,
                overlap
            )

        # Sort and clean merged segments
        merged_segments = [seg for seg in merged_segments if seg.get('words')]
        merged_segments.sort(key=lambda seg: seg.get('start', 0.0))

        # Reassign sequential segment ids to keep ordering consistent
        for idx, segment in enumerate(merged_segments):
            segment['id'] = idx
            self._update_segment_metadata(segment)

        merged_text = ' '.join(
            segment.get('text', '').strip()
            for segment in merged_segments
            if segment.get('text')
        ).strip()

        merged_result = {
            'text': merged_text,
            'segments': merged_segments,
            'language': base_result.get('language', ''),
            'speech_activity': self._merge_vad_segments(all_vad_segments)
        }

        return merged_result

    def _merge_segments_with_overlap(
        self,
        merged_segments: List[Dict[str, Any]],
        new_segments: List[Dict[str, Any]],
        boundary: float,
        overlap: float
    ) -> None:
        """Merge new segments into the running transcript with overlap reconciliation."""
        if not new_segments:
            return

        if not merged_segments:
            # Nothing to reconcile, take the segments as-is.
            for segment in copy.deepcopy(new_segments):
                self._update_segment_metadata(segment)
                merged_segments.append(segment)
            return

        window_start = max(0.0, boundary - overlap)
        window_end = boundary + overlap

        # Prepare copies of the new segments so we can safely mutate them
        new_segments_copy = copy.deepcopy(new_segments)

        left_refs = self._collect_word_refs(merged_segments, window_start, window_end, boundary, side='left')
        right_refs, right_keep_keys = self._prepare_right_word_refs(
            new_segments_copy, window_start, window_end, boundary
        )

        remove_word_ids: set[int] = set()

        if left_refs and right_refs:
            alignment_ops = self._align_overlap_words(left_refs, right_refs, boundary)
            resolved_remove_ids, resolved_keep_keys = self._resolve_alignment(
                alignment_ops,
                left_refs,
                right_refs,
                boundary
            )
            remove_word_ids.update(resolved_remove_ids)
            right_keep_keys.update(resolved_keep_keys)
        elif not left_refs and right_refs:
            # No overlap on the left, keep everything from the right window
            right_keep_keys.update(ref['key'] for ref in right_refs)

        if remove_word_ids:
            self._remove_words_by_id(merged_segments, remove_word_ids)

        filtered_segments = self._filter_new_segments(
            new_segments_copy,
            right_keep_keys
        )

        merged_segments.extend(filtered_segments)
        merged_segments[:] = [segment for segment in merged_segments if segment.get('words')]
        merged_segments.sort(key=lambda seg: seg.get('start', 0.0))

    def _collect_word_refs(
        self,
        segments: List[Dict[str, Any]],
        window_start: float,
        window_end: float,
        boundary: float,
        side: str
    ) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for segment_idx, segment in enumerate(segments):
            words = segment.get('words', []) or []
            for word_idx, word in enumerate(words):
                if not isinstance(word, dict):
                    continue
                start = float(word.get('start', segment.get('start', 0.0)))
                end = float(word.get('end', start))
                if end <= window_start or start >= window_end:
                    continue
                refs.append(self._build_word_ref(word, segment_idx, word_idx, boundary, side))
        refs.sort(key=lambda ref: (ref['start'], ref['end']))
        return refs

    def _prepare_right_word_refs(
        self,
        segments: List[Dict[str, Any]],
        window_start: float,
        window_end: float,
        boundary: float
    ) -> Tuple[List[Dict[str, Any]], set[Tuple[int, int]]]:
        overlap_refs: List[Dict[str, Any]] = []
        keep_keys: set[Tuple[int, int]] = set()

        for segment_idx, segment in enumerate(segments):
            words = segment.get('words', []) or []
            for word_idx, word in enumerate(words):
                if not isinstance(word, dict):
                    continue
                start = float(word.get('start', segment.get('start', 0.0)))
                end = float(word.get('end', start))
                key = (segment_idx, word_idx)

                if end <= window_start:
                    # Completely precedes the overlap – drop it
                    continue
                if start >= window_end:
                    keep_keys.add(key)
                    continue

                overlap_refs.append(self._build_word_ref(word, segment_idx, word_idx, boundary, side='right'))

        overlap_refs.sort(key=lambda ref: (ref['start'], ref['end']))
        return overlap_refs, keep_keys

    def _build_word_ref(
        self,
        word: Dict[str, Any],
        segment_idx: int,
        word_idx: int,
        boundary: float,
        side: str
    ) -> Dict[str, Any]:
        start = float(word.get('start', 0.0))
        end = float(word.get('end', start))
        text = self._word_text(word)
        norm = self._normalize_token(text)
        distance = (
            max(0.0, boundary - end) if side == 'left'
            else max(0.0, start - boundary)
        )

        return {
            'word': word,
            'segment_idx': segment_idx,
            'word_idx': word_idx,
            'start': start,
            'end': end,
            'mid': (start + end) / 2.0,
            'text': text,
            'norm': norm,
            'confidence': float(word.get('confidence', 0.0)),
            'distance': distance,
            'side': side,
            'key': (segment_idx, word_idx)
        }

    def _align_overlap_words(
        self,
        left_refs: List[Dict[str, Any]],
        right_refs: List[Dict[str, Any]],
        boundary: float
    ) -> List[Tuple[str, Optional[int], Optional[int]]]:
        n = len(left_refs)
        m = len(right_refs)
        if n == 0 and m == 0:
            return []

        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        back: List[List[Optional[Tuple[str, Optional[int], Optional[int]]]]] = [
            [None] * (m + 1) for _ in range(n + 1)
        ]

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + self._delete_cost(left_refs[i - 1])
            back[i][0] = ('delete', i - 1, None)
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + self._insert_cost(right_refs[j - 1])
            back[0][j] = ('insert', None, j - 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_cost = dp[i - 1][j - 1] + self._match_cost(left_refs[i - 1], right_refs[j - 1])
                delete_cost = dp[i - 1][j] + self._delete_cost(left_refs[i - 1])
                insert_cost = dp[i][j - 1] + self._insert_cost(right_refs[j - 1])

                best_cost = match_cost
                best_op = ('match', i - 1, j - 1)

                if delete_cost < best_cost:
                    best_cost = delete_cost
                    best_op = ('delete', i - 1, None)

                if insert_cost < best_cost:
                    best_cost = insert_cost
                    best_op = ('insert', None, j - 1)

                dp[i][j] = best_cost
                back[i][j] = best_op

        ops: List[Tuple[str, Optional[int], Optional[int]]] = []
        i, j = n, m
        while i > 0 or j > 0:
            op = back[i][j]
            if op is None:
                break
            ops.append(op)
            action, li, rj = op
            if action == 'match':
                i -= 1
                j -= 1
            elif action == 'delete':
                i -= 1
            elif action == 'insert':
                j -= 1
        ops.reverse()
        return ops

    def _match_cost(self, left_ref: Dict[str, Any], right_ref: Dict[str, Any]) -> float:
        time_diff = abs(left_ref['mid'] - right_ref['mid'])
        time_penalty = min(time_diff, 0.5) * 2.0

        left_norm = left_ref['norm']
        right_norm = right_ref['norm']

        if not left_norm and not right_norm:
            text_penalty = 0.2
        elif self._is_disfluency_marker_norm(left_norm) and not self._is_disfluency_marker_norm(right_norm):
            text_penalty = 0.7
        elif self._is_disfluency_marker_norm(right_norm) and not self._is_disfluency_marker_norm(left_norm):
            text_penalty = 0.1
        elif left_norm == right_norm:
            text_penalty = 0.0
        else:
            similarity = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
            if similarity >= 0.85:
                text_penalty = 0.1
            elif similarity >= 0.6:
                text_penalty = 0.35
            else:
                text_penalty = 0.75

        return time_penalty + text_penalty

    def _delete_cost(self, left_ref: Dict[str, Any]) -> float:
        # Slight preference to keep existing words unless a good match is available
        duration = max(0.05, left_ref['end'] - left_ref['start'])
        return 0.45 + 0.1 * duration

    def _insert_cost(self, right_ref: Dict[str, Any]) -> float:
        duration = max(0.05, right_ref['end'] - right_ref['start'])
        return 0.45 + 0.1 * duration

    def _resolve_alignment(
        self,
        ops: List[Tuple[str, Optional[int], Optional[int]]],
        left_refs: List[Dict[str, Any]],
        right_refs: List[Dict[str, Any]],
        boundary: float
    ) -> Tuple[set[int], set[Tuple[int, int]]]:
        remove_word_ids: set[int] = set()
        keep_right_keys: set[Tuple[int, int]] = set()

        for op, left_index, right_index in ops:
            if op == 'match' and left_index is not None and right_index is not None:
                left_ref = left_refs[left_index]
                right_ref = right_refs[right_index]
                preferred = self._choose_preferred_word(left_ref, right_ref)
                if preferred == 'right':
                    remove_word_ids.add(id(left_ref['word']))
                    keep_right_keys.add(right_ref['key'])
                else:
                    # Prefer left: drop the overlapping right word
                    continue
            elif op == 'insert' and right_index is not None:
                right_ref = right_refs[right_index]
                keep_right_keys.add(right_ref['key'])
            # Deletes correspond to left-only words we already keep

        return remove_word_ids, keep_right_keys

    def _choose_preferred_word(self, left_ref: Dict[str, Any], right_ref: Dict[str, Any]) -> str:
        left_norm = left_ref['norm']
        right_norm = right_ref['norm']

        left_marker = self._is_disfluency_marker_norm(left_norm)
        right_marker = self._is_disfluency_marker_norm(right_norm)

        if left_marker and not right_marker:
            return 'right'
        if right_marker and not left_marker:
            return 'left'

        if not left_norm and right_norm:
            return 'right'
        if left_norm and not right_norm:
            return 'left'

        if left_norm == right_norm:
            # Identical tokens – keep the existing one
            return 'left'

        similarity = self._token_similarity(left_norm, right_norm)
        left_distance = left_ref['distance']
        right_distance = right_ref['distance']

        if similarity < 0.85:
            # Material disagreement: prefer the word farther from the boundary.
            distance_delta = abs(right_distance - left_distance)
            if distance_delta > 0.01:
                return 'right' if right_distance > left_distance else 'left'
            # No clear distance winner – fall back to the newer token.
            return 'right'

        # Words are almost the same: keep the existing one unless the newer
        # word is significantly better positioned (farther from boundary).
        if right_distance - left_distance > 0.05:
            return 'right'
        if left_distance - right_distance > 0.05:
            return 'left'

        # Everything else being equal, prefer the later chunk.
        return 'right'

    @staticmethod
    def _token_similarity(left_norm: str, right_norm: str) -> float:
        if not left_norm or not right_norm:
            return 0.0
        return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()

    def _remove_words_by_id(self, segments: List[Dict[str, Any]], remove_ids: set[int]) -> None:
        if not remove_ids:
            return
        for segment in segments:
            words = segment.get('words', []) or []
            if not words:
                continue
            filtered = [word for word in words if id(word) not in remove_ids]
            if len(filtered) != len(words):
                segment['words'] = filtered
                keep_segment = self._update_segment_metadata(segment)
                if not keep_segment:
                    segment['words'] = []

    def _filter_new_segments(
        self,
        segments: List[Dict[str, Any]],
        keep_keys: set[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        filtered_segments: List[Dict[str, Any]] = []

        for segment_idx, segment in enumerate(segments):
            words = segment.get('words', []) or []
            if not words:
                continue

            kept_words = [
                word for word_idx, word in enumerate(words)
                if (segment_idx, word_idx) in keep_keys
            ]

            if not kept_words:
                continue

            segment['words'] = kept_words
            if self._update_segment_metadata(segment):
                filtered_segments.append(segment)

        return filtered_segments

    def _update_segment_metadata(self, segment: Dict[str, Any]) -> bool:
        words = segment.get('words', []) or []
        if not words:
            segment['text'] = ''
            return False

        start = min(float(word.get('start', segment.get('start', 0.0))) for word in words)
        end = max(float(word.get('end', word.get('start', start))) for word in words)
        segment['start'] = start
        segment['end'] = end

        text_tokens = [self._word_text(word) for word in words if self._word_text(word)]
        segment['text'] = ' '.join(text_tokens).strip()

        return True

    @staticmethod
    def _word_text(word: Dict[str, Any]) -> str:
        if not isinstance(word, dict):
            return ''
        token = word.get('word')
        if isinstance(token, str) and token.strip():
            return token.strip()
        token = word.get('text')
        if isinstance(token, str):
            return token.strip()
        return ''

    @staticmethod
    def _normalize_token(token: str) -> str:
        if not token:
            return ''
        token = token.strip().lower()
        allowed = set('[]*')
        translation = {ord(ch): None for ch in string.punctuation if ch not in allowed}
        return token.translate(translation)

    @staticmethod
    def _is_disfluency_marker_norm(norm_token: str) -> bool:
        return bool(norm_token) and bool(DISFLUENCY_MARKER_PATTERN.match(norm_token))

    def _merge_vad_segments(self, spans: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Merge overlapping VAD spans from chunk outputs."""
        normalized: List[Dict[str, float]] = []
        for span in spans:
            start = span.get('start')
            end = span.get('end')
            if start is None or end is None:
                continue
            normalized.append({'start': float(start), 'end': float(end)})

        if not normalized:
            return []

        normalized.sort(key=lambda item: item['start'])
        merged: List[Dict[str, float]] = [normalized[0]]

        for current in normalized[1:]:
            last = merged[-1]
            if current['start'] <= last['end'] + 1e-6:
                last['end'] = max(last['end'], current['end'])
            else:
                merged.append(current)

        return merged


class TranscriptionPipeline:
    """Main transcription pipeline orchestrator."""
    
    def __init__(self, config: TranscriptionConfig, whisper_settings: WhisperSettings):
        self.config = config
        self.whisper_settings = whisper_settings
        self.resource_manager = ResourceManager()
        self.audio_processor = AudioProcessor(config)
        self.chunk_processor = ChunkProcessor(config, whisper_settings, self.resource_manager)
        self.result_merger = ResultMerger(config)
    
    def process_audio_file(self, audiofile_path: str, boundaries: List[float], 
                          output_path: str) -> Dict[str, Any]:
        """Process an entire audio file using chunking."""
        chunk_results = []
        
        print(f"\n🎬 Processing {len(boundaries)-1} chunks...")
        
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            chunk_duration = end_time - start_time
            
            print(f"\n📝 Chunk {i+1}/{len(boundaries)-1}: {start_time:.1f}s to {end_time:.1f}s ({chunk_duration:.1f}s)")
            
            chunk_start_time = time.time()
            
            try:
                result = self.chunk_processor.process_chunk_in_subprocess(
                    audiofile_path, start_time, end_time, i
                )
                chunk_results.append(result)
                
                # Monitor progress
                self._monitor_progress(i, len(boundaries)-1, chunk_start_time, chunk_duration)
                
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")
                # Could implement retry logic here
                raise
        
        # Merge all chunk results
        print("\n🔀 Merging chunk results...")
        merged_result = self.result_merger.merge_chunk_results(chunk_results, boundaries)
        
        return merged_result
    
    def _monitor_progress(self, chunk_id: int, total_chunks: int, 
                         chunk_start_time: float, chunk_duration: float):
        """Monitor and display processing progress."""
        current_chunk_time = time.time() - chunk_start_time
        remaining_chunks = total_chunks - chunk_id - 1
        estimated_remaining = current_chunk_time * remaining_chunks if remaining_chunks > 0 else 0
        
        print(f"Progress: {chunk_id+1}/{total_chunks} ({(chunk_id+1)/total_chunks*100:.1f}%)")
        print(f"Chunk duration: {chunk_duration:.1f}s | Processed in: {current_chunk_time:.1f}s")
        print(f"Estimated remaining: {estimated_remaining/60:.1f}min")
        
        # Warning for slow chunks
        if current_chunk_time > chunk_duration * 3:
            print("⚠️ WARNING: Current chunk is very slow - possible memory pressure!")
