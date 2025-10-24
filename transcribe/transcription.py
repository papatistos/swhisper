"""Transcription processing with chunking and subprocess management."""

import os
import time
import math
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
                    device: str = "mps"):
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
            overlap_duration = 2.0
            overlap_samples = int(overlap_duration * 16000)
            start_sample = max(0, int(start_time * 16000) - overlap_samples)
            end_sample = int(end_time * 16000) + overlap_samples
            
            # Load chunk using soundfile
            with sf.SoundFile(audiofile_path) as f:
                orig_sr = f.samplerate
                orig_start = int(start_sample * orig_sr / 16000)
                orig_end = int(end_sample * orig_sr / 16000)
                
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
            time_offset = start_sample / 16000
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
            args=(audiofile_path, start_time, end_time, self.config.model_str, 
                  self.whisper_settings.to_dict(), output_queue, self.config.device)
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
        
        merged_result = {
            'text': '',
            'segments': [],
            'language': chunk_results[0][0].get('language', ''),
            'speech_activity': []
        }

        all_vad_segments: List[Dict[str, Any]] = []
        
        for i, (result, time_offset) in enumerate(chunk_results):
            if i == 0:
                # First chunk: take everything
                merged_result['text'] += result.get('text', '')
                merged_result['segments'].extend(result.get('segments', []))
            else:
                # Subsequent chunks: filter out overlap region
                chunk_start_time = boundaries[i]
                overlap_threshold = chunk_start_time + self.config.overlap_duration
                
                # Only include segments that start after the overlap region
                for segment in result.get('segments', []):
                    if segment['start'] > overlap_threshold:
                        merged_result['text'] += ' ' + segment.get('text', '')
                        merged_result['segments'].append(segment)
            all_vad_segments.extend(result.get('speech_activity', []))
        
        merged_result['speech_activity'] = self._merge_vad_segments(all_vad_segments)

        return merged_result

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
