"""Audio processing and chunking utilities."""

import os
import numpy as np
import soundfile as sf
import librosa
import gc
from typing import Tuple, List
from scipy.signal import resample

from .config import TranscriptionConfig


class AudioProcessor:
    """Handle audio loading, chunking, and format conversion."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
    
    def get_audio_duration(self, audiofile_path: str) -> Tuple[float, int]:
        """Get audio duration without loading the entire file into memory."""
        with sf.SoundFile(audiofile_path) as f:
            duration = len(f) / f.samplerate
            sample_rate = f.samplerate
        return duration, sample_rate
    
    def load_audio_chunk(self, audiofile_path: str, start_time: float, 
                        end_time: float, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """Load only a specific chunk of audio from file."""
        overlap_samples = int(self.config.overlap_duration * sample_rate)
        start_sample = max(0, int(start_time * sample_rate) - overlap_samples)
        end_sample = int(end_time * sample_rate) + overlap_samples
        
        # Use soundfile for efficient chunk loading
        with sf.SoundFile(audiofile_path) as f:
            # Get original sample rate
            orig_sr = f.samplerate
            
            # Calculate actual start/end in original sample rate
            orig_start = int(start_sample * orig_sr / sample_rate)
            orig_end = int(end_sample * orig_sr / sample_rate)
            
            # Seek and read
            f.seek(orig_start)
            chunk_audio = f.read(orig_end - orig_start, dtype='float32')
            
            # Convert to mono if stereo
            if len(chunk_audio.shape) > 1:
                chunk_audio = np.mean(chunk_audio, axis=1)
            
            # Resample if needed
            if orig_sr != sample_rate:
                target_length = int(len(chunk_audio) * sample_rate / orig_sr)
                chunk_audio = resample(chunk_audio, target_length)
        
        return chunk_audio.astype(np.float32), start_sample


class SpeechAnalyzer:
    """Analyze speech patterns for intelligent chunking."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
    
    def detect_speech_segments_from_preview(self, audiofile_path: str, 
                                          sample_rate: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze speech patterns using a heavily downsampled version."""
        print("Loading downsampled audio for speech analysis...")
        
        # Load at very low sample rate for analysis only
        audio_preview, _ = librosa.load(audiofile_path, sr=sample_rate)
        
        # Detect speech segments on low-res audio
        times, speech_mask = self._detect_speech_segments_basic(audio_preview, sample_rate)
        
        # Clean up preview immediately
        del audio_preview
        gc.collect()
        gc.collect()
        
        return times, speech_mask
    
    def _detect_speech_segments_basic(self, audio_data: np.ndarray, sample_rate: int,
                                    frame_length: int = 1024, hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Basic speech detection with smaller frames for memory efficiency."""
        # Calculate RMS energy with smaller frames
        frame_energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frame_energy.append(energy)
        
        frame_energy = np.array(frame_energy)
        
        # Simple threshold
        threshold = np.percentile(frame_energy, 30)
        speech_mask = frame_energy > threshold
        
        # Convert to time
        times = np.arange(len(speech_mask)) * hop_length / sample_rate
        
        return times, speech_mask


class ChunkBoundaryFinder:
    """Find optimal chunk boundaries that respect speech patterns."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
    
    def find_optimal_boundaries(self, times: np.ndarray, speech_mask: np.ndarray) -> List[float]:
        """Find optimal boundaries for audio chunks that respect speech patterns."""
        boundaries = [0]  # Always start at the beginning
        current_pos = 0
        
        while current_pos < len(times):
            # Find our target position
            target_time = times[current_pos] + self.config.target_chunk_duration
            
            # If target is beyond the end, we're done
            if target_time >= times[-1]:
                break
                
            # Find the closest time index to our target
            target_idx = np.argmin(np.abs(times - target_time))
            
            # Look for silence periods around our target position
            search_window = int(30 * len(times) / times[-1])  # 30-second search window
            search_start = max(0, target_idx - search_window)
            search_end = min(len(times), target_idx + search_window)
            
            # Find silent regions within our search window
            silence_regions = self._find_silence_regions(
                times[search_start:search_end], 
                speech_mask[search_start:search_end]
            )
            
            if silence_regions:
                # Choose the silence region closest to our target time
                best_silence = min(silence_regions, 
                                 key=lambda x: abs(times[search_start + x[0]] - target_time))
                # Use the middle of the silence region as our boundary
                boundary_idx = search_start + (best_silence[0] + best_silence[1]) // 2
                boundary_time = times[boundary_idx]
            else:
                # No good silence found, use target time (not ideal but necessary)
                boundary_time = target_time
                print(f"Warning: No optimal silence found near {target_time:.1f}s, using target time")
            
            boundaries.append(boundary_time)
            current_pos = np.argmin(np.abs(times - boundary_time))
        
        # Always end with the full duration
        boundaries.append(times[-1])
        
        return boundaries
    
    def _find_silence_regions(self, times: np.ndarray, speech_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous silence regions of at least min_duration seconds."""
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_speech in enumerate(speech_mask):
            if not is_speech and not in_silence:
                # Starting a silence region
                in_silence = True
                silence_start = i
            elif is_speech and in_silence:
                # Ending a silence region
                silence_duration = times[i] - times[silence_start]
                if silence_duration >= self.config.min_silence_duration:
                    silence_regions.append((silence_start, i))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = times[-1] - times[silence_start]
            if silence_duration >= self.config.min_silence_duration:
                silence_regions.append((silence_start, len(speech_mask)))
        
        return silence_regions
