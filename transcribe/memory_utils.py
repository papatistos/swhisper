"""Memory and resource management utilities."""

import gc
import sys
import ctypes
import torch
import numpy as np
from typing import Optional
import psutil
import subprocess
import re

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import whisper_timestamped as whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class ResourceManager:
    """Manages system resources and memory cleanup."""
    
    def __init__(self):
        self.current_whisper_model: Optional[object] = None
        self.current_audio_data: Optional[object] = None
        self.current_subprocess: Optional[subprocess.Popen] = None
    
    def clear_device_memory(self) -> None:
        """Perform comprehensive memory cleanup."""
        print("Performing comprehensive memory cleanup...")
        
        # Clear global resources
        if self.current_whisper_model is not None:
            del self.current_whisper_model
            self.current_whisper_model = None
        if self.current_audio_data is not None:
            del self.current_audio_data
            self.current_audio_data = None
        
        # Multiple garbage collection passes
        for _ in range(5):
            gc.collect()
        
        # Clear PyTorch caches with synchronization
        if hasattr(torch, "mps") and torch.backends.mps.is_built():
            torch.mps.synchronize()
            torch.mps.empty_cache()
            torch.mps.empty_cache()  # Call twice
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        
        # Try to clear NumPy memory pools
        try:
            # Force NumPy to release memory
            np.fft.fftfreq(1)  # This sometimes helps clear FFT caches
        except:
            pass
        
        # Clear any librosa caches
        if HAS_LIBROSA:
            try:
                librosa.cache.clear()
            except:
                pass
        
        # Clear whisper caches
        if HAS_WHISPER:
            try:
                if hasattr(whisper, 'cache'):
                    whisper.cache.clear()
            except:
                pass
        
        # Force Python to release memory to OS (aggressive)
        try:
            if sys.platform == 'darwin':  # macOS
                libc = ctypes.CDLL("libc.dylib")
            else:  # Linux
                libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
        
        # Final garbage collection
        for _ in range(3):
            gc.collect()
    
    def cleanup_resources(self) -> None:
        """Comprehensive cleanup of all resources, but preserve checkpoints."""
        print("\n🧹 Cleaning up resources...")
        
        # Terminate any running subprocess
        if self.current_subprocess is not None:
            try:
                self.current_subprocess.terminate()
                self.current_subprocess.wait(timeout=5)
            except:
                try:
                    self.current_subprocess.kill()
                    self.current_subprocess.wait(timeout=2)
                except:
                    pass
            self.current_subprocess = None
        
        # Clear whisper model
        if self.current_whisper_model is not None:
            del self.current_whisper_model
            self.current_whisper_model = None
        
        # Clear audio data
        if self.current_audio_data is not None:
            del self.current_audio_data
            self.current_audio_data = None
        
        # Comprehensive memory cleanup
        self.clear_device_memory()
        
        print("✅ Done (temp workspace preserved for resuming)")


class MemoryMonitor:
    """Monitor system memory usage."""
    
    @staticmethod
    def get_memory_info() -> str:
        """Get current memory usage as formatted string."""
        try:
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024
            
            if sys.platform == 'darwin':
                return MemoryMonitor._get_macos_memory_info(process, rss_mb)
            else:
                return MemoryMonitor._get_linux_memory_info(process, rss_mb)
                
        except (ImportError, psutil.NoSuchProcess):
            return "Memory info unavailable"
    
    @staticmethod
    def _get_macos_memory_info(process: psutil.Process, rss_mb: float) -> str:
        """Get detailed memory info on macOS using vmmap."""
        try:
            command = f"vmmap -summary {process.pid}"
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'TOTAL' in line and 'minus' not in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            dirty_size = parts[3]
                            return f"RSS: {rss_mb:.0f}MB, Dirty: {dirty_size}"
        except Exception:
            pass
        
        return f"RSS: {rss_mb:.0f}MB"
    
    @staticmethod
    def _get_linux_memory_info(process: psutil.Process, rss_mb: float) -> str:
        """Get detailed memory info on Linux."""
        try:
            memory_full = process.memory_full_info()
            uss_mb = memory_full.uss / 1024 / 1024
            pss_mb = memory_full.pss / 1024 / 1024
            return f"RSS: {rss_mb:.0f}MB, USS: {uss_mb:.0f}MB, PSS: {pss_mb:.0f}MB"
        except (AttributeError, psutil.AccessDenied):
            return f"RSS: {rss_mb:.0f}MB"
    
    @staticmethod
    def parse_dirty_memory(dirty_str: str) -> float:
        """Parse dirty memory string and return value in MB."""
        match = re.search(r'Dirty:\s*([\d\.]+)([GM]?)', dirty_str)
        if not match:
            return 0
        value, unit = float(match.group(1)), match.group(2)
        if unit.startswith('G'):
            value *= 1024
        return value


# Global resource manager instance
resource_manager = ResourceManager()
