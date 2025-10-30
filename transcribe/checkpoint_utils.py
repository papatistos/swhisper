"""Checkpoint management for resumable transcription."""

import os
import pickle
import time
from typing import Dict, Any, Optional, Tuple, List

from .config import TranscriptionConfig


class CheckpointManager:
    """Manage checkpoint files for resumable transcription."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        # Optional explicit output directory (usually set by workspace manager)
        self._output_dir: Optional[str] = None
    
    def save_checkpoint(self, audiofile_path: str, chunk_results: List[Tuple[Dict[str, Any], float]],
                       current_chunk: int, boundaries: List[float], output_path: str,
                       settings: Dict[str, Any]) -> str:
        """Save current processing state to a checkpoint file.

        Args:
            audiofile_path: Path to the audio file being processed
            chunk_results: List of (result_dict, time_offset) tuples for completed chunks
            current_chunk: Index of the next chunk to process (0-based)
            boundaries: List of chunk boundary timestamps
            output_path: Final output path (optional, for reference only)
            settings: Whisper settings dictionary

        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_data = {
            'audiofile_path': audiofile_path,
            'chunk_results': chunk_results,
            'current_chunk': current_chunk,
            'boundaries': boundaries,
            'output_path': output_path,  # Stored for reference, not used during resume
            'settings': settings,
            'timestamp': time.strftime("%Y%m%d-%H%M%S"),
            'total_chunks': len(boundaries) - 1
        }
        
        # Create checkpoint filename based on audio file
        audio_basename = os.path.splitext(os.path.basename(audiofile_path))[0]
        checkpoint_path = self._get_checkpoint_path(audio_basename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"✓ Checkpoint saved: chunk {current_chunk}/{checkpoint_data['total_chunks']}")
        return checkpoint_path
    
    def load_checkpoint(self, audiofile_path: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Load checkpoint if it exists for this audio file."""
        audio_basename = os.path.splitext(os.path.basename(audiofile_path))[0]
        checkpoint_path = self._get_checkpoint_path(audio_basename)
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    # Verify that the checkpoint is for the same file
                    if checkpoint_data.get('audiofile_path') == audiofile_path:
                        print(f"📋 Found checkpoint: resuming from chunk {checkpoint_data['current_chunk']}/{checkpoint_data['total_chunks']}")
                        return checkpoint_data, checkpoint_path
                    else:
                        print("⚠️ Checkpoint found, but for a different audio file. Ignoring.")
                        return None, checkpoint_path
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                return None, checkpoint_path
        # No checkpoint found - return (None, "") so callers know nothing to cleanup
        return None, ""
    
    def cleanup_checkpoint(self, checkpoint_path: str):
        """Remove checkpoint file when processing is complete."""
        if not self.config.preserve_checkpoints:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    print(f"🧹 Checkpoint cleaned up")
            except Exception as e:
                print(f"Error cleaning up checkpoint: {e}")
        else:
            print(f"💾 Checkpoint preserved for resuming: {os.path.basename(checkpoint_path)}")
    
    def cleanup_old_checkpoints(self, output_dir: str, max_age_days: int = 7):
        """Clean up checkpoint files older than max_age_days."""
        if not os.path.exists(output_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        checkpoint_files = []
        for filename in os.listdir(output_dir):
            if filename.startswith('.checkpoint_') and filename.endswith('.pkl'):
                checkpoint_path = os.path.join(output_dir, filename)
                try:
                    file_age = current_time - os.path.getmtime(checkpoint_path)
                    if file_age > max_age_seconds:
                        checkpoint_files.append((checkpoint_path, file_age / (24 * 60 * 60)))
                except OSError:
                    pass
        
        if checkpoint_files:
            print(f"🧹 Found {len(checkpoint_files)} old checkpoint files (>{max_age_days} days):")
            for checkpoint_path, age_days in checkpoint_files:
                try:
                    os.remove(checkpoint_path)
                    print(f"   ✅ Removed {os.path.basename(checkpoint_path)} ({age_days:.1f} days old)")
                except Exception as e:
                    print(f"   ❌ Failed to remove {os.path.basename(checkpoint_path)}: {e}")
        else:
            print(f"✅ No old checkpoints found (>{max_age_days} days)")
    
    def _get_checkpoint_path(self, audio_basename: str) -> str:
        """Get the checkpoint file path for an audio file."""
        # This would need to be implemented to get the output directory
        # For now, we'll assume it's available
        output_dir = self._get_output_dir()
        return os.path.join(output_dir, f".checkpoint_{audio_basename}.pkl")
    
    def _get_output_dir(self) -> str:
        """Get the output directory - this should be injected or passed in."""
        # Prefer an explicitly-set output dir (e.g. temp workspace output)
        if self._output_dir:
            return self._output_dir

        # Fall back to config's audio_dir/json_dir layout if available
        try:
            # If config has json_dir, place checkpoints in that output directory next to audio
            base = getattr(self.config, 'audio_dir', None) or os.getcwd()
            json_dir = getattr(self.config, 'json_dir', 'whisper-json-output')
            candidate = os.path.join(base, json_dir)
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except Exception:
            return os.getcwd()

    def set_output_dir(self, output_dir: str):
        """Explicitly set the directory where checkpoint files will be stored."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._output_dir = output_dir
