"""Cache for pyannote diarization results."""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any


class DiarizationCache:
    """Cache pyannote diarization results in RTTM format."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize the diarization cache.
        
        Args:
            cache_dir: Directory to store cached RTTM files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(
        self,
        audio_path: str,
        model: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a cache key based on audio file and diarization parameters.
        
        Args:
            audio_path: Path to the audio file
            model: Diarization model identifier
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            pipeline_config: Additional pipeline configuration
        
        Returns:
            Cache key string
        """
        # Get audio file modification time and size
        audio_stat = os.stat(audio_path)
        audio_mtime = audio_stat.st_mtime
        audio_size = audio_stat.st_size
        
        # Create a string with all parameters
        audio_basename = os.path.basename(audio_path)
        key_parts = [
            audio_basename,
            str(audio_size),
            str(audio_mtime),
            model,
            str(min_speakers),
            str(max_speakers),
        ]
        
        # Add pipeline config if provided
        if pipeline_config:
            # Sort keys for consistent ordering
            config_str = '_'.join(f"{k}={v}" for k, v in sorted(pipeline_config.items()))
            key_parts.append(config_str)
        
        # Create a hash of the key parts
        key_string = '||'.join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
        
        # Use audio basename without extension + hash
        audio_name = Path(audio_basename).stem
        return f"{audio_name}_{key_hash}"
    
    def get_cache_path(
        self,
        audio_path: str,
        model: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        pipeline_config: Optional[Dict[str, Any]] = None,
        exclusive: bool = False
    ) -> Path:
        """
        Get the cache file path for given parameters.
        
        Args:
            audio_path: Path to the audio file
            model: Diarization model identifier
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            pipeline_config: Additional pipeline configuration
            exclusive: Whether this is for exclusive diarization
        
        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key(audio_path, model, min_speakers, max_speakers, pipeline_config)
        suffix = "_exclusive" if exclusive else ""
        filename = f"{cache_key}{suffix}.rttm"
        return self.cache_dir / filename
    
    def exists(
        self,
        audio_path: str,
        model: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        pipeline_config: Optional[Dict[str, Any]] = None,
        exclusive: bool = False
    ) -> bool:
        """
        Check if cache exists for given parameters.
        
        Args:
            audio_path: Path to the audio file
            model: Diarization model identifier
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            pipeline_config: Additional pipeline configuration
            exclusive: Whether this is for exclusive diarization
        
        Returns:
            True if cache exists, False otherwise
        """
        cache_path = self.get_cache_path(audio_path, model, min_speakers, max_speakers, pipeline_config, exclusive)
        return cache_path.exists()
    
    def load(
        self,
        audio_path: str,
        model: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        pipeline_config: Optional[Dict[str, Any]] = None,
        exclusive: bool = False
    ):
        """
        Load cached diarization result.
        
        Args:
            audio_path: Path to the audio file
            model: Diarization model identifier
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            pipeline_config: Additional pipeline configuration
            exclusive: Whether this is for exclusive diarization
        
        Returns:
            Pyannote Annotation object or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(audio_path, model, min_speakers, max_speakers, pipeline_config, exclusive)
        
        if not cache_path.exists():
            return None
        
        try:
            from pyannote.database.util import load_rttm
            
            # Load RTTM file - returns a dict with uri as key
            rttm_dict = load_rttm(str(cache_path))
            
            # RTTM files can contain multiple URIs, but we expect just one
            if len(rttm_dict) == 0:
                return None
            
            # Get the first (and should be only) annotation
            annotation = list(rttm_dict.values())[0]
            return annotation
            
        except Exception as e:
            print(f"  -> Warning: Failed to load diarization cache: {e}")
            return None
    
    def save(
        self,
        annotation,
        audio_path: str,
        model: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        pipeline_config: Optional[Dict[str, Any]] = None,
        exclusive: bool = False
    ) -> None:
        """
        Save diarization result to cache.
        
        Args:
            annotation: Pyannote Annotation object
            audio_path: Path to the audio file
            model: Diarization model identifier
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            pipeline_config: Additional pipeline configuration
            exclusive: Whether this is for exclusive diarization
        """
        cache_path = self.get_cache_path(audio_path, model, min_speakers, max_speakers, pipeline_config, exclusive)
        
        try:
            # Create cache directory if it doesn't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as RTTM
            # Use audio basename as URI, replacing spaces with underscores
            # (RTTM format doesn't allow spaces in URIs)
            uri = os.path.basename(audio_path).replace(' ', '_')
            
            # Temporarily set the annotation's uri attribute for serialization
            original_uri = getattr(annotation, 'uri', None)
            annotation.uri = uri
            
            print(f"  -> Writing cache to: {cache_path}")
            with open(cache_path, 'w') as rttm_file:
                annotation.write_rttm(rttm_file)
            print(f"  -> Cache file written successfully")
            
            # Restore original uri
            if original_uri is not None:
                annotation.uri = original_uri
            
        except Exception as e:
            print(f"  -> Warning: Failed to save diarization cache: {e}")
    
    def clear(self) -> int:
        """
        Clear all cached diarization files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.rttm"):
                cache_file.unlink()
                count += 1
        return count
