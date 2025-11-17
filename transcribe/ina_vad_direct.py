"""
Direct inaSpeechSegmenter VAD integration (no subprocess).

This module provides VAD using inaSpeechSegmenter installed directly
in the swhisper environment. Much simpler than subprocess approach.
"""

from typing import List, Tuple, Optional
from inaSpeechSegmenter import Segmenter


class InaSpeechVADDirect:
    """
    VAD provider using inaSpeechSegmenter (direct import).
    
    Uses inaSpeechSegmenter directly from the main environment.
    Requires inaSpeechSegmenter 0.8.0+ with NumPy 2.x compatibility.
    """
    
    def __init__(
        self,
        detect_gender: bool = False,
        vad_engine: str = 'smn'
    ):
        """
        Initialize InaSpeechSegmenter VAD.
        
        Args:
            detect_gender: If True, returns male/female labels. If False, returns 'speech'
            vad_engine: 'smn' (speech/music/noise) or 'sm' (speech/music)
        """
        self.detect_gender = detect_gender
        self.vad_engine = vad_engine
        
        # Initialize the segmenter
        self.segmenter = Segmenter(
            vad_engine=vad_engine,
            detect_gender=detect_gender
        )
    
    def get_speech_segments(
        self,
        audio_path: str,
        start_sec: Optional[float] = None,
        stop_sec: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Get speech segments from audio file.
        
        Args:
            audio_path: Path to audio file (any format supported by ffmpeg)
            start_sec: Optional start time in seconds
            stop_sec: Optional stop time in seconds
            
        Returns:
            List of (start, end) tuples in seconds, suitable for whisper VAD parameter
        """
        # Run segmentation
        segments = self.segmenter(audio_path)
        
        # Filter to speech segments only
        # Labels: 'speech', 'music', 'noise', 'male', 'female', 'noEnergy'
        speech_labels = {'speech', 'male', 'female'}
        
        speech_segments = [
            (start, end)
            for label, start, end in segments
            if label in speech_labels
        ]
        
        # Apply time range filter if specified
        if start_sec is not None or stop_sec is not None:
            start_sec = start_sec or 0.0
            stop_sec = stop_sec or float('inf')
            
            speech_segments = [
                (max(start, start_sec), min(end, stop_sec))
                for start, end in speech_segments
                if end > start_sec and start < stop_sec
            ]
        
        return speech_segments


def create_ina_vad_direct(
    detect_gender: bool = False,
    vad_engine: str = 'smn'
) -> InaSpeechVADDirect:
    """
    Factory function to create InaSpeechSegmenter VAD provider (direct import).
    
    Args:
        detect_gender: If True, distinguishes male/female. If False, all speech is labeled 'speech'
        vad_engine: 'smn' for speech/music/noise or 'sm' for speech/music
        
    Returns:
        InaSpeechVADDirect instance
    """
    return InaSpeechVADDirect(
        detect_gender=detect_gender,
        vad_engine=vad_engine
    )
