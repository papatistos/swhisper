"""
Speaker-aware chunking for diarization-first transcription pipeline.

This module provides utilities to:
1. Extract speaker segments from pyannote diarization results
2. Create time-bounded chunks that respect speaker boundaries
3. Convert speaker segments to VAD format for whisper-timestamped
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_speaker_segments(diarization_result: Any, use_exclusive: bool = True) -> List[Dict[str, Any]]:
    """
    Extract speaker segments from pyannote diarization result.
    
    Args:
        diarization_result: pyannote.audio Annotation object or DiarizeOutput object
        use_exclusive: If True, use exclusive (non-overlapping) segments. 
                      If False, use standard (potentially overlapping) segments.
        
    Returns:
        List of dicts with 'start', 'end', and 'speaker' keys, sorted by start time
        
    Example:
        [
            {'start': 0.0, 'end': 15.5, 'speaker': 'SPEAKER_00'},
            {'start': 15.5, 'end': 30.2, 'speaker': 'SPEAKER_01'},
            ...
        ]
    """
    segments = []
    
    # Handle DiarizeOutput object from pyannote 4 community model
    # It has speaker_diarization and exclusive_speaker_diarization attributes
    annotation = diarization_result
    if hasattr(diarization_result, 'speaker_diarization'):
        if use_exclusive and hasattr(diarization_result, 'exclusive_speaker_diarization'):
            # Use exclusive (non-overlapping) for clean segment boundaries
            annotation = diarization_result.exclusive_speaker_diarization
        else:
            # Use standard (overlapping) to detect crosstalk
            annotation = diarization_result.speaker_diarization
    
    # pyannote diarization result has itertracks() method
    # Returns (segment, track, label) tuples
    for segment, _, speaker_label in annotation.itertracks(yield_label=True):
        segments.append({
            'start': float(segment.start),  # Ensure float, not Segment object attribute
            'end': float(segment.end),      # Ensure float, not Segment object attribute
            'speaker': str(speaker_label)
        })
    
    # Sort by start time (should already be sorted, but ensure it)
    segments.sort(key=lambda x: x['start'])
    
    return segments


def has_overlapping_speech(
    timestamp: float,
    overlapping_segments: List[Dict[str, Any]],
    tolerance: float = 0.5
) -> bool:
    """
    Check if there is overlapping speech (crosstalk) at or near a given timestamp.
    
    Args:
        timestamp: Time in seconds to check for overlap
        overlapping_segments: Speaker segments from standard (overlapping) diarization
        tolerance: Time window in seconds to check around the timestamp (default 0.5s)
        
    Returns:
        True if multiple speakers are talking at the same time near this timestamp
    """
    # Find all segments that overlap with the time window
    active_speakers = set()
    
    for segment in overlapping_segments:
        # Check if segment overlaps with our time window
        if (segment['start'] <= timestamp + tolerance and 
            segment['end'] >= timestamp - tolerance):
            active_speakers.add(segment['speaker'])
    
    # If more than one speaker is active, there's overlap
    return len(active_speakers) > 1


def create_speaker_aware_chunks(
    speaker_segments: List[Dict[str, Any]], 
    target_duration: float = 180.0,
    max_duration: float = 200.0,
    min_chunk_duration: float = 30.0,
    overlapping_segments: Optional[List[Dict[str, Any]]] = None,
    avoid_crosstalk: bool = True
) -> List[Dict[str, Any]]:
    """
    Create time-bounded chunks that respect speaker boundaries.
    
    Strategy: Group consecutive speaker turns into chunks of approximately
    target_duration, but only cut at speaker boundaries. Optionally avoids
    cutting during overlapping speech (crosstalk) if overlapping segments provided.
    
    Args:
        speaker_segments: List of speaker segments from extract_speaker_segments() (exclusive)
        target_duration: Target chunk duration in seconds (default 180s = 3 minutes)
        max_duration: Maximum chunk duration before forcing a split (default 200s)
        min_chunk_duration: Minimum chunk duration to avoid too-small chunks (default 30s)
        overlapping_segments: Optional list of segments with overlaps to detect crosstalk
        avoid_crosstalk: If True and overlapping_segments provided, avoid cutting during crosstalk
        
    Returns:
        List of chunk dicts, each containing:
        - 'chunk_start': Start time of chunk in original audio
        - 'chunk_end': End time of chunk in original audio
        - 'segments': List of speaker segments within this chunk
        - 'chunk_index': 0-based chunk number
        
    Example:
        [
            {
                'chunk_start': 0.0,
                'chunk_end': 185.3,
                'segments': [{'start': 0.0, 'end': 15.5, 'speaker': 'SPEAKER_00'}, ...],
                'chunk_index': 0
            },
            ...
        ]
    """
    if not speaker_segments:
        return []
    
    chunks = []
    current_chunk_start = speaker_segments[0]['start']
    current_chunk_segments = []
    crosstalk_avoidances = 0  # Track how many times we avoided crosstalk
    
    for segment in speaker_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        
        # Check if adding this segment would exceed our target duration
        chunk_duration_with_segment = segment_end - current_chunk_start
        
        # Decision logic:
        # 1. If we haven't reached target duration, keep adding segments
        # 2. If we're past target but under max, only split if we have enough in current chunk
        # 3. If we're past max duration, force a split (safety for very long monologues)
        
        should_split = False
        
        if chunk_duration_with_segment > max_duration:
            # Force split - chunk is too long (safety limit)
            should_split = True
        elif chunk_duration_with_segment > target_duration:
            # We're over target - split if current chunk is substantial enough
            current_chunk_duration = (current_chunk_segments[-1]['end'] - current_chunk_start 
                                     if current_chunk_segments else 0)
            if current_chunk_duration >= min_chunk_duration:
                # Check if there's crosstalk at the proposed cut point
                proposed_cut_point = current_chunk_segments[-1]['end']
                
                if avoid_crosstalk and overlapping_segments:
                    if has_overlapping_speech(proposed_cut_point, overlapping_segments):
                        # Don't split here - there's crosstalk at this boundary
                        # Continue to next segment and check again
                        should_split = False
                        crosstalk_avoidances += 1
                        logger.info(
                            f"Avoided chunk boundary at {proposed_cut_point:.2f}s due to overlapping speech. "
                            f"Continuing chunk to avoid cutting during crosstalk."
                        )
                    else:
                        # Clean boundary - safe to split
                        should_split = True
                else:
                    # No overlap checking requested, split normally
                    should_split = True
        
        if should_split and current_chunk_segments:
            # Create chunk from accumulated segments
            chunks.append({
                'chunk_start': current_chunk_start,
                'chunk_end': current_chunk_segments[-1]['end'],
                'segments': current_chunk_segments.copy(),
                'chunk_index': len(chunks),
                'crosstalk_avoidances': crosstalk_avoidances
            })
            
            # Reset counter for next chunk
            crosstalk_avoidances = 0
            
            # Start new chunk with current segment
            current_chunk_start = segment_start
            current_chunk_segments = [segment]
        else:
            # Add segment to current chunk
            current_chunk_segments.append(segment)
    
    # Don't forget the last chunk
    if current_chunk_segments:
        chunks.append({
            'chunk_start': current_chunk_start,
            'chunk_end': current_chunk_segments[-1]['end'],
            'segments': current_chunk_segments.copy(),
            'chunk_index': len(chunks),
            'crosstalk_avoidances': crosstalk_avoidances  # Include statistics
        })
    
    # Log summary of crosstalk avoidance
    total_avoidances = sum(chunk.get('crosstalk_avoidances', 0) for chunk in chunks)
    if total_avoidances > 0:
        logger.info(
            f"Crosstalk avoidance summary: Avoided {total_avoidances} chunk boundaries "
            f"due to overlapping speech across {len(chunks)} chunks"
        )
    
    return chunks


def convert_segments_to_vad_format(
    segments: List[Dict[str, Any]], 
    chunk_start: float
) -> List[Tuple[float, float]]:
    """
    Convert speaker segments to VAD format for whisper-timestamped.
    
    Adjusts timestamps to be relative to the chunk start time, as whisper
    expects timestamps relative to the audio chunk it receives.
    
    Note: whisper-timestamped expects a list of (start, end) tuples,
    NOT a list of dicts with 'start' and 'end' keys.
    
    Args:
        segments: List of speaker segments with absolute timestamps
        chunk_start: Start time of the audio chunk in the original file
        
    Returns:
        List of (start, end) tuples with chunk-relative timestamps
        
    Example:
        # Input: segments for chunk starting at 60.0s
        segments = [{'start': 60.0, 'end': 75.5, 'speaker': 'SPEAKER_00'}, ...]
        chunk_start = 60.0
        
        # Output: timestamps relative to chunk start as tuples
        [(0.0, 15.5), (15.5, 30.2), ...]
    """
    vad_segments = []
    
    for segment in segments:
        # Convert to chunk-relative time and ensure float type
        relative_start = float(segment['start']) - float(chunk_start)
        relative_end = float(segment['end']) - float(chunk_start)
        
        # Ensure non-negative times (safety check)
        relative_start = max(0.0, relative_start)
        relative_end = max(relative_start, relative_end)
        
        # Return as tuple (start, end) not dict
        vad_segments.append((relative_start, relative_end))
    
    return vad_segments


def get_speaker_for_timestamp(
    timestamp: float,
    speaker_segments: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Find which speaker was active at a given timestamp.
    
    Args:
        timestamp: Time in seconds (absolute time in original audio)
        speaker_segments: List of speaker segments with absolute timestamps
        
    Returns:
        Speaker label if found, None if timestamp is in a gap/silence
    """
    for segment in speaker_segments:
        if segment['start'] <= timestamp <= segment['end']:
            return segment['speaker']
    
    return None


def analyze_chunk_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of chunks for debugging/logging.
    
    Args:
        chunks: List of chunk dicts from create_speaker_aware_chunks()
        
    Returns:
        Dict with statistics about chunk distribution
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_duration': 0.0,
            'avg_chunk_duration': 0.0,
            'min_chunk_duration': 0.0,
            'max_chunk_duration': 0.0
        }
    
    durations = [chunk['chunk_end'] - chunk['chunk_start'] for chunk in chunks]
    total_duration = chunks[-1]['chunk_end'] - chunks[0]['chunk_start']
    
    return {
        'total_chunks': len(chunks),
        'total_duration': total_duration,
        'avg_chunk_duration': sum(durations) / len(durations),
        'min_chunk_duration': min(durations),
        'max_chunk_duration': max(durations),
        'chunk_durations': durations
    }
