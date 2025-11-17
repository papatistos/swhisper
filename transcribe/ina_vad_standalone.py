#!/usr/bin/env python3
"""
Standalone inaSpeechSegmenter VAD script for use in isolated conda environment.

This script runs in the ina_vad environment (with NumPy 1.x) and outputs
VAD results as JSON to stdout for consumption by the main swhisper pipeline.

Usage:
    conda run -n ina_vad python ina_vad_standalone.py <audio_path> [--start START] [--stop STOP]

Output (JSON):
    {
        "success": true,
        "segments": [
            {"start": 0.0, "end": 10.5, "label": "speech"},
            {"start": 15.2, "end": 25.8, "label": "speech"},
            ...
        ],
        "stats": {
            "total_duration": 194.9,
            "speech_duration": 180.2,
            "music_duration": 5.3,
            "noise_duration": 9.4
        }
    }

Error output:
    {
        "success": false,
        "error": "Error message here"
    }
"""

import sys
import json
import argparse
from pathlib import Path


def run_ina_vad(audio_path: str, start_sec: float = None, stop_sec: float = None) -> dict:
    """
    Run inaSpeechSegmenter on audio file.
    
    Args:
        audio_path: Path to audio file
        start_sec: Optional start time in seconds
        stop_sec: Optional stop time in seconds
        
    Returns:
        Dictionary with success status, segments, and stats
    """
    try:
        # Suppress TensorFlow/Keras logging
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        from inaSpeechSegmenter import Segmenter
        
        # Verify file exists
        if not Path(audio_path).exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Initialize segmenter
        try:
            seg = Segmenter(
                vad_engine='smn',  # speech/music/noise
                detect_gender=False  # We only need VAD, not gender
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize Segmenter: {e}"
            }
        
        # Run segmentation
        try:
            segmentation = seg(audio_path, start_sec=start_sec, stop_sec=stop_sec)
        except Exception as e:
            return {
                "success": False,
                "error": f"Segmentation failed: {e}"
            }
        
        # Convert to list and calculate stats
        segments = []
        stats = {
            "total_duration": 0.0,
            "speech_duration": 0.0,
            "music_duration": 0.0,
            "noise_duration": 0.0
        }
        
        for label, start, end in segmentation:
            duration = end - start
            segments.append({
                "start": float(start),
                "end": float(end),
                "label": str(label)
            })
            
            # Update stats
            stats["total_duration"] = max(stats["total_duration"], float(end))
            
            if label in ['speech', 'male', 'female']:
                stats["speech_duration"] += duration
            elif label == 'music':
                stats["music_duration"] += duration
            elif label in ['noise', 'noEnergy']:
                stats["noise_duration"] += duration
        
        return {
            "success": True,
            "segments": segments,
            "stats": stats,
            "audio_path": audio_path,
            "start_sec": start_sec,
            "stop_sec": stop_sec
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {type(e).__name__}: {e}"
        }


def main():
    """Command-line interface."""
    # Suppress TensorFlow warnings early
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_KERAS_VERBOSE'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    
    # Also suppress Keras progress bars globally
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(
        description='Run inaSpeechSegmenter VAD on audio file'
    )
    parser.add_argument(
        'audio_path',
        help='Path to audio file'
    )
    parser.add_argument(
        '--start',
        type=float,
        default=None,
        help='Start time in seconds (optional)'
    )
    parser.add_argument(
        '--stop',
        type=float,
        default=None,
        help='Stop time in seconds (optional)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress stderr output from inaSpeechSegmenter'
    )
    
    args = parser.parse_args()
    
    # Suppress stderr if requested (inaSpeechSegmenter is verbose)
    if args.quiet:
        sys.stderr = open(os.devnull, 'w')
    
    # Run VAD
    result = run_ina_vad(args.audio_path, args.start, args.stop)
    
    # Output JSON to stdout
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
