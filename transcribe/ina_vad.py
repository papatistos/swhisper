"""
InaSpeechSegmenter VAD integration for swhisper.

This module provides a VAD (Voice Activity Detection) implementation using
inaSpeechSegmenter, ranked #1 in French TV/radio benchmarks against Silero,
Pyannote, and other VAD systems.

inaSpeechSegmenter provides:
- Superior speech detection (speech/music/noise classification)
- Optional gender detection (male/female)
- CNN-based segmentation optimized for broadcast content

This implementation uses a subprocess approach to run inaSpeechSegmenter in
a separate conda environment (ina_vad) to avoid NumPy/dependency conflicts.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class InaSpeechVAD:
    """
    VAD provider using inaSpeechSegmenter via subprocess.
    
    This runs inaSpeechSegmenter in a separate conda environment (ina_vad)
    to avoid dependency conflicts with the main swhisper environment.
    """
    
    def __init__(
        self,
        conda_env: str = 'ina_vad',
        detect_gender: bool = False,
        vad_engine: str = 'smn',
        verify_env: bool = False  # Skip verification by default for speed
    ):
        """
        Initialize InaSpeechSegmenter VAD subprocess wrapper.
        
        Args:
            conda_env: Name of conda environment with inaSpeechSegmenter installed
            detect_gender: If True, returns male/female labels. If False, returns 'speech'
            vad_engine: 'smn' (speech/music/noise) or 'sm' (speech/music)
            verify_env: If True, verify environment before first use (slow)
        """
        self.conda_env = conda_env
        self.detect_gender = detect_gender
        self.vad_engine = vad_engine
        
        # Find the standalone script
        script_dir = Path(__file__).parent
        self.standalone_script = script_dir / "ina_vad_standalone.py"
        
        if not self.standalone_script.exists():
            raise FileNotFoundError(
                f"inaSpeechSegmenter standalone script not found at: {self.standalone_script}\n"
                f"Expected to find: transcribe/ina_vad_standalone.py"
            )
        
        # Optionally verify conda environment exists
        if verify_env:
            self._verify_environment()
    
    def _verify_environment(self):
        """Verify that the ina_vad conda environment exists and has inaSpeechSegmenter."""
        try:
            result = subprocess.run(
                ['conda', 'run', '-n', self.conda_env, 'python', '-c', 
                 'import inaSpeechSegmenter; print("OK")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0 or 'OK' not in result.stdout:
                raise RuntimeError(
                    f"Conda environment '{self.conda_env}' exists but inaSpeechSegmenter is not installed.\n"
                    f"Install it with:\n"
                    f"  conda run -n {self.conda_env} pip install git+https://github.com/ina-foss/inaSpeechSegmenter.git"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timeout verifying conda environment '{self.conda_env}'")
        except FileNotFoundError:
            raise RuntimeError(
                "conda command not found. Make sure conda is in your PATH."
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not verify conda environment '{self.conda_env}': {e}\n"
                f"Create it with:\n"
                f"  conda create -n {self.conda_env} python=3.10 ffmpeg -y\n"
                f"  conda run -n {self.conda_env} pip install git+https://github.com/ina-foss/inaSpeechSegmenter.git"
            )
    
    def get_speech_segments(
        self,
        audio_path: str,
        start_sec: Optional[float] = None,
        stop_sec: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Get speech segments from audio file using inaSpeechSegmenter.
        
        Args:
            audio_path: Path to audio file (any format supported by ffmpeg)
            start_sec: Optional start time in seconds
            stop_sec: Optional stop time in seconds
            
        Returns:
            List of (start, end) tuples in seconds, suitable for whisper VAD parameter
        """
        # Build command
        cmd = [
            'conda', 'run', '-n', self.conda_env,
            'python', str(self.standalone_script),
            audio_path,
            '--quiet'
        ]
        
        if start_sec is not None:
            cmd.extend(['--start', str(start_sec)])
        if stop_sec is not None:
            cmd.extend(['--stop', str(stop_sec)])
        
        # Run subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"inaSpeechSegmenter subprocess failed with code {result.returncode}\n"
                    f"stderr: {result.stderr}\n"
                    f"stdout: {result.stdout}"
                )
            
            # Parse JSON output (strip any non-JSON prefix lines like progress bars)
            try:
                # Find the first '{' to start of JSON
                stdout = result.stdout
                json_start = stdout.find('{')
                if json_start > 0:
                    stdout = stdout[json_start:]
                response = json.loads(stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse inaSpeechSegmenter output as JSON: {e}\n"
                    f"stdout: {result.stdout[:500]}"
                )
            
            if not response.get('success'):
                raise RuntimeError(
                    f"inaSpeechSegmenter failed: {response.get('error', 'Unknown error')}"
                )
            
            # Extract speech segments
            speech_labels = {'speech', 'male', 'female'}
            speech_segments = [
                (seg['start'], seg['end'])
                for seg in response['segments']
                if seg['label'] in speech_labels
            ]
            
            return speech_segments
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"inaSpeechSegmenter subprocess timed out after 10 minutes processing: {audio_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to run inaSpeechSegmenter: {e}"
            )


def create_ina_vad_provider(
    conda_env: str = 'ina_vad',
    detect_gender: bool = False,
    vad_engine: str = 'smn'
) -> InaSpeechVAD:
    """
    Factory function to create InaSpeechSegmenter VAD provider.
    
    Args:
        conda_env: Name of conda environment with inaSpeechSegmenter installed
        detect_gender: If True, distinguishes male/female. If False, all speech is labeled 'speech'
        vad_engine: 'smn' for speech/music/noise or 'sm' for speech/music
        
    Returns:
        Configured InaSpeechVAD instance
    """
    return InaSpeechVAD(
        conda_env=conda_env,
        detect_gender=detect_gender,
        vad_engine=vad_engine
    )
