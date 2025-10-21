"""Configuration settings for diarization processing."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from path_settings import get_path_settings
except ModuleNotFoundError:  # Allow running modules directly from the diarize package
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from path_settings import get_path_settings

_PATH_SETTINGS = get_path_settings()

DEFAULT_DIARIZE_AUDIO_DIR = str(
    _PATH_SETTINGS.diarize_default_audio_dir
    or _PATH_SETTINGS.audio_dir
    or os.getenv("TRANSCRIBE_AUDIO_DIR")
    or "audio"
)


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""
    
    # Authentication
    hugging_face_token: str = os.getenv("HUGGING_FACE_TOKEN", "")
    
    # Diarization settings
    min_speakers: int = 2
    max_speakers: int = 4  
    device: str = "mps"  # "mps", "cuda", or "cpu"
    
    # Processing options
    force_reprocess: bool = True

    # Advanced speaker separation settings
    use_temporal_constraints: bool = True
    speaker_change_penalty: float = 0.1  # Penalty for frequent speaker changes
    embedding_distance_threshold: float = 0.6  # Lower values = Stricter matching - speakers must sound very similar to be grouped together
    
    # Pipeline configuration
    pipeline_model: str = "pyannote/speaker-diarization-3.1"
    segmentation_threshold: float = 0.2  # Lower threshold for more sensitive speaker changes
    min_duration_on: float = None   # default seems to be 0.0
    min_duration_off: float = None  # default seems to be 0.0
    clustering_method: str = "centroid"
    clustering_min_cluster_size: int = 300   
    clustering_threshold: float = 0.1     # (was: .15) Lower threshold for stricter clustering
    
# Default values from pyannote
# embedding_distance_threshold: 0.7 (default)
# lustering_threshold: 0.15 (typical default)
# segmentation_threshold: 0.3 (voice activity detection)
# min_duration_on: 0.0 (no minimum duration filter)
# clustering_threshold: 0.15 (default for clustering)
# clustering_min_cluster_size: 10-15?
# 

    # Silence detection settings
    min_silence_duration: float = 0.2                       # (rounded) silence durations below this value will be ignored
    include_silence_markers: bool = True                    # silence duration in transcript
    log_silence_gaps: bool = False                          # If true, between-word-gap-durations will be logged in separate file
    silence_gap_linebreak_threshold: Optional[float] = 1.0  # Surround long silences with blank lines in transcripts. Set to 0 for no linebreaks
    
    # Word-level processing
    smoothing_enabled: bool = True
    min_speaker_words: int = 3
    preserve_markers: bool = True                           # Preserve markers for sounds that could not be transcribed
    preserved_markers: List[str] = None
    
    # Output preamble for transcript files
    output_preamble: str = """Note 1: If the transcript contains markers like [*], [**], [***] they indicate sounds that could not be transcribed. The number of asterisks roughly reflects the sound's duration in tenths of a second. This can be turned off in the swisper settings.

Note 2: You may also find parentheses with a number inside, e.g. (.3). These indicate silences and the number indicates the duration of the silence in seconds. Currently, not all silences are reliably detected, but those that are may be useful, especially when they indicate longer pauses. This can also be turned off in the swhisper settings. Long silences are surrounded by blank lines (even when there is no speaker change). This can be adjusted or turned off in the swhisper settings.

Note 3: Speaker detection is not perfect. The transcript may show too many different speakers, but the excess (or unknown) speakers should have few turns attributed to them, so it should be easy to fix manually.
"""
    
    def get_preamble_with_transcript_id(self, transcript_id: str) -> str:
        """Generate preamble with transcript ID included."""
        return f"""Transcript-ID: {transcript_id}

{self.output_preamble}"""
        
    # Parameter testing options
    enable_parameter_testing: bool = False
    test_output_dir: str = "parameter_tests"
    test_max_combinations: int = 320
    test_single_file_mode: bool = True                      # Test on one file first before full testing
    
    # Parameter testing notes:
    # - Only enable if you want to find optimal embedding_distance_threshold values
    # - With max_speakers=2 (fixed), most parameter testing is meaningless
    
    # Directory settings
    output_dir: str = "transcripts"
    log_dir: str = "transcripts"
    default_audio_dir: str = DEFAULT_DIARIZE_AUDIO_DIR
    default_json_input_dir: str = "whisper-json-output"
    default_audio_subdir: str = ""                         # Leave empty if audio files are directly in audio_dir
    
    def __post_init__(self):
        """Initialize default values that can't be set in field definition."""
        if self.preserved_markers is None:
            self.preserved_markers = ["[DISCONTINUITY]", "[SILENCE]", "[OVERLAP]", "[*]"]
    
    @property
    def audio_dir(self) -> str:
        """Get audio directory from environment or default."""
        return os.environ.get('TRANSCRIBE_AUDIO_DIR', self.default_audio_dir)
    
    @property
    def json_input_dir(self) -> str:
        """Get JSON input directory from environment or default."""
        return os.environ.get('JSON_INPUT_DIR', self.default_json_input_dir)
    
    @property
    def audio_subdir(self) -> str:
        """Get audio subdirectory from environment or default."""
        return os.environ.get('AUDIO_SUBDIR', self.default_audio_subdir)
    
    @property
    def final_json_input_dir(self) -> str:
        """Get final JSON input directory path."""
        return os.path.join(self.audio_dir, self.json_input_dir)
    
    @property
    def final_output_dir(self) -> str:
        """Get final output directory path."""
        return os.path.join(self.audio_dir, self.output_dir)
    
    @property
    def final_log_dir(self) -> str:
        """Get final log directory path."""
        return os.path.join(self.audio_dir, self.log_dir)
    
    @property
    def final_audio_dir(self) -> str:
        """Get final audio directory path."""
        return os.path.join(self.audio_dir, self.audio_subdir) if self.audio_subdir else self.audio_dir
    
    def get_pipeline_config(self) -> Dict:
        """Get pipeline configuration dictionary."""
        config = {
            "segmentation": {
                "threshold": self.segmentation_threshold,
                "min_duration_on": self.min_duration_on,
                "min_duration_off": self.min_duration_off,
            },
            "clustering": {
                "method": self.clustering_method,
                "min_cluster_size": self.clustering_min_cluster_size,
                "threshold": self.clustering_threshold,
            }
        }
        
        # Add advanced settings if available
        if hasattr(self, 'use_temporal_constraints') and self.use_temporal_constraints:
            config["temporal_constraints"] = True
            config["speaker_change_penalty"] = getattr(self, 'speaker_change_penalty', 0.1)
            
        if hasattr(self, 'embedding_distance_threshold'):
            config["embedding_distance_threshold"] = self.embedding_distance_threshold
            
        return config
    
    def get_output_subdirs(self) -> Dict[str, str]:
        """Get output subdirectory paths."""
        return {
            'vtt': os.path.join(self.final_output_dir, 'vtt'),
            'rttm': os.path.join(self.final_output_dir, 'rttm'), 
            'rtf': os.path.join(self.final_output_dir, 'rtf'),
            'txt': os.path.join(self.final_output_dir, 'txt'),
            'stats': os.path.join(self.final_output_dir, 'stats'),
            'logs': os.path.join(self.final_output_dir, 'logs')
        }


# Default configuration instance
DEFAULT_DIARIZATION_CONFIG = DiarizationConfig()
