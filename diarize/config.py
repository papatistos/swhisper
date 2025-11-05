"""Configuration settings for diarization processing."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from path_settings import get_path_settings
except ModuleNotFoundError:                                 # Allow running modules directly from the diarize package
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from path_settings import get_path_settings

_PATH_SETTINGS = get_path_settings()


def _get_env_bool(key: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ('true', '1', 'yes', 'on')


def _get_env_int(key: str, default: Optional[int]) -> Optional[int]:
    """Parse integer environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Parse float environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _default_hf_token() -> str:
    """Resolve Hugging Face token from multiple supported environment variables."""
    for env_var in ("HUGGINGFACE_ACCESS_TOKEN", "HUGGING_FACE_TOKEN"):
        token = os.getenv(env_var)
        if token:
            return token
    return ""


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
    hugging_face_token: str = field(default_factory=_default_hf_token)

    # Diarization settings
    min_speakers: int = _get_env_int("SWHISPER_MIN_SPEAKERS", 1)
    max_speakers: Optional[int] = _get_env_int("SWHISPER_MAX_SPEAKERS", None)
    device: str = os.getenv("SWHISPER_DIARIZE_DEVICE", "mps")  # "mps", "cuda", or "cpu"

    # Processing options
    force_reprocess: bool = _get_env_bool("SWHISPER_DIARIZE_FORCE_REPROCESS", False)  # Set to True to reprocess files even if output already exists (i.e. .ok files are present)

    # Advanced speaker separation settings
    use_temporal_constraints: bool = _get_env_bool("SWHISPER_TEMPORAL_CONSTRAINTS", True)  # Might be ignored by post 3.1 models,  ignored by precision-2
    speaker_change_penalty: float = _get_env_float("SWHISPER_SPEAKER_CHANGE_PENALTY", 0.1)  # Penalty for frequent speaker changes (default 0.1) - Might be ignored by post 3.1 models,  ignored by precision-2
    embedding_distance_threshold: float = _get_env_float("SWHISPER_EMBEDDING_THRESHOLD", 0.6)  # Lower values = Stricter matching - speakers must sound very similar to be grouped together (default 0.6 ?) - Might be ignored by post 3.1 models, ignored by precision-2

    # Pipeline configuration
    pipeline_model: str = os.getenv("SWHISPER_PIPELINE_MODEL", "pyannote/speaker-diarization-3.1")  # this is ignored when diarize4.py is used

    # Community-1 pipeline parameters (VBxClustering) - used when diarize4.py is used
    vbx_clustering_threshold: float = _get_env_float("SWHISPER_VBX_THRESHOLD", 0.6)  # VBx clustering threshold (default: 0.6) - lower values create more speakers (speakers must sound very similar to be grouped)
    vbx_fa: float = _get_env_float("SWHISPER_VBX_FA", 0.07)  # False alarm probability for speaker activity (default: 0.07) - lower = more conservative speech detection
    vbx_fb: float = _get_env_float("SWHISPER_VBX_FB", 0.8)  # Probability of not detecting a speaker (default: 0.8), - lower = more aggressive speaker separation
    community_min_duration_off: float = _get_env_float("SWHISPER_COMMUNITY_MIN_DURATION_OFF", 0.0)  # Minimum duration for non-speech segments (default: 0.0)
    # segmentation_threshold? community-1 apparently does not expose it.

    # Legacy 3.1 pipeline parameters (AgglomerativeClustering)
    segmentation_threshold: float = _get_env_float("SWHISPER_SEGMENTATION_THRESHOLD", 0.01)  # Lower threshold for more sensitive speaker changes (this concerns segmentation)
    min_duration_on: float = _get_env_float("SWHISPER_MIN_DURATION_ON", 0.0) if os.getenv("SWHISPER_MIN_DURATION_ON") else None  # default seems to be 0.0
    min_duration_off: float = _get_env_float("SWHISPER_MIN_DURATION_OFF", 0.0) if os.getenv("SWHISPER_MIN_DURATION_OFF") else None  # default seems to be 0.0
    clustering_method: str = os.getenv("SWHISPER_CLUSTERING_METHOD", "centroid")
    clustering_min_cluster_size: int = _get_env_int("SWHISPER_CLUSTERING_MIN_SIZE", 15)  # Minimum number of frames (not segments!) to form a cluster
    clustering_threshold: float = _get_env_float("SWHISPER_CLUSTERING_THRESHOLD", 0.15)  # (was: .15) Lower threshold for stricter clustering

# embedding_distance_threshold: 0.7 (default)
# lustering_threshold: 0.15 (typical default)
# segmentation_threshold: 0.3 (voice activity detection)
# min_duration_on: 0.0 (no minimum duration filter)
# clustering_threshold: 0.15 (default for clustering)
# clustering_min_cluster_size: 10-15?
#




    # Pyannote 4 settings (some of the above are also applied)
    use_exclusive_speaker_diarization: bool = _get_env_bool("SWHISPER_USE_EXCLUSIVE_DIARIZATION", True)  # Use exclusive speaker diarization stream if available

    # Premium diarization service (pyannote precision-2)
    use_precision_service: bool = _get_env_bool("SWHISPER_USE_PRECISION", False)
    precision_pipeline_model: str = os.getenv("SWHISPER_PRECISION_MODEL", "pyannote/speaker-diarization-precision-2")
    precision_api_token: Optional[str] = None
    precision_token_env_vars: Tuple[str, ...] = ("PYANNOTEAI_API_KEY", "PYANNOTE_API_KEY")

    # Silence detection settings (post-processing)
    min_silence_duration: float = _get_env_float("SWHISPER_MIN_SILENCE_DURATION", 0.2)  # (rounded) silence durations below this value will be ignored
    include_silence_markers: bool = _get_env_bool("SWHISPER_INCLUDE_SILENCE", True)  # silence duration in transcript
    log_silence_gaps: bool = _get_env_bool("SWHISPER_LOG_SILENCE", False)  # If true, between-word-gap-durations will be logged in separate file
    silence_gap_linebreak_threshold: Optional[float] = _get_env_float("SWHISPER_SILENCE_LINEBREAK", 1.0) if os.getenv("SWHISPER_SILENCE_LINEBREAK") else 1.0  # Surround long silences with blank lines in transcripts. Set to 0 for no linebreaks

    # Timestamp correction
    whisper_timestamp_offset: float = _get_env_float("SWHISPER_WHISPER_TIMESTAMP_OFFSET", 0.1)  # Offset to add to all Whisper timestamps for better alignment with audio and pyannote segments (default: 0.1s)

    # Word-level processing
    smoothing_enabled: bool = _get_env_bool("SWHISPER_SMOOTHING_ENABLED", True)
    min_speaker_words: int = _get_env_int("SWHISPER_MIN_SPEAKER_WORDS", 3)
    preserve_markers: bool = _get_env_bool("SWHISPER_PRESERVE_MARKERS", True)  # Preserve disfluency markers for sounds that could not be transcribed
    preserved_markers: List[str] = None

    # Targeted re-transcription for empty diarization turns
    backfill_missing_turns: bool = _get_env_bool("SWHISPER_BACKFILL_ENABLED", True)
    backfill_model: str = os.getenv("SWHISPER_BACKFILL_MODEL", "KBLab/kb-whisper-large")
    backfill_device: Optional[str] = os.getenv("SWHISPER_BACKFILL_DEVICE") if os.getenv("SWHISPER_BACKFILL_DEVICE") else None
    backfill_overlap: float = _get_env_float("SWHISPER_BACKFILL_OVERLAP", 0.5)
    backfill_save_audio_snippets: bool = _get_env_bool("SWHISPER_BACKFILL_SAVE_AUDIO", False)
    backfill_snippet_dir: Optional[str] = os.getenv("SWHISPER_BACKFILL_SNIPPET_DIR") if os.getenv("SWHISPER_BACKFILL_SNIPPET_DIR") else None
    backfill_cache_enabled: bool = _get_env_bool("SWHISPER_BACKFILL_CACHE_ENABLED", True)
    backfill_cache_dir: Optional[str] = os.getenv("SWHISPER_BACKFILL_CACHE_DIR") if os.getenv("SWHISPER_BACKFILL_CACHE_DIR") else None
    backfill_min_duration: float = _get_env_float("SWHISPER_BACKFILL_MIN_DURATION", 0.1)
    backfill_ignore_words: List[str] = None  # Words to replace with disfluency markers (e.g., hallucinations like "Balans")

    # Diarization caching (cache raw pyannote results)
    diarization_cache_enabled: bool = _get_env_bool("SWHISPER_DIARIZATION_CACHE_ENABLED", True)
    diarization_cache_dir: Optional[str] = os.getenv("SWHISPER_DIARIZATION_CACHE_DIR") if os.getenv("SWHISPER_DIARIZATION_CACHE_DIR") else None


    
    # Output formatting
    tsv_word_per_line: bool = _get_env_bool("SWHISPER_TSV_WORD_PER_LINE", True)  # If True, TSV output writes one word per line instead of one segment per line

    # Output preamble for transcript files
    output_preamble: str = os.getenv("SWHISPER_OUTPUT_PREAMBLE", """This transcript was generated using swhisper (https://github.com/papatistos/swhisper). 
                                     
Note 1: If the transcript contains markers like [*], [**], [***] they indicate sounds that could not be transcribed. The number of asterisks roughly reflects the sound's duration in tenths of a second. This can be turned off in the swisper settings. If there is a number before the asterisks, e.g. [S02**], this indicates overlapping speech by another speaker (in this case: speaker 02). Not all (possibly not even the majority of) overlapping speech is detected.

Note 2: You may also find parentheses with a number inside, e.g. (.3). These indicate silences and the number indicates the duration of the silence in seconds. Currently, not all silences are reliably detected, but those that are may be useful, especially when they indicate longer pauses. This can also be turned off in the swhisper settings. Long silences are surrounded by blank lines (even when there is no speaker change). This can be adjusted or turned off in the swhisper settings.

Note 3: Speaker detection is not perfect. The transcript may show too few or too many different speakers, but the excess (or unknown) speakers should have few turns attributed to them, so it should be easy to fix manually.
""")
    
    def get_preamble_with_transcript_id(self, transcript_id: str) -> str:
        """Generate preamble with transcript ID included."""
        return f"""Transcript-ID: {transcript_id}

{self.output_preamble}"""
        
    # Parameter testing options
    enable_parameter_testing: bool = _get_env_bool("SWHISPER_PARAMETER_TESTING", False)
    test_output_dir: str = os.getenv("SWHISPER_TEST_OUTPUT_DIR", "parameter_tests")
    test_max_combinations: int = _get_env_int("SWHISPER_TEST_MAX_COMBINATIONS", 320)
    test_single_file_mode: bool = _get_env_bool("SWHISPER_TEST_SINGLE_FILE", True)  # Test on one file first before full testing

    # Parameter testing notes:
    # - Only enable if you want to find optimal embedding_distance_threshold values
    # - With max_speakers=2 (fixed), most parameter testing is meaningless

    # Directory settings
    output_dir: str = os.getenv("SWHISPER_OUTPUT_DIR", "transcripts")
    log_dir: str = os.getenv("SWHISPER_LOG_DIR", "transcripts")
    default_audio_dir: str = DEFAULT_DIARIZE_AUDIO_DIR
    default_json_input_dir: str = "whisper-json-output"
    default_audio_subdir: str = os.getenv("SWHISPER_AUDIO_SUBDIR", "")  # Leave empty if audio files are directly in audio_dir
    
    def __post_init__(self):
        """Initialize default values that can't be set in field definition."""
        if self.preserved_markers is None:
            self.preserved_markers = ["[DISCONTINUITY]", "[SILENCE]", "[OVERLAP]", "[*]"]
        if not getattr(self, 'backfill_model', None):
            self.backfill_model = "KBLab/kb-whisper-large"
        if getattr(self, 'backfill_overlap', 0.0) < 0.0:
            self.backfill_overlap = 0.0
        if getattr(self, 'backfill_min_duration', 0.0) < 0.0:
            self.backfill_min_duration = 0.0
        if not getattr(self, 'precision_api_token', None):
            for env_var in getattr(self, 'precision_token_env_vars', ()):  # type: ignore[arg-type]
                token = os.getenv(env_var, "").strip()
                if token:
                    self.precision_api_token = token
                    break
        # Parse comma-separated list of words to ignore in backfill
        if self.backfill_ignore_words is None:
            ignore_words_str = os.getenv("SWHISPER_BACKFILL_IGNORE_WORDS", "")
            if ignore_words_str:
                # Split by comma and normalize (strip whitespace, convert to lowercase for case-insensitive matching)
                self.backfill_ignore_words = [w.strip().lower() for w in ignore_words_str.split(",") if w.strip()]
            else:
                self.backfill_ignore_words = []
    
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

    def get_backfill_cache_dir(self) -> str:
        """Resolve directory used for backfill cache storage."""
        base_dir = getattr(self, 'backfill_cache_dir', None)
        if not base_dir:
            base_dir = os.path.join(self.final_log_dir, "backfill_cache")
        return base_dir
    
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
        
        # Add VBx clustering parameters (for community-1 pipeline)
        if hasattr(self, 'vbx_clustering_threshold'):
            config["vbx_clustering_threshold"] = self.vbx_clustering_threshold
        if hasattr(self, 'vbx_fa'):
            config["vbx_fa"] = self.vbx_fa
        if hasattr(self, 'vbx_fb'):
            config["vbx_fb"] = self.vbx_fb
        if hasattr(self, 'community_min_duration_off'):
            config["community_min_duration_off"] = self.community_min_duration_off
        
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
            'tsv': os.path.join(self.final_output_dir, 'tsv'),
            'json': os.path.join(self.final_output_dir, 'json'),
            'stats': os.path.join(self.final_output_dir, 'stats'),
            'logs': os.path.join(self.final_output_dir, 'logs')
        }


# Default configuration instance
DEFAULT_DIARIZATION_CONFIG = DiarizationConfig()
