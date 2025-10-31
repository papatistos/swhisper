"""Configuration settings for the transcription system."""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re

from path_settings import get_path_settings

_PATH_SETTINGS = get_path_settings()

DEFAULT_AUDIO_DIR = str(_PATH_SETTINGS.audio_dir) if _PATH_SETTINGS.audio_dir else os.getenv(
    "TRANSCRIBE_AUDIO_DIR", "audio"
)
DEFAULT_TEMP_DIR = str(_PATH_SETTINGS.temp_dir) if _PATH_SETTINGS.temp_dir else os.getenv(
    "SWHISPER_TEMP_DIR"
)

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

def _get_env_list_float(key: str, default: List[float]) -> List[float]:
    """Parse comma-separated float list from environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return [float(x.strip()) for x in val.split(',')]
    except ValueError:
        return default

@dataclass
class TranscriptionConfig:
    """Main configuration for transcription settings."""

    # Audio directories
    audio_dir: str = DEFAULT_AUDIO_DIR
    json_dir: str = os.getenv("SWHISPER_JSON_DIR", "whisper-json-output")

    # Model settings
    model_str: str = os.getenv("SWHISPER_MODEL", "KBLab/kb-whisper-large")
    revision: str = os.getenv("SWHISPER_MODEL_REVISION", "strict")
    device: str = os.getenv("SWHISPER_DEVICE", "mps")

    # Processing limits
    file_limit: Optional[int] = _get_env_int("SWHISPER_FILE_LIMIT", None)
    force_reprocess: bool = _get_env_bool("SWHISPER_FORCE_REPROCESS", False)  # if true, files are transcribed even if transcriptions already exist

    # Temporary directory configuration
    custom_temp_dir: Optional[str] = DEFAULT_TEMP_DIR
    preserve_checkpoints: bool = _get_env_bool("SWHISPER_PRESERVE_CHECKPOINTS", True)

    # Chunking configuration
    target_chunk_duration: int = _get_env_int("SWHISPER_CHUNK_DURATION", 180)    # size in seconds of audio chunks to be processed individually (to handle memory constraints)
    min_silence_duration: float = _get_env_float("SWHISPER_MIN_SILENCE", 0.5)    # Minimum duration of silence to place the chunk boundary
    overlap_duration: float = _get_env_float("SWHISPER_OVERLAP_DURATION", 1.0)   # increasing this from 1 to make sure there are enough words to reliabley stitch the transcript together. But not too long, otherwise it will be removed as duplicate

    # Environment checks
    check_revision: bool = _get_env_bool("SWHISPER_CHECK_REVISION", False)
    check_environment: bool = _get_env_bool("SWHISPER_CHECK_ENVIRONMENT", True)


@dataclass
class WhisperSettings:
    """Whisper-specific transcription settings (for whisper_timestamped)."""

    language: str = os.getenv("SWHISPER_LANGUAGE", "sv")
    task: str = os.getenv("SWHISPER_TASK", "transcribe")
    remove_punctuation_from_words: bool = False
    compute_word_confidence: bool = _get_env_bool("SWHISPER_COMPUTE_CONFIDENCE", True)  # this does not seem to work with the KBLab model, though it should according to this: https://huggingface.co/KBLab/kb-whisper-large/discussions/14
    include_punctuation_in_confidence: bool = False
    refine_whisper_precision: float = _get_env_float("SWHISPER_REFINE_PRECISION", 0.5)
    min_word_duration: float = _get_env_float("SWHISPER_MIN_WORD_DURATION", 0.02)
    plot_word_alignment: bool = False
    word_alignment_most_top_layers: Optional[int] = _get_env_int("SWHISPER_ALIGNMENT_LAYERS", 0) if os.getenv("SWHISPER_ALIGNMENT_LAYERS") else None
    remove_empty_words: bool = False
    seed: int = _get_env_int("SWHISPER_SEED", 1234)
    vad: str = os.getenv("SWHISPER_VAD", "silero:v3.1")  # silero:v3.1, silero:v4.0, auditok or None. With None, no VAD is applied before transcription.
    detect_disfluencies: bool = _get_env_bool("SWHISPER_DETECT_DISFLUENCIES", True)
    trust_whisper_timestamps: bool = _get_env_bool("SWHISPER_TRUST_TIMESTAMPS", False)
    naive_approach: bool = False  # I think this gets overridden if beam_size and best_of are set, so we are using naive mode anyway
    beam_size: int = _get_env_int("SWHISPER_BEAM_SIZE", 5)  # less efficient but better results
    best_of: int = _get_env_int("SWHISPER_BEST_OF", 5)  # less efficient but better results
    temperature: List[float] = None  # if None, set below in __post_init__
    compression_ratio_threshold: float = _get_env_float("SWHISPER_COMPRESSION_THRESHOLD", 2.4)
    logprob_threshold: float = _get_env_float("SWHISPER_LOGPROB_THRESHOLD", -1.0)  # value between 0 and -1 . (-1 effectuively disables the threshold, i.e. no segment will be excluded because of low confidence)
    no_speech_threshold: float = _get_env_float("SWHISPER_NO_SPEECH_THRESHOLD", 0.6)  # we have to be more than 60% sure there is no speech to dismiss a segment as silence
    fp16: Optional[bool] = _get_env_bool("SWHISPER_FP16", True) if os.getenv("SWHISPER_FP16") else None
    condition_on_previous_text: bool = _get_env_bool("SWHISPER_CONDITION_ON_PREVIOUS", True)
    initial_prompt: str = os.getenv("SWHISPER_INITIAL_PROMPT", "")
    suppress_tokens: str = os.getenv("SWHISPER_SUPPRESS_TOKENS", "50364")
    sample_len: Optional[int] = None
    verbose: bool = _get_env_bool("SWHISPER_VERBOSE", True)
    
    def __post_init__(self):
        if self.temperature is None:
            # Check environment variable first
            temp_str = os.getenv("SWHISPER_TEMPERATURE")
            if temp_str:
                try:
                    self.temperature = [float(x.strip()) for x in temp_str.split(',')]
                except ValueError:
                    self.temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            else:
                self.temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # less efficient but better results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for whisper.transcribe()."""
        return {
            "language": self.language,
            "task": self.task,
            "remove_punctuation_from_words": self.remove_punctuation_from_words,
            "compute_word_confidence": self.compute_word_confidence,
            "include_punctuation_in_confidence": self.include_punctuation_in_confidence,
            "refine_whisper_precision": self.refine_whisper_precision,
            "min_word_duration": self.min_word_duration,
            "plot_word_alignment": self.plot_word_alignment,
            "word_alignment_most_top_layers": self.word_alignment_most_top_layers,
            "remove_empty_words": self.remove_empty_words,
            "seed": self.seed,
            "vad": self.vad,
            "detect_disfluencies": self.detect_disfluencies,
            "trust_whisper_timestamps": self.trust_whisper_timestamps,
            "naive_approach": self.naive_approach,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "fp16": self.fp16,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self.initial_prompt,
            "suppress_tokens": self.suppress_tokens,
            "sample_len": self.sample_len,
            "verbose": self.verbose
        }

# Default instances
DEFAULT_CONFIG = TranscriptionConfig()
DEFAULT_WHISPER_SETTINGS = WhisperSettings()
