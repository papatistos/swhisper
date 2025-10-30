"""Configuration settings for the transcription system."""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from path_settings import get_path_settings

_PATH_SETTINGS = get_path_settings()

DEFAULT_AUDIO_DIR = str(_PATH_SETTINGS.audio_dir) if _PATH_SETTINGS.audio_dir else os.getenv(
    "TRANSCRIBE_AUDIO_DIR", "audio"
)
DEFAULT_TEMP_DIR = str(_PATH_SETTINGS.temp_dir) if _PATH_SETTINGS.temp_dir else os.getenv(
    "SWHISPER_TEMP_DIR"
)

@dataclass
class TranscriptionConfig:
    """Main configuration for transcription settings."""
    
    # Audio directories
    audio_dir: str = DEFAULT_AUDIO_DIR
    json_dir: str = "whisper-json-output"
    
    # Model settings  
    model_str: str = "KBLab/kb-whisper-large"
    revision: str = "strict"
    device: str = "mps"
    
    # Processing limits
    file_limit: int = 3
    
    # Temporary directory configuration
    custom_temp_dir: Optional[str] = DEFAULT_TEMP_DIR
    preserve_checkpoints: bool = True
    
    # Chunking configuration
    target_chunk_duration: int = 180  # 3 minutes
    min_silence_duration: float = 0.5  # 0.5 seconds
    overlap_duration: float = 1.0  # 1 second
    
    # Environment checks
    check_revision: bool = False
    check_environment: bool = True

@dataclass 
class WhisperSettings:
    """Whisper-specific transcription settings (for whisper_timestamped)."""
    
    language: str = 'sv'
    task: str = 'transcribe'
    remove_punctuation_from_words: bool = False
    compute_word_confidence: bool = True                   # this does not seem to work with the KBLab model, though it should according to this: https://huggingface.co/KBLab/kb-whisper-large/discussions/14
    include_punctuation_in_confidence: bool = False
    refine_whisper_precision: float = 0.5
    min_word_duration: float = 0.02
    plot_word_alignment: bool = False
    word_alignment_most_top_layers: Optional[int] = None
    remove_empty_words: bool = False
    seed: int = 1234
    vad: str = "silero:v3.1"                               # silero:v3.1, silero:v4.0, auditok or None. With None, no VAD is applied before transcription.
    detect_disfluencies: bool = True
    trust_whisper_timestamps: bool = False
    naive_approach: bool = False                           # I think this gets overridden if beam_size and best_of are set, so we are using naive mode anyway
    beam_size: int = 5                                     # less efficient but better results
    best_of: int = 5                                       # less efficient but better results
    temperature: List[float] = None                        # if None, set below in __post_init__
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0                        # value between 0 and -1 . (-1 effectuively disables the threshold, i.e. no segment will be excluded because of low confidence)
    no_speech_threshold: float = 0.6                       # we have to be more than 60% sure there is no speech to dismiss a segment as silence
    fp16: Optional[bool] = None
    condition_on_previous_text: bool = False
    initial_prompt: str = 'öh öhm nä nähä jo jaha jaha jaså jaja oj ojojoj nja njäh nja näe haha hehe hihi asså ju mm mhm hmm eh ehm äh ähm'
    suppress_tokens: str = '50364'
    sample_len: Optional[int] = None
    verbose: bool = True
    
    def __post_init__(self):
        if self.temperature is None:
            self.temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # less efficient but better results
    
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
