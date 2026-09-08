"""Transcript rendering that does not depend on the ML stack.

``diarize`` owns everything that needs ``torch``/``whisper``; this package owns
turning an already-diarized segment list into readable output. Keeping the two
apart lets the web UI and the CLI re-render a transcript in milliseconds
without loading a model.
"""

from .markers import (
    STATIC_MARKERS,
    DISFLUENCY_MARKER_PATTERN,
    SPACED_MARKER_PATTERN,
    MARKER_INLINE_HINT_PATTERN,
    MARKER_PREFIX_HINT_PATTERN,
    strip_marker_hint,
    split_marker_hint_and_body,
    inject_hint_into_marker,
    format_speaker_hint,
    is_marker_token,
    is_disfluency_marker,
    is_silence_token,
    parse_silence_duration,
    get_word_text,
    set_word_text,
)
from .paragraphs import create_paragraph_text_from_words
from .options import RenderOptions
from .derive import derive_segments, DEFAULT_MIN_SILENCE
from .formatters import (
    TranscriptFormatter,
    TXTFormatter,
    RTFFormatter,
    SPEAKER_STATS_WARNING,
)

__all__ = [
    'STATIC_MARKERS',
    'DISFLUENCY_MARKER_PATTERN',
    'SPACED_MARKER_PATTERN',
    'MARKER_INLINE_HINT_PATTERN',
    'MARKER_PREFIX_HINT_PATTERN',
    'strip_marker_hint',
    'split_marker_hint_and_body',
    'inject_hint_into_marker',
    'format_speaker_hint',
    'is_marker_token',
    'is_disfluency_marker',
    'is_silence_token',
    'parse_silence_duration',
    'get_word_text',
    'set_word_text',
    'create_paragraph_text_from_words',
    'RenderOptions',
    'derive_segments',
    'DEFAULT_MIN_SILENCE',
    'TranscriptFormatter',
    'TXTFormatter',
    'RTFFormatter',
    'SPEAKER_STATS_WARNING',
]
