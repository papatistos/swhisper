"""Marker and silence-token vocabulary shared by the pipeline and the renderer.

This module is deliberately free of ``torch``/``whisper`` imports so the web UI
and the standalone renderer can reuse it without loading the ML stack.
"""

import math
import re
from typing import Any, Dict, Optional, Tuple


def parse_silence_duration(token: str) -> Optional[float]:
    """Extract numeric silence duration from a token like '(1.3)'."""
    if not token:
        return None

    stripped = token.strip()
    if stripped.startswith('(') and stripped.endswith(')'):
        stripped = stripped[1:-1]

    stripped = stripped.replace(',', '.')
    if not stripped:
        return None

    try:
        return float(stripped)
    except ValueError:
        return None


def get_word_text(word: Dict[str, Any]) -> str:
    """Fetch the primary textual representation for a word token."""
    value = word.get('word')
    if isinstance(value, str) and value:
        return value
    value = word.get('text')
    return value if isinstance(value, str) else ""


def set_word_text(word: Dict[str, Any], new_text: str) -> None:
    """Synchronize word/text fields after updating marker contents."""
    word['word'] = new_text
    word['text'] = new_text


def marker_asterisk_count(duration: float) -> int:
    """Return one asterisk per started 0.1 s of a marker, from 1 to 50.

    Rounding the tenths first stops float error from adding an asterisk to an
    exact tenth, e.g. 3.62 - 3.42 = 0.20000000000000018 s.
    """
    return max(1, min(50, math.ceil(round(duration * 10, 6))))


STATIC_MARKERS = {"[DISCONTINUITY]", "[SILENCE]", "[OVERLAP]"}
DISFLUENCY_MARKER_PATTERN = re.compile(r"^\[\*+\]$")
SPACED_MARKER_PATTERN = re.compile(r"^\[\s\*{1,50}\s\]$")
MARKER_INLINE_HINT_PATTERN = re.compile(r"^\[(\s*)(S\d{2}(?:[+/]S\d{2})*)(\*+)(\s*)\]$")
MARKER_PREFIX_HINT_PATTERN = re.compile(r"^(S\d{2}(?:[+/]S\d{2})*)\s+")


def strip_marker_hint(token: str) -> str:
    """Remove leading speaker hint prefixes from a marker token."""
    if not isinstance(token, str):
        return ""
    cleaned = token.strip()
    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        return f"[{leading_spaces}{stars}{trailing_spaces}]"
    return MARKER_PREFIX_HINT_PATTERN.sub("", cleaned)


def split_marker_hint_and_body(token: str) -> Tuple[str, str]:
    """Return the hint prefix (if present) and the base marker body."""
    if not isinstance(token, str):
        return "", ""
    cleaned = token.strip()
    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        hint = inline.group(2)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        base_marker = f"[{leading_spaces}{stars}{trailing_spaces}]"
        return hint, base_marker
    match = MARKER_PREFIX_HINT_PATTERN.match(cleaned)
    if match:
        hint = match.group(1)
        remainder = cleaned[match.end():].lstrip()
        return hint, remainder
    return "", cleaned


def inject_hint_into_marker(marker_body: str, hint: str) -> str:
    """Place the speaker hint immediately inside the marker brackets."""
    if not marker_body or not hint:
        return marker_body

    cleaned = marker_body.strip()

    inline = MARKER_INLINE_HINT_PATTERN.match(cleaned)
    if inline:
        leading_spaces = inline.group(1)
        stars = inline.group(3)
        trailing_spaces = inline.group(4)
        return f"[{leading_spaces}{hint}{stars}{trailing_spaces}]"

    if cleaned.startswith('[') and cleaned.endswith(']'):
        inner = cleaned[1:-1]
        if inner:
            first_star = inner.find('*')
            if first_star != -1:
                return f"[{inner[:first_star]}{hint}{inner[first_star:]}]"
            if set(inner.replace(' ', '')) <= {'*'}:
                return f"[{hint}{inner}]"

    return f"{hint} {marker_body}".strip()


def format_speaker_hint(label: Optional[str]) -> str:
    """Convert diarization speaker labels into the compact Sxx hint form."""
    if not label or not isinstance(label, str):
        return ""

    if label.upper() in {"SILENCE", "UNKNOWN"}:
        return ""

    match = re.search(r"(\d+)", label)
    if match:
        try:
            index = int(match.group(1))
            return f"S{index + 1:02d}"
        except ValueError:
            pass

    alnum = re.sub(r"[^A-Za-z0-9]", "", label).upper()
    if not alnum:
        return ""
    if len(alnum) <= 3:
        return alnum
    return alnum[:3]


def _normalize_speaker_label(label: Optional[str]) -> str:
    """Normalize speaker labels for comparisons, ignoring formatting differences."""
    if not label or not isinstance(label, str):
        return ""

    cleaned = label.strip().upper()
    match = re.search(r"(\d+)", cleaned)
    if match:
        digits = match.group(1).lstrip('0')
        return digits or "0"

    alnum = re.sub(r"[^A-Z0-9]", "", cleaned)
    return alnum


def is_marker_token(token: str) -> bool:
    """Return True for preserved markers, including variable-length asterisk forms."""
    if not token:
        return False
    cleaned = strip_marker_hint(token)
    return (
        cleaned in STATIC_MARKERS
        or bool(DISFLUENCY_MARKER_PATTERN.match(cleaned))
        or bool(SPACED_MARKER_PATTERN.match(cleaned))
    )


def is_silence_token(word: Dict[str, Any]) -> bool:
    """Identify silence placeholders regardless of explicit flags."""
    if not isinstance(word, dict):
        return False

    if word.get('is_silence_marker', False):
        return True

    if word.get('speaker') == 'SILENCE':
        return True

    token = word.get('word', word.get('text', ''))
    if not isinstance(token, str):
        return False

    cleaned = token.strip()
    if not cleaned:
        return False

    if cleaned.startswith('(') and cleaned.endswith(')'):
        if parse_silence_duration(cleaned) is not None:
            return True

    return False


def is_disfluency_marker(token: str) -> bool:
    """Return True for the asterisk markers the pipeline strips when
    ``preserve_markers`` is off — the same predicate as
    ``WordProcessor.remove_disfluency_markers_from_segments``."""
    if not token:
        return False
    cleaned = strip_marker_hint(token)
    return bool(cleaned and DISFLUENCY_MARKER_PATTERN.match(cleaned))


# Historic private alias; ``diarize`` still imports this name.
_is_silence_token = is_silence_token
