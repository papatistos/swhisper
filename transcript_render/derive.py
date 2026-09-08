"""Turn the canonical diarized segments into the view a reader asked for.

The pipeline bakes pause durations into the word list as literal ``(1.3)``
tokens. That is what makes hiding a disfluency marker destructive: drop the
``[**]`` word and the pauses printed on either side of it are now wrong, since
the marker's own span belongs to neither.

So there are two paths here.

* Nothing removed. The stored pauses are still correct, so they are left
  exactly where they are and only display filters apply. The default view is
  therefore the pipeline's own output, unchanged.
* Markers removed. The stored pauses are stale and get rebuilt from the
  surviving word timings, using the same rule as
  ``SilenceMarkerProcessor.add_word_level_silence_markers``.

Rebuilding cannot reproduce the pipeline byte-for-byte in general, because
``BackfillMerger`` splits silences around words recovered on a second pass and
reattaches the halves to neighbouring segments. Pause *durations* survive that
faithfully; which segment a pause hangs off can shift. That only affects the
rebuild path, where the pauses are being merged anyway.
"""

import copy
from typing import Any, Dict, List, Optional

from .markers import (
    get_word_text,
    is_disfluency_marker,
    is_marker_token,
    is_silence_token,
    parse_silence_duration,
)
from .options import RenderOptions

# What diarize/config.py uses when the JSON does not say.
DEFAULT_MIN_SILENCE = 0.2


def _format_silence(rounded: float) -> str:
    """Render a rounded gap the way the pipeline does: '(1.3)', '(.4)'."""
    return f"({rounded:.1f})".replace("(0.", "(.")


def _word_end(word: Dict[str, Any]) -> float:
    value = word.get('end', word.get('start', 0))
    return 0.0 if value is None else value


def _word_start(word: Dict[str, Any], fallback: float) -> float:
    value = word.get('start', word.get('end', fallback))
    return fallback if value is None else value


def _pause_seconds(word: Dict[str, Any]) -> Optional[float]:
    """How long a stored pause token is, in seconds."""
    recorded = word.get('silence_seconds')
    if recorded is not None:
        return recorded
    parsed = parse_silence_duration(get_word_text(word))
    if parsed is not None:
        return parsed
    start, end = word.get('start'), word.get('end')
    if start is not None and end is not None:
        return round(max(0.0, end - start), 1)
    return None


# ---------------------------------------------------------------------------
# Presentation: applied to both paths, never changes durations
# ---------------------------------------------------------------------------

def _present(segments: List[Dict[str, Any]], options: RenderOptions) -> List[Dict[str, Any]]:
    """Apply the display-only filters: hide pause text, drop short pauses.

    With default options this is a no-op on everything the formatters read, so
    the rendered result is identical to the pipeline's own.
    """
    hide_text = not options.pauses
    minimum = options.min_pause
    if not hide_text and minimum <= 0:
        return segments  # nothing to do; the stored data is already the answer

    out: List[Dict[str, Any]] = []
    for segment in segments:
        words: List[Dict[str, Any]] = []
        for word in segment.get('words', []) or []:
            if not is_silence_token(word):
                words.append(word)
                continue
            seconds = _pause_seconds(word)
            if seconds is not None and seconds < minimum:
                continue
            pause = dict(word)
            if seconds is not None:
                # Record it so a pause whose text is hidden can still break the
                # paragraph without the renderer parsing the token back.
                pause['silence_seconds'] = seconds
            if hide_text:
                pause['word'] = ''
                pause['text'] = ''
            words.append(pause)

        if not words and segment.get('is_silence_marker', False):
            continue
        new_segment = copy.copy(segment)
        new_segment['words'] = words
        if segment.get('is_silence_marker', False) and words:
            seconds = _pause_seconds(words[0])
            if seconds is not None:
                new_segment['silence_seconds'] = seconds
            if hide_text:
                new_segment['text'] = ''
        out.append(new_segment)
    return out


# ---------------------------------------------------------------------------
# Rebuild: only when removing words has invalidated the stored pauses
# ---------------------------------------------------------------------------

def _strip(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop every pause and every disfluency marker.

    Returns the segment list as it stood before pauses were computed, which is
    the only state they can be derived from again.

    A segment flagged ``is_silence_marker`` is not necessarily a pure pause:
    ``BackfillMerger`` leaves words it recovered inside the flagged segment. So
    the flag alone cannot decide what to drop — only a segment with nothing but
    pauses left in it is discarded.
    """
    base: List[Dict[str, Any]] = []
    for segment in segments:
        words = [
            word for word in (segment.get('words', []) or [])
            if not is_silence_token(word)
            and not is_disfluency_marker(get_word_text(word))
        ]
        if not words and segment.get('is_silence_marker', False):
            continue
        new_segment = copy.copy(segment)
        new_segment['words'] = words
        base.append(new_segment)
    return base


def _align_boundaries(segments: List[Dict[str, Any]]) -> None:
    """Match segment start/end to the first and last surviving word.

    Mirrors ``WordProcessor.align_segment_boundaries_to_words``, which the
    pipeline runs after removing markers for the same reason.
    """
    for segment in segments:
        words = segment.get('words') or []
        if not words:
            continue
        if words[0].get('start') is not None:
            segment['start'] = words[0]['start']
        if words[-1].get('end') is not None:
            segment['end'] = words[-1]['end']


def _make_pause(start: float, end: float, rounded: float, show_text: bool,
                after_disfluency: bool) -> Dict[str, Any]:
    text = _format_silence(rounded) if show_text else ''
    return {
        'start': start,
        'end': end,
        'word': text,
        'text': text,
        'speaker': 'SILENCE',
        'confidence': 1.0,
        'is_silence_marker': True,
        'after_disfluency': after_disfluency,
        'silence_seconds': rounded,
    }


def _rederive(segments: List[Dict[str, Any]], minimum: float,
              show_text: bool) -> List[Dict[str, Any]]:
    """Rebuild pauses from word timings after markers have been removed."""
    base = _strip(segments)
    _align_boundaries(base)
    out: List[Dict[str, Any]] = []

    for index, segment in enumerate(base):
        words = segment.get('words') or []
        if not words:
            out.append(segment)
            continue

        enhanced: List[Dict[str, Any]] = []
        for i, word in enumerate(words):
            enhanced.append(word)
            if i + 1 >= len(words):
                continue
            current_end = _word_end(word)
            next_start = _word_start(words[i + 1], current_end)
            gap = next_start - current_end
            if gap >= minimum:
                enhanced.append(_make_pause(
                    current_end, next_start, round(gap, 1), show_text,
                    after_disfluency=is_marker_token(get_word_text(word)),
                ))

        segment['words'] = enhanced
        out.append(segment)

        # Gap to the next segment becomes a standalone pause segment. Look past
        # segments left wordless by filtering, so the pause they used to sit in
        # is reported rather than silently swallowed.
        next_words = None
        for later in base[index + 1:]:
            candidate = later.get('words') or []
            if candidate:
                next_words = candidate
                break
        if not next_words:
            continue
        current_end = _word_end(words[-1])
        next_start = _word_start(next_words[0], current_end)
        gap = next_start - current_end
        if gap < minimum:
            continue
        rounded = round(gap, 1)
        pause = _make_pause(current_end, next_start, rounded, show_text,
                            after_disfluency=False)
        out.append({
            'start': current_end,
            'end': next_start,
            'text': pause['text'],
            'speaker': 'SILENCE',
            'speaker_confidence': 1.0,
            'is_silence_marker': True,
            'silence_seconds': rounded,
            'words': [pause],
        })

    return out


def derive_segments(segments: List[Dict[str, Any]],
                    options: Optional[RenderOptions] = None,
                    pipeline_min_silence: float = DEFAULT_MIN_SILENCE
                    ) -> List[Dict[str, Any]]:
    """Return a new segment list presenting *segments* under *options*.

    *pipeline_min_silence* is the shortest gap the run recorded as a pause; it
    bounds how fine the rebuild may go, so a rebuilt transcript stays
    comparable with the stored one. Read it from the JSON metadata when
    present.

    The input is never mutated. Word dicts are shared where unchanged.
    """
    options = options or RenderOptions()
    if options.disfluencies:
        return _present(segments, options)
    minimum = max(options.min_pause, pipeline_min_silence)
    return _present(_rederive(segments, minimum, show_text=options.pauses), options)
