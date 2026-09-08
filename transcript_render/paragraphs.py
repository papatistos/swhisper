"""Reconstruct readable paragraph text from a segment's word list.

Kept free of ML imports so both the diarization pipeline and the on-demand
renderer share one implementation.
"""

from typing import Any, Dict, List, Optional

from .markers import parse_silence_duration


def create_paragraph_text_from_words(
    segment: Dict[str, Any],
    gap_threshold: Optional[float] = None
) -> str:
    """Reconstruct segment text from words while optionally adding blank lines around long silences."""
    words = segment.get('words', [])
    if not words:
        return segment.get('text', '').strip()

    # Behaviour when no threshold is provided
    if not gap_threshold or gap_threshold <= 0:
        word_texts = []
        for word in words:
            word_text = word.get('word', word.get('text', ''))
            if word_text:  # Preserve markers and silence tokens
                word_texts.append(word_text)
        return ' '.join(word_texts).strip()

    lines: List[str] = []
    current_tokens: List[str] = []

    def flush_current_tokens() -> None:
        if current_tokens:
            lines.append(' '.join(current_tokens).strip())
            current_tokens.clear()

    for word in words:
        word_text = word.get('word', word.get('text', ''))
        if not word_text:
            continue

        add_line_breaks = False
        if word.get('is_silence_marker', False):
            duration = parse_silence_duration(word_text)
            if duration is not None and duration >= gap_threshold:
                add_line_breaks = True

        if add_line_breaks:
            flush_current_tokens()
            # Always add blank line before long silence markers
            # (even at start of segment, for visual separation)
            if not lines:
                # First item is a long silence - add leading blank
                lines.append('')
            elif lines[-1] != '':
                # Add blank line after non-blank line
                lines.append('')
            lines.append(word_text.strip())
            lines.append('')
        else:
            current_tokens.append(word_text.strip())

    flush_current_tokens()

    if not lines:
        return ' '.join(current_tokens).strip()

    while lines and lines[0] == '':
        lines.pop(0)
    while lines and lines[-1] == '':
        lines.pop()

    cleaned_lines: List[str] = []
    previous_blank = False
    for line in lines:
        if line == '':
            if previous_blank:
                continue
            previous_blank = True
        else:
            previous_blank = False
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
