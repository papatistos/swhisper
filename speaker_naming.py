"""Interview role suggestions from the final aligned transcript."""
from transcript_render.markers import (
    STATIC_MARKERS, get_word_text, is_backfill_failure_marker,
    is_disfluency_marker, parse_silence_duration,
)


def interview_names(segments, config):
    """Rank known speakers by words; ties retain first appearance order."""
    if getattr(config, 'recording_type', '').strip().lower() != 'interview':
        return {}
    counts = {}
    for segment in segments or []:
        label = segment.get('speaker')
        if not isinstance(label, str) or not label or label.upper() in {'SILENCE', 'UNKNOWN'}:
            continue
        counts.setdefault(label, 0)
        words = segment.get('words') or [{'text': segment.get('text', '')}]
        for word in words:
            text = get_word_text(word).strip()
            if (word.get('is_silence_marker') or text in STATIC_MARKERS
                    or is_disfluency_marker(text) or is_backfill_failure_marker(text)):
                continue
            for token in text.split():
                if (token in STATIC_MARKERS or is_disfluency_marker(token)
                        or is_backfill_failure_marker(token)
                        or (token.startswith('(') and token.endswith(')')
                            and parse_silence_duration(token) is not None)):
                    continue
                if any(char.isalnum() for char in token):
                    counts[label] += 1
    ranked = sorted((label for label in counts if counts[label]), key=lambda label: -counts[label])
    roles = [getattr(config, 'interviewee_name', '') or 'INTERVIEWEE',
             getattr(config, 'interviewer_name', '') or 'INTERVIEWER']
    names = {label: name.strip() for label, name in zip(ranked, roles)}
    # Use the same rules as manually entered names.
    from webui_speakers import validate
    return validate(names, list(counts))
