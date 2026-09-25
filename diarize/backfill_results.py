"""Checkpointed backfill results, independent of model runtimes."""

from transcript_render.markers import get_word_text, is_low_confidence, is_marker_token, marker_asterisk_count


def turn_key(turn):
    return f"{turn.get('speaker', 'UNKNOWN')}_{float(turn['start']):.3f}_{float(turn['end']):.3f}"


def word_count(segment):
    if not segment or segment.get('is_placeholder'):
        return 0
    return sum(bool(get_word_text(w).strip()) and not is_marker_token(get_word_text(w))
               for w in segment.get('words', []))


def failure_segment(turn, reason):
    """A failed attempt is distinct from successfully detected unintelligible audio."""
    start, end = float(turn['start']), float(turn['end'])
    text = '[' + '-' * marker_asterisk_count(max(0, end - start)) + ']'
    word = dict(start=start, end=end, word=text, text=text,
                speaker=turn.get('speaker', 'UNKNOWN'), confidence=0.0,
                is_backfill=True, is_backfill_failure=True, backfill_error=reason)
    return dict(start=start, end=end, text=text, speaker=word['speaker'],
                words=[word], is_backfill=True, is_backfill_failure=True)


def summarize(outcome, attempted, error_occurred=False):
    """The backfill report fields for an outcome.

    Gaps rejected for low confidence keep their [-] markers but do not make the
    backfill incomplete.
    """
    failed = outcome.get('failed_turns', [])
    unconfident = sum(is_low_confidence(t.get('backfill_error')) for t in failed)
    recovered = outcome.get('segments', [])
    incomplete = (error_occurred or outcome.get('error') or len(failed) > unconfident
                  or outcome.get('enrichment_errors'))
    return dict(
        status=('partial' if recovered else 'failed') if incomplete else
               'completed' if recovered else 'no_words_recovered',
        attempted_turns=attempted,
        failed_turns=len(failed) - unconfident,
        low_confidence_turns=unconfident,
        recovered_words=outcome.get('word_count', 0))


class BackfillResults:
    """Keep completed turns when a worker fails before its final reply."""

    def __init__(self, turns, cached=None):
        self.turns = turns
        self.errors = {}
        cached = cached or {}
        self.segments = {turn_key(t): cached[turn_key(t)] for t in turns
                         if turn_key(t) in cached and not cached[turn_key(t)].get('is_backfill_failure')}

    def record(self, key, segment, error=None):
        if segment is not None:
            self.segments[key] = segment
        if error:
            self.errors[key] = error

    def outcome(self, error=None, enrichment_errors=None):
        recovered, failed, segments = [], [], []
        for turn in self.turns:
            segment = self.segments.get(turn_key(turn))
            if segment is None:
                reason = self.errors.get(turn_key(turn))
                failed.append(dict(turn, backfill_error=reason) if reason else turn)
            else:
                segments.append(segment)
                recovered.append(dict(turn, word_count=word_count(segment)))
        result = dict(segments=segments, recovered_turns=recovered,
                      failed_turns=failed, word_count=sum(map(word_count, segments)),
                      placeholders=sum(bool(s.get('is_placeholder')) for s in segments))
        if error:
            result['error'] = error
        if enrichment_errors:
            result['enrichment_errors'] = enrichment_errors
        return result
