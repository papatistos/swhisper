"""Decoding policy for short backfill snippets, separate from full transcription."""
import math


def _validate_scores(values, settings):
    checks = [('avg_logprob', settings.get('logprob_threshold', -1.0), 'below'),
              ('compression_ratio', settings.get('compression_ratio_threshold', 2.4), 'above')]
    for field, threshold, direction in checks:
        if threshold is None:
            continue
        value = values.get(field)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f'Backfill rejected: missing or invalid {field}')
        value = float(value)
        if (direction == 'below' and value < threshold) or (direction == 'above' and value > threshold):
            raise ValueError(f'Backfill rejected: {field} {value:.3f} {direction} threshold {threshold:g}')


def transcribe_with_policy(transcribe, model, audio, settings, try_harder=False):
    """Check the decoding result before Whisper can drop empty/boundary tokens."""
    if try_harder or not callable(getattr(model, 'decode', None)):
        return transcribe(model, audio, **settings)
    original_decode = model.decode
    had_override = 'decode' in vars(model)
    previous_override = vars(model).get('decode')

    def checked_decode(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        candidates = result if isinstance(result, list) else [result]
        for candidate in candidates:
            values = {field: getattr(candidate, field, None) for field in
                      ('avg_logprob', 'compression_ratio', 'no_speech_prob')}
            no_speech = settings.get('no_speech_threshold', .6)
            logprob = settings.get('logprob_threshold', -1.0)
            # Match Whisper's explicit silence exception. This is a completed
            # no-speech result, not a failed decoding attempt.
            silence = (no_speech is not None and logprob is not None
                       and values['no_speech_prob'] is not None
                       and values['avg_logprob'] is not None
                       and values['no_speech_prob'] > no_speech
                       and values['avg_logprob'] < logprob)
            if not silence:
                _validate_scores(values, settings)
        return result

    model.decode = checked_decode
    try:
        return transcribe(model, audio, **settings)
    finally:
        if had_override:
            model.decode = previous_override
        else:
            del model.decode


def decoding_settings(settings, try_harder=False):
    result = dict(settings)
    if not try_harder:
        result['temperature'] = [0.0]
    return result


def validate_result(result, settings, start, end):
    """Reject failed Whisper quality checks instead of accepting its last fallback.

    The timestamps must already refer to the original recording. Only segments
    containing a word that would be kept in this gap affect its acceptance.
    Empty/no-speech results remain successful unintelligible-audio placeholders.
    """
    for segment in result.get('segments', []):
        if not any(w.get('end', start) > start and w.get('start', end) < end
                   for w in segment.get('words', [])):
            continue
        _validate_scores(segment, settings)
