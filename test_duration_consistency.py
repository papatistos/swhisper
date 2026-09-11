"""Duration-sum invariant for SilenceMarkerProcessor.add_word_level_silence_markers.

The pipeline's contract is that a transcript's words plus its inserted
silence markers tile the whole recording with no unaccounted time: every word
duration and every silence-marker duration should sum back up to
(last word end - first word start). This checks that the invariant holds --
and pins the one case where it currently does not: gaps shorter than
min_silence_duration get no marker at all, so their time silently
disappears from the reconstructed total.
"""

import pytest

from diarize.utils import SilenceMarkerProcessor


def _total_word_duration(segments):
    total = 0.0
    for segment in segments:
        for word in segment.get('words', []):
            total += word['end'] - word['start']
    return total


def _timeline_span(segments):
    first_start = segments[0]['words'][0]['start']
    last_end = segments[-1]['words'][-1]['end']
    return last_end - first_start


def test_gaps_above_threshold_are_fully_accounted_for():
    segments = [
        {
            'start': 0.0, 'end': 2.0,
            'words': [
                {'start': 0.0, 'end': 1.0, 'word': 'Hello'},
                {'start': 1.0, 'end': 2.0, 'word': 'world'},
            ],
        },
        # 2.0s gap to the next segment (cross-segment gap, well above the 0.2s default threshold)
        {
            'start': 4.0, 'end': 7.0,
            'words': [
                {'start': 4.0, 'end': 5.0, 'word': 'foo'},
                # 1.5s gap within this segment (also above threshold)
                {'start': 6.5, 'end': 7.0, 'word': 'bar'},
            ],
        },
    ]

    enhanced = SilenceMarkerProcessor.add_word_level_silence_markers(segments)

    assert _total_word_duration(enhanced) == pytest.approx(_timeline_span(segments))


def test_subthreshold_gaps_are_dropped_not_counted():
    # A gap of 0.1s is below the default min_silence_duration of 0.2s: no
    # silence marker gets inserted for it, so it's simply missing from the
    # reconstructed timeline. This test documents that behavior so a future
    # change to the threshold is a deliberate, visible change here.
    segments = [
        {
            'start': 0.0, 'end': 2.0,
            'words': [
                {'start': 0.0, 'end': 1.0, 'word': 'a'},
                {'start': 1.1, 'end': 2.0, 'word': 'b'},
            ],
        },
    ]

    enhanced = SilenceMarkerProcessor.add_word_level_silence_markers(segments)

    assert len(enhanced[0]['words']) == 2, "sub-threshold gap should not get a silence marker"
    dropped = _timeline_span(segments) - _total_word_duration(enhanced)
    assert dropped == pytest.approx(0.1)
