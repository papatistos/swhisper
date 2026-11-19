"""Round-trip test: is the output of each pipeline stage still valid input
for the next?

remove_nonoverlapping_markers -> add_word_level_silence_markers -> TXTFormatter.render

Each of these three is unit-tested on its own elsewhere; this checks the
seams between them instead of each function in isolation, using synthetic
segments -- no real audio or models involved.
"""

from pyannote.core import Annotation, Segment

from diarize.config import DiarizationConfig
from diarize.utils import SilenceMarkerProcessor, WordProcessor
from transcript_render.formatters import TXTFormatter


def test_pipeline_round_trip_preserves_every_real_word():
    diarization = Annotation()
    diarization[Segment(0.0, 5.0)] = "SPEAKER_00"
    diarization[Segment(8.0, 12.0)] = "SPEAKER_01"
    # Gap 5.0-8.0: no speaker segment covers this range.

    segments = [
        {
            'start': 0.0, 'end': 6.0, 'speaker': 'SPEAKER_00',
            'words': [
                {'start': 0.0, 'end': 1.0, 'word': 'Hello'},
                {'start': 1.1, 'end': 1.6, 'word': '[*]'},   # overlaps SPEAKER_00: kept as a marker
                {'start': 1.7, 'end': 2.5, 'word': 'world'},
                {'start': 5.5, 'end': 6.0, 'word': '[*]'},   # in the 5.0-8.0 gap: no overlap, removed
            ],
        },
        {
            'start': 8.0, 'end': 11.0, 'speaker': 'SPEAKER_01',
            'words': [
                {'start': 8.0, 'end': 8.5, 'word': 'Goodbye'},
                # 2.0s internal gap, above the default 0.2s silence-marker threshold
                {'start': 10.5, 'end': 11.0, 'word': 'friend'},
            ],
        },
    ]

    stats = WordProcessor.remove_nonoverlapping_markers(segments, diarization)
    assert stats == {'total_markers': 2, 'removed': 1, 'kept_as_markers': 1}

    enhanced = SilenceMarkerProcessor.add_word_level_silence_markers(segments)

    config = DiarizationConfig(silence_gap_linebreak_threshold=1.0, output_preamble=None)
    text = TXTFormatter().render(enhanced, config=config, stats=False)

    # Every real word from the input survives the whole chain exactly once.
    for real_word in ('Hello', 'world', 'Goodbye', 'friend'):
        assert text.count(real_word) == 1, f"'{real_word}' should appear exactly once in:\n{text}"

    # The marker that overlapped a speaker segment survives; render() must
    # still be able to consume a word dict remove_nonoverlapping_markers left
    # behind (no 'is_silence_marker' key at all, not even False).
    assert '[*]' in text

    # The word-level gap inside SPEAKER_01's segment surfaces as a silence marker.
    assert '(2.0)' in text
