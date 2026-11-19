"""Tests for WordProcessor.remove_nonoverlapping_markers: disfluency markers
([*], [**], ...) are removed when they don't overlap any pyannote speaker
segment, and kept when they do.
"""

from pyannote.core import Annotation, Segment

from diarize.utils import WordProcessor


def _mock_diarization():
    annotation = Annotation()
    annotation[Segment(0.0, 5.0)] = "SPEAKER_00"
    annotation[Segment(5.5, 10.0)] = "SPEAKER_01"
    annotation[Segment(10.5, 15.0)] = "SPEAKER_00"
    # Gap from 15.0 to 17.0: no speaker segment covers this range.
    annotation[Segment(17.0, 20.0)] = "SPEAKER_01"
    return annotation


def _segments():
    return [
        {
            'start': 0.0, 'end': 5.2, 'text': 'Hello [*] world', 'speaker': 'SPEAKER_00',
            'words': [
                {'start': 0.0, 'end': 0.5, 'word': 'Hello', 'speaker': 'SPEAKER_00'},
                {'start': 0.6, 'end': 1.1, 'word': '[*]', 'speaker': 'SPEAKER_00'},  # overlaps SPEAKER_00
                {'start': 1.2, 'end': 1.8, 'word': 'world', 'speaker': 'SPEAKER_00'},
            ],
        },
        {
            'start': 15.0, 'end': 17.0, 'text': '[*]', 'speaker': 'UNKNOWN',
            'words': [
                {'start': 15.0, 'end': 17.0, 'word': '[*]', 'speaker': 'UNKNOWN'},  # in the 15.0-17.0 gap
            ],
        },
        {
            'start': 17.0, 'end': 20.0, 'text': 'Final [*] statement', 'speaker': 'SPEAKER_01',
            'words': [
                {'start': 17.0, 'end': 17.5, 'word': 'Final', 'speaker': 'SPEAKER_01'},
                {'start': 17.6, 'end': 18.0, 'word': '[*]', 'speaker': 'SPEAKER_01'},  # overlaps SPEAKER_01
                {'start': 18.1, 'end': 18.8, 'word': 'statement', 'speaker': 'SPEAKER_01'},
            ],
        },
    ]


def test_marker_overlapping_a_speaker_segment_is_kept():
    segments = _segments()
    WordProcessor.remove_nonoverlapping_markers(segments, _mock_diarization())

    words = segments[0]['words']
    assert any(w['word'] == '[*]' for w in words), "marker overlapping SPEAKER_00 should be kept"

    words = segments[2]['words']
    assert any(w['word'] == '[*]' for w in words), "marker overlapping SPEAKER_01 should be kept"


def test_marker_in_a_speaker_gap_is_removed():
    segments = _segments()
    WordProcessor.remove_nonoverlapping_markers(segments, _mock_diarization())

    assert segments[1]['words'] == [], "marker in the 15.0-17.0 gap (no speaker overlap) should be removed"


def test_removal_statistics():
    segments = _segments()
    stats = WordProcessor.remove_nonoverlapping_markers(segments, _mock_diarization())

    assert stats == {'total_markers': 3, 'removed': 1, 'kept_as_markers': 2}
