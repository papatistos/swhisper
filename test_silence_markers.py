"""Tests for silence-marker blank-line behaviour.

WordProcessor.create_paragraph_text_from_words puts blank lines around a
silence marker that's at or above gap_threshold, but trims a leading/trailing
blank line at the very edge of a segment's own text (paragraph-level blank
lines around a *standalone* silence segment are added separately, by
TXTFormatter, when it stitches segments into paragraphs -- so a segment must
not also add its own edge blanks or they'd double up).
"""

from diarize.config import DiarizationConfig
from diarize.utils import WordProcessor
from transcript_render.formatters import TXTFormatter

GAP_THRESHOLD = 1.0


def test_long_silence_mid_segment_gets_surrounding_blank_lines():
    segment = {
        'start': 0.0, 'end': 10.0, 'speaker': 'SPEAKER_04',
        'words': [
            {'start': 0.0, 'end': 0.5, 'word': 'Ja', 'speaker': 'SPEAKER_04', 'is_silence_marker': False},
            {'start': 0.5, 'end': 5.6, 'word': '(5.1)', 'speaker': 'SILENCE', 'is_silence_marker': True},
            {'start': 5.6, 'end': 6.0, 'word': 'ja', 'speaker': 'SPEAKER_04', 'is_silence_marker': False},
        ],
    }

    text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=GAP_THRESHOLD)

    assert text == "Ja\n\n(5.1)\n\nja"


def test_long_silence_at_segment_start_does_not_leave_a_leading_blank_line():
    segment = {
        'start': 0.0, 'end': 10.0, 'speaker': 'SILENCE',
        'words': [
            {'start': 0.0, 'end': 5.1, 'word': '(5.1)', 'speaker': 'SILENCE', 'is_silence_marker': True},
            {'start': 5.1, 'end': 5.5, 'word': 'ja', 'speaker': 'SPEAKER_04', 'is_silence_marker': False},
        ],
    }

    text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=GAP_THRESHOLD)

    assert not text.startswith('\n'), "leading blank line should be trimmed, not left in the segment's own text"
    assert text == "(5.1)\n\nja"


def test_txt_formatter_surrounds_standalone_silence_segment_with_blank_lines():
    segments = [
        {
            'start': 361.434, 'end': 364.844, 'speaker': 'SPEAKER_04',
            'words': [
                {'text': 'Precis', 'start': 360.0, 'end': 361.0, 'speaker': 'SPEAKER_04'},
                {'text': 'Ja', 'start': 364.664, 'end': 364.844, 'speaker': 'SPEAKER_04'},
            ],
        },
        {
            'start': 364.844, 'end': 369.965, 'speaker': 'SILENCE', 'is_silence_marker': True,
            'words': [
                {'word': '(5.1)', 'start': 364.844, 'end': 369.965, 'speaker': 'SILENCE', 'is_silence_marker': True},
            ],
        },
        {
            'start': 370.834, 'end': 372.654, 'speaker': 'SPEAKER_04',
            'words': [
                {'text': 'Ida', 'start': 370.834, 'end': 372.054, 'speaker': 'SPEAKER_04'},
            ],
        },
    ]
    config = DiarizationConfig(silence_gap_linebreak_threshold=GAP_THRESHOLD, output_preamble=None)

    text = TXTFormatter().render(segments, config=config, stats=False)

    assert "\n\n(5.1)\n\n" in text
