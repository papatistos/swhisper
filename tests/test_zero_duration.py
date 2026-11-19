"""Tests for transcribe.fix_zero_duration_words."""

import pytest

from transcribe import fix_zero_duration_words


def _durations(words):
    return [w['end'] - w['start'] for w in words]


def test_zero_duration_word_gets_minimum_duration():
    result = {
        'segments': [{
            'start': 375.0,
            'end': 376.0,
            'text': 'Hello Ja there',
            'words': [
                {'text': 'Hello', 'start': 375.0, 'end': 375.2},
                {'text': 'Ja', 'start': 375.274, 'end': 375.274},
                {'text': 'there', 'start': 375.5, 'end': 376.0},
            ],
        }],
    }

    fixed = fix_zero_duration_words(result, min_duration=0.02)
    words = fixed['segments'][0]['words']

    assert words[1]['end'] - words[1]['start'] >= 0.019
    assert words[1]['start'] >= words[0]['end']
    assert words[2]['start'] >= words[1]['end']


@pytest.mark.parametrize(
    "name, words",
    [
        (
            "zero-duration at beginning",
            [
                {'text': 'Ja', 'start': 0.0, 'end': 0.0},
                {'text': 'hello', 'start': 0.3, 'end': 0.5},
                {'text': 'world', 'start': 0.7, 'end': 1.0},
            ],
        ),
        (
            "zero-duration at end",
            [
                {'text': 'hello', 'start': 0.0, 'end': 0.3},
                {'text': 'world', 'start': 0.5, 'end': 0.8},
                {'text': 'Ja', 'start': 0.9, 'end': 0.9},
            ],
        ),
        (
            "multiple zero-duration words",
            [
                {'text': 'hello', 'start': 0.0, 'end': 0.3},
                {'text': 'Ja', 'start': 0.4, 'end': 0.4},
                {'text': 'world', 'start': 0.6, 'end': 1.0},
                {'text': 'Nej', 'start': 1.2, 'end': 1.2},
                {'text': 'okay', 'start': 1.5, 'end': 2.0},
            ],
        ),
        (
            "zero-duration touching the next word",
            [
                {'text': 'hello', 'start': 0.0, 'end': 0.3},
                {'text': 'Ja', 'start': 0.4, 'end': 0.4},
                {'text': 'world', 'start': 0.4, 'end': 1.0},
            ],
        ),
    ],
)
def test_zero_duration_edge_cases(name, words):
    result = {'segments': [{'start': words[0]['start'], 'end': words[-1]['end'], 'text': name, 'words': words}]}

    fixed = fix_zero_duration_words(result, min_duration=0.02)
    fixed_words = fixed['segments'][0]['words']

    for i, word in enumerate(fixed_words):
        duration = word['end'] - word['start']
        assert duration >= 0.02 - 0.001, f"{name}: word {i} still under min_duration"
        if i > 0:
            gap = word['start'] - fixed_words[i - 1]['end']
            assert gap >= -0.001, f"{name}: word {i} overlaps word {i - 1}"
