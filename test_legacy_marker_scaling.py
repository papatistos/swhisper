"""Tests for WordProcessor.rescale_legacy_disfluency_markers: transcription
JSONs written before 2025-10-21 contain only single-asterisk [*] markers, and
diarize.py rescales them on load from each marker's own start/end times.
"""

from diarize.utils import WordProcessor


def _legacy_result():
    # Shape copied from a June 2025 whisper-json-output file: words carry only
    # 'text', and the segment text leaves the markers out.
    return {
        'segments': [
            {
                'start': 0.8, 'end': 3.72, 'text': ' Så missar jag inte viktiga saker.',
                'words': [
                    {'text': '[*]', 'start': 0.8, 'end': 1.26, 'confidence': 0},
                    {'text': 'Så', 'start': 1.84, 'end': 2.1, 'confidence': 0.086},
                    {'text': 'viktiga', 'start': 2.82, 'end': 3.42, 'confidence': 0.0},
                    {'text': '[*]', 'start': 3.42, 'end': 3.62, 'confidence': 0},
                    {'text': 'saker.', 'start': 3.62, 'end': 3.72, 'confidence': 0.0},
                ],
            },
        ],
    }


def test_rescales_single_asterisk_markers_by_duration():
    result = _legacy_result()

    rescaled = WordProcessor.rescale_legacy_disfluency_markers(result)

    words = result['segments'][0]['words']
    assert words[0]['word'] == words[0]['text'] == '[*****]'
    assert words[3]['word'] == words[3]['text'] == '[**]'
    assert result['segments'][0]['text'] == '[*****] Så viktiga [**] saker.'
    assert rescaled == 2


def test_leaves_already_scaled_markers_unchanged():
    result = _legacy_result()
    WordProcessor.rescale_legacy_disfluency_markers(result)

    rescaled = WordProcessor.rescale_legacy_disfluency_markers(result)

    assert [w['text'] for w in result['segments'][0]['words']] == [
        '[*****]', 'Så', 'viktiga', '[**]', 'saker.',
    ]
    assert rescaled == 0
