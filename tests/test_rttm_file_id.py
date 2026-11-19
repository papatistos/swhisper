"""Tests that RTTMFormatter writes a whitespace-free file id that standard parsers can read."""

from pyannote.database.util import load_rttm

from diarize.output import RTTMFormatter

AUDIO_BASENAME = "2762_recording 11 august LC.wav"
EXPECTED_URI = "2762_recording_11_august_LC"


def _segments():
    return [
        {'start': 0.15, 'end': 1.63, 'speaker': 'SPEAKER_00', 'speaker_confidence': 1.0,
         'words': [{'start': 0.15, 'end': 0.6, 'speaker': 'SPEAKER_00', 'confidence': 0.9},
                   {'start': 0.7, 'end': 1.63, 'speaker': 'SPEAKER_00', 'confidence': 0.8}]},
        {'start': 2.0, 'end': 2.5, 'speaker': 'SILENCE', 'words': []},
        {'start': 3.0, 'end': 4.25, 'speaker': 'SPEAKER_01', 'speaker_confidence': 0.7,
         'words': [{'start': 3.0, 'end': 4.25, 'speaker': 'SPEAKER_01', 'confidence': 0.7}]},
    ]


def _tracks(annotation):
    return [(round(seg.start, 3), round(seg.end, 3), label)
            for seg, _, label in annotation.itertracks(yield_label=True)]


def test_rttm_with_spaces_in_name_loads_with_pyannote(tmp_path):
    path = tmp_path / "out.rttm"
    count = RTTMFormatter().format(_segments(), str(path), audio_basename=AUDIO_BASENAME)

    loaded = load_rttm(str(path))

    assert count == 2
    assert list(loaded) == [EXPECTED_URI]
    assert _tracks(loaded[EXPECTED_URI]) == [(0.15, 1.63, 'SPEAKER_00'), (3.0, 4.25, 'SPEAKER_01')]


def test_detailed_rttm_with_spaces_in_name_loads_with_pyannote(tmp_path):
    path = tmp_path / "out_detailed.rttm"
    count = RTTMFormatter().format_detailed(_segments(), str(path), audio_basename=AUDIO_BASENAME)

    loaded = load_rttm(str(path))

    assert count == 3
    assert list(loaded) == [EXPECTED_URI]
    assert _tracks(loaded[EXPECTED_URI]) == [
        (0.15, 0.6, 'SPEAKER_00'), (0.7, 1.63, 'SPEAKER_00'), (3.0, 4.25, 'SPEAKER_01')]


def test_rttm_without_spaces_keeps_name(tmp_path):
    path = tmp_path / "out.rttm"
    RTTMFormatter().format(_segments(), str(path), audio_basename="interview_01.wav")

    assert list(load_rttm(str(path))) == ["interview_01"]
