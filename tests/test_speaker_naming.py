"""Interview names use final words, independently of the diarization backend."""
from types import SimpleNamespace

import pytest

from speaker_naming import interview_names


def config(**changes):
    return SimpleNamespace(**dict(dict(recording_type='interview', interviewer_name='INTERVIEWER',
                                      interviewee_name='INTERVIEWEE'), **changes))


def segment(label, text, **changes):
    return dict(dict(speaker=label, text=text,
                     words=[{'word': word} for word in text.split()]), **changes)


def test_ranks_total_words_not_turn_count_or_duration():
    segments = [segment('A', 'Question?', end=500), segment('B', 'A longer answer here', end=1),
                segment('C', 'Hi'), segment('A', 'Why?'), segment('A', 'Really?')]
    assert interview_names(segments, config(interviewee_name='Anna Svensson', interviewer_name='Chris')) == {
        'B': 'Anna Svensson', 'A': 'Chris'}
    assert segments[0]['speaker'] == 'A'


@pytest.mark.parametrize('recording_type', ['', 'meeting'])
def test_opt_in_only(recording_type):
    assert interview_names([segment('A', 'Hello')], config(recording_type=recording_type)) == {}


def test_markers_unknowns_and_silence_do_not_win_roles():
    segments = [segment('A', '[*] [S02**] [---] [SILENCE] (1.2) ...'),
                segment('Unknown', 'many many many many words'), segment('SILENCE', 'silence'),
                segment('B', 'One two'), segment('C', 'Three')]
    assert interview_names(segments, config()) == {'B': 'INTERVIEWEE', 'C': 'INTERVIEWER'}


def test_ties_and_text_only_segments():
    assert interview_names([segment('B', 'hello', words=[]), segment('A', 'world', words=[])], config()) == {
        'B': 'INTERVIEWEE', 'A': 'INTERVIEWER'}
    assert interview_names([], config()) == {}
    assert interview_names([segment('A', '')], config()) == {}
    assert interview_names([segment('A', 'hello')], config()) == {'A': 'INTERVIEWEE'}


def test_text_tokens_and_flagged_silence():
    segments = [segment('A', '', words=[{'text': 'one two three'},
                                      {'text': 'pause', 'is_silence_marker': True}]),
                segment('B', 'one two three four')]
    assert interview_names(segments, config()) == {'B': 'INTERVIEWEE', 'A': 'INTERVIEWER'}


def test_duplicate_role_names_rejected():
    with pytest.raises(ValueError, match='share the name'):
        interview_names([segment('A', 'one'), segment('B', 'two')],
                        config(interviewer_name='Same', interviewee_name='Same'))


def test_diarization_first_exports_keep_ids_and_supply_editable_names(tmp_path):
    import json
    from diarize.config import DiarizationConfig
    from transcribe5 import TranscriptionAppV5
    from webui_speakers import names_for, transcript_key

    app = TranscriptionAppV5.__new__(TranscriptionAppV5)
    app.diarization_config = DiarizationConfig(recording_type='interview', output_preamble='',
                                               interviewee_name='Anna Svensson', interviewer_name='INTERVIEWER')
    segments = [dict(speaker='SPEAKER_00', start=0, end=1, text='Why?',
                     words=[dict(word='Why?', start=0, end=1)]),
                dict(speaker='SPEAKER_01', start=1, end=3, text='My answer',
                     words=[dict(word='My', start=1, end=2), dict(word='answer', start=2, end=3)])]
    for item in segments:
        for word in item['words']:
            word['speaker'] = item['speaker']
    app._generate_output_formats({'segments': segments}, str(tmp_path / 'a.wav'), 'a', str(tmp_path))
    exported = tmp_path / 'transcripts'
    path = next((exported / 'json').glob('*_diarized.json'))
    raw = json.loads(path.read_text())
    assert raw['segments'] == segments
    assert names_for(str(tmp_path), transcript_key(str(tmp_path), str(path))) == {
        'SPEAKER_01': 'Anna Svensson', 'SPEAKER_00': 'INTERVIEWER'}
    for kind in ('txt', 'rtf', 'tsv', 'vtt'):
        text = next((exported / kind).glob('*')).read_text()
        assert 'Anna Svensson' in text
        assert 'INTERVIEWER' in text
    for path in (exported / 'rttm').glob('*.rttm'):
        assert 'SPEAKER_01' in path.read_text()


def test_env_defaults_and_scribe_support(monkeypatch):
    from diarize.config import DiarizationConfig
    from scribe_processing import capability

    for key in ('RECORDING_TYPE', 'INTERVIEWEE_NAME', 'INTERVIEWER_NAME'):
        monkeypatch.delenv(key, raising=False)
        assert capability(key)['supported']
    defaults = DiarizationConfig()
    assert (defaults.recording_type, defaults.interviewer_name, defaults.interviewee_name) == (
        '', 'INTERVIEWER', 'INTERVIEWEE')
    monkeypatch.setenv('RECORDING_TYPE', 'interview')
    monkeypatch.setenv('INTERVIEWEE_NAME', 'Anna')
    assert interview_names([segment('A', 'Hello')], DiarizationConfig()) == {'A': 'Anna'}


def test_per_file_names_reach_all_pipeline_configs_without_leaking(tmp_path):
    from diarize.config import DiarizationConfig
    from sidecar import apply_sidecar_to_diarization_config, sidecar_overrides
    from scribe_processing import configure, effective_settings

    global_config = DiarizationConfig(recording_type='', interviewer_name='Global researcher',
                                      interviewee_name='Global participant')
    (tmp_path / 'a.swhisper').write_text('RECORDING_TYPE=interview\nINTERVIEWER_NAME=Chris\nINTERVIEWEE_NAME=Anna\n')
    segments = [segment('A', 'Why'), segment('B', 'My answer')]
    expected = {'B': 'Anna', 'A': 'Chris'}
    local, _ = apply_sidecar_to_diarization_config(global_config, str(tmp_path), 'a')
    assert interview_names(segments, local) == expected
    with sidecar_overrides(str(tmp_path), 'a', diarization_config=global_config):
        assert interview_names(segments, global_config) == expected
    assert interview_names(segments, global_config) == {}
    scribe = configure(global_config, tmp_path / 'a.wav', tmp_path / 'scribe-run')
    assert interview_names(segments, scribe) == expected
    assert effective_settings(scribe)['interviewee_name'] == 'Anna'
    (tmp_path / 'a.swhisper').write_text('RECORDING_TYPE=\n')
    global_config.recording_type = 'interview'
    local, _ = apply_sidecar_to_diarization_config(global_config, str(tmp_path), 'a')
    assert interview_names(segments, local) == {}
    assert global_config.recording_type == 'interview'
