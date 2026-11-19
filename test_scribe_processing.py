"""Scribe recovery integration with simulated model outputs, no network/models."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from pyannote.core import Annotation, Segment

import scribe_browser_auth
import transcribe_scribe
from diarize.pipeline import DiarizationPipeline
from diarize.utils import BackfillCache, BackfillTranscriber
from scribe_api import ScribeClient
from scribe_processing import capability


class _BrowserProvider:
    """Stands in for the BrowserOS Neo sign-in so no browser is opened."""

    def __init__(self, interactive):
        pass

    def get_token(self, force=False):
        return 'token'

    def close(self):
        pass


def _browser_auth():
    return patch.object(scribe_browser_auth, 'BrowserTokenProvider', _BrowserProvider)


def test_capabilities_distinguish_remote_and_recovery():
    assert capability('SWHISPER_MIN_SPEAKERS')['scope'] == 'Local processing after Scribe'
    assert capability('SWHISPER_INITIAL_PROMPT')['scope'] == 'Local backfill only'
    assert capability('SWHISPER_LANGUAGE')['scope'] == 'Scribe and local backfill'
    assert capability('SWHISPER_BACKFILL_ENABLED')['supported']
    assert capability('SWHISPER_INITIAL_PROMPT')['supported']
    assert 'backfill only' in capability('SWHISPER_INITIAL_PROMPT')['description']
    assert not capability('SWHISPER_VAD')['supported']
    assert not capability('SWHISPER_USE_PRECISION')['supported']
    assert not capability('SWHISPER_WHISPER_TIMESTAMP_OFFSET')['supported']
    assert capability('SWHISPER_SCRIBE_TIMESTAMP_OFFSET')['supported']


def test_cache_invalidates_changed_recovery_settings(tmp_path):
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    def cache(settings, sensevoice=False):
        return BackfillCache(str(tmp_path / 'cache'), str(audio), 'model', 'cpu', 0.5,
                             sensevoice_enabled=sensevoice, processing_settings=settings)
    cache({'prompt': 'one', 'ignore': []}).save('audio', {'turn': {'text': 'hej'}})
    assert cache({'prompt': 'one', 'ignore': []}).load('audio')
    assert not cache({'prompt': 'two', 'ignore': []}).load('audio')
    assert not cache({'prompt': 'one', 'ignore': ['hej']}).load('audio')
    assert not cache({'prompt': 'one', 'ignore': []}, True).load('audio')


def test_saved_scribe_result_local_recovery_exports_and_resume(tmp_path):
    source = tmp_path / 'interview.wav'
    source.write_bytes(b'original')
    sidecar = source.with_suffix('.swhisper')
    sidecar.write_text('SWHISPER_BACKFILL_ENABLED=true\nSWHISPER_SMOOTHING_ENABLED=false\n'
                       'SWHISPER_BACKFILL_ENABLE_SENSEVOICE=true\nSWHISPER_INITIAL_PROMPT=Names\n')
    work = tmp_path / 'run'
    transcript = {'full_transcription': 'Hej slut', 'segments': [
        {'start': 0, 'end': 3, 'text': 'Hej slut'}]}
    words = {'version': 1, 'words': [
        {'t': 'Hej', 's': 0.1, 'e': 0.5}, {'t': 'slut', 's': 2.5, 'e': 2.9}]}
    annotation = Annotation()
    annotation[Segment(0, .7)] = 'SPEAKER_00'
    annotation[Segment(1, 2)] = 'SPEAKER_01'
    annotation[Segment(2.3, 3)] = 'SPEAKER_00'
    recovered = {'segments': [{'start': 1.2, 'end': 1.6, 'speaker': 'SPEAKER_01',
                              'text': 'ja', 'words': [{'start': 1.2, 'end': 1.6,
                              'text': 'ja', 'speaker': 'SPEAKER_01', 'emotion': 'happy'}]}],
                 'word_count': 1, 'failed_turns': []}
    def remote(self, wav, state, state_path, language):
        state['job_id'] = 'simulated'
        return copy.deepcopy(transcript), copy.deepcopy(words)
    argv = [str(source), '--work-dir', str(work), '--auth', 'browseros']
    with patch.dict('os.environ', {'SUNET_SCRIBE_ENCRYPT_PWD': 'test-password'}), _browser_auth(), \
            patch.object(transcribe_scribe, 'prepare_audio', side_effect=lambda src, dst: dst.write_bytes(b'staged')), \
            patch.object(ScribeClient, 'transcribe', autospec=True, side_effect=remote) as service, \
            patch('diarize4.ensure_pyannote_v4'), \
            patch.object(DiarizationPipeline, 'diarize', return_value=annotation), \
            patch.object(BackfillTranscriber, 'transcribe_turns_subprocess', return_value=recovered) as recovery:
        assert transcribe_scribe.main(argv) == 0
        recovery.assert_called_once()
        normalized = (work / 'whisper-json-output/audio.json').read_bytes()
        outputs = list((work / 'transcripts/json').glob('*_diarized.json'))
        assert len(outputs) == 1
        enriched = json.loads(outputs[0].read_text())
        actual = [w for s in enriched['segments'] for w in s['words'] if not w.get('is_silence_marker')]
        assert [w['text'] for w in actual] == ['Hej', 'ja', 'slut']
        assert actual[1]['transcription_source'] == 'local-whisper-backfill'
        assert actual[1]['emotion'] == 'happy'
        assert actual[0]['transcription_source'] == 'sunet-scribe'
        for segment in enriched['segments']:
            for word in segment['words']:
                if word.get('is_silence_marker'):
                    assert word['end'] <= 1.2 or word['start'] >= 1.6
        # A settings-only rerun must not call SUNET or recovery after disabling it.
        sidecar.write_text('SWHISPER_BACKFILL_ENABLED=false\n')
        assert transcribe_scribe.main(argv) == 0
        assert len(list((work / 'transcripts/json').glob('*_diarized.json'))) == 2
        assert service.call_count == 1
        assert recovery.call_count == 1
        assert (work / 'whisper-json-output/audio.json').read_bytes() == normalized
        records = [json.loads(p.read_text()) for p in (work / 'local-runs').glob('*.json')]
        assert len(records) == 2
        assert all(r['status'] == 'completed' for r in records)
        assert {r['settings']['backfill_missing_turns'] for r in records} == {True, False}
        assert source.read_bytes() == b'original'
        # Recovery failure must be visible while preserving usable remote words.
        sidecar.write_text('SWHISPER_BACKFILL_ENABLED=true\n')
        recovery.return_value = {'segments': [], 'failed_turns': [{'start': 1, 'end': 2}],
                                 'word_count': 0, 'error': 'subprocess_failed'}
        assert transcribe_scribe.main(argv) == 0
        assert service.call_count == 1
        records = [json.loads(p.read_text()) for p in (work / 'local-runs').glob('*.json')]
        assert any(r.get('backfill_outcome', {}).get('status') == 'failed' for r in records)
        latest = max((work / 'transcripts/json').glob('*_diarized.json'), key=lambda p: p.stat().st_mtime_ns)
        failed_export = json.loads(latest.read_text())
        assert failed_export['metadata']['backfill_outcome']['status'] == 'failed'
        failure_words = [w for s in failed_export['segments'] for w in s['words'] if w.get('is_backfill_failure')]
        assert len(failure_words) == 1
        assert failure_words[0]['text'] == '[----------]'
        latest_txt = max((work / 'transcripts/txt').glob('*.txt'), key=lambda p: p.stat().st_mtime_ns)
        assert 'Backfill failures:' in latest_txt.read_text()
        assert '[----------]' in latest_txt.read_text()
        assert (work / 'whisper-json-output/audio.json').read_bytes() == normalized


def test_rerun_with_saved_transcript_needs_no_credentials_or_browser(tmp_path, monkeypatch):

    source = tmp_path / 'interview.wav'
    source.write_bytes(b'original')
    source.with_suffix('.swhisper').write_text('SWHISPER_BACKFILL_ENABLED=false\n')
    work = tmp_path / 'run'
    transcript = {'full_transcription': 'Hej', 'segments': [{'start': 0, 'end': 1, 'text': 'Hej'}]}
    words = {'version': 1, 'words': [{'t': 'Hej', 's': 0.1, 'e': 0.5}]}
    annotation = Annotation()
    annotation[Segment(0, 1)] = 'SPEAKER_00'
    def remote(self, wav, state, state_path, language):
        state['job_id'] = 'simulated'
        return copy.deepcopy(transcript), copy.deepcopy(words)
    staged = patch.object(transcribe_scribe, 'prepare_audio', side_effect=lambda src, dst: dst.write_bytes(b'staged'))
    with staged, patch('diarize4.ensure_pyannote_v4'), patch.object(DiarizationPipeline, 'diarize', return_value=annotation):
        with patch.dict('os.environ', {'SUNET_SCRIBE_ENCRYPT_PWD': 'test-password'}), _browser_auth(), \
                patch.object(ScribeClient, 'transcribe', autospec=True, side_effect=remote):
            assert transcribe_scribe.main([str(source), '--work-dir', str(work), '--auth', 'browseros']) == 0
        # Rerun as the web UI does: no password, no SUNET access, no browser.
        for name in ('SUNET_SCRIBE_ENCRYPT_PWD', 'SWHISPER_SCRIBE_PASSWORD'):
            monkeypatch.delenv(name, raising=False)
        with patch.object(ScribeClient, '__init__', side_effect=AssertionError('SUNET contacted')), \
                patch.object(scribe_browser_auth, 'BrowserTokenProvider', side_effect=AssertionError('BrowserOS opened')):
            assert transcribe_scribe.main([str(source), '--work-dir', str(work), '--auth', 'browseros']) == 0
            # The rerun reuses the diarization from the same model.
            assert len(list((work / 'transcripts/json').glob('*_diarized.json'))) == 1
            assert transcribe_scribe.main([str(source), '--work-dir', str(work), '--auth', 'browseros', '--rediarize']) == 0
    assert len(list((work / 'transcripts/json').glob('*_diarized.json'))) == 2


def test_configure_keeps_nemotron_and_forces_community_otherwise(tmp_path):
    from diarize.config import DiarizationConfig
    from scribe_processing import configure

    source = tmp_path / 'interview.wav'
    source.write_bytes(b'audio')
    for selected, expected in (('nvidia/Nemotron-3-Diarization', 'nvidia/Nemotron-3-Diarization'),
                               ('pyannote/speaker-diarization-3.1', 'pyannote/speaker-diarization-community-1')):
        config = DiarizationConfig()
        config.pipeline_model = selected
        assert configure(config, source, tmp_path / 'run').pipeline_model == expected


def test_scribe_run_diarizes_with_nemotron_when_selected(tmp_path):
    from diarize.config import DiarizationConfig
    from diarize.nemotron import NemotronDiarizer, NemotronOutput

    class NemotronConfig(DiarizationConfig):
        def __post_init__(self):
            super().__post_init__()
            self.pipeline_model = 'nvidia/Nemotron-3-Diarization'

    source = tmp_path / 'interview.wav'
    source.write_bytes(b'original')
    source.with_suffix('.swhisper').write_text('SWHISPER_BACKFILL_ENABLED=false\n')
    work = tmp_path / 'run'
    transcript = {'full_transcription': 'Hej slut', 'segments': [{'start': 0, 'end': 3, 'text': 'Hej slut'}]}
    words = {'version': 1, 'words': [{'t': 'Hej', 's': 0.1, 'e': 0.5}, {'t': 'slut', 's': 2.5, 'e': 2.9}]}
    annotation = Annotation()
    annotation[Segment(0, .7)] = 'SPEAKER_00'
    annotation[Segment(2.3, 3)] = 'SPEAKER_01'
    def remote(self, wav, state, state_path, language):
        state['job_id'] = 'simulated'
        return copy.deepcopy(transcript), copy.deepcopy(words)
    argv = [str(source), '--work-dir', str(work), '--auth', 'browseros']
    with patch.dict('os.environ', {'SUNET_SCRIBE_ENCRYPT_PWD': 'test-password'}), _browser_auth(), \
            patch('diarize.DiarizationConfig', NemotronConfig), \
            patch.object(transcribe_scribe, 'prepare_audio', side_effect=lambda src, dst: dst.write_bytes(b'staged')), \
            patch.object(ScribeClient, 'transcribe', autospec=True, side_effect=remote), \
            patch('diarize4.ensure_pyannote_v4'), \
            patch.object(NemotronDiarizer, 'diarize', return_value=NemotronOutput(annotation, annotation)) as nemotron:
        assert transcribe_scribe.main(argv) == 0
    nemotron.assert_called_once()
    records = [json.loads(p.read_text()) for p in (work / 'local-runs').glob('*.json')]
    assert [r['settings']['pipeline_model'] for r in records] == ['nvidia/Nemotron-3-Diarization']
    assert records[0]['status'] == 'completed'


def _scribe_fixture(tmp_path):
    source = tmp_path / 'interview.wav'
    source.write_bytes(b'original')
    source.with_suffix('.swhisper').write_text('SWHISPER_BACKFILL_ENABLED=false\n')
    annotation = Annotation()
    annotation[Segment(0, 1)] = 'SPEAKER_00'
    calls = []
    def remote(self, wav, state, state_path, language):
        calls.append(language)
        state['job_id'] = f'job-{len(calls)}'
        text = f'Hej{len(calls)}'
        return ({'full_transcription': text, 'segments': [{'start': 0, 'end': 1, 'text': text}]},
                {'version': 1, 'words': [{'t': text, 's': 0.1, 'e': 0.5}]})
    return source, annotation, calls, remote


def test_diarization_model_option_selects_nemotron_for_one_run(tmp_path):
    from diarize.nemotron import NemotronDiarizer, NemotronOutput

    source, annotation, _, remote = _scribe_fixture(tmp_path)
    work = tmp_path / 'run'
    argv = [str(source), '--work-dir', str(work), '--auth', 'browseros', '--diarization-model', 'nemotron']
    with patch.dict('os.environ', {'SUNET_SCRIBE_ENCRYPT_PWD': 'p'}), _browser_auth(), \
            patch.object(transcribe_scribe, 'prepare_audio', side_effect=lambda src, dst: dst.write_bytes(b'staged')), \
            patch.object(ScribeClient, 'transcribe', autospec=True, side_effect=remote), \
            patch('diarize4.ensure_pyannote_v4'), \
            patch.object(NemotronDiarizer, 'diarize', return_value=NemotronOutput(annotation, annotation)) as nemotron:
        assert transcribe_scribe.main(argv) == 0
    nemotron.assert_called_once()
    record = json.loads(next((work / 'local-runs').glob('*.json')).read_text())
    assert record['settings']['pipeline_model'] == 'nvidia/Nemotron-3-Diarization'


def test_retranscribe_uploads_again_and_keeps_the_previous_transcript(tmp_path):
    source, annotation, calls, remote = _scribe_fixture(tmp_path)
    work = tmp_path / 'run'
    argv = [str(source), '--work-dir', str(work), '--auth', 'browseros']
    with patch.dict('os.environ', {'SUNET_SCRIBE_ENCRYPT_PWD': 'p'}), _browser_auth(), \
            patch.object(transcribe_scribe, 'prepare_audio', side_effect=lambda src, dst: dst.write_bytes(b'staged')), \
            patch.object(ScribeClient, 'transcribe', autospec=True, side_effect=remote), \
            patch('diarize4.ensure_pyannote_v4'), \
            patch.object(DiarizationPipeline, 'diarize', return_value=annotation):
        assert transcribe_scribe.main(argv) == 0
        assert transcribe_scribe.main(argv) == 0
        assert len(calls) == 1  # a plain rerun reuses the saved transcript
        first = (work / 'whisper-json-output/audio.json').read_text()
        assert transcribe_scribe.main(argv + ['--retranscribe']) == 0
        assert len(calls) == 2
    superseded = list((work / 'superseded').iterdir())
    assert len(superseded) == 1
    assert (superseded[0] / 'audio.json').read_text() == first
    assert json.loads((superseded[0] / 'scribe-state.json').read_text())['job_id'] == 'job-1'
    state = json.loads((work / 'scribe-state.json').read_text())
    assert state['job_id'] == 'job-2'
    assert 'Hej2' in (work / 'whisper-json-output/audio.json').read_text()


def test_retranscribe_refuses_an_unfinished_job_and_job_adoption(tmp_path):
    import pytest

    source, _, _, _ = _scribe_fixture(tmp_path)
    work = tmp_path / 'run'
    work.mkdir()
    (work / 'audio.wav').write_bytes(b'staged')
    identity = {'source': str(source), 'sha256': transcribe_scribe.sha256(source), 'language': 'Swedish (verbatim)',
                'base_url': 'https://scribe.sunet.se', 'adapter_version': 1, 'audio_format': 'pcm_s16le/16000/mono'}
    (work / 'scribe-state.json').write_text(json.dumps({
        'identity': identity, 'job_id': 'pending-job', 'wav_sha256': transcribe_scribe.sha256(work / 'audio.wav')}))
    assert transcribe_scribe.main([str(source), '--work-dir', str(work), '--auth', 'browseros', '--retranscribe']) == 1
    assert json.loads((work / 'scribe-state.json').read_text())['job_id'] == 'pending-job'
    with pytest.raises(SystemExit):
        transcribe_scribe.main([str(source), '--work-dir', str(work), '--retranscribe', '--job-id', 'x'])


def test_scribe_language_follows_the_local_whisper_codes():
    from scribe_processing import language_code, scribe_language
    assert scribe_language('en') == 'English (verbatim)'
    assert scribe_language('en', 'standard') == 'English'
    assert scribe_language('sv') == 'Swedish (verbatim)'
    assert scribe_language('de') == 'German'  # no verbatim model
    assert scribe_language('nb', 'standard') == 'Norwegian'
    # Saved runs and the CLI may name the Scribe language directly.
    assert scribe_language('Swedish (verbatim)', 'standard') == 'Swedish (verbatim)'
    assert scribe_language('swedish', 'standard') == 'Swedish'
    assert language_code('English (verbatim)') == 'en'
    assert language_code('Northern Sámi (Experimental)') == 'no'


def test_scribe_language_refuses_what_the_worker_cannot_run():
    import pytest
    from scribe_processing import scribe_language
    for language in ('ja', '', 'Klingon'):
        with pytest.raises(ValueError, match='SWHISPER_LANGUAGE'):
            scribe_language(language)
    with pytest.raises(ValueError):
        scribe_language('sv', 'fast')


def test_recording_language_prefers_the_sidecar(tmp_path, monkeypatch):
    from scribe_processing import recording_language
    source = tmp_path / 'talk.wav'
    source.write_bytes(b'x')
    monkeypatch.delenv('SWHISPER_LANGUAGE', raising=False)
    assert recording_language(source) == 'sv'
    monkeypatch.setenv('SWHISPER_LANGUAGE', 'fi')
    assert recording_language(source) == 'fi'
    source.with_suffix('.swhisper').write_text('SWHISPER_LANGUAGE=en\n')
    assert recording_language(source) == 'en'
