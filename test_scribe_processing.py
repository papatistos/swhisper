"""Scribe recovery integration with simulated model outputs, no network/models."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from pyannote.core import Annotation, Segment

import transcribe_scribe
from diarize.pipeline import DiarizationPipeline
from diarize.utils import BackfillCache, BackfillTranscriber
from scribe_api import ScribeClient
from scribe_processing import capability


def test_capabilities_distinguish_remote_and_recovery():
    assert capability('SWHISPER_MIN_SPEAKERS')['scope'] == 'Local processing after Scribe'
    assert capability('SWHISPER_INITIAL_PROMPT')['scope'] == 'Local backfill only'
    assert capability('SWHISPER_LANGUAGE')['scope'] == 'Unavailable for Scribe'
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
    argv = [str(source), '--work-dir', str(work), '--auth', 'env']
    with patch.dict('os.environ', {'SUNET_SCRIBE_API_KEY': 'test-key', 'SUNET_SCRIBE_ENCRYPT_PWD': 'test-password'}), \
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
        assert (work / 'whisper-json-output/audio.json').read_bytes() == normalized
