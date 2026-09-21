from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_errors import failure_reason


@pytest.mark.parametrize('missing', [True, False])
def test_working_copy_is_rebuilt_without_upload_or_transcript_changes(tmp_path, missing):
    import json
    from types import SimpleNamespace
    from scribe_api import ScribeError
    import transcribe_scribe

    source = tmp_path / 'recording.m4a'
    source.write_bytes(b'original recording')
    work = tmp_path / 'run'
    work.mkdir()
    audio = work / 'audio.wav'
    audio.write_bytes(b'original working copy')
    args = SimpleNamespace(language='Swedish (verbatim)', base_url='https://scribe.sunet.se', job_id=None, transcription_only=True)
    state = {'identity': {'source': str(source), 'sha256': transcribe_scribe.sha256(source),
                         'language': args.language, 'base_url': args.base_url, 'adapter_version': 1,
                         'audio_format': 'pcm_s16le/16000/mono'},
             'wav_sha256': transcribe_scribe.sha256(audio)}
    state_path = work / 'scribe-state.json'
    state_path.write_text(json.dumps(state))
    transcript = work / 'whisper-json-output' / 'audio.json'
    transcript.parent.mkdir()
    transcript.write_text('saved transcript')
    state['output_sha256'] = transcribe_scribe.sha256(transcript)
    state['job_id'] = 'existing-job'
    state_path.write_text(json.dumps(state))
    if missing:
        audio.unlink()
    else:
        audio.write_bytes(b'changed working copy')
    def rebuild(src, destination):
        destination.write_bytes(b'rebuilt working copy')
    with patch.object(transcribe_scribe, 'ScribeClient') as remote, \
            patch.object(transcribe_scribe, 'prepare_audio', side_effect=rebuild) as prepare:
        transcribe_scribe.run_locked(args, source, work)
        transcribe_scribe.run_locked(args, source, work)
        prepare.assert_called_once()
        remote.assert_not_called()
    assert audio.read_bytes() == b'rebuilt working copy'
    assert transcript.read_text() == 'saved transcript'
    updated = json.loads(state_path.read_text())
    assert updated['wav_sha256'] == state['wav_sha256']
    assert updated['local_wav_sha256'] == transcribe_scribe.sha256(audio)
    assert updated['job_id'] == 'existing-job'
    assert source.read_bytes() == b'original recording'
    if not missing:
        backups = list((work / 'replaced-audio').glob('*.wav'))
        assert len(backups) == 1
        assert backups[0].read_bytes() == b'changed working copy'
    source.write_bytes(b'different original')
    with patch.object(transcribe_scribe, 'prepare_audio') as prepare, pytest.raises(ScribeError):
        transcribe_scribe.run_locked(args, source, work)
    prepare.assert_not_called()


def test_rebuild_failure_preserves_previous_copy_and_state(tmp_path):
    import transcribe_scribe
    from scribe_api import ScribeError
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'old')
    state = {'identity': {'sha256': 'unchanged'}, 'wav_sha256': 'remote-checksum'}
    with patch.object(transcribe_scribe, 'prepare_audio', side_effect=ScribeError('Conversion failed')):
        with pytest.raises(ScribeError):
            transcribe_scribe.restore_working_audio(tmp_path / 'source', audio, state, tmp_path / 'state.json')
    assert audio.read_bytes() == b'old'
    assert 'local_wav_sha256' not in state


def test_missing_symlink_target_explains_cache_failure(tmp_path):
    cache = tmp_path / 'hub'
    target = tmp_path / 'disconnected-drive' / 'models'
    cache.symlink_to(target)
    missing = cache / 'community-1'
    reason = failure_reason(FileNotFoundError(2, 'No such file or directory', str(missing)))
    assert str(target / 'community-1') in reason
    assert 'Reconnect the drive' in reason


def test_ordinary_missing_file_does_not_claim_drive_failure(tmp_path):
    reason = failure_reason(FileNotFoundError(2, 'Missing', str(tmp_path / 'audio.wav')))
    assert 'FileNotFoundError' in reason
    assert 'Reconnect' not in reason


@pytest.mark.parametrize('code,stopped', [(1, False), (0, False), (130, True)])
def test_failure_reason_is_last_after_cleanup_and_history(tmp_path, code, stopped):
    from webui import AppState
    state = AppState(str(tmp_path))
    state.run_id = 'failure-test'
    state.finished_at = None
    state.stop_requested = stopped
    log = tmp_path / 'run.log'
    log.write_text('Error: Local model cache is unavailable.\nCleaning up resources...\nCleanup completed\n')
    state.run_log_path = str(log)
    def history():
        with state.lock:
            state._append_log_line_locked('Run recorded in history')
    with patch.object(state, '_write_history_entry', side_effect=history), \
            patch.object(state, '_remove_active_metadata_locked'):
        state._finish_run('failure-test', code)
    lines = log.read_text().splitlines()
    if code and not stopped:
        assert lines[-1] == '❌ Failure reason: Local model cache is unavailable.'
    else:
        assert not any('Failure reason:' in line for line in lines)
