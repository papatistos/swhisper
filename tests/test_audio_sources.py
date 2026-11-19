"""Source selection agrees across previews and the processing queue."""
from types import SimpleNamespace

import pytest

from audio_sources import source_audio_files
from webui_run import preview_run


def inputs(root, names):
    for name in names:
        (root / name).write_bytes(b'fixture')


def configs():
    return (SimpleNamespace(json_dir='whisper-json-output', force_reprocess=False, file_limit=None),
            SimpleNamespace(json_input_dir='whisper-json-output', output_dir='transcripts', force_reprocess=False))


def test_sources_hide_converted_wavs_but_keep_original_wavs(tmp_path):
    inputs(tmp_path, ['interview.wav', 'interview.m4a', 'original.WAV', '09-58-40.m4a', 'notes.txt'])
    (tmp_path / 'directory.mp3').mkdir()
    assert set(source_audio_files(tmp_path)) == {'interview.m4a', 'original.WAV', '09-58-40.m4a'}


@pytest.mark.parametrize('pipeline', ['transcribe4', 'transcribe5'])
def test_reprocessing_queues_each_recording_once_in_preview_and_execution(tmp_path, monkeypatch, pipeline):
    from transcribe.file_utils import FileManager
    inputs(tmp_path, ['interview.wav', 'interview.m4a', 'original.wav'])
    json_dir = tmp_path / 'whisper-json-output'
    json_dir.mkdir()
    for base in ['interview', 'original']:
        (json_dir / (base + '.json')).write_text('{}')
    tc, dc = configs()
    monkeypatch.delenv('SWHISPER_ONLY_FILES', raising=False)
    monkeypatch.setenv('SWHISPER_RUN_REPROCESS', 'true')
    plan = preview_run(str(tmp_path), pipeline, tc, dc, reprocess=True)
    rows = [r for r in plan['files'] if r['stage'] != 'Diarization']
    queued = FileManager(tc).get_audio_files_to_process(str(tmp_path), str(json_dir))
    assert [r['file'] for r in rows] == queued
    assert set(queued) == {'interview.m4a', 'original.wav'}
    assert all(r['action'] == 'reprocess' for r in rows)
    assert not any('share a basename' in note for note in plan['notes'])
    # File limits count source recordings, not conversion copies.
    tc.file_limit = 1
    plan = preview_run(str(tmp_path), pipeline, tc, dc, reprocess=True)
    assert [r['file'] for r in plan['files'] if r['stage'] != 'Diarization' and r['action'] != 'skip'] == \
        FileManager(tc).get_audio_files_to_process(str(tmp_path), str(json_dir))


def test_diarization_preview_labels_the_original_recording(tmp_path):
    inputs(tmp_path, ['interview.m4a', 'interview.wav', 'original.wav'])
    (tmp_path / 'whisper-json-output').mkdir()
    for base in ['interview', 'original']:
        (tmp_path / 'whisper-json-output' / (base + '.json')).write_text('{}')
    plan = preview_run(str(tmp_path), 'diarize4', *configs(), reprocess=True)
    assert {r['file'] for r in plan['files']} == {'interview.m4a', 'original.wav'}


def test_multiple_non_wav_sources_remain_visible(tmp_path):
    inputs(tmp_path, ['interview.wav', 'interview.m4a', 'interview.mp3'])
    assert set(source_audio_files(tmp_path)) == {'interview.m4a', 'interview.mp3'}
    plan = preview_run(str(tmp_path), 'transcribe5', *configs())
    assert any('share a basename' in note for note in plan['notes'])


def test_selected_recording_and_added_conversion_do_not_change_preview(tmp_path):
    inputs(tmp_path, ['interview.m4a', 'other.wav'])
    first = preview_run(str(tmp_path), 'transcribe5', *configs(), only={'interview'}, reprocess=True)
    inputs(tmp_path, ['interview.wav'])
    second = preview_run(str(tmp_path), 'transcribe5', *configs(), only={'interview'}, reprocess=True)
    assert first == second
    assert [r['file'] for r in second['files']] == ['interview.m4a']
