"""Backfill must preserve progress and distinguish failed work from empty speech."""
import copy
import gc
import queue
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from diarize import utils
from diarize.backfill_results import BackfillResults, failure_segment, turn_key, word_count
from transcript_render import RenderOptions, derive_segments
from transcript_render.markers import is_marker_token, with_backfill_legend
from transcript_render.formatters import TXTFormatter, RTFFormatter


def turn(start=1, end=1.21):
    return dict(start=start, end=end, speaker='SPEAKER_00')


def recovered(t, text='hej'):
    w = dict(t, word=text, text=text, is_backfill=True)
    return dict(t, text=text, words=[w])


@pytest.mark.parametrize('duration,token', [(0.1, '[-]'), (0.2, '[--]'),
                                          (0.201, '[---]'), (6, '[' + '-' * 50 + ']')])
def test_failure_marker_duration_and_word_count(duration, token):
    segment = failure_segment(turn(3.42, 3.42 + duration), 'timeout')
    assert segment['text'] == token
    assert is_marker_token(token)
    assert word_count(segment) == 0
    assert segment['words'][0]['is_backfill_failure']


def test_failure_markers_survive_gaps_silences_overlap_and_hidden_disfluencies():
    original = [recovered(turn(1, 2)), recovered(turn(1.5, 2.5), 'du'),
                dict(start=3, end=4, speaker='SILENCE', is_silence_marker=True,
                     words=[dict(start=3, end=4, text='(1.0)', speaker='SILENCE', is_silence_marker=True)])]
    gaps = [turn(0, .2), turn(1.6, 1.8), turn(2.7, 2.9), turn(3.2, 3.5), turn(5, 5.3)]
    merged = utils.BackfillMerger.merge_backfilled_words(original, [failure_segment(t, 'error') for t in gaps])
    markers = [w for s in merged for w in s['words'] if w.get('is_backfill_failure')]
    assert len(markers) == len(gaps)
    rendered = derive_segments(merged, RenderOptions(disfluencies=False))
    assert sum(w.get('is_backfill_failure', False) for s in rendered for w in s['words']) == len(gaps)


def test_failure_legend_in_custom_preamble_and_exports(tmp_path):
    config = SimpleNamespace(output_preamble='Custom introduction')
    segments = [failure_segment(turn(), 'timeout')]
    txt, rtf = tmp_path / 'out.txt', tmp_path / 'out.rtf'
    TXTFormatter().format(segments, str(txt), config=config)
    RTFFormatter().format(segments, str(rtf), config=config)
    for path in (txt, rtf):
        assert 'Backfill failures:' in path.read_text()
        assert '[---]' in path.read_text()
    assert with_backfill_legend(with_backfill_legend('Custom')) == with_backfill_legend('Custom')


def test_parent_checkpoints_and_retains_partial_results(monkeypatch):
    turns = [turn(), turn(2, 2.5)]
    first = recovered(turns[0])
    clock = [0.0]
    events = [('turn', dict(key=turn_key(turns[0]), segment=first))]
    class Output:
        def get(self, timeout):
            clock[0] += timeout
            if events:
                return events.pop(0)
            raise queue.Empty
        def cancel_join_thread(self): pass
        def close(self): pass
    class Input(Output):
        def put(self, value): pass
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: True,
                              terminate=lambda: None, kill=lambda: None, join=lambda **kw: None)
    queues = iter([Input(), Output()])
    context = SimpleNamespace(Queue=lambda: next(queues), Process=lambda **kw: process)
    monkeypatch.setattr(utils.mp, 'get_context', lambda _: context)
    monkeypatch.setattr(utils.time, 'monotonic', lambda: clock[0])
    saved = []
    cache = SimpleNamespace(load=lambda _: {}, save=lambda key, data: saved.append(copy.deepcopy(data)))
    transcriber = utils.BackfillTranscriber('unused', 'unused', 'cpu', {}, cache=cache)
    result = transcriber.transcribe_turns_subprocess(turns, 'audio', timeout=1)
    assert result['word_count'] == 1
    assert result['failed_turns'] == [turns[1]]
    assert result['segments'] == [first]
    assert 'progress' in result['error']
    assert saved[-1][turn_key(turns[0])] == first


def test_total_batch_can_exceed_timeout_with_progress(monkeypatch):
    turns = [turn(i, i + .2) for i in range(5)]
    clock = [0.0]
    events = [('turn', dict(key=turn_key(t), segment=recovered(t))) for t in turns]
    events.append(('success', {}))
    class Q:
        def put(self, value): pass
        def get(self, timeout):
            clock[0] += .8
            return events.pop(0)
        def cancel_join_thread(self): pass
        def close(self): pass
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: False, join=lambda **kw: None)
    context = SimpleNamespace(Queue=Q, Process=lambda **kw: process)
    monkeypatch.setattr(utils.mp, 'get_context', lambda _: context)
    monkeypatch.setattr(utils.time, 'monotonic', lambda: clock[0])
    result = utils.BackfillTranscriber('unused', 'unused', 'cpu', {}).transcribe_turns_subprocess(turns, timeout=1)
    assert clock[0] > 4
    assert result['word_count'] == 5
    assert not result['failed_turns']
    assert 'error' not in result


def test_worker_releases_whisper_before_sensevoice_and_streams(monkeypatch, tmp_path):
    audio = tmp_path / 'audio.wav'
    sf.write(audio, np.zeros(48000, dtype='float32'), 16000)
    references = []
    order = []
    class Model: pass
    def load(*args, **kw):
        model = Model()
        references.append(weakref.ref(model))
        return model
    def transcribe(model, audio, **kw):
        order.append('whisper')
        return {'segments': [dict(start=.1, end=.2, text='hej',
                                  avg_logprob=-.2, compression_ratio=.8,
                                  words=[dict(start=.1, end=.2, word='hej')])]}
    class Provider:
        def __init__(self, **kw):
            gc.collect()
            assert references[0]() is None
        def transcribe(self, **kw):
            order.append('sensevoice')
            return dict(text='hello', raw_text='hello', event='Speech')
        def close(self): pass
    import diarize.sensevoice_provider
    monkeypatch.setattr(utils.whisper, 'load_model', load)
    monkeypatch.setattr(utils.whisper, 'transcribe', transcribe)
    monkeypatch.setattr(diarize.sensevoice_provider, 'SenseVoiceProvider', Provider)
    inp, out = queue.Queue(), queue.Queue()
    inp.put((dict(audio_path=str(audio), model_name='fake', device='cpu', settings={},
                  overlap_duration=0, sample_rate=16000, enable_sensevoice=True),
             [turn(0, .3), turn(1, 1.3)], {}, 'audio'))
    utils._backfill_subprocess_worker(inp, out)
    events = []
    while not out.empty(): events.append(out.get())
    assert events[-1][0] == 'success', events
    assert order == ['whisper', 'whisper', 'sensevoice', 'sensevoice']
    updates = [data for status, data in events if status == 'turn']
    assert len(updates) == 4
    assert 'sensevoice_text' not in updates[0]['segment']
    assert updates[-1]['segment']['sensevoice_status'] == 'completed'


def test_cached_placeholders_are_not_counted_as_words():
    t = turn()
    result = BackfillResults([t], {turn_key(t): dict(recovered(t, '[***]'), is_placeholder=True)}).outcome()
    assert result['word_count'] == 0
    assert result['placeholders'] == 1


def test_worker_exit_preserves_cached_success_and_leaves_only_pending_failed(monkeypatch):
    turns = [turn(), turn(2, 3)]
    first = recovered(turns[0])
    class Q:
        def put(self, value): pass
        def get(self, timeout): raise queue.Empty
        def cancel_join_thread(self): pass
        def close(self): pass
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: False, exitcode=9)
    context = SimpleNamespace(Queue=Q, Process=lambda **kw: process)
    monkeypatch.setattr(utils.mp, 'get_context', lambda _: context)
    cache = SimpleNamespace(load=lambda _: {turn_key(turns[0]): first})
    result = utils.BackfillTranscriber('unused', 'unused', 'cpu', {}, cache=cache).transcribe_turns_subprocess(turns, 'audio')
    assert result['word_count'] == 1
    assert result['failed_turns'] == [turns[1]]
    assert 'code 9' in result['error']


def test_incomplete_run_is_reported_in_progress_and_history(tmp_path):
    import time
    import json
    import webui
    state = webui.AppState(str(tmp_path))
    state.pipeline = 'diarize4'
    state.run_id = 'test'
    state.started_at = time.time()
    state.run_audio_dir = str(tmp_path)
    state.run_log_path = str(tmp_path / 'run.log')
    (tmp_path / 'run.log').write_text('DIARIZATION LOG FOR: interview.wav\n'
                                   '⚠️ Transcript exported for interview.wav; backfill incomplete\n'
                                   '📄 Completion marker created: interview.ok\n')
    state._finish_run('test', 0)
    result = state.progress.snapshot()
    assert result['backfill_incomplete'] == 1
    assert result['completed']['Diarization'] == 1
    history = json.loads((tmp_path / webui.HISTORY_FILENAME).read_text().splitlines()[-1])
    assert history['backfill_incomplete'] == 1
    assert not history['success']
    assert any('backfill incomplete' in line for line in state.log_lines)


def test_incomplete_completion_marker_is_scheduled_for_retry(tmp_path):
    from webui_run import preview_run
    (tmp_path / 'a.wav').write_bytes(b'audio')
    (tmp_path / 'whisper-json-output').mkdir()
    (tmp_path / 'whisper-json-output/a.json').write_text('{}')
    (tmp_path / 'transcripts').mkdir()
    (tmp_path / 'transcripts/a.ok').write_text('# Backfill status: partial\n')
    tc = SimpleNamespace(json_dir='whisper-json-output', force_reprocess=False, file_limit=None)
    dc = SimpleNamespace(json_input_dir='whisper-json-output', output_dir='transcripts', force_reprocess=False)
    plan = preview_run(str(tmp_path), 'diarize4', tc, dc)
    assert plan['files'][0]['action'] == 'reprocess'
    assert 'backfill' in plan['files'][0]['reason']
