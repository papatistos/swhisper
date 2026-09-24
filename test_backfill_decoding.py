"""Default backfill is one attempt; uncertain final fallbacks are opt-in."""
import copy
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from diarize.backfill_decoding import decoding_settings, validate_result, transcribe_with_policy
from diarize.backfill_results import BackfillResults, failure_segment, turn_key
from diarize.utils import BackfillCache, BackfillTranscriber, _backfill_subprocess_worker


def candidate(score=-.2, ratio=1.0, start=.1, end=.2):
    return dict(start=start, end=end, text='hej', avg_logprob=score, compression_ratio=ratio,
                words=[dict(start=start, end=end, word='hej')])


def test_default_attempt_does_not_mutate_transcription_settings():
    original = dict(temperature=[0, .2, .4, .6, .8, 1], beam_size=5, best_of=5)
    before = copy.deepcopy(original)
    assert decoding_settings(original)['temperature'] == [0.0]
    assert decoding_settings(original, True) == before
    assert original == before
    assert BackfillTranscriber('unused', 'unused', 'cpu', original).settings['temperature'] == [0.0]


def test_reject_before_low_confidence_decoding_can_become_empty_result():
    class Model:
        def decode(self):
            return SimpleNamespace(avg_logprob=-1.42, compression_ratio=.8, no_speech_prob=0.01)
    model = Model()
    def transcribe(model, audio, **settings):
        model.decode()
        return {'segments': []}
    with pytest.raises(ValueError, match='avg_logprob'):
        transcribe_with_policy(transcribe, model, None, {})
    assert 'decode' not in vars(model)
    assert transcribe_with_policy(transcribe, model, None, {}, True) == {'segments': []}


def test_decoder_silence_exception_and_existing_override_are_preserved():
    decode = lambda: SimpleNamespace(avg_logprob=-2, compression_ratio=.8, no_speech_prob=.9)
    model = SimpleNamespace(decode=decode)
    def transcribe(model, audio, **settings):
        model.decode()
        return {'segments': []}
    assert transcribe_with_policy(transcribe, model, None, {}) == {'segments': []}
    assert model.decode is decode


@pytest.mark.parametrize('score,ratio,reason', [(-1.42, 1, 'avg_logprob'),
                                             (-.2, 3, 'compression_ratio'),
                                             (float('nan'), 1, 'invalid'),
                                             (None, 1, 'missing')])
def test_failed_quality_checks_are_rejected(score, ratio, reason):
    with pytest.raises(ValueError, match=reason):
        validate_result({'segments': [candidate(score, ratio)]}, {}, 0, .3)


def test_accepted_silence_and_padding_do_not_fail():
    validate_result({'segments': [candidate(-1, 2.4)]}, {}, 0, .3)
    validate_result({'segments': []}, {}, 0, .3)
    validate_result({'segments': [candidate(-2, 3, 1, 1.2)]}, {}, 0, .3)
    validate_result({'segments': [candidate(-2, 3)]},
                    dict(logprob_threshold=None, compression_ratio_threshold=None), 0, .3)


@pytest.mark.parametrize('try_harder', [False, True])
def test_worker_rejects_uncertain_text_only_in_default_mode(monkeypatch, tmp_path, try_harder):
    import diarize.utils as utils
    audio = tmp_path / 'audio.wav'
    sf.write(audio, np.zeros(16000, dtype='float32'), 16000)
    seen = []
    monkeypatch.setattr(utils.whisper, 'load_model', lambda *a, **kw: object())
    def transcribe(model, audio, **settings):
        seen.append(settings)
        return {'segments': [candidate(-1.42)]}
    monkeypatch.setattr(utils.whisper, 'transcribe', transcribe)
    turn = dict(start=0, end=.3, speaker='SPEAKER_00')
    settings = dict(temperature=[0, .2, .4, .6, .8, 1])
    inp, out = queue.Queue(), queue.Queue()
    inp.put((dict(audio_path=str(audio), model_name='fake', device='cpu', settings=settings,
                  overlap_duration=0, sample_rate=16000, try_harder=try_harder), [turn], {}, 'test'))
    _backfill_subprocess_worker(inp, out)
    partial = BackfillResults([turn])
    events = []
    while not out.empty():
        status, result = out.get()
        events.append(status)
        if status == 'turn':
            partial.record(result['key'], result['segment'], result.get('error'))
    assert events[-1] == 'success'
    result = partial.outcome()
    assert len(seen) == 1
    assert seen[0]['temperature'] == (settings['temperature'] if try_harder else [0.0])
    if try_harder:
        assert result['word_count'] == 1
        assert not result['failed_turns']
    else:
        assert result['word_count'] == 0
        failed = result['failed_turns'][0]
        assert 'avg_logprob' in failed['backfill_error']
        assert failure_segment(failed, failed['backfill_error'])['text'] == '[---]'


def test_switching_retry_mode_invalidates_backfill_cache(tmp_path):
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    def cache(try_harder):
        return BackfillCache(str(tmp_path / 'cache'), str(audio), 'model', 'cpu', .5,
                             try_harder=try_harder)
    cache(True).save('audio', {'turn': candidate(-1.42)})
    assert cache(True).load('audio')
    assert not cache(False).load('audio')
    cache(False).save('audio', {'turn': candidate(-.2)})
    assert not cache(True).load('audio')


def test_setting_is_available_globally_per_file_and_for_scribe(tmp_path):
    from diarize.config import DiarizationConfig
    from sidecar import apply_sidecar_to_diarization_config
    from scribe_processing import capability, effective_settings
    from webui_schema import PARAM_GROUPS, PARAM_HELP
    config = DiarizationConfig(hugging_face_token='')
    assert config.backfill_try_harder is False
    key = 'SWHISPER_BACKFILL_TRY_HARDER'
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    audio.with_suffix('.swhisper').write_text(key + '=true\n')
    config, _ = apply_sidecar_to_diarization_config(config, str(tmp_path), 'audio')
    assert config.backfill_try_harder is True
    assert effective_settings(config)['backfill_try_harder'] is True
    assert capability(key)['supported']
    assert capability(key)['scope'] == 'Local backfill only'
    assert any(key in group['keys'] for group in PARAM_GROUPS)
    assert key in PARAM_HELP
