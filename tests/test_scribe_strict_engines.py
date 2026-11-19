import json
from types import SimpleNamespace

from scribe_processing import enforce_scribe_nemotron


def test_worker_pins_engines_after_sidecar_and_removes_speaker_truncation(tmp_path):
    config = SimpleNamespace(pipeline_model='pyannote', backfill_missing_turns=True,
                             backfill_enable_sensevoice=True, max_speakers=2)
    assert enforce_scribe_nemotron(config,tmp_path) is False
    assert config.pipeline_model=='nvidia/Nemotron-3-Diarization'
    assert config.backfill_missing_turns is False
    assert config.backfill_enable_sensevoice is False
    assert config.max_speakers is None


def test_strict_request_reprocesses_legacy_backfill_but_reuses_strict_output(tmp_path):
    marker=tmp_path/'transcripts/audio.ok';marker.parent.mkdir()
    marker.write_text('# Start: 20261111-130000\n')
    path=marker.parent/'json/audio_20261111-130000_diarized.json';path.parent.mkdir()
    for backfill, expected in [(True,True),(False,False)]:
        path.write_text(json.dumps({'metadata':{'processing':{'settings':{'backfill_missing_turns':backfill,'max_speakers':None}}}}))
        assert enforce_scribe_nemotron(SimpleNamespace(),tmp_path) is expected
    path.write_text('{}')
    assert enforce_scribe_nemotron(SimpleNamespace(),tmp_path) is True
