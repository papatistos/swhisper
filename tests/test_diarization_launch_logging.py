"""Launch labels describe the effective model, not the entry point's history."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from diarization_status import COMMUNITY_MODEL, LEGACY_MODEL, PRECISION_MODEL

NEMOTRON = 'nvidia/Nemotron-3-Diarization'


@pytest.mark.parametrize('model,precision,expected', [
    (COMMUNITY_MODEL, False, COMMUNITY_MODEL),
    (NEMOTRON, False, NEMOTRON),
    (LEGACY_MODEL, False, COMMUNITY_MODEL),
    (NEMOTRON, True, PRECISION_MODEL),
    ('custom/model', False, 'custom/model'),
])
def test_direct_and_transcription_launch_labels(monkeypatch, capsys, model, precision, expected):
    import diarize4
    import transcribe4
    config = SimpleNamespace(audio_dir='/synthetic/audio', pipeline_model=model,
                             use_precision_service=precision,
                             precision_pipeline_model=PRECISION_MODEL,
                             precision_api_token='test', hugging_face_token='test')
    monkeypatch.setattr(diarize4, 'DiarizationConfig', lambda: config)
    monkeypatch.setattr(diarize4, 'ensure_pyannote_v4', lambda: None)
    invoked = Mock()
    monkeypatch.setattr(diarize4, '_invoke_legacy_main', invoked)
    monkeypatch.setattr(transcribe4.resource_manager, 'clear_device_memory', lambda: None)
    for launch in (lambda: diarize4.run(config),
                   lambda: transcribe4.TranscriptionAppV4._run_diarization_pipeline(None)):
        config.pipeline_model = model
        launch()
        log = capsys.readouterr().out
        assert f'Launching diarization pipeline ({expected})' in log
        if expected in (NEMOTRON, 'custom/model'):
            assert 'pyannote' not in log.lower()
    assert invoked.call_count == 2
    assert config.pipeline_model == expected


def test_diarization_first_banner_is_model_neutral(monkeypatch, capsys):
    import transcribe5
    app = Mock()
    monkeypatch.setattr(transcribe5, 'TranscriptionAppV5', lambda: app)
    transcribe5.main()
    assert 'pyannote' not in capsys.readouterr().out.lower()
    app.run.assert_called_once_with()


def test_webui_standard_pipeline_label_is_model_neutral():
    from webui import PIPELINES
    assert PIPELINES['transcribe4']['label'] == 'Standard pipeline (transcribe → diarize)'
