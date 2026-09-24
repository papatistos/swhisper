"""Model-aware completion checks using temporary artifacts, without inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from diarization_status import COMMUNITY_MODEL, LEGACY_MODEL, effective_v4_model, model_run_reason
from run_scope import DIARIZATION_MODELS

NEMOTRON = DIARIZATION_MODELS['nemotron']


def legacy_output(root, model=COMMUNITY_MODEL):
    marker = root / 'interview.ok'
    marker.write_text('# Start: 20260902-021419\n')
    artifact = root / 'json/interview_20260902-021419_diarized.json'
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps({'metadata': {'diarization_model': model}}))
    return marker, artifact


def test_legacy_marker_uses_its_exact_json_and_tracks_preview_evidence(tmp_path):
    marker, artifact = legacy_output(tmp_path)
    (artifact.parent / 'interview_20260924-123000_diarized.json').write_text(
        json.dumps({'metadata': {'diarization_model': NEMOTRON}}))
    evidence = []
    assert model_run_reason(marker, COMMUNITY_MODEL, evidence.append) is None
    assert evidence == [artifact]
    assert COMMUNITY_MODEL in model_run_reason(marker, NEMOTRON)


def test_missing_or_invalid_provenance_does_not_claim_model_completed(tmp_path):
    marker, artifact = legacy_output(tmp_path)
    for content in ['{}', 'null', 'broken JSON']:
        artifact.write_text(content)
        assert 'no recorded diarization model' in model_run_reason(marker, NEMOTRON)
    artifact.unlink()
    assert model_run_reason(marker, NEMOTRON)


def test_new_marker_can_verify_model_without_reading_transcript(tmp_path):
    marker = tmp_path / 'interview.ok'
    marker.write_text(f'# Diarization model: {NEMOTRON}\n')
    stamp = Mock()
    assert model_run_reason(marker, NEMOTRON.upper(), stamp) is None
    stamp.assert_not_called()
    assert model_run_reason(marker, COMMUNITY_MODEL)


def test_v4_resolves_actual_default_and_precision_override():
    assert effective_v4_model(SimpleNamespace(pipeline_model=LEGACY_MODEL)) == COMMUNITY_MODEL
    assert effective_v4_model(SimpleNamespace(pipeline_model=NEMOTRON)) == NEMOTRON
    assert effective_v4_model(SimpleNamespace(pipeline_model=NEMOTRON, use_precision_service=True,
                                             precision_pipeline_model='precision')) == 'precision'


@pytest.fixture(scope='module')
def pipeline():
    spec = importlib.util.spec_from_file_location('diarize_status_test', Path(__file__).with_name('diarize.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pipeline_enters_processing_for_new_model_and_skips_same_model(tmp_path, monkeypatch, pipeline):
    marker, artifact = legacy_output(tmp_path)
    class ProcessingReached(Exception):
        pass
    config = SimpleNamespace(final_json_input_dir=str(tmp_path), final_audio_dir=str(tmp_path),
                             final_output_dir=str(tmp_path), force_reprocess=False,
                             pipeline_model=NEMOTRON,
                             get_output_subdirs=Mock(side_effect=ProcessingReached))
    monkeypatch.delenv('SWHISPER_RUN_REPROCESS', raising=False)
    with pytest.raises(ProcessingReached):
        pipeline.process_file(config, 'interview.json', 1, 1, apply_sidecar=False)
    config.pipeline_model = COMMUNITY_MODEL
    assert pipeline.process_file(config, 'interview.json', 1, 1, apply_sidecar=False)
    assert config.get_output_subdirs.call_count == 1
    # Checks do not rewrite either artifact.
    assert marker.read_text() == '# Start: 20260902-021419\n'
    assert json.loads(artifact.read_text())['metadata']['diarization_model'] == COMMUNITY_MODEL


def test_completion_marker_records_model(tmp_path, pipeline):
    config = SimpleNamespace(final_output_dir=str(tmp_path), audio_subdir='', audio_dir=str(tmp_path),
                             json_input_dir='whisper-json-output', output_dir='transcripts', pipeline_model=NEMOTRON)
    pipeline.create_completion_marker(config, 'interview', '20260924-123000', {}, {'status': 'complete'})
    marker = tmp_path / 'interview.ok'
    assert model_run_reason(marker, NEMOTRON) is None
    assert '# Backfill status: complete' in marker.read_text()
