"""Exercise real log/export paths with synthetic annotations, without inference."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyannote.core import Annotation, Segment

from diarize.config import DiarizationConfig
from diarization_status import COMMUNITY_MODEL

NEMOTRON = 'nvidia/Nemotron-3-Diarization'


@pytest.fixture(scope='module')
def pipeline():
    spec = importlib.util.spec_from_file_location(
        'diarize_logging_test', Path(__file__).with_name('diarize.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('model', [COMMUNITY_MODEL, NEMOTRON, None])
@pytest.mark.parametrize('exclusive_enabled', [False, True])
@pytest.mark.parametrize('cached', [False, True])
def test_processing_logs_and_segment_exports(tmp_path, monkeypatch, pipeline,
                                             model, exclusive_enabled, cached):
    monkeypatch.setenv('TRANSCRIBE_AUDIO_DIR', str(tmp_path))
    monkeypatch.setenv('AUDIO_SUBDIR', '')
    monkeypatch.setenv('JSON_INPUT_DIR', 'input')
    config = DiarizationConfig(
        pipeline_model=model, output_dir='output', force_reprocess=True,
        use_exclusive_speaker_diarization=exclusive_enabled,
        diarization_cache_enabled=cached, diarization_cache_dir=None,
        whisper_timestamp_offset=0, backfill_missing_turns=False,
        smoothing_enabled=False, recording_type='', log_silence_gaps=False,
        output_preamble='', device='cpu')
    for directory in config.get_output_subdirs().values():
        Path(directory).mkdir(parents=True)
    Path(config.final_json_input_dir).mkdir()
    (tmp_path / 'interview.wav').write_bytes(b'synthetic audio placeholder')
    (Path(config.final_json_input_dir) / 'interview.json').write_text(json.dumps({
        'language': 'sv', 'segments': [
            {'start': 0.25, 'end': 1.25, 'text': 'Hej',
             'words': [{'start': 0.25, 'end': 1.25, 'word': 'Hej'}]},
            {'start': 1.5, 'end': 2.5, 'text': 'Ja',
             'words': [{'start': 1.5, 'end': 2.5, 'word': 'Ja'}]},
        ]}))
    standard = Annotation()
    standard[Segment(0.25, 1.75)] = 'SPEAKER_00'
    standard[Segment(1.5, 2.5)] = 'SPEAKER_01'
    exclusive = Annotation()
    exclusive[Segment(0.25, 1.5)] = 'SPEAKER_00'
    exclusive[Segment(1.5, 2.5)] = 'SPEAKER_01'
    output = SimpleNamespace(speaker_diarization=standard,
                             exclusive_speaker_diarization=exclusive)
    calls = []

    def infer(self, path):
        calls.append(path)
        return output

    monkeypatch.setattr(pipeline.DiarizationPipeline, 'diarize', infer)
    if cached:
        from diarize import DiarizationCache
        monkeypatch.setattr(DiarizationCache, 'load',
                            lambda self, *args, **kwargs:
                            exclusive if kwargs['exclusive'] else standard)
    # Check that the per-file override, not the outer configuration, names the model.
    monkeypatch.setattr(pipeline, 'apply_sidecar_to_diarization_config',
                        lambda *args: (config, None))
    outer_config = DiarizationConfig(pipeline_model='outer/model')
    assert pipeline.process_file(outer_config, 'interview.json', 1, 1, raise_errors=True)
    assert len(calls) == (0 if cached else 1)
    logs = list((tmp_path / 'output/logs').glob('*.log'))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert f'Diarization model: {model or "unspecified model"}' in log
    assert 'Saving diarization segment boundaries (overlapping timeline)' in log
    assert 'Saving diarization segment boundaries (exclusive timeline)' in log
    assert 'Diarization complete. (Duration:' in log
    assert 'outer/model' not in log
    if model != COMMUNITY_MODEL:
        assert 'pyannote' not in log.lower()
    if not exclusive_enabled:
        assert 'Exclusive diarization stream available but disabled in config.' in log
    for timeline, annotation in [('overlapping', standard), ('exclusive', exclusive)]:
        exports = list((tmp_path / 'output/tsv').glob(f'*_diarization_segments_{timeline}.tsv'))
        assert len(exports) == 1
        with exports[0].open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        expected = [(s.start, s.end, s.duration, label)
                    for s, _, label in annotation.itertracks(yield_label=True)]
        assert [(float(r['start']), float(r['end']), float(r['duration']), r['speaker'])
                for r in rows] == expected
        assert list(rows[0]) == ['segment_index', 'start', 'end', 'duration', 'speaker', 'confidence']


@pytest.mark.parametrize('model', [COMMUNITY_MODEL, NEMOTRON, 'custom/model', None, '', '  '])
def test_model_display_name_uses_config_or_neutral_fallback(model):
    from diarization_status import diarization_model_label
    assert diarization_model_label(SimpleNamespace(pipeline_model=model)) == (
        model.strip() if model and model.strip() else 'unspecified model')
    assert diarization_model_label(SimpleNamespace()) == 'unspecified model'
