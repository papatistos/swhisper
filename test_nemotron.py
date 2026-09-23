"""Tests for the Nemotron 3 Diarization backend.

Pure logic only: the worker subprocess is replaced by a fake, so no model is loaded.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from diarize.config import DiarizationConfig
from diarize.nemotron import (
    NemotronDiarizer,
    binarize,
    is_nemotron_model,
    resolve_worker_python,
)

STEP = 0.01


def speaker_probabilities(frames, channels, turns):
    """Build [frames, channels] probabilities with 0.9 inside each (channel, start_s, end_s) turn."""
    probabilities = np.full((frames, channels), 0.05, dtype=np.float32)
    for channel, start, end in turns:
        probabilities[round(start / STEP):round(end / STEP), channel] = 0.9
    return probabilities


def turns_of(annotation):
    return [(round(s.start, 2), round(s.end, 2), label) for s, _, label in annotation.itertracks(yield_label=True)]


def test_is_nemotron_model_matches_only_the_nemotron_id():
    assert is_nemotron_model("nvidia/Nemotron-3-Diarization")
    assert is_nemotron_model(" NVIDIA/nemotron-3-diarization ")
    assert not is_nemotron_model("pyannote/speaker-diarization-community-1")
    assert not is_nemotron_model(None)


def test_worker_python_defaults_to_the_nemo_env_next_to_the_current_env(monkeypatch):
    monkeypatch.setattr("sys.prefix", "/opt/anaconda3/envs/swhisper")
    assert resolve_worker_python() == "/opt/anaconda3/envs/nemo/bin/python"
    assert resolve_worker_python("/custom/python") == "/custom/python"


def test_standard_keeps_overlap_and_exclusive_picks_the_most_probable_speaker():
    probabilities = speaker_probabilities(300, 8, [(0, 0.0, 2.0), (1, 1.5, 3.0)])
    probabilities[150:200, 1] = 0.95  # speaker 1 is louder during the overlap

    output = binarize(probabilities, STEP, min_duration_on=0.0, min_duration_off=0.0)

    assert turns_of(output.speaker_diarization) == [
        (0.0, 2.0, "SPEAKER_00"),
        (1.5, 3.0, "SPEAKER_01"),
    ]
    assert turns_of(output.exclusive_speaker_diarization) == [
        (0.0, 1.5, "SPEAKER_00"),
        (1.5, 3.0, "SPEAKER_01"),
    ]


def test_short_pauses_are_closed_and_short_turns_are_dropped():
    probabilities = speaker_probabilities(500, 8, [(0, 0.0, 1.0), (0, 1.2, 2.0), (1, 3.0, 3.05)])

    output = binarize(probabilities, STEP, min_duration_on=0.1, min_duration_off=0.3)

    assert turns_of(output.speaker_diarization) == [(0.0, 2.0, "SPEAKER_00")]


def test_speakers_are_numbered_by_first_appearance():
    probabilities = speaker_probabilities(500, 8, [(5, 0.0, 1.0), (2, 2.0, 3.0)])

    output = binarize(probabilities, STEP)

    assert turns_of(output.speaker_diarization) == [
        (0.0, 1.0, "SPEAKER_00"),
        (2.0, 3.0, "SPEAKER_01"),
    ]


def test_max_speakers_gives_speech_of_dropped_speakers_to_kept_ones():
    probabilities = speaker_probabilities(1000, 8, [(0, 0.0, 4.0), (1, 4.0, 8.0), (2, 8.0, 9.0)])
    probabilities[800:900, 1] = 0.3  # channel 1 is the closest kept speaker during channel 2's turn

    output = binarize(probabilities, STEP, max_speakers=2)

    assert output.speaker_diarization.labels() == ["SPEAKER_00", "SPEAKER_01"]
    assert turns_of(output.exclusive_speaker_diarization) == [
        (0.0, 4.0, "SPEAKER_00"),
        (4.0, 9.0, "SPEAKER_01"),
    ]


def test_silence_produces_empty_annotations():
    output = binarize(np.zeros((100, 8), dtype=np.float32), STEP)

    assert len(output.speaker_diarization) == 0
    assert len(output.exclusive_speaker_diarization) == 0


def test_pipeline_config_holds_nemotron_settings_for_the_cache_key():
    config = DiarizationConfig()
    config.pipeline_model = "nvidia/Nemotron-3-Diarization"

    assert config.get_pipeline_config() == {
        "nemotron_threshold": config.nemotron_threshold,
        "nemotron_min_duration_on": config.nemotron_min_duration_on,
        "nemotron_min_duration_off": config.nemotron_min_duration_off,
    }


def fake_worker(probabilities, calls):
    """Stand in for subprocess.run: save the probabilities where the worker would."""
    def run(command, **kwargs):
        calls.append(command)
        np.save(command[3], probabilities)
        info = {"frames": len(probabilities), "channels": 8, "frame_step": STEP,
                "audio_seconds": len(probabilities) * STEP, "elapsed_seconds": 1.0}
        return subprocess.CompletedProcess(command, 0, stdout="loading...\n" + json.dumps(info) + "\n", stderr="")
    return run


def test_diarizer_runs_the_worker_and_returns_both_annotations(monkeypatch, tmp_path):
    python = tmp_path / "python"
    python.touch()
    config = DiarizationConfig()
    config.nemotron_python = str(python)
    config.nemotron_device = "cpu"
    config.min_speakers, config.max_speakers = 1, None
    calls = []
    monkeypatch.setattr(subprocess, "run", fake_worker(speaker_probabilities(300, 8, [(0, 0.0, 1.0), (3, 1.5, 2.5)]), calls))

    output = NemotronDiarizer(config).diarize("/audio/interview.m4a")

    assert calls[0][0] == str(python)
    assert Path(calls[0][1]).name == "nemotron_worker.py"
    assert calls[0][2] == "/audio/interview.m4a"
    assert calls[0][-2:] == ["--device", "cpu"]
    assert turns_of(output.speaker_diarization) == [(0.0, 1.0, "SPEAKER_00"), (1.5, 2.5, "SPEAKER_01")]
    assert turns_of(output.exclusive_speaker_diarization) == [(0.0, 1.0, "SPEAKER_00"), (1.5, 2.5, "SPEAKER_01")]


def test_diarizer_reports_worker_failures(monkeypatch, tmp_path):
    python = tmp_path / "python"
    python.touch()
    config = DiarizationConfig()
    config.nemotron_python = str(python)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 1, stdout="", stderr="Traceback ...\nRuntimeError: ffmpeg could not decode"))

    with pytest.raises(RuntimeError, match="ffmpeg could not decode"):
        NemotronDiarizer(config).diarize("/audio/broken.m4a")


def test_diarizer_explains_how_to_create_a_missing_environment(tmp_path):
    config = DiarizationConfig()
    config.nemotron_python = str(tmp_path / "missing" / "python")

    with pytest.raises(RuntimeError, match="conda create -n nemo"):
        NemotronDiarizer(config).diarize("/audio/interview.m4a")


def test_diarization_pipeline_delegates_to_nemotron(monkeypatch):
    from diarize import pipeline as pipeline_module

    config = DiarizationConfig()
    config.pipeline_model = "nvidia/Nemotron-3-Diarization"
    sentinel = object()
    monkeypatch.setattr(pipeline_module.NemotronDiarizer, "diarize", lambda self, path: sentinel)

    assert pipeline_module.DiarizationPipeline(config).diarize("/audio/interview.m4a") is sentinel
