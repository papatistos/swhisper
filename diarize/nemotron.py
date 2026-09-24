"""NVIDIA Nemotron 3 Diarization backend.

The model needs Python >= 3.12 and transformers built from source, so it runs in a
separate Python environment through nemotron_worker.py. This module starts that
worker, reads back the per-frame speaker probabilities, and turns them into the
same pyannote annotations the rest of swhisper consumes.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyannote.core import Annotation, Segment

from .config import NEMOTRON_MODEL_ID, is_nemotron_model  # noqa: F401 - re-exported

NEMOTRON_MAX_SPEAKERS = 8
WORKER_PATH = Path(__file__).with_name("nemotron_worker.py")

SETUP_HINT = (
    "Create the environment with:\n"
    "  conda create -n nemo python=3.12\n"
    "  conda run -n nemo pip install torch numpy librosa git+https://github.com/huggingface/transformers\n"
    "or point SWHISPER_NEMOTRON_PYTHON at the python binary of an environment that has them."
)


def resolve_worker_python(configured: Optional[str] = None) -> str:
    """Find the Python that runs the worker: the configured one, or the "nemo" conda env."""
    if configured:
        return configured
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return str(prefix.parent / "nemo" / "bin" / "python")
    return str(prefix / "envs" / "nemo" / "bin" / "python")


@dataclass
class NemotronOutput:
    """Mirrors the two annotations of pyannote.audio 4's DiarizeOutput."""

    speaker_diarization: Annotation
    exclusive_speaker_diarization: Annotation


def _runs(active: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end) frame indices of each run of True values."""
    edges = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _segments(active: np.ndarray, frame_step: float, min_on: float, min_off: float) -> List[Tuple[float, float]]:
    """Turn a frame mask into segments, closing pauses < min_off and dropping turns < min_on."""
    merged: List[List[float]] = []
    for start, end in _runs(active):
        start_s, end_s = start * frame_step, end * frame_step
        if merged and start_s - merged[-1][1] < min_off:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    return [(s, e) for s, e in merged if e - s >= min_on]


def binarize(
    probabilities: np.ndarray,
    frame_step: float = 0.01,
    threshold: float = 0.5,
    min_duration_on: float = 0.1,
    min_duration_off: float = 0.3,
    max_speakers: Optional[int] = None,
) -> NemotronOutput:
    """Convert [frames, channels] speaker probabilities into pyannote annotations.

    The standard annotation keeps overlapping speech. The exclusive annotation
    gives each speech frame to the single most probable speaker. When
    max_speakers is set, only the most active channels are kept, and speech of
    the other channels goes to the most probable kept channel.

    WARNING: Do not use max_speakers with Nemotron; leave it as None. This
    post-processing heuristic does not constrain the model's inference. It
    forces speech onto retained speakers even when their probabilities are
    very low, so it can produce misleading speaker labels. It has no value
    as a model speaker-count constraint and should not be used for that purpose.
    """
    active = probabilities > threshold
    kept = [c for c in range(probabilities.shape[1]) if active[:, c].any()]

    if max_speakers and len(kept) > max_speakers:
        kept = sorted(sorted(kept, key=lambda c: -active[:, c].sum())[:max_speakers])
        dropped_speech = np.delete(active, kept, axis=1).any(axis=1)
        best_kept = np.array(kept)[probabilities[:, kept].argmax(axis=1)]
        frames = np.flatnonzero(dropped_speech)
        active[frames, best_kept[frames]] = True

    kept_probabilities = probabilities[:, kept] if kept else np.zeros((len(probabilities), 0))
    any_speech = active[:, kept].any(axis=1) if kept else np.zeros(len(probabilities), dtype=bool)
    winner = np.array(kept)[kept_probabilities.argmax(axis=1)] if kept else None

    standard: Dict[int, List[Tuple[float, float]]] = {}
    exclusive: Dict[int, List[Tuple[float, float]]] = {}
    for channel in kept:
        standard[channel] = _segments(active[:, channel], frame_step, min_duration_on, min_duration_off)
        exclusive[channel] = _segments(any_speech & (winner == channel), frame_step, min_duration_on, min_duration_off)

    # Number speakers SPEAKER_00, SPEAKER_01, ... in order of first appearance.
    order = sorted((c for c in kept if standard[c]), key=lambda c: standard[c][0][0])
    labels = {channel: f"SPEAKER_{index:02d}" for index, channel in enumerate(order)}

    output = NemotronOutput(Annotation(), Annotation())
    for source, annotation in ((standard, output.speaker_diarization), (exclusive, output.exclusive_speaker_diarization)):
        for channel, label in labels.items():
            for start, end in source[channel]:
                annotation[Segment(start, end)] = label
    return output


class NemotronDiarizer:
    """Runs Nemotron 3 Diarization in its own environment and returns pyannote annotations."""

    def __init__(self, config):
        self.config = config

    def diarize(self, audio_path: str) -> NemotronOutput:
        python = resolve_worker_python(getattr(self.config, "nemotron_python", None))
        if not os.path.exists(python):
            raise RuntimeError(f"Nemotron environment not found: {python}\n{SETUP_HINT}")

        max_speakers = self.config.max_speakers
        if max_speakers and max_speakers > NEMOTRON_MAX_SPEAKERS:
            print(f"  -> Note: Nemotron tracks at most {NEMOTRON_MAX_SPEAKERS} speakers; SWHISPER_MAX_SPEAKERS={max_speakers} cannot be reached.")
        if self.config.min_speakers and self.config.min_speakers > 1:
            print(f"  -> Note: Nemotron ignores SWHISPER_MIN_SPEAKERS={self.config.min_speakers}.")

        device = getattr(self.config, "nemotron_device", "cpu")
        print(f"  -> Running {NEMOTRON_MODEL_ID} on {device} (worker: {python})...")
        sys.stdout.flush()

        with tempfile.TemporaryDirectory(prefix="swhisper-nemotron-") as tmp:
            output_npy = os.path.join(tmp, "probabilities.npy")
            result = subprocess.run(
                [python, str(WORKER_PATH), audio_path, output_npy,
                 "--model", NEMOTRON_MODEL_ID, "--device", device],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
                raise RuntimeError(f"Nemotron worker failed (exit {result.returncode}):\n{stderr_tail}")
            info = json.loads(result.stdout.strip().splitlines()[-1])
            probabilities = np.load(output_npy)

        print(f"  -> Nemotron processed {info['audio_seconds'] / 60:.1f} min of audio in {info['elapsed_seconds']:.0f} s.")

        output = binarize(
            probabilities,
            frame_step=info["frame_step"],
            threshold=self.config.nemotron_threshold,
            min_duration_on=self.config.nemotron_min_duration_on,
            min_duration_off=self.config.nemotron_min_duration_off,
            max_speakers=max_speakers,
        )
        print(f"  -> Nemotron found {len(output.speaker_diarization.labels())} speakers.")
        return output
