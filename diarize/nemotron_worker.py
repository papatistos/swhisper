"""Run Nemotron 3 Diarization on one audio file and save its speaker probabilities.

This script runs in its own Python environment (Python >= 3.12 with transformers
built from source), not in the swhisper environment. It imports nothing from
swhisper. The swhisper side (diarize/nemotron.py) starts it as a subprocess and
turns the saved probabilities into pyannote annotations.

Usage:
    python nemotron_worker.py AUDIO_PATH OUTPUT_NPY [--model ID] [--device cpu]

The output file holds a float32 array of shape [frames, 8]: the probability that
each of the model's eight speaker channels is active in each 10 ms frame. The
last line on stdout is a JSON object describing the run.
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

FRAME_STEP_SECONDS = 0.01


def load_audio(path: str, sampling_rate: int) -> np.ndarray:
    """Decode any format ffmpeg reads (wav, m4a, mp3, video, ...) to mono float32."""
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(sampling_rate), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(decoded.stdout, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio_path")
    parser.add_argument("output_npy")
    parser.add_argument("--model", default="nvidia/Nemotron-3-Diarization")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(args.model).to(args.device).eval()

    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(args.audio_path, sampling_rate=sampling_rate)
    inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        logits = model(**inputs).logits[0]  # (frames, 8)

    probabilities = torch.sigmoid(logits).float().cpu().numpy()
    np.save(args.output_npy, probabilities)

    print(json.dumps({
        "frames": int(probabilities.shape[0]),
        "channels": int(probabilities.shape[1]),
        "frame_step": FRAME_STEP_SECONDS,
        "audio_seconds": round(len(audio) / sampling_rate, 3),
        "elapsed_seconds": round(time.time() - started, 2),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
