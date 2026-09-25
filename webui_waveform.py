"""Small peak envelopes for the review players, without loading entire recordings.

The bounded memory cache is keyed by file identity. Source audio is only read.
"""
from array import array
from functools import lru_cache
from pathlib import Path
import math
import subprocess
import threading

import numpy as np
import soundfile as sf
import webui_durations


PEAKS = 2048


def waveform(path: Path, start: float = 0, end: float | None = None) -> dict:
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError("Choose a finite, increasing waveform interval.")
    path = path.resolve()
    stat = path.stat()
    return _cached(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, start, end)


@lru_cache(maxsize=32)
def _cached(path, size, modified, changed, start, end):
    try:
        audio = sf.SoundFile(path)
    except (RuntimeError, sf.LibsndfileError):
        return _compressed(path, start, end)
    with audio:
        if not len(audio):
            raise ValueError("The audio file is empty.")
        duration = len(audio) / audio.samplerate
        first = int(start * audio.samplerate)
        last = min(len(audio), math.ceil((end if end is not None else duration) * audio.samplerate))
        if first >= last:
            raise ValueError("The waveform interval is outside the recording.")
        audio.seek(first)
        edges = np.linspace(first, last, min(PEAKS, last - first) + 1, dtype=np.int64)
        peaks = []
        for start, end in zip(edges[:-1], edges[1:]):
            peak = 0.0
            remaining = int(end - start)
            while remaining:
                chunk = audio.read(min(remaining, 65536), dtype="float32", always_2d=True)
                if not len(chunk):
                    raise ValueError("The audio file could not be read completely.")
                peak = max(peak, float(np.max(np.abs(chunk))))
                remaining -= len(chunk)
            peaks.append(peak)
        return _result(duration, peaks, first / audio.samplerate, last / audio.samplerate)


def _result(duration, peaks, start=0, end=None):
    values = np.nan_to_num(np.asarray(peaks), nan=0, posinf=1, neginf=0)
    return {"duration": duration, "start": start, "end": duration if end is None else end,
            "peaks": np.clip(values, 0, 1).round(5).tolist()}


def _compressed(path, start=0, end=None):
    """FFmpeg fallback for containers libsndfile cannot open, e.g. M4A.

    Keep one peak per 32 ms, not the PCM. Stereo channels stay separate until
    taking their absolute maximum so opposite-phase audio cannot cancel out.
    """
    peaks = array("f")
    frames = 0
    duration = None
    block_frames = 256
    seek_args, length_args = [], []
    if start or end is not None:
        duration = webui_durations.measure(path)
        if duration is None or not math.isfinite(duration) or start >= duration:
            raise ValueError("The waveform interval is outside the recording.")
        end = min(end if end is not None else duration, duration)
        seek_args = ["-ss", str(start)]
        length_args = ["-t", str(end - start)]
        block_frames = max(1, min(256, math.ceil((end - start) * 8000 / PEAKS)))
    with subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", *seek_args, "-i", path, *length_args, "-vn",
         "-ar", "8000", "-ac", "2", "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as process:
        timer = threading.Timer(120, process.kill)
        timer.daemon = True
        timer.start()
        try:
            while True:
                chunk = process.stdout.read(block_frames * 2 * 4)
                if not chunk:
                    break
                samples = np.frombuffer(chunk, dtype="<f4")
                frames += len(samples) // 2
                peaks.append(float(np.max(np.abs(samples))))
            if process.wait() != 0 or not frames:
                raise ValueError("The audio format could not be decoded for the waveform.")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
    values = np.frombuffer(peaks, dtype=np.float32)
    # Include every small peak touched by an output bin, preserving short bursts.
    edges = np.linspace(0, frames / block_frames, min(PEAKS, len(values)) + 1)
    overview = [float(values[int(start):max(int(start) + 1, int(np.ceil(end)))].max())
                for start, end in zip(edges[:-1], edges[1:])]
    return _result(duration or frames / 8000, overview, start, start + frames / 8000)
