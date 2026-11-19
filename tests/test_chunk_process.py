"""Exercise process lifetime and result transfer without loading ASR models."""

import os
from types import SimpleNamespace

import pytest

from transcribe import chunk_process


def echo_worker(config, path, start, end, settings, connection):
    connection.send(("ok", {"pid": os.getpid(), "settings": settings,
                            "start": start, "end": end, "text": "x" * 1_000_000}))
    connection.close()


def crash_worker(config, path, start, end, settings, connection):
    os._exit(7)


def error_worker(config, path, start, end, settings, connection):
    connection.send(("error", "ValueError: invalid audio"))
    connection.close()


def test_each_chunk_uses_a_new_process_and_transfers_large_results(monkeypatch):
    monkeypatch.setattr(chunk_process, "_transcribe_chunk", echo_worker)
    manager = SimpleNamespace(current_subprocess=None)
    settings = {"vad": [(0.2, 2.8)], "temperature": [0.0, 0.2], "beam_size": 5}
    results = [chunk_process.transcribe_chunk_in_subprocess(
        None, "unused.wav", 100.0, 103.0, settings, manager, timeout=60,
    ) for _ in range(2)]
    assert results[0]["pid"] != results[1]["pid"]
    for result in results:
        assert result["pid"] != os.getpid()
        assert result["settings"] == settings
        assert (result["start"], result["end"]) == (100.0, 103.0)
        assert len(result["text"]) == 1_000_000
        with pytest.raises(ProcessLookupError):
            os.kill(result["pid"], 0)
    assert manager.current_subprocess is None


@pytest.mark.parametrize("worker, message", [
    (crash_worker, "without a result.*exit code 7"),
    (error_worker, "ValueError: invalid audio"),
])
def test_worker_failures_are_reported_and_reaped(monkeypatch, worker, message):
    monkeypatch.setattr(chunk_process, "_transcribe_chunk", worker)
    manager = SimpleNamespace(current_subprocess=None)
    with pytest.raises(RuntimeError, match=message):
        chunk_process.transcribe_chunk_in_subprocess(
            None, "unused.wav", 0, 1, {}, manager, timeout=60,
        )
    assert manager.current_subprocess is None


def test_timeout_terminates_and_reaps_worker(monkeypatch):
    monkeypatch.setattr(chunk_process, "_transcribe_chunk", echo_worker)
    manager = SimpleNamespace(current_subprocess=None)
    children_before = {p.pid for p in chunk_process.mp.active_children()}
    with pytest.raises(TimeoutError, match="timed out"):
        chunk_process.transcribe_chunk_in_subprocess(
            None, "unused.wav", 0, 1, {}, manager, timeout=0,
        )
    assert {p.pid for p in chunk_process.mp.active_children()} == children_before
    assert manager.current_subprocess is None


def test_worker_preserves_audio_loading_and_decoding_settings(monkeypatch):
    import whisper_timestamped as whisper
    from transcribe.audio_utils import AudioProcessor

    config = SimpleNamespace(model_str="test-model", device="cpu")
    model, audio = object(), object()
    calls, messages = [], []
    settings = {"vad": [(0.0, 2.0)], "beam_size": 5}
    monkeypatch.setattr(AudioProcessor, "load_audio_chunk",
                        lambda self, *args: (calls.append(args) or audio, 0))
    monkeypatch.setattr(whisper, "load_model",
                        lambda name, **kwargs: calls.append((name, kwargs)) or model)
    monkeypatch.setattr(whisper, "transcribe",
                        lambda m, a, **kwargs: calls.append((m, a, kwargs)) or {"segments": []})
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    chunk_process._transcribe_chunk(config, "test.wav", 10.0, 12.0, settings, connection)
    assert calls == [("test.wav", 10.0, 12.0),
                     ("test-model", {"device": "cpu"}), (model, audio, settings)]
    assert messages == [("ok", {"segments": []})]
