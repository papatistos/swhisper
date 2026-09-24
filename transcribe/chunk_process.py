"""Run one diarization-guided Whisper chunk in a fresh process.

Process exit releases Metal allocations and native caches that Python garbage
collection and torch.mps.empty_cache() cannot reliably release between chunks.
"""

import multiprocessing as mp
from multiprocessing.connection import wait


def _transcribe_chunk(config, audio_path, start, end, settings, connection):
    try:
        import whisper_timestamped as whisper
        from .audio_utils import AudioProcessor

        audio, _ = AudioProcessor(config).load_audio_chunk(audio_path, start, end)
        model = whisper.load_model(config.model_str, device=config.device)
        result = whisper.transcribe(model, audio, **settings)
        connection.send(("ok", result))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def transcribe_chunk_in_subprocess(config, audio_path, start, end, settings,
                                  resource_manager, timeout=1800):
    """Return chunk-relative results, preserving the supplied decoder and VAD settings."""
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_transcribe_chunk,
        args=(config, audio_path, start, end, settings, writer),
    )
    started = False
    try:
        process.start()
        started = True
        resource_manager.current_subprocess = process
        writer.close()
        if not wait([reader, process.sentinel], timeout=timeout):
            raise TimeoutError(f"Whisper chunk timed out after {timeout:g} seconds")
        # Drain the pipe before joining: a result can exceed the pipe buffer.
        try:
            status, result = reader.recv()
        except EOFError as exc:
            process.join(timeout=5)
            raise RuntimeError(
                f"Whisper chunk worker exited without a result (exit code {process.exitcode})"
            ) from exc
        process.join(timeout=10)
        if process.is_alive():
            raise RuntimeError("Whisper chunk worker did not exit after returning its result")
        if status != "ok":
            raise RuntimeError(f"Whisper chunk failed: {result}")
        if process.exitcode != 0:
            raise RuntimeError(f"Whisper chunk worker failed (exit code {process.exitcode})")
        return result
    finally:
        if started and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        resource_manager.current_subprocess = None
        reader.close()
        writer.close()
        if not started or not process.is_alive():
            process.close()
