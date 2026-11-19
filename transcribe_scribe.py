#!/usr/bin/env python3
"""Single-recording SUNET transcription prototype with local diarization."""

import argparse
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from scribe_api import ScribeClient, ScribeError, atomic_json, normalize_result
from audio_integrity import pcm_checksum, checksum_without_jpr_metadata
from run_scope import DIARIZATION_MODELS
# State of the finished remote job; a retranscription starts a new one.
REMOTE_JOB_KEYS = ("job_id", "upload_started", "queue_started", "adopted_job", "output_sha256")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_audio(source, wav_path):
    """Match local pipeline input: 16 kHz, mono PCM; never modify the source."""
    temporary = wav_path.with_suffix(".tmp.wav")
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(temporary)
        ], check=True, capture_output=True)
    except FileNotFoundError:
        raise ScribeError("ffmpeg is required to prepare the audio.") from None
    except subprocess.CalledProcessError:
        raise ScribeError("ffmpeg could not prepare the audio; check the source file.") from None
    os.replace(temporary, wav_path)


def restore_working_audio(source, wav_path, state, state_path):
    """Rebuild disposable audio from the verified source; retain remote provenance."""
    candidate = wav_path.with_name(f"audio-rebuild-{uuid.uuid4().hex}.wav")
    print("Rebuilding the local audio copy from the original recording...", flush=True)
    try:
        prepare_audio(source, candidate)
        if sha256(source) != state['identity']['sha256']:
            raise ScribeError("The original recording changed during audio preparation. Retry with the original recording.")
        checksum = sha256(candidate)
        # Keep the previous copy for diagnosis without treating it as input again.
        if wav_path.exists():
            backup = wav_path.parent / 'replaced-audio'
            backup.mkdir(exist_ok=True)
            os.replace(wav_path, backup / f"audio-{uuid.uuid4().hex}.wav")
        os.replace(candidate, wav_path)
        # wav_sha256 describes the audio uploaded to SUNET and must stay unchanged.
        if not state.get('job_id') and not state.get('upload_started'):
            state['wav_sha256'] = checksum
        state['local_wav_sha256'] = checksum
        state['local_pcm_sha256'] = pcm_checksum(wav_path)
        atomic_json(state_path, state)
    finally:
        candidate.unlink(missing_ok=True)
        candidate.with_suffix('.tmp.wav').unlink(missing_ok=True)
    print("Local audio copy rebuilt. Continuing with the saved Scribe job.", flush=True)


def local_diarization(source, work_dir, json_name, diarization_model=None, force=False, strict_scribe_nemotron=False):
    # Imports delayed until after remote transcription. Local Whisper loads only if backfill needs it.
    from diarize import DiarizationConfig
    from diarize4 import ensure_pyannote_v4, resolve_pyannote_token

    # Legacy path environment variables must not redirect this isolated run.
    class PrototypeConfig(DiarizationConfig):
        @property
        def audio_dir(self):
            return str(work_dir)

        @property
        def json_input_dir(self):
            return "whisper-json-output"

        @property
        def audio_subdir(self):
            return ""

    from scribe_processing import configure, diarization_reason, effective_settings
    from transcribe import WhisperSettings
    from sidecar import sidecar_overrides

    config = PrototypeConfig()
    if diarization_model:
        config.pipeline_model = DIARIZATION_MODELS[diarization_model]
    config = configure(config, source, work_dir)
    if strict_scribe_nemotron:
        from scribe_processing import enforce_scribe_nemotron
        force = enforce_scribe_nemotron(config, work_dir) or force
    reason = diarization_reason(config, source, work_dir, force)
    if reason is None:
        print("Reusing the local diarization of this Scribe transcript; "
              "force reprocessing to run it again.", flush=True)
        return
    print(f"Local diarization: {reason}.", flush=True)
    # The decision above already covered the marker; process_file must not skip.
    config.force_reprocess = True
    whisper_settings = WhisperSettings()
    with sidecar_overrides(str(source.parent), source.stem, whisper_settings=whisper_settings):
        config.backfill_whisper_settings = whisper_settings.to_dict()
    config.backfill_whisper_settings.update(language="sv", task="transcribe", vad=None, verbose=False)
    config.processing_metadata = {
        "backend": "sunet-scribe", "timing_source": "scribe",
        "backfill_backend": "local-whisper", "settings": effective_settings(config),
        "backfill_whisper_settings": config.backfill_whisper_settings,
    }
    config.hugging_face_token = resolve_pyannote_token(config.hugging_face_token)
    ensure_pyannote_v4()
    for directory in config.get_output_subdirs().values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "scribe_local_diarize", Path(__file__).with_name("diarize.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(config.processing_metadata, status="running", started_at=datetime.now(timezone.utc).isoformat())
    record_path = work_dir / "local-runs" / (uuid.uuid4().hex + ".json")
    atomic_json(record_path, record)
    model = 'Nemotron' if config.uses_nemotron else 'community-1'
    print(f"Local {model} diarization; local Whisper backfill {'on' if config.backfill_missing_turns else 'off'}; "
          f"SenseVoice {'on (backfill snippets only)' if config.backfill_enable_sensevoice else 'off'}; "
          f"timestamp offset {config.whisper_timestamp_offset:g} seconds.", flush=True)
    try:
        if not module.process_file(config, json_name, 1, 1, apply_sidecar=False, raise_errors=True):
            raise ScribeError("Local diarization failed. Remote transcription remains saved for reuse.")
        record['status'] = 'completed'
    except Exception as exc:
        from pipeline_errors import failure_reason
        reason = failure_reason(exc)
        record.update(status='failed', failure_reason=reason)
        raise ScribeError(f"Local diarization failed: {reason} Remote transcription remains saved for reuse.") from exc
    except BaseException:
        record['status'] = 'failed_or_interrupted'
        raise
    finally:
        record.update(config.processing_metadata)
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(record_path, record)


def library_client(args, password):
    """A checked scribe-api client for --auth oidc or auto, else None for the fallback.

    In auto mode, missing OIDC credentials or a refused GET /me select the
    BrowserOS fallback. Nothing has been uploaded at that point.
    """
    if args.auth not in ("auto", "oidc"):
        return None
    import scribe_library
    if args.auth == "auto" and not scribe_library.available():
        print("scribe-api or its OIDC client credentials are not configured; "
              "using the BrowserOS fallback.", flush=True)
        return None
    client = scribe_library.LibraryScribeClient(
        password, args.base_url, poll_interval=args.poll_interval, wait_seconds=args.wait_seconds)
    try:
        client.check()
    except ScribeError as exc:
        client.close()
        if args.auth == "oidc":
            raise
        print(f"scribe-api could not sign in ({exc}) Using the BrowserOS fallback.", flush=True)
        return None
    print("Using scribe-api with OIDC client credentials.", flush=True)
    return client


def run(args):
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        raise ScribeError("Audio source does not exist or is not a file.")
    work_dir = (args.work_dir or source.parent / "scribe-prototype" / source.stem).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    # The directory belongs to one recording/settings combination. Never mix runs.
    with (work_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ScribeError("Another prototype run is using this work directory.") from None
        run_locked(args, source, work_dir)


def supersede_remote_result(work_dir, state, state_path, output_path):
    """Move the saved SUNET result aside and clear the finished job, so the next step uploads again."""
    target = work_dir / "superseded" / datetime.now().strftime("%Y%m%d-%H%M%S")
    target.mkdir(parents=True)
    atomic_json(target / "scribe-state.json", state)
    for path in (output_path, work_dir / "scribe-transcript.json", work_dir / "scribe-words.json"):
        if path.exists():
            path.rename(target / path.name)
    for key in REMOTE_JOB_KEYS:
        state.pop(key, None)
    atomic_json(state_path, state)
    print(f"Retranscribing: previous Scribe transcript moved to {target}", flush=True)


def run_locked(args, source, work_dir):
    identity = {"source": str(source), "sha256": sha256(source), "language": args.language,
                "base_url": args.base_url.rstrip("/"), "adapter_version": 1,
                "audio_format": "pcm_s16le/16000/mono"}
    state_path = work_dir / "scribe-state.json"
    wav_path = work_dir / "audio.wav"
    output_path = work_dir / "whisper-json-output" / "audio.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ScribeError(
                f"Cannot resume this saved run: the original recording or the Scribe settings no longer match. "
                f"Previous results are preserved in {work_dir}. Restore the original recording/settings to resume, "
                "or copy the recording into a separate folder and select that folder in the WebUI to start a new Scribe run. "
                "A new run may upload the recording again.")
    else:
        if any(p.name != ".lock" for p in work_dir.iterdir()):
            raise ScribeError("Work directory is not empty and has no prototype state. Choose a new directory.")
        state = {"identity": identity}
        atomic_json(state_path, state)
    if args.job_id:
        if state.get("job_id") and state["job_id"] != args.job_id:
            raise ScribeError("The saved job differs from --job-id; choose a new work directory.")
        state["job_id"] = args.job_id
        state["adopted_job"] = True
        atomic_json(state_path, state)

    if not state.get("wav_sha256"):
        prepare_audio(source, wav_path)
        state["wav_sha256"] = sha256(wav_path)
        state['local_pcm_sha256'] = pcm_checksum(wav_path)
        atomic_json(state_path, state)
    elif not wav_path.exists():
        restore_working_audio(source, wav_path, state, state_path)
    else:
        expected = state.get('local_wav_sha256', state['wav_sha256'])
        current = sha256(wav_path)
        if current != expected:
            audio_checksum = pcm_checksum(wav_path)
            same_audio = (audio_checksum is not None and audio_checksum == state.get('local_pcm_sha256'))
            if not same_audio and not state.get('local_pcm_sha256'):
                same_audio = checksum_without_jpr_metadata(wav_path) == expected
            if same_audio:
                state['local_pcm_sha256'] = audio_checksum
                state['local_wav_sha256'] = current
                atomic_json(state_path, state)
                print("Audio verified; only WAV metadata changed.", flush=True)
            else:
                restore_working_audio(source, wav_path, state, state_path)
        elif not state.get('local_pcm_sha256'):
            audio_checksum = pcm_checksum(wav_path)
            if audio_checksum is not None:
                state['local_pcm_sha256'] = audio_checksum
                atomic_json(state_path, state)

    new_transcript = False
    if getattr(args, "retranscribe", False):
        if output_path.exists():
            supersede_remote_result(work_dir, state, state_path, output_path)
        elif state.get("job_id") or state.get("upload_started"):
            raise ScribeError("A Scribe job for this recording has not finished. Run without --retranscribe to resume it.")
    if output_path.exists():
        if sha256(output_path) != state.get("output_sha256"):
            raise ScribeError(
                "The saved transcript has changed and must be restored before this run can continue.")
        print(f"Reusing saved Scribe transcription: {output_path}", flush=True)
    else:
        password = (os.getenv("SUNET_SCRIBE_ENCRYPT_PWD", "").strip()
                    or os.getenv("SWHISPER_SCRIBE_PASSWORD", "").strip()
                    or os.getenv("SCRIBE_ENCRYPTION_PASSWORD", "").strip())
        if not password and sys.stdin.isatty():
            password = getpass.getpass("Scribe encryption password (hidden): ")
        client = library_client(args, password)
        # Without a scribe-api client, the run signs in through BrowserOS Neo.
        provider = None
        if client is None:
            if args.base_url.rstrip("/") != "https://scribe.sunet.se":
                raise ScribeError("BrowserOS authentication is restricted to https://scribe.sunet.se.")
            from scribe_browser_auth import BrowserTokenProvider
            provider = BrowserTokenProvider(interactive=sys.stdin.isatty())
        try:
            if client is None:
                client = ScribeClient("", password, args.base_url, token_provider=provider,
                                      poll_interval=args.poll_interval, wait_seconds=args.wait_seconds)
            transcript, words = client.transcribe(wav_path, state, state_path, args.language)
        finally:
            if provider:
                provider.close()
            if hasattr(client, "close"):
                client.close()
        # Outside whisper-json-output so diarize.py never treats raw data as input.
        atomic_json(work_dir / "scribe-transcript.json", transcript)
        atomic_json(work_dir / "scribe-words.json", words)
        result = normalize_result(transcript, words, language="sv")
        result["scribe"] = {
            "job_id": state["job_id"], "requested_language": args.language,
            "source_sha256": identity["sha256"], "audio_sha256": state["wav_sha256"],
            "base_url": identity["base_url"], "adapter_version": 1,
            "speaker_labels_used": False, "processing_stage": "remote_transcription",
            "timestamp_offset": 0.0, "adopted_job": state.get("adopted_job", False),
            "transport": getattr(client, "transport", "swhisper-http"),
        }
        atomic_json(output_path, result)
        state["output_sha256"] = sha256(output_path)
        atomic_json(state_path, state)
        new_transcript = True
        print(f"Saved timed transcription: {output_path}", flush=True)
    if not args.transcription_only:
        # Diarization is aligned to the transcript, so a new one needs a new run.
        local_diarization(source, work_dir, output_path.name, getattr(args, "diarization_model", None),
                          force=new_transcript or getattr(args, "rediarize", False),
                          strict_scribe_nemotron=getattr(args, "strict_scribe_nemotron", False))
    print(f"Prototype results: {work_dir}", flush=True)


def main(argv=None):
    from path_settings import get_path_settings
    get_path_settings()  # Load .swhisper.env before constructing local config objects.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="One recording to upload to SUNET")
    parser.add_argument("--work-dir", type=Path, help="Isolated run directory; reused on restart")
    parser.add_argument("--transcription-only", action="store_true", help="Save JSON without local diarization")
    parser.add_argument("--language", choices=["Swedish (verbatim)", "Swedish"],
                        default="Swedish (verbatim)", help="Scribe language selector")
    parser.add_argument("--auth", choices=["auto", "oidc", "browseros"], default="auto",
                        help="auto (default): scribe-api with OIDC client credentials, falling back to "
                             "BrowserOS Neo; oidc: scribe-api only; browseros: BrowserOS Neo only")
    parser.add_argument("--diarization-model", choices=sorted(DIARIZATION_MODELS),
                        help="Local diarization model for this run (default: SWHISPER_PIPELINE_MODEL, else community-1)")
    parser.add_argument("--strict-scribe-nemotron", action="store_true",
                        help="Require Nemotron and disable non-Scribe backfill, overriding sidecars")
    parser.add_argument("--retranscribe", action="store_true",
                        help="Upload again for a new SUNET transcript; the saved one moves to superseded/")
    parser.add_argument("--rediarize", action="store_true",
                        help="Run local diarization again although a result from this model exists")
    parser.add_argument("--job-id", help="Adopt an existing job for this exact audio and language; verify these in Scribe")
    parser.add_argument("--base-url", default="https://scribe.sunet.se", help="Scribe HTTPS origin")
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument("--wait-seconds", type=float, default=86400)
    args = parser.parse_args(argv)
    if args.retranscribe and args.job_id:
        parser.error("--retranscribe starts a new job and cannot be combined with --job-id")
    try:
        run(args)
    except (ScribeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped locally. Any remote job continues; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
