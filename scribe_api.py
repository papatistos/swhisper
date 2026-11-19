"""Small SUNET Scribe client and strict word-timing adapter.

Contract reviewed against SUNET's 2026-09-08 OpenAPI and public worker.
No local speech models are imported by this module.
"""

import json
import math
import os
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit


class ScribeError(RuntimeError):
    """An actionable remote-processing or result-validation error.

    ``status`` is the HTTP status when the service refused a request.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_payload(value):
    """A transcript or word-data result as a dict; the API may JSON-encode it."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ScribeError(
                "Result is empty, encrypted, or invalid JSON. Check the Scribe encryption "
                "password and word-data availability. No transcript was accepted."
            ) from None
    if not isinstance(value, dict):
        raise ScribeError("Unexpected Scribe result payload.")
    return value


MISSING_PASSWORD = (
    "Scribe job completed, but no encryption password is configured. "
    "Set SWHISPER_SCRIBE_PASSWORD to the Scribe account's encryption password and rerun."
)


class RemoteJobs:
    """Resumable upload, queue and poll of one Scribe job.

    Subclasses provide the transport: ``upload``, ``status``, ``queue`` and
    ``results``, plus ``poll_interval`` and ``wait_seconds``.
    """

    def transcribe(self, wav_path, state, state_path, language):
        if not state.get("job_id"):
            if state.get("upload_started"):
                raise ScribeError(
                    "A previous upload has an unknown outcome. Inspect Scribe and use --job-id "
                    "to adopt that job; do not blindly submit the recording again."
                )
            state["upload_started"] = True
            atomic_json(state_path, state)
            try:
                state["job_id"] = self.upload(wav_path)
            except ScribeError as exc:
                # Auth/quota rejection occurs before a job is created. Clear
                # this marker so a refreshed credential can retry safely.
                if exc.status in (401, 403):
                    state.pop("upload_started", None)
                    atomic_json(state_path, state)
                raise
            atomic_json(state_path, state)
        job_id = state["job_id"]
        print(f"Scribe job: {job_id}", flush=True)
        deadline = time.monotonic() + self.wait_seconds
        previous = None
        while True:
            status = self.status(job_id)
            if status != previous:
                print(f"Scribe status: {status}", flush=True)
                previous = status
            if status == "completed":
                break
            if status in ("failed", "deleted"):
                raise ScribeError(f"Scribe job {status}; inspect it in Scribe.")
            if status == "uploaded":
                if state.get("queue_started"):
                    raise ScribeError(
                        "Queue request has an uncertain outcome and job still says uploaded. "
                        "Check or queue it in Scribe, then rerun."
                    )
                state["queue_started"] = True
                atomic_json(state_path, state)
                self.queue(job_id, language)
            elif status not in ("pending", "in_progress", "uploading"):
                raise ScribeError(f"Unrecognized Scribe job status: {status!r}.")
            if time.monotonic() >= deadline:
                raise ScribeError("Stopped waiting for Scribe. Rerun to resume the same job.")
            time.sleep(self.poll_interval)
        return self.results(job_id)


class ScribeClient(RemoteJobs):
    """Hand-written HTTP transport, kept as the fallback to scribe_library."""

    def __init__(self, token, password, base_url="https://scribe.sunet.se",
                 session=None, poll_interval=10, wait_seconds=86400, token_provider=None):
        self.token_provider = token_provider
        if token_provider:
            token = token_provider.get_token()
        if not token:
            raise ScribeError("Set SUNET_SCRIBE_API_KEY or SWHISPER_SCRIBE_TOKEN to a valid bearer credential.")
        # The current SUNET credential is an OIDC JWT. Give a useful local
        # error for an expired token before starting an upload. Opaque bearer
        # credentials remain supported and are checked by the server.
        parts = token.split(".")
        if len(parts) == 3:
            try:
                payload = json.loads(__import__("base64").urlsafe_b64decode(
                    parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                ))
                if isinstance(payload.get("exp"), (int, float)) and payload["exp"] <= time.time():
                    raise ScribeError("The SUNET bearer credential has expired. Obtain a fresh one and rerun.")
            except ScribeError:
                raise
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                pass
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ScribeError("Scribe base URL must be an HTTPS origin without credentials or a path.")
        if poll_interval <= 0 or wait_seconds <= 0:
            raise ScribeError("Polling interval and maximum wait must be positive.")
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.token = token
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.wait_seconds = wait_seconds

    def _request(self, method, path, **kwargs):
        import requests
        headers = dict(kwargs.pop("headers", {}))
        if self.token_provider:
            self.token = self.token_provider.get_token()
        headers["Authorization"] = "Bearer " + self.token
        try:
            response = self.session.request(
                method, self.base_url + "/api/v1/transcriber" + path,
                headers=headers, timeout=(30, 600), allow_redirects=False, **kwargs
            )
        except requests.RequestException:
            # Do not echo request bodies, credentials, or remote transcript content.
            raise ScribeError("Scribe connection failed. Job state was preserved; rerun to resume.") from None
        # Retry only read-only requests. Never replay an upload or queue mutation.
        if response.status_code == 401 and method == "GET" and self.token_provider:
            self.token = self.token_provider.get_token(force=True)
            headers["Authorization"] = "Bearer " + self.token
            try:
                response = self.session.request(
                    method, self.base_url + "/api/v1/transcriber" + path,
                    headers=headers, timeout=(30, 600), allow_redirects=False, **kwargs)
            except requests.RequestException:
                raise ScribeError("Scribe connection failed. Rerun to resume.") from None
        if response.status_code in (401, 403):
            raise ScribeError(
                f"Scribe returned HTTP {response.status_code}. Check token, API access and quota; "
                "refresh credentials locally and rerun to resume.", status=response.status_code
            )
        if not 200 <= response.status_code < 300:
            raise ScribeError(f"Scribe returned HTTP {response.status_code}; job state was preserved.",
                              status=response.status_code)
        try:
            envelope = response.json()
        except ValueError:
            raise ScribeError("Scribe returned a non-JSON response.") from None
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise ScribeError("Unexpected Scribe response envelope.")
        result = envelope["result"]
        if isinstance(result, dict) and result.get("error"):
            raise ScribeError("Scribe reported an error. Inspect the job in Scribe; state was preserved.")
        return result

    def upload(self, wav_path):
        # requests' files= buffers the entire multipart upload. Stream instead.
        boundary = uuid.uuid4().hex
        prefix = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode()
        suffix = f"\r\n--{boundary}--\r\n".encode()
        size = Path(wav_path).stat().st_size

        class Multipart:
            def __len__(self):
                return len(prefix) + size + len(suffix)

            def __iter__(self):
                yield prefix
                with open(wav_path, "rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        yield chunk
                yield suffix

        result = self._request("POST", "", data=Multipart(), headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        })
        if not isinstance(result, dict) or not isinstance(result.get("uuid"), str):
            raise ScribeError("Upload response did not contain a job UUID; inspect Scribe before retrying.")
        return result["uuid"]

    def status(self, job_id):
        result = self._request("GET", "", params={"job_id": job_id},
                               json={"encryption_password": self.password})
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            raise ScribeError("Unexpected Scribe job status response.")
        return result["status"]

    def queue(self, job_id, language):
        self._request("PUT", "/" + quote(job_id, safe=""), json={
            "language": language, "speakers": 0, "output_format": "txt",
            "encryption_password": self.password,
        })

    def result(self, job_id, suffix):
        if not self.password:
            raise ScribeError(MISSING_PASSWORD)
        return parse_payload(self._request("GET", "/" + quote(job_id, safe="") + suffix,
                                           json={"encryption_password": self.password}))

    def results(self, job_id):
        return self.result(job_id, "/result/txt"), self.result(job_id, "/words")


def normalize_result(transcript, word_data, language="sv"):
    """Keep every word and original timing; discard all remote speaker labels.

    Chunk boundaries are only grouping metadata. Local pyannote assigns speakers
    independently to each word. Refuse missing/mismatched words rather than
    quietly exporting a less complete transcript.
    """
    if word_data.get("version") != 1:
        raise ScribeError("Unsupported or missing Scribe word-data version.")
    raw_words = word_data.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise ScribeError("Scribe returned no word timings; detailed transcription cannot proceed.")
    chunks = transcript.get("segments")
    if not isinstance(chunks, list) or not chunks:
        raise ScribeError("Scribe returned no transcript segments.")

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ScribeError("Invalid numeric value in Scribe word/segment data.")
        return float(value)

    def interval(item, start_key, end_key):
        start, end = number(item.get(start_key)), number(item.get(end_key))
        if start < 0 or end < start:
            raise ScribeError("Invalid Scribe timestamp interval.")
        return start, end

    segments = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise ScribeError("Invalid Scribe transcript segment.")
        start, end = interval(chunk, "start", "end")
        segments.append({"start": start, "end": end, "text": chunk["text"], "words": []})
    if any(a["start"] > b["start"] for a, b in zip(segments, segments[1:])):
        raise ScribeError("Scribe transcript segments are not chronological.")
    words = []
    for raw in raw_words:
        if not isinstance(raw, dict) or not isinstance(raw.get("t"), str) or not raw["t"].strip():
            raise ScribeError("Invalid Scribe word text.")
        start, end = interval(raw, "s", "e")
        word = {"text": raw["t"], "start": start, "end": end}
        if "c" in raw:
            confidence = number(raw["c"])
            if not 0 <= confidence <= 1:
                raise ScribeError("Invalid Scribe word confidence.")
            word["confidence"] = confidence
        words.append(word)
    if any(a["start"] > b["start"] for a, b in zip(words, words[1:])):
        raise ScribeError("Scribe words are not chronological.")
    compact = lambda text: "".join(text.split())
    combined = " ".join(word["text"] for word in words)
    original = transcript.get("full_transcription")
    if not isinstance(original, str) or compact(original) != compact(combined):
        raise ScribeError("Transcript text and timed words differ. Inspect raw results; no words were guessed.")
    if compact(" ".join(s["text"] for s in segments)) != compact(combined):
        raise ScribeError("Transcript segments and timed words differ; no transcript was accepted.")
    index = 0
    for word in words:
        middle = (word["start"] + word["end"]) / 2
        while index + 1 < len(segments) and middle >= segments[index + 1]["start"]:
            index += 1
        segments[index]["words"].append(word)
    segments = [s for s in segments if s["words"]]
    for index, segment in enumerate(segments):
        segment.update(id=index, text=" ".join(w["text"] for w in segment["words"]),
                       start=min(w["start"] for w in segment["words"]),
                       end=max(w["end"] for w in segment["words"]))
    return {"text": original, "language": language, "segments": segments}
