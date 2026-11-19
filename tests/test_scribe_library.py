"""scribe-api transport with a simulated HTTP session; no network."""
import json

import pytest

scribe = pytest.importorskip("scribe")

import scribe_library
from scribe_api import ScribeError
from scribe_library import LibraryScribeClient


@pytest.fixture(autouse=True)
def no_saved_login(tmp_path, monkeypatch):
    """Never read the developer's real ~/.config/scribe/token.json."""
    monkeypatch.setenv("SCRIBE_TOKEN_FILE", str(tmp_path / "no-login.json"))


def save_test_login(tmp_path, monkeypatch):
    path = tmp_path / "login.json"
    path.write_text(json.dumps({"issuer": "https://idp.example/realms/r", "client_id": "automations",
                                "refresh_token": "refresh"}))
    path.chmod(0o600)
    monkeypatch.setenv("SCRIBE_TOKEN_FILE", str(path))


class Response:
    def __init__(self, status, result=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = {}
        self.text = ""
        self._body = {"result": result}

    def json(self):
        return self._body


class Session:
    """Answers from a list of (method, path suffix, Response) and records each request."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        expected_method, suffix, response = self.answers.pop(0)
        assert method == expected_method and url.endswith(suffix), (method, url)
        return response

    def close(self):
        pass


def client(answers, password="pw"):
    library = scribe.ScribeClient("token", base_url="https://scribe.example/api/v1",
                                  encryption_password=password, requests_per_second=None)
    library._session = Session(answers)
    return LibraryScribeClient(password, poll_interval=0.001, client=library), library._session


def test_full_job_through_the_library(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    transcript = {"full_transcription": "Hej", "segments": [{"start": 0, "end": 1, "text": "Hej"}]}
    words = {"version": 1, "words": [{"t": "Hej", "s": 0.1, "e": 0.5}]}
    remote, session = client([
        ("POST", "/transcriber", Response(200, {"uuid": "job-1"})),
        ("GET", "/transcriber", Response(200, {"uuid": "job-1", "status": "uploaded"})),
        ("PUT", "/transcriber/job-1", Response(200, {})),
        ("GET", "/transcriber", Response(200, {"uuid": "job-1", "status": "completed"})),
        ("GET", "/transcriber/job-1/result/txt", Response(200, json.dumps(transcript))),
        ("GET", "/transcriber/job-1/words", Response(200, json.dumps(words))),
    ])
    state = {}
    assert remote.transcribe(wav, state, tmp_path / "state.json", "Swedish (verbatim)") == (transcript, words)
    assert state["job_id"] == "job-1" and state["queue_started"]
    queued = session.requests[2][2]["json"]
    # The worker's verbatim key is sent unchanged, which scribe-api's start() would refuse.
    assert queued == {"language": "Swedish (verbatim)", "speakers": 0, "output_format": "TXT",
                      "encryption_password": "pw"}


def test_plain_language_uses_the_library_start():
    remote, session = client([("PUT", "/transcriber/job-1", Response(200, {}))])
    remote.queue("job-1", "Swedish")
    assert session.requests[0][2]["json"]["language"] == "Swedish"


@pytest.mark.parametrize("language", ["Klingon", "Finnish (verbatim)"])
def test_languages_the_worker_cannot_run_are_refused_before_queueing(language):
    remote, session = client([])
    with pytest.raises(ScribeError):
        remote.queue("job-1", language)
    assert session.requests == []


def test_refused_upload_clears_the_marker_so_a_new_credential_can_retry(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    remote, _ = client([("POST", "/transcriber", Response(401))])
    state = {}
    with pytest.raises(ScribeError) as error:
        remote.transcribe(wav, state, tmp_path / "state.json", "Swedish")
    assert error.value.status == 401
    assert "upload_started" not in state


def test_quota_on_verbatim_queue_keeps_status():
    remote, _ = client([("PUT", "/transcriber/job-1", Response(403))])
    with pytest.raises(ScribeError) as error:
        remote.queue("job-1", "Swedish (verbatim)")
    assert error.value.status == 403


def test_results_need_the_encryption_password():
    remote, session = client([], password="")
    with pytest.raises(ScribeError, match="encryption password"):
        remote.results("job-1")
    assert session.requests == []


def test_available_needs_all_oidc_variables(monkeypatch):
    for name in scribe_library.OIDC_VARIABLES:
        monkeypatch.setenv(name, "x")
    assert scribe_library.available()
    monkeypatch.delenv("OIDC_CLIENT_SECRET")
    assert not scribe_library.available()


def run_cli(tmp_path, monkeypatch, auth, check_error=None, oidc=True, login=False, sidecar=None, extra=()):
    """Run transcribe_scribe up to the remote step; return which transport transcribed."""
    import transcribe_scribe
    import scribe_api
    import scribe_browser_auth

    for name in scribe_library.OIDC_VARIABLES:
        if oidc:
            monkeypatch.setenv(name, "x")
        else:
            monkeypatch.delenv(name, raising=False)
    if login:
        save_test_login(tmp_path, monkeypatch)
    monkeypatch.setenv("SUNET_SCRIBE_ENCRYPT_PWD", "pw")
    source = tmp_path / "interview.wav"
    source.write_bytes(b"original")
    if sidecar:
        source.with_suffix(".swhisper").write_text(sidecar)
    used = []
    languages = []

    def remote(self, wav, state, state_path, language):
        used.append(type(self).__name__)
        languages.append(language)
        state["job_id"] = "simulated"
        return ({"full_transcription": "Hej", "segments": [{"start": 0, "end": 1, "text": "Hej"}]},
                {"version": 1, "words": [{"t": "Hej", "s": 0.1, "e": 0.5}]})

    def check(self):
        if check_error:
            raise check_error

    class Browser:
        def __init__(self, interactive):
            used.append("browser")

        def get_token(self, force=False):
            return "token"

        def close(self):
            pass

    monkeypatch.setattr(transcribe_scribe, "prepare_audio", lambda src, dst: dst.write_bytes(b"staged"))
    monkeypatch.setattr(LibraryScribeClient, "check", check)
    monkeypatch.setattr(LibraryScribeClient, "transcribe", remote)
    monkeypatch.setattr(scribe_api.ScribeClient, "transcribe", remote)
    monkeypatch.setattr(scribe_browser_auth, "BrowserTokenProvider", Browser)
    code = transcribe_scribe.main([str(source), "--work-dir", str(tmp_path / "run"),
                                   "--transcription-only", "--auth", auth, *extra])
    output = tmp_path / "run/whisper-json-output/audio.json"
    saved = json.loads(output.read_text()) if output.exists() else None
    run_cli.last = dict(languages=languages, saved=saved)
    return code, used, saved["scribe"]["transport"] if saved else None


def test_auto_uses_scribe_api_when_oidc_is_configured(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "auto") == (0, ["LibraryScribeClient"], "scribe-api")


def test_auto_falls_back_to_browseros_without_oidc(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "auto", oidc=False) == (0, ["browser", "ScribeClient"], "swhisper-http")


def test_auto_falls_back_to_browseros_when_oidc_is_refused(tmp_path, monkeypatch):
    refused = ScribeError("Scribe returned HTTP 401.", status=401)
    assert run_cli(tmp_path, monkeypatch, "auto", check_error=refused) == (0, ["browser", "ScribeClient"], "swhisper-http")


def test_oidc_mode_has_no_fallback(tmp_path, monkeypatch):
    refused = ScribeError("Scribe returned HTTP 401.", status=401)
    assert run_cli(tmp_path, monkeypatch, "oidc", check_error=refused) == (1, [], None)


def test_auto_uses_the_saved_login_without_oidc(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "auto", oidc=False, login=True) == (0, ["LibraryScribeClient"], "scribe-api")


def test_saved_login_comes_before_client_credentials(tmp_path, monkeypatch):
    for name in scribe_library.OIDC_VARIABLES:
        monkeypatch.setenv(name, "x")
    assert scribe_library.sign_in_method() == "client-credentials"
    save_test_login(tmp_path, monkeypatch)
    assert scribe_library.sign_in_method() == "saved-login"
    remote = LibraryScribeClient("pw")
    assert remote.sign_in == "saved-login"
    remote.close()


def test_no_login_at_all_names_the_login_command(monkeypatch):
    for name in scribe_library.OIDC_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert not scribe_library.available()
    with pytest.raises(ScribeError, match=r"python scribe_login\.py login"):
        LibraryScribeClient("pw")


def test_an_expired_login_is_reported_and_counts_as_refused(tmp_path):
    def expired():
        raise scribe.AuthError("Your login has expired. Run `python -m scribe login` to sign in.")
    library = scribe.ScribeClient(expired, requests_per_second=None)
    remote = LibraryScribeClient("pw", client=library)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    state = {}
    with pytest.raises(ScribeError, match="Your login has expired") as error:
        remote.transcribe(wav, state, tmp_path / "state.json", "Swedish")
    # No job can exist, so a rerun after signing in may upload.
    assert error.value.status == 401
    assert "python scribe_login.py login" in str(error.value)
    assert "python -m scribe login" not in str(error.value)
    assert "upload_started" not in state


def test_scribe_uses_the_recording_language_like_local_whisper(tmp_path, monkeypatch):
    monkeypatch.setenv("SWHISPER_LANGUAGE", "sv")
    assert run_cli(tmp_path, monkeypatch, "auto", sidecar="SWHISPER_LANGUAGE=en\n")[0] == 0
    assert run_cli.last["languages"] == ["English (verbatim)"]
    assert run_cli.last["saved"]["language"] == "en"
    assert run_cli.last["saved"]["scribe"]["requested_language"] == "English (verbatim)"


def test_standard_mode_and_explicit_language(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "auto", extra=("--language", "de", "--mode", "standard"))[0] == 0
    assert run_cli.last["languages"] == ["German"]
    assert run_cli.last["saved"]["language"] == "de"


def test_a_language_scribe_cannot_run_stops_before_any_upload(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "auto", sidecar="SWHISPER_LANGUAGE=ja\n") == (1, [], None)
