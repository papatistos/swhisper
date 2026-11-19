"""SUNET Scribe transport through the official scribe-api client.

https://github.com/SUNET/scribe-api authenticates with an OIDC
client-credentials grant and renews the token itself, so no browser is
involved. The job loop, state file and normalization stay in scribe_api.
"""

import os

from scribe_api import MISSING_PASSWORD, RemoteJobs, ScribeError, parse_payload

VERBATIM = " (verbatim)"
# Languages with a verbatim model in scribe-worker utils/settings.py.
VERBATIM_LANGUAGES = ("Swedish", "English", "Norwegian")
OIDC_VARIABLES = ("OIDC_TOKEN_ENDPOINT", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET")


def oidc_credentials():
    """The OIDC client credentials from the environment, or None if any is unset."""
    values = [os.getenv(name, "").strip() for name in OIDC_VARIABLES]
    return tuple(values) if all(values) else None


def available():
    """True when scribe-api is installed and OIDC client credentials are set."""
    try:
        import scribe  # noqa: F401
    except ImportError:
        return False
    return oidc_credentials() is not None


class LibraryScribeClient(RemoteJobs):
    """Scribe jobs through scribe-api; the same interface as scribe_api.ScribeClient."""

    transport = "scribe-api"

    def __init__(self, password, base_url="https://scribe.sunet.se", poll_interval=10,
                 wait_seconds=86400, client=None):
        if poll_interval <= 0 or wait_seconds <= 0:
            raise ScribeError("Polling interval and maximum wait must be positive.")
        if client is None:
            try:
                import scribe
            except ImportError:
                raise ScribeError("scribe-api is not installed. Install it with "
                                  "pip install git+https://github.com/SUNET/scribe-api") from None
            credentials = oidc_credentials()
            if credentials is None:
                raise ScribeError("Set " + ", ".join(OIDC_VARIABLES) + " to use scribe-api.")
            client = scribe.ScribeClient(
                token=lambda: scribe.get_token(*credentials),
                base_url=base_url.rstrip("/") + "/api/v1",
                encryption_password=password or None,
            )
        self.client = client
        self.password = password
        self.poll_interval = poll_interval
        self.wait_seconds = wait_seconds

    def close(self):
        self.client.close()

    def _call(self, method, *args, **kwargs):
        """Run one scribe-api call, translating its errors without echoing response bodies."""
        import scribe
        try:
            return method(*args, **kwargs)
        except scribe.QuotaExceeded:
            raise ScribeError("Scribe returned HTTP 403: the transcription quota is used up. "
                              "Job state was preserved.", status=403) from None
        except scribe.AuthError as exc:
            # A token that cannot be fetched is refused before any job exists.
            raise ScribeError(f"Scribe returned HTTP {exc.status or 401}. Check the OIDC client "
                              "credentials and API access, then rerun to resume.", status=exc.status or 401) from None
        except scribe.NotFound:
            raise ScribeError("Scribe returned HTTP 404: the job does not exist or is not yours.",
                              status=404) from None
        except scribe.RateLimited:
            raise ScribeError("Scribe kept returning HTTP 429. Job state was preserved; rerun later.",
                              status=429) from None
        except scribe.ScribeError as exc:
            if exc.status:
                raise ScribeError(f"Scribe returned HTTP {exc.status}; job state was preserved.",
                                  status=exc.status) from None
            raise ScribeError("Scribe connection failed. Job state was preserved; rerun to resume.") from None

    def check(self):
        """Fetch a token and call GET /me; raises ScribeError if refused."""
        self._call(self.client.check)

    def upload(self, wav_path):
        job_id = self._call(self.client.upload, str(wav_path))
        if not isinstance(job_id, str) or not job_id:
            raise ScribeError("Upload response did not contain a job UUID; inspect Scribe before retrying.")
        return job_id

    def status(self, job_id):
        status = self._call(self.client.job, job_id).get("status")
        if not isinstance(status, str):
            raise ScribeError("Unexpected Scribe job status response.")
        return status

    def queue(self, job_id, language):
        import scribe
        base = language.removesuffix(VERBATIM)
        try:
            scribe.resolve_language(base)
        except ValueError:
            raise ScribeError(f"Scribe does not support the language {language!r}.") from None
        if base == language:
            self._call(self.client.start, job_id, language=language, speakers=0, output_format="TXT")
            return
        # An unknown language leaves the job in_progress forever, so check here.
        if base not in VERBATIM_LANGUAGES:
            raise ScribeError(f"Scribe has no verbatim model for {base}.")
        # scribe-api 1.0.0 rejects "<language> (verbatim)" although the worker
        # accepts it, so send the same request through the client's request path.
        self._call(self.client._request, "PUT", f"/transcriber/{job_id}", json={
            "language": language, "speakers": 0, "output_format": "TXT",
            "encryption_password": self.password,
        })

    def results(self, job_id):
        if not self.password:
            raise ScribeError(MISSING_PASSWORD)
        return (parse_payload(self._call(self.client.result, job_id, "txt")),
                parse_payload(self._call(self.client.words, job_id)))
