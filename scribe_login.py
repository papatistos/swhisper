"""Save a Scribe device login with a request for offline access."""

import argparse
import base64
import json
import math
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from scribe import signin
from scribe.auth import require_https
from scribe.errors import AuthError


def endpoint(issuer, name):
    """Build a credential endpoint and require HTTPS."""
    url = f"{issuer.rstrip('/')}/protocol/openid-connect/{name}"
    # The library permits HTTP on localhost; this command always requires HTTPS.
    if urlsplit(url).scheme != "https":
        raise AuthError("Credentials require an HTTPS issuer.")
    require_https(url)
    return url


def claims(token):
    """Read JWT metadata without checking its signature."""
    try:
        payload = token.split(".")[1]
        value = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return value if isinstance(value, dict) else {}
    except (ValueError, IndexError, TypeError, AttributeError):
        return {}


def token_kind(token):
    """Return a safe description of the refresh token type."""
    kind = claims(token).get("typ")
    return kind if kind in ("Offline", "Refresh") else "Unknown"


def post(url, data, timeout):
    """Post credentials without following redirects or exposing response bodies."""
    try:
        return requests.post(url, data=data, timeout=timeout, allow_redirects=False)
    except requests.RequestException:
        raise AuthError("The authentication request failed.") from None


def login(timeout=10):
    """Request offline access through the device flow and save the login."""
    issuer, client_id = signin.oidc_settings()
    response = post(endpoint(issuer, "auth/device"),
                    {"client_id": client_id, "scope": "openid offline_access"}, timeout)
    if not 200 <= response.status_code < 300:
        raise AuthError(f"Could not start login (HTTP {response.status_code}).")
    try:
        device = response.json()
        device_code = device["device_code"]
        url = device.get("verification_uri_complete") or device["verification_uri"]
        code = device["user_code"]
        interval = float(device.get("interval", 5))
        lifetime = float(device.get("expires_in", 600))
        if not math.isfinite(interval) or interval <= 0 or not math.isfinite(lifetime) or lifetime <= 0:
            raise ValueError
        if not all(isinstance(v, str) and v for v in (device_code, url, code)):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError):
        raise AuthError("The provider returned an invalid device authorization response.") from None
    deadline = time.monotonic() + lifetime
    print(f"Open {url}\nand check that the code is: {code}", flush=True)
    token_url = endpoint(issuer, "token")
    while time.monotonic() < deadline:
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
        if time.monotonic() >= deadline:
            break
        response = post(token_url, {"grant_type": signin.DEVICE_GRANT,
                                   "device_code": device_code, "client_id": client_id}, timeout)
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, TypeError):
            raise AuthError("The provider returned an invalid token response.") from None
        if 200 <= response.status_code < 300:
            refresh = body.get("refresh_token")
            if not isinstance(refresh, str) or not refresh:
                raise AuthError("The provider sent no refresh token.")
            signin.save_login(issuer, client_id, refresh)
            if token_kind(refresh) == "Offline":
                print("Login saved. Offline access was granted.")
            else:
                print("Login saved and usable, but offline access was not confirmed "
                      "(the refresh token type is not Offline). You may need to sign in again daily.")
            return
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "access_denied":
            raise AuthError("Login was denied.")
        if error == "expired_token":
            break
        raise AuthError(f"Login failed (HTTP {response.status_code}).")
    raise AuthError("Login timed out; the code expired. Try again.")


def timestamp(value):
    """Format a JWT timestamp without exposing unrecognized claim values."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "Unknown"
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return "Unknown (timestamp outside the supported range)"


def status():
    """Describe the saved login using unverified local metadata only."""
    saved = signin.load_login()
    if saved is None:
        print("Not signed in (no readable saved login).")
        return
    token = saved["refresh_token"]
    metadata = claims(token)
    print("Signed in (saved login; server validity has not been checked).")
    print(f"Refresh token type: {token_kind(token)}.")
    print(f"Issued at: {timestamp(metadata.get('iat'))}.")
    expiry = metadata.get("exp")
    print(f"Expires if unused (JWT exp): {timestamp(expiry) if expiry else 'No expiry recorded'}.")
    print("JWT claims are unverified. Server idle and maximum session limits may expire access earlier.")


def logout(timeout=10):
    """Revoke the saved refresh token and always remove the local login."""
    saved = signin.load_login()
    failed = False
    try:
        if saved:
            response = post(endpoint(saved["issuer"], "revoke"),
                            {"token": saved["refresh_token"], "token_type_hint": "refresh_token",
                             "client_id": saved["client_id"]}, timeout)
            failed = not 200 <= response.status_code < 300
    except AuthError:
        failed = True
    finally:
        removed = signin.logout()
    print("Local login deleted." if removed else "No local login was saved.")
    if failed:
        print("Server revocation failed. The remote session may still be active.")
    elif saved:
        print("Server revocation succeeded.")
    return not failed


def main(argv=None):
    """Run a login, status, or logout command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("login", "status", "logout"))
    args = parser.parse_args(argv)
    try:
        result = globals()[args.command]()
        return 1 if result is False else 0
    except AuthError as exc:
        print(str(exc))
        return 1
    except OSError:
        print("Could not access the saved login file.")
        return 1
    except KeyboardInterrupt:
        print("Login cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
