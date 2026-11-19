"""Exercise offline login without network access or personal credentials."""

import base64
import json
from unittest.mock import Mock

import pytest
import requests
from scribe import signin
from scribe.errors import AuthError

import scribe_login as cli


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBE_TOKEN_FILE", str(tmp_path / "token.json"))
    monkeypatch.setenv("SCRIBE_OIDC_ISSUER", "https://identity.example/realms/test")
    monkeypatch.setenv("SCRIBE_OIDC_CLIENT_ID", "automations")
    monkeypatch.setattr(requests, "post", Mock(side_effect=AssertionError("Network is forbidden.")))


def jwt(**metadata):
    payload = base64.urlsafe_b64encode(json.dumps(metadata).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def response(body, ok=True):
    return Mock(ok=ok, status_code=200 if ok else 400, json=Mock(return_value=body))


def device_flow(monkeypatch, answers, lifetime=100):
    clock = [0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli.time, "sleep", sleep)
    start = response({"device_code": "private-device", "verification_uri": "https://identity.example/verify",
                      "user_code": "ABCD", "interval": 1, "expires_in": lifetime})
    post = Mock(side_effect=[start] + answers)
    monkeypatch.setattr(requests, "post", post)
    return post, sleeps


@pytest.mark.parametrize("kind", ["Offline", "Refresh"])
def test_polling_and_library_compatibility(monkeypatch, capsys, kind):
    token = jwt(typ=kind, iat=1700000000)
    post, sleeps = device_flow(monkeypatch, [response({"error": "authorization_pending"}, False),
                                           response({"error": "slow_down"}, False),
                                           response({"refresh_token": token})])
    cli.login()
    assert sleeps == [1, 1, 6]
    assert post.call_args_list[0].kwargs['data']['scope'] == 'openid offline_access'
    assert all(call.kwargs['allow_redirects'] is False for call in post.call_args_list)
    output = capsys.readouterr().out
    assert token not in output
    assert 'ABCD' in output
    assert ('Offline access was granted.' in output) == (kind == 'Offline')
    assert signin.load_login()['refresh_token'] == token
    assert signin.token_path().stat().st_mode & 0o777 == 0o600
    refresh = Mock(return_value={"access_token": "access", "expires_in": 60})
    monkeypatch.setattr(signin, 'post_token', refresh)
    assert signin.saved_login_token_source()().access_token == 'access'
    assert refresh.call_args.args[1]['refresh_token'] == token


@pytest.mark.parametrize('error,message', [('access_denied', 'denied'), ('expired_token', 'expired')])
def test_terminal_poll_errors(monkeypatch, error, message):
    device_flow(monkeypatch, [response({'error': error}, False)])
    with pytest.raises(AuthError, match=message):
        cli.login()
    assert signin.load_login() is None


def test_local_expiration(monkeypatch):
    post, _ = device_flow(monkeypatch, [], lifetime=1)
    with pytest.raises(AuthError, match='expired'):
        cli.login()
    assert post.call_count == 1


@pytest.mark.parametrize('metadata', [dict(typ='Offline', iat=1700000000),
                                     dict(typ='Offline', exp=0),
                                     dict(typ='Offline', exp=999999999999999999),
                                     dict(typ='Refresh', iat=1700000000, exp=1700030000)])
def test_status_metadata(capsys, metadata):
    token = jwt(**metadata)
    signin.save_login('https://identity.example', 'automations', token)
    cli.status()
    output = capsys.readouterr().out
    assert token not in output
    assert f"type: {metadata['typ']}" in output
    assert 'Issued at:' in output and 'Expires if unused' in output
    if metadata.get('iat'):
        assert '2023-11-14' in output


@pytest.mark.parametrize('token', ['not-a-jwt', jwt(typ='secret-claim', iat='secret-claim')])
def test_status_never_echoes_unrecognized_metadata(capsys, token):
    signin.save_login('https://identity.example', 'automations', token)
    cli.status()
    output = capsys.readouterr().out
    assert token not in output
    assert 'secret-claim' not in output
    assert 'Unknown' in output


def test_not_signed_in(capsys):
    cli.status()
    assert 'Not signed in' in capsys.readouterr().out


@pytest.mark.parametrize('result', [response({}), response({}, False), requests.ConnectionError('secret-token')])
def test_logout(monkeypatch, capsys, result):
    signin.save_login('https://identity.example', 'automations', 'secret-token')
    post = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(requests, 'post', post)
    success = cli.logout()
    assert success == (not isinstance(result, Exception) and result.ok)
    assert signin.load_login() is None
    assert post.call_args.args[0].endswith('/protocol/openid-connect/revoke')
    assert post.call_args.kwargs['data'] == {'token': 'secret-token',
                                           'token_type_hint': 'refresh_token', 'client_id': 'automations'}
    output = capsys.readouterr().out
    assert 'secret-token' not in output
    assert ('revocation failed' in output) == (not success)


def test_logout_without_login(capsys):
    assert cli.logout()
    assert 'No local login' in capsys.readouterr().out


def test_insecure_issuer_is_rejected(monkeypatch):
    monkeypatch.setenv('SCRIBE_OIDC_ISSUER', 'http://localhost/realm')
    with pytest.raises(AuthError, match='HTTPS'):
        cli.login()
    signin.save_login('http://localhost/realm', 'automations', 'secret-token')
    assert not cli.logout()
    assert signin.load_login() is None


def test_error_body_is_not_printed(monkeypatch, capsys):
    device_flow(monkeypatch, [response({'error': 'secret-token', 'error_description': 'secret-token'}, False)])
    assert cli.main(['login']) == 1
    assert 'secret-token' not in capsys.readouterr().out


@pytest.mark.parametrize('body', [{}, {'refresh_token': None}, [], {'error': 'private-value'}])
def test_invalid_success_preserves_existing_login(monkeypatch, body):
    signin.save_login('https://identity.example', 'automations', 'previous-token')
    device_flow(monkeypatch, [response(body)])
    with pytest.raises(AuthError):
        cli.login()
    assert signin.load_login()['refresh_token'] == 'previous-token'


def test_prompt_is_flushed_before_polling(monkeypatch):
    import builtins

    printer = Mock()
    monkeypatch.setattr(builtins, 'print', printer)
    post, _ = device_flow(monkeypatch, [response({'refresh_token': jwt(typ='Offline')})])
    original = post.side_effect
    answers = iter(original)

    def answer(*args, **kwargs):
        if args[0].endswith('/token'):
            assert printer.call_args.kwargs['flush'] is True
            assert 'ABCD' in printer.call_args.args[0]
        return next(answers)

    post.side_effect = answer
    cli.login()
