"""Exercise the emitted JavaScript against Neo's page-handle SDK contract."""
import base64
import json
import shutil
import subprocess
import time

import pytest

from scribe_api import ScribeError
from scribe_browser_auth import BrowserTokenProvider


NODE = shutil.which('node')
SDK = r'''
const fs = require('fs');
const {code, token, loggedIn, dialogReady} = JSON.parse(fs.readFileSync(0, 'utf8'));
let dialogOpen = false;
const dialog = {
  innerText: 'User token',
  querySelectorAll: selector => selector === 'input,textarea'
    ? [{value: token}] : [{innerText: 'CLOSE', click: () => {dialogOpen = false;}}]
};
const document = {
  querySelectorAll: selector => selector === 'div'
    ? [{children: [], textContent: 'Username', nextElementSibling: {click: () => {dialogOpen = true;}}}]
    : dialogOpen ? [dialog] : []
};
const page = {
  id: 17,
  goto: async () => ({status: 200}),
  snapshot: async () => ({url: loggedIn ? 'https://scribe.sunet.se/user' : 'https://login.example/'}),
  read: async () => loggedIn ? 'Username\nUser settings' : 'Log in',
  waitForText: async () => ({matched: dialogReady && dialogOpen}),
  evaluateInPage: async code => ({value: new Function('document', 'getComputedStyle', code)(document, () => ({cursor: 'pointer'}))}),
  close: async () => true,
};
const browser = {open: async () => page, page: id => {
  if (id !== 17) throw new Error('Invalid page id');
  return page;
}};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
new AsyncFunction('browser', code)(browser).then(value => {
  process.stdout.write(JSON.stringify({ok: true, value}));
}).catch(() => process.stdout.write(JSON.stringify({ok: false})));
'''


class ContractProvider(BrowserTokenProvider):
    def __init__(self, *, logged_in=True, dialog_ready=True):
        super().__init__(interactive=False)
        payload = base64.urlsafe_b64encode(json.dumps({'exp': time.time() + 300}).encode()).decode().rstrip('=')
        self.expected_token = 'eyJheader.' + payload + '.signature'
        self.logged_in = logged_in
        self.dialog_ready = dialog_ready

    def rpc(self, method, params):
        if method == 'initialize' or params.get('name') == 'name_session':
            return {}
        assert method == 'tools/call' and params['name'] == 'run'
        result = subprocess.run([NODE, '-e', SDK], input=json.dumps({
            'code': params['arguments']['code'], 'token': self.expected_token,
            'loggedIn': self.logged_in, 'dialogReady': self.dialog_ready,
        }), capture_output=True, text=True, check=True, timeout=10)
        return {'structuredContent': json.loads(result.stdout)}


@pytest.mark.skipif(not NODE, reason='Node is needed to execute the emitted browser scripts')
def test_current_neo_sdk_login_and_cleanup():
    p = ContractProvider()
    try:
        assert p.get_token() == p.expected_token
        assert p.page == 17
    finally:
        p.close()
    assert p.page is None


@pytest.mark.skipif(not NODE, reason='Node is needed to execute the emitted browser scripts')
def test_logged_out_still_requests_login():
    p = ContractProvider(logged_in=False)
    try:
        with pytest.raises(ScribeError, match='Log in'):
            p.get_token()
        assert p.token is None
    finally:
        p.close()


@pytest.mark.skipif(not NODE, reason='Node is needed to execute the emitted browser scripts')
def test_missing_token_dialog_refuses_credentials():
    p = ContractProvider(dialog_ready=False)
    try:
        with pytest.raises(ScribeError, match='No credential was accepted'):
            p.get_token()
        assert p.token is None
    finally:
        p.close()


def test_tool_failure_does_not_claim_user_is_logged_out(monkeypatch):
    p = BrowserTokenProvider(interactive=False)
    monkeypatch.setattr(p, 'rpc', lambda *args: {'isError': True})
    try:
        with pytest.raises(ScribeError, match='API compatibility'):
            p.call('run', {'code': ''})
    finally:
        p.close()
