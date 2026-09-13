"""Obtain Scribe bearer credentials through its existing BrowserOS Neo login UI.

Tokens stay in memory. The only MCP origin allowed is IPv4 loopback.
"""
import base64
import json
import time
from urllib.parse import urlsplit

from scribe_api import ScribeError


def expiry(token):
    try:
        part = token.split('.')[1]
        return float(json.loads(base64.urlsafe_b64decode(part + '=' * (-len(part) % 4)))['exp'])
    except (ValueError, KeyError, IndexError, TypeError):
        raise ScribeError('Scribe UI did not return a JWT with an expiry.') from None


class BrowserTokenProvider:
    def __init__(self, endpoint='http://127.0.0.1:9010/mcp', interactive=True):
        import requests
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ScribeError('BrowserOS MCP endpoint must be on http://127.0.0.1.')
        self.endpoint = endpoint
        self.http = requests.Session()
        self.http.trust_env = False
        self.headers = {'Accept': 'application/json, text/event-stream', 'Content-Type': 'application/json'}
        self.counter = 0
        self.session = None
        self.page = None
        self.token = None
        self.interactive = interactive

    def rpc(self, method, params):
        import requests
        self.counter += 1
        try:
            r = self.http.post(self.endpoint, headers=self.headers, json={
                'jsonrpc': '2.0', 'id': self.counter, 'method': method, 'params': params}, timeout=45)
            r.raise_for_status()
            if r.headers.get('mcp-session-id'):
                self.headers['Mcp-Session-Id'] = r.headers['mcp-session-id']
            if 'text/event-stream' in r.headers.get('content-type', ''):
                messages = [json.loads(line[5:].strip()) for line in r.text.splitlines()
                            if line.startswith('data:') and line[5:].strip()]
                data = next(m for m in messages if m.get('id') == self.counter)
            else:
                data = r.json()
            if 'error' in data:
                raise ValueError()
            return data['result']
        except (requests.RequestException, ValueError, KeyError, StopIteration):
            raise ScribeError('Cannot communicate with BrowserOS Neo. Start it and check its MCP server connection.') from None

    def call(self, name, arguments):
        if self.session:
            arguments = dict(arguments, session=self.session)
        r = self.rpc('tools/call', {'name': name, 'arguments': arguments})
        self.session = r.get('_meta', {}).get('com.browseros.neo/session', self.session)
        if r.get('isError'):
            raise ScribeError('BrowserOS Neo could not inspect Scribe. Check its connection and login.')
        return r

    def run(self, code):
        r = self.call('run', {'code': code, 'timeout': 30000})
        result = r.get('structuredContent')
        if result is None:
            for item in r.get('content', []):
                if item.get('type') == 'text':
                    try:
                        candidate = json.loads(item['text'])
                        if isinstance(candidate, dict) and 'ok' in candidate:
                            result = candidate
                    except ValueError:
                        pass
        if not result or not result.get('ok'):
            raise ScribeError('BrowserOS Neo returned an unexpected script result. No credential was accepted.')
        return result.get('value')

    def connect(self):
        self.rpc('initialize', {'protocolVersion': '2025-03-26', 'capabilities': {},
                               'clientInfo': {'name': 'swhisper', 'version': '1'}})
        # Neo versions without explicit browser-session metadata use the MCP session.
        self.session = self.headers.get('Mcp-Session-Id')
        self.call('name_session', {'name': 'scribe authentication', 'category': 'dev-tools',
                                  'summary': 'Obtain current API credentials from the existing login UI'})
        self.page = self.run("return await browser.pages.newPage('https://scribe.sunet.se/user');")

    def close(self):
        if self.page is not None:
            try:
                self.run('await browser.pages.close(' + str(self.page) + '); return true;')
            except ScribeError:
                pass
            self.page = None
        self.http.close()

    def get_token(self, force=False):
        if self.token and not force and expiry(self.token) > time.time() + 45:
            return self.token
        if self.page is None:
            self.connect()
        # Reload lets the existing UI refresh its server-side authentication.
        result = self.run('''
const page = PAGE;
await browser.nav(page).goto('https://scribe.sunet.se/user');
try { await browser.wait(page, {for:'text', value:'Username', timeout:15000}); } catch (_) {}
const state = await browser.observe(page).snapshot();
if (!state.url.startsWith('https://scribe.sunet.se/')) return {loginRequired:true};
const text = await browser.read(page);
if (!text.includes('Username') || !text.includes('User settings')) return {loginRequired:true};
await browser.evaluate(page, {code: `
const label = Array.from(document.querySelectorAll('div')).find(e => e.children.length === 0 && e.textContent.trim() === 'Username');
const control = label?.nextElementSibling;
if (!control || getComputedStyle(control).cursor !== 'pointer') throw new Error('Username control missing');
control.click(); return true;`});
await browser.wait(page, {for:'text', value:'Your user token is used to authenticate API requests.'});
const token = await browser.evaluate(page, {code: `
const dialog = Array.from(document.querySelectorAll('[role="dialog"]')).find(e => e.innerText.includes('User token'));
const token = Array.from(dialog?.querySelectorAll('input,textarea') || []).map(e=>e.value).find(v=>/^eyJ[^.]+\\.[^.]+\\.[^.]+$/.test(v));
return token || null;`});
await browser.evaluate(page, {code: `
const dialog = Array.from(document.querySelectorAll('[role="dialog"]')).find(e=>e.innerText.includes('User token'));
Array.from(dialog?.querySelectorAll('button') || []).find(e=>e.innerText.trim().toUpperCase()==='CLOSE')?.click(); return true;`});
return {token:token.value};
'''.replace('PAGE', str(self.page)))
        if result.get('loginRequired'):
            message = 'Log in to https://scribe.sunet.se/ in BrowserOS Neo before continuing.'
            if not self.interactive:
                raise ScribeError(message + ' Then rerun the same command to resume.')
            input(message + ' Press Enter when logged in: ')
            # One retry, so a missing login cannot become an endless prompt loop.
            previous = self.interactive
            self.interactive = False
            try:
                return self.get_token(force=True)
            finally:
                self.interactive = previous
        token = result.get('token')
        if not isinstance(token, str) or expiry(token) <= time.time() + 10:
            raise ScribeError('The Scribe UI has no fresh token. Log in again in BrowserOS Neo and rerun to resume.')
        self.token = token
        return token
