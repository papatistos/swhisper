import base64
import json
import time
import unittest
from unittest.mock import Mock, patch
from scribe_browser_auth import BrowserTokenProvider, expiry
from scribe_api import ScribeClient, ScribeError
from scribe_checks import Session, Response


def jwt(seconds):
    payload = base64.urlsafe_b64encode(json.dumps({'exp':time.time()+seconds}).encode()).decode().rstrip('=')
    return 'eyJ.'+payload+'.signature'


class BrowserAuthChecks(unittest.TestCase):
    def provider(self, result):
        p=BrowserTokenProvider(interactive=False)
        p.page=1
        p.run=Mock(return_value=result)
        return p

    def test_rejects_remote_mcp(self):
        with self.assertRaises(ScribeError):
            BrowserTokenProvider('http://example.com/mcp')

    def test_logged_out_stops_without_token(self):
        p=self.provider({'loginRequired':True})
        with self.assertRaisesRegex(ScribeError,'Log in'):
            p.get_token()
        self.assertIsNone(p.token)

    def test_interactive_login_rechecks(self):
        p=self.provider(None);p.interactive=True
        token=jwt(300)
        p.run.side_effect=[{'loginRequired':True},{'token':token}]
        with patch('builtins.input') as prompt:
            self.assertEqual(p.get_token(),token)
            prompt.assert_called_once()

    def test_expired_dialog_rejected(self):
        with self.assertRaises(ScribeError):
            self.provider({'token':jwt(-60)}).get_token()

    def test_cache_and_refresh(self):
        p=self.provider({'token':jwt(300)})
        first=p.get_token();self.assertEqual(first,p.get_token())
        self.assertEqual(p.run.call_count,1)
        p.token=jwt(5);p.get_token();self.assertEqual(p.run.call_count,2)

    def test_401_read_refreshes_once(self):
        provider=Mock();provider.get_token.side_effect=['first','first','second']
        session=Session([Response({},401),Response({'status':'completed'})])
        client=ScribeClient('', '',session=session,token_provider=provider)
        self.assertEqual(client.status('job'),'completed')
        self.assertEqual(session.calls[-1][2]['headers']['Authorization'],'Bearer second')

    def test_upload_not_retried_on_401(self):
        provider=Mock();provider.get_token.return_value='token'
        session=Session([Response({},401)])
        client=ScribeClient('', '',session=session,token_provider=provider)
        with self.assertRaises(ScribeError):client._request('POST','')
        self.assertEqual(len(session.calls),1)
