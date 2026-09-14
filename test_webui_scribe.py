"""Scribe WebUI tests use temporary recordings and simulated external services."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import webui


class ScribeWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'a.wav').write_bytes(b'original')
        (self.root / 'b.wav').write_bytes(b'other')
        self.state = webui.AppState(str(self.root))
        p = patch.object(webui, 'state', self.state)
        p.start()
        self.addCleanup(p.stop)

    def test_artifacts_are_readable_without_exposing_arbitrary_files(self):
        from fastapi.testclient import TestClient
        work = Path(webui.api_run_preview('scribe')['files'][0]['work_dir'])
        work.mkdir(parents=True)
        result = work / 'whisper-json-output' / 'audio.json'
        result.parent.mkdir()
        result.write_text('{"text":"simulated transcript"}')
        (work / 'secret.env').write_text('must not be served')
        (work / 'escape.txt').symlink_to(self.root / 'a.wav')
        with TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            listing = client.get('/api/scribe/artifacts')
            self.assertEqual(listing.status_code, 200)
            files = listing.json()['files']
            self.assertEqual(len(files), 1)
            self.assertIn('audio.json', files[0]['path'])
            fetched = client.get('/api/scribe/artifact', params={'path': files[0]['path']})
            self.assertEqual(fetched.json()['text'], 'simulated transcript')
            for path in ('../a.wav', 'a.wav', str(result), str((work / 'secret.env').relative_to(self.root)), str((work / 'escape.txt').relative_to(self.root))):
                self.assertIn(client.get('/api/scribe/artifact', params={'path': path}).status_code, (400, 404))

    def test_batch_stops_before_changed_or_redirected_input(self):
        import json
        import webui_scribe
        plan = webui_scribe.preview_scribe(self.root)
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            (self.root / 'a.wav').write_bytes(b'changed after approval')
            self.assertEqual(webui_scribe.main([json.dumps(plan)]), 1)
            adapter.assert_not_called()
        plan = webui_scribe.preview_scribe(self.root)
        with patch('transcribe_scribe.main', side_effect=[1, 0]) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(plan)]), 1)
            self.assertEqual(adapter.call_count, 1)
        work = Path(plan['files'][0]['work_dir'])
        work.parent.mkdir()
        work.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            webui_scribe.preview_scribe(self.root)

    def test_http_launch_executes_fixed_batch_in_noninteractive_child(self):
        import json
        import time
        from fastapi.testclient import TestClient
        # Only the external transcription/local model boundary is simulated.
        # AppState, subprocess creation, batch runner, logs and history are real.
        fixture = self.root / 'simulated_scribe.py'
        fixture.write_text('''import sys, json
from pathlib import Path
sys.path.insert(0, ''' + repr(str(webui.PROJECT_ROOT)) + ''')
import webui_scribe, transcribe_scribe
assert not sys.stdin.isatty()
def simulated(argv):
    print('SIMULATED adapter argv: ' + json.dumps(argv), flush=True)
    work = Path(argv[argv.index('--work-dir') + 1])
    work.mkdir(parents=True, exist_ok=True)
    (work / 'scribe-state.json').write_text('{"simulated":true}')
    return 0
transcribe_scribe.main = simulated
sys.exit(webui_scribe.main())
''')
        with patch.dict(webui.PIPELINES, scribe={'script': str(fixture), 'label': 'Simulated Scribe'}), TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            preview = client.get('/api/run/preview', params={'pipeline': 'scribe', 'selected': 'a'}).json()
            result = client.post('/api/run', json=dict(pipeline='scribe', selected='a', language=preview['language'], remote_upload=True, preview_token=preview['token']))
            self.assertEqual(result.status_code, 200, result.text)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if (self.root / webui.HISTORY_FILENAME).exists():
                    break
                time.sleep(.02)
            self.assertEqual(self.state.returncode, 0, self.state.log_since())
            log = '\n'.join(self.state.log_since()['lines'])
            self.assertIn('--auth', log)
            self.assertIn('browseros', log)
            self.assertIn(str(self.root / 'a.wav'), log)
            self.assertNotIn(str(self.root / 'b.wav'), log)
            history = json.loads((self.root / webui.HISTORY_FILENAME).read_text())
            self.assertEqual(history['scribe']['language'], 'Swedish (verbatim)')
            self.assertEqual(history['scribe']['files'], ['a.wav'])
            self.assertTrue(any('scribe-state.json' in f for f in history['files_produced']))
            self.assertEqual((self.root / 'a.wav').read_bytes(), b'original')

    def test_launch_requires_remote_consent_and_fresh_preview(self):
        preview = webui.api_run_preview('scribe', selected='a')
        with patch.object(self.state, 'start') as start:
            for body in ({'pipeline': 'scribe'}, {'pipeline': 'scribe', 'preview_token': preview['token']}):
                with self.assertRaises(webui.HTTPException):
                    webui.api_run(body)
            body = dict(pipeline='scribe', selected='a', language=preview['language'],
                        preview_token=preview['token'], remote_upload=True)
            webui.api_run(body)
            self.assertEqual(start.call_args.args[0], 'scribe')
            self.assertEqual(start.call_args.kwargs['plan']['files'], preview['files'])
            (self.root / 'a.swhisper').write_text('SWHISPER_NUM_SPEAKERS=2')
            with self.assertRaises(webui.HTTPException):
                webui.api_run(body)

    def test_selected_scope_validates_identity_before_remote_start(self):
        selected = webui.api_run_preview('scribe', selected='a', language='Swedish')
        self.assertEqual([r['file'] for r in selected['files']], ['a.wav'])
        self.assertEqual(selected['language'], 'Swedish')
        token = selected['token']
        (self.root / 'a.wav').write_bytes(b'changed')
        self.assertNotEqual(token, webui.api_run_preview('scribe', selected='a', language='Swedish')['token'])
        (self.root / 'a.mp3').write_bytes(b'ambiguous')
        for selected in ('a', '../a', 'missing'):
            with self.assertRaises(webui.HTTPException):
                webui.api_run_preview('scribe', selected=selected)
        with self.assertRaises(webui.HTTPException):
            webui.api_run_preview('scribe', language='English')
        (self.root / 'outside.wav').symlink_to('/etc/hosts')
        with self.assertRaises(webui.HTTPException):
            webui.api_run_preview('scribe')

    def test_scribe_preview_is_explicit_and_isolated(self):
        result = webui.api_run_preview('scribe')
        self.assertEqual(result['language'], 'Swedish (verbatim)')
        self.assertEqual([r['file'] for r in result['files']], ['a.wav', 'b.wav'])
        self.assertTrue(all(r['work_dir'].startswith(str(self.root / 'scribe-webui')) for r in result['files']))
        self.assertIn('upload', ' '.join(result['notes']))
        self.assertEqual(list(self.root.iterdir()).__len__(), 2)


if __name__ == '__main__':
    unittest.main()
