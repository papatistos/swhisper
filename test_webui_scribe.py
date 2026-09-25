"""Scribe WebUI tests use temporary recordings and simulated external services."""
import json
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

    def test_preview_prefers_source_over_converted_wav(self):
        (self.root / 'a.m4a').write_bytes(b'source recording')
        plan = webui.api_run_preview('scribe', selected='a')
        self.assertEqual([row['file'] for row in plan['files']], ['a.m4a'])
        batch = webui.api_run_preview('scribe')
        self.assertEqual([row['file'] for row in batch['files']], ['a.m4a', 'b.wav'])

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

    def scribe_transcript(self, key, source='a.wav'):
        work = self.root / 'scribe-webui' / key
        txt = work / 'transcripts/txt/audio_20260914-030512.txt'
        txt.parent.mkdir(parents=True)
        txt.write_text('Scribe transcript for ' + source)
        (work / 'scribe-state.json').write_text(json.dumps({
            'identity': {'source': str(self.root / source)}}))
        return txt

    def test_scribe_transcripts_join_original_recording_versions(self):
        from fastapi.testclient import TestClient
        txt = self.scribe_transcript('first')
        self.scribe_transcript('second-language')
        self.scribe_transcript('other-recording', 'b.wav')
        legacy = Path(self.state.dirs()['output']) / 'txt/a_20260913-120000.txt'
        legacy.parent.mkdir(parents=True)
        legacy.write_text('Local transcript')
        diarized = txt.parent.parent / 'json/audio_20260914-030512_diarized.json'
        diarized.parent.mkdir()
        diarized.write_text(json.dumps({'segments': [
            {'start': 0, 'end': 1, 'speaker': 'S01', 'text': 'Hello'}]}))
        version = str(txt.relative_to(self.root))
        with TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            response = client.get('/api/transcript/a', params={'version': version})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(len(data['versions']), 3)
            self.assertEqual(data['content'], 'Scribe transcript for a.wav')
            self.assertTrue(data['can_render'])
            self.assertEqual(sum(v['pipeline'] == 'Scribe' for v in data['versions']), 2)
            rendered = client.post('/api/render/a', json={'version': version})
            self.assertEqual(rendered.status_code, 200, rendered.text)
            self.assertIn('Hello', rendered.json()['content'])
            self.assertEqual(client.get('/api/transcript/b', params={'version': version}).status_code, 400)
            local = client.get('/api/transcript/a', params={'version': str(legacy.relative_to(self.root))})
            self.assertEqual(local.json()['content'], 'Local transcript')

    def test_scribe_discovery_ignores_invalid_state_and_symlinks(self):
        txt = self.scribe_transcript('valid')
        invalid = self.scribe_transcript('invalid')
        (invalid.parents[2] / 'scribe-state.json').write_text('{broken')
        escaped = self.scribe_transcript('escaped')
        escaped.unlink()
        escaped.symlink_to(txt)
        (self.root / 'scribe-webui/linked').symlink_to(txt.parents[2], target_is_directory=True)
        self.scribe_transcript('outside', '../a.wav')
        versions = webui._transcript_versions(self.state.dirs(), 'a')
        self.assertEqual([v['id'] for v in versions], [str(txt.relative_to(self.root))])

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

    def test_preview_works_from_a_server_thread_in_a_fresh_process(self):
        # The web server runs requests in worker threads. Importing the diarize
        # package there installs signal handlers and fails, so the preview must
        # not import it. A fresh process keeps earlier tests from masking this.
        import subprocess
        import sys
        script = f'''import sys, threading
sys.path.insert(0, {str(webui.PROJECT_ROOT)!r})
from webui_scribe import preview_scribe
result = {{}}
def run():
    try:
        result['files'] = [r['file'] for r in preview_scribe({str(self.root)!r}, 'a')['files']]
    except Exception as exc:
        result['error'] = repr(exc)
thread = threading.Thread(target=run)
thread.start()
thread.join()
print(result)
assert result.get('files') == ['a.wav'], result
assert 'diarize.utils' not in sys.modules
'''
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

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

    def test_rerun_of_saved_transcript_needs_no_upload_consent(self):
        work = Path(webui.api_run_preview('scribe', selected='a')['files'][0]['work_dir'])
        saved = work / 'whisper-json-output' / 'audio.json'
        saved.parent.mkdir(parents=True)
        saved.write_text('{}')
        preview = webui.api_run_preview('scribe', selected='a')
        self.assertFalse(preview['remote'])
        self.assertFalse(preview['files'][0]['remote'])
        self.assertIn('no upload', preview['files'][0]['reason'])
        with patch.object(self.state, 'start') as start:
            webui.api_run(dict(pipeline='scribe', selected='a', language=preview['language'],
                               preview_token=preview['token']))
            start.assert_called_once()
        # Without a saved transcript the same request needs consent again.
        saved.unlink()
        preview = webui.api_run_preview('scribe', selected='a')
        self.assertTrue(preview['remote'])
        with patch.object(self.state, 'start') as start, self.assertRaises(webui.HTTPException):
            webui.api_run(dict(pipeline='scribe', selected='a', language=preview['language'],
                               preview_token=preview['token']))
        start.assert_not_called()

    def test_saved_diarization_from_the_selected_model_is_reused(self):
        from webui_scribe import preview_scribe
        work = Path(preview_scribe(self.root, selected='a')['files'][0]['work_dir'])
        saved = work / 'whisper-json-output' / 'audio.json'
        saved.parent.mkdir(parents=True)
        saved.write_text('{}')
        marker = work / 'transcripts' / 'audio.ok'
        marker.parent.mkdir(parents=True)
        marker.write_text('# Diarization model: pyannote/speaker-diarization-community-1\n')
        row = preview_scribe(self.root, selected='a', diarization_model='community-1')['files'][0]
        self.assertEqual(row['action'], 'skip')
        self.assertEqual(row['reusable'], {'transcription': True, 'diarization': True})
        self.assertIn('reuses the local', row['reason'])
        forced = preview_scribe(self.root, selected='a', diarization_model='community-1', rediarize=True)
        self.assertEqual(forced['files'][0]['action'], 'process')
        self.assertTrue(forced['rediarize'])
        other = preview_scribe(self.root, selected='a', diarization_model='nemotron')['files'][0]
        self.assertEqual(other['action'], 'process')
        self.assertFalse(other['reusable']['diarization'])
        # A new Scribe transcript always needs a new diarization.
        again = preview_scribe(self.root, selected='a', diarization_model='community-1', retranscribe=True)
        self.assertEqual(again['files'][0]['action'], 'process')
        self.assertIn('New Scribe transcript', again['files'][0]['reason'])

    def test_named_recordings_upload_again_without_new_consent(self):
        from webui_scribe import preview_scribe
        for row in preview_scribe(self.root)['files']:
            saved = Path(row['work_dir']) / 'whisper-json-output' / 'audio.json'
            saved.parent.mkdir(parents=True)
            saved.write_text('{}')
        plan = preview_scribe(self.root, retranscribe={'a'})
        a, b = plan['files']
        self.assertEqual((a['remote'], a['retranscribe'], a['consent']), (True, True, False))
        self.assertEqual((b['remote'], b['retranscribe']), (False, False))
        self.assertTrue(plan['remote'])
        self.assertFalse(plan['consent_required'], 'Recordings uploaded before need no new consent')
        self.assertEqual(plan['retranscribe'], ['a'])
        (self.root / 'c.wav').write_bytes(b'new')
        fresh = preview_scribe(self.root, selected='c')
        self.assertTrue(fresh['files'][0]['consent'])
        self.assertTrue(fresh['consent_required'])

    def test_ticked_recordings_limit_a_scribe_run(self):
        (self.root / 'c.wav').write_bytes(b'third')
        preview = webui.api_run_preview('scribe', ['a', 'c'])
        self.assertEqual([r['file'] for r in preview['files']], ['a.wav', 'c.wav'])
        self.assertEqual(preview['only'], ['a', 'c'])
        with self.assertRaises(webui.HTTPException):
            webui.api_run_preview('scribe', ['missing'])
        (self.root / 'a.mp3').write_bytes(b'same basename')
        (self.root / 'a.m4a').write_bytes(b'another source')
        with self.assertRaises(webui.HTTPException):
            webui.api_run_preview('scribe', ['a'])
        (self.root / 'a.mp3').unlink()
        (self.root / 'a.m4a').unlink()
        import webui_scribe
        plan = webui_scribe.preview_scribe(self.root, only={'c'})
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(plan)]), 0)
        self.assertEqual(adapter.call_count, 1)
        self.assertIn(str(self.root / 'c.wav'), adapter.call_args.args[0])

    def test_batch_runs_only_reviewed_files_that_need_work(self):
        import webui_scribe
        work = Path(webui_scribe.preview_scribe(self.root, selected='a')['files'][0]['work_dir'])
        (work / 'whisper-json-output').mkdir(parents=True)
        (work / 'whisper-json-output' / 'audio.json').write_text('{}')
        (work / 'transcripts').mkdir()
        (work / 'transcripts' / 'audio.ok').write_text('# Diarization model: pyannote/speaker-diarization-community-1\n')
        plan = webui_scribe.preview_scribe(self.root, diarization_model='community-1')
        self.assertEqual([r['action'] for r in plan['files']], ['skip', 'process'])
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(plan)]), 0)
        self.assertEqual(adapter.call_count, 1)
        self.assertIn(str(self.root / 'b.wav'), adapter.call_args.args[0])
        plan = webui_scribe.preview_scribe(self.root, selected='a', diarization_model='community-1', rediarize=True)
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(plan)]), 0)
        self.assertIn('--rediarize', adapter.call_args.args[0])

    def test_selected_scope_validates_identity_before_remote_start(self):
        selected = webui.api_run_preview('scribe', selected='a', language='Swedish')
        self.assertEqual([r['file'] for r in selected['files']], ['a.wav'])
        self.assertEqual(selected['language'], 'Swedish')
        token = selected['token']
        (self.root / 'a.wav').write_bytes(b'changed')
        self.assertNotEqual(token, webui.api_run_preview('scribe', selected='a', language='Swedish')['token'])
        (self.root / 'a.mp3').write_bytes(b'ambiguous')
        (self.root / 'a.m4a').write_bytes(b'another source')
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
