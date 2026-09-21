"""Global settings, export, and audio tests using temporary files only."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

import path_settings
import webui
from env_settings import parse_env, env_snapshot, update_env


class RemainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in [('state', webui.AppState(str(self.root))), ('GLOBAL_ENV_PATH', self.root / '.env'),
                            ('_tconfig', webui._tconfig), ('_dconfig', webui._dconfig)]:
            p = patch.object(webui, name, value);p.start();self.addCleanup(p.stop)
        p = patch.dict(os.environ);p.start();self.addCleanup(p.stop)
        for key in list(os.environ):
            if key.startswith('SWHISPER_') or key == 'JSON_INPUT_DIR': os.environ.pop(key)
        p = patch.object(path_settings, '_load_env_file', lambda: None);p.start();self.addCleanup(p.stop)
        self.client = TestClient(webui.app, base_url="http://127.0.0.1:8722")
        self.addCleanup(self.client.close)

    def write(self, name, text=''):
        path = self.root / name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text(text);return path

    def transcript(self):
        self.write('a.wav', 'mock audio')
        original = self.write('transcripts/txt/a_20260913-120000.txt', 'Original transcript')
        self.write('transcripts/json/a_20260913-120000_diarized.json', json.dumps({'segments':[
            {'start':1.0,'end':3.0,'speaker':'S01','text':'Hello world','words':[
                {'start':1.0,'end':2.0,'word':'Hello'},{'start':2.0,'end':3.0,'word':'world'}]}]}))
        return original

    def test_env_round_trip_and_unrelated_lines(self):
        path=self.write('.env','# Header\nTOKEN=private-value\nSWHISPER_LANGUAGE=sv # language\n# saved path\n')
        _, revision=env_snapshot(path)
        update_env(path,{'SWHISPER_LANGUAGE':'en','SWHISPER_INITIAL_PROMPT':'A # quote " plus\nnewline','SWHISPER_OUTPUT_PREAMBLE':''},
                   {'SWHISPER_LANGUAGE','SWHISPER_INITIAL_PROMPT','SWHISPER_OUTPUT_PREAMBLE'},revision)
        text=path.read_text();self.assertIn('TOKEN=private-value\n',text);self.assertIn('# language',text)
        self.assertEqual(parse_env(text)['SWHISPER_INITIAL_PROMPT'],'A # quote " plus\nnewline')
        self.assertEqual(parse_env(text)['SWHISPER_OUTPUT_PREAMBLE'],'')
        self.assertEqual(parse_env('X=first\\nsecond\n')['X'],'first\nsecond')

    def test_env_stale_revision_rejected(self):
        path=self.write('.env','A=1\n');_,revision=env_snapshot(path);path.write_text('A=2\n')
        with self.assertRaises(ValueError):update_env(path,{'A':'3'},{'A'},revision)
        self.assertEqual(path.read_text(),'A=2\n')

    def test_global_schema_and_save_refreshes_defaults(self):
        self.write('.env','# Keep this\nSECRET_TOKEN=hidden\n')
        before=self.client.get('/api/settings/global');self.assertEqual(before.status_code,200)
        form=before.json();self.assertNotIn('SECRET_TOKEN',form['values']);self.assertIn('SWHISPER_MODEL',form['params'])
        response=self.client.put('/api/settings/global',json={'values':{'SWHISPER_LANGUAGE':'en','SWHISPER_FORCE_REPROCESS':'true','SWHISPER_INITIAL_PROMPT':''},'revision':form['revision']})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(webui._global_defaults()['SWHISPER_LANGUAGE'],'en')
        self.assertEqual(webui._tconfig.TranscriptionConfig().force_reprocess,True)
        self.assertEqual(os.environ['SWHISPER_INITIAL_PROMPT'],'')
        self.assertIn('SECRET_TOKEN=hidden',webui.GLOBAL_ENV_PATH.read_text())
        reset=self.client.put('/api/settings/global',json={'values':{},'revision':response.json()['revision']})
        self.assertEqual(reset.status_code,200,reset.text)
        self.assertEqual(webui._global_defaults()['SWHISPER_LANGUAGE'],'sv')

    def test_global_validation_and_running_lock(self):
        form=self.client.get('/api/settings/global').json()
        body={'values':{'SWHISPER_MIN_SPEAKERS':'4','SWHISPER_MAX_SPEAKERS':'2'},'revision':form['revision']}
        self.assertEqual(self.client.put('/api/settings/global',json=body).status_code,422)
        self.assertFalse(webui.GLOBAL_ENV_PATH.exists())
        with patch.object(webui.state,'running',return_value=True):
            self.assertEqual(self.client.put('/api/settings/global',json={'values':{},'revision':form['revision']}).status_code,409)

    def test_per_file_numeric_bounds(self):
        response=self.client.put('/api/sidecar/a',json={'SWHISPER_NUM_SPEAKERS':'0'})
        self.assertEqual(response.status_code,422)
        self.assertIn('SWHISPER_NUM_SPEAKERS',response.json()['errors'])

    def test_export_creates_copies_and_requires_explicit_replace(self):
        original=self.transcript();body={'formats':['txt','rtf'],'name':'review','destination':'exports','options':{'stats':False}}
        plan=self.client.post('/api/render/a/export-preview',json=body)
        self.assertEqual(plan.status_code,200,plan.text)
        self.assertEqual(plan.json()['existing'],[])
        response=self.client.post('/api/render/a/save',json={**body,'token':plan.json()['token']})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(original.read_text(),'Original transcript')
        self.assertTrue((self.root/'exports/review.rtf').exists())
        self.assertIn('Hello',(self.root/'exports/review.txt').read_text())
        next_plan=self.client.post('/api/render/a/export-preview',json=body).json()
        self.assertEqual(len(next_plan['existing']),2)
        self.assertEqual(self.client.post('/api/render/a/save',json={**body,'token':next_plan['token']}).status_code,409)
        self.assertEqual(self.client.post('/api/render/a/save',json={**body,'token':next_plan['token'],'overwrite':True}).status_code,200)

    def test_export_protects_originals_and_paths(self):
        self.transcript()
        for body in [{'name':'a_20260913-120000','destination':'transcripts/txt'}, {'name':'../escape'}, {'destination':'../outside'}]:
            response=self.client.post('/api/render/a/export-preview',json=body)
            self.assertIn(response.status_code,(400,409),response.text)
        self.assertEqual(self.client.post('/api/render/a/save',json={'name':'review'}).status_code,409)

    def test_export_detects_conflict_after_preview(self):
        self.transcript();body={'name':'review'}
        plan=self.client.post('/api/render/a/export-preview',json=body).json()
        target=self.write('exports/review.txt','Someone else wrote this')
        response=self.client.post('/api/render/a/save',json={**body,'token':plan['token'],'overwrite':True})
        self.assertEqual(response.status_code,409)
        self.assertEqual(target.read_text(),'Someone else wrote this')

    def test_native_save_as_writes_the_selected_path(self):
        original = self.transcript()
        target = self.root / 'chosen folder' / 'review.rtf'
        scripts = []

        def fake_run(argv, **kwargs):
            scripts.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, stdout=str(target) + '\n', stderr='')

        with patch.object(webui.subprocess, 'run', fake_run):
            choice = self.client.post('/api/render/a/save-as', json={'format': 'rtf', 'options': {'stats': False}})
        self.assertEqual(choice.status_code, 200, choice.text)
        self.assertIn('choose file name', scripts[0])
        self.assertIn('default name', scripts[0])
        self.assertIn('.rtf', scripts[0])
        self.assertEqual(choice.json()['path'], str(target))

        saved = self.client.post('/api/render/a/save-selected', json={'token': choice.json()['token']})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()['written'], [str(target)])
        self.assertTrue(target.exists())
        self.assertIn('Hello', target.read_text())
        self.assertEqual(original.read_text(), 'Original transcript')

    def test_native_save_as_cancel_returns_without_an_error_or_export(self):
        self.transcript()

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 1, stdout='', stderr='execution error: User canceled. (-128)')

        with patch.object(webui.subprocess, 'run', fake_run):
            response = self.client.post('/api/render/a/save-as', json={'format': 'txt', 'options': {'stats': False}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'canceled': True})
        self.assertEqual(webui.state.export_selections, {})

    def test_native_save_as_keeps_the_selected_format_extension(self):
        self.transcript()

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout=str(self.root / 'review.rtf') + '\n', stderr='')

        with patch.object(webui.subprocess, 'run', fake_run):
            response = self.client.post('/api/render/a/save-as', json={'format': 'txt', 'options': {'stats': False}})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn('Choose a .txt filename', response.json()['detail'])

    def test_native_save_as_reports_write_errors_clearly(self):
        self.transcript()
        target = self.root / 'review.txt'

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout=str(target) + '\n', stderr='')

        with patch.object(webui.subprocess, 'run', fake_run):
            choice = self.client.post('/api/render/a/save-as', json={'format': 'txt', 'options': {'stats': False}})
        with patch.object(webui.tempfile, 'mkstemp', side_effect=OSError('disk full')):
            response = self.client.post('/api/render/a/save-selected', json={'token': choice.json()['token']})
        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn('Could not export transcript: disk full', response.json()['detail'])

    def test_view_options_do_not_modify_original(self):
        original=self.transcript();response=self.client.post('/api/render/a',json={'options':{'stats':False,'preamble':False}})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(original.read_text(),'Original transcript')
        self.assertEqual(response.json()['timeline'][0]['text'],'Hello world')

    def test_env_loader_keeps_empty_values_and_hashes(self):
        self.write('.swhisper.env', 'TEST_SWHISPER_EMPTY=""\nTEST_SWHISPER_PROMPT="A # title"\n')
        # Exercise the actual loader instead of the test's model-config stub.
        import importlib.util
        spec=importlib.util.spec_from_file_location('path_settings_loader_test', Path(path_settings.__file__))
        module=importlib.util.module_from_spec(spec)
        import sys
        with patch.dict(sys.modules, {'path_settings_loader_test':module}):
            spec.loader.exec_module(module)
            module.__file__=str(self.root/'path_settings.py')
            module._load_env_file()
        self.assertEqual(os.environ['TEST_SWHISPER_EMPTY'],'')
        self.assertEqual(os.environ['TEST_SWHISPER_PROMPT'],'A # title')

    def test_global_change_invalidates_run_preview(self):
        self.write('a.wav')
        before=self.client.get('/api/run/preview',params={'pipeline':'transcribe4'}).json()['token']
        self.write('.env','SWHISPER_MODEL=different-model\n')
        after=self.client.get('/api/run/preview',params={'pipeline':'transcribe4'}).json()['token']
        self.assertNotEqual(before,after)

    def test_speaker_names_rename_views_but_not_stored_transcripts(self):
        original = self.transcript()
        detected = self.client.get("/api/speakers/a").json()
        self.assertEqual(detected["detected"], ["S01"])
        for bad in [{"NOPE": "X"}, {"S01": "a\nb"}, {"S01": "x" * 61}, "nope"]:
            self.assertEqual(self.client.put("/api/speakers/a", json={"names": bad}).status_code, 422, bad)
        saved = self.client.put("/api/speakers/a", json={"names": {"S01": "Interviewer"}})
        self.assertEqual(saved.status_code, 200, saved.text)
        rendered = self.client.post("/api/render/a", json={"options": {}}).json()
        self.assertIn("Interviewer", rendered["content"])
        self.assertNotIn("S01", rendered["content"])
        self.assertEqual(rendered["timeline"][0]["speaker"], "Interviewer")
        self.assertEqual(self.client.get("/api/timeline/a").json()["segments"][0]["speaker"], "Interviewer")
        self.assertEqual(self.client.get("/api/transcript/a").json()["speaker_names"], {"S01": "Interviewer"})
        # The file the pipeline wrote is never rewritten.
        self.assertEqual(original.read_text(), "Original transcript")
        self.assertEqual(self.client.put("/api/speakers/a", json={"names": {"S01": ""}}).json()["names"], {})

    def test_reveal_stays_inside_the_audio_folder(self):
        self.transcript()
        calls = []
        with patch("subprocess.Popen", lambda command, **kw: calls.append(command)):
            ok = self.client.post("/api/reveal",
                                  json={"path": "transcripts/txt/a_20260913-120000.txt"})
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertEqual(len(calls), 1)
            self.assertIn("a_20260913-120000.txt", calls[0][-1])
            for bad in ["../../etc/passwd", "/etc/passwd", "missing.txt"]:
                response = self.client.post("/api/reveal", json={"path": bad})
                self.assertIn(response.status_code, (400, 404), bad)
            self.assertEqual(len(calls), 1)   # nothing else was launched

    def test_rejects_foreign_host_and_cross_origin_requests(self):
        self.assertEqual(self.client.get("/api/run/status").status_code, 200)
        # DNS rebinding: a name that resolves to 127.0.0.1 but is not local.
        rebound = self.client.get("/api/run/status", headers={"Host": "attacker.example"})
        self.assertEqual(rebound.status_code, 403)
        # A bodyless POST is CORS-simple, so the Origin check is what stops it.
        foreign = self.client.post("/api/run/stop", headers={"Origin": "https://attacker.example"})
        self.assertEqual(foreign.status_code, 403)
        same = self.client.post("/api/run/stop", headers={"Origin": "http://127.0.0.1:8722"})
        self.assertEqual(same.status_code, 409)   # reaches the handler: no run

    def test_timeline_and_audio_range(self):
        self.transcript();timeline=self.client.get('/api/timeline/a')
        self.assertEqual(timeline.status_code,200,timeline.text)
        self.assertEqual(timeline.json()['segments'][0]['start'],1.0)
        audio=self.client.get('/api/audio/a',headers={'Range':'bytes=0-3'})
        self.assertEqual(audio.status_code,206)
        self.assertEqual(audio.content,b'mock')
        self.assertEqual(self.client.get('/api/audio/missing').status_code,404)


if __name__=='__main__':unittest.main()
