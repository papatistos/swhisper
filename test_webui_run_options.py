"""Per-run diarization model and reprocess options in the web UI, with no models or remote calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import webui
import webui_scribe


class RunOptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'a.wav').write_bytes(b'original')
        self.state = webui.AppState(str(self.root))
        p = patch.object(webui, 'state', self.state)
        p.start()
        self.addCleanup(p.stop)

    def transcribed(self):
        json_dir = self.root / 'whisper-json-output'
        json_dir.mkdir()
        (json_dir / 'a.json').write_text('{}')
        output = Path(self.state.dirs()['output'])
        output.mkdir(parents=True)
        (output / 'a.ok').write_text('')

    def test_local_preview_reprocesses_existing_outputs_only_when_asked(self):
        self.transcribed()
        plain = webui.api_run_preview('transcribe4', None)
        self.assertEqual({r['action'] for r in plain['files']}, {'skip'})
        forced = webui.api_run_preview('transcribe4', None, reprocess=True, diarization_model='nemotron')
        self.assertEqual({r['action'] for r in forced['files']}, {'reprocess'})
        self.assertIn('requested for this run', forced['files'][0]['reason'])
        self.assertEqual(forced['diarization_model'], 'nemotron')
        self.assertNotEqual(plain['token'], forced['token'])

    def test_run_passes_options_to_the_pipeline_process(self):
        preview = webui.api_run_preview('diarize4', None, diarization_model='nemotron', reprocess=True)
        with patch('webui.subprocess.Popen', **{'return_value.pid': 12345}) as popen, patch.object(self.state, '_watch_process'):
            webui.api_run(dict(pipeline='diarize4', preview_token=preview['token'],
                               diarization_model='nemotron', reprocess=True))
        env = popen.call_args.kwargs['env']
        self.assertEqual(env['SWHISPER_PIPELINE_MODEL'], 'nvidia/Nemotron-3-Diarization')
        self.assertEqual(env['SWHISPER_RUN_REPROCESS'], 'true')

    def test_run_without_options_keeps_the_saved_model_and_does_not_reprocess(self):
        with patch.dict('os.environ', {'SWHISPER_RUN_REPROCESS': 'true', 'SWHISPER_PIPELINE_MODEL': 'saved/model'}), \
                patch('webui.subprocess.Popen', **{'return_value.pid': 12345}) as popen, patch.object(self.state, '_watch_process'):
            webui.api_run(dict(pipeline='diarize4'))
        env = popen.call_args.kwargs['env']
        self.assertEqual(env['SWHISPER_PIPELINE_MODEL'], 'saved/model')
        self.assertNotIn('SWHISPER_RUN_REPROCESS', env)

    def test_options_are_validated_and_bound_to_the_preview(self):
        with self.assertRaises(webui.HTTPException):
            webui.api_run_preview('diarize4', None, diarization_model='unknown')
        with self.assertRaises(webui.HTTPException):
            webui.api_run(dict(pipeline='diarize4', diarization_model='unknown'))
        preview = webui.api_run_preview('diarize4', None)
        with patch.object(self.state, 'start') as start, self.assertRaises(webui.HTTPException):
            webui.api_run(dict(pipeline='diarize4', preview_token=preview['token'], reprocess=True))
        start.assert_not_called()

    def test_scribe_retranscribe_needs_consent_and_reaches_the_adapter(self):
        work = Path(webui.api_run_preview('scribe', selected='a')['files'][0]['work_dir'])
        saved = work / 'whisper-json-output' / 'audio.json'
        saved.parent.mkdir(parents=True)
        saved.write_text('{}')
        rerun = webui.api_run_preview('scribe', selected='a', diarization_model='nemotron')
        self.assertFalse(rerun['remote'])
        self.assertEqual(rerun['files'][0]['local_settings']['pipeline_model'], 'nvidia/Nemotron-3-Diarization')
        again = webui.api_run_preview('scribe', selected='a', diarization_model='nemotron', reprocess=True)
        self.assertTrue(again['remote'])
        self.assertIn('Uploads again', again['files'][0]['reason'])
        with patch.object(self.state, 'start') as start, self.assertRaises(webui.HTTPException):
            webui.api_run(dict(pipeline='scribe', selected='a', language=again['language'],
                               preview_token=again['token'], diarization_model='nemotron', reprocess=True))
        start.assert_not_called()
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(again)]), 0)
        argv = adapter.call_args.args[0]
        self.assertEqual(argv[argv.index('--diarization-model') + 1], 'nemotron')
        self.assertIn('--retranscribe', argv)
