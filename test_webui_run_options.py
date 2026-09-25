"""Per-run diarization model and reprocess options in the web UI, with no models or remote calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import webui
import webui_scribe
from diarization_status import COMMUNITY_MODEL
from run_scope import DIARIZATION_MODELS


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

    def transcribed(self, model=COMMUNITY_MODEL):
        json_dir = self.root / 'whisper-json-output'
        json_dir.mkdir()
        (json_dir / 'a.json').write_text('{}')
        output = Path(self.state.dirs()['output'])
        output.mkdir(parents=True)
        (output / 'a.ok').write_text(f'# Diarization model: {model}\n')

    def test_local_preview_reprocesses_existing_outputs_only_when_asked(self):
        self.transcribed()
        plain = webui.api_run_preview('transcribe4', None)
        self.assertEqual({r['action'] for r in plain['files']}, {'skip'})
        forced = webui.api_run_preview('transcribe4', None, reprocess=True, diarization_model='nemotron')
        # A different model already explains the diarization rerun.
        self.assertEqual([(r['stage'], r['action']) for r in forced['files']],
                         [('Transcription', 'reprocess'), ('Diarization', 'process')])
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

    def test_run_reprocessing_does_not_persist_in_file_settings(self):
        self.transcribed()
        sidecar = self.root / 'a.swhisper'
        sidecar.write_text('# Keep this comment\nSWHISPER_FORCE_REPROCESS=false\n')
        before = (sidecar.read_bytes(), sidecar.stat().st_mtime_ns)
        preview = webui.api_run_preview('transcribe4', ['a'], reprocess=True)
        self.assertEqual({r['action'] for r in preview['files']}, {'reprocess'})
        with patch('webui.subprocess.Popen', **{'return_value.pid': 12345}), patch.object(self.state, '_watch_process'):
            webui.api_run(dict(pipeline='transcribe4', only=['a'],
                               preview_token=preview['token'], reprocess=True))
        self.assertEqual((sidecar.read_bytes(), sidecar.stat().st_mtime_ns), before)
        next_preview = webui.api_run_preview('transcribe4', ['a'], reprocess=False)
        self.assertEqual({r['action'] for r in next_preview['files']}, {'skip'})

    def test_selected_existing_diarization_explains_how_to_rerun(self):
        self.transcribed(DIARIZATION_MODELS['nemotron'])
        plain = webui.api_run_preview('diarize4', ['a'], diarization_model='nemotron')
        self.assertEqual([r['action'] for r in plain['files']], ['skip'])
        self.assertIn('Run is disabled', plain['notes'][0])
        self.assertIn('Re-diarize', plain['notes'][0])
        forced = webui.api_run_preview('diarize4', ['a'], diarization_model='nemotron', reprocess=True)
        self.assertEqual([r['action'] for r in forced['files']], ['reprocess'])
        self.assertFalse(any('Run is disabled' in note for note in forced['notes']))
        self.assertNotEqual(plain['token'], forced['token'])

    def test_first_run_with_new_model_does_not_require_reprocessing(self):
        self.transcribed()
        plan = webui.api_run_preview('diarize4', ['a'], diarization_model='nemotron')
        self.assertEqual([r['action'] for r in plan['files']], ['process'])
        self.assertIn(COMMUNITY_MODEL, plan['files'][0]['reason'])
        self.assertFalse(plan['reprocess'])
        with patch.object(self.state, 'start') as start:
            webui.api_run(dict(pipeline='diarize4', only=['a'], preview_token=plan['token'],
                               diarization_model='nemotron', reprocess=False))
        self.assertEqual(start.call_args.kwargs['options'],
                         {'diarization_model': 'nemotron', 'reprocess': False, 'retranscribe': [], 'rediarize': []})

    def test_named_recordings_are_redone_and_reach_the_pipeline(self):
        self.transcribed()
        plan = webui.api_run_preview('transcribe4', ['a'], diarization_model='community-1', rediarize=['a'])
        self.assertEqual([(r['stage'], r['action']) for r in plan['files']],
                         [('Transcription', 'skip'), ('Diarization', 'reprocess')])
        self.assertEqual(plan['rediarize'], ['a'])
        again = webui.api_run_preview('transcribe4', ['a'], diarization_model='community-1', retranscribe=['a'])
        self.assertEqual([r['action'] for r in again['files']], ['reprocess', 'reprocess'],
                         'A new transcript is diarized again')
        with patch('webui.subprocess.Popen', **{'return_value.pid': 12345}) as popen, patch.object(self.state, '_watch_process'):
            webui.api_run(dict(pipeline='transcribe4', only=['a'], preview_token=plan['token'],
                               diarization_model='community-1', rediarize=['a']))
        env = popen.call_args.kwargs['env']
        self.assertEqual(env['SWHISPER_RUN_REDIARIZE'], 'a')
        self.assertNotIn('SWHISPER_RUN_RETRANSCRIBE', env)
        self.assertNotIn('SWHISPER_RUN_REPROCESS', env)
        for bad in (['../a'], 'a', [3]):
            with self.assertRaises(webui.HTTPException):
                webui.api_run(dict(pipeline='transcribe4', rediarize=bad))

    def test_preview_reports_which_files_have_reusable_results(self):
        self.transcribed()
        (self.root / 'b.wav').write_bytes(b'new')
        for rediarize in ([], ['a']):
            plan = webui.api_run_preview('transcribe4', None, diarization_model='community-1', rediarize=rediarize)
            self.assertEqual(plan['reusable'], {'a': {'transcription': True, 'diarization': True}})
        other = webui.api_run_preview('transcribe4', None, diarization_model='nemotron')
        self.assertEqual(other['reusable'], {'a': {'transcription': True, 'diarization': False}})

    def precision_switch(self, on):
        original = webui._dconfig.DiarizationConfig
        def config():
            made = original()
            made.use_precision_service = on
            return made
        return patch.object(webui._dconfig, 'DiarizationConfig', config)

    def test_precision_is_offered_only_by_the_saved_switch_and_wins_when_chosen(self):
        with self.precision_switch(False):
            self.assertNotIn('precision', [m['id'] for m in webui.api_meta()['diarization_models']])
            with self.assertRaises(webui.HTTPException):
                webui.api_run_preview('transcribe4', None, diarization_model='precision')
        with self.precision_switch(True):
            meta = webui.api_meta()
            self.assertIn('precision', [m['id'] for m in meta['diarization_models']])
            self.assertEqual(meta['diarization_default'], 'community-1')
            with self.assertRaises(webui.HTTPException):
                webui.api_run_preview('scribe', selected='a', diarization_model='precision')
            with patch('webui.subprocess.Popen', **{'return_value.pid': 12345}) as popen, patch.object(self.state, '_watch_process'):
                webui.api_run(dict(pipeline='transcribe4', diarization_model='nemotron'))
            self.assertEqual(popen.call_args.kwargs['env']['SWHISPER_USE_PRECISION'], 'false')
            self.state.process = None
            with patch('webui.subprocess.Popen', **{'return_value.pid': 12345}) as popen, patch.object(self.state, '_watch_process'):
                webui.api_run(dict(pipeline='transcribe4', diarization_model='precision'))
            self.assertEqual(popen.call_args.kwargs['env']['SWHISPER_USE_PRECISION'], 'true')

    def test_file_list_marks_recordings_with_unrecovered_backfill_gaps(self):
        self.transcribed()
        (self.root / 'b.wav').write_bytes(b'other')
        # Listing files starts background duration probes; keep them out of later Popen mocks.
        warm = patch.object(webui.webui_durations, 'warm', return_value=False)
        warm.start()
        self.addCleanup(warm.stop)
        files = {f['base']: f for f in webui.api_files()['files']}
        self.assertEqual((files['a']['backfill_incomplete'], files['b']['backfill_incomplete']), (False, False))
        marker = Path(self.state.dirs()['output']) / 'a.ok'
        marker.write_text(marker.read_text() + '# Backfill status: partial\n')
        self.assertTrue({f['base']: f for f in webui.api_files()['files']}['a']['backfill_incomplete'])

    def test_versions_count_unrecovered_backfill_gaps(self):
        txt = Path(self.state.dirs()['output']) / 'txt'
        txt.mkdir(parents=True)
        (txt / 'a_20260925-100000.txt').write_text('INTERVIEWER: Hej [--] och [ - ] mm [*]\n')
        (txt / 'a_20260925-110000.txt').write_text('INTERVIEWER: Hej [*] och [ ** ]\n')
        versions = {v['timestamp']: v['backfill_gaps'] for v in webui.api_transcript_versions('a')['versions']}
        self.assertEqual(versions, {'20260925-100000': 2, '20260925-110000': 0})

    def test_meta_offers_local_whisper_and_scribe_as_transcription_choices(self):
        meta = webui.api_meta()
        self.assertEqual([p['id'] for p in meta['pipelines'] if p['choice']], ['transcribe4', 'scribe'])
        self.assertEqual(meta['scribe_languages'], ['Swedish (verbatim)', 'Swedish'])

    def test_model_change_in_standard_pipeline_reuses_transcription(self):
        self.transcribed()
        plan = webui.api_run_preview('transcribe4', ['a'], diarization_model='nemotron')
        self.assertEqual([(r['stage'], r['action']) for r in plan['files']],
                         [('Transcription', 'skip'), ('Diarization', 'process')])

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

    def test_scribe_retranscribe_of_an_uploaded_recording_needs_no_new_consent(self):
        work = Path(webui.api_run_preview('scribe', selected='a')['files'][0]['work_dir'])
        saved = work / 'whisper-json-output' / 'audio.json'
        saved.parent.mkdir(parents=True)
        saved.write_text('{}')
        rerun = webui.api_run_preview('scribe', selected='a', diarization_model='nemotron')
        self.assertFalse(rerun['remote'])
        self.assertEqual(rerun['files'][0]['local_settings']['pipeline_model'], 'nvidia/Nemotron-3-Diarization')
        again = webui.api_run_preview('scribe', selected='a', diarization_model='nemotron', retranscribe=['a'])
        self.assertTrue(again['remote'])
        self.assertFalse(again['consent_required'])
        self.assertIn('Uploads again', again['files'][0]['reason'])
        with patch.object(self.state, 'start') as start:
            webui.api_run(dict(pipeline='scribe', selected='a', language=again['language'],
                               preview_token=again['token'], diarization_model='nemotron', retranscribe=['a']))
        start.assert_called_once()
        with patch('transcribe_scribe.main', return_value=0) as adapter:
            self.assertEqual(webui_scribe.main([json.dumps(again)]), 0)
        argv = adapter.call_args.args[0]
        self.assertEqual(argv[argv.index('--diarization-model') + 1], 'nemotron')
        self.assertIn('--retranscribe', argv)
