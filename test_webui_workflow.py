"""WebUI workflow tests: only temporary files, no models or real transcription."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import webui
from sidecar import parse_sidecar_file
from webui_run import preview_run, inspect_progress, ProgressTracker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tc = SimpleNamespace(json_dir='whisper-json-output', force_reprocess=False, file_limit=None)
        self.dc = SimpleNamespace(json_input_dir='whisper-json-output', output_dir='transcripts', force_reprocess=False)
        self.state = webui.AppState(str(self.root))
        self.patch = patch.object(webui, 'state', self.state)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, name, content=''):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def preview(self, pipeline='transcribe4'):
        return preview_run(str(self.root), pipeline, self.tc, self.dc)

    def test_unknown_values_survive_save_and_clear(self):
        self.write('a.wav')
        self.write('a.swhisper', 'FUTURE_SETTING=keep me\nSWHISPER_NUM_SPEAKERS=2\n')
        result = webui.api_put_sidecar('a', {'_comment': 'new note', 'SWHISPER_NUM_SPEAKERS': '3'})
        self.assertTrue(result['exists'])
        self.assertEqual(parse_sidecar_file(str(self.root / 'a.swhisper'))['FUTURE_SETTING'], 'keep me')
        webui.api_put_sidecar('a', {'_comment': ''})
        self.assertEqual(parse_sidecar_file(str(self.root / 'a.swhisper')), {'FUTURE_SETTING': 'keep me'})

    def test_comment_edit_preserves_mtime_with_unknown_keys(self):
        self.write('a.wav')
        p = self.write('a.swhisper', 'FUTURE_SETTING=keep\nSWHISPER_NUM_SPEAKERS=2\n')
        os.utime(p, (1000000, 1000000))
        webui.api_put_sidecar('a', {'_comment': 'comment only', 'SWHISPER_NUM_SPEAKERS': '2'})
        self.assertEqual(p.stat().st_mtime, 1000000)

    def test_invalid_save_does_not_touch_disk(self):
        p = self.write('a.swhisper', 'SWHISPER_NUM_SPEAKERS=2\n')
        before = p.read_bytes()
        result = webui.api_put_sidecar('a', {'SWHISPER_NUM_SPEAKERS': 'nope'})
        self.assertEqual(result.status_code, 422)
        self.assertEqual(p.read_bytes(), before)

    def test_preview_lists_existing_json_even_without_audio(self):
        self.write('new.m4a')
        self.write('whisper-json-output/orphan.json', '{}')
        rows = self.preview()['files']
        self.assertTrue(any(r['file'] == 'new.m4a' and r['action'] == 'process' for r in rows))
        self.assertTrue(any(r['file'] == 'orphan' and r['action'] == 'blocked' for r in rows))
        self.assertTrue(any(r['file'] == 'new.m4a' and 'depends on' in r['reason'] for r in rows))

    def test_force_and_sidecar_override_and_skip(self):
        self.write('a.wav')
        self.write('whisper-json-output/a.json', '{}')
        marker = self.write('transcripts/a.ok')
        sidecar = self.write('a.swhisper', 'SWHISPER_DIARIZE_FORCE_REPROCESS=false\n')
        os.utime(sidecar, (10, 10))
        self.dc.force_reprocess = True
        rows = self.preview('diarize4')['files']
        self.assertEqual(rows[0]['action'], 'skip')
        sidecar.write_text('SWHISPER_DIARIZE_FORCE_REPROCESS=true\n')
        self.assertEqual(self.preview('diarize4')['files'][0]['action'], 'reprocess')

    def test_v5_uses_transcription_selection(self):
        self.write('a.wav')
        self.write('whisper-json-output/a.json', '{}')
        sidecar = self.write('a.swhisper', 'SWHISPER_DIARIZE_FORCE_REPROCESS=true\n')
        os.utime(sidecar, (10, 10))
        rows = self.preview('transcribe5')['files']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['action'], 'skip')

    def test_file_limit_and_global_force(self):
        for name in ['a', 'b']:
            self.write(name + '.wav')
            self.write('whisper-json-output/' + name + '.json', '{}')
        self.tc.force_reprocess = True
        self.tc.file_limit = 1
        rows = [r for r in self.preview()['files'] if r['stage'] == 'Transcription']
        self.assertEqual([r['action'] for r in rows].count('reprocess'), 1)
        self.assertEqual([r['reason'] for r in rows].count('Outside the configured file limit'), 1)

    def test_sidecar_staleness_and_token(self):
        self.write('a.wav')
        self.write('whisper-json-output/a.json', '{}')
        first = self.preview()
        self.assertEqual(first['token'], self.preview()['token'])
        self.write('a.swhisper', 'SWHISPER_LANGUAGE=sv\n')
        second = self.preview()
        self.assertNotEqual(first['token'], second['token'])
        self.assertEqual(second['files'][0]['action'], 'reprocess')

    def test_mismatched_json_directories_explained(self):
        self.write('a.wav')
        self.dc.json_input_dir = 'other-json'
        result = self.preview()
        self.assertEqual(len(result['files']), 1)
        self.assertTrue(any('will not be picked up' in n for n in result['notes']))

    def test_stale_launch_rejected_before_process_start(self):
        with patch.object(webui, 'api_run_preview', return_value={'token': 'new'}), patch.object(self.state, 'start') as start:
            with self.assertRaises(webui.HTTPException) as error:
                webui.api_run({'pipeline': 'transcribe4', 'preview_token': 'old'})
            self.assertEqual(error.exception.status_code, 409)
            start.assert_not_called()

    def test_progress_counts_success_not_skips_or_failure(self):
        lines = ['📦 Audio file 1/2: a.wav', '🎵 Processing (1/2): a.wav',
                 '✅ Saved 30 segments, 500 characters', '📦 Audio file 2/2: b.wav',
                 '❌ Skipping b.wav (conversion failed)', '🎯 Launching diarization pipeline...',
                 '🎯 DIARIZATION LOG FOR: a.wav', 'Processing file 1 of 2',
                 '📄 Completion marker created: a.ok',
                 '2026-09-13 - Skipping file 2 of 2: b.wav']
        p = inspect_progress(lines, 'transcribe4')
        self.assertEqual(p['completed'], {'Transcription': 1, 'Diarization': 1})
        self.assertEqual(p['current_file'], 'b.wav')
        self.assertEqual(p['total_files'], 2)
        self.assertEqual(p['error_count'], 1)

    def test_progress_not_started_and_offline_persistent_log(self):
        self.assertIsNone(inspect_progress([], None)['current_file'])
        log = self.write('run.log', '📦 Audio file 1/1: a.wav\n❌ Error during transcription: broken\n')
        self.state.run_log_path = str(log)
        self.state.pipeline = 'transcribe4'
        self.state.returncode = 0
        status = webui.api_run_status()
        self.assertEqual(status['progress']['error_count'], 1)
        self.assertEqual(status['progress']['current_file'], 'a.wav')
        self.assertFalse(status['running'])

    def test_log_reads_only_what_was_appended(self):
        log = self.write('run.log', '')
        read, offset = webui.AppState._read_lines_from(str(log), 0)
        self.assertEqual((read, offset), ([], 0))
        log.write_bytes(b'first\nhalf')
        read, offset = webui.AppState._read_lines_from(str(log), offset)
        self.assertEqual(read, ['first'])          # the half line is withheld
        log.write_bytes(b'first\nhalf line\n')
        read, offset = webui.AppState._read_lines_from(str(log), offset)
        self.assertEqual(read, ['half line'])      # and never repeated
        self.assertEqual(webui.AppState._read_lines_from(str(log), offset), ([], offset))

    def test_log_survives_split_newlines_and_replacement(self):
        log = self.write('run.log', '')
        log.write_bytes(b'one\r')
        read, offset = webui.AppState._read_lines_from(str(log), 0)
        self.assertEqual(read, ['one'])
        log.write_bytes(b'one\r\ntwo\n')          # CRLF split across two reads
        self.assertEqual(webui.AppState._read_lines_from(str(log), offset)[0], ['two'])
        log.write_bytes(b'short\n')                # a new run replaces the file
        self.assertEqual(webui.AppState._read_lines_from(str(log), 99)[0], ['short'])

    def test_incremental_progress_matches_a_full_scan(self):
        lines = ['📦 Audio file 1/2: a.wav', '✅ Saved 30 segments, 500 characters',
                 '🎯 DIARIZATION LOG FOR: a.wav', '📄 Completion marker created: a.ok',
                 'ERROR: something broke']
        tracker = ProgressTracker('transcribe4')
        for line in lines:
            tracker.feed([line])
        self.assertEqual(tracker.snapshot(), inspect_progress(lines, 'transcribe4'))
        self.assertEqual(tracker.line_count, len(lines))

    def test_indexed_lookups_match_the_per_file_scan(self):
        import webui_files as wf
        dirs = self.state.dirs()
        for name in ["a.wav", "b.wav", "c d.wav"]:
            self.write(name)
        for name in ["transcripts/txt/a.txt", "transcripts/txt/b_20260913-120000.txt",
                     "transcripts/logs/a_20260913-120000.log",
                     "transcripts/logs/b_2026-09-13 120000.log",
                     "transcripts/logs/a_silence_gaps.log"]:
            self.write(name, "x")
        bases = wf._basenames(dirs["audio"])
        fast_t, fast_l = wf.transcript_bases(dirs), wf.log_bases(dirs)
        self.assertEqual({b for b in bases if b in fast_t},
                         {b for b in bases if wf._transcript_versions(dirs, b)})
        self.assertEqual({b for b in bases if b in fast_l},
                         {b for b in bases if wf._log_versions(dirs, b)})
        self.assertEqual({b for b in bases if b in fast_t}, {"a", "b"})
        self.assertEqual({b for b in bases if b in fast_l}, {"a", "b"})

        # A Scribe export is named after its work folder, so the index has to
        # resolve it through the recorded identity exactly as the scan does.
        import json
        work = self.root / "scribe-webui" / "run1"
        (work / "transcripts" / "txt").mkdir(parents=True)
        (work / "transcripts" / "txt" / "audio_20260914-030512.txt").write_text("t")
        (work / "scribe-state.json").write_text(
            json.dumps({"identity": {"source": str(self.root.resolve() / "c d.wav")}}))
        fast_t = wf.transcript_bases(dirs)
        self.assertEqual({b for b in bases if b in fast_t},
                         {b for b in bases if wf._transcript_versions(dirs, b)})
        self.assertIn("c d", fast_t)

    def test_run_preview_and_launch_honour_a_selection(self):
        from fastapi.testclient import TestClient
        for name in ["a.wav", "b.wav", "c.wav"]:
            self.write(name)
        with TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            folder = client.get("/api/run/preview", params={"pipeline": "transcribe4"}).json()
            chosen = client.get("/api/run/preview",
                                params={"pipeline": "transcribe4", "only": ["a", "c"]}).json()
            self.assertIsNone(folder["only"])
            self.assertEqual(chosen["only"], ["a", "c"])
            self.assertEqual({r["base"] for r in chosen["files"]}, {"a", "c"})
            self.assertIn("restricted to 2", chosen["notes"][0])
            # A selection is part of what the token commits to.
            self.assertNotEqual(folder["token"], chosen["token"])
            self.assertEqual(client.post("/api/run", json={"pipeline": "transcribe4",
                                                          "only": ["a"],
                                                          "preview_token": folder["token"]}).status_code, 409)
            for bad in ([], ["missing"], ["../escape"]):
                self.assertEqual(client.post("/api/run", json={"pipeline": "transcribe4",
                                                              "only": bad}).status_code, 400, bad)
            self.assertEqual(client.post("/api/run", json={"pipeline": "transcribe4",
                                                          "only": "a"}).status_code, 422)

    def test_selection_reaches_the_child_environment(self):
        seen = {}

        class FakeProcess:
            pid = 4321
            def poll(self):
                return None

        def fake_popen(args, **kwargs):
            seen["only"] = kwargs["env"].get(webui.ONLY_FILES_ENV)
            return FakeProcess()

        self.write("a.wav")
        with patch("subprocess.Popen", fake_popen), patch.object(threading, "Thread"):
            self.state.start("transcribe4", ["a"])
            self.assertEqual(seen["only"], "a")
            self.state.process = None
            self.state.pid = None
            self.state.start("transcribe4", None)
            self.assertIsNone(seen["only"])

    def test_old_run_logs_and_history_are_pruned(self):
        run_dir = self.root / webui.UI_RUN_LOG_DIR
        run_dir.mkdir(parents=True, exist_ok=True)
        for index in range(webui.MAX_RUN_LOGS + 12):
            path = run_dir / f"ui-run-20260913-1200{index:02d}-transcribe4-{index:06d}.log"
            path.write_text("x")
            os.utime(path, (index, index))
        keep = run_dir / "ui-run-20260913-120000-transcribe4-keepme.log"
        keep.write_text("active")
        os.utime(keep, (0, 0))          # oldest, but it is the current run
        history = self.root / webui.HISTORY_FILENAME
        history.write_text("".join(f'{{"n": {i}}}\n' for i in range(webui.MAX_HISTORY_ENTRIES + 40)))
        self.state.run_log_path = str(keep)
        self.state._prune_old_runs(str(self.root))
        remaining = sorted(p.name for p in run_dir.glob("ui-run-*.log"))
        # The newest MAX_RUN_LOGS survive, plus the active run's own log, which
        # is the oldest here and would otherwise have been deleted underneath it.
        self.assertEqual(len(remaining), webui.MAX_RUN_LOGS + 1)
        self.assertIn(keep.name, remaining)
        kept_history = history.read_text().splitlines()
        self.assertEqual(len(kept_history), webui.MAX_HISTORY_ENTRIES)
        self.assertIn(str(webui.MAX_HISTORY_ENTRIES + 39), kept_history[-1])

    def test_file_list_warns_when_the_json_folders_differ(self):
        from fastapi.testclient import TestClient
        self.write("a.wav")
        with TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            self.assertIsNone(client.get("/api/files").json()["note"])
            with patch.dict(os.environ, {"JSON_INPUT_DIR": "other-json"}):
                note = client.get("/api/files").json()["note"]
            self.assertIn("other-json", note)
            self.assertIn("whisper-json-output", note)

    def test_diarization_first_current_step(self):
        progress = inspect_progress(['🎵 Processing (1/3): a.wav',
                                     '🎤 Step 4: Transcribing chunks with speaker-guided VAD...'], 'transcribe5')
        self.assertEqual(progress['current_file'], 'a.wav')
        self.assertIn('Transcribing chunks', progress['detail'])
        self.assertEqual(progress['completed']['Transcription'], 0)

    def test_http_routes_without_model_processes(self):
        from fastapi.testclient import TestClient
        self.write('a.wav')
        with TestClient(webui.app, base_url="http://127.0.0.1:8722") as client:
            preview = client.get('/api/run/preview', params={'pipeline': 'transcribe4'})
            self.assertEqual(preview.status_code, 200)
            self.assertIn('token', preview.json())
            self.assertEqual(client.get('/api/run/preview', params={'pipeline': 'bad'}).status_code, 400)
            self.assertEqual(client.get('/api/run/status').status_code, 200)
            result = client.put('/api/sidecar/a', json={'SWHISPER_NUM_SPEAKERS': '3', '_comment': 'note'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(client.get('/api/sidecar/a').json()['comment'], 'note')


if __name__ == '__main__':
    unittest.main()
