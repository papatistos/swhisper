"""The macOS folder picker endpoint returns a path without changing the audio directory."""
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import webui


class DirPickTests(unittest.TestCase):
    def test_pick_returns_dialog_path_without_switching_directory(self):
        picked = tempfile.mkdtemp(prefix="picked audio ")
        self.addCleanup(shutil.rmtree, picked, ignore_errors=True)

        def fake_run(argv, **kwargs):
            self.assertEqual(argv[0], "osascript")
            return subprocess.CompletedProcess(argv, 0, stdout=picked + "/\n", stderr="")

        with patch.object(webui.subprocess, "run", fake_run), \
             patch.object(webui.state, "running", return_value=False):
            result = webui.api_pick_audio_dir()
        self.assertEqual(result["path"], os.path.realpath(picked))

    def test_canceled_dialog_is_a_clear_error(self):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="execution error: User canceled. (-128)")

        with patch.object(webui.subprocess, "run", fake_run):
            with self.assertRaises(webui.HTTPException) as ctx:
                webui.api_pick_audio_dir()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("canceled", ctx.exception.detail.lower())

    def test_failed_dialog_reports_osascript_error(self):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="some osascript failure")

        with patch.object(webui.subprocess, "run", fake_run):
            with self.assertRaises(webui.HTTPException) as ctx:
                webui.api_pick_audio_dir()
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("some osascript failure", ctx.exception.detail)

    def test_non_directory_result_is_rejected(self):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="/tmp/does-not-exist-9ef3\n", stderr="")

        with patch.object(webui.subprocess, "run", fake_run):
            with self.assertRaises(webui.HTTPException) as ctx:
                webui.api_pick_audio_dir()
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
