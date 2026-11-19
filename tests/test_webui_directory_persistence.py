"""Folder selection survives server restarts without changing pipeline defaults."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import path_settings
import webui


class DirectoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.old = self.root / "old"
        self.selected = self.root / "Interview data å # selected"
        self.old.mkdir()
        self.selected.mkdir()
        self.env = self.root / ".swhisper.env"
        self.env.write_text(f'SWHISPER_AUDIO_DIR="{self.old}"\nOTHER=keep\n')
        for mocked in [
            patch.object(path_settings, "__file__", str(self.root / "path_settings.py")),
            patch.object(webui, "get_path_settings", return_value=SimpleNamespace(audio_dir=self.old)),
        ]:
            mocked.start()
            self.addCleanup(mocked.stop)

    def select(self):
        state = webui.AppState()
        with patch.object(webui, "state", state):
            webui.api_set_audio_dir({"path": str(self.selected)})
        return state

    def test_selection_survives_fresh_server_state(self):
        self.assertEqual(webui.AppState().audio_dir(), str(self.old))
        state = self.select()
        self.assertEqual(state.audio_dir(), str(self.selected))
        restarted = webui.AppState()
        self.assertEqual(restarted.audio_dir(), str(self.selected))
        self.assertFalse(restarted.audio_dir_pinned)
        self.assertIn(f'SWHISPER_AUDIO_DIR="{self.old}"\n', self.env.read_text())
        self.assertIn("OTHER=keep\n", self.env.read_text())
        self.assertIn(f'#SWHISPER_AUDIO_DIR="{self.selected}"\n', self.env.read_text())
        self.assertIn(str(self.selected), path_settings.get_commented_audio_dirs())
        # Re-selecting the same folder must not duplicate it in the env history.
        self.select()
        self.assertEqual(path_settings.get_commented_audio_dirs().count(str(self.selected)), 1)

    def test_explicit_startup_directory_wins(self):
        self.select()
        explicit = webui.AppState(str(self.old))
        self.assertEqual(explicit.audio_dir(), str(self.old))
        self.assertTrue(explicit.audio_dir_pinned)
        self.assertEqual(webui.AppState().audio_dir(), str(self.selected))

    def test_unavailable_or_corrupt_saved_directory_falls_back(self):
        for saved in ["broken JSON", "null", "[]", "{}",
                      json.dumps({"audio_dir": "relative/path"}),
                      json.dumps({"audio_dir": str(self.root / "missing")})]:
            with self.subTest(saved=saved):
                (self.root / ".swhisper-webui.json").write_text(saved)
                self.assertEqual(webui.AppState().audio_dir(), str(self.old))

    def test_failed_save_keeps_previous_selection(self):
        self.select()
        state = webui.AppState()
        with patch.object(webui, "state", state), patch.object(
            webui, "remember_webui_audio_dir", side_effect=OSError("disk full")
        ):
            with self.assertRaises(webui.HTTPException) as caught:
                webui.api_set_audio_dir({"path": str(self.old)})
        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(state.audio_dir(), str(self.selected))
        self.assertEqual(webui.AppState().audio_dir(), str(self.selected))

    def test_running_pipeline_rejects_selection_without_saving(self):
        state = webui.AppState()
        with patch.object(webui, "state", state), patch.object(state, "running", return_value=True):
            with self.assertRaises(webui.HTTPException) as caught:
                webui.api_set_audio_dir({"path": str(self.selected)})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertFalse((self.root / ".swhisper-webui.json").exists())
