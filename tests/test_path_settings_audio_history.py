"""Audio-directory history keeps the latest selection active across a restart."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import path_settings


class RememberAudioDirectoryTests(unittest.TestCase):
    def test_latest_selection_is_a_unique_history_entry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".swhisper.env").write_text(
                'SWHISPER_AUDIO_DIR="/recordings/older"\n'
                '#SWHISPER_AUDIO_DIR="/recordings/earlier"\n',
                encoding="utf-8",
            )
            with patch.object(path_settings, "__file__", str(root / "path_settings.py")):
                path_settings.remember_audio_dir("/recordings/latest")
                saved = (root / ".swhisper.env").read_text(encoding="utf-8")
                history = path_settings.get_commented_audio_dirs()

        self.assertIn('SWHISPER_AUDIO_DIR="/recordings/older"', saved)
        self.assertIn('#SWHISPER_AUDIO_DIR="/recordings/latest"', saved)
        self.assertNotIn('#SWHISPER_AUDIO_DIR="/recordings/older"', saved)
        self.assertEqual(history, ["/recordings/earlier", "/recordings/latest"])


if __name__ == "__main__":
    unittest.main()
