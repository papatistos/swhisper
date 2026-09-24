"""SWHISPER_ONLY_FILES selection, with no pipeline or model imports."""
import os
import unittest
from unittest.mock import patch

import run_scope


class RunScopeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(run_scope.ENV_VAR, None)

    def set(self, value):
        os.environ[run_scope.ENV_VAR] = value

    def test_unset_runs_the_whole_folder(self):
        self.assertIsNone(run_scope.selected_bases())
        self.assertEqual(run_scope.keep_selected(['a.wav', 'b.wav']), ['a.wav', 'b.wav'])
        self.assertEqual(run_scope.describe(), '')

    def test_blank_value_is_not_an_empty_selection(self):
        self.set('   \n  \n')
        self.assertIsNone(run_scope.selected_bases())
        self.assertEqual(run_scope.keep_selected(['a.wav']), ['a.wav'])

    def test_extension_and_directory_are_ignored(self):
        self.set('a.wav\n/somewhere/b.m4a\nc')
        self.assertEqual(run_scope.selected_bases(), {'a', 'b', 'c'})
        self.assertEqual(run_scope.keep_selected(['a.wav', 'b.json', 'c.log', 'd.wav']),
                         ['a.wav', 'b.json', 'c.log'])

    def test_commas_in_a_filename_are_preserved(self):
        self.set('Interview, part 2.m4a')
        self.assertEqual(run_scope.selected_bases(), {'Interview, part 2'})
        self.assertEqual(run_scope.keep_selected(['Interview, part 2.m4a', 'Interview.m4a']),
                         ['Interview, part 2.m4a'])

    def test_caller_order_is_preserved(self):
        self.set('b\na')
        self.assertEqual(run_scope.keep_selected(['a.wav', 'b.wav']), ['a.wav', 'b.wav'])

    def test_describe_summarises_long_selections(self):
        self.set('\n'.join(f'file{i}' for i in range(9)))
        self.assertIn('9 selected file(s)', run_scope.describe())
        self.assertIn('and 4 more', run_scope.describe())


if __name__ == '__main__':
    unittest.main()


class RunReprocessTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(run_scope.REPROCESS_ENV_VAR, None)

    def test_reprocess_is_off_unless_set_to_a_true_value(self):
        self.assertFalse(run_scope.reprocess_requested())
        for value in ('false', '0', ''):
            os.environ[run_scope.REPROCESS_ENV_VAR] = value
            self.assertFalse(run_scope.reprocess_requested())
        for value in ('true', '1', 'Yes'):
            os.environ[run_scope.REPROCESS_ENV_VAR] = value
            self.assertTrue(run_scope.reprocess_requested())

    def test_run_reprocess_retranscribes_despite_a_sidecar_that_says_no(self):
        import tempfile
        from pathlib import Path
        from transcribe.config import TranscriptionConfig
        from transcribe.file_utils import FileManager

        with tempfile.TemporaryDirectory() as tmp:
            audio, output = Path(tmp, 'audio'), Path(tmp, 'json')
            audio.mkdir()
            output.mkdir()
            (audio / 'a.wav').write_bytes(b'')
            (audio / 'a.swhisper').write_text('SWHISPER_FORCE_REPROCESS=false\n')
            (output / 'a.json').write_text('{}')
            config = TranscriptionConfig()
            config.force_reprocess, config.file_limit = False, None
            manager = FileManager(config)
            with patch('transcribe.file_utils.ENABLE_EARLY_FORMAT_SUMMARY', False):
                self.assertEqual(manager.get_audio_files_to_process(str(audio), str(output)), [])
                os.environ[run_scope.REPROCESS_ENV_VAR] = 'true'
                self.assertEqual(manager.get_audio_files_to_process(str(audio), str(output)), ['a.wav'])
