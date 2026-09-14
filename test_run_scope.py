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
