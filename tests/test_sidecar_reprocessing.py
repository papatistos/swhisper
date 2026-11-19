"""Reprocessing decisions for saved force options and actual settings changes."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidecar import STAGE_DIARIZE, STAGE_TRANSCRIBE, sidecar_requests_reprocess


class SidecarReprocessingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.output = self.root / 'a.output'
        self.output.write_text('existing output')
        os.utime(self.output, (100, 100))
        self.sidecar = self.root / 'a.swhisper'
        env = patch.dict(os.environ, {'SWHISPER_SIDECAR_AUTOREPROCESS': 'true'})
        env.start()
        self.addCleanup(env.stop)

    def settings(self, text):
        self.sidecar.write_text(text)
        os.utime(self.sidecar, (200, 200))

    def decision(self, stage):
        return sidecar_requests_reprocess(str(self.root), 'a', str(self.output), stage)

    def test_newer_disabled_force_options_do_not_invalidate_either_stage(self):
        for key in ('SWHISPER_FORCE_REPROCESS', 'SWHISPER_DIARIZE_FORCE_REPROCESS'):
            for value in ('false', '0', 'no', 'off'):
                with self.subTest(key=key, value=value):
                    self.settings(f'{key}={value}\n')
                    for stage in (STAGE_TRANSCRIBE, STAGE_DIARIZE):
                        self.assertEqual(self.decision(stage), (False, ''))

    def test_invalid_force_option_does_not_invalidate_output(self):
        self.settings('SWHISPER_FORCE_REPROCESS=invalid\n')
        self.assertEqual(self.decision(STAGE_TRANSCRIBE), (False, ''))
        self.assertEqual(self.decision(STAGE_DIARIZE), (False, ''))

    def test_disabling_force_after_a_run_clears_the_request(self):
        self.settings('SWHISPER_DIARIZE_FORCE_REPROCESS=true\n')
        self.assertTrue(self.decision(STAGE_DIARIZE)[0])
        self.settings('SWHISPER_DIARIZE_FORCE_REPROCESS=false\n')
        self.assertFalse(self.decision(STAGE_DIARIZE)[0])

    def test_enabled_force_remains_active_after_successful_output(self):
        for key, stage in (('SWHISPER_FORCE_REPROCESS', STAGE_TRANSCRIBE),
                           ('SWHISPER_DIARIZE_FORCE_REPROCESS', STAGE_DIARIZE)):
            with self.subTest(stage=stage):
                self.settings(f'{key}=true\n')
                os.utime(self.output, (300, 300))
                self.assertEqual(self.decision(stage), (True, f'{key}=true in sidecar'))

    def test_disabled_force_does_not_hide_changed_transcription_settings(self):
        self.settings('SWHISPER_FORCE_REPROCESS=false\nSWHISPER_LANGUAGE=sv\n')
        for stage in (STAGE_TRANSCRIBE, STAGE_DIARIZE):
            self.assertEqual(self.decision(stage), (True, 'sidecar modified after last processing'))

    def test_disabled_force_does_not_hide_changed_speaker_settings(self):
        self.settings('SWHISPER_FORCE_REPROCESS=false\nSWHISPER_NUM_SPEAKERS=2\n')
        self.assertFalse(self.decision(STAGE_TRANSCRIBE)[0])
        self.assertTrue(self.decision(STAGE_DIARIZE)[0])

    def test_interview_settings_do_not_invalidate_processing(self):
        for setting in ('RECORDING_TYPE=interview', 'RECORDING_TYPE=',
                        'INTERVIEWER_NAME=Chris', 'INTERVIEWEE_NAME=Anna'):
            with self.subTest(setting=setting):
                self.settings(setting + '\n')
                self.assertFalse(self.decision(STAGE_TRANSCRIBE)[0])
                self.assertFalse(self.decision(STAGE_DIARIZE)[0])

    def test_new_transcription_force_still_invalidates_older_diarization(self):
        self.settings('SWHISPER_FORCE_REPROCESS=true\n')
        self.assertTrue(self.decision(STAGE_DIARIZE)[0])


if __name__ == '__main__':
    unittest.main()
