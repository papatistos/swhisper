"""Waveform extraction and API checks using generated audio only."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import webui
import webui_waveform


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_stereo_peaks_preserve_bursts_and_silence(self):
        samples = np.zeros((16000, 2), dtype=np.float32)
        samples[100, 0] = .75
        samples[8100, 1] = -.5
        path = self.root / 'a.wav'
        sf.write(path, samples, 8000, subtype='FLOAT')
        before = path.read_bytes()
        result = webui_waveform.waveform(path)
        self.assertEqual(result['duration'], 2)
        self.assertEqual(len(result['peaks']), 2048)
        self.assertEqual(max(result['peaks'][:1024]), .75)
        self.assertEqual(max(result['peaks'][1024:]), .5)
        self.assertEqual(sum(p > 0 for p in result['peaks']), 2)
        self.assertEqual(path.read_bytes(), before)

    def test_short_silent_file_and_cache_invalidation(self):
        path = self.root / 'a.wav'
        sf.write(path, np.zeros(3), 8000, subtype='FLOAT')
        first = webui_waveform.waveform(path)
        self.assertEqual(first['peaks'], [0, 0, 0])
        with patch.object(sf, 'SoundFile', side_effect=AssertionError('Cache missed')):
            self.assertIs(webui_waveform.waveform(path), first)
        previous = path.stat()
        sf.write(path, np.ones(3) * .5, 8000, subtype='FLOAT')
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
        self.assertEqual(webui_waveform.waveform(path)['peaks'], [.5, .5, .5])

    def test_zoom_reads_the_requested_window_at_higher_resolution(self):
        path = self.root / 'a.wav'
        samples = np.zeros(80000, dtype=np.float32)
        samples[40000] = .4
        samples[40008] = .8
        sf.write(path, samples, 8000, subtype='FLOAT')
        overview = webui_waveform.waveform(path)
        detail = webui_waveform.waveform(path, 5, 5.1)
        self.assertEqual(detail['duration'], 10)
        self.assertEqual(detail['start'], 5)
        self.assertAlmostEqual(detail['end'], 5.1)
        self.assertEqual(sum(p > 0 for p in overview['peaks']), 1)
        self.assertEqual(sum(p > 0 for p in detail['peaks']), 2)
        self.assertEqual(detail['peaks'][0], .4)
        self.assertEqual(detail['peaks'][8], .8)
        self.assertEqual(max(webui_waveform.waveform(path, 6, 7)['peaks']), 0)
        self.assertEqual(webui_waveform.waveform(path, 9, 20)['end'], 10)
        with patch.object(sf, 'SoundFile', side_effect=AssertionError('Cache missed')):
            self.assertIs(webui_waveform.waveform(path, 5, 5.1), detail)

    def test_invalid_and_outside_windows_are_rejected(self):
        path = self.root / 'a.wav'
        sf.write(path, np.zeros(8000), 8000)
        for start, end in [(-1, .5), (1, 1), (.5, .4), (float('nan'), 1), (0, float('inf')), (2, 3)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                webui_waveform.waveform(path, start, end)

    def test_compressed_fallback(self):
        wav, m4a = self.root / 'a.wav', self.root / 'a.m4a'
        samples = np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000) * .5
        sf.write(wav, np.column_stack([samples, -samples]), 8000)
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(wav), str(m4a)], check=True)
        result = webui_waveform.waveform(m4a)
        self.assertAlmostEqual(result['duration'], 1, delta=.2)
        self.assertTrue(0 < len(result['peaks']) <= 2048)
        self.assertGreater(max(result['peaks']), .3)
        detail = webui_waveform.waveform(m4a, .2, .25)
        self.assertAlmostEqual(detail['duration'], 1, delta=.2)
        self.assertAlmostEqual(detail['start'], .2)
        self.assertAlmostEqual(detail['end'], .25, delta=1/8000)
        self.assertGreater(len(detail['peaks']), 100)
        self.assertGreater(max(detail['peaks']), .3)

    def test_api_missing_invalid_and_wav_preference(self):
        sf.write(self.root / 'a.wav', np.zeros(8000), 8000)
        (self.root / 'a.mp3').write_bytes(b'invalid compressed audio')
        (self.root / 'broken.wav').write_bytes(b'invalid audio')
        with patch.object(webui, 'state', webui.AppState(str(self.root))):
            with TestClient(webui.app, base_url='http://127.0.0.1:8722') as client:
                response = client.get('/api/waveform/a')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['duration'], 1)
                detail = client.get('/api/waveform/a?start=0.2&end=0.4')
                self.assertEqual(detail.status_code, 200, detail.text)
                self.assertEqual((detail.json()['start'], detail.json()['end']), (.2, .4))
                self.assertEqual(client.get('/api/waveform/a?start=nan').status_code, 422)
                self.assertEqual(client.get('/api/waveform/a?start=0.6&end=0.3').status_code, 422)
                for name in ['webui_waveform_speakers.js', 'webui_waveform_speakers.css']:
                    self.assertEqual(client.get('/' + name).status_code, 200)
                    self.assertIn('/' + name + '?v=', client.get('/').text)
                self.assertEqual(client.get('/api/waveform/missing').status_code, 404)
                self.assertEqual(client.get('/api/waveform/broken').status_code, 422)
                self.assertEqual(client.get('/api/waveform/..%5Csecret').status_code, 400)


if __name__ == '__main__':
    unittest.main()
