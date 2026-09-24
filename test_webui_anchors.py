"""The transcript and render endpoints return the anchors the comparison view syncs on."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import path_settings
import webui
from test_transcript_anchors import SEGMENTS


class ApiAnchorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        p = patch.object(webui, 'state', webui.AppState(str(self.root)));p.start();self.addCleanup(p.stop)
        p = patch.dict(os.environ);p.start();self.addCleanup(p.stop)
        for key in list(os.environ):
            if key.startswith('SWHISPER_') or key == 'JSON_INPUT_DIR': os.environ.pop(key)
        p = patch.object(path_settings, '_load_env_file', lambda: None);p.start();self.addCleanup(p.stop)
        self.client = TestClient(webui.app, base_url="http://127.0.0.1:8722")
        self.addCleanup(self.client.close)
        (self.root / 'a.wav').write_text('mock audio')
        (self.root / 'transcripts/txt').mkdir(parents=True)
        (self.root / 'transcripts/json').mkdir(parents=True)

    def test_render_returns_anchors_from_segment_times(self):
        (self.root / 'transcripts/txt/a_20260913-120000.txt').write_text('stored')
        (self.root / 'transcripts/json/a_20260913-120000_diarized.json').write_text(json.dumps({'segments': SEGMENTS}))
        data = self.client.post('/api/render/a', json={'options': {}}).json()
        lines = data['content'].split('\n')
        self.assertEqual([(lines[a['line']], a['start'], a['end']) for a in data['anchors']][0],
                         ('S01 [00:00:01]:', 1.0, 5.0))

    def test_stored_transcript_returns_anchors_from_its_headers(self):
        (self.root / 'transcripts/txt/a_20260913-120000.txt').write_text("S01 [00:00:04]:\nHello\n")
        data = self.client.get('/api/transcript/a').json()
        self.assertEqual(data['anchors'], [{'line': 0, 'start': 4.0, 'end': 4.0}])


if __name__ == '__main__':
    unittest.main()
