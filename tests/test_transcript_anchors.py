"""Time anchors that let the comparison view line up two transcripts by time."""
import unittest

from transcript_render import TXTFormatter, anchors_from_txt


def words(*items):
    return [{'word': text, 'start': start, 'end': end} for text, start, end in items]


SEGMENTS = [
    {'start': 1.0, 'end': 3.0, 'speaker': 'S01', 'words': words(('Hello', 1.0, 2.0), ('world', 2.0, 3.0))},
    {'start': 3.5, 'end': 5.0, 'speaker': 'S01', 'words': words(('again', 3.5, 5.0))},
    {'start': 6.0, 'end': 9.5, 'speaker': 'S02', 'words': words(('Hi', 6.0, 6.5), ('(2.0)', 6.5, 8.5), ('there', 8.5, 9.5))},
    {'start': 3700.0, 'end': 3702.0, 'speaker': 'S01', 'words': words(('Late', 3700.0, 3702.0))},
]


class Config:
    silence_gap_linebreak_threshold = 1.0
    output_preamble = "Preamble line one\nPreamble line two"

    def get_preamble_with_transcript_id(self, transcript_id):
        return self.output_preamble


class FormatterAnchorTests(unittest.TestCase):
    def test_render_with_anchors_returns_the_same_text_as_render(self):
        text, _ = TXTFormatter().render_with_anchors(SEGMENTS, config=Config(), stats=False)
        self.assertEqual(text, TXTFormatter().render(SEGMENTS, config=Config(), stats=False))

    def test_each_anchor_points_at_its_paragraph_header(self):
        text, anchors = TXTFormatter().render_with_anchors(SEGMENTS, config=Config(), stats=False)
        lines = text.split('\n')
        self.assertEqual([lines[a['line']] for a in anchors],
                         ['S01 [00:00:01]:', 'S02 [00:00:06]:', 'S01 [01:01:40]:'])

    def test_anchors_span_the_first_to_the_last_segment_of_a_paragraph(self):
        _, anchors = TXTFormatter().render_with_anchors(SEGMENTS, config=Config(), stats=False)
        self.assertEqual([(a['start'], a['end']) for a in anchors],
                         [(1.0, 5.0), (6.0, 9.5), (3700.0, 3702.0)])

    def test_anchor_lines_survive_pause_breaks_inside_a_paragraph(self):
        segments = [dict(s) for s in SEGMENTS]
        segments[2]['words'] = [dict(w) for w in segments[2]['words']]
        segments[2]['words'][1]['is_silence_marker'] = True
        text, anchors = TXTFormatter().render_with_anchors(segments, config=Config(), stats=False)
        self.assertIn('\n\n(2.0)\n\n', text)
        self.assertEqual(text.split('\n')[anchors[-1]['line']], 'S01 [01:01:40]:')


class TxtAnchorTests(unittest.TestCase):
    def test_headers_become_anchors_that_end_where_the_next_begins(self):
        text = TXTFormatter().render(SEGMENTS, config=Config(), stats=False)
        anchors = anchors_from_txt(text)
        self.assertEqual([(a['start'], a['end']) for a in anchors],
                         [(1.0, 6.0), (6.0, 3700.0), (3700.0, 3700.0)])
        self.assertEqual(text.split('\n')[anchors[1]['line']], 'S02 [00:00:06]:')

    def test_renamed_speakers_and_body_text_are_handled(self):
        text = "Intro\n\nAnna Berg [00:01:02]:\nShe said [00:00:05]: nothing\n\nS02 [00:01:10]:\nOk\n"
        self.assertEqual([(a['line'], a['start']) for a in anchors_from_txt(text)], [(2, 62.0), (5, 70.0)])

    def test_text_without_headers_has_no_anchors(self):
        self.assertEqual(anchors_from_txt("Just a transcript"), [])


if __name__ == '__main__':
    unittest.main()
