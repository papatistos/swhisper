"""Check the renderer against transcripts the pipeline already produced.

The contract this package rests on is that rendering a diarized JSON with
default options reproduces the .txt the pipeline wrote from the same run. Run
it against a real output directory:

    python3 -m transcript_render.selftest \\
        "/path/to/Interview data/transcripts"

Exits non-zero if any transcript fails to reproduce.
"""

import json
import os
import re
import sys
import time
from typing import List, Optional

from .derive import derive_segments
from .formatters import TXTFormatter
from .markers import get_word_text, is_disfluency_marker, is_silence_token
from .options import RenderOptions

_HEADING = re.compile(r'^(?:SPEAKER_\d+|UNKNOWN|SILENCE) \[\d\d:\d\d:\d\d\]:$', re.M)


class _Config:
    """The formatter reads these two attributes off the diarization config."""

    def __init__(self, threshold: Optional[float]):
        self.silence_gap_linebreak_threshold = threshold
        self.output_preamble = ""


def _transcript_body(text: str) -> str:
    """The speaker paragraphs, without the preamble and statistics header."""
    match = _HEADING.search(text)
    return text[match.start():] if match else text


def _render(segments, options: RenderOptions, path: str) -> str:
    TXTFormatter().format(derive_segments(segments, options), path,
                          config=_Config(options.linebreak_threshold))
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def run(transcripts_dir: str) -> int:
    json_dir = os.path.join(transcripts_dir, 'json')
    txt_dir = os.path.join(transcripts_dir, 'txt')
    if not os.path.isdir(json_dir) or not os.path.isdir(txt_dir):
        print(f"Expected {json_dir} and {txt_dir} to exist.")
        return 2

    names: List[str] = sorted(f for f in os.listdir(json_dir)
                              if f.endswith('_diarized.json'))
    if not names:
        print(f"No *_diarized.json in {json_dir}")
        return 2

    scratch = os.path.join(transcripts_dir, '.selftest.txt')
    passed = failed = skipped = 0
    try:
        for name in names:
            stem = name[:-len('_diarized.json')]
            stored_path = os.path.join(txt_dir, stem + '.txt')
            if not os.path.exists(stored_path):
                skipped += 1
                print(f"  SKIP       no stored txt for {stem}")
                continue

            with open(os.path.join(json_dir, name), encoding='utf-8') as handle:
                segments = json.load(handle)['segments']

            started = time.time()
            rendered = _render(segments, RenderOptions(), scratch)
            elapsed = (time.time() - started) * 1000

            with open(stored_path, encoding='utf-8') as handle:
                stored = handle.read()

            if _transcript_body(rendered) == _transcript_body(stored):
                passed += 1
                print(f"  OK  {elapsed:5.0f}ms  {stem}")
            else:
                failed += 1
                print(f"  FAIL          {stem}")

            # Hiding markers must convert their spans into pause time rather
            # than dropping them, which is the whole reason pauses are rebuilt
            # instead of edited in place.
            hidden = derive_segments(segments, RenderOptions(disfluencies=False))
            gained = _pause_total(hidden) - _pause_total(
                derive_segments(segments, RenderOptions()))
            removed = _marker_span(segments)
            if removed > 0 and gained < removed * 0.9:
                failed += 1
                print(f"      pause budget: gained {gained:.0f}s for {removed:.0f}s "
                      f"of marker removed — expected most of it to convert")
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)

    print(f"\npassed={passed} failed={failed} skipped={skipped}")
    return 1 if failed else 0


def _pause_total(segments) -> float:
    return sum((w.get('end') or 0) - (w.get('start') or 0)
               for s in segments for w in (s.get('words') or [])
               if is_silence_token(w))


def _marker_span(segments) -> float:
    return sum((w.get('end') or 0) - (w.get('start') or 0)
               for s in segments for w in (s.get('words') or [])
               if is_disfluency_marker(get_word_text(w)))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    return run(sys.argv[1])


if __name__ == '__main__':
    raise SystemExit(main())
