#!/usr/bin/env python3
"""Re-render a diarized transcript without re-running the pipeline.

Which markers and pauses a transcript shows is recoverable from the diarized
JSON, so producing a different view is a matter of milliseconds::

    python render_transcript.py run_diarized.json --no-disfluencies -o clean.txt
    python render_transcript.py run_diarized.json --no-pauses --format rtf
    python render_transcript.py run_diarized.json --all -o out/

Hiding disfluency markers folds each marker's span into the pauses around it,
so the durations stay true to the recording rather than understating it.

With no -o, the transcript goes to stdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcript_render import (  # noqa: E402
    DEFAULT_MIN_SILENCE,
    RenderOptions,
    RTFFormatter,
    TXTFormatter,
    derive_segments,
)


def _load_diarize_defaults():
    """The preamble and threshold defaults, without importing torch."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diarize', 'config.py')
    spec = importlib.util.spec_from_file_location('swhisper_render_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DiarizationConfig()


def _config(options: RenderOptions):
    defaults = _load_diarize_defaults()

    class _Config:
        silence_gap_linebreak_threshold = options.linebreak_threshold
        output_preamble = defaults.output_preamble if options.preamble else ""
        get_preamble_with_transcript_id = defaults.get_preamble_with_transcript_id

    return _Config()


def _write(segments, options: RenderOptions, metadata: Dict[str, Any],
           fmt: str, out_path: Optional[str]) -> None:
    formatter = TXTFormatter() if fmt == 'txt' else RTFFormatter()
    text = formatter.render(
        segments,
        config=_config(options),
        transcript_id=metadata.get('timestamp'),
        speaker_stats=metadata.get('speaker_stats'),
        stats=options.stats,
    )
    if out_path is None:
        sys.stdout.write(text)
        return
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('json_path', help="a *_diarized.json written by the pipeline")
    parser.add_argument('-o', '--output',
                        help="output file, or a directory when using --all")
    parser.add_argument('--format', choices=['txt', 'rtf'], default='txt')

    parser.add_argument('--no-disfluencies', action='store_true',
                        help="hide [*] markers and fold their spans into the pauses")
    parser.add_argument('--no-pauses', action='store_true',
                        help="hide the (1.3) pause durations")
    parser.add_argument('--no-breaks', action='store_true',
                        help="do not put blank lines around long pauses")
    parser.add_argument('--break-at', type=float, default=1.0, metavar='SECONDS',
                        help="pause length that triggers a blank line (default 1.0)")
    parser.add_argument('--hide-under', type=float, default=0.0, metavar='SECONDS',
                        help="hide pauses shorter than this (default 0, show all)")
    parser.add_argument('--no-stats', action='store_true',
                        help="omit the per-speaker statistics header")
    parser.add_argument('--no-preamble', action='store_true',
                        help="omit the explanatory preamble")
    parser.add_argument('--all', action='store_true',
                        help="write every marker/pause/break combination into --output")

    args = parser.parse_args(argv)

    try:
        with open(args.json_path, encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {args.json_path}: {exc}", file=sys.stderr)
        return 2

    segments = data.get('segments') or []
    metadata = data.get('metadata') or {}
    floor = metadata.get('min_silence_duration')
    if not isinstance(floor, (int, float)):
        floor = DEFAULT_MIN_SILENCE

    base = dict(
        pause_linebreak_threshold=args.break_at,
        min_pause=args.hide_under,
        stats=not args.no_stats,
        preamble=not args.no_preamble,
    )
    stem = os.path.basename(args.json_path)
    for suffix in ('_diarized.json', '.json'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    if args.all:
        directory = args.output or '.'
        os.makedirs(directory, exist_ok=True)
        for disfl, pause, breaks in itertools.product([True, False], repeat=3):
            options = RenderOptions(disfluencies=disfl, pauses=pause,
                                    pause_linebreaks=breaks, **base)
            derived = derive_segments(segments, options, pipeline_min_silence=float(floor))
            path = os.path.join(directory, f"{stem}{options.suffix()}.{args.format}")
            _write(derived, options, metadata, args.format, path)
        return 0

    options = RenderOptions(
        disfluencies=not args.no_disfluencies,
        pauses=not args.no_pauses,
        pause_linebreaks=not args.no_breaks,
        **base,
    )
    derived = derive_segments(segments, options, pipeline_min_silence=float(floor))
    _write(derived, options, metadata, args.format, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
