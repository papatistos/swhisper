#!/usr/bin/env python3
"""Add or remove the preamble + speaker-stats header on an existing TXT
transcript, without re-running transcription or diarization.

The header mirrors what a fresh pipeline run writes: the configured preamble
(respecting any per-file sidecar override), optionally followed by the
"Approximate speaker statistics" table, followed by a separator line of 90
"=" characters. That separator line is also how an existing header is
detected and stripped again, so add/remove is idempotent in both directions.

The per-speaker numbers for the table come from the run's ``_analysis.json``
(saved next to the txt file, under a sibling ``stats/`` directory) if one
exists. That file only stores per-speaker word count, speech duration and
turn count, not per-speaker pause time, so the PAUSES column is shown as
"n/a" rather than a guessed number. Runs that never saved an analysis file
still get the preamble, just without the stats table underneath it, matching
what the pipeline itself does when it has nothing to tabulate.

This module deliberately does NOT import the ``diarize`` package: that pulls
in torch/pyannote/whisper and costs upwards of ten seconds to import, which
is fine for a batch pipeline run but not for an interactive toggle button.
Instead it loads ``diarize/config.py`` directly (bypassing ``diarize/__init__.py``)
to reuse the pipeline's own preamble text as the single source of truth.

CLI usage::

    python transcript_stats.py add    <audio_dir> <base> [--version <id>]
    python transcript_stats.py remove <audio_dir> <base> [--version <id>]
    python transcript_stats.py status <audio_dir> <base> [--version <id>]

Without --version, the newest txt version for that base is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sidecar import apply_sidecar_to_diarization_config  # noqa: E402  (lightweight, no ML deps)

HEADER_SEPARATOR = "=" * 90
_MAX_HEADER_SCAN_LINES = 200  # the header is always near the top; bound the scan

STATS_WARNING = (
    "Approximate speaker statistics:\n"
    "(may be very inaccurate, especially if number of detected speakers is incorrect)\n"
    "Reconstructed from a saved analysis file, not from a live run — per-speaker\n"
    "pause time wasn't recorded, so it is shown as n/a rather than guessed."
)

_diarization_config_module = None  # lazy-loaded, cached


def _load_diarization_config_module():
    """Load diarize/config.py directly, bypassing diarize/__init__.py so we
    don't pull in torch/pyannote/whisper just to read the default preamble."""
    global _diarization_config_module
    if _diarization_config_module is None:
        spec = importlib.util.spec_from_file_location(
            "diarize_config_standalone", PROJECT_ROOT / "diarize" / "config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _diarization_config_module = module
    return _diarization_config_module


def has_stats_header(content: str) -> bool:
    """True if *content* already starts with a preamble+stats header."""
    return _find_header_end(content) is not None


def _find_header_end(content: str) -> Optional[int]:
    """Return the character offset right after the header block, or None."""
    lines = content.split("\n")
    for i, line in enumerate(lines[:_MAX_HEADER_SCAN_LINES]):
        if line.strip() == HEADER_SEPARATOR:
            j = i + 1
            if j < len(lines) and lines[j].strip() == "":
                j += 1
            return sum(len(l) + 1 for l in lines[:j])
    return None


def remove_header(content: str) -> str:
    """Strip an existing preamble+stats header, if any; unchanged otherwise."""
    end = _find_header_end(content)
    return content[end:] if end is not None else content


def find_analysis_json(txt_path: str) -> Optional[str]:
    """The ``_analysis.json`` a pipeline run saved alongside this txt version,
    if any (mirrors ``<txt_dir>/../stats/<stem>_analysis.json``)."""
    txt_dir = os.path.dirname(txt_path)  # .../transcripts/txt
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    stats_dir = os.path.join(os.path.dirname(txt_dir), "stats")  # .../transcripts/stats
    candidate = os.path.join(stats_dir, f"{stem}_analysis.json")
    return candidate if os.path.isfile(candidate) else None


def _load_final_speaker_stats(stats_json_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    try:
        with open(stats_json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if "final_speaker_stats" in data:
        return data["final_speaker_stats"]
    if isinstance(data.get("summary"), dict) and "final_speaker_stats" in data["summary"]:
        return data["summary"]["final_speaker_stats"]
    return None


def _speaker_sort_key(label: str) -> tuple:
    m = re.match(r"^SPEAKER_(\d+)$", label)
    if m:
        return (0, f"{int(m.group(1)):03d}")
    return (1, label)


def build_stats_table(final_speaker_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    """Render the "Approximate speaker statistics" table from the aggregate
    per-speaker numbers in an ``_analysis.json`` (words, duration, segments).

    This mirrors the columns/format of the pipeline's own table but is
    computed straight from those aggregates rather than by re-walking
    word-level segments, since that finer-grained data isn't preserved.
    """
    names = [n for n in final_speaker_stats if n not in ("UNKNOWN", "SILENCE")]
    names.sort(key=_speaker_sort_key)
    if "UNKNOWN" in final_speaker_stats:
        names.append("UNKNOWN")
    if not names:
        return []

    columns = [
        ("speaker", "", "left"),
        ("turns", "TURNS", "right"),
        ("words", "WORDS", "right"),
        ("words_per_turn", "WORD/TURN", "right"),
        ("speech", "SPEECH (s)", "right"),
        ("speed", "SPEED", "right"),
        ("pauses", "PAUSES", "right"),
    ]

    def build_row(name: str, words: float, speech: float, turns: float) -> Dict[str, str]:
        speed = (words / speech * 60.0) if speech > 0 else 0.0
        words_per_turn = (words / turns) if turns > 0 else 0.0
        return {
            "speaker": name,
            "turns": str(int(round(turns))),
            "words": str(int(round(words))),
            "words_per_turn": f"{words_per_turn:.1f}" if words_per_turn > 0 else "--",
            "speech": f"{speech:.1f}",
            "speed": f"{int(round(speed))} wpm" if speech > 0 else "--",
            "pauses": "n/a",
        }

    rows: List[Dict[str, str]] = []
    total_words = total_speech = total_turns = 0.0
    for name in names:
        stats = final_speaker_stats[name]
        words = float(stats.get("words", 0) or 0)
        speech = float(stats.get("duration", 0.0) or 0.0)
        turns = float(stats.get("segments", 0) or 0)
        rows.append(build_row(name, words, speech, turns))
        total_words += words
        total_speech += speech
        total_turns += turns

    total_row = build_row("Total", total_words, total_speech, total_turns)

    all_rows = rows + [total_row]
    widths = {key: max(len(label), max(len(r[key]) for r in all_rows))
              for key, label, _alignment in columns}

    def format_row(row: Dict[str, str]) -> str:
        parts = []
        for key, _label, alignment in columns:
            width = widths[key]
            value = row[key]
            parts.append(f"{value:<{width}}" if alignment == "left" else f"{value:>{width}}")
        return "  ".join(parts)

    header = format_row({key: label for key, label, _alignment in columns})
    lines = [STATS_WARNING, "", header, ""]
    lines.extend(format_row(row) for row in rows)
    lines.append(format_row(total_row))
    return lines


def _transcript_id_from_filename(txt_path: str, base: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    if stem.startswith(base + "_"):
        return stem[len(base) + 1:]
    return None


def _preamble_text(audio_dir: str, base: str, transcript_id: Optional[str]) -> str:
    config_module = _load_diarization_config_module()
    config, _sidecar_path = apply_sidecar_to_diarization_config(
        config_module.DiarizationConfig(), audio_dir, base)
    if not config.output_preamble:
        return ""
    if transcript_id:
        return config.get_preamble_with_transcript_id(transcript_id)
    return config.output_preamble


def build_header(audio_dir: str, base: str, txt_path: str) -> str:
    """The full preamble(+stats table) block to prepend to *txt_path*'s content."""
    transcript_id = _transcript_id_from_filename(txt_path, base)
    preamble = _preamble_text(audio_dir, base, transcript_id)
    if not preamble:
        return ""
    block = [preamble, ""]

    stats_json_path = find_analysis_json(txt_path)
    final_speaker_stats = _load_final_speaker_stats(stats_json_path) if stats_json_path else None
    table_lines = build_stats_table(final_speaker_stats) if final_speaker_stats else []
    if table_lines:
        block.extend(table_lines)
        block.append("")

    block.append(HEADER_SEPARATOR)
    block.append("")
    return "\n".join(block)


def add_stats(txt_path: str, audio_dir: str, base: str) -> bool:
    """Prepend the preamble(+stats) header. Returns True if the file changed."""
    content = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    body = remove_header(content)  # refresh rather than duplicate if already present
    header = build_header(audio_dir, base, txt_path)
    new_content = header + body
    if new_content == content:
        return False
    Path(txt_path).write_text(new_content, encoding="utf-8")
    return True


def remove_stats(txt_path: str) -> bool:
    """Strip an existing preamble(+stats) header. Returns True if it changed."""
    content = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    new_content = remove_header(content)
    if new_content == content:
        return False
    Path(txt_path).write_text(new_content, encoding="utf-8")
    return True


def _resolve_txt_path(audio_dir: str, base: str, version: Optional[str]) -> str:
    """Find the txt file for *base* — the given version id, or the newest one."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import webui  # local import: webui.py has the version-listing logic already

    state = webui.AppState()
    state.audio_dir_override = audio_dir
    dirs = state.dirs()
    versions = webui._transcript_versions(dirs, base)
    if not versions:
        raise SystemExit(f"No txt transcript found for '{base}' in {audio_dir}")
    if version:
        selected = next((v for v in versions if v["id"] == version), None)
        if selected is None:
            raise SystemExit(f"Unknown version '{version}' for '{base}'")
    else:
        selected = versions[0]
    return os.path.join(dirs["audio"], selected["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["add", "remove", "status"])
    parser.add_argument("audio_dir")
    parser.add_argument("base")
    parser.add_argument("--version", help="Transcript version id (e.g. 'transcripts/txt/<file>.txt'); default: newest")
    args = parser.parse_args()

    audio_dir = os.path.abspath(args.audio_dir)
    txt_path = _resolve_txt_path(audio_dir, args.base, args.version)

    if args.action == "status":
        content = Path(txt_path).read_text(encoding="utf-8", errors="replace")
        print(f"{txt_path}: {'has' if has_stats_header(content) else 'no'} stats header")
    elif args.action == "add":
        changed = add_stats(txt_path, audio_dir, args.base)
        print(f"{'added' if changed else 'already present'}: {txt_path}")
    else:
        changed = remove_stats(txt_path)
        print(f"{'removed' if changed else 'already absent'}: {txt_path}")


if __name__ == "__main__":
    main()
