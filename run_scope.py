"""Restrict a pipeline run to named files.

A run normally processes the whole audio folder. ``SWHISPER_ONLY_FILES`` narrows
it to the listed basenames, which is what the web UI sets when you launch a run
for a selection rather than for the folder.

The value is a newline separated list, because a filename may contain a comma
but cannot contain a newline. Extensions and directories are ignored, so
``a.wav``, ``/path/to/a.m4a`` and ``a`` all select the same recording.

``SWHISPER_RUN_REPROCESS=true`` makes the run redo files that already have
output, for every stage of the pipeline. Unlike the saved force settings, it
applies to this run only and wins over per-recording sidecar settings.
"""
import os

ENV_VAR = "SWHISPER_ONLY_FILES"
REPROCESS_ENV_VAR = "SWHISPER_RUN_REPROCESS"


def _base(name):
    return os.path.splitext(os.path.basename(name))[0]


def selected_bases():
    """Basenames the run is restricted to, or None to run the whole folder."""
    raw = os.getenv(ENV_VAR, "")
    bases = {_base(line.strip()) for line in raw.splitlines() if line.strip()}
    return bases or None


def keep_selected(names):
    """Filter *names* down to the selection, preserving the caller's order."""
    bases = selected_bases()
    if bases is None:
        return list(names)
    return [name for name in names if _base(name) in bases]


def reprocess_requested():
    """True when this run must redo files that already have output."""
    return os.getenv(REPROCESS_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def describe():
    """One line for a run log, or an empty string when nothing is restricted."""
    bases = selected_bases()
    if not bases:
        return ""
    listed = ", ".join(sorted(bases)[:5])
    more = f" and {len(bases) - 5} more" if len(bases) > 5 else ""
    return f"Restricted to {len(bases)} selected file(s): {listed}{more}"
