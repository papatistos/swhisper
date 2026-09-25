"""Restrict a pipeline run to named files.

A run normally processes the whole audio folder. ``SWHISPER_ONLY_FILES`` narrows
it to the listed basenames, which is what the web UI sets when you launch a run
for a selection rather than for the folder.

The value is a newline separated list, because a filename may contain a comma
but cannot contain a newline. Extensions and directories are ignored, so
``a.wav``, ``/path/to/a.m4a`` and ``a`` all select the same recording.

``SWHISPER_RUN_REPROCESS=true`` makes the run redo files that already have
output, for every stage of the pipeline. ``SWHISPER_RUN_REPROCESS=diarization``
redoes diarization only and reuses existing transcriptions. Unlike the saved
force settings, it applies to this run only and wins over per-recording sidecar
settings. There is no transcription-only value, because diarization is aligned
to the transcript and must follow a new one.

``SWHISPER_RUN_RETRANSCRIBE`` and ``SWHISPER_RUN_REDIARIZE`` name recordings
to redo in this run only, in the same newline separated form as
``SWHISPER_ONLY_FILES``. A recording that is transcribed again is diarized
again too.
"""
import os

ENV_VAR = "SWHISPER_ONLY_FILES"
REPROCESS_ENV_VAR = "SWHISPER_RUN_REPROCESS"
RETRANSCRIBE_ENV_VAR = "SWHISPER_RUN_RETRANSCRIBE"
REDIARIZE_ENV_VAR = "SWHISPER_RUN_REDIARIZE"
STAGE_TRANSCRIPTION = "transcription"
STAGE_DIARIZATION = "diarization"

# Diarization models a single run can choose, by short name.
DIARIZATION_MODELS = {
    "community-1": "pyannote/speaker-diarization-community-1",
    "nemotron": "nvidia/Nemotron-3-Diarization",
}


def _base(name):
    return os.path.splitext(os.path.basename(name))[0]


def _listed(var):
    raw = os.getenv(var, "")
    return {_base(line.strip()) for line in raw.splitlines() if line.strip()}


def selected_bases():
    """Basenames the run is restricted to, or None to run the whole folder."""
    return _listed(ENV_VAR) or None


def keep_selected(names):
    """Filter *names* down to the selection, preserving the caller's order."""
    bases = selected_bases()
    if bases is None:
        return list(names)
    return [name for name in names if _base(name) in bases]


def reprocess_requested(stage=None, base=None):
    """True when this run must redo existing output in *stage*.

    Without *base* only the whole-run setting counts; with it, the recordings
    named for this run count too.
    """
    value = os.getenv(REPROCESS_ENV_VAR, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value == STAGE_DIARIZATION and stage == STAGE_DIARIZATION:
        return True
    if base is None:
        return False
    base = _base(base)
    if base in _listed(RETRANSCRIBE_ENV_VAR):
        return True
    return stage == STAGE_DIARIZATION and base in _listed(REDIARIZE_ENV_VAR)


def describe():
    """One line for a run log, or an empty string when nothing is restricted."""
    bases = selected_bases()
    if not bases:
        return ""
    listed = ", ".join(sorted(bases)[:5])
    more = f" and {len(bases) - 5} more" if len(bases) > 5 else ""
    return f"Restricted to {len(bases)} selected file(s): {listed}{more}"
