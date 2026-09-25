"""Read model provenance without importing or running diarization models."""
import json
import re
from pathlib import Path

COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_MODEL = "pyannote/speaker-diarization-3.1"
PRECISION_MODEL = "pyannote/speaker-diarization-precision-2"


def diarization_model_label(config):
    """Describe the active configuration without guessing a backend."""
    model = getattr(config, 'pipeline_model', None)
    return model.strip() if isinstance(model, str) and model.strip() else 'unspecified model'


def effective_v4_model(config):
    """The model diarize4 actually runs, including its legacy-default upgrade."""
    if getattr(config, 'use_precision_service', False):
        return getattr(config, 'precision_pipeline_model', PRECISION_MODEL)
    model = (getattr(config, 'pipeline_model', '') or '').strip()
    return COMMUNITY_MODEL if model in ('', LEGACY_MODEL) else model


def backfill_incomplete(marker):
    """True when the run behind *marker* could not recover every missing turn.

    The transcript is complete; the unrecovered gaps are marked [-] and the
    next run retries them.
    """
    try:
        text = Path(marker).read_text(encoding='utf-8')
    except OSError:
        return False
    return any(f'# Backfill status: {status}' in text for status in ('failed', 'partial'))


def model_run_reason(marker, requested_model, stamp=None):
    """Why existing output cannot satisfy this model, or None when it can.

    Older completion markers lack the model. Their exact run timestamp links
    to the enriched JSON, whose metadata already records it. The optional
    callback includes that file in the WebUI preview's stale-state detection.
    """
    if not requested_model:
        return None
    marker = Path(marker)
    text = marker.read_text(encoding='utf-8')
    recorded = re.search(r'^# Diarization model: (.+)$', text, re.MULTILINE)
    model = recorded[1].strip() if recorded else None
    if not model:
        started = re.search(r'^# Start: (\d{8}-\d{6})$', text, re.MULTILINE)
        if started:
            artifact = marker.parent / 'json' / f'{marker.stem}_{started[1]}_diarized.json'
            if stamp:
                stamp(artifact)
            try:
                saved = json.loads(artifact.read_text(encoding='utf-8'))
                model = saved['metadata']['diarization_model']
            except (OSError, ValueError, KeyError, TypeError):
                model = None
    if not isinstance(model, str) or not model.strip():
        return f'Existing output has no recorded diarization model; run {requested_model}'
    if model.strip().casefold() != requested_model.strip().casefold():
        return f'Run {requested_model}; existing output was produced by {model}'
    return None
