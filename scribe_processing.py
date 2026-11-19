"""Scribe local-processing policy and UI capabilities, without model imports."""
import os
from pathlib import Path

from diarization_status import model_run_reason
from sidecar import (SIDECAR_PARAMS, STAGE_DIARIZE, apply_sidecar_to_diarization_config,
                     find_sidecar, parse_sidecar_file, sidecar_requests_reprocess)

# Languages the Scribe worker has a model for (scribe-worker utils/settings.py),
# by the ISO 639-1 code that SWHISPER_LANGUAGE uses for local Whisper.
SCRIBE_LANGUAGES = {
    'sv': 'Swedish', 'en': 'English', 'no': 'Norwegian', 'fi': 'Finnish', 'da': 'Danish',
    'is': 'Icelandic', 'se': 'Northern Sámi (Experimental)', 'fr': 'French', 'de': 'German',
    'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian', 'uk': 'Ukrainian', 'pt': 'Portuguese',
    'nl': 'Dutch',
}
# Bokmål and Nynorsk use the Norwegian model.
LANGUAGE_ALIASES = {'nb': 'no', 'nn': 'no'}
# Languages with a separate "<name> (verbatim)" model in the worker.
VERBATIM_LANGUAGES = ('Swedish', 'English', 'Norwegian')
VERBATIM = ' (verbatim)'
SCRIBE_MODES = ('verbatim', 'standard')


def recording_language(source):
    """SWHISPER_LANGUAGE for one recording, as local Whisper resolves it:
    the recording's .swhisper sidecar, else the global setting, else sv."""
    source = Path(source)
    sidecar = find_sidecar(str(source.parent), source.stem)
    if sidecar:
        value = parse_sidecar_file(sidecar).get('SWHISPER_LANGUAGE', '').strip()
        if value:
            return value
    # An explicitly empty value asks local Whisper to detect the language;
    # Scribe cannot, so scribe_language() refuses it instead of guessing.
    value = os.getenv('SWHISPER_LANGUAGE')
    return 'sv' if value is None else value.strip()


def scribe_language(language, mode='verbatim'):
    """The Scribe language for a SWHISPER_LANGUAGE code or a Scribe language name.

    *mode* "verbatim" selects the verbatim model where Scribe has one. A name
    that already ends in " (verbatim)" is kept. Raises ValueError for a
    language Scribe cannot transcribe, because the worker would leave such a
    job in_progress forever.
    """
    if mode not in SCRIBE_MODES:
        raise ValueError(f'Unknown Scribe mode {mode!r}; use verbatim or standard.')
    key = language.strip()
    verbatim = mode == 'verbatim'
    if key.lower().endswith(VERBATIM):
        key, verbatim = key[:-len(VERBATIM)].strip(), True
    code = LANGUAGE_ALIASES.get(key.lower(), key.lower())
    name = SCRIBE_LANGUAGES.get(code) or next(
        (n for n in SCRIBE_LANGUAGES.values() if n.lower() == key.lower()), None)
    if name is None:
        raise ValueError(f'Scribe cannot transcribe SWHISPER_LANGUAGE={language!r}. '
                         f'Use one of: {", ".join(sorted(SCRIBE_LANGUAGES))}.')
    return name + VERBATIM if verbatim and name in VERBATIM_LANGUAGES else name


def language_code(scribe_name):
    """The ISO 639-1 code for a Scribe language name, for local Whisper and exports."""
    base = scribe_name.removesuffix(VERBATIM)
    code = next(c for c, n in SCRIBE_LANGUAGES.items() if n == base)
    # Whisper has no Northern Sámi; the worker also uses its Norwegian model.
    return 'no' if code == 'se' else code


UNSUPPORTED = {
    'SWHISPER_USE_PRECISION': 'Scribe uses local community-1 or Nemotron; Precision is not integrated.',
    'SWHISPER_PRECISION_MODEL': 'Scribe uses local community-1 or Nemotron; Precision is not integrated.',
    'SWHISPER_WHISPER_TIMESTAMP_OFFSET': 'Use the separate Scribe timestamp offset, which defaults to zero.',
    'SWHISPER_OUTPUT_DIR': 'Scribe exports stay inside its isolated work directory.',
    'JSON_INPUT_DIR': 'Scribe reads its saved normalized remote result from its isolated work directory.',
    'SWHISPER_VERBOSE': 'Local backfill sets Whisper verbose output off.',
}
LEGACY_TUNING = {
    'SWHISPER_SEGMENTATION_THRESHOLD', 'SWHISPER_MIN_DURATION_ON', 'SWHISPER_MIN_DURATION_OFF',
    'SWHISPER_CLUSTERING_METHOD', 'SWHISPER_CLUSTERING_MIN_SIZE', 'SWHISPER_CLUSTERING_THRESHOLD',
    'SWHISPER_EMBEDDING_THRESHOLD', 'SWHISPER_SPEAKER_CHANGE_PENALTY', 'SWHISPER_TEMPORAL_CONSTRAINTS',
}


def capability(key):
    if key == 'SWHISPER_SCRIBE_TIMESTAMP_OFFSET':
        return dict(supported=True, scope='Scribe timestamps', description='Scribe only: shift returned timestamps locally; default 0 seconds. No model re-alignment.')
    if key in UNSUPPORTED:
        return dict(supported=False, scope='Unavailable for Scribe', description=UNSUPPORTED[key])
    if key in LEGACY_TUNING:
        return dict(supported=False, scope='Unavailable for Scribe', description='Not applied by the community-1 path. Use VBx and community segmentation controls.')
    if key == 'SWHISPER_LANGUAGE':
        return dict(supported=True, scope='Scribe and local backfill',
                    description='Selects the Scribe language and the language of local backfill, as for local Whisper.')
    param = SIDECAR_PARAMS.get(key)
    if (param and param.target == 'whisper' and key not in {'SWHISPER_LANGUAGE', 'SWHISPER_TASK', 'SWHISPER_VAD'}) or key in {'SWHISPER_ALIGNMENT_LAYERS', 'SWHISPER_FP16'}:
        return dict(supported=True, scope='Local backfill only', description='Local Whisper backfill only; does not change the Scribe transcription. Recovery uses the language of the recording with VAD off.')
    if key.startswith('SWHISPER_BACKFILL_'):
        return dict(supported=True, scope='Local backfill only', description='Local Whisper recovery of missing turns only. SenseVoice analyzes recovery snippets only.')
    if (param and param.target == 'diarization') or key in {
        'SWHISPER_DIARIZE_DEVICE', 'RECORDING_TYPE', 'INTERVIEWER_NAME', 'INTERVIEWEE_NAME'
    }:
        return dict(supported=True, scope='Local processing after Scribe', description='Applies to local processing after Scribe transcription.')
    return dict(supported=False, scope='Unavailable for Scribe', description='Does not control SUNET transcription or local recovery. Scribe uses SWHISPER_LANGUAGE and the verbatim or standard choice; recovery uses its own Whisper defaults. Remote prompting, disfluency and alignment controls are unavailable.')


def configure(config, source, work_dir):
    """Apply the source sidecar once, then enforce isolated local processing."""
    source, work_dir = Path(source), Path(work_dir)
    config, _ = apply_sidecar_to_diarization_config(config, str(source.parent), source.stem)
    if not config.uses_nemotron:
        config.pipeline_model = 'pyannote/speaker-diarization-community-1'
    config.use_precision_service = False
    config.whisper_timestamp_offset = config.scribe_timestamp_offset
    config.output_dir = str(work_dir / 'transcripts')
    config.log_dir = str(work_dir / 'transcripts' / 'logs')
    config.diarization_cache_dir = str(work_dir / '.diarization_cache')
    config.backfill_cache_dir = str(work_dir / '.backfill_cache')
    config.backfill_snippet_dir = str(work_dir / 'backfill_snippets')
    config.enable_parameter_testing = False
    config.backfill_enable_sensevoice = bool(config.backfill_missing_turns and config.backfill_enable_sensevoice)
    return config


def completion_marker(work_dir):
    """The local diarization marker for the saved Scribe transcript audio.json."""
    return Path(work_dir) / 'transcripts' / 'audio.ok'


def diarization_reason(config, source, work_dir, force=False):
    """Why local diarization must run after Scribe, or None to reuse the last result.

    *config* is the configured result of configure(). The recording's sidecar
    sits next to the source, not in the isolated work directory.
    """
    source = Path(source)
    marker = completion_marker(work_dir)
    if not marker.is_file():
        return 'No local diarization yet'
    if force:
        return 'Force reprocessing requested for this run'
    if config.force_reprocess:
        return 'Force reprocessing enabled'
    reason = model_run_reason(marker, config.pipeline_model)
    if reason:
        return reason
    if any(f'# Backfill status: {status}' in marker.read_text(encoding='utf-8')
           for status in ('failed', 'partial')):
        return 'Previous backfill was incomplete; resume from saved gaps'
    redo, reason = sidecar_requests_reprocess(str(source.parent), source.stem, str(marker), STAGE_DIARIZE)
    return reason if redo else None


def effective_settings(config):
    """Allowlisted settings only; never serialize credentials from the config."""
    names = ('pipeline_model', 'device', 'min_speakers', 'max_speakers',
             'recording_type', 'interviewer_name', 'interviewee_name',
             'whisper_timestamp_offset', 'smoothing_enabled', 'min_speaker_words',
             'preserve_markers', 'min_silence_duration', 'include_silence_markers',
             'backfill_missing_turns', 'backfill_try_harder', 'backfill_model', 'backfill_device',
             'backfill_overlap', 'backfill_min_duration', 'backfill_ignore_words',
             'backfill_enable_sensevoice', 'backfill_sensevoice_model',
             'backfill_sensevoice_device', 'backfill_sensevoice_language')
    result = {name: getattr(config, name) for name in names}
    result['backfill_device'] = config.backfill_device or config.device
    result['backfill_sensevoice_device'] = config.backfill_sensevoice_device or result['backfill_device']
    result['pipeline_parameters_requested'] = config.get_pipeline_config()
    return result


def enforce_scribe_nemotron(config, work_dir):
    """Pin the worker's promised engines after applying global/sidecar settings.

    Return whether a previous result must be replaced because it used local
    backfill or speaker truncation. The remote Scribe result remains reusable.
    """
    import json
    import re
    config.pipeline_model = 'nvidia/Nemotron-3-Diarization'
    config.backfill_missing_turns = False
    config.backfill_enable_sensevoice = False
    config.max_speakers = None
    marker = completion_marker(work_dir)
    if not marker.exists():
        return False
    try:
        stamp = re.search(r'^# Start: (\d{8}-\d{6})$', marker.read_text(), re.M)[1]
        metadata = json.loads((marker.parent / 'json' / f'audio_{stamp}_diarized.json').read_text())['metadata']
        settings = metadata['processing']['settings']
        return settings.get('backfill_missing_turns') is not False or settings.get('max_speakers') is not None
    except (OSError, ValueError, KeyError, TypeError):
        return True
