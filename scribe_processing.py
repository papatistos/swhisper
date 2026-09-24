"""Scribe local-processing policy and UI capabilities, without model imports."""
from pathlib import Path

from sidecar import SIDECAR_PARAMS, apply_sidecar_to_diarization_config


UNSUPPORTED = {
    'SWHISPER_USE_PRECISION': 'Scribe uses local community-1 or Nemotron; Precision is not integrated.',
    'SWHISPER_PRECISION_MODEL': 'Scribe uses local community-1 or Nemotron; Precision is not integrated.',
    'SWHISPER_WHISPER_TIMESTAMP_OFFSET': 'Use the separate Scribe timestamp offset, which defaults to zero.',
    'SWHISPER_DIARIZE_FORCE_REPROCESS': 'Scribe always repeats local processing and exports; inference may reuse its cache.',
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
    param = SIDECAR_PARAMS.get(key)
    if (param and param.target == 'whisper' and key not in {'SWHISPER_LANGUAGE', 'SWHISPER_TASK', 'SWHISPER_VAD'}) or key in {'SWHISPER_ALIGNMENT_LAYERS', 'SWHISPER_FP16'}:
        return dict(supported=True, scope='Local backfill only', description='Local Whisper backfill only; does not change the Scribe transcription. Recovery always uses Swedish transcription with VAD off.')
    if key.startswith('SWHISPER_BACKFILL_'):
        return dict(supported=True, scope='Local backfill only', description='Local Whisper recovery of missing turns only. SenseVoice analyzes recovery snippets only.')
    if (param and param.target == 'diarization') or key == 'SWHISPER_DIARIZE_DEVICE':
        return dict(supported=True, scope='Local processing after Scribe', description='Applies to local processing after Scribe transcription.')
    return dict(supported=False, scope='Unavailable for Scribe', description='Does not control SUNET transcription or local recovery. Scribe uses its language selector; recovery uses its own Whisper defaults. Remote prompting, disfluency and alignment controls are unavailable.')


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
    config.force_reprocess = True
    config.backfill_enable_sensevoice = bool(config.backfill_missing_turns and config.backfill_enable_sensevoice)
    return config


def effective_settings(config):
    """Allowlisted settings only; never serialize credentials from the config."""
    names = ('pipeline_model', 'device', 'min_speakers', 'max_speakers',
             'whisper_timestamp_offset', 'smoothing_enabled', 'min_speaker_words',
             'preserve_markers', 'min_silence_duration', 'include_silence_markers',
             'backfill_missing_turns', 'backfill_model', 'backfill_device',
             'backfill_overlap', 'backfill_min_duration', 'backfill_ignore_words',
             'backfill_enable_sensevoice', 'backfill_sensevoice_model',
             'backfill_sensevoice_device', 'backfill_sensevoice_language')
    result = {name: getattr(config, name) for name in names}
    result['backfill_device'] = config.backfill_device or config.device
    result['backfill_sensevoice_device'] = config.backfill_sensevoice_device or result['backfill_device']
    result['pipeline_parameters_requested'] = config.get_pipeline_config()
    return result
