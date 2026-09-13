"""Form schema for the settings editors: grouping, labels, help, validation.

Pure data plus two functions over ``SIDECAR_PARAMS``. Nothing here touches the
filesystem or the running pipeline, so it can be imported and tested on its own.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from sidecar import SIDECAR_PARAMS


GLOBAL_EXTRAS = {
    'SWHISPER_MODEL': ('transcription', 'model_str', 'str', 'Whisper model'),
    'SWHISPER_MODEL_REVISION': ('transcription', 'revision', 'str', 'Model revision'),
    'SWHISPER_DEVICE': ('transcription', 'device', 'str', 'Transcription device'),
    'SWHISPER_DIARIZE_DEVICE': ('diarization', 'device', 'str', 'Diarization device'),
    'SWHISPER_FILE_LIMIT': ('transcription', 'file_limit', 'int', 'Files per run'),
    'SWHISPER_JSON_DIR': ('transcription', 'json_dir', 'str', 'Transcription JSON folder'),
    'JSON_INPUT_DIR': ('diarization', 'json_input_dir', 'str', 'Diarization JSON input folder'),
    'SWHISPER_OUTPUT_DIR': ('diarization', 'output_dir', 'str', 'Transcript output folder'),
    'SWHISPER_PRESERVE_CHECKPOINTS': ('transcription', 'preserve_checkpoints', 'bool', 'Preserve checkpoints'),
    'SWHISPER_SPEAKER_VAD': ('transcription', 'use_speaker_segments_as_vad', 'bool', 'Use speaker segments for voice detection'),
    'SWHISPER_SPEAKER_CHUNKING': ('transcription', 'speaker_aware_chunking', 'bool', 'Split audio at speaker boundaries'),
    'SWHISPER_USE_PRECISION': ('diarization', 'use_precision_service', 'bool', 'Use Precision diarization service'),
    'SWHISPER_PRECISION_MODEL': ('diarization', 'precision_pipeline_model', 'str', 'Precision model'),
    'SWHISPER_BACKFILL_MODEL': ('diarization', 'backfill_model', 'str', 'Backfill model'),
    'SWHISPER_BACKFILL_DEVICE': ('diarization', 'backfill_device', 'str', 'Backfill device'),
    'SWHISPER_FP16': ('whisper', 'fp16', 'bool', 'Half precision'),
    'SWHISPER_ALIGNMENT_LAYERS': ('whisper', 'word_alignment_most_top_layers', 'int', 'Alignment layers'),
    'SWHISPER_VERBOSE': ('whisper', 'verbose', 'bool', 'Detailed transcription logging'),
}


def parameter_ui(key):
    labels = {'NUM_SPEAKERS': 'Number of speakers', 'MIN_SPEAKERS': 'Minimum speakers',
              'MAX_SPEAKERS': 'Maximum speakers', 'FORCE_REPROCESS': 'Reprocess transcription',
              'DIARIZE_FORCE_REPROCESS': 'Reprocess diarization', 'VAD': 'Voice activity detection',
              'INITIAL_PROMPT': 'Transcription context', 'CONDITION_ON_PREVIOUS': 'Use previous text as context',
              'OUTPUT_PREAMBLE': 'Transcript introduction'}
    short = key.removeprefix('SWHISPER_')
    result = {'label': labels.get(short, short.replace('_', ' ').capitalize())}
    if short in ('NUM_SPEAKERS', 'MIN_SPEAKERS', 'MAX_SPEAKERS', 'BEAM_SIZE', 'BEST_OF', 'FILE_LIMIT'):
        result['min'] = 1
    if short in ('NO_SPEECH_THRESHOLD',):
        result.update(min=0, max=1)
    if any(part in short for part in ('DURATION', 'OVERLAP_DURATION')):
        result['min'] = 0
    if short == 'TASK':
        result['choices'] = ['transcribe', 'translate']
    return result


def validate_settings(values, specs):
    errors = {}
    for key, raw in values.items():
        spec = specs.get(key)
        if spec is None:
            errors[key] = 'Not an editable setting'
            continue
        try:
            if key in SIDECAR_PARAMS:
                value = SIDECAR_PARAMS[key].parse(raw)
            elif spec['type'] == 'bool':
                value = SIDECAR_PARAMS['SWHISPER_FORCE_REPROCESS'].parse(raw)
            elif spec['type'] == 'int':
                value = int(raw)
            elif spec['type'] == 'float':
                value = float(raw)
            else:
                value = raw
            numeric = value if isinstance(value, list) else [value]
            if any(isinstance(v, float) and not math.isfinite(v) for v in numeric):
                raise ValueError('Must be a finite number')
            if 'min' in spec and value < spec['min']:
                raise ValueError(f'Must be at least {spec["min"]}')
            if 'max' in spec and value > spec['max']:
                raise ValueError(f'Must be at most {spec["max"]}')
            if 'choices' in spec and value not in spec['choices']:
                raise ValueError('Choose ' + ' or '.join(spec['choices']))
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
    low, high = values.get('SWHISPER_MIN_SPEAKERS'), values.get('SWHISPER_MAX_SPEAKERS')
    if low and high and not errors and int(low) > int(high):
        errors['SWHISPER_MAX_SPEAKERS'] = 'Must be at least the minimum speaker count'
    return errors

PARAM_GROUPS: List[Dict[str, Any]] = [
    {"name": "Speakers", "keys": [
        "SWHISPER_NUM_SPEAKERS", "SWHISPER_MIN_SPEAKERS", "SWHISPER_MAX_SPEAKERS"]},
    {"name": "Reprocess", "keys": [
        "SWHISPER_FORCE_REPROCESS", "SWHISPER_DIARIZE_FORCE_REPROCESS"]},
    {"name": "Whisper", "keys": [
        "SWHISPER_LANGUAGE", "SWHISPER_TASK", "SWHISPER_INITIAL_PROMPT", "SWHISPER_VAD",
        "SWHISPER_BEAM_SIZE", "SWHISPER_BEST_OF", "SWHISPER_TEMPERATURE",
        "SWHISPER_COMPRESSION_THRESHOLD", "SWHISPER_LOGPROB_THRESHOLD",
        "SWHISPER_NO_SPEECH_THRESHOLD", "SWHISPER_CONDITION_ON_PREVIOUS",
        "SWHISPER_DETECT_DISFLUENCIES", "SWHISPER_SEED", "SWHISPER_REFINE_PRECISION",
        "SWHISPER_MIN_WORD_DURATION", "SWHISPER_TRUST_TIMESTAMPS",
        "SWHISPER_SUPPRESS_TOKENS", "SWHISPER_COMPUTE_CONFIDENCE"]},
    {"name": "Chunking", "keys": [
        "SWHISPER_CHUNK_DURATION", "SWHISPER_MIN_SILENCE", "SWHISPER_OVERLAP_DURATION",
        "SWHISPER_MAX_CHUNK_DURATION", "SWHISPER_MIN_CHUNK_DURATION",
        "SWHISPER_VAD_GAP_MERGE_THRESHOLD"]},
    {"name": "Diarization tuning", "keys": [
        "SWHISPER_VBX_THRESHOLD", "SWHISPER_VBX_FA", "SWHISPER_VBX_FB",
        "SWHISPER_COMMUNITY_MIN_DURATION_OFF", "SWHISPER_SEGMENTATION_THRESHOLD",
        "SWHISPER_MIN_DURATION_ON", "SWHISPER_MIN_DURATION_OFF",
        "SWHISPER_CLUSTERING_METHOD", "SWHISPER_CLUSTERING_MIN_SIZE",
        "SWHISPER_CLUSTERING_THRESHOLD", "SWHISPER_EMBEDDING_THRESHOLD",
        "SWHISPER_SPEAKER_CHANGE_PENALTY", "SWHISPER_TEMPORAL_CONSTRAINTS",
        "SWHISPER_USE_EXCLUSIVE_DIARIZATION"]},
    {"name": "Alignment & silence", "keys": [
        "SWHISPER_WHISPER_TIMESTAMP_OFFSET", "SWHISPER_SMOOTHING_ENABLED",
        "SWHISPER_MIN_SPEAKER_WORDS", "SWHISPER_PRESERVE_MARKERS",
        "SWHISPER_MIN_SILENCE_DURATION", "SWHISPER_INCLUDE_SILENCE",
        "SWHISPER_LOG_SILENCE", "SWHISPER_SILENCE_LINEBREAK"]},
    {"name": "Backfill", "keys": [
        "SWHISPER_BACKFILL_ENABLED", "SWHISPER_BACKFILL_OVERLAP",
        "SWHISPER_BACKFILL_MIN_DURATION", "SWHISPER_BACKFILL_IGNORE_WORDS",
        "SWHISPER_BACKFILL_ENABLE_SENSEVOICE", "SWHISPER_BACKFILL_SENSEVOICE_LANGUAGE"]},
    {"name": "Output", "keys": [
        "SWHISPER_TSV_WORD_PER_LINE", "SWHISPER_OUTPUT_PREAMBLE"]},
]

# Mouseover help texts for the sidecar form
PARAM_HELP: Dict[str, str] = {
    "SWHISPER_NUM_SPEAKERS": "Exact number of speakers for this file — sets both min and max. Tip from the README: specifying 1–2 more speakers than actually present often works better.",
    "SWHISPER_MIN_SPEAKERS": "Lower bound for the number of speakers pyannote may detect (default 1).",
    "SWHISPER_MAX_SPEAKERS": "Upper bound for the number of speakers. The README recommends 1–2 more than the real number; post-processing often removes the excess.",
    "SWHISPER_FORCE_REPROCESS": "Re-run transcription for this file even though whisper JSON output already exists.",
    "SWHISPER_DIARIZE_FORCE_REPROCESS": "Re-run diarization for this file even though its .ok completion marker exists.",
    "SWHISPER_LANGUAGE": "Language code passed to Whisper (global default: sv). Use e.g. 'en' for an English recording.",
    "SWHISPER_TASK": "'transcribe' keeps the source language; 'translate' translates to English.",
    "SWHISPER_INITIAL_PROMPT": "Text used to prime Whisper — e.g. names, places, or terminology that occur in this recording. Improves spelling of rare words.",
    "SWHISPER_VAD": "Voice activity detection applied before transcription: silero:v3.1, silero:v4.0, auditok, or None (no VAD).",
    "SWHISPER_BEAM_SIZE": "Beam search width (default 5). Higher = potentially better output, slower.",
    "SWHISPER_BEST_OF": "Number of candidates when sampling at non-zero temperature (default 5).",
    "SWHISPER_TEMPERATURE": "Comma-separated fallback temperatures, e.g. 0.0,0.2,0.4. Whisper retries with the next value when decoding fails quality checks.",
    "SWHISPER_COMPRESSION_THRESHOLD": "Decoding counts as failed if the text's gzip compression ratio exceeds this (default 2.4) — a guard against repetitive hallucination.",
    "SWHISPER_LOGPROB_THRESHOLD": "Segments with average log-probability below this count as failed. -1 effectively disables the check.",
    "SWHISPER_NO_SPEECH_THRESHOLD": "Probability above which a segment is dismissed as silence (default 0.6).",
    "SWHISPER_CONDITION_ON_PREVIOUS": "Feed the previous transcript window as context. More consistent style, but errors and hallucinations can propagate.",
    "SWHISPER_DETECT_DISFLUENCIES": "Mark sounds that could not be transcribed as [*] disfluency markers.",
    "SWHISPER_SEED": "Random seed for reproducibility (default 1234).",
    "SWHISPER_REFINE_PRECISION": "Precision (in seconds) for refining word timestamps (default 0.5).",
    "SWHISPER_MIN_WORD_DURATION": "Minimum duration a word may get in the timestamps (default 0.02 s).",
    "SWHISPER_TRUST_TIMESTAMPS": "Trust Whisper's own segment timestamps instead of recomputing them (default false).",
    "SWHISPER_SUPPRESS_TOKENS": "Comma-separated token ids Whisper must not produce (default 50364).",
    "SWHISPER_COMPUTE_CONFIDENCE": "Compute word-level confidence scores. Note: may not work with the KBLab model.",
    "SWHISPER_CHUNK_DURATION": "Target chunk length in seconds for long files (default 180). Boundaries are placed at silences.",
    "SWHISPER_MIN_SILENCE": "Minimum silence duration (s) where a chunk boundary may be placed (default 0.5).",
    "SWHISPER_OVERLAP_DURATION": "Overlap between consecutive chunks in seconds, used to stitch transcripts together (default 1.0).",
    "SWHISPER_MAX_CHUNK_DURATION": "Diarization-first mode: maximum chunk duration before a split is forced (default 200 s).",
    "SWHISPER_MIN_CHUNK_DURATION": "Diarization-first mode: minimum chunk duration, to avoid tiny chunks (default 30 s).",
    "SWHISPER_VAD_GAP_MERGE_THRESHOLD": "Diarization-first mode: merge speaker segments separated by gaps shorter than this (default 0.5 s).",
    "SWHISPER_VBX_THRESHOLD": "VBx clustering threshold for the community-1 model (default 0.6). Lower values create more speakers.",
    "SWHISPER_VBX_FA": "False-alarm probability for speaker activity (default 0.07). Lower = more conservative speech detection.",
    "SWHISPER_VBX_FB": "Probability of not detecting a speaker (default 0.8). Lower = more aggressive speaker separation.",
    "SWHISPER_COMMUNITY_MIN_DURATION_OFF": "Minimum duration of non-speech segments for the community-1 model (default 0.0).",
    "SWHISPER_SEGMENTATION_THRESHOLD": "Legacy 3.1 pipeline: sensitivity to speaker changes in segmentation (default 0.01, lower = more sensitive).",
    "SWHISPER_MIN_DURATION_ON": "Legacy 3.1 pipeline: minimum duration of speech segments (s).",
    "SWHISPER_MIN_DURATION_OFF": "Legacy 3.1 pipeline: minimum duration of non-speech segments (s).",
    "SWHISPER_CLUSTERING_METHOD": "Legacy 3.1 pipeline: clustering linkage method (default centroid).",
    "SWHISPER_CLUSTERING_MIN_SIZE": "Legacy 3.1 pipeline: minimum number of frames to form a speaker cluster (default 15).",
    "SWHISPER_CLUSTERING_THRESHOLD": "Legacy 3.1 pipeline: clustering distance threshold (default 0.15). Lower = stricter clustering.",
    "SWHISPER_EMBEDDING_THRESHOLD": "Speaker embedding distance threshold (default 0.6). Lower = voices must be more similar to be merged into one speaker.",
    "SWHISPER_SPEAKER_CHANGE_PENALTY": "Penalty for frequent speaker changes (default 0.1). Ignored by newer pyannote models.",
    "SWHISPER_TEMPORAL_CONSTRAINTS": "Use temporal constraints during diarization. Ignored by newer pyannote models.",
    "SWHISPER_USE_EXCLUSIVE_DIARIZATION": "Use pyannote 4's exclusive (non-overlapping) diarization stream when available.",
    "SWHISPER_WHISPER_TIMESTAMP_OFFSET": "Shift all Whisper timestamps by this many seconds before alignment (default 0.1 — the README notes a systematic ~0.1 s shift).",
    "SWHISPER_SMOOTHING_ENABLED": "Smooth word-level speaker assignment to reduce spurious single-word speaker flips.",
    "SWHISPER_MIN_SPEAKER_WORDS": "Minimum consecutive words required to accept a speaker change during smoothing (default 3).",
    "SWHISPER_PRESERVE_MARKERS": "Keep [*] disfluency markers in the diarized transcript.",
    "SWHISPER_MIN_SILENCE_DURATION": "Ignore silences shorter than this when writing silence markers (default 0.2 s).",
    "SWHISPER_INCLUDE_SILENCE": "Write silence durations into the transcript in CA notation, e.g. (.3).",
    "SWHISPER_LOG_SILENCE": "Write a separate log file with between-word gap durations.",
    "SWHISPER_SILENCE_LINEBREAK": "Surround silences of at least this many seconds with blank lines (default 1.0; 0 disables). Affects rtf/txt output only.",
    "SWHISPER_BACKFILL_ENABLED": "Re-transcribe speaker turns that got no words. Recovers missed speech but can introduce hallucinations.",
    "SWHISPER_BACKFILL_OVERLAP": "Extra seconds of audio included around each backfilled segment (default 0.5).",
    "SWHISPER_BACKFILL_MIN_DURATION": "Skip backfilling segments shorter than this (default 0.1 s).",
    "SWHISPER_BACKFILL_IGNORE_WORDS": "Comma-separated words to treat as backfill hallucinations (replaced by markers), e.g. Balans.",
    "SWHISPER_BACKFILL_ENABLE_SENSEVOICE": "Run SenseVoice emotion and audio-event detection (laughter, crying, …) on backfilled segments.",
    "SWHISPER_BACKFILL_SENSEVOICE_LANGUAGE": "SenseVoice language: auto, zh, en, yue, ja, or ko.",
    "SWHISPER_TSV_WORD_PER_LINE": "TSV output: one word per line (true) or one segment per line (false).",
    "SWHISPER_OUTPUT_PREAMBLE": "Custom preamble text at the top of transcript files. Use \\n for line breaks.",
}

_TYPE_BY_PARSER = {
    "_parse_bool": "bool",
    "_parse_int": "int",
    "_parse_float": "float",
    "_parse_str": "str",
    "_parse_float_list": "float_list",
    "_parse_word_list": "word_list",
}

