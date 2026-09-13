"""Read-only run planning and progress inspection; never imports model packages."""
import hashlib
import json
import os
import re
from pathlib import Path

from sidecar import (SIDECAR_PARAMS, STAGE_DIARIZE, STAGE_TRANSCRIBE,
                     find_sidecar, parse_sidecar_file, sidecar_requests_reprocess)

FORMATS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}


def preview_run(audio_dir, pipeline, tc, dc):
    """Mirror FileManager and diarize.process_file selection, without conversion.

    Stage two is conditional on stage one's outputs. Actual conversion/model
    failures cannot be predicted. Keep os.listdir order for FILE_LIMIT slicing.
    """
    audio = Path(audio_dir)
    json_dir = audio / tc.json_dir
    diar_json = audio / dc.json_input_dir
    output = audio / dc.output_dir
    entries = os.listdir(audio)
    rows, candidates, evidence = [], [], []

    def stamp(path):
        try:
            stat = path.stat()
            evidence.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            evidence.append((str(path), None))

    def decision(base, artifact, stage, force):
        stamp(artifact)
        sidecar = find_sidecar(str(audio), base)
        if sidecar:
            stamp(Path(sidecar))
        if not artifact.exists():
            return 'process', 'No existing output'
        if force:
            return 'reprocess', 'Force reprocessing enabled'
        redo, reason = sidecar_requests_reprocess(str(audio), base, str(artifact), stage)
        return ('reprocess', reason) if redo else ('skip', 'Existing output; no reprocessing requested')

    if pipeline != 'diarize4':
        for name in entries:
            if Path(name).suffix.lower() not in FORMATS:
                continue
            stamp(audio / name)
            base = Path(name).stem
            action, reason = decision(base, json_dir / (base + '.json'), STAGE_TRANSCRIBE, tc.force_reprocess)
            row = dict(file=name, base=base, stage='Diarization + transcription' if pipeline == 'transcribe5' else 'Transcription', action=action, reason=reason)
            rows.append(row)
            if action != 'skip':
                candidates.append(row)
        selected = candidates if tc.file_limit is None else candidates[:tc.file_limit]
        for row in candidates:
            if row not in selected:
                row.update(action='skip', reason='Outside the configured file limit')
    else:
        selected = []

    if pipeline != 'transcribe5':
        names = [p.stem for p in diar_json.iterdir() if p.name.endswith('.json')] if diar_json.is_dir() else []
        new_bases = {r['base'] for r in selected} if json_dir == diar_json else set()
        for base in sorted(set(names) | new_bases):
            force = dc.force_reprocess
            sidecar = find_sidecar(str(audio), base)
            if sidecar:
                raw = parse_sidecar_file(sidecar).get('SWHISPER_DIARIZE_FORCE_REPROCESS')
                if raw is not None:
                    try:
                        force = SIDECAR_PARAMS['SWHISPER_DIARIZE_FORCE_REPROCESS'].parse(raw)
                    except ValueError:
                        pass
            action, reason = decision(base, output / (base + '.ok'), STAGE_DIARIZE, force)
            stamp(diar_json / (base + '.json'))
            stamp(audio / (base + '.wav'))
            if base not in new_bases and not (audio / (base + '.wav')).exists() and action != 'skip':
                action, reason = 'blocked', 'Required WAV audio is missing'
            elif base in new_bases:
                reason += '; depends on transcription/conversion succeeding'
            rows.append(dict(file=base + '.json', base=base, stage='Diarization', action=action, reason=reason))

    notes = ['This runs the folder. Selecting a file in the editor does not restrict the run.',
             'Preview uses saved settings and current outputs. Conversion, model errors, and files changed during a run can affect the result.']
    if pipeline == 'transcribe5':
        notes.append('Diarization-first selects files using transcription JSON and transcription reprocessing rules; a diarization-only override does not select an already transcribed file.')
    if pipeline == 'transcribe4' and json_dir != diar_json:
        notes.append(f'Transcription writes to {json_dir}; diarization reads {diar_json}. New transcription outputs will not be picked up automatically.')
    if tc.file_limit is not None and pipeline != 'diarize4':
        notes.append(f'Transcription file limit: {tc.file_limit}. Selection follows the pipeline filesystem order.')
    if len({r['base'] for r in rows if r['stage'] != 'Diarization'}) < len([r for r in rows if r['stage'] != 'Diarization']):
        notes.append('Multiple audio files share a basename and therefore share outputs; the pipeline may process that basename more than once.')
    result = dict(audio_dir=str(audio), pipeline=pipeline, files=rows, notes=notes)
    payload = json.dumps([result, evidence], sort_keys=True).encode()
    result['token'] = hashlib.sha256(payload).hexdigest()
    return result


def inspect_progress(lines, pipeline):
    """Only count explicit success records, not inferred success from exit 0."""
    stage, current, index, total = 'Starting', None, None, None
    detail = 'Preparing models and files'
    completed = {'Transcription': set(), 'Diarization': set()}
    errors = []
    for raw in lines:
        line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw).strip()
        match = re.search(r'Audio file (\d+)/(\d+): (.+)', line)
        if match:
            index, total, current = int(match[1]), int(match[2]), match[3]
            stage = 'Diarization + transcription' if pipeline == 'transcribe5' else 'Transcription'
            detail = 'Preparing audio'
        match = re.search(r'Processing \((\d+)/(\d+)\): (.+)', line)
        if match:
            index, total, current = int(match[1]), int(match[2]), match[3]
            stage = 'Diarization + transcription' if pipeline == 'transcribe5' else 'Transcription'
            detail = 'Processing audio'
        if 'Launching diarization pipeline' in line:
            stage, current, index, total = 'Diarization', None, None, None
            detail = 'Preparing diarization'
        if 'DIARIZATION LOG FOR:' in line:
            stage, current = 'Diarization', line.split('DIARIZATION LOG FOR:', 1)[1].strip()
            detail = 'Loading file'
        step = re.search(r'Step \d+: (.+)', line)
        if step:
            detail = step[1]
        elif 'Loading Whisper model' in line:
            detail = 'Loading Whisper model'
        elif '🔄 Transcribing...' in line:
            detail = 'Transcribing audio'
        match = re.search(r'Processing file (\d+) of (\d+)', line)
        if match:
            index, total = int(match[1]), int(match[2])
        match = re.search(r'Skipping file (\d+) of (\d+): (.+)', line)
        if match:
            stage, current, index, total = 'Diarization', match[3], int(match[1]), int(match[2])
            detail = 'Skipping existing output'
        if re.search(r'✅ Saved \d+ segments', line) and current:
            completed['Transcription'].add(current)
            detail = 'Transcription saved'
        if 'Completion marker created:' in line and current:
            completed['Diarization'].add(current)
            detail = 'Diarization saved'
        if (('❌' in line and 'No new audio files' not in line) or
                'ERROR:' in line or re.search(r'^Failed: [1-9]\d*$', line) or
                line.startswith('Traceback (most recent call last)')):
            errors.append(line)
    return dict(stage=stage, detail=detail, current_file=current, file_index=index, total_files=total,
                completed={k: len(v) for k, v in completed.items()},
                error_count=len(errors), last_error=errors[-1] if errors else None)
