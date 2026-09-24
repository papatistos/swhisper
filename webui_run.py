"""Read-only run planning and progress inspection; never imports model packages."""
import hashlib
import json
import os
import re
from pathlib import Path

from sidecar import (SIDECAR_PARAMS, STAGE_DIARIZE, STAGE_TRANSCRIBE,
                     find_sidecar, parse_sidecar_file, sidecar_requests_reprocess)

FORMATS = {'.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg'}


def preview_run(audio_dir, pipeline, tc, dc, only=None, reprocess=False):
    """Mirror FileManager and diarize.process_file selection, without conversion.

    Stage two is conditional on stage one's outputs. Actual conversion/model
    failures cannot be predicted. Keep os.listdir order for FILE_LIMIT slicing.

    *only* is a set of basenames the run is restricted to, or None for the
    whole folder; it mirrors what SWHISPER_ONLY_FILES does inside the pipeline.
    *reprocess* mirrors SWHISPER_RUN_REPROCESS: redo existing outputs in this run.
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
        if reprocess:
            return 'reprocess', 'Reprocessing requested for this run'
        if force:
            return 'reprocess', 'Force reprocessing enabled'
        if stage == STAGE_DIARIZE and any(
                f'# Backfill status: {status}' in artifact.read_text(encoding='utf-8')
                for status in ('failed', 'partial')):
            return 'reprocess', 'Previous backfill was incomplete; resume from saved gaps'
        redo, reason = sidecar_requests_reprocess(str(audio), base, str(artifact), stage)
        return ('reprocess', reason) if redo else ('skip', 'Existing output; no reprocessing requested')

    if pipeline != 'diarize4':
        for name in entries:
            if Path(name).suffix.lower() not in FORMATS:
                continue
            if only is not None and Path(name).stem not in only:
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
            if only is not None and base not in only:
                continue
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

    scope = (f'This run is restricted to {len(only)} selected file(s).' if only is not None else
             'This runs the folder. Selecting a file in the editor does not restrict the run.')
    notes = [scope,
             'Preview uses saved settings and current outputs. Conversion, model errors, and files changed during a run can affect the result.']
    if pipeline == 'transcribe5':
        notes.append('Diarization-first selects files using transcription JSON and transcription reprocessing rules; a diarization-only override does not select an already transcribed file.')
    if pipeline == 'transcribe4' and json_dir != diar_json:
        notes.append(f'Transcription writes to {json_dir}; diarization reads {diar_json}. New transcription outputs will not be picked up automatically.')
    if tc.file_limit is not None and pipeline != 'diarize4':
        notes.append(f'Transcription file limit: {tc.file_limit}. Selection follows the pipeline filesystem order.')
    if len({r['base'] for r in rows if r['stage'] != 'Diarization'}) < len([r for r in rows if r['stage'] != 'Diarization']):
        notes.append('Multiple audio files share a basename and therefore share outputs; the pipeline may process that basename more than once.')
    result = dict(audio_dir=str(audio), pipeline=pipeline, files=rows, notes=notes,
                  only=sorted(only) if only is not None else None, reprocess=reprocess)
    payload = json.dumps([result, evidence], sort_keys=True).encode()
    result['token'] = hashlib.sha256(payload).hexdigest()
    return result


class ProgressTracker:
    """Incremental reader of a run's log.

    A run's log only grows, so re-reading it from the start on every poll costs
    more with every minute the run lasts. The tracker keeps the state the scan
    produced and consumes each line exactly once.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.stage = 'Starting'
        self.detail = 'Preparing models and files'
        self.current = None
        self.index = None
        self.total = None
        self.completed = {'Transcription': set(), 'Diarization': set()}
        self.errors = []
        self.backfill_incomplete = set()
        self.line_count = 0

    def feed(self, lines):
        """Only count explicit success records, not inferred success from exit 0."""
        for raw in lines:
            self.line_count += 1
            line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw).strip()
            match = re.search(r'Audio file (\d+)/(\d+): (.+)', line)
            if match:
                self.index, self.total, self.current = int(match[1]), int(match[2]), match[3]
                self.stage = 'Diarization + transcription' if self.pipeline == 'transcribe5' else 'Transcription'
                self.detail = 'Preparing audio'
            match = re.search(r'Processing \((\d+)/(\d+)\): (.+)', line)
            if match:
                self.index, self.total, self.current = int(match[1]), int(match[2]), match[3]
                self.stage = 'Diarization + transcription' if self.pipeline == 'transcribe5' else 'Transcription'
                self.detail = 'Processing audio'
            if 'Launching diarization pipeline' in line:
                self.stage, self.current, self.index, self.total = 'Diarization', None, None, None
                self.detail = 'Preparing diarization'
            if 'DIARIZATION LOG FOR:' in line:
                self.stage = 'Diarization'
                self.current = line.split('DIARIZATION LOG FOR:', 1)[1].strip()
                self.detail = 'Loading file'
            step = re.search(r'Step \d+: (.+)', line)
            if step:
                self.detail = step[1]
            elif 'Loading Whisper model' in line:
                self.detail = 'Loading Whisper model'
            elif '🔄 Transcribing...' in line:
                self.detail = 'Transcribing audio'
            if 'Transcript exported for ' in line and '; backfill incomplete' in line:
                self.backfill_incomplete.add(self.current or line)
                self.completed['Diarization'].add(self.current or line)
                self.detail = 'Transcript saved; backfill incomplete'
            match = re.search(r'Processing file (\d+) of (\d+)', line)
            if match:
                self.index, self.total = int(match[1]), int(match[2])
            match = re.search(r'Skipping file (\d+) of (\d+): (.+)', line)
            if match:
                self.stage, self.current = 'Diarization', match[3]
                self.index, self.total = int(match[1]), int(match[2])
                self.detail = 'Skipping existing output'
            if re.search(r'✅ Saved \d+ segments', line) and self.current:
                self.completed['Transcription'].add(self.current)
                self.detail = 'Transcription saved'
            if 'Completion marker created:' in line and self.current:
                self.completed['Diarization'].add(self.current)
                self.detail = ('Transcript saved; backfill incomplete' if self.current in self.backfill_incomplete
                               else 'Diarization saved')
            if (('❌' in line and 'No new audio files' not in line) or
                    'ERROR:' in line or re.search(r'^Failed: [1-9]\d*$', line) or
                    line.startswith('Traceback (most recent call last)')):
                self.errors.append(line)
        return self

    def snapshot(self):
        return dict(stage=self.stage, detail=self.detail, current_file=self.current,
                    file_index=self.index, total_files=self.total,
                    completed={k: len(v) for k, v in self.completed.items()},
                    error_count=len(self.errors), backfill_incomplete=len(self.backfill_incomplete),
                    last_error=self.errors[-1] if self.errors else None)


def inspect_progress(lines, pipeline):
    """Scan a whole log at once. Prefer ProgressTracker for a live run."""
    return ProgressTracker(pipeline).feed(lines).snapshot()
