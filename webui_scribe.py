"""Read-only Scribe planning. No credentials, model imports, or remote calls."""
import hashlib
import json
from pathlib import Path

from webui_run import FORMATS

LANGUAGES = ('Swedish (verbatim)', 'Swedish')
NOTES = [
    'Scribe may upload the listed recordings to https://scribe.sunet.se. Confirm only if remote processing is permitted.',
    'BrowserOS Neo must be logged into Scribe. The server environment must supply the encryption password. Noninteractive authentication failures appear in the run log.',
    'Local Whisper model, language, decoding, VAD, chunking, force-reprocessing and file-limit settings do not apply. Use the Scribe language and scope controls.',
    'Local community-1 diarization and exports use saved speaker and export settings. Precision, backfill, SenseVoice, parameter testing and timestamp correction are disabled.',
    'Each start resumes validated remote transcription but repeats local diarization and exports. Uncertain uploads or queue requests are never automatically replayed.',
    'Stop cancels local waiting and processing only. Remote jobs continue. Start with the same selection and language to resume.',
    'Original recordings and local results are preserved. Scribe output, state and cache stay in scribe-webui.',
]


def preview_scribe(audio_dir, selected='', language=LANGUAGES[0]):
    audio = Path(audio_dir).resolve()
    if language not in LANGUAGES:
        raise ValueError('Unsupported Scribe language')
    sources = [p for p in sorted(audio.iterdir()) if p.suffix.lower() in FORMATS and p.is_file()]
    if selected:
        sources = [p for p in sources if p.stem == selected]
        if len(sources) != 1:
            raise ValueError('Select one unambiguous recording basename, or use the explicit folder batch.')
    rows = []
    for source in sources:
        if source.is_file() and source.suffix.lower() in FORMATS:
            if source.is_symlink():
                raise ValueError('Scribe does not upload symlinked recordings')
            key = hashlib.sha256((source.name + language).encode()).hexdigest()[:16]
            work = audio / 'scribe-webui' / key
            if work.resolve() != work:
                raise ValueError('Scribe work directories must not use symlinks')
            evidence = []
            for path in [source, source.with_suffix('.swhisper'), work / 'scribe-state.json']:
                if path.exists():
                    stat = path.stat()
                    evidence.append((str(path), stat.st_mtime_ns, stat.st_size))
                else:
                    evidence.append((str(path), None))
            fingerprint = hashlib.sha256(json.dumps(evidence).encode()).hexdigest()
            rows.append(dict(file=source.name, base=source.stem, work_dir=str(work), fingerprint=fingerprint,
                             stage='Scribe + local diarization', action='process',
                             reason=f'Resume or upload; local exports in {work / "transcripts"}'))
    result = dict(audio_dir=str(audio), pipeline='scribe', language=language, selected=selected,
                  files=rows, notes=NOTES)
    result['token'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def main(argv=None):
    """Run only the reviewed files, serially, with the conservative adapter."""
    import sys
    import transcribe_scribe
    plan = json.loads((sys.argv[1:] if argv is None else argv)[0])
    for index, row in enumerate(plan['files'], 1):
        try:
            current = preview_scribe(plan['audio_dir'], language=plan['language'])
            actual = next((r for r in current['files'] if r['file'] == row['file']), None)
            if actual != row:
                raise ValueError('Recording, settings or Scribe state changed after preview. Review again.')
        except (ValueError, OSError) as exc:
            print(f'Error: {exc}', file=sys.stderr, flush=True)
            return 1
        print(f"Scribe file {index}/{len(plan['files'])}: {row['file']}", flush=True)
        code = transcribe_scribe.main([
            str(Path(plan['audio_dir']) / row['file']),
            '--work-dir', row['work_dir'], '--language', plan['language'],
            '--auth', 'browseros',
        ])
        if code:
            return code
        print(f"Scribe completed: {row['file']}", flush=True)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
