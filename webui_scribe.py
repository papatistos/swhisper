"""Read-only Scribe planning. No credentials, model imports, or remote calls."""
import hashlib
import json
from pathlib import Path

from run_scope import DIARIZATION_MODELS
from webui_run import FORMATS
from audio_sources import source_audio_files
from scribe_processing import completion_marker, diarization_reason, effective_settings

LANGUAGES = ('Swedish (verbatim)', 'Swedish')
NOTES = [
    'Scribe may upload the listed recordings to https://scribe.sunet.se. Confirm only if remote processing is permitted.',
    'BrowserOS Neo must be logged into Scribe. The server environment must supply the encryption password. Noninteractive authentication failures appear in the run log.',
    'SUNET controls the initial transcription. Whisper decoding and prompt settings affect local backfill only; language, task and VAD are fixed for recovery. Choose the Scribe language in the transcription selector.',
    'Local diarization (community-1, or Nemotron when selected), Whisper backfill and exports use saved settings. SenseVoice is optional for backfill snippets only. Scribe timestamp offset defaults to zero. Precision and parameter testing remain unavailable.',
    'Each start resumes validated remote transcription. Local diarization runs when it has no result from the selected model, when a new Scribe transcript arrives, or when forced. Uncertain uploads or queue requests are never automatically replayed.',
    'Stop cancels local waiting and processing only. Remote jobs continue. Start with the same selection and language to resume.',
    'Original recordings and local results are preserved. Scribe output, state and cache stay in scribe-webui.',
]


def local_config(source, work, diarization_model=''):
    # Load only the config module, avoiding diarize/__init__ and model imports.
    import importlib.util
    import sys
    from scribe_processing import configure
    name = 'scribe_preview_config'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('diarize') / 'config.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = module.DiarizationConfig()
    if diarization_model:
        config.pipeline_model = DIARIZATION_MODELS[diarization_model]
    return configure(config, source, work)


def _named(choice, base):
    """True for every recording, or a collection of basenames to redo."""
    return choice is True or (not isinstance(choice, bool) and base in choice)


def previously_uploaded(work):
    """Whether this recording already went to SUNET, so a new upload needs no new consent."""
    try:
        state = json.loads((work / 'scribe-state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    from transcribe_scribe import REMOTE_JOB_KEYS
    return (isinstance(state, dict) and any(state.get(key) for key in REMOTE_JOB_KEYS)
            or (work / 'whisper-json-output' / 'audio.json').is_file()
            or (work / 'superseded').is_dir())


def preview_scribe(audio_dir, selected='', language=LANGUAGES[0], diarization_model='', retranscribe=False,
                   rediarize=False, only=None):
    """Plan a Scribe run for the whole folder, one *selected* recording, or the
    basenames in *only*. *retranscribe* and *rediarize* are True for every
    recording, or collections of basenames to redo in this run."""
    audio = Path(audio_dir).resolve()
    if language not in LANGUAGES:
        raise ValueError('Unsupported Scribe language')
    if diarization_model and diarization_model not in DIARIZATION_MODELS:
        raise ValueError('Unsupported diarization model')
    sources = [audio / name for name in sorted(source_audio_files(audio))]
    if selected:
        sources = [p for p in sources if p.stem == selected]
        if len(sources) != 1:
            raise ValueError('Select one unambiguous recording basename, or use the explicit folder batch.')
    if only is not None:
        sources = [p for p in sources if p.stem in only]
        stems = [p.stem for p in sources]
        if not only or sorted(stems) != sorted(set(only)):
            raise ValueError('Each selected recording must match exactly one source file in this folder.')
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
            config = local_config(source, work, diarization_model)
            settings = effective_settings(config)
            recovery = f"local Whisper backfill {settings['backfill_model']}" if settings['backfill_missing_turns'] else "backfill off"
            # transcribe_scribe contacts SUNET only when no saved transcript exists,
            # or when this run explicitly retranscribes.
            saved = (work / 'whisper-json-output' / 'audio.json').is_file()
            again = _named(retranscribe, source.stem)
            remote = again or not saved
            stamp = completion_marker(work)
            evidence.append((str(stamp), stamp.stat().st_mtime_ns if stamp.exists() else None))
            fingerprint = hashlib.sha256(json.dumps(evidence).encode()).hexdigest()
            # A new transcript always needs a new diarization.
            reuse_reason = diarization_reason(config, source, work)
            diarize_reason = ('New Scribe transcript' if remote and saved
                              else diarization_reason(config, source, work, _named(rediarize, source.stem)))
            transcription = ("Uploads again; the saved Scribe transcript moves to superseded/" if saved and again
                             else "Resume or upload" if remote else "Reuses the saved Scribe transcript; no upload or SUNET login")
            diarization = (f'reuses the local {settings["pipeline_model"]} diarization' if diarize_reason is None
                           else f'diarization {settings["pipeline_model"]} ({diarize_reason})')
            rows.append(dict(local_settings=settings, file=source.name, base=source.stem, work_dir=str(work), fingerprint=fingerprint,
                             stage='Scribe + local diarization' if remote else 'Local diarization of saved Scribe transcript',
                             action='process' if remote or diarize_reason else 'skip', remote=remote,
                             consent=remote and not previously_uploaded(work),
                             retranscribe=again, rediarize=_named(rediarize, source.stem),
                             reusable=dict(transcription=saved, diarization=saved and reuse_reason is None),
                             reason=f'{transcription}; {diarization}; {recovery}; SenseVoice {"on" if settings["backfill_enable_sensevoice"] else "off"}; offset {settings["whisper_timestamp_offset"]:g}s; local exports in {work / "transcripts"}'))
    listed = lambda choice: choice if isinstance(choice, bool) else sorted(choice)
    result = dict(audio_dir=str(audio), pipeline='scribe', language=language, selected=selected,
                  only=sorted(only) if only is not None else None,
                  files=rows, notes=NOTES, remote=any(row['remote'] for row in rows),
                  consent_required=any(row['consent'] for row in rows),
                  diarization_model=diarization_model, retranscribe=listed(retranscribe),
                  rediarize=listed(rediarize))
    result['token'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def main(argv=None):
    """Run only the reviewed files, serially, with the conservative adapter."""
    import sys
    import transcribe_scribe
    plan = json.loads((sys.argv[1:] if argv is None else argv)[0])
    files = [row for row in plan['files'] if row['action'] != 'skip']
    for index, row in enumerate(files, 1):
        try:
            current = preview_scribe(plan['audio_dir'], language=plan['language'], only=plan.get('only'),
                                     diarization_model=plan.get('diarization_model', ''),
                                     retranscribe=plan.get('retranscribe', False),
                                     rediarize=plan.get('rediarize', False))
            actual = next((r for r in current['files'] if r['file'] == row['file']), None)
            if actual != row:
                raise ValueError('Recording, settings or Scribe state changed after preview. Review again.')
        except (ValueError, OSError) as exc:
            print(f'Error: {exc}', file=sys.stderr, flush=True)
            return 1
        print(f"Scribe file {index}/{len(files)}: {row['file']}", flush=True)
        code = transcribe_scribe.main([
            str(Path(plan['audio_dir']) / row['file']),
            '--work-dir', row['work_dir'], '--language', plan['language'],
            '--auth', 'browseros',
        ] + (['--diarization-model', plan['diarization_model']] if plan.get('diarization_model') else [])
          + (['--retranscribe'] if row.get('retranscribe') else [])
          + (['--rediarize'] if row.get('rediarize') else []))
        if code:
            return code
        print(f"Scribe completed: {row['file']}", flush=True)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
