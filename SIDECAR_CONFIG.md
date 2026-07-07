# Per-file configuration (sidecar files)

> There is also a browser-based editor for these settings: `python webui.py` (requires `pip install fastapi uvicorn`), then open http://127.0.0.1:8722. It lists your audio files with their processing status, edits sidecars through a form, and starts/stops pipeline runs with live log output.

Most settings in `.swhisper.env` apply to every audio file in a run. Sidecar files let you override selected settings **for a single audio file** — most commonly the number of speakers.

## Quick start

Create a file next to the audio file, named after its basename with a `.swhisper` extension:

```
audio/
├── interview1.wav
├── interview1.swhisper   ← settings for interview1.wav only
└── interview2.wav        ← uses global settings
```

The sidecar uses the same `KEY=value` syntax and the same variable names as `.swhisper.env`:

```bash
# interview1.swhisper
SWHISPER_NUM_SPEAKERS=3                       # shorthand: sets min AND max speakers
SWHISPER_INITIAL_PROMPT="Intervju om arbetet"
SWHISPER_BACKFILL_IGNORE_WORDS=Balans,Tack
```

That's it — all pipelines (`transcribe.py`, `transcribe4.py`, `transcribe5.py`, and standalone `diarize.py`/`diarize4.py`) pick sidecars up automatically. Overrides apply to that file only; every other file uses the global configuration.

## Bulk editing via CSV

If you have many files, editing a table is easier than creating sidecar files one by one:

```bash
python sidecar_tools.py template            # writes swhisper_settings.csv to the audio dir
# ... edit the CSV in Excel/Numbers: one row per audio file, one column per setting;
#     fill in cells only where a file should deviate from the global config ...
python sidecar_tools.py apply audio/swhisper_settings.csv --dry-run   # preview
python sidecar_tools.py apply audio/swhisper_settings.csv             # write sidecars
```

Notes on the CSV workflow: `template` pre-fills the table from existing sidecar files, so it doubles as an export of the current state and is safe to re-run. `template --all` includes a column for every allowed key (default is a curated subset). `apply` only touches files for rows that have at least one non-empty setting; add `--prune` to delete sidecars for rows you've cleared. Rows with invalid values are skipped with an error, leaving any existing sidecar untouched.

## Allowed settings

Run `python sidecar_tools.py list-keys` for the authoritative list. In summary:

**Diarization** (the usual suspects): `SWHISPER_NUM_SPEAKERS` (shorthand for min=max), `SWHISPER_MIN_SPEAKERS`, `SWHISPER_MAX_SPEAKERS`, clustering parameters (`SWHISPER_VBX_*`, `SWHISPER_CLUSTERING_*`, `SWHISPER_SEGMENTATION_THRESHOLD`, `SWHISPER_EMBEDDING_THRESHOLD`, …), alignment and formatting (`SWHISPER_WHISPER_TIMESTAMP_OFFSET`, `SWHISPER_SMOOTHING_ENABLED`, `SWHISPER_MIN_SPEAKER_WORDS`, `SWHISPER_PRESERVE_MARKERS`, silence settings, `SWHISPER_OUTPUT_PREAMBLE`, `SWHISPER_TSV_WORD_PER_LINE`), and backfill options (`SWHISPER_BACKFILL_ENABLED`, `SWHISPER_BACKFILL_IGNORE_WORDS`, `SWHISPER_BACKFILL_ENABLE_SENSEVOICE`, …).

**Whisper decoding**: `SWHISPER_LANGUAGE`, `SWHISPER_TASK`, `SWHISPER_INITIAL_PROMPT`, `SWHISPER_VAD`, `SWHISPER_BEAM_SIZE`, `SWHISPER_BEST_OF`, `SWHISPER_TEMPERATURE`, thresholds, `SWHISPER_DETECT_DISFLUENCIES`, `SWHISPER_SEED`, ….

**Chunking**: `SWHISPER_CHUNK_DURATION`, `SWHISPER_MIN_SILENCE`, `SWHISPER_OVERLAP_DURATION`, and the diarization-first chunking parameters.

**Per-file reprocessing**: `SWHISPER_FORCE_REPROCESS=true` re-runs transcription for this file only; `SWHISPER_DIARIZE_FORCE_REPROCESS=true` re-runs diarization for this file only.

### Deliberately not allowed per file

Settings that are global by nature are rejected with a warning explaining why: authentication tokens, directory paths, device selection (`SWHISPER_DEVICE`, …), model ids (`SWHISPER_MODEL`, `SWHISPER_PIPELINE_MODEL`, `SWHISPER_USE_PRECISION`, …), pipeline-mode switches (`SWHISPER_DIARIZATION_FIRST`), batch options (`SWHISPER_FILE_LIMIT`), caching toggles, and parameter-testing options. Unknown keys and unparseable values also produce warnings and are ignored — a broken sidecar never aborts a run.

## Reprocessing semantics

When a file has already been processed, its sidecar can still trigger reprocessing:

1. **Force flags** (see above) always win.
2. **Staleness**: if the sidecar was modified *after* the file's output was written (the whisper JSON for the transcription stage, the `.ok` marker for diarization), the file is reprocessed automatically. This is stage-aware: a sidecar containing only diarization keys (e.g. `SWHISPER_MAX_SPEAKERS`) will re-run diarization but **not** re-run the expensive transcription step. Changed transcription keys re-run both stages (new JSON invalidates the diarized transcript). Disable auto-reprocessing globally with `SWHISPER_SIDECAR_AUTOREPROCESS=false` in `.swhisper.env`.

## Edge cases and details

- **Precedence**: sidecar > `.swhisper.env` > built-in defaults. Sidecar values are applied per file and fully restored afterwards (also on errors), so files later in the queue are unaffected.
- **`SWHISPER_NUM_SPEAKERS` vs explicit min/max**: if both appear, the shorthand is applied first and explicit `MIN`/`MAX_SPEAKERS` take precedence (with a warning), regardless of line order.
- **Diarization cache**: cache keys include speaker counts and clustering parameters, so per-file overrides get their own cache entries — no stale results, and switching a value back re-uses the earlier cache.
- **Checkpoint resume**: if transcription resumes from a mid-file checkpoint, chunks transcribed before a sidecar change keep their original settings (a note is printed). Delete the checkpoint to re-transcribe consistently.
- **Basename matching**: `interview1.swhisper` matches any `interview1.*` audio file. If you keep both `interview1.wav` and `interview1.mp3` in the same directory, they share the sidecar (they'd also collide on all other outputs, so avoid that anyway).
- **Values with `#` or spaces** should be quoted, exactly as in `.swhisper.env`. Use `\n` for line breaks (e.g. in `SWHISPER_OUTPUT_PREAMBLE`). Lists are comma-separated.
