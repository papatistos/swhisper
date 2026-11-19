# swhisper - Swedish Whisper Transcription and Diarization

swhisper transcribes and diarizes recordings using the Swedish Whisper model [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) (the `strict`, more verbatim version) with [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) for word timestamps, and [pyannote](https://github.com/pyannote/pyannote-audio) or [Nemotron 3](https://huggingface.co/nvidia/Nemotron-3-Diarization) for speaker diarization. [SUNET Scribe](https://scribe.sunet.se) can do the transcription instead of local Whisper.

Until proven otherwise (and please do!), I believe this is the best available open-source solution for Swedish speech recognition and diarization. It is work in progress; PRs and suggestions are welcome.

- It prioritizes quality over speed, as academic research needs.
- It was developed and tested on a MacBook Pro M4 with 24 GB RAM, using the Apple GPU. Other systems, especially CUDA, are untested.

## Quick start

1. Copy the example configuration:
   ```bash
   cp .swhisper.env.example .swhisper.env
   ```
2. Set at least these two values in `.swhisper.env`:
   ```bash
   HUGGING_FACE_TOKEN="your_huggingface_token_here"
   SWHISPER_AUDIO_DIR="/path/to/your/audio/files"
   ```
3. Run the whole pipeline on the audio folder:
   ```bash
   python transcribe4.py   # Whisper, then pyannote 4 (community-1) diarization
   ```
   - `python transcribe.py` does the same with pyannote 3.1.
   - Stop it at any time with Ctrl+C. A restart resumes where it left off.
   - Raw transcripts go to `whisper-json-output/` in the audio folder. Diarization reads them from there.
   - To diarize already transcribed files only, run `python diarize4.py` (or `python diarize.py`).
   - Finished files are skipped. To redo one, delete its `.ok` file in `transcripts/`, or see [Re-running](#re-running).

Or use the [web UI](#web-ui) for all of this.

## Choosing models

### Transcription

- **Local Whisper** (default): KB-Whisper strict on your machine.
- **SUNET Scribe**: remote transcription, followed by local diarization. See [SUNET Scribe](#sunet-scribe-remote-transcription).

### Diarization

- **pyannote community-1** (default for `transcribe4.py`, `diarize4.py` and the web UI): local and free.
- **Nemotron 3**: local and fast, up to 8 speakers. See [Nemotron 3 diarization](#nemotron-3-diarization).
- **pyannote Precision-2**: PyannoteAI's hosted service. It fixes many segmentation and attribution errors of the open models, but uploads the audio. Set `SWHISPER_USE_PRECISION=true` and `PYANNOTEAI_API_KEY` (account at [pyannote.ai](https://pyannote.ai/)). The web UI then offers it as a choice.

### Nemotron 3 diarization

- [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) runs locally, tracks up to 8 speakers, and diarizes an hour of audio in about 30 seconds on an Apple Silicon CPU.
- It needs its own conda environment (Python 3.12 and an unreleased `transformers`):
  ```bash
  conda create -n nemo python=3.12
  conda run -n nemo pip install torch numpy librosa git+https://github.com/huggingface/transformers
  ```
- Use it always with `SWHISPER_PIPELINE_MODEL=nvidia/Nemotron-3-Diarization` in `.swhisper.env`.
- Use it for one run by choosing "Nemotron 3" in the web UI, with `SWHISPER_PIPELINE_MODEL=nvidia/Nemotron-3-Diarization python diarize4.py`, or with `--diarization-model nemotron` for Scribe.
- swhisper finds the `nemo` environment next to the `swhisper` environment. Set `SWHISPER_NEMOTRON_PYTHON` if it lives elsewhere. `ffmpeg` must be on the `PATH`.
- Do not set `SWHISPER_MAX_SPEAKERS`. It only relabels speech after inference, forcing extra speakers onto retained ones. `SWHISPER_MIN_SPEAKERS` and the pyannote tuning settings (`SWHISPER_VBX_*`, `SWHISPER_CLUSTERING_*`, …) do not apply.

### SUNET Scribe (remote transcription)

[SUNET Scribe](https://scribe.sunet.se) transcribes remotely; swhisper then runs local diarization, backfilling and the usual exports on the returned words. Only upload recordings that may be processed remotely.

- **Run it** from the web UI, or for one recording with `python transcribe_scribe.py recording.m4a`.
- **Sign in once** with SUNET's [scribe-api](https://github.com/SUNET/scribe-api) client:
  ```bash
  conda activate swhisper
  pip install "git+https://github.com/SUNET/scribe-api@5e707e2579dfadf055272d847749fcbd17907c09"
  python -m scribe login
  ```
  `login` prints a URL and a code. Check that the page shows the same code, and sign in with your SUNET login. The login is saved in `~/.config/scribe/token.json` and renews itself while you use it. Run `login` again when a run reports that it has expired.
- **Encryption password**: set `SUNET_SCRIBE_ENCRYPT_PWD` in `.swhisper.env` to your Scribe encryption password, because results are encrypted.
- **Language**: Scribe uses each recording's `SWHISPER_LANGUAGE`, from its sidecar or the global settings, as local Whisper does. Scribe cannot detect the language, and with the wrong one it translates: English audio with `sv` comes back as Swedish text. Supported: `sv`, `en`, `no`, `fi`, `da`, `is`, `se`, `fr`, `de`, `es`, `it`, `ru`, `uk`, `pt`, `nl`. The run preview shows each recording's language and refuses anything else.
- **Verbatim or standard**: verbatim (the default) uses KB-Whisper strict for Swedish and the verbatim models for English and Norwegian. Other languages use their normal model in both modes. On the command line, use `--mode standard`, or `--language` to override the recording's language.
- More detail: [SCRIBE_WEBUI.md](SCRIBE_WEBUI.md) 

## Web UI

```bash
pip install fastapi uvicorn   # one-time
python webui.py               # then open http://127.0.0.1:8722
python webui.py --restart     # restart, also when another terminal started it
python webui.py --status      # show the running server and its PID
python webui.py --stop        # stop the server
```

- **Files**: lists the recordings with their status (transcribed, diarized, own settings, will reprocess) and length, with search, filters and sorting.
- **Settings**: edits each recording's `.swhisper` sidecar, or several at once (tick them first). Empty fields inherit the global settings, which are editable too.
- **Run**: the transcription selector chooses Local Whisper or SUNET Scribe (verbatim or standard). The diarization selector chooses community-1, Nemotron 3, or Precision-2 when enabled. Tick recordings to run only those; otherwise the whole folder runs. A preview shows what will run before anything starts.
- **Reuse**: existing results are reused. A different diarization model diarizes again without transcribing again. Tick **Re-transcribe** or **Re-diarize** under a recording to redo that result in the next run.
- **Transcripts**: view and export transcripts, show or hide markers and pauses (see [Changing the view](#changing-the-view-without-re-running-anything)), play the audio from any paragraph, and review word timestamps.
- **Speaker names**: map `SPEAKER_00` and so on to names, stored in `speaker-names.json`. Pipeline outputs are never rewritten.
- **Comparison**: shows two transcript versions of a recording side by side, kept in sync by time.
- **Folders and restarts**: the UI remembers the last folder (`--audio-dir` overrides it for a session). Restart and stop leave running jobs alone, and the next UI reconnects to them.
- **Security**: the server only listens on localhost and refuses cross-origin and non-loopback requests.

`transcribe5.py` (diarization-first) is not in the web UI; run it from the command line.

## Re-running

- A run skips recordings that already have a result from the selected models.
- Web UI: tick **Re-diarize** or **Re-transcribe** under the recording. Re-transcribing also re-diarizes.
- Command line, for one run: `SWHISPER_RUN_REDIARIZE` or `SWHISPER_RUN_RETRANSCRIBE` take newline separated basenames. `SWHISPER_RUN_REPROCESS=diarization` or `=true` applies to every recording.
- `SWHISPER_ONLY_FILES` (newline separated basenames) restricts a run to those recordings.
- Choosing a different diarization model keeps the previous transcripts.
- Scribe: `--retranscribe` uploads again and moves the previous SUNET transcript to `superseded/`. `--rediarize` diarizes again.

## Output

Each transcript includes speaker statistics. They are only as good as the transcription and diarization, and almost worthless with the wrong number of speakers.

````
                   TURNS          WORDS  WORD/TURN          SPEECH    SPEED        PAUSES

SPEAKER_00   141 (27.9%)    791 (29.2%)        5.6   223.4 (30.3%)  212 wpm           32s
SPEAKER_01   155 (30.6%)    928 (34.3%)        6.0   229.8 (31.2%)  242 wpm           37s
SPEAKER_02   188 (37.2%)    963 (35.6%)        5.1   269.9 (36.6%)  214 wpm           57s
UNKNOWN        22 (4.3%)      25 (0.9%)        1.1     13.8 (1.9%)  109 wpm           16s
GAPS                                                                                 165s
Total       506 (100.0%)  2707 (100.0%)        5.3  737.0 (100.0%)  220 wpm  306s (29.4%)
````

- **Word timestamps** are in the `.json` files in `whisper-json-output/`. They come from whisper-timestamped and are not changed by diarization.
- **Pauses** between words can be shown in CA notation, e.g. `(.3)`. Their accuracy is not yet checked; they may be underestimated.
- **Disfluency markers** `[*]` mark sounds that could not be transcribed, one `*` per 0.1 s. They do not yet distinguish unidentified speech from background noise. `SWHISPER_PRESERVE_MARKERS=false` removes them.
- **Line breaks**: pauses of 1 s or more get blank lines around them in txt and rtf output. Change the threshold with `SWHISPER_SILENCE_LINEBREAK` (`0` disables).
- **Preamble**: `SWHISPER_OUTPUT_PREAMBLE` sets the text at the top of transcript files (`\n` for line breaks).
- **Interview speaker names**: with `RECORDING_TYPE=interview`, the speaker with the most words becomes `INTERVIEWEE_NAME` (default `INTERVIEWEE`) and the second `INTERVIEWER_NAME` (default `INTERVIEWER`). Set it globally or per recording. The names appear in new text exports and the web UI; JSON and RTTM keep the speaker IDs. Names you save yourself take precedence.
- **Statistics tool**: `diarize/` has a tool for comparing diarization results across settings. See `diarize/STATS_ANALYSIS_README.md`.
- **SenseVoice** (optional): emotion and audio-event detection (laughter, applause, …) on backfilled segments. Enable with `SWHISPER_BACKFILL_ENABLE_SENSEVOICE=true`. See [SENSEVOICE_INTEGRATION.md](SENSEVOICE_INTEGRATION.md).

### Changing the view without re-running anything

The diarized JSON keeps everything needed to show or hide markers and pauses, so the transcript viewer in the web UI changes these instantly:

| Toggle | What it does |
| --- | --- |
| disfluency markers | show or hide `[*]` / `[S02**]` |
| pause durations | show or hide `(1.3)` |
| line break at *n* s | blank line around pauses of at least *n* seconds |
| hide under *n* s | drop pauses shorter than *n* seconds |

- **Reset** returns to exactly what the pipeline produced. **Save as…** writes the current view to txt or rtf next to the transcript.
- Hiding markers merges the pauses around them, because markers take up real time: `Rastlös. (1.7) [***] (.5) [***] Jobbar` becomes `Rastlös. (4.9) Jobbar`.
- From the command line:
  ```bash
  python render_transcript.py run_diarized.json --no-disfluencies -o clean.txt
  python render_transcript.py run_diarized.json --no-pauses --format rtf -o plain.rtf
  python render_transcript.py run_diarized.json --all -o variants/   # all 8 combinations
  ```
- Two settings still need a re-run, because they change what data exists: `SWHISPER_DETECT_DISFLUENCIES` and `SWHISPER_PRESERVE_MARKERS=false`.
- To check that re-rendering reproduces a run exactly: `python -m transcript_render.selftest /path/to/audio/transcripts`.

## Configuration

- All settings live in `.swhisper.env` (over 100, each with its default in `.swhisper.env.example`). Uncomment and change only what you need.
- Priority, highest first: environment variables on the command line (e.g. `SWHISPER_DEVICE=cuda python transcribe4.py`), then `.swhisper.env`, then the code defaults.

### Per-file settings (sidecar files)

- A file named after the recording with a `.swhisper` extension (e.g. `interview1.swhisper` for `interview1.wav`) overrides selected settings for that recording, in the same `KEY=value` syntax:
  ```bash
  # interview1.swhisper
  SWHISPER_NUM_SPEAKERS=3     # sets min and max speakers for this file
  ```
- For many files: `python sidecar_tools.py template` writes a spreadsheet-friendly table, and `python sidecar_tools.py apply <csv>` turns it into sidecar files.
- Changing a processing setting in a sidecar reprocesses that recording, and only the affected stage. Naming settings (recording type, interview names) do not.
- See [SIDECAR_CONFIG.md](SIDECAR_CONFIG.md) for the allowed settings and precedence rules.

### Key settings

| Variable | Purpose | Default |
| --- | --- | --- |
| **Authentication** | | |
| `HUGGING_FACE_TOKEN` | Hugging Face token for pyannote models | *required* |
| `PYANNOTEAI_API_KEY` | PyannoteAI API key for Precision-2 | *optional* |
| `SUNET_SCRIBE_ENCRYPT_PWD` | Scribe encryption password | *for Scribe* |
| **Paths** | | |
| `SWHISPER_AUDIO_DIR` | Source audio files directory | `"audio"` |
| `SWHISPER_TEMP_DIR` | Temporary files and checkpoints | *none* |
| `SWHISPER_OUTPUT_DIR` | Diarization output directory (relative to the audio directory) | `"transcripts"` |
| **Transcription** | | |
| `SWHISPER_MODEL` | Whisper model from Hugging Face | `"KBLab/kb-whisper-large"` |
| `SWHISPER_DEVICE` | Processing device | `"mps"` |
| `SWHISPER_LANGUAGE` | Transcription language, for local Whisper and Scribe | `"sv"` |
| `SWHISPER_FILE_LIMIT` | Max files to process | *unlimited* |
| `SWHISPER_WHISPER_TIMESTAMP_OFFSET` | Seconds added to all Whisper timestamps | `0.1` |
| **Diarization** | | |
| `SWHISPER_MIN_SPEAKERS` | Minimum speakers | `1` |
| `SWHISPER_MAX_SPEAKERS` | Maximum speakers | *none* |
| `SWHISPER_PIPELINE_MODEL` | Diarization pipeline (`nvidia/Nemotron-3-Diarization` selects Nemotron) | `"pyannote/speaker-diarization-3.1"` for `diarize.py`; `transcribe4.py`, `diarize4.py` and the web UI use community-1 |
| `SWHISPER_USE_PRECISION` | Offer or use the Precision-2 service | `false` |
| `SWHISPER_NEMOTRON_PYTHON` | Python binary of the environment that runs Nemotron | the `nemo` conda env |
| **Processing** | | |
| `SWHISPER_BACKFILL_ENABLED` | Re-transcribe empty turns | `true` |
| `SWHISPER_BACKFILL_TRY_HARDER` | Use the configured decoding retries for backfill | `false` |
| `SWHISPER_BACKFILL_IGNORE_WORDS` | Words to discard when backfilling | *none* |
| `SWHISPER_INCLUDE_SILENCE` | Include silence markers | `true` |
| `SWHISPER_PRESERVE_MARKERS` | Preserve disfluency markers | `true` |

## Limitations and known issues

- **Verbatim**: KB-Whisper strict is the most verbatim model available, but still drops many sounds and filler words that some transcription standards need.
- **Diarization** is the weakest step. Expect some manual speaker correction.
  - Tune the diarization settings in `.swhisper.env` for your audio.
  - Setting `SWHISPER_MAX_SPEAKERS` 1–2 above the real number often works better for pyannote; post-processing sometimes removes the extra speakers. (Not for Nemotron.)
  - Precision-2 fixes many errors but uploads the audio, which research data often may not.
- **Missed speech**: voice activity detection (silero and auditok) misses some speech, which then is not transcribed.
  - Backfilling re-transcribes diarized turns that got no words. It recovers some words but can hallucinate on short segments.
  - Unrecovered turns get a `[ * ]` marker (with spaces, unlike Whisper's `[*]`).
  - Words that are often hallucinated can be ignored with `SWHISPER_BACKFILL_IGNORE_WORDS`. `SWHISPER_BACKFILL_ENABLED=false` turns backfilling off.
  - By default, backfill makes one decoding attempt at temperature 0, and results that fail the confidence or repetition checks become `[-]` markers. `SWHISPER_BACKFILL_TRY_HARDER=true` (globally or per file) uses the configured temperature retries instead. Changing it invalidates that mode's backfill cache.
- **Timestamp shift**: whisper-timestamped word timestamps seem about 0.1 s early. `SWHISPER_WHISPER_TIMESTAMP_OFFSET` (default `0.1`) shifts them before diarization; `0.0` disables it.

## Acknowledgments

This project builds upon the excellent work of:

- **[KBLab](https://huggingface.co/KBLab)** for the [kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) Swedish Whisper model (which is based on **OpenAI**'s original [Whisper](https://github.com/openai/whisper) speech recognition model).
- **[whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)** by Jérôme Louradour for precise word-level timestamps
- **[pyannote-audio](https://github.com/pyannote/pyannote-audio)** by Hervé Bredin for speaker diarization capabilities
- **[scribe-api](https://github.com/SUNET/scribe-api)** by Kristofer Hallin at [SUNET](https://www.sunet.se) for access to SUNET Scribe

## Citations

If you use this pipeline in your research, please cite the repo:
```bibtex
@misc{papatistos2025swhisper,
  title={Swhisper - Swedish Whisper Transcription and Diarization},
  author={Papatistos},
  journal={GitHub repository},
  year={2025},
  publisher={GitHub},
  howpublished = {\url{https://github.com/papatistos/swhisper}}
}
```
Please also cite the other works used in the pipeline as indicated in their respective repositories.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
