# swhisper - Swedish Whisper Transcription and Diarization

Speech transcription and diarization pipeline using the (strict) Swedish Whisper model [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) with [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) and [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Performance and limitations
Until proven otherwise (and please do!), I believe that this is currently the best available open-source solution for Swedish speech recognition and diarization. 

The excellent transcription quality is due to the KBLab Whisper model, which has been fine-tuned on a large Swedish dataset. The pipeline uses the `strict` version of the model, which prioritizes verbatim transcription as much as possible (unfortunately, the model still filters out a lot of sounds and filler words that are relevant for certain transcription standards).

The **limitations** are mostly in the diarization step. I tried the `precision-2` model and it seems to solve a lot of the segmentation/attribution errors that I had with the open source models. But because researchers are often not allowed to upload sensitive audio data to third-party services, I will continue to develop this pipeline for the open-source models. If nothing prevents you from using PyannoteAI's cloud service, I recommend using the `precision-2` model by setting `SWHISPER_USE_PRECISION=true` in `.swhisper.env`. (You will need to set up an account and get an API key from [pyannote.ai](https://pyannote.ai/).)

A second limitation is in the voice activity detection (VAD) step, which currently misses a few speech segments (both with silero and auditok), which entails that those audio segments are not transcribed. I have tried to address this in the pipeline by automatically (re-)transcribing speaker segments (from pyannote) that did not get any words assigned ("empty turns"). Not perfect, but it does recover some of the missing words.

Depending on your audio-files and settings, you may still have to do some manual work to assign the right speakers, but it's still much better than starting from scratch. Try to tweak the diarization parameters in `.swhisper.env` to see what works for your audio. I recommend specifying 1-2 more speakers than you have in the audio (via `SWHISPER_MAX_SPEAKERS`), it seems to work better and my post-processing sometimes manages to remove the extra speakers.

- Although the pipeline is fully functional, it is still work in progress. PRs and suggestions are welcome!

- The scripts have been developed and tested on a MacBook Pro M4 with 24 GB RAM. It works with the Apple GPU but I'm not sure if they will run out of the box on other systems, especially with CUDA support.

- The pipeline prioritizes quality over speed (as this is what counts in academic research), so, if you're in a hurry, this is probably not for you. 

- Word-level timestamps are provided in the `.json` output files in the `whisper-json-output` directory. They come from the `whisper-timestamped` model and are not affected by diarization.

- there is a basic stats analysis tool in `diarize/` that can help compare diarization results across different parameter settings. See `diarize/STATS_ANALYSIS_README.md` for details.

- **Optional SenseVoice integration** for emotion recognition and audio event detection on backfilled segments (laughter, crying, applause, etc.). Enable with `SWHISPER_BACKFILL_ENABLE_SENSEVOICE=true` in `.swhisper.env`. See [SENSEVOICE_INTEGRATION.md](SENSEVOICE_INTEGRATION.md) for details.

- Each transcript includes basic speaker statistics in a table like the one below. Note, however, that these stats depend on the quality of both the transcription and diarization. They are almost worthless if the wrong number of speakers has been detected.

````
                   TURNS          WORDS  WORD/TURN          SPEECH    SPEED        PAUSES

SPEAKER_00   141 (27.9%)    791 (29.2%)        5.6   223.4 (30.3%)  212 wpm           32s
SPEAKER_01   155 (30.6%)    928 (34.3%)        6.0   229.8 (31.2%)  242 wpm           37s
SPEAKER_02   188 (37.2%)    963 (35.6%)        5.1   269.9 (36.6%)  214 wpm           57s
UNKNOWN        22 (4.3%)      25 (0.9%)        1.1     13.8 (1.9%)  109 wpm           16s
GAPS                                                                                 165s
Total       506 (100.0%)  2707 (100.0%)        5.3  737.0 (100.0%)  220 wpm  306s (29.4%)
````

- When the voice activity detection algorithm (VAD) failed to detect speech in parts of the audio, those parts will not be transcribed in the initial transcription step. If the diarization step then identifies a speaker segment in such a part, we send this segment for transcription and fill in the results into the "empty turn" (backfilling). While this recovers some missing words, it can also introduce hallucination artifacts (possibly due to the short duration of these segments). To at least partially work around this, you can specify a list of words to ignore during backfilling. For example, if you notice that the word "Balans" is frequently hallucinated in empty turns, you can add it to the ignore list in the `.swhisper.env` file using the `SWHISPER_BACKFILL_IGNORE_WORDS` variable. Alternatively, you can disable backfilling entirely via `SWHISPER_BACKFILL_ENABLED=false`.

- I am seeing what looks like a relatively consisten time-shift of about 0.1s in the word-level timestamps provided by `whisper-timestamped` (relative to the actual audio). I'm not sure why this is happening, but I introduced an option to manually adjust such a shift via `SWHISPER_TIMESTAMP_ADJUSTMENT` in `.swhisper.env` (set to a negative value to move timestamps earlier). This adjustment is applied before diarization, so the diarized transcript will reflect the adjusted timestamps. Note that this is a manual workaround and may not be necessary for all use cases, but it is set to `-0.1` by default.


### Transcript formatting options

Set `RECORDING_TYPE=interview` in `.swhisper.env` to name speakers automatically after diarization and word alignment. The speaker with the most transcribed words receives `INTERVIEWEE_NAME` (default `INTERVIEWEE`); the second receives `INTERVIEWER_NAME` (default `INTERVIEWER`). Leave `RECORDING_TYPE` empty to disable this. Silence, unknown speakers and non-word markers do not count. Ties follow first appearance; other speakers keep their labels.

These options are available under **Common settings** in both the global settings and each recording's settings. Per-file values override the global defaults. Choose **Interview** to enable automatic naming and modal suggestions, **Not specified** to disable both for that file, or **Use global** to inherit the global recording type. Saved manual names remain in place.

Names appear in new text exports and the WebUI. The diarized JSON and RTTM retain speaker IDs, with the automatic name mapping stored in JSON metadata. For an existing transcript, open **Speaker names** to review the proposed names and save them. Saved manual names take precedence, and each transcript version has its own mapping.

- The duration of silences between words can be included in the transcript (using CA notation, e.g. (.3) for a .3 second silence). I have not yet investigate the accuracy of these durations. They are based on the word-level provided by `whisper-timestamped` and I have a feeling that they might be underestimated...

- There is also an option to include disfluence markers (as `[*]`) for sounds that could not be transcribed. By default, these markers are preserved in the diarized transcript, but can be disabled via `SWHISPER_PRESERVE_MARKERS=false`. The discontinuity markers are longer (more asterisks) the longer the unidentified sound is (one `*` per 0.1 s). It looks like the current settings don't properly distinguish between unidentified speech and background noise. Tweaking of the whisper-timestamped settings (in `.swhisper.env`) might help.

- The output preamble for transcript files can be customized via `SWHISPER_OUTPUT_PREAMBLE` in `.swhisper.env`. Use `\n` for line breaks. This allows you to include specific notes or instructions for users reviewing the transcripts.

- By default, silence of 1 second or longer are surrounded by blank lines in the transcript (even when there is no speaker change) to improve readability. This threshold can be adjusted via `SWHISPER_SILENCE_LINEBREAK` (set to `0` to disable). This setting only concerns rtf and txt output files.

### Changing the view without re-running anything

The settings above decide what a run *writes*, but they do not decide what you
can *read*. Everything needed to show or hide markers and pauses is kept in the
diarized JSON, so any of those choices can be changed after the fact, in
milliseconds, from the transcript viewer in the web UI:

| Toggle | What it does |
| --- | --- |
| disfluency markers | show or hide `[*]` / `[S02**]` |
| pause durations | show or hide `(1.3)` |
| line break at *n* s | blank line around pauses of at least *n* seconds |
| hide under *n* s | drop pauses shorter than *n* seconds |

**Reset** returns to exactly what the pipeline produced, byte for byte.
**Save as…** writes the current view to `txt` or `rtf` next to the transcript,
with a suffix naming it (`..._nodisfl_nopause.txt`).

Hiding disfluency markers is not the same as deleting them. A marker occupies
real time, so removing it merges the pauses on either side:

    with markers      Rastlös. (1.7) [***************] (.5) [*************] Jobbar
    without markers   Rastlös. (4.9) Jobbar

Deleting the markers and keeping the printed durations would claim 2.2 s of
silence where the recording has 4.9 s. The pauses are therefore rebuilt from
the word timings rather than edited in place.

The same thing from the command line, for scripting or for people who want the
files up front:

```bash
python render_transcript.py run_diarized.json --no-disfluencies -o clean.txt
python render_transcript.py run_diarized.json --no-pauses --format rtf -o plain.rtf
python render_transcript.py run_diarized.json --all -o variants/   # all 8 combinations
```

`--all` takes about 0.25 s and 1.1 MB per hour-long interview, so generating
every combination up front is perfectly affordable — it is just rarely what you
want, since the toggles are instant and sixteen near-identical files are hard
to tell apart.

Two settings stay run-time settings, because they change what data exists
rather than how it is shown. `SWHISPER_DETECT_DISFLUENCIES` controls what
Whisper emits, and `SWHISPER_PRESERVE_MARKERS=false` discards markers before
diarization, which also shifts the pauses. Leave both at their defaults and the
toggles cover the same ground without a re-run.

To check that re-rendering still reproduces a run exactly:

```bash
python -m transcript_render.selftest /path/to/audio/transcripts
```



### To-dos
- [x] migrate to pyannote-audio 4.0 (and the `community-1` model)
  - to use pyannote 4 with `community-1` use `diarize4.py` (instead of `diarize.py`) or `transcribe4.py` for the entire pipeline
  - there is also an option to use the premium `precision-2` diarization service from pyannote (see `.swhisper.env` for details)
- [x] do something with speaker segments that didn't get any words
  - these are now reprocessed by (re-)transcribing them and backfilling the words in the diarized transcript. While this does find the odd word here and there, it also sometimes introduces artifacts (possibly due to the short duration of these segments). If no word has been transcried, we add a discontinuity marker `[ * ]` instead, because we trust pyannote that there is some voice activity there. These markers have spaces around them to distinguish them from the regular disfluency markers (`[*]`) added by whisper-timestamped.
- [x] combine settings in one config file that is read by both `config.py`s
- [x] fix caching of empty turns (the current code that should persist those transcriptions doesn't seem to work)
- [ ] add (more) speaker stats
- [x] add speaker stats to the preamble
- [x] per-file settings via sidecar files and/or csv table (see [SIDECAR_CONFIG.md](SIDECAR_CONFIG.md))
- [ ] improve documentation (let me know what is particularly unclear)
- [ ] add command line options?

## Configuration and usage

All configuration is managed through a single `.swhisper.env` file that contains over 100 customizable parameters for both transcription and diarization. The file includes:

- **Authentication tokens** (HuggingFace, PyannoteAI)
- **Directory paths** (audio, output, temp, logs)
- **Transcription settings** (model, device, chunk size, file limits)
- **Whisper parameters** (language, VAD, beam size, temperature, thresholds)
- **Diarization settings** (speakers, pipeline model, clustering parameters)
- **Output formatting** (silence markers, preamble text, TSV format)
- **Backfill settings** (re-transcription of empty turns, caching)
- **Advanced options** (parameter testing, precision-2 service)

Each setting in the file shows its default value, so you only need to uncomment and modify the settings you want to change.

### Per-file settings (sidecar files)

Selected settings — most importantly the number of speakers — can be overridden **per audio file** by placing a sidecar file next to the audio file, named after its basename with a `.swhisper` extension (e.g. `interview1.swhisper` for `interview1.wav`). Sidecars use the same `KEY=value` syntax and variable names as `.swhisper.env`:

```bash
# interview1.swhisper
SWHISPER_NUM_SPEAKERS=3     # shorthand: sets min AND max speakers for this file
```

For bulk editing there is a CSV workflow: `python sidecar_tools.py template` generates a spreadsheet-friendly table (one row per audio file), and `python sidecar_tools.py apply <csv>` turns it into sidecar files. Editing processing settings in a sidecar automatically triggers reprocessing of the affected file (stage-aware: diarization-only changes don't re-run transcription). Recording type and interview names are naming settings; changing them in the WebUI does not request reprocessing. See [SIDECAR_CONFIG.md](SIDECAR_CONFIG.md) for the list of allowed settings, precedence rules, and edge cases.

### Web UI

A local browser UI covers per-file settings and run control:

```bash
pip install fastapi uvicorn   # one-time
python webui.py               # then open http://127.0.0.1:8722
python webui.py --restart     # restart even if another terminal or agent started it
python webui.py --status      # show the running server and its PID
python webui.py --stop        # stop the UI server
```

Starting an already running WebUI opens that instance and prints the restart and stop commands. Restart and stop leave pipeline jobs running; the next WebUI reconnects to them. Restart runs in the current terminal, where Ctrl+C also stops the UI. Use the same `--port` for these commands if you chose a custom port. Process discovery uses `lsof` and `ps` and only stops a Python WebUI from this checkout owned by your user. Startup options must be supplied again when restarting.

It shows your audio files with per-file status (transcribed / diarized / per-file settings / will-reprocess) and recording length, edits the `.swhisper` sidecar files through a grouped form (empty fields = global settings, shown as placeholders), and starts/stops runs with live log output. The directory selector remembers your last folder across restarts in the local `.swhisper-webui.json` file and adds it to the saved-folder history in `.swhisper.env`. If that folder is unavailable, the UI falls back to `SWHISPER_AUDIO_DIR`. `--audio-dir` overrides both for the session. Reconnecting to an existing run can select that run's folder.

The run controls have two selectors. The transcription selector chooses **Local Whisper** (`transcribe4.py`) or **SUNET Scribe (remote)**, which is offered as **verbatim** or **standard** (see [SUNET Scribe](#sunet-scribe-remote-transcription) below). The diarization selector chooses **pyannote community-1** or **Nemotron 3**, and it also offers **pyannote Precision-2** when `SWHISPER_USE_PRECISION=true`. A run reuses existing results: a recording that already has a transcript is not transcribed again, and a recording that already has a diarization from the selected model is not diarized again. To add or change speakers without redoing the transcription, choose a different diarization model and select Run. When a recording already has a result from the selected model, **Re-transcribe** or **Re-diarize** appears under its name in the file list; tick it to redo that result in the next run. Re-transcribing also re-diarizes, because diarization is aligned to the transcript. The ticks apply to the next run only. The diarization-first pipeline `transcribe5.py` is not offered in the web UI; run it from the command line. Local Whisper runs each Whisper chunk in a separate process so its model and native GPU allocations are released on exit. This prevents allocations from accumulating across chunks; it does not cap an individual chunk's peak memory.

Tick files or click **Select all shown** to select the files matching the current search and filter. The Settings tab applies to all checked files: shared values are shown normally, differing values are marked **Mixed values**, and **Save to N files** applies only fields you edit. Each file keeps its other settings and comments. Selections remain checked when filtered out, with a count of hidden selections beside the button. **Clear selection** returns to settings for the open file. Changing the selection prompts you to save or discard any draft. The same selection also controls which files run.

- **Run a selection.** Tick files in the list to run only those, then select "Run". The "Run Preview" modal shows the selection count, lists what will run, and puts already-finished files behind a disclosure. The preview and processing queue prefer an original non-WAV recording over its same-basename converted WAV; recordings available only as WAV remain inputs. To rerun an existing result, tick "Re-transcribe" or "Re-diarize" under the recording before opening the preview. A different diarization model runs automatically when the current output was produced by another model. Internally this sets `SWHISPER_ONLY_FILES`, a newline separated list of basenames that both pipeline stages honour, so the same restriction works from a shell.
- **Name the speakers.** "Speakers…" in the transcript view maps `SPEAKER_00` and friends to real names, stored in `speaker-names.json` in the audio folder. Names are applied when the transcript is re-rendered from the diarized JSON, so they appear in the view, the timestamped review list and exports, and never rewrite a transcript the pipeline produced.
- **Compare two versions.** The Comparison tab shows two transcript versions of one recording side by side. Both panes use the same Show settings, which are shared with the Transcript tab. Scrolling one pane keeps the same moment of the recording in the middle of the other: each paragraph's start and end time come from the word timestamps in the diarized JSON, and positions inside a paragraph are interpolated. A version without a diarized JSON is shown as the pipeline wrote it, and its `[hh:mm:ss]` paragraph headers are used for the syncing instead. Selecting a speaker heading plays the recording from that paragraph. While it plays, both panes keep the moment being played in the middle and highlight its paragraph; scrolling by hand pauses this until you select a heading or play again. "Review with timestamps" switches both panes to their lists of timestamped segments, which stay synced in the same way.
- **Theme.** The interface follows the operating system's light or dark setting.

The server binds to localhost only and additionally refuses requests whose `Host` is not a loopback name and cross-origin requests, which closes DNS rebinding and stops another page in your browser from driving the API.

### SUNET Scribe (remote transcription)

[SUNET Scribe](https://scribe.sunet.se) can do the initial transcription instead of local Whisper. swhisper uploads the recording, waits for the job, and then runs local diarization, backfilling and the usual exports on the returned words. Only confirm the upload if remote processing is permitted for the recording. Run it from the web UI, or for one recording with `python transcribe_scribe.py recording.m4a`.

**Sign in once.** swhisper talks to Scribe through SUNET's [scribe-api](https://github.com/SUNET/scribe-api) client. Install it in the `swhisper` environment and sign in:

```bash
conda activate swhisper
pip install "git+https://github.com/SUNET/scribe-api@5e707e2579dfadf055272d847749fcbd17907c09"
python -m scribe login
```

`login` prints a URL and a code. Open the URL, check that the page shows the same code, and sign in with your usual SUNET login. The login is saved in `~/.config/scribe/token.json` and renews itself while you use it; run `login` again when a run reports that it has expired. Results are encrypted, so also set `SUNET_SCRIBE_ENCRYPT_PWD` in `.swhisper.env` to your Scribe encryption password.

**Language.** Scribe transcribes each recording in its `SWHISPER_LANGUAGE`, from the recording's `.swhisper` sidecar or the global settings, exactly as local Whisper does. Set the language before the run: Scribe does not detect it, and with the wrong language it translates, so English audio with `sv` comes back as Swedish text. Scribe supports `sv`, `en`, `no`, `fi`, `da`, `is`, `se`, `fr`, `de`, `es`, `it`, `ru`, `uk`, `pt` and `nl`; the run preview shows each recording's Scribe language and refuses any other value before uploading.

**Verbatim or standard.** Verbatim (the default) uses the verbatim model where Scribe has one: KB-Whisper strict for Swedish, and verbatim variants for English and Norwegian. Standard uses the ordinary model. Other languages use their ordinary model in both modes. On the command line, use `--mode standard`, or `--language` to override the recording's language.

More detail is in [SCRIBE_WEBUI.md](SCRIBE_WEBUI.md) and [SCRIBE_PROTOTYPE.md](SCRIBE_PROTOTYPE.md).

### Quick start

1. Copy the example configuration file:
   ```bash
   cp .swhisper.env.example .swhisper.env
   ```

2. Edit `.swhisper.env` and set at minimum:
   ```bash
   # Required: Authentication tokens
   HUGGING_FACE_TOKEN="your_huggingface_token_here"

   # Required: Audio directory
   SWHISPER_AUDIO_DIR="/path/to/your/audio/files"
   ```

3. Optionally customize other settings (all have sensible defaults):
   ```bash
   # Example: Adjust speaker detection
   SWHISPER_MIN_SPEAKERS=2
   SWHISPER_MAX_SPEAKERS=5

   # Example: Change processing device
   SWHISPER_DEVICE="cuda"  # default: "mps"

   # Example: Use premium diarization service (the web UI offers it as a
   # diarization model choice instead of always using it)
   SWHISPER_USE_PRECISION=true
   PYANNOTEAI_API_KEY="your_pyannote_api_key"
   ```

### Using NVIDIA Nemotron 3 Diarization

swhisper can use [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) in place of pyannote. The model runs locally and tracks at most eight speakers, so use it only for recordings with eight speakers or fewer. On an Apple Silicon Mac it diarizes an hour of audio in about 30 seconds on the CPU.

The model needs Python 3.12 and a version of `transformers` that is not yet released, so it runs in its own conda environment:

```bash
conda create -n nemo python=3.12
conda run -n nemo pip install torch numpy librosa git+https://github.com/huggingface/transformers
```

Then select the model in `.swhisper.env`:

```bash
SWHISPER_PIPELINE_MODEL=nvidia/Nemotron-3-Diarization
```

swhisper finds the `nemo` environment next to the `swhisper` environment. If your environment lives elsewhere, set `SWHISPER_NEMOTRON_PYTHON` to its `python` binary. `ffmpeg` must be on the `PATH`.

To use Nemotron for one run only, leave `.swhisper.env` as it is:

- In the web UI, choose "Diarization: Nemotron 3" in the diarization selector.
- On the command line, set the variable for that command, for example `SWHISPER_PIPELINE_MODEL=nvidia/Nemotron-3-Diarization python diarize4.py`.
- For Scribe, use `python transcribe_scribe.py recording.m4a --diarization-model nemotron`.

Diarization checks which model produced the current output. Choosing a different model runs diarization and preserves the previous timestamped transcripts. Older outputs are identified from the saved JSON metadata; if their model cannot be verified, diarization runs again. To repeat a diarization with the same model, tick "Re-diarize" under the recording in the web UI, or set `SWHISPER_RUN_REDIARIZE` to the recording's basename for the command (`SWHISPER_RUN_REPROCESS=diarization` does this for every recording). To repeat the transcription as well, tick "Re-transcribe", or set `SWHISPER_RUN_RETRANSCRIBE` (`SWHISPER_RUN_REPROCESS=true` for every recording). The two variables take newline separated basenames, like `SWHISPER_ONLY_FILES`. These apply to one run only and override per-recording reprocess settings. For Scribe, "Re-transcribe (upload again)" matches `--retranscribe`: it uploads the recording again and moves the previous SUNET transcript to `superseded/` in the recording's Scribe folder. Upload consent is asked only before a recording's first upload. Without it, a Scribe run reuses the saved SUNET transcript and needs no credentials. A Scribe run also reuses the local diarization of that transcript when it came from the selected model; `--rediarize` runs it again.

Do not use `SWHISPER_MAX_SPEAKERS` with Nemotron; leave it unset. The current implementation only limits the output after inference. It keeps the most active speakers and forces the others' speech onto a retained speaker, even when that speaker's probability is very low. This can produce misleading labels and provides no meaningful constraint on the model's speaker detection. Nemotron ignores `SWHISPER_MIN_SPEAKERS`, and the pyannote tuning settings (`SWHISPER_VBX_*`, `SWHISPER_CLUSTERING_*`, …) do not apply to it.

### Configuration priority

Settings are loaded in this order (highest to lowest priority):
1. Command-line environment variables (e.g., `SWHISPER_DEVICE=cuda python transcribe.py`)
2. Variables in `.swhisper.env` file
3. Default values in the code

### Key configuration variables

| Variable | Purpose | Default |
| --- | --- | --- |
| **Authentication** | | |
| `HUGGING_FACE_TOKEN` | Hugging Face token for pyannote models | *required* |
| `PYANNOTEAI_API_KEY` | PyannoteAI API key for precision-2 model | *optional* |
| **Paths** | | |
| `SWHISPER_AUDIO_DIR` | Source audio files directory | `"audio"` |
| `SWHISPER_TEMP_DIR` | Temporary files and checkpoints | *none* |
| `SWHISPER_OUTPUT_DIR` | Diarization output directory (relative to audio_dir) | `"transcripts"` |
| **Transcription** | | |
| `SWHISPER_MODEL` | Whisper model from HuggingFace | `"KBLab/kb-whisper-large"` |
| `SWHISPER_DEVICE` | Processing device | `"mps"` |
| `SWHISPER_LANGUAGE` | Transcription language, for local Whisper and Scribe | `"sv"` |
| `SWHISPER_FILE_LIMIT` | Max files to process | *unlimited* |
| **Diarization** | | |
| `SWHISPER_MIN_SPEAKERS` | Minimum speakers | `1` |
| `SWHISPER_MAX_SPEAKERS` | Maximum speakers | *none* |
| `SWHISPER_PIPELINE_MODEL` | Diarization pipeline (`nvidia/Nemotron-3-Diarization` selects Nemotron) | `"pyannote/speaker-diarization-3.1"` |
| `SWHISPER_USE_PRECISION` | Use commercial precision-2 service | `false` |
| `SWHISPER_NEMOTRON_PYTHON` | Python binary of the environment that runs Nemotron | the `nemo` conda env |
| **Processing** | | |
| `SWHISPER_BACKFILL_ENABLED` | Re-transcribe empty turns | `true` |
| `SWHISPER_BACKFILL_TRY_HARDER` | Use the configured decoding retries for backfill | `false` |
| `SWHISPER_INCLUDE_SILENCE` | Include silence markers | `true` |
| `SWHISPER_PRESERVE_MARKERS` | Preserve disfluency markers | `true` |

See `.swhisper.env.example` for the complete list of all 100+ available settings with descriptions and defaults.

Backfill makes one decoding attempt at temperature 0 by default. Results that
fail the configured confidence or repetition checks become duration-scaled `[-]`
markers. Set `SWHISPER_BACKFILL_TRY_HARDER=true` globally or in a file's `.swhisper`
sidecar to use the configured temperature retries and accept Whisper's final
fallback if no attempt passes, as in earlier versions. This affects backfill
only, and changing it invalidates the backfill cache for that mode.

### Running the transcription and diarization pipeline§

To run the full transcription and diarization pipeline, simply execute: 

```bash
python transcribe.py
```
After successful transcription of all audio files in the specified audio directory, diarization will be automatically initiated.

The transcription can be stopped at any time (e.g., using Ctrl+C), and when you restart the script, it will resume from where it left off.

Once an audio file has been successfully transcribed, the temporary `.json` file with the raw transcript will be moved to the `whisper-json-output` directory inside the audio directory. This file is used as input for the diarization step. 

To run only the diarization step on previously transcribed files, use:

```bash
python diarize.py
```

Sucessfully processed files will be skipped. If you want to reprocess a specific file, simply delete its ok-file (e.g., `audio1.ok`) from the `transcritps` directory.


## Acknowledgments

This project builds upon the excellent work of:

- **[KBLab](https://huggingface.co/KBLab)** for the [kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) Swedish Whisper model (which is based on **OpenAI**'s original [Whisper](https://github.com/openai/whisper) speech recognition model). 
- **[whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)** by Jérôme Louradour for precise word-level timestamps
- **[pyannote-audio](https://github.com/pyannote/pyannote-audio)** by Hervé Bredin for speaker diarization capabilities 

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
Please also cite the other works used in the pipeline as inidicated in their respective repositories.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
