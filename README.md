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


### Transcript formatting options

- The duration of silences between words can be included in the transcript (using CA notation, e.g. (.3) for a .3 second silence). I have not yet investigate the accuracy of these durations. They are based on the word-level provided by `whisper-timestamped` and I have a feeling that they might be underestimated...

- There is also an option to include disfluence markers (as `[*]`) for sounds that could not be transcribed. By default, these markers are preserved in the diarized transcript, but can be disabled via `SWHISPER_PRESERVE_MARKERS=false`. The discontinuity markers are longer (more asterisks) the longer the unidentified sound is (one `*` per 0.1 s). It looks like the current settings don't properly distinguish between unidentified speech and background noise. Tweaking of the whisper-timestamped settings (in `.swhisper.env`) might help.

- The output preamble for transcript files can be customized via `SWHISPER_OUTPUT_PREAMBLE` in `.swhisper.env`. Use `\n` for line breaks. This allows you to include specific notes or instructions for users reviewing the transcripts.

- By default, silence of 1 second or longer are surrounded by blank lines in the transcript (even when there is no speaker change) to improve readability. This threshold can be adjusted via `SWHISPER_SILENCE_LINEBREAK` (set to `0` to disable). This setting only concerns rtf and txt output files.


### To-dos
- [x] migrate to pyannote-audio 4.0 (and the `community-1` model)
  - to use pyannote 4 with `community-1` use `diarize4.py` (instead of `diarize.py`) or `transcribe4.py` for the entire pipeline
  - there is also an option to use the premium `precision-2` diarization service from pyannote (see `.swhisper.env` for details)
- [x] do something with speaker segments that didn't get any words
  - these are now reprocessed by (re-)transcribing them and backfilling the words in the diarized transcript. While this does find the odd word here and there, it also sometimes introduces artifacts (possibly due to the short duration of these segments). If no word has been transcried, we add a discontinuity marker `[ * ]` instead, because we trust pyannote that there is some voice activity there. These markers have spaces around them to distinguish them from the regular disfluency markers (`[*]`) added by whisper-timestamped.
- [x] combine settings in one config file that is read by both `config.py`s
- [x] fix caching of empty turns (the current code that should persist those transcriptions doesn't seem to work)
- [ ] add (more) speaker stats
- [ ] add speaker stats to the preamble
- [ ] process files via csv input list (with custom settings per file)
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

   # Example: Use premium diarization service
   SWHISPER_USE_PRECISION=true
   PYANNOTEAI_API_KEY="your_pyannote_api_key"
   ```

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
| `SWHISPER_LANGUAGE` | Transcription language | `"sv"` |
| `SWHISPER_FILE_LIMIT` | Max files to process | `3` |
| **Diarization** | | |
| `SWHISPER_MIN_SPEAKERS` | Minimum speakers | `2` |
| `SWHISPER_MAX_SPEAKERS` | Maximum speakers | `2` |
| `SWHISPER_PIPELINE_MODEL` | Diarization pipeline | `"pyannote/speaker-diarization-3.1"` |
| `SWHISPER_USE_PRECISION` | Use commercial precision-2 service | `false` |
| **Processing** | | |
| `SWHISPER_BACKFILL_ENABLED` | Re-transcribe empty turns | `true` |
| `SWHISPER_INCLUDE_SILENCE` | Include silence markers | `true` |
| `SWHISPER_PRESERVE_MARKERS` | Preserve disfluency markers | `true` |

See `.swhisper.env.example` for the complete list of all 100+ available settings with descriptions and defaults.

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

- **[KBLab](https://huggingface.co/KBLab)** for the [kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) Swedish Whisper model. 
- **[whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)** by Jérôme Louradour for precise word-level timestamps
- **[pyannote-audio](https://github.com/pyannote/pyannote-audio)** by Hervé Bredin for speaker diarization capabilities
- **OpenAI** for the original [Whisper](https://github.com/openai/whisper) speech recognition model

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
