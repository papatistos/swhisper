# swhisper - Swedish Whisper Transcription and Diarization

Speech transcription and diarization pipeline using the (strict) Swedish Whisper model [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) with [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) and [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Performance and limitations
Until proven otherwise (and please do!), I believe that this is currently the best available open-source solution for Swedish speech recognition and diarization. 

The excellent transcription quality is due to the KBLab Whisper model, which has been fine-tuned on a large Swedish dataset. The **limitations** are in the diarization step (though I have not tried the latest community-1 version of pyannote). Depending on your audio-files and settings, you may have to do some manual work to assign the right speakers, but it's still much better than starting from scratch. Try to tweak the parameters in `diarize/config.py` to see what works for your audio. I recommend specifying 1-2 more speakers than you have in the audio, it seems to work better and my post-processing sometimes manages to remove the extra speakers.

- Although the pipeline is fully functional, it is still work in progress. PRs and suggestions are welcome!

- The scripts have been developed and tested on a MacBook Pro M4 with 24 GB RAM. It works with the Apple GPU but I'm not sure if they will run out of the box on other systems, especially with CUDA support.

- The pipeline prioritizes quality over speed (as this is what counts in academic research), so, if you're in a hurry, this is probably not for you. 

- Word-level timestamps are provided in the `.json` output files in the `whisper-json-output` directory. They come from the `whisper-timestamped` model and are not affected by diarization.

- there is a basic stats analysis tool in `diarize/` that can help compare diarization results across different parameter settings. See `diarize/STATS_ANALYSIS_README.md` for details.


### Transcript formatting options

- The duration of silences between words can be included in the transcript (using CA notation, e.g. (.3) for a .3 second silence). I have not yet investigate the accuracy of these durations. They are based on the word-level provided by `whisper-timestamped` and I have a feeling that they might be underestimated...

- There is also an option to include disfluence markers (as [*]) for sounds that could not be transcribed By default, these markers are preserved in the diarized transcript, but there is a setting to remove them. The discontinuity markers are longer (more asterisks) the longer the unidentified sound is (one * per .1 s). It looks like the current settings don't properly distinguish between unidentified speech and background noise. Tweaking of the whisper-timestamped settings (in `transcribe/config.py`) might help.

- The output preamble for transcript files can be customized in the `DiarizationConfig` class. This allows you to include specific notes or instructions for users reviewing the transcripts.

- By default, silence of 1 second or longer are surrounded by blank lines in the transcript (even when there is no speaker change) to improve readability. This threshold can be adjusted in `diarize/config.py` through the `silence_newline_threshold` parameter (set to `0`to disable). This setting only concerns rtf and txt output files.


### To-dos
- [x] migrate to pyannote-audio 4.0 (and the `community-1` model)
  - to use pyannote 4 with `community-1` use `diarize4.py` (instead of `diarize.py`) or `transcribe4.py` for the entire pipeline
  - there is also an option to use the premium `precision-2` diarization service from pyannote (see `diarize/config.py` for details)
- [x] do something with speaker segments that didn't get any words
  - these are now reprocessed by (re-)transcribing them and backfilling the words in the diarized transcript. While this does find the odd word here and there, it also sometimes introduces artifacts (possibly due to the short duration of these segments). If no word has been transcried, we add a discontinuity marker `[ * ]` instead, because we trust pyannote that there is some voice activity there. These markers have spaces around them to distinguish them from the regular disfluency markers (`[*]`) added by whisper-timestamped.
- [ ] fix caching of empty turns (the current code that should persist those transcriptions doesn't seem to work)
- [ ] add (more) speaker stats
- [ ] improve documentation (let me know what is particularly unclear)
- [ ] combine settings in one config file that is read by both `config.py`s
- [ ] add command line options?

## Configuration and usage

- relevant paths are specified in a `.swhisper.env` file or through environment variables (see below)
- huggingface token for pyannote models can also be set in the `.swhisper.env` file or through the `HUGGING_FACE_TOKEN` environment variable
- model and processing settings can be adjusted in `config.py`
- you can run transcription and diarization through `transcribe.py` (`transcribe.py` will call `config.py` after transcription is completed)
- `diarize.py` can also be called on its own. It will look for previously transcribed files



### Local path configuration

The scripts are started by simply calling them without any arguments. All settings are read from the configuration files and relevant paths are managed through environment variables loaded from an optional `.swhisper.env` file. Copy `.swhisper.env.example` to `.swhisper.env` and set the paths that apply to your machine:

```bash
cp .swhisper.env.example .swhisper.env
open .swhisper.env  # Edit the file with your preferred editor
```

Supported variables:

| Variable | Purpose |
| --- | --- |
| `SWHISPER_AUDIO_DIR` | Default folder containing source audio files for transcription |
| `SWHISPER_TEMP_DIR` | Custom directory for temporary workspaces and checkpoints |
| `SWHISPER_DIARIZE_AUDIO_DIR` | Default diarization audio directory (falls back to `SWHISPER_AUDIO_DIR` if omitted) |
| `HUGGING_FACE_TOKEN` | Hugging Face token for accessing pyannote pretrained model |


### Environment variables

You can also set the same values through regular environment variables instead of the
`.swhisper.env` file. For example:

```bash
export SWHISPER_AUDIO_DIR=/data/audio
export SWHISPER_TEMP_DIR=/tmp/swhisper
```

The path settings module (`path_settings.py`) reads both the environment and the
optional `.swhisper.env` file, so choose whichever workflow fits your deployment.

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
