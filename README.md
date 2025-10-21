# swhisper - Swedish Whisper Transcription and Diarization

Speech transcription and diarization pipeline using the (strict) Swedish Whisper model [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) with [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) and [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Performance and limitations
Until proven otherwise (and please do!), I believe that this is currently the best available open-source solution for Swedish speech recognition and diarization. 

The excellent transcription quality is due to the KBLab Whisper model, which has been fine-tuned on a large Swedish dataset. The **limitations** are in the diarization step (though I have not tried the latest community-1 version of pyannote). Depending on your audio-files and settings, you may have to do some manual work to assign the right speakers, but it's still much better than starting from scratch. Try to tweak the parameters in `diarize/config.py` to see what works for your audio. I recommend specifying 1-2 more speakers than you have in the audio, it seems to work better and my post-processing sometimes manages to remove the extra speakers.

- Please note that this is very much work in progress. PRs and suggestions are welcome!

- The scripts have been developed and tested on a MacBook Pro M4 with 24 GB RAM. It works with the Apple GPU but I'm not sure if they will run out of the box on other systems, especially with CUDA support.

- The pipeline prioritizes quality over speed (as this is what counts in academic research), so, if you're in a hurry, this is probably not for you. 

- Word-level timestamps are provided in the `.json` output files in the `whisper-json-output` directory.

### To-dos
- [ ] migrate to pyannote-audio 4.0 (and the community-1 model)
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

Paths are managed through environment variables loaded from an optional `.swhisper.env`
file. Copy `.swhisper.env.example` to `.swhisper.env` and set the paths that apply to
your machine:

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
