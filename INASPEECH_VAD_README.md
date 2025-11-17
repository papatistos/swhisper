# inaSpeechSegmenter VAD Integration

This document describes the integration of inaSpeechSegmenter as an alternative VAD (Voice Activity Detection) engine for swhisper.

## Overview

inaSpeechSegmenter is a CNN-based audio segmentation toolkit that was **ranked #1** against 6 open-source VAD systems (including Silero, Pyannote, LIUM_SpkDiarization, Rvad, and Speechbrain) on a French TV and radio benchmark.

### Key Features

- **Superior VAD Performance**: Outperforms Silero and other VAD systems on broadcast content
- **Speech/Music/Noise Classification**: Distinguishes between speech, music, and noise
- **Optimized for Broadcast Content**: Designed for TV and radio with complex audio environments
- **Gender Detection**: Optional male/female classification during VAD stage
- **Proven Track Record**: Used by French Audiovisual Regulation Authority since 2020
- **Simple Integration**: Direct import in swhisper environment (no subprocess needed)

### Architecture

The integration uses **direct import** from the main swhisper environment:

1. **VAD Stage**: inaSpeechSegmenter replaces Silero for speech detection
2. **VAD Segments Passed to Whisper**: Detected speech segments are passed to whisper-timestamped's `vad` parameter
3. **Transcription**: Whisper (KBLab model) transcribes using the INA VAD segments
4. **Diarization**: pyannote.audio 4 performs post-transcription speaker diarization (optional)

## Installation

### Install inaSpeechSegmenter in swhisper Environment

```bash
# Activate your swhisper conda environment
conda activate swhisper

# Install inaSpeechSegmenter from GitHub (has NumPy 2.x compatibility)
pip install git+https://github.com/ina-foss/inaSpeechSegmenter.git
```

**Important**: Install from GitHub, not PyPI! The GitHub version (0.8.0+) has NumPy 2.x compatibility fixes from November 8, 2024. The PyPI version (0.7.6) has compatibility issues.

### Verify Installation

```bash
# Test the installation
python -c "from inaSpeechSegmenter import Segmenter; print('✅ inaSpeechSegmenter installed')"
```

## Usage

### Basic Usage

Use `transcribe4_ina.py` instead of `transcribe4.py`:

```bash
# Activate your main swhisper environment
conda activate swhisper

# Run transcription with inaSpeechSegmenter VAD
python transcribe4_ina.py
```

This will:
1. Extract speech segments using inaSpeechSegmenter (direct import in swhisper environment)
2. Pass VAD segments to whisper-timestamped via the `vad=` parameter
3. Transcribe with Whisper using only the detected speech regions
4. Generate all output formats (JSON, VTT, RTTM, etc.)

### Configuration

The inaSpeechSegmenter VAD can be configured by modifying the `create_ina_vad_direct()` call in `transcribe4_ina.py`:

```python
self.ina_vad = create_ina_vad_direct(
    detect_gender=False,          # Set to True for male/female classification during VAD
    vad_engine='smn'              # 'smn' (speech/music/noise) or 'sm' (speech/music)
)
```

#### VAD Engine Options

- **'smn'** (default): Speech/Music/Noise classification
  - More recent engine
  - Better at handling noise
  - Recommended for most use cases

- **'sm'**: Speech/Music classification
  - Original engine from ICASSP 2017 and MIREX 2018
  - Noise is classified as either speech or music
  - Slightly faster

#### Gender Detection

Set `detect_gender=True` to distinguish male/female speech during VAD:
- Provides earlier gender classification
- May help with transcription quality
- Note: swhisper also does gender detection in the diarization stage (if enabled)

## Comparison: inaSpeechSegmenter vs. Silero

### Advantages of inaSpeechSegmenter

1. **Better Accuracy**: Ranked #1 in French TV/radio benchmark
2. **Music/Noise Handling**: Explicitly distinguishes speech from music and noise
3. **Broadcast Optimization**: Designed for complex audio environments
4. **Proven in Production**: Used by French regulatory authority since 2020

### Advantages of Silero

1. **Simpler**: No additional dependencies
2. **Faster**: Slightly faster processing
3. **Integrated**: Built into whisper-timestamped

### When to Use inaSpeechSegmenter

Use inaSpeechSegmenter when:
- Audio contains significant music or background noise
- You're transcribing broadcast content (TV, radio, podcasts)
- You need the highest VAD accuracy
- You're willing to install an additional dependency

Use Silero (original pipeline) when:
- Audio is clean speech with minimal background
- You want the simplest setup
- Processing speed is critical

## Technical Details

### How It Works

The integration uses **direct import** from inaSpeechSegmenter:

1. **INA VAD Analysis**: Analyze full audio file with inaSpeechSegmenter
   ```python
   segmenter = Segmenter(vad_engine='smn', detect_gender=False)
   segments = segmenter(audio_path)  # Returns [(label, start, end), ...]
   ```

2. **Speech Filtering**: Only speech/male/female segments are kept
   ```python
   speech_segments = [(start, end) 
                      for label, start, end in segments 
                      if label in {"speech", "male", "female"}]
   ```

3. **Pass to Whisper**: Segments are set in `whisper_settings.vad` parameter
   ```python
   self.whisper_settings.vad = speech_segments  # [(0.0, 5.2), (8.1, 15.3), ...]
   ```

4. **Whisper Transcription**: whisper-timestamped uses VAD segments directly
   ```python
   whisper.transcribe(model, audio, vad=speech_segments, ...)
   ```

5. **Diarization** (optional): pyannote assigns speakers to transcribed words

### VAD Format

inaSpeechSegmenter provides segments as:

```python
[
    ('male', 0.0, 5.2),      # Male speech 0-5.2s
    ('music', 5.2, 8.1),     # Music 5.2-8.1s
    ('female', 8.1, 15.3),   # Female speech 8.1-15.3s
    ('noise', 15.3, 16.0),   # Noise 15.3-16.0s
]
```

Converted to Whisper VAD format (list of tuples):

```python
[
    (0.0, 5.2),    # Speech segment 1
    (8.1, 15.3),   # Speech segment 2
]
```

This format is passed directly to whisper-timestamped's `vad` parameter.

### Why Direct Import Works Now

- **November 8, 2024 fix**: inaSpeechSegmenter GitHub version was patched for NumPy 2.x compatibility
- **Same NumPy version**: Both swhisper and inaSpeechSegmenter now work with NumPy 2.2.6+
- **No conflicts**: All dependencies are compatible in a single environment
- **Simpler**: No subprocess overhead, faster startup, easier debugging

## Troubleshooting

### Import Error: inaSpeechSegmenter not found

Make sure you installed from GitHub:

```bash
# Activate swhisper environment
conda activate swhisper

# Install from GitHub (not PyPI)
pip install git+https://github.com/ina-foss/inaSpeechSegmenter.git
```

### NumPy Compatibility Issues

The PyPI version (0.7.6) has NumPy 2.x issues. Use GitHub version (0.8.0+):

```bash
# Check version
python -c "import inaSpeechSegmenter; print(inaSpeechSegmenter.__version__)"

# Should show 0.8.0 or higher
```

### FFmpeg Not Found

```bash
# Install ffmpeg in swhisper environment
conda install -c conda-forge ffmpeg
```

### TensorFlow Issues

If you encounter TensorFlow errors:

```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow
```

## Performance Considerations

### Processing Time

inaSpeechSegmenter adds overhead compared to Silero:
- **Silero**: VAD is fast, integrated with Whisper (negligible time)
- **inaSpeechSegmenter**: Separate VAD pass before transcription

For a 1-hour audio file:
- Silero VAD: ~negligible (integrated)
- inaSpeechSegmenter: +2-5 minutes (VAD analysis)

The improved accuracy often justifies the additional time.

### Startup Time

- **First use**: ~2-3 seconds to load TensorFlow models
- **Subsequent calls**: Models stay in memory (faster)
- **Direct import**: No subprocess overhead

### GPU Acceleration

inaSpeechSegmenter benefits from GPU acceleration:
- CPU: Slower but works on any machine
- GPU: Significantly faster (TensorFlow will auto-detect)
- Automatically uses available GPU if present

## References

### Papers

**inaSpeechSegmenter:**
- Doukhan et al. (2018). "An Open-Source Speaker Gender Detection Framework for Monitoring Gender Equality." ICASSP 2018.
- Doukhan et al. (2024). "InaGVAD: A Challenging French TV and Radio Corpus." LREC-COLING 2024.

**Benchmark Results:**
- Ranked #1 against Silero, Pyannote, LIUM, Rvad, Speechbrain, and other VAD systems
- Dataset: French TV and radio broadcast content

### Links

- [GitHub Repository](https://github.com/ina-foss/inaSpeechSegmenter)
- [PyPI Package](https://pypi.org/project/inaSpeechSegmenter/)
- [API Tutorial (Jupyter Notebook)](https://github.com/ina-foss/inaSpeechSegmenter/blob/master/tutorials/API_Tutorial.ipynb)

## License

inaSpeechSegmenter is released under the MIT License.

## Implementation Files

- `transcribe4_ina.py`: Main entry point for INA VAD transcription
- `transcribe/ina_vad_direct.py`: Direct import wrapper for inaSpeechSegmenter
- `INASPEECH_COMPATIBILITY_ISSUES.md`: NumPy compatibility notes
- `INASPEECH_INTEGRATION_SUMMARY.md`: Integration implementation details
