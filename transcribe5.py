#!/usr/bin/env python3
"""
swhisper - Diarization-First Transcription Pipeline

This entry point implements a diarization-first workflow where speaker
diarization is performed BEFORE transcription. The detected speaker segments
are then used to guide the transcription process, providing:

1. More accurate speaker attribution (diarization sees full context)
2. Superior VAD from pyannote instead of Silero
3. Natural chunk boundaries at speaker changes
4. Memory-safe processing for long files

Architecture:
- Inherits from transcribe4.py to use pyannote.audio 4 framework
- Uses DiarizationFirstPipeline for the reversed workflow
- Maintains compatibility with existing checkpoint and output systems
"""

import importlib.util
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure local packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import transcribe4 module to inherit pyannote 4 settings
_BASE_TRANSCRIBE4_PATH = Path(__file__).with_name("transcribe4.py")

_spec = importlib.util.spec_from_file_location(
    "swhisper_transcribe4",
    _BASE_TRANSCRIBE4_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load transcribe4 module for inheritance.")
_transcribe4_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transcribe4_module)

# Import configuration and utilities from transcribe4
TranscriptionConfig = _transcribe4_module.TranscriptionConfig
WhisperSettings = _transcribe4_module.WhisperSettings
DEFAULT_CONFIG = _transcribe4_module.DEFAULT_CONFIG
DEFAULT_WHISPER_SETTINGS = _transcribe4_module.DEFAULT_WHISPER_SETTINGS
resource_manager = _transcribe4_module.resource_manager

# Import diarization-first pipeline
from transcribe.diarization_first import DiarizationFirstPipeline

# Import diarization config for pyannote 4
from diarize4 import run as diarize4_run
from diarize import DiarizationConfig


class TranscriptionAppV5(_transcribe4_module.TranscriptionAppV4):
    """
    Diarization-first transcription pipeline using pyannote.audio 4.
    
    This class inherits from TranscriptionAppV4 to maintain pyannote 4
    compatibility, but overrides the processing workflow to run
    diarization before transcription.
    """

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        whisper_settings: Optional[WhisperSettings] = None,
        diarization_config: Optional[DiarizationConfig] = None,
    ) -> None:
        # Initialize parent (gets workspace manager, checkpoint manager, etc.)
        super().__init__(config, whisper_settings)
        
        # Store diarization config
        self.diarization_config = diarization_config or DiarizationConfig()
        
        # Create diarization-first pipeline instead of standard pipeline
        self.diarization_first_pipeline = DiarizationFirstPipeline(
            self.config,
            self.whisper_settings,
            self.checkpoint_manager,
            self.diarization_config  # Pass diarization config
        )

    def _process_single_file(
        self, 
        wav_file: str, 
        audio_dir: str, 
        output_dir: str,
        file_index: int, 
        total_files: int,
        original_label: Optional[str] = None
    ):
        """
        Process a single audio file using diarization-first approach.
        
        This overrides the parent method to use the new pipeline.
        """
        audiofile_path = os.path.join(audio_dir, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        file_label = original_label or wav_file
        
        print(f"\n🎵 Processing ({file_index}/{total_files}): {file_label}")
        print("-" * 40)
        
        # TODO: Add checkpoint support for diarization-first pipeline
        # For now, process from start each time
        
        # Process using diarization-first pipeline
        result = self.diarization_first_pipeline.process_audio_file(
            audiofile_path,
            output_path,
            diarization_config=self.diarization_config,
            file_label=file_label,
            file_index=file_index,
            total_files=total_files
        )
        
        # Save final result
        self._save_transcription_result(result, output_path)

        # Persist VAD segments if available (speaker segments in this case)
        vad_output_path = os.path.join(output_dir, f"{base_name}_vad.tsv")
        self._save_speaker_segments(result, vad_output_path)

    def _save_speaker_segments(self, result: dict, vad_path: str) -> bool:
        """
        Write speaker segments to TSV for downstream analysis.
        
        This is similar to _save_vad_segments but uses speaker_segments
        from the diarization-first pipeline.
        """
        segments = result.get('speaker_segments')
        if segments is None:
            print("⚠️ No speaker segment data found; skipping TSV export")
            return False

        os.makedirs(os.path.dirname(vad_path), exist_ok=True)

        try:
            import csv
            with open(vad_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(["segment_id", "start", "end", "duration", "speaker"])
                for idx, seg in enumerate(segments, start=1):
                    start = float(seg.get('start', 0.0))
                    end = float(seg.get('end', start))
                    duration = max(0.0, end - start)
                    speaker = seg.get('speaker', 'UNKNOWN')
                    writer.writerow([idx, f"{start:.3f}", f"{end:.3f}", f"{duration:.3f}", speaker])

            print(f"📄 Saved speaker segments to: {os.path.basename(vad_path)}")
            return True
        except Exception as exc:
            print(f"⚠️ Failed to write speaker segments TSV: {exc}")
            return False

    def _run_diarization_pipeline(self) -> None:
        """
        Override to prevent double diarization.
        
        Since we run diarization BEFORE transcription in this pipeline,
        we don't need to run it again after transcription.
        """
        print("\n" + "=" * 60)
        print("✅ Diarization already completed (diarization-first mode)")
        print("   Skipping post-transcription diarization step")


def main() -> None:
    """Main entry point for the diarization-first transcription workflow."""
    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)

    # Print banner
    print("=" * 60)
    print("🎯 swhisper - Diarization-First Pipeline (pyannote 4)")
    print("=" * 60)
    print("This pipeline runs speaker diarization BEFORE transcription")
    print("to provide more accurate speaker attribution.")
    print("=" * 60)

    # Create application with diarization-first pipeline
    app = TranscriptionAppV5()
    app.run()


if __name__ == "__main__":
    main()
