#!/usr/bin/env python3
"""
swhisper - Swedish Whisper Transcription and Diarization

Copyright (c) 2025 papatistos
Licensed under the MIT License
https://github.com/papatistos/swhisper

Audio Transcription with Speech-Aware Chunking

This script automatically chunks long audio files at natural speech boundaries
to prevent memory issues while maintaining transcription accuracy.
"""

import shutil
import csv
import os
import sys
import json
import signal as sys_signal
import atexit
import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcribe import (
    TranscriptionConfig, WhisperSettings, DEFAULT_CONFIG, DEFAULT_WHISPER_SETTINGS,
    ResourceManager, resource_manager, 
    AudioProcessor, SpeechAnalyzer, ChunkBoundaryFinder,
    FileManager, TranscriptionPipeline, WorkspaceManager, CheckpointManager
)


class TranscriptionApp:
    """Main transcription application."""
    
    def __init__(self, config: TranscriptionConfig = None, whisper_settings: WhisperSettings = None):
        self.config = config or DEFAULT_CONFIG
        self.whisper_settings = whisper_settings or DEFAULT_WHISPER_SETTINGS
        
        # Initialize components
        self.workspace_manager = WorkspaceManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.file_manager = FileManager(self.config)
        self.audio_processor = AudioProcessor(self.config)
        self.speech_analyzer = SpeechAnalyzer(self.config)
        self.boundary_finder = ChunkBoundaryFinder(self.config)
        self.transcription_pipeline = TranscriptionPipeline(
            self.config, self.whisper_settings, self.checkpoint_manager
        )
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def run(self):
        """Run the main transcription process."""
        print("🚀 Starting Audio Transcription Pipeline")
        print("=" * 60)

        try:
            # Verify model availability before starting
            self._verify_model_availability()

            # Setup workspace
            workspace_dir = self.workspace_manager.setup_temp_workspace()
            source_audio_dir = self.workspace_manager.get_source_audio_dir()
            output_dir = self.workspace_manager.get_output_dir()
            # Use a single, deterministic checkpoint root for both save and load.
            checkpoint_root = getattr(self.workspace_manager, "temp_dir", None) or output_dir
            self.checkpoint_manager.set_output_dir(checkpoint_root)
            print(f"💾 Checkpoints will be read/written in: {checkpoint_root}")

            # Find and convert audio files
            source_output_dir = os.path.join(self.workspace_manager.original_audio_dir, self.config.json_dir)
            wav_files = self.file_manager.find_and_convert_audio_files(audio_dir, output_dir, source_output_dir)
            
            if not wav_files:
                print("✅ No new audio files to process.")
                return
            
            # Process each audio file
            for wav_file in wav_files:
                self._process_single_file(wav_file, audio_dir, output_dir)
            
            print("\n" + "=" * 60)
            print("✅ All transcriptions completed successfully!")

            # Automatically clean up temp workspace
            self.workspace_manager.cleanup_temp_workspace()

            # Run diarization pipeline now that transcripts exist
            self._run_diarization_pipeline()
            
        except KeyboardInterrupt:
            print("\n⚠️  Process interrupted by user.")
            resource_manager.cleanup_resources()
        except Exception as e:
            print(f"\n❌ Error during transcription: {e}")
            resource_manager.cleanup_resources()
            raise
        finally:
            self._cleanup()

    def _process_single_file(self, wav_file: str, audio_dir: str, output_dir: str):
        """Process a single audio file."""
        audiofile_path = os.path.join(audio_dir, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")
        
        print(f"\n🎵 Processing: {wav_file}")
        print("-" * 40)
        
        # Check for existing checkpoint
        checkpoint_data, checkpoint_path = self.checkpoint_manager.load_checkpoint(audiofile_path)
        
        # Fallback: probe multiple plausible roots so read/write match exactly
        if not checkpoint_data:
            current_root = getattr(self.checkpoint_manager, "output_dir", None)
            candidates = []
            # 1) current configured root (already checked, but keep for order)
            if current_root:
                candidates.append(current_root)
            # 2) temp output directory
            try:
                temp_output_dir = self.workspace_manager.get_output_dir()
                if temp_output_dir:
                    candidates.append(temp_output_dir)
            except Exception:
                pass
            # 3) temp workspace root
            temp_root = getattr(self.workspace_manager, "temp_dir", None)
            if temp_root:
                candidates.append(temp_root)

            # Try each unique candidate (preserve order)
            seen = set()
            for root in [c for c in candidates if not (c in seen or seen.add(c))]:
                if not os.path.isdir(root):
                    continue
                if current_root != root:
                    self.checkpoint_manager.set_output_dir(root)
                alt_data, alt_path = self.checkpoint_manager.load_checkpoint(audiofile_path)
                if alt_data:
                    checkpoint_data, checkpoint_path = alt_data, alt_path
                    print(f"📍 Using checkpoint from: {root}")
                    # Keep manager pointing at the root where we found the checkpoint,
                    # so subsequent saves go to the same place.
                    break
            else:
                # Nothing found; restore original root if we changed it
                if current_root and getattr(self.checkpoint_manager, "output_dir", None) != current_root:
                    self.checkpoint_manager.set_output_dir(current_root)
                if current_root:
                    print(f"🔎 No checkpoints found in searched locations. Current root: {current_root}")
        
        if checkpoint_data:
            # Resume from checkpoint - extract saved state
            start_chunk = checkpoint_data['current_chunk']
            chunk_results = checkpoint_data['chunk_results']
            boundaries = checkpoint_data['boundaries']

            print(f"📋 Resuming from chunk {start_chunk+1}/{len(boundaries)-1}...")

            # Use unified processing path with resume parameters
            result = self.transcription_pipeline.process_audio_file(
                audiofile_path, boundaries, output_path, start_chunk, chunk_results
            )
        else:
            # Start fresh processing
            result = self._process_from_start(audiofile_path, output_path)
        
        # Save final result
        self._save_transcription_result(result, output_path)

        # Persist VAD segments if available
        vad_output_path = os.path.join(output_dir, f"{base_name}_vad.tsv")
        self._save_vad_segments(result, vad_output_path)
        
        # Clean up checkpoint on successful completion
        if checkpoint_path:
            self.checkpoint_manager.cleanup_checkpoint(checkpoint_path)
    
    def _process_from_start(self, audiofile_path: str, output_path: str) -> dict:
        """Process audio file from the beginning."""
        # Get audio duration
        duration, sample_rate = self.audio_processor.get_audio_duration(audiofile_path)
        print(f"📊 Audio duration: {duration/60:.1f} minutes ({sample_rate}Hz)")
        
        # Determine if chunking is needed
        if duration <= self.config.target_chunk_duration:
            print("🔄 File is short enough - processing as single chunk")
            return self._process_single_chunk(audiofile_path)
        
        print("🔄 File requires chunking - analyzing speech patterns...")
        
        # Analyze speech patterns for optimal chunking
        times, speech_mask = self.speech_analyzer.detect_speech_segments_from_preview(audiofile_path)
        
        # Find optimal chunk boundaries
        boundaries = self.boundary_finder.find_optimal_boundaries(times, speech_mask)
        
        print(f"📏 Found {len(boundaries)-1} optimal chunks:")
        for i in range(len(boundaries)-1):
            chunk_duration = boundaries[i+1] - boundaries[i]
            print(f"   Chunk {i+1}: {boundaries[i]:.1f}s - {boundaries[i+1]:.1f}s ({chunk_duration:.1f}s)")
        
        # Process chunks
        return self.transcription_pipeline.process_audio_file(audiofile_path, boundaries, output_path)
    
    def _process_single_chunk(self, audiofile_path: str) -> dict:
        """Process a single audio file without chunking."""
        # Implementation for single chunk processing
        # This would use the existing whisper transcription directly
        import whisper_timestamped as whisper
        
        with self._safe_whisper_model() as model:
            print("🔄 Loading audio...")
            audio_data, _ = self.audio_processor.load_audio_chunk(
                audiofile_path, 0, float('inf')
            )
            
            print("🔄 Transcribing...")
            result = whisper.transcribe(model, audio_data, **self.whisper_settings.to_dict())
            
            return result
    
    def _save_transcription_result(self, result: dict, output_path: str):
        """Save transcription result to JSON file, archiving any existing file."""
        
        # Archive existing file if it exists
        if os.path.exists(output_path):
            try:
                # Create archive directory
                output_dir = os.path.dirname(output_path)
                archive_dir = os.path.join(output_dir, "archive")
                os.makedirs(archive_dir, exist_ok=True)
                
                # Create timestamped filename for the old file
                modification_time = os.path.getmtime(output_path)
                timestamp = datetime.fromtimestamp(modification_time).strftime('%Y%m%d-%H%M%S')
                base_name = os.path.splitext(os.path.basename(output_path))[0]
                archive_path = os.path.join(archive_dir, f"{base_name}_{timestamp}.json")
                
                # Move the old file
                print(f"  -> Archiving existing file to: {os.path.relpath(archive_path, output_dir)}")
                os.rename(output_path, archive_path)
            except Exception as e:
                print(f"  -> ⚠️ Warning: Could not archive existing file: {e}")

        print(f"💾 Saving transcription to: {os.path.basename(output_path)}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Print summary
        if 'segments' in result:
            total_segments = len(result['segments'])
            total_text_length = len(result.get('text', ''))
            print(f"✅ Saved {total_segments} segments, {total_text_length} characters")

    def _save_vad_segments(self, result: dict, vad_path: str) -> bool:
        """Write VAD boundaries to TSV for downstream analysis."""
        segments = result.get('speech_activity')
        if segments is None:
            print("⚠️ No VAD data found; skipping TSV export")
            return False

        os.makedirs(os.path.dirname(vad_path), exist_ok=True)

        try:
            with open(vad_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(["segment_id", "start", "end", "duration"])
                for idx, span in enumerate(segments, start=1):
                    start = float(span.get('start', 0.0))
                    end = float(span.get('end', start))
                    duration = max(0.0, end - start)
                    writer.writerow([idx, f"{start:.3f}", f"{end:.3f}", f"{duration:.3f}"])

            print(f"📄 Saved VAD segments to: {os.path.basename(vad_path)}")
            return True
        except Exception as exc:
            print(f"⚠️ Failed to write VAD TSV: {exc}")
            return False
    
    @contextmanager
    def _safe_whisper_model(self):
        """Context manager for safe whisper model handling."""
        model = None
        try:
            print("🔄 Loading Whisper model...")
            import whisper_timestamped as whisper
            model = whisper.load_model(self.config.model_str, device=self.config.device)
            resource_manager.current_whisper_model = model
            yield model
        finally:
            if model is not None:
                del model
                resource_manager.current_whisper_model = None
            resource_manager.clear_device_memory()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_names = {
                sys_signal.SIGINT: "SIGINT (Ctrl+C)",
                sys_signal.SIGTERM: "SIGTERM",
                sys_signal.SIGQUIT: "SIGQUIT"
            }
            
            signal_name = signal_names.get(signum, f"Signal {signum}")
            print(f"\n⚠️  Received {signal_name}. Initiating graceful shutdown...")
            
            resource_manager.cleanup_resources()
            
            print("🔄 Transcription interrupted. Progress saved to checkpoints.")
            print("💡 Re-run the script to resume from where it left off.")
            
            exit_code = 130 if signum == sys_signal.SIGINT else 1
            sys.exit(exit_code)
        
        # Register signal handlers
        sys_signal.signal(sys_signal.SIGINT, signal_handler)
        sys_signal.signal(sys_signal.SIGTERM, signal_handler)
        if hasattr(sys_signal, 'SIGQUIT'):
            sys_signal.signal(sys_signal.SIGQUIT, signal_handler)
        
        # Register exit handler
        atexit.register(self._safe_exit_cleanup)
    
    def _safe_exit_cleanup(self):
        """Safe cleanup function for exit handler."""
        try:
            resource_manager.cleanup_resources()
            root = getattr(self.checkpoint_manager, 'output_dir', None)
            if root and os.path.isdir(root):
                print(f"💾 Checkpoints preserved in: {root}")
                print("💡 Re-run the script to resume from where it left off")
        except Exception as e:
            print(f"Warning: Error during exit cleanup: {e}")
    
    def _verify_model_availability(self):
        """Verify that the Whisper model is available, downloading if necessary."""
        from transcribe.transcription import is_model_cached

        model_name = self.config.model_str
        print(f"🔍 Checking model availability: {model_name}")

        if is_model_cached(model_name):
            print(f"✅ Model found in local cache - all processing will be offline")
            return

        print(f"⚠️  Model not found in cache")
        print(f"📥 Will download model from HuggingFace on first use")
        print(f"💡 This is a one-time download (model will be cached for future use)")

        # Optional: Pre-download the model in the main process
        # This ensures download happens once, not in every subprocess
        try:
            import whisper_timestamped as whisper
            print(f"📦 Pre-loading model to cache...")
            model = whisper.load_model(model_name, device=self.config.device)
            del model
            resource_manager.clear_device_memory()
            print(f"✅ Model cached successfully - subsequent chunks will load from cache")
        except Exception as e:
            print(f"⚠️  Could not pre-load model: {e}")
            print(f"   Will attempt to download in subprocesses instead")

    def _cleanup(self):
        """Final cleanup."""
        resource_manager.clear_device_memory()

    def _run_diarization_pipeline(self) -> None:
        """Invoke diarization pipeline after successful transcription."""
        print("\n" + "=" * 60)
        print("🎯 Launching diarization pipeline...")
        try:
            from diarize import main as diarize_main
        except ImportError as exc:
            print(f"❌ Unable to start diarization: {exc}")
            return

        try:
            diarize_main()
        except SystemExit as exit_info:
            # Propagate successful exit, but continue cleanup on non-zero codes
            if exit_info.code not in (None, 0):
                print(f"❌ Diarization exited with status {exit_info.code}")
        except Exception as exc:
            print(f"❌ Unexpected error during diarization: {exc}")


def main():
    """Main entry point."""
    # Required for macOS multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # It's okay if the start method was already set elsewhere
        pass
    # Create and run the application
    app = TranscriptionApp()
    app.run()


if __name__ == '__main__':
    main()
