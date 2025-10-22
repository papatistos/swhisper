#!/usr/bin/env python3
"""
swhisper - Swedish Whisper Transcription and Diarization

Copyright (c) 2025 papatistos
Licensed under the MIT License
https://github.com/papatistos/swhisper

Speaker Diarization Script

This script reads JSON output from transcription, performs speaker diarization
on corresponding audio files, aligns the results, and exports transcripts in
multiple formats using a modular architecture.
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

import warnings

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.set_audio_backend has been deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.get_audio_backend has been deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Module 'speechbrain.pretrained' was deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torchaudio.backend.common.AudioMetaData` has been moved",
    category=UserWarning,
)

if __name__ == "__main__":
    print("Preparing for diarization. This may take a while. Please be patient...", flush=True)

# Import our refactored modules
from diarize import (
    DiarizationConfig, DEFAULT_DIARIZATION_CONFIG,
    DiarizationPipeline, SpeakerAligner,
    DiarizationAnalyzer, SegmentAnalyzer, BoundaryAnalyzer, StatsExporter,
    VTTFormatter, RTTMFormatter, RTFFormatter, TXTFormatter
)
from diarize.utils import logger_manager


# Ensure progress messages appear immediately even when stdout is buffered
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# Get script version from last modified timestamp
SCRIPT_VERSION = datetime.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y-%m-%d %H:%M:%S')


def create_completion_marker(config: DiarizationConfig, base_filename: str, 
                           log_timestamp: str, output_files: dict) -> None:
    """Create completion marker file with processing details."""
    completion_marker = os.path.join(config.final_output_dir, f"{base_filename}.ok")
    audio_file_location = f"{config.audio_subdir}/{base_filename}.wav" if config.audio_subdir else f"{base_filename}.wav"
    
    completion_content = f"""# Diarization completed successfully
# Start: {log_timestamp}
# End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Audio dir: {config.audio_dir}
# Audio file:      └── {audio_file_location}
# Raw transcript:  └── {os.path.join(config.json_input_dir, base_filename + '.json')}
#
# Output files generated:
#                  └── {config.output_dir}/
#                       ├── vtt/{output_files.get('vtt', 'N/A')}
#                       ├── rttm/{output_files.get('rttm', 'N/A')}
#                       ├── rttm/{output_files.get('rttm_detailed', 'N/A')}
#                       ├── rtf/{output_files.get('rtf', 'N/A')}
#                       ├── txt/{output_files.get('txt', 'N/A')}
#                       ├── stats/{output_files.get('stats', 'N/A')}
#                       ├── logs/{output_files.get('log', 'N/A')}
#                       └── logs/{output_files.get('silence_gap_log', 'N/A')}
"""

    with open(completion_marker, 'w', encoding='utf-8') as f:
        f.write(completion_content)


def process_file(config: DiarizationConfig, json_file: str, processed_files: int, total_files: int) -> bool:
    """
    Process a single audio file through the diarization pipeline.
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    json_path = os.path.join(config.final_json_input_dir, json_file)
    
    # Reconstruct the original audio file path from the JSON filename
    audio_basename = f"{os.path.splitext(json_file)[0]}.wav"
    audiofile_path = os.path.join(config.final_audio_dir, audio_basename)
    base_filename = os.path.splitext(audio_basename)[0]
    
    # Check for completion marker file (.ok)
    completion_marker = os.path.join(config.final_output_dir, f"{base_filename}.ok")
    
    if os.path.exists(completion_marker) and not config.force_reprocess:
        print("-" * 80)
        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{starttime} - Skipping file {processed_files} of {total_files}: {audio_basename}')
        print(f"  -> Already processed (marker file exists: {base_filename}.ok)")
        print(f"  -> Set FORCE_REPROCESS = True to reprocess existing files (or delete .ok files).")
        return True

    # Create per-file logger using context manager
    log_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    subdirs = config.get_output_subdirs()
    log_file_path = os.path.join(subdirs['logs'], f"{base_filename}_{log_timestamp}.log")
    gap_log_filename = None
    gap_log_path = None
    if config.include_silence_markers and getattr(config, 'log_silence_gaps', False):
        gap_log_filename = f"{base_filename}_{log_timestamp}_silence_gaps.log"
        gap_log_path = os.path.join(subdirs['logs'], gap_log_filename)
    
    with logger_manager.safe_logger(log_file_path) as logger:
        try:
            print("-" * 80)
            print(f"🎯 DIARIZATION LOG FOR: {audio_basename}")
            print(f"Timestamp: {log_timestamp}")
            print(f"Processing file {processed_files} of {total_files}")
            print(f"Audio file: {audio_basename}")
            print(f"JSON input: ./{config.json_input_dir}/{json_file}")
            print(f"Log file: ./logs/{base_filename}_{log_timestamp}.log")
            print("-" * 80)

            if not os.path.exists(audiofile_path):
                print(f"  -> ERROR: Corresponding audio file not found at: {audiofile_path}")
                return False
    
            # 1. Load the transcription data from the JSON file
            starttime = datetime.now()
            print(f"{starttime.strftime('%Y-%m-%d %H:%M:%S')} - Step 1: Loading transcription from {json_file}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                whisper_result = json.load(f)
            endtime = datetime.now()
            duration = endtime - starttime
            print(f"{endtime.strftime('%Y-%m-%d %H:%M:%S')} - Transcription loaded. (Duration: {duration})")

            # 2. Diarize with Pyannote 
            starttime = datetime.now()
            print(f"{starttime.strftime('%Y-%m-%d %H:%M:%S')} - Step 2: Performing speaker diarization... (this can take a while)")
            print("  -> Creating diarization pipeline object...")
            sys.stdout.flush()  # Force immediate output
            
            diarization_pipeline = DiarizationPipeline(config)
            print("  -> Loading AI models and running diarization on audio file...")
            sys.stdout.flush()  # Force immediate output
            diarization_result = diarization_pipeline.diarize(audiofile_path)
            DiarizationAnalyzer.analyze_diarization_result(diarization_result)
            
            # Save raw RTTM output (for reference)
            sanitized_filename = base_filename.replace(" ", "_")
            raw_rttm_path = os.path.join(config.final_json_input_dir, f"{sanitized_filename}_{log_timestamp}_speakers_only.rttm")
            
            # Create RTTM content directly (no temp file needed)
            rttm_lines = []
            for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                start_time = turn.start
                duration_time = turn.duration
                rttm_line = f"SPEAKER {sanitized_filename} 1 {start_time:.3f} {duration_time:.3f} <NA> <NA> {speaker_label} 1.000 <NA>"
                rttm_lines.append(rttm_line)
            
            with open(raw_rttm_path, "w") as rttm:
                rttm.write('\n'.join(rttm_lines))
            
            endtime = datetime.now()
            duration = endtime - starttime
            print(f"{endtime.strftime('%Y-%m-%d %H:%M:%S')} - Diarization complete. (Duration: {duration})")

            # 3. Align Whisper segments with diarization results
            starttime = datetime.now()
            print(f"{starttime.strftime('%Y-%m-%d %H:%M:%S')} - Step 3: Aligning transcription with speaker segments...")
            sys.stdout.flush()  # Force immediate output
            speaker_aligner = SpeakerAligner(config)
            alignment_result = speaker_aligner.align_segments_with_speakers(
                whisper_result,
                diarization_result,
                gap_log_path=gap_log_path if gap_log_path else None
            )
            
            segments = alignment_result['segments']
            word_stats = alignment_result['word_stats']
            segment_stats = alignment_result['segment_stats']
            
            endtime = datetime.now()
            duration = endtime - starttime
            print(f"{endtime.strftime('%Y-%m-%d %H:%M:%S')} - Alignment complete. (Duration: {duration})")
            if gap_log_path and os.path.exists(gap_log_path):
                print(f"Silence gap log saved to logs/{gap_log_filename}")

            # 4. Analysis
            speaker_stats = SegmentAnalyzer.analyze_final_segments(segments)
            boundary_stats = BoundaryAnalyzer.analyze_boundary_issues(segments, diarization_result, return_data=True)

            # 5. Create settings dictionary for stats
            settings = {
                "script_version": f"diarize.py (modified: {SCRIPT_VERSION})",
                "audio_file": audio_basename,
                "json_input_file": json_file,
                "device": config.device,
                "diarization": {
                    "model": config.pipeline_model,
                    "min_speakers": config.min_speakers,
                    "max_speakers": config.max_speakers,
                    "pipeline_config": config.get_pipeline_config()
                },
                "word_level_processing": {
                    "smoothing_enabled": config.smoothing_enabled,
                    "min_speaker_words": config.min_speaker_words,
                    "preserve_markers": config.preserve_markers,
                    "markers_preserved": config.preserved_markers
                },
                "output_formats": ["vtt", "rttm", "rttm_detailed", "rtf", "txt", "stats_json"] 
            }

            # 6. Save all output formats
            print()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Step 4: Generating output files...")
            sys.stdout.flush()  # Force immediate output
            output_files = {}
            
            # VTT format
            print(f'Saving transcript for "{audio_basename}" in VTT format...')
            vtt_filename = f"{base_filename}_{log_timestamp}.vtt"
            vtt_output_path = os.path.join(subdirs['vtt'], vtt_filename)
            VTTFormatter().format(segments, vtt_output_path)
            print(f"VTT file saved to vtt/{vtt_filename}")
            output_files['vtt'] = vtt_filename
            
            # RTTM format (standard)
            print(f'Saving RTTM file...')
            rttm_filename = f"{base_filename}_{log_timestamp}.rttm"
            rttm_output_path = os.path.join(subdirs['rttm'], rttm_filename)
            rttm_entries = RTTMFormatter().format(segments, rttm_output_path, audio_basename=audio_basename)
            print(f"RTTM file saved to rttm/{rttm_filename} ({rttm_entries} speaker turns)")
            output_files['rttm'] = rttm_filename
            
            # RTTM format (detailed word-level)
            print(f'Saving detailed word-level RTTM file...')
            detailed_rttm_filename = f"{base_filename}_{log_timestamp}_detailed.rttm"
            detailed_rttm_output_path = os.path.join(subdirs['rttm'], detailed_rttm_filename)
            detailed_rttm_entries = RTTMFormatter().format_detailed(segments, detailed_rttm_output_path, audio_basename=audio_basename)
            print(f"Detailed RTTM file saved to rttm/{detailed_rttm_filename} ({detailed_rttm_entries} word entries)")
            output_files['rttm_detailed'] = detailed_rttm_filename
            
            # RTF format
            print(f'Saving speaker-grouped transcript in RTF format...')
            rtf_filename = f"{base_filename}_{log_timestamp}.rtf" 
            rtf_output_path = os.path.join(subdirs['rtf'], rtf_filename)
            RTFFormatter().format(segments, rtf_output_path, config=config, transcript_id=log_timestamp)
            print(f"RTF file saved to rtf/{rtf_filename}")
            output_files['rtf'] = rtf_filename
            
            # TXT format
            print(f'Saving speaker-grouped transcript in TXT format...')
            txt_filename = f"{base_filename}_{log_timestamp}.txt"
            txt_output_path = os.path.join(subdirs['txt'], txt_filename)
            TXTFormatter().format(segments, txt_output_path, config=config, transcript_id=log_timestamp)
            print(f"TXT file saved to txt/{txt_filename}")
            output_files['txt'] = txt_filename

            # Stats JSON
            print(f'Saving analysis statistics...')
            stats_filename = f"{base_filename}_{log_timestamp}_analysis.json"  
            stats_output_path = os.path.join(subdirs['stats'], stats_filename)
            StatsExporter.save_analysis_stats(
                segments, diarization_result, word_stats, segment_stats, 
                speaker_stats, boundary_stats, settings, stats_output_path
            )
            print(f"Analysis stats saved to stats/{stats_filename}")
            output_files['stats'] = stats_filename
            output_files['log'] = f"{base_filename}_{log_timestamp}.log"
            if gap_log_path and os.path.exists(gap_log_path):
                output_files['silence_gap_log'] = gap_log_filename

            # Create completion marker
            create_completion_marker(config, base_filename, log_timestamp, output_files)

            print(f"✅ Successfully completed processing of {audio_basename}")
            print(f"📄 Completion marker created: {base_filename}.ok")
            print("-" * 80)
            
            return True

        except Exception as e:
            print(f"{datetime.now().strftime('%Y%m%d-%H%M%S')} - An error occurred while processing {json_file}: {e}")
            import traceback
            print(traceback.format_exc())
            print(f"❌ Processing failed :-(")
            return False
            
        finally:               
            # Clean up per-file resources
            if 'whisper_result' in locals():
                del whisper_result
            if 'diarization_result' in locals():
                del diarization_result
            from diarize.utils import DeviceManager
            DeviceManager.clear_device_memory()


def main():
    """Main processing function."""
    config = DEFAULT_DIARIZATION_CONFIG
    
    # Create the final output directory if it doesn't exist
    os.makedirs(config.final_output_dir, exist_ok=True)
    
    # Create organized subfolders for different file types
    subdirs = config.get_output_subdirs()
    for folder_path in subdirs.values():
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"🎯 DIARIZATION STARTED")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Input directory: {config.audio_dir}")
    if config.audio_subdir:
        print(f"   └── 🎵 Audio files: {config.audio_subdir}/ (from temp workspace)")
        print(f"   └── 📄 Raw transcript: {config.json_input_dir}/")
    else:
        print(f"   └── 🎵 Audio files: ./ (standalone mode)")
        print(f"   └── 📄 Raw transcript: {config.json_input_dir}/")
    print(f"   └── 📁 Final output: {config.output_dir}/")
    print(f"       ├── vtt/ (VTT transcripts)")
    print(f"       ├── rttm/ (RTTM speaker timing files)")
    print(f"       ├── rtf/ (Rich text documents)")
    print(f"       ├── txt/ (Plain text transcripts)")
    print(f"       ├── stats/ (Analysis JSON files)")
    print(f"       └── logs/ (Processing logs)")
    print("=" * 80)
    
    try:
        # Find JSON files to process
        try:
            json_files = [f for f in os.listdir(config.final_json_input_dir) if f.endswith('.json')]
            if not json_files:
                print(f"No .json files found in '{config.final_json_input_dir}'. Please run transcription first.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"Error: The directory '{config.final_json_input_dir}' was not found.")
            sys.exit(1)

        total_files = len(json_files)
        processed_files = 0
        successful_files = 0

        # Process each JSON file
        for json_file in json_files:
            processed_files += 1
            success = process_file(config, json_file, processed_files, total_files)
            if success:
                successful_files += 1

        print("-" * 80)
        print(f"🎯 DIARIZATION COMPLETED")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {successful_files}")
        print(f"Failed: {total_files - successful_files}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
