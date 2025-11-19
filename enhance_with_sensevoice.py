#!/usr/bin/env python3
"""
Enhance diarized JSON transcripts with SenseVoice emotion and event detection.

This script processes existing diarization JSON files and adds SenseVoice metadata
to backfilled segments and silence markers. It runs independently after the main
diarization pipeline, avoiding memory pressure from loading multiple models.

Usage:
    python enhance_with_sensevoice.py <json_file> [options]
    
Examples:
    # Process single file
    python enhance_with_sensevoice.py output.json
    
    # Process with custom settings
    python enhance_with_sensevoice.py output.json --model iic/SenseVoiceSmall --device cuda:0
    
    # Process all JSON files in a directory
    python enhance_with_sensevoice.py transcripts/json/*.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from diarize.sensevoice_provider import SenseVoiceProvider


class SenseVoiceEnhancer:
    """Enhance diarized transcripts with SenseVoice emotion and event detection."""
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: Optional[str] = None,
        language: str = "auto",
        verbose: bool = True,
        filter_language_tags: bool = False,
        process_all_segments: bool = False
    ):
        """
        Initialize the enhancer.
        
        Args:
            model_name: SenseVoice model identifier
            device: Device to run on (cuda:0, cpu, mps, etc.)
            language: Language hint (auto, zh, en, yue, ja, ko)
            verbose: Print progress messages
            filter_language_tags: If True, only save language when it's <|nospeech|> (ignore unreliable language classifications)
            process_all_segments: If True, process all segments (not just backfill and silence)
        """
        self.model_name = model_name
        self.device = device or "mps"
        self.language = language
        self.verbose = verbose
        self.filter_language_tags = filter_language_tags
        self.process_all_segments = process_all_segments
        self.provider = None
        self.sample_rate = 16000
        
    def _ensure_provider_loaded(self) -> None:
        """Lazy load the SenseVoice provider."""
        if self.provider is None:
            if self.verbose:
                print(f"Loading SenseVoice model: {self.model_name}")
                print(f"  Device: {self.device}")
                print(f"  Language: {self.language}")
            
            self.provider = SenseVoiceProvider(
                model_name=self.model_name,
                device=self.device,
                use_vad=False  # No VAD needed for short snippets
            )
            
            # Ensure the underlying model is loaded now so we surface errors early
            try:
                self.provider._ensure_model_loaded()  # type: ignore[attr-defined]
            except Exception:
                # If loading fails, remove provider reference so we can retry later if desired
                self.provider = None
                raise
            
            if self.verbose:
                print("✓ SenseVoice model loaded successfully\n")
    
    def _load_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file and return as numpy array."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
    
    def _extract_audio_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
        overlap: float = 0.5
    ) -> np.ndarray:
        """Extract audio segment with padding."""
        padded_start = max(0.0, start - overlap)
        padded_end = end + overlap
        
        start_sample = int(round(padded_start * self.sample_rate))
        end_sample = int(round(padded_end * self.sample_rate))
        end_sample = min(end_sample, len(audio))
        
        if end_sample <= start_sample:
            end_sample = min(start_sample + int(0.1 * self.sample_rate), len(audio))
        
        return audio[start_sample:end_sample]
    
    def _transcribe_with_sensevoice(
        self,
        audio_segment: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Transcribe audio segment with SenseVoice."""
        self._ensure_provider_loaded()
        
        try:
            result = self.provider.transcribe(
                audio_segment,
                language=self.language
            )
            return result
        except Exception as e:
            if self.verbose:
                print(f"  Warning: SenseVoice transcription failed: {e}")
            return None
    
    def _should_process_segment(self, segment: Dict[str, Any]) -> bool:
        """Determine if segment should be processed by SenseVoice."""
        # Already has SenseVoice data
        if 'sensevoice_text' in segment:
            return False
        
        # If processing all segments, include this one
        if self.process_all_segments:
            return True
        
        # Check if segment is marked as backfill
        if segment.get('is_backfill') or segment.get('contains_backfill'):
            return True
        
        # Check if segment contains silence markers
        words = segment.get('words', [])
        for word in words:
            if word.get('is_silence_marker'):
                return True
        
        return False
    
    def enhance_json_file(
        self,
        json_path: str,
        audio_path: Optional[str] = None,
        output_path: Optional[str] = None,
        backup: bool = True
    ) -> bool:
        """
        Enhance a diarized JSON file with SenseVoice data.
        
        Args:
            json_path: Path to diarized JSON file
            audio_path: Path to audio file (auto-detected if None)
            output_path: Output path (overwrites input if None)
            backup: Create backup of original file
            
        Returns:
            True if successful, False otherwise
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            print(f"Error: JSON file not found: {json_path}")
            return False
        
        if self.verbose:
            print(f"Processing: {json_path.name}")
        
        # Load JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False
        
        segments = data.get('segments', [])
        metadata = data.get('metadata', {})
        
        # Determine audio file path
        if audio_path is None:
            audio_filename = metadata.get('audio_file')
            if not audio_filename:
                print("Error: Cannot determine audio file path")
                return False
            
            # Try to find audio file relative to JSON location
            audio_path = json_path.parent.parent.parent / audio_filename
            if not audio_path.exists():
                print(f"Error: Audio file not found: {audio_path}")
                return False
        
        if self.verbose:
            print(f"  Audio file: {audio_path}")
        
        # Load audio
        audio = self._load_audio_file(str(audio_path))
        if audio is None:
            return False
        
        # Find segments to process
        segments_to_process = []
        for i, segment in enumerate(segments):
            if self._should_process_segment(segment):
                segments_to_process.append((i, segment))
        
        if not segments_to_process:
            if self.verbose:
                print("  No segments need SenseVoice processing")
            return True
        
        if self.verbose:
            print(f"  Found {len(segments_to_process)} segments to process")
        
        # Process segments
        processed_count = 0
        skipped_count = 0
        
        for seg_idx, segment in segments_to_process:
            start = segment.get('start', 0.0)
            end = segment.get('end', start + 0.1)
            speaker = segment.get('speaker', 'UNKNOWN')
            
            segment_log = None
            if self.verbose:
                seg_text = segment.get('text', '') or ''
                if isinstance(seg_text, str):
                    seg_text = seg_text.strip().replace('\n', ' ')
                else:
                    seg_text = str(seg_text)
                segment_log = f"  [{seg_idx+1}/{len(segments)}] {speaker}: {start:.2f}s - {end:.2f}s | text: '{seg_text}'"
            
            # Extract audio segment
            audio_segment = self._extract_audio_segment(audio, start, end)
            
            if len(audio_segment) < int(0.05 * self.sample_rate):
                if self.verbose and segment_log:
                    print(f"{segment_log}\n      ✗ skipped (too short)")
                skipped_count += 1
                continue
            
            # Transcribe with SenseVoice
            sensevoice_result = self._transcribe_with_sensevoice(audio_segment)
            
            if sensevoice_result:
                # Check language for filtering
                language = sensevoice_result.get('language')
                is_nospeech = language == '<|nospeech|>'
                save_language = not self.filter_language_tags or is_nospeech

                # Temporarily remove words so SenseVoice fields appear before them in JSON
                words_list = None
                if 'words' in segment:
                    words_list = segment.pop('words')
                
                # Add to segment
                # Only save text/language if not filtering or if it's nospeech
                if save_language:
                    segment['sensevoice_text'] = sensevoice_result.get('text', '')
                    segment['sensevoice_language'] = language
                
                # Always save raw text, emotion, and event (these are useful regardless)
                segment['sensevoice_raw_text'] = sensevoice_result.get('raw_text', '')
                segment['sensevoice_emotion'] = sensevoice_result.get('emotion')
                segment['sensevoice_event'] = sensevoice_result.get('event')

                if words_list is not None:
                    segment['words'] = words_list
                
                if self.verbose and segment_log:
                    emotion = sensevoice_result.get('emotion', '')
                    event = sensevoice_result.get('event', '')
                    tags = []
                    if emotion:
                        tags.append(f"Emotion: {emotion}")
                    if event:
                        tags.append(f"Event: {event}")
                    tag_str = ', '.join(tags) if tags else 'no tags'
                    print(f"{segment_log}\n      ✓ {tag_str}" )
                
                processed_count += 1
            else:
                if self.verbose and segment_log:
                    print(f"{segment_log}\n      ✗ failed")
                skipped_count += 1
        
        if self.verbose:
            print(f"\n  Processed: {processed_count} segments")
            if skipped_count > 0:
                print(f"  Skipped: {skipped_count} segments")
        
        # Save enhanced JSON
        if output_path is None:
            output_path = json_path
            
            # Create backup if requested
            if backup and processed_count > 0:
                backup_path = json_path.with_suffix('.json.backup')
                if self.verbose:
                    print(f"  Creating backup: {backup_path.name}")
                import shutil
                shutil.copy2(json_path, backup_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                print(f"  ✓ Saved enhanced JSON: {Path(output_path).name}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.provider:
            # SenseVoiceProvider cleanup is handled internally
            self.provider = None


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhance diarized JSON transcripts with SenseVoice emotion and event detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  %(prog)s output.json
  
  # Process with custom audio path
  %(prog)s output.json --audio audio/file.wav
  
  # Use GPU
  %(prog)s output.json --device cuda:0
  
  # Process multiple files
  %(prog)s transcripts/json/*.json
        """
    )
    
    parser.add_argument(
        'json_files',
        nargs='+',
        help='Path(s) to diarized JSON file(s)'
    )
    
    parser.add_argument(
        '--audio',
        help='Path to audio file (auto-detected from JSON metadata if not provided)'
    )
    
    parser.add_argument(
        '--model',
        default='iic/SenseVoiceSmall',
        help='SenseVoice model to use (default: iic/SenseVoiceSmall)'
    )
    
    parser.add_argument(
        '--device',
        default='mps',
        help='Device to run on: cpu, cuda:0, mps, etc. (default: cpu)'
    )
    
    parser.add_argument(
        '--language',
        default='auto',
        help='Language hint: auto, zh, en, yue, ja, ko (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        help='Output path (overwrites input if not specified)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original files'
    )
    
    parser.add_argument(
        '--dont-filter-language',
        action='store_true',
        help='Save all language tags (by default, only <|nospeech|> is saved to avoid unreliable classifications)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all segments (by default, only backfill and silence segments are processed)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = SenseVoiceEnhancer(
        model_name=args.model,
        device=args.device,
        language=args.language,
        verbose=not args.quiet,
        filter_language_tags=not args.dont_filter_language,  # Default is True (filtering enabled)
        process_all_segments=args.all
    )
    
    # Process files
    success_count = 0
    fail_count = 0
    
    print(f"SenseVoice Enhancement Tool")
    print(f"{'=' * 60}\n")
    
    for json_file in args.json_files:
        try:
            success = enhancer.enhance_json_file(
                json_path=json_file,
                audio_path=args.audio,
                output_path=args.output,
                backup=not args.no_backup
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            fail_count += 1
    
    # Cleanup
    enhancer.cleanup()
    
    # Summary
    print(f"{'=' * 60}")
    print(f"Summary:")
    print(f"  Successfully processed: {success_count}")
    if fail_count > 0:
        print(f"  Failed: {fail_count}")
    print()
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
