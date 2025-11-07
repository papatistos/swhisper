"""File conversion and management utilities."""

import os
import shutil
import subprocess
import tempfile
import soundfile as sf
from typing import List, Tuple, Optional

# Local tuning knob for developers while iterating on staged processing.
# Toggle to True to restore eager format validation and detailed counts.
ENABLE_EARLY_FORMAT_SUMMARY = False

from .config import TranscriptionConfig


class AudioConverter:
    """Handle audio file format conversion."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.supported_formats = ['.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg']
    
    def convert_to_wav(self, input_path: str, output_path: str, target_sr: int = 16000) -> bool:
        """Convert various audio/video formats to 16kHz mono WAV using ffmpeg."""
        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            raise Exception("ffmpeg not found. Please install ffmpeg: brew install ffmpeg")
        
        same_path = os.path.abspath(input_path) == os.path.abspath(output_path)
        temp_output_path = output_path
        temp_created = False

        if same_path:
            # ffmpeg cannot overwrite in-place; use a temporary file then replace.
            fd, temp_output_path = tempfile.mkstemp(suffix=".tmp.wav", dir=os.path.dirname(output_path) or None)
            os.close(fd)
            os.unlink(temp_output_path)
            temp_created = True

        try:
            # Use ffmpeg to convert to 16kHz mono WAV
            cmd = [
                'ffmpeg',
                '-i', input_path,           # Input file
                '-ar', str(target_sr),      # Sample rate: 16000 Hz
                '-ac', '1',                 # Channels: 1 (mono)
                '-c:a', 'pcm_s16le',       # Codec: 16-bit PCM
                '-y',                       # Overwrite output file
                '-hide_banner',             # Suppress banner for cleaner logs
                '-loglevel', 'error',       # Only show errors
                temp_output_path            # Output file (may be temporary)
            ]
            
            # Run ffmpeg with a timeout to prevent hanging
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=300,  # 5-minute timeout
                encoding='utf-8',
                errors='replace'
            )
            if temp_created:
                os.replace(temp_output_path, output_path)
            return True
            
        except subprocess.TimeoutExpired as e:
            print(f"❌ FFmpeg timed out converting '{os.path.basename(input_path)}'.")
            print("   This often happens if the file is very large or not fully synced from cloud storage.")
            print("   SOLUTION: In Finder, right-click the folder and select 'Always Keep on This Device', then retry.")
            print(f"   FFmpeg stderr: {e.stderr or 'No output'}")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed to convert '{os.path.basename(input_path)}'. This could be due to a corrupt file or path issues.")
            print(f"   FFmpeg stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during conversion: {e}")
            return False
        finally:
            if temp_created and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except OSError:
                    pass
    
    def is_wav_ready(self, wav_path: str) -> bool:
        """Check if WAV file is ready for processing (16kHz mono)."""
        try:
            with sf.SoundFile(wav_path) as f:
                return f.samplerate == 16000 and f.channels == 1
        except Exception:
            return False

    def probe_wav(self, wav_path: str) -> Tuple[Optional[int], Optional[int]]:
        """Inspect WAV metadata, returning sample rate and channel count if available."""
        try:
            with sf.SoundFile(wav_path) as f:
                return f.samplerate, f.channels
        except Exception:
            return None, None


class FileManager:
    """Manage file discovery and organization."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.converter = AudioConverter(config)
    
    def find_audio_files(self, audio_dir: str) -> List[str]:
        """Find all supported audio/video files in directory."""
        try:
            files_in_dir = os.listdir(audio_dir)
            
            audio_files = []
            for f in files_in_dir:
                if any(f.lower().endswith(ext) for ext in self.converter.supported_formats):
                    # Check if it has at least 4 alphabetic characters
                    if sum(1 for char in f if char.isalpha()) >= 4:
                        audio_files.append(f)
            
            return audio_files
            
        except FileNotFoundError:
            print(f"Error: The directory '{audio_dir}' was not found.")
            return []
    
    def get_audio_files_to_process(self, audio_dir: str, source_output_dir: str) -> List[str]:
        """Identify audio files that still need processing, respecting file limits."""
        audio_files = self.find_audio_files(audio_dir)
        print(f"Found {len(audio_files)} audio/video files")

        if not audio_files:
            return []

        if not self.config.force_reprocess and os.path.exists(source_output_dir):
            existing_json_basenames = {
                os.path.splitext(f)[0]
                for f in os.listdir(source_output_dir)
                if f.endswith('.json')
            }
            unprocessed_files = []
            for audio_file in audio_files:
                base_name = os.path.splitext(audio_file)[0]
                if base_name in existing_json_basenames:
                    print(f"  ⏩ {audio_file} (already processed, found in source)")
                else:
                    unprocessed_files.append(audio_file)
        else:
            unprocessed_files = audio_files

        if self.config.file_limit is not None:
            files_to_consider = unprocessed_files[:self.config.file_limit]
        else:
            files_to_consider = unprocessed_files

        if not files_to_consider:
            print("❌ No new audio files to process.")
            return []

        if ENABLE_EARLY_FORMAT_SUMMARY:
            ready_wavs: List[str] = []
            wavs_to_fix: List[str] = []
            non_wavs: List[str] = []

            for filename in files_to_consider:
                if filename.lower().endswith('.wav'):
                    input_path = os.path.join(audio_dir, filename)
                    if self.converter.is_wav_ready(input_path):
                        ready_wavs.append(filename)
                    else:
                        wavs_to_fix.append(filename)
                else:
                    non_wavs.append(filename)

            print("\n📊 File Status Summary:")
            print(f"   Ready WAV files: {len(ready_wavs)}")
            print(f"   WAV files needing format conversion: {len(wavs_to_fix)}")
            print(f"   Non-WAV files needing conversion: {len(non_wavs)}")
        else:
            wav_candidates = [f for f in files_to_consider if f.lower().endswith('.wav')]
            non_wav_candidates = [f for f in files_to_consider if not f.lower().endswith('.wav')]

            print("\n📊 File Status Summary:")
            print(f"   WAV files detected: {len(wav_candidates)} (format will be verified when staged)")
            print(f"   Non-WAV files: {len(non_wav_candidates)} (will convert on demand)")

        listed = sorted(files_to_consider)
        preview_count = min(10, len(listed))
        print(f"\n🎵 Queued {len(files_to_consider)} files for processing:")
        for f in listed[:preview_count]:
            print(f"  - {f}")
        if len(listed) > preview_count:
            print(f"  … and {len(listed) - preview_count} more")

        return files_to_consider

    def prepare_staged_audio_file(self, staging_dir: str, staged_filename: str) -> Optional[str]:
        """Ensure the staged file is a 16kHz mono WAV and return its filename."""
        staged_path = os.path.join(staging_dir, staged_filename)
        if not os.path.exists(staged_path):
            print(f"❌ Staged file missing: {staged_filename}")
            return None

        if staged_filename.lower().endswith('.wav'):
            sample_rate, channels = self.converter.probe_wav(staged_path)
            ready = sample_rate == 16000 and (channels in (None, 1))

            if ready:
                info = "16kHz"
                if channels:
                    info += f", {channels} channel{'s' if channels != 1 else ''}"
                print(f"  ✅ {staged_filename} ({info})")
                return staged_filename

            print(f"  🔄 {staged_filename} (needs format conversion)")
            if sample_rate and channels:
                print(
                    f"      Detected {sample_rate}Hz, {channels} channel"
                    f"{'s' if channels != 1 else ''}; converting to 16kHz mono"
                )
            elif sample_rate:
                print(f"      Detected {sample_rate}Hz; converting to 16kHz mono")
            elif channels:
                print(
                    f"      Detected {channels} channel"
                    f"{'s' if channels != 1 else ''}; ensuring 16kHz mono"
                )
            else:
                print("      Could not read WAV metadata reliably; forcing conversion")

            if self._convert_wav_file(staging_dir, staged_filename):
                return staged_filename
            return None

        print(f"  🔄 {staged_filename} (converting staged copy to WAV)")
        if self._convert_non_wav_file(staging_dir, staged_filename):
            base_name = os.path.splitext(staged_filename)[0]
            wav_filename = f"{base_name}.wav"
            original_path = staged_path
            if os.path.exists(original_path):
                try:
                    os.remove(original_path)
                except Exception as exc:
                    print(f"⚠️  Warning: Could not remove staged original '{staged_filename}': {exc}")
            return wav_filename

        return None

    def find_and_convert_audio_files(self, audio_dir: str, output_dir: str, source_output_dir: str) -> List[str]:
        """Deprecated helper retained for backward compatibility."""
        print("⚠️  find_and_convert_audio_files is deprecated; using staged processing plan instead.")
        return self.get_audio_files_to_process(audio_dir, source_output_dir)
    
    def _convert_wav_file(self, audio_dir: str, wav_file: str) -> bool:
        """Convert WAV file to correct format."""
        input_path = os.path.join(audio_dir, wav_file)
        output_path = input_path  # Overwrite the same file
        print(f"🔄 Converting WAV format: {wav_file}")
        
        if self.converter.convert_to_wav(input_path, output_path):
            print(f"  ✅ {wav_file} ✓ converted to 16kHz mono")
            return True
        else:
            print(f"  ❌ {wav_file} failed format conversion")
            return False
    
    def _convert_non_wav_file(self, audio_dir: str, audio_file: str) -> bool:
        """Convert non-WAV file to WAV format."""
        input_path = os.path.join(audio_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        wav_filename = f"{base_name}.wav"
        wav_path = os.path.join(audio_dir, wav_filename)
        
        ext = os.path.splitext(audio_file)[1]
        print(f"🔄 Converting {ext} to WAV: {audio_file}")
        
        if self.converter.convert_to_wav(input_path, wav_path):
            print(f"  ✅ {audio_file} successfully converted to {wav_filename}")
            return True
        else:
            print(f"  ❌ {audio_file} failed conversion")
            return False
