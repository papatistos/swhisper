"""Workspace and temporary directory management."""

import os
import tempfile
import shutil
import time
from typing import Optional

from .config import TranscriptionConfig


class WorkspaceManager:
    """Manage temporary workspace for robust processing."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.temp_dir: Optional[str] = None
        self.original_audio_dir: Optional[str] = None
        self.temp_audio_dir: Optional[str] = None
        self.temp_output_dir: Optional[str] = None
    
    def setup_temp_workspace(self) -> str:
        """Set up a temporary workspace for robust processing."""
        # Store original paths
        self.original_audio_dir = self.config.audio_dir
        
        # Create main temp directory
        if self.config.custom_temp_dir:
            # Use custom temp directory with session name
            session_name = f"transcription_{time.strftime('%Y%m%d_%H%M%S')}"
            self.temp_dir = os.path.join(self.config.custom_temp_dir, session_name)
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"🏗️  Using custom temp location: {self.config.custom_temp_dir}")
            print(f"   📁 Session folder: {session_name}")
        else:
            # Use system temp with cryptic name
            self.temp_dir = tempfile.mkdtemp(prefix="transcribe_", suffix="_workspace")
            print(f"🏗️  Using system temp location")
        
        self.temp_audio_dir = os.path.join(self.temp_dir, "audio")
        self.temp_output_dir = os.path.join(self.temp_dir, self.config.json_dir)
        
        # Create subdirectories
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        print(f"   📁 Workspace: {self.temp_dir}")
        print(f"   📁 Audio files: {self.temp_audio_dir}")
        print(f"   📁 Output files: {self.temp_output_dir}")
        print(f"   💾 Checkpoint preservation: {'Enabled' if self.config.preserve_checkpoints else 'Disabled'}")
        
        # Copy audio files to temp workspace
        self._copy_audio_files_to_temp()
        
        return self.temp_dir
    
    def _copy_audio_files_to_temp(self):
        """Copy audio files from source to temp directory."""
        try:
            files_in_dir = os.listdir(self.original_audio_dir)
            audio_files = [f for f in files_in_dir if self._is_audio_file(f)]
            
            if not audio_files:
                print("📋 No audio files found to copy")
                return
            
            print(f"📋 Copying {len(audio_files)} audio files to temp workspace...")
            
            copied_files = []
            failed_files = []
            
            for audio_file in audio_files:
                src_path = os.path.join(self.original_audio_dir, audio_file)
                dst_path = os.path.join(self.temp_audio_dir, audio_file)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(audio_file)
                    print(f"  ✅ {audio_file}")
                except Exception as e:
                    failed_files.append(audio_file)
                    print(f"  ❌ {audio_file}: {e}")
            
            if failed_files:
                print(f"⚠️  Warning: {len(failed_files)} files could not be copied")
            
            print(f"✅ Successfully copied {len(copied_files)} files to temp workspace")
            
        except Exception as e:
            print(f"❌ Error copying files to temp workspace: {e}")
    
    def _is_audio_file(self, filename: str) -> bool:
        """Check if file is a supported audio format."""
        supported_formats = ['.wav', '.m4a', '.mp3', '.mov', '.mp4', '.flac', '.aac', '.ogg']
        return (any(filename.lower().endswith(ext) for ext in supported_formats) and
                sum(1 for char in filename if char.isalpha()) >= 4)
    
    def get_output_dir(self) -> str:
        """Get the current output directory (temp or original)."""
        if self.temp_output_dir:
            return self.temp_output_dir
        elif self.temp_dir:
            return os.path.join(self.temp_dir, self.config.json_dir)
        else:
            # Fallback to original directory
            fallback_dir = os.path.join(self.original_audio_dir or self.config.audio_dir, self.config.json_dir)
            os.makedirs(fallback_dir, exist_ok=True)
            
            if not self._check_write_permissions(fallback_dir):
                print(f"❌ No write permission to fallback directory: {fallback_dir}")
                print("💡 Setting up temporary workspace instead...")
                if not self.temp_dir:
                    self.setup_temp_workspace()
                return os.path.join(self.temp_dir, self.config.json_dir)
            
            return fallback_dir
    
    def get_audio_dir(self) -> str:
        """Get the current audio directory (temp or original)."""
        return self.temp_audio_dir if self.temp_audio_dir else self.config.audio_dir
    
    def _check_write_permissions(self, directory: str) -> bool:
        """Check if we have write permissions to a directory."""
        try:
            test_file = os.path.join(directory, ".write_test_" + str(time.time()))
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except (OSError, IOError, PermissionError):
            return False
    
    def copy_results_to_source(self) -> bool:
        """Copy results from temp directory back to source directory."""
        if not self.original_audio_dir or not self.temp_output_dir:
            return False
        
        # Check if source directory exists and is writable
        if not os.path.exists(self.original_audio_dir):
            print(f"❌ Source directory no longer exists: {self.original_audio_dir}")
            return False
        
        if not self._check_write_permissions(self.original_audio_dir):
            print(f"❌ No write permission to source directory: {self.original_audio_dir}")
            print("💡 You can manually copy the results from the temp directory:")
            print(f"   cp -r {self.temp_output_dir} {self.original_audio_dir}/")
            return False
        
        # Ask user if they want to copy results back
        print(f"\n📂 Processing completed successfully!")
        print(f"   📁 Results are in temp directory: {self.temp_output_dir}")
        print(f"   📁 Source directory: {self.original_audio_dir}")
        
        response = input("\n🤔 Copy transcript folder to source directory? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            try:
                dest_dir = os.path.join(self.original_audio_dir, self.config.json_dir)
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(self.temp_output_dir, dest_dir)
                print(f"✅ Results copied to: {dest_dir}")
                return True
            except Exception as e:
                print(f"❌ Error copying results: {e}")
                return False
        else:
            print(f"📝 Results remain in temp directory: {self.temp_output_dir}")
            print(f"💡 Temp directory will be cleaned up when script exits")
            return False
    
    def copy_single_result_to_source(self, temp_file_path: str, base_name: str) -> bool:
        """Copy a single transcription result to the source directory immediately after completion."""
        if not self.original_audio_dir or not self.temp_output_dir:
            return False
        
        # Check if source directory exists and is writable
        if not os.path.exists(self.original_audio_dir):
            print(f"⚠️  Source directory no longer exists: {self.original_audio_dir}")
            print(f"   File remains in temp: {temp_file_path}")
            return False
        
        # Ensure destination directory exists
        dest_dir = os.path.join(self.original_audio_dir, self.config.json_dir)
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not create destination directory: {e}")
            return False
        
        if not self._check_write_permissions(dest_dir):
            print(f"⚠️  No write permission to: {dest_dir}")
            print(f"   File remains in temp: {temp_file_path}")
            return False
        
        # Copy the file
        dest_path = os.path.join(dest_dir, f"{base_name}.json")
        try:
            shutil.copy2(temp_file_path, dest_path)
            print(f"📋 Copied to source: {self.config.json_dir}/{base_name}.json")
            return True
        except Exception as e:
            print(f"⚠️  Error copying file to source: {e}")
            print(f"   File remains in temp: {temp_file_path}")
            return False
    
    def offer_cleanup_temp_workspace(self):
        """Offer to clean up temp workspace after all files are processed and copied."""
        if not self.temp_dir or not os.path.exists(self.temp_dir):
            return
        
        print(f"\n🧹 Temp workspace: {self.temp_dir}")
        
        # Check if checkpoints are preserved
        if self.config.preserve_checkpoints:
            print("⚠️  Note: This will remove any preserved checkpoints")
        
        response = input("🤔 Clean up temporary workspace? (Y/n): ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            self.cleanup_temp_workspace()
        else:
            print(f"📝 Temp workspace preserved: {self.temp_dir}")
            if self.config.preserve_checkpoints:
                print(f"💾 Checkpoints remain available for resuming")
    
    def cleanup_temp_workspace(self):
        """Clean up the temporary workspace."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temp directory {self.temp_dir}: {e}")
            finally:
                self.temp_dir = None
