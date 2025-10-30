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
            # Use the specified custom temp directory directly
            self.temp_dir = self.config.custom_temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"🏗️  Using custom temp location for workspace: {self.temp_dir}")
        else:
            # Use a fixed directory inside the system temp location
            self.temp_dir = os.path.join(tempfile.gettempdir(), "swhisper_workspace")
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"🏗️  Using persistent workspace in system temp location: {self.temp_dir}")
        
        self.temp_audio_dir = os.path.join(self.temp_dir, "audio")
        self.temp_output_dir = os.path.join(self.temp_dir, self.config.json_dir)
        
        # Create subdirectories
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        print(f"   📁 Workspace: {self.temp_dir}")
        print(f"   📁 Audio files: {self.temp_audio_dir}")
        print(f"   📁 Output files: {self.temp_output_dir}")
        print(f"   💾 Checkpoint preservation: {'Enabled' if self.config.preserve_checkpoints else 'Disabled'}")
        
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
        """Get the output directory path within the source folder."""
        output_dir = os.path.join(self.original_audio_dir or self.config.audio_dir, self.config.json_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
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
