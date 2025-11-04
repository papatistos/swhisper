"""Workspace and temporary directory management."""

import errno
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

        # Clean up any existing audio files from previous runs
        self._cleanup_temp_audio()
        
        # Create subdirectories
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        print(f"   📁 Workspace: {self.temp_dir}")
        print(f"   📁 Audio files: {self.temp_audio_dir}")
        print(f"   📁 Output files: {self.temp_output_dir}")
        print(f"   💾 Checkpoint preservation: {'Enabled' if self.config.preserve_checkpoints else 'Disabled'}")
        
        return self.temp_dir
    
    def stage_audio_file(self, filename: str) -> Optional[str]:
        """Copy a single audio file into the temp workspace, replacing any previous file."""
        if not self.original_audio_dir:
            self.original_audio_dir = self.config.audio_dir

        if not self.original_audio_dir:
            print("❌ No source audio directory configured")
            return None

        if not self.temp_audio_dir:
            print("❌ Temporary audio workspace is not initialized")
            return None

        src_path = os.path.join(self.original_audio_dir, filename)
        if not os.path.exists(src_path):
            print(f"❌ Source file not found: {filename}")
            return None

        # Try to ensure the file is hydrated locally if it lives in cloud storage.
        if not self._ensure_local_file(src_path):
            return None

        # Remove any previously staged audio before copying the next file.
        self._cleanup_temp_audio()

        os.makedirs(self.temp_audio_dir, exist_ok=True)
        dst_path = os.path.join(self.temp_audio_dir, filename)

        try:
            shutil.copy2(src_path, dst_path)
            print(f"🚚 Staged '{filename}' → {self.temp_audio_dir}")
            return dst_path
        except Exception as exc:
            # Retry once after forcing hydration, in case the cloud provider
            # delivered the bytes lazily.
            if self._ensure_local_file(src_path, force=True):
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"🚚 Staged '{filename}' → {self.temp_audio_dir}")
                    return dst_path
                except Exception as retry_exc:
                    print(f"❌ Failed to stage {filename}: {retry_exc}")
                    return None

            print(f"❌ Failed to stage {filename}: {exc}")
            return None

    def _ensure_local_file(self, path: str, *, force: bool = False) -> bool:
        """Best-effort hydration for cloud-managed files (OneDrive, iCloud, etc.)."""
        if not os.path.exists(path):
            print(f"❌ Source file not found on disk: {os.path.basename(path)}")
            return False

        # When not forcing, try a single quick touch to avoid unnecessary waits.
        attempts = 2 if not force else 4
        backoff = 1.0

        for attempt in range(1, attempts + 1):
            try:
                with open(path, "rb") as handle:
                    handle.read(4096)
                return True
            except (FileNotFoundError, PermissionError, OSError) as exc:
                errno_code = getattr(exc, "errno", None)
                transient_codes = {
                    errno.EBUSY,
                    errno.EIO,
                    errno.ENOENT,
                    errno.EAGAIN,
                    errno.EACCES,
                }
                transient = errno_code in transient_codes or isinstance(exc, PermissionError)

                if not transient and not isinstance(exc, FileNotFoundError):
                    print(f"❌ Unable to access {os.path.basename(path)}: {exc}")
                    return False

                print(
                    f"⏳ Waiting for cloud file to hydrate: {os.path.basename(path)}"
                    f" (attempt {attempt}/{attempts})"
                )
                time.sleep(backoff)
                backoff *= 1.5

        print(f"❌ File never became available locally: {os.path.basename(path)}")
        return False
    
    def _cleanup_temp_audio(self):
        """Clean up any existing audio files from previous runs to ensure fresh start."""
        if not self.temp_audio_dir:
            return

        if os.path.exists(self.temp_audio_dir):
            try:
                # Remove all files in the temp audio directory
                removed_any = False
                for item in os.listdir(self.temp_audio_dir):
                    item_path = os.path.join(self.temp_audio_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            removed_any = True
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            removed_any = True
                    except Exception as e:
                        print(f"⚠️  Warning: Could not remove {item_path}: {e}")
                if removed_any:
                    print("🧹 Cleaned up existing temp audio files")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean temp audio directory: {e}")

    def get_source_audio_dir(self) -> str:
        """Return the directory where original audio files reside."""
        return self.original_audio_dir or self.config.audio_dir

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
                # First clean up audio files specifically
                if self.temp_audio_dir and os.path.exists(self.temp_audio_dir):
                    for item in os.listdir(self.temp_audio_dir):
                        item_path = os.path.join(self.temp_audio_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception as e:
                            print(f"⚠️  Warning: Could not remove {item_path}: {e}")

                # Then remove the entire temp workspace
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temp directory {self.temp_dir}: {e}")
            finally:
                self.temp_dir = None
