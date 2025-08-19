"""
Data Security and Cleanup Module
Handles secure cleanup of temporary files, PHI protection, and security procedures.
"""

import logging
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import time
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCleanupManager:
    """Manages secure cleanup of temporary data and PHI protection."""
    
    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_directories: List[str] = []
        self.cleanup_thread = None
        self.cleanup_enabled = True
        self.max_file_age = timedelta(hours=2)  # Files older than 2 hours are cleaned
        self.phi_patterns = [
            # Social Security Number patterns
            r'\b\d{3}-?\d{2}-?\d{4}\b',
            r'\b\d{9}\b',
            
            # Phone number patterns
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            
            # Date of birth patterns (MM/DD/YYYY, etc.)
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            
            # Credit card numbers (basic pattern)
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        ]
    
    def register_temp_file(self, filepath: str) -> str:
        """Register a temporary file for cleanup."""
        try:
            if filepath and os.path.exists(filepath):
                self.temp_files.append(filepath)
                logger.debug(f"Registered temp file for cleanup: {os.path.basename(filepath)}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to register temp file {filepath}: {e}")
            return filepath
    
    def register_temp_directory(self, dirpath: str) -> str:
        """Register a temporary directory for cleanup."""
        try:
            if dirpath and os.path.exists(dirpath):
                self.temp_directories.append(dirpath)
                logger.debug(f"Registered temp directory for cleanup: {os.path.basename(dirpath)}")
            return dirpath
        except Exception as e:
            logger.error(f"Failed to register temp directory {dirpath}: {e}")
            return dirpath
    
    def create_secure_temp_file(self, suffix: str = "", prefix: str = "medical_") -> str:
        """Create a secure temporary file and register it for cleanup."""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(temp_fd)  # Close the file descriptor
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(temp_path, 0o600)
            
            self.register_temp_file(temp_path)
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {e}")
            raise
    
    def create_secure_temp_directory(self, suffix: str = "", prefix: str = "medical_") -> str:
        """Create a secure temporary directory and register it for cleanup."""
        try:
            temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            
            # Set restrictive permissions (owner read/write/execute only)
            os.chmod(temp_dir, 0o700)
            
            self.register_temp_directory(temp_dir)
            return temp_dir
            
        except Exception as e:
            logger.error(f"Failed to create secure temp directory: {e}")
            raise
    
    def mask_phi_in_logs(self, text: str) -> str:
        """Mask PHI in log messages and other text."""
        try:
            masked_text = text
            
            # Apply PHI masking patterns
            for pattern in self.phi_patterns:
                # Replace with asterisks, preserving some structure
                def mask_match(match):
                    original = match.group()
                    if len(original) <= 4:
                        return '*' * len(original)
                    elif len(original) <= 8:
                        return original[:2] + '*' * (len(original) - 4) + original[-2:]
                    else:
                        return original[:3] + '*' * (len(original) - 6) + original[-3:]
                
                masked_text = re.sub(pattern, mask_match, masked_text)
            
            return masked_text
            
        except Exception as e:
            logger.error(f"PHI masking failed: {e}")
            return text  # Return original text if masking fails
    
    def secure_delete_file(self, filepath: str) -> bool:
        """Securely delete a file by overwriting before deletion."""
        try:
            if not os.path.exists(filepath):
                return True
            
            # Get file size
            file_size = os.path.getsize(filepath)
            
            # Overwrite file with random data multiple times
            with open(filepath, 'r+b') as file:
                for _ in range(3):  # 3 passes for security
                    file.seek(0)
                    file.write(os.urandom(file_size))
                    file.flush()
                    os.fsync(file.fileno())
            
            # Delete the file
            os.remove(filepath)
            logger.debug(f"Securely deleted file: {os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"Secure file deletion failed for {filepath}: {e}")
            try:
                # Fallback to regular deletion
                if os.path.exists(filepath):
                    os.remove(filepath)
                return True
            except:
                return False
    
    def secure_delete_directory(self, dirpath: str) -> bool:
        """Securely delete a directory and all its contents."""
        try:
            if not os.path.exists(dirpath):
                return True
            
            # Recursively secure delete all files
            for root, dirs, files in os.walk(dirpath, topdown=False):
                # Delete files
                for file in files:
                    filepath = os.path.join(root, file)
                    self.secure_delete_file(filepath)
                
                # Delete empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                    except OSError:
                        pass  # Directory not empty, will be handled in next iteration
            
            # Delete the root directory
            try:
                os.rmdir(dirpath)
                logger.debug(f"Securely deleted directory: {os.path.basename(dirpath)}")
                return True
            except OSError:
                # Fallback to shutil.rmtree
                shutil.rmtree(dirpath, ignore_errors=True)
                return True
                
        except Exception as e:
            logger.error(f"Secure directory deletion failed for {dirpath}: {e}")
            try:
                # Fallback to regular deletion
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath, ignore_errors=True)
                return True
            except:
                return False
    
    def cleanup_temp_files(self) -> int:
        """Clean up all registered temporary files."""
        cleaned_count = 0
        
        # Clean up files
        for filepath in self.temp_files[:]:  # Copy list to avoid modification during iteration
            try:
                if self.secure_delete_file(filepath):
                    self.temp_files.remove(filepath)
                    cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to clean up temp file {filepath}: {e}")
        
        # Clean up directories
        for dirpath in self.temp_directories[:]:  # Copy list to avoid modification during iteration
            try:
                if self.secure_delete_directory(dirpath):
                    self.temp_directories.remove(dirpath)
                    cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to clean up temp directory {dirpath}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files/directories")
        
        return cleaned_count
    
    def cleanup_old_files(self, directory: str = None) -> int:
        """Clean up old files in specified directory or system temp."""
        if directory is None:
            directory = tempfile.gettempdir()
        
        cleaned_count = 0
        cutoff_time = datetime.now() - self.max_file_age
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.startswith('medical_'):  # Only clean our temporary files
                        filepath = os.path.join(root, file)
                        try:
                            # Check file age
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if file_time < cutoff_time:
                                if self.secure_delete_file(filepath):
                                    cleaned_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to clean old file {filepath}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old temporary files")
            
        except Exception as e:
            logger.error(f"Old file cleanup failed: {e}")
        
        return cleaned_count
    
    def start_background_cleanup(self, interval_minutes: int = 30):
        """Start background cleanup thread."""
        if self.cleanup_thread is not None:
            return  # Already running
        
        def cleanup_worker():
            while self.cleanup_enabled:
                try:
                    # Clean up registered temp files
                    self.cleanup_temp_files()
                    
                    # Clean up old files
                    self.cleanup_old_files()
                    
                    # Wait for next cleanup cycle
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"Started background cleanup thread (interval: {interval_minutes} minutes)")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self.cleanup_enabled = False
        if self.cleanup_thread is not None:
            self.cleanup_thread.join(timeout=5)
            self.cleanup_thread = None
            logger.info("Stopped background cleanup thread")
    
    def emergency_cleanup(self) -> Dict[str, int]:
        """Perform emergency cleanup of all temporary data."""
        logger.warning("Performing emergency data cleanup")
        
        results = {
            "temp_files_cleaned": 0,
            "old_files_cleaned": 0,
            "errors": 0
        }
        
        try:
            # Clean up registered temp files
            results["temp_files_cleaned"] = self.cleanup_temp_files()
            
            # Clean up old files
            results["old_files_cleaned"] = self.cleanup_old_files()
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
            results["errors"] += 1
        
        logger.info(f"Emergency cleanup completed: {results}")
        return results
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status and statistics."""
        return {
            "temp_files_registered": len(self.temp_files),
            "temp_directories_registered": len(self.temp_directories),
            "background_cleanup_enabled": self.cleanup_enabled,
            "background_cleanup_running": self.cleanup_thread is not None and self.cleanup_thread.is_alive(),
            "max_file_age_hours": self.max_file_age.total_seconds() / 3600,
            "phi_patterns_count": len(self.phi_patterns)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_background_cleanup()
            self.cleanup_temp_files()
        except:
            pass  # Ignore errors during cleanup


# Global cleanup manager instance
data_cleanup_manager = DataCleanupManager()

# Convenience functions
def mask_phi(text: str) -> str:
    """Convenience function to mask PHI in text."""
    return data_cleanup_manager.mask_phi_in_logs(text)

def create_secure_temp_file(suffix: str = "", prefix: str = "medical_") -> str:
    """Convenience function to create secure temp file."""
    return data_cleanup_manager.create_secure_temp_file(suffix, prefix)

def create_secure_temp_directory(suffix: str = "", prefix: str = "medical_") -> str:
    """Convenience function to create secure temp directory."""
    return data_cleanup_manager.create_secure_temp_directory(suffix, prefix)

def register_for_cleanup(path: str) -> str:
    """Convenience function to register file/directory for cleanup."""
    if os.path.isdir(path):
        return data_cleanup_manager.register_temp_directory(path)
    else:
        return data_cleanup_manager.register_temp_file(path)

def emergency_cleanup() -> Dict[str, int]:
    """Convenience function for emergency cleanup."""
    return data_cleanup_manager.emergency_cleanup()