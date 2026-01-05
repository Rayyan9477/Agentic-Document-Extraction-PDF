"""
File utility functions for document extraction.

Provides secure file handling operations with proper error handling
and HIPAA-compliant practices.
"""

import hashlib
import os
import re
import tempfile
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from src.config import get_logger


logger = get_logger(__name__)


class FileOperationError(Exception):
    """Exception raised for file operation errors."""


class FileLockError(Exception):
    """Exception raised when file lock cannot be acquired."""


def ensure_directory(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.

    Raises:
        FileOperationError: If directory cannot be created.
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("directory_ensured", path=str(path))
        return path
    except PermissionError as e:
        raise FileOperationError(f"Permission denied creating directory: {path}") from e
    except OSError as e:
        raise FileOperationError(f"Failed to create directory: {path}") from e


def get_file_hash(
    file_path: Path | str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Compute hash of a file.

    Args:
        file_path: Path to file to hash.
        algorithm: Hash algorithm (sha256, md5, sha1).
        chunk_size: Size of chunks to read.

    Returns:
        Hexadecimal hash string.

    Raises:
        FileOperationError: If file cannot be read.
        ValueError: If invalid algorithm specified.
    """
    algorithms = {
        "sha256": hashlib.sha256,
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}. Use: {list(algorithms.keys())}")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileOperationError(f"File not found: {file_path}")

    hasher = algorithms[algorithm]()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    except PermissionError as e:
        raise FileOperationError(f"Permission denied reading file: {file_path}") from e
    except OSError as e:
        raise FileOperationError(f"Error reading file: {file_path}") from e


def safe_filename(
    filename: str,
    max_length: int = 255,
    replacement: str = "_",
) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename.
        max_length: Maximum length of filename.
        replacement: Character to replace invalid characters with.

    Returns:
        Safe filename string.
    """
    # Remove or replace invalid characters
    # Windows: <>:"/\\|?*
    # Unix: / and null byte
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    safe_name = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(". ")

    # Collapse multiple replacements
    safe_name = re.sub(f"{re.escape(replacement)}+", replacement, safe_name)

    # Truncate if too long
    if len(safe_name) > max_length:
        # Preserve extension
        name, ext = os.path.splitext(safe_name)
        max_name_length = max_length - len(ext)
        safe_name = name[:max_name_length] + ext

    # Fallback for empty names
    if not safe_name or safe_name == replacement:
        safe_name = "unnamed"

    return safe_name


def get_temp_path(
    prefix: str = "doc_extract_",
    suffix: str = "",
    directory: Path | str | None = None,
) -> Path:
    """
    Get a temporary file path.

    Args:
        prefix: Prefix for temp file name.
        suffix: Suffix for temp file name.
        directory: Directory for temp file.

    Returns:
        Path to temporary file.
    """
    if directory:
        ensure_directory(directory)
        dir_path = Path(directory)
    else:
        dir_path = Path(tempfile.gettempdir())

    # Generate unique name using timestamp and random
    timestamp = int(time.time() * 1000000)
    temp_name = f"{prefix}{timestamp}{suffix}"

    return dir_path / temp_name


def cleanup_temp_files(
    directory: Path | str,
    pattern: str = "doc_extract_*",
    max_age_hours: float = 24.0,
) -> int:
    """
    Clean up temporary files older than specified age.

    Args:
        directory: Directory to clean.
        pattern: Glob pattern for files to clean.
        max_age_hours: Maximum age in hours before cleanup.

    Returns:
        Number of files deleted.
    """
    directory = Path(directory)

    if not directory.exists():
        return 0

    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0

    for file_path in directory.glob(pattern):
        try:
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
                logger.debug("temp_file_deleted", path=str(file_path))
        except OSError as e:
            logger.warning("temp_file_delete_failed", path=str(file_path), error=str(e))

    if deleted_count > 0:
        logger.info("temp_files_cleaned", directory=str(directory), count=deleted_count)

    return deleted_count


@contextmanager
def atomic_write(
    file_path: Path | str,
    mode: str = "w",
    encoding: str | None = "utf-8",
) -> Generator:
    """
    Context manager for atomic file writes.

    Writes to a temporary file first, then atomically moves to target.
    Ensures file is not corrupted on write failure.

    Args:
        file_path: Target file path.
        mode: Write mode ('w' or 'wb').
        encoding: File encoding (None for binary mode).

    Yields:
        File object for writing.

    Raises:
        FileOperationError: If write operation fails.

    Example:
        with atomic_write("/path/to/file.txt") as f:
            f.write("content")
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    # Create temp file in same directory for atomic rename
    temp_path = file_path.with_suffix(f".tmp.{int(time.time() * 1000)}")

    try:
        if "b" in mode:
            with open(temp_path, mode) as f:
                yield f
        else:
            with open(temp_path, mode, encoding=encoding) as f:
                yield f

        # Atomic rename (on POSIX, replaces existing file)
        temp_path.replace(file_path)
        logger.debug("atomic_write_complete", path=str(file_path))

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise FileOperationError(f"Atomic write failed: {e}") from e


class FileLock:
    """
    Simple file-based lock for process synchronization.

    Uses lock files to prevent concurrent access to resources.
    Supports timeout and automatic cleanup.

    Example:
        lock = FileLock("/path/to/resource.lock")
        with lock:
            # Exclusive access to resource
            process_resource()
    """

    def __init__(
        self,
        lock_path: Path | str,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        """
        Initialize file lock.

        Args:
            lock_path: Path to lock file.
            timeout: Maximum time to wait for lock in seconds.
            poll_interval: Time between lock attempts in seconds.
        """
        self._lock_path = Path(lock_path)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._locked = False

    def acquire(self) -> bool:
        """
        Acquire the lock.

        Returns:
            True if lock acquired successfully.

        Raises:
            FileLockError: If lock cannot be acquired within timeout.
        """
        start_time = time.time()

        while (time.time() - start_time) < self._timeout:
            try:
                # Create lock file exclusively
                fd = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.close(fd)
                self._locked = True
                logger.debug("file_lock_acquired", path=str(self._lock_path))
                return True

            except FileExistsError:
                # Lock file exists, check if stale
                if self._is_stale_lock():
                    self._cleanup_stale_lock()
                    continue

                time.sleep(self._poll_interval)

            except OSError as e:
                raise FileLockError(f"Failed to create lock file: {e}") from e

        raise FileLockError(f"Timeout acquiring lock: {self._lock_path}")

    def release(self) -> None:
        """Release the lock."""
        if self._locked and self._lock_path.exists():
            try:
                self._lock_path.unlink()
                self._locked = False
                logger.debug("file_lock_released", path=str(self._lock_path))
            except OSError as e:
                logger.warning("file_lock_release_failed", path=str(self._lock_path), error=str(e))

    def _is_stale_lock(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if lock file is stale (older than max age)."""
        try:
            if self._lock_path.exists():
                age = time.time() - self._lock_path.stat().st_mtime
                return age > max_age_seconds
        except OSError:
            pass
        return False

    def _cleanup_stale_lock(self) -> None:
        """Remove stale lock file."""
        try:
            self._lock_path.unlink()
            logger.warning("stale_lock_cleaned", path=str(self._lock_path))
        except OSError:
            pass

    def __enter__(self) -> "FileLock":
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Context manager exit."""
        self.release()


def iter_files(
    directory: Path | str,
    pattern: str = "*",
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Iterate over files in a directory.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.
        recursive: Whether to search recursively.

    Yields:
        Path objects for matching files.
    """
    directory = Path(directory)

    if not directory.exists():
        return

    if recursive:
        yield from directory.rglob(pattern)
    else:
        yield from directory.glob(pattern)


def get_file_size(file_path: Path | str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file.

    Returns:
        File size in bytes.

    Raises:
        FileOperationError: If file not found or cannot be read.
    """
    file_path = Path(file_path)

    try:
        return file_path.stat().st_size
    except FileNotFoundError as e:
        raise FileOperationError(f"File not found: {file_path}") from e
    except OSError as e:
        raise FileOperationError(f"Cannot read file size: {file_path}") from e
