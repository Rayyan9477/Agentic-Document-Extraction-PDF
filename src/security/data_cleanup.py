"""
Secure Data Cleanup Module for HIPAA-Compliant PHI Deletion.

Provides secure file deletion using multiple overwrite passes,
secure memory wiping, and automated data retention management.
"""

from __future__ import annotations

import gc
import os
import secrets
import stat
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import structlog


logger = structlog.get_logger(__name__)


class DeletionMethod(str, Enum):
    """Secure deletion methods."""

    # Standard methods
    SIMPLE = "simple"  # Single pass with zeros
    DOD_3PASS = "dod_3pass"  # DoD 5220.22-M (3 passes)
    DOD_7PASS = "dod_7pass"  # DoD 5220.22-M ECE (7 passes)
    GUTMANN = "gutmann"  # Gutmann method (35 passes)
    RANDOM = "random"  # Random data passes

    # Custom
    CUSTOM = "custom"


class CleanupError(Exception):
    """Exception raised during cleanup operations."""


class SecureDeletionError(Exception):
    """Exception raised when secure deletion fails."""


@dataclass(slots=True)
class DeletionResult:
    """Result of a secure deletion operation."""

    path: Path
    success: bool
    method: DeletionMethod
    passes_completed: int
    bytes_overwritten: int
    duration_seconds: float
    error: str | None = None


@dataclass(slots=True)
class RetentionPolicy:
    """Data retention policy configuration."""

    max_age_days: int = 90
    min_age_days: int = 0
    file_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(default_factory=list)
    deletion_method: DeletionMethod = DeletionMethod.DOD_3PASS
    dry_run: bool = False


@dataclass(slots=True)
class CleanupStats:
    """Statistics from a cleanup operation."""

    files_scanned: int = 0
    files_deleted: int = 0
    files_failed: int = 0
    bytes_deleted: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class SecureOverwriter:
    """
    Implements secure overwrite patterns for data destruction.

    Supports multiple industry-standard deletion methods including
    DoD 5220.22-M and Gutmann.
    """

    # DoD 5220.22-M patterns (3-pass)
    DOD_3PASS_PATTERNS: list[bytes | None] = [
        b"\x00",  # Pass 1: All zeros
        b"\xff",  # Pass 2: All ones
        None,  # Pass 3: Random data
    ]

    # DoD 5220.22-M ECE patterns (7-pass)
    DOD_7PASS_PATTERNS: list[bytes | None] = [
        b"\x00",  # Pass 1: All zeros
        b"\xff",  # Pass 2: All ones
        None,  # Pass 3: Random
        b"\x00",  # Pass 4: All zeros
        b"\xff",  # Pass 5: All ones
        None,  # Pass 6: Random
        None,  # Pass 7: Random
    ]

    # Gutmann patterns (35 passes)
    GUTMANN_PATTERNS: list[bytes | None] = [
        None,
        None,
        None,
        None,  # Passes 1-4: Random
        b"\x55",
        b"\xaa",
        b"\x92\x49\x24",
        b"\x49\x24\x92",  # Passes 5-8
        b"\x24\x92\x49",
        b"\x00",
        b"\x11",
        b"\x22",  # Passes 9-12
        b"\x33",
        b"\x44",
        b"\x55",
        b"\x66",  # Passes 13-16
        b"\x77",
        b"\x88",
        b"\x99",
        b"\xaa",  # Passes 17-20
        b"\xbb",
        b"\xcc",
        b"\xdd",
        b"\xee",  # Passes 21-24
        b"\xff",
        b"\x92\x49\x24",
        b"\x49\x24\x92",
        b"\x24\x92\x49",  # Passes 25-28
        b"\x6d\xb6\xdb",
        b"\xb6\xdb\x6d",
        b"\xdb\x6d\xb6",  # Passes 29-31
        None,
        None,
        None,
        None,  # Passes 32-35: Random
    ]

    def __init__(self, buffer_size: int = 64 * 1024) -> None:
        """
        Initialize secure overwriter.

        Args:
            buffer_size: Size of write buffer in bytes.
        """
        self._buffer_size = buffer_size

    def get_patterns(self, method: DeletionMethod) -> list[bytes | None]:
        """Get overwrite patterns for the specified method."""
        if method == DeletionMethod.SIMPLE:
            return [b"\x00"]
        if method == DeletionMethod.DOD_3PASS:
            return self.DOD_3PASS_PATTERNS.copy()
        if method == DeletionMethod.DOD_7PASS:
            return self.DOD_7PASS_PATTERNS.copy()
        if method == DeletionMethod.GUTMANN:
            return self.GUTMANN_PATTERNS.copy()
        if method == DeletionMethod.RANDOM:
            return [None, None, None]  # 3 random passes
        return self.DOD_3PASS_PATTERNS.copy()

    def create_pattern_buffer(
        self,
        pattern: bytes | None,
        size: int,
    ) -> bytes:
        """
        Create a buffer filled with the specified pattern.

        Args:
            pattern: Pattern bytes or None for random.
            size: Buffer size.

        Returns:
            Buffer filled with pattern.
        """
        if pattern is None:
            # Generate random data
            return secrets.token_bytes(size)

        # Repeat pattern to fill buffer
        pattern_len = len(pattern)
        repeats = (size // pattern_len) + 1
        return (pattern * repeats)[:size]

    def overwrite_file(
        self,
        file_path: Path,
        method: DeletionMethod = DeletionMethod.DOD_3PASS,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DeletionResult:
        """
        Securely overwrite a file.

        Args:
            file_path: Path to file.
            method: Deletion method to use.
            progress_callback: Optional callback(pass_num, total_passes).

        Returns:
            DeletionResult with operation details.
        """
        start_time = time.time()
        patterns = self.get_patterns(method)
        total_passes = len(patterns)
        passes_completed = 0
        bytes_overwritten = 0

        try:
            # Get file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return DeletionResult(
                    path=file_path,
                    success=True,
                    method=method,
                    passes_completed=0,
                    bytes_overwritten=0,
                    duration_seconds=time.time() - start_time,
                )

            # Make file writable
            file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

            # Perform overwrite passes
            for pass_num, pattern in enumerate(patterns, 1):
                if progress_callback:
                    progress_callback(pass_num, total_passes)

                self._overwrite_pass(file_path, file_size, pattern)
                passes_completed += 1
                bytes_overwritten += file_size

            return DeletionResult(
                path=file_path,
                success=True,
                method=method,
                passes_completed=passes_completed,
                bytes_overwritten=bytes_overwritten,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return DeletionResult(
                path=file_path,
                success=False,
                method=method,
                passes_completed=passes_completed,
                bytes_overwritten=bytes_overwritten,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _overwrite_pass(
        self,
        file_path: Path,
        file_size: int,
        pattern: bytes | None,
    ) -> None:
        """Perform a single overwrite pass."""
        with open(file_path, "r+b") as f:
            remaining = file_size
            while remaining > 0:
                chunk_size = min(self._buffer_size, remaining)
                buffer = self.create_pattern_buffer(pattern, chunk_size)
                f.write(buffer)
                remaining -= chunk_size

            # Ensure data is flushed to disk
            f.flush()
            os.fsync(f.fileno())


class SecureDataCleanup:
    """
    Main class for secure data cleanup operations.

    Provides secure file deletion, directory cleanup, and
    automated retention management.
    """

    def __init__(
        self,
        overwrite_passes: int = 3,
        deletion_method: DeletionMethod = DeletionMethod.DOD_3PASS,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize secure data cleanup.

        Args:
            overwrite_passes: Number of overwrite passes for custom method.
            deletion_method: Default deletion method.
            max_workers: Max parallel workers for batch operations.
        """
        self._overwrite_passes = overwrite_passes
        self._deletion_method = deletion_method
        self._max_workers = max_workers
        self._overwriter = SecureOverwriter()

    def secure_delete_file(
        self,
        file_path: Path | str,
        method: DeletionMethod | None = None,
        verify: bool = True,
    ) -> DeletionResult:
        """
        Securely delete a file.

        Args:
            file_path: Path to file to delete.
            method: Deletion method (uses default if None).
            verify: Verify deletion success.

        Returns:
            DeletionResult with operation details.
        """
        file_path = Path(file_path)
        method = method or self._deletion_method

        if not file_path.exists():
            return DeletionResult(
                path=file_path,
                success=True,
                method=method,
                passes_completed=0,
                bytes_overwritten=0,
                duration_seconds=0.0,
                error="File does not exist",
            )

        if not file_path.is_file():
            return DeletionResult(
                path=file_path,
                success=False,
                method=method,
                passes_completed=0,
                bytes_overwritten=0,
                duration_seconds=0.0,
                error="Path is not a file",
            )

        # Perform secure overwrite
        result = self._overwriter.overwrite_file(file_path, method)

        if result.success:
            try:
                # Rename to random name before deletion (makes recovery harder)
                random_name = file_path.parent / secrets.token_hex(16)
                file_path.rename(random_name)

                # Delete the file
                random_name.unlink()

                if verify and random_name.exists():
                    result.success = False
                    result.error = "File still exists after deletion"

            except Exception as e:
                result.success = False
                result.error = f"Failed to delete file: {e}"

        logger.info(
            "secure_delete_complete",
            path=str(file_path),
            success=result.success,
            method=method.value,
            passes=result.passes_completed,
            bytes=result.bytes_overwritten,
            duration=result.duration_seconds,
        )

        return result

    def secure_delete_directory(
        self,
        dir_path: Path | str,
        recursive: bool = True,
        method: DeletionMethod | None = None,
    ) -> CleanupStats:
        """
        Securely delete all files in a directory.

        Args:
            dir_path: Directory path.
            recursive: Delete subdirectories.
            method: Deletion method.

        Returns:
            CleanupStats with operation summary.
        """
        dir_path = Path(dir_path)
        method = method or self._deletion_method
        stats = CleanupStats()
        start_time = time.time()

        if not dir_path.exists():
            stats.duration_seconds = time.time() - start_time
            return stats

        if not dir_path.is_dir():
            stats.errors.append(f"{dir_path} is not a directory")
            stats.duration_seconds = time.time() - start_time
            return stats

        # Collect all files
        files_to_delete: list[Path] = []
        dirs_to_delete: list[Path] = []

        if recursive:
            for item in dir_path.rglob("*"):
                if item.is_file():
                    files_to_delete.append(item)
                elif item.is_dir():
                    dirs_to_delete.append(item)
        else:
            for item in dir_path.iterdir():
                if item.is_file():
                    files_to_delete.append(item)

        stats.files_scanned = len(files_to_delete)

        # Delete files in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.secure_delete_file, f, method): f for f in files_to_delete
            }

            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    stats.files_deleted += 1
                    stats.bytes_deleted += result.bytes_overwritten
                else:
                    stats.files_failed += 1
                    if result.error:
                        stats.errors.append(f"{result.path}: {result.error}")

        # Remove empty directories (in reverse order for nested dirs)
        for subdir in sorted(dirs_to_delete, reverse=True):
            try:
                subdir.rmdir()
            except OSError:
                pass  # Directory not empty or already removed

        # Remove the main directory if empty
        try:
            if recursive:
                dir_path.rmdir()
        except OSError:
            pass

        stats.duration_seconds = time.time() - start_time

        logger.info(
            "secure_delete_directory_complete",
            path=str(dir_path),
            files_deleted=stats.files_deleted,
            files_failed=stats.files_failed,
            bytes_deleted=stats.bytes_deleted,
            duration=stats.duration_seconds,
        )

        return stats

    def secure_wipe_memory(self, data: bytearray) -> None:
        """
        Securely wipe sensitive data from memory.

        Args:
            data: Bytearray to wipe.
        """
        if not isinstance(data, bytearray):
            return

        # Overwrite with random data
        random_data = secrets.token_bytes(len(data))
        for i in range(len(data)):
            data[i] = random_data[i]

        # Overwrite with zeros
        for i in range(len(data)):
            data[i] = 0

        # Force garbage collection
        gc.collect()

    def secure_wipe_string(self, s: str) -> None:
        """
        Attempt to wipe a string from memory.

        Note: Python strings are immutable, so this has limited effectiveness.
        Use bytearray for sensitive data when possible.

        Args:
            s: String to wipe (reference will be cleared).
        """
        # We can't actually modify the string, but we can trigger GC
        gc.collect()


class MemorySecurityManager:
    """
    Manages secure memory handling for sensitive data.

    Provides context managers and utilities for handling PHI
    and other sensitive data in memory.
    """

    def __init__(self) -> None:
        """Initialize memory security manager."""
        self._sensitive_refs: list[bytearray] = []

    def allocate_secure_buffer(self, size: int) -> bytearray:
        """
        Allocate a secure buffer that will be wiped on cleanup.

        Args:
            size: Buffer size in bytes.

        Returns:
            Secure bytearray.
        """
        buffer = bytearray(size)
        self._sensitive_refs.append(buffer)
        return buffer

    def register_sensitive(self, data: bytearray) -> None:
        """
        Register a bytearray for secure cleanup.

        Args:
            data: Bytearray containing sensitive data.
        """
        if data not in self._sensitive_refs:
            self._sensitive_refs.append(data)

    def cleanup_all(self) -> None:
        """Securely wipe all registered sensitive data."""
        cleanup = SecureDataCleanup()
        for data in self._sensitive_refs:
            cleanup.secure_wipe_memory(data)
        self._sensitive_refs.clear()
        gc.collect()

    def __enter__(self) -> MemorySecurityManager:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup."""
        self.cleanup_all()


class RetentionManager:
    """
    Manages automated data retention and cleanup.

    Implements configurable retention policies for PHI and
    other sensitive data.
    """

    def __init__(
        self,
        policies: list[RetentionPolicy] | None = None,
        cleanup: SecureDataCleanup | None = None,
    ) -> None:
        """
        Initialize retention manager.

        Args:
            policies: Retention policies to apply.
            cleanup: Secure cleanup instance.
        """
        self._policies = policies or []
        self._cleanup = cleanup or SecureDataCleanup()

    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy."""
        self._policies.append(policy)

    def remove_policy(self, policy: RetentionPolicy) -> None:
        """Remove a retention policy."""
        if policy in self._policies:
            self._policies.remove(policy)

    def scan_directory(
        self,
        directory: Path | str,
        policy: RetentionPolicy,
    ) -> list[Path]:
        """
        Scan directory for files matching retention policy.

        Args:
            directory: Directory to scan.
            policy: Retention policy to apply.

        Returns:
            List of files eligible for deletion.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        now = datetime.now(UTC)
        min_age = timedelta(days=policy.min_age_days)
        max_age = timedelta(days=policy.max_age_days)

        eligible_files: list[Path] = []

        for pattern in policy.file_patterns:
            for file_path in directory.rglob(pattern):
                if not file_path.is_file():
                    continue

                # Check exclude patterns
                excluded = False
                for exclude in policy.exclude_patterns:
                    if file_path.match(exclude):
                        excluded = True
                        break

                if excluded:
                    continue

                # Check file age
                try:
                    mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime,
                        tz=UTC,
                    )
                    age = now - mtime

                    if min_age <= age <= max_age or age > max_age:
                        eligible_files.append(file_path)

                except OSError:
                    continue

        return eligible_files

    def apply_policy(
        self,
        directory: Path | str,
        policy: RetentionPolicy | None = None,
    ) -> CleanupStats:
        """
        Apply retention policy to a directory.

        Args:
            directory: Directory to clean.
            policy: Specific policy to apply (uses all if None).

        Returns:
            CleanupStats summarizing the operation.
        """
        directory = Path(directory)
        policies = [policy] if policy else self._policies
        stats = CleanupStats()
        start_time = time.time()

        for pol in policies:
            # Find eligible files
            eligible = self.scan_directory(directory, pol)
            stats.files_scanned += len(eligible)

            if pol.dry_run:
                logger.info(
                    "retention_dry_run",
                    directory=str(directory),
                    policy_max_age=pol.max_age_days,
                    files_would_delete=len(eligible),
                )
                continue

            # Delete eligible files
            for file_path in eligible:
                result = self._cleanup.secure_delete_file(
                    file_path,
                    method=pol.deletion_method,
                )

                if result.success:
                    stats.files_deleted += 1
                    stats.bytes_deleted += result.bytes_overwritten
                else:
                    stats.files_failed += 1
                    if result.error:
                        stats.errors.append(f"{file_path}: {result.error}")

        stats.duration_seconds = time.time() - start_time

        logger.info(
            "retention_policy_applied",
            directory=str(directory),
            files_scanned=stats.files_scanned,
            files_deleted=stats.files_deleted,
            files_failed=stats.files_failed,
            bytes_deleted=stats.bytes_deleted,
            duration=stats.duration_seconds,
        )

        return stats

    def apply_all_policies(
        self,
        directories: list[Path | str],
    ) -> dict[str, CleanupStats]:
        """
        Apply all policies to multiple directories.

        Args:
            directories: List of directories to process.

        Returns:
            Dictionary mapping directory to cleanup stats.
        """
        results: dict[str, CleanupStats] = {}

        for directory in directories:
            directory = Path(directory)
            results[str(directory)] = self.apply_policy(directory)

        return results


class TempFileManager:
    """
    Manages temporary files with automatic secure cleanup.

    Ensures all temporary files containing PHI are securely
    deleted after use.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        cleanup_method: DeletionMethod = DeletionMethod.DOD_3PASS,
    ) -> None:
        """
        Initialize temp file manager.

        Args:
            base_dir: Base directory for temp files.
            cleanup_method: Secure deletion method.
        """
        self._base_dir = Path(base_dir) if base_dir else Path("./temp")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_method = cleanup_method
        self._cleanup = SecureDataCleanup(deletion_method=cleanup_method)
        self._tracked_files: set[Path] = set()
        self._tracked_dirs: set[Path] = set()

    def create_temp_file(
        self,
        prefix: str = "tmp_",
        suffix: str = "",
        subdir: str | None = None,
    ) -> Path:
        """
        Create a tracked temporary file.

        Args:
            prefix: File name prefix.
            suffix: File name suffix.
            subdir: Optional subdirectory.

        Returns:
            Path to new temp file.
        """
        if subdir:
            target_dir = self._base_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self._base_dir

        filename = f"{prefix}{secrets.token_hex(8)}{suffix}"
        file_path = target_dir / filename

        # Create empty file
        file_path.touch()
        self._tracked_files.add(file_path)

        return file_path

    def create_temp_dir(self, prefix: str = "tmpdir_") -> Path:
        """
        Create a tracked temporary directory.

        Args:
            prefix: Directory name prefix.

        Returns:
            Path to new temp directory.
        """
        dir_name = f"{prefix}{secrets.token_hex(8)}"
        dir_path = self._base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        self._tracked_dirs.add(dir_path)
        return dir_path

    def track_file(self, file_path: Path | str) -> None:
        """Track an external file for cleanup."""
        self._tracked_files.add(Path(file_path))

    def track_directory(self, dir_path: Path | str) -> None:
        """Track an external directory for cleanup."""
        self._tracked_dirs.add(Path(dir_path))

    def cleanup_file(self, file_path: Path | str) -> DeletionResult:
        """Securely delete a specific tracked file."""
        file_path = Path(file_path)
        result = self._cleanup.secure_delete_file(file_path, self._cleanup_method)
        self._tracked_files.discard(file_path)
        return result

    def cleanup_directory(self, dir_path: Path | str) -> CleanupStats:
        """Securely delete a specific tracked directory."""
        dir_path = Path(dir_path)
        stats = self._cleanup.secure_delete_directory(dir_path, recursive=True)
        self._tracked_dirs.discard(dir_path)
        return stats

    def cleanup_all(self) -> CleanupStats:
        """Securely delete all tracked files and directories."""
        stats = CleanupStats()
        start_time = time.time()

        # Cleanup files
        for file_path in list(self._tracked_files):
            result = self.cleanup_file(file_path)
            stats.files_scanned += 1
            if result.success:
                stats.files_deleted += 1
                stats.bytes_deleted += result.bytes_overwritten
            else:
                stats.files_failed += 1
                if result.error:
                    stats.errors.append(f"{file_path}: {result.error}")

        # Cleanup directories
        for dir_path in list(self._tracked_dirs):
            dir_stats = self.cleanup_directory(dir_path)
            stats.files_scanned += dir_stats.files_scanned
            stats.files_deleted += dir_stats.files_deleted
            stats.files_failed += dir_stats.files_failed
            stats.bytes_deleted += dir_stats.bytes_deleted
            stats.errors.extend(dir_stats.errors)

        stats.duration_seconds = time.time() - start_time
        return stats

    def __enter__(self) -> TempFileManager:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager with cleanup."""
        self.cleanup_all()
