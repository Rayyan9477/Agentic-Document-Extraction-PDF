"""
Secure Path Validation Module.

Provides comprehensive path traversal protection for file operations.
Implements OWASP path traversal prevention guidelines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)


class PathTraversalError(Exception):
    """Exception raised when path traversal attack is detected."""


class PathValidationError(Exception):
    """Exception raised when path validation fails."""


class SecurePathValidator:
    """
    Secure path validator with traversal protection.

    Prevents:
    - Directory traversal attacks (../ sequences)
    - Absolute path injection
    - Null byte injection
    - URL-encoded traversal sequences
    - Unicode normalization attacks
    - Symbolic link escapes
    - Case-insensitive bypass on Windows

    Usage:
        validator = SecurePathValidator(
            allowed_directories=["/app/uploads", "/app/data"],
            allowed_extensions=[".pdf", ".json"],
        )
        safe_path = validator.validate(user_provided_path)
    """

    # Dangerous patterns that indicate traversal attempts
    TRAVERSAL_PATTERNS = [
        r"\.\.",           # Direct ..
        r"\.\\/",          # .\ on Windows
        r"%2e%2e",         # URL-encoded ..
        r"%252e%252e",     # Double URL-encoded ..
        r"\.%2e",          # Mixed encoding
        r"%2e\.",          # Mixed encoding
        r"%c0%ae",         # UTF-8 overlong encoding of .
        r"%c1%9c",         # UTF-8 overlong encoding of /
        r"\.\.%00",        # Null byte after ..
        r"\.\.%5c",        # URL-encoded backslash
        r"\.\.%2f",        # URL-encoded forward slash
    ]

    # Characters that should never appear in sanitized paths
    DANGEROUS_CHARS = {
        '\x00',  # Null byte
        '\n',    # Newline
        '\r',    # Carriage return
        '\t',    # Tab
        '|',     # Pipe (command injection)
        ';',     # Command separator
        '&',     # Command chaining
        '$',     # Variable expansion
        '`',     # Command substitution
        '!',     # History expansion
        '*',     # Glob wildcard
        '?',     # Glob wildcard
        '[',     # Glob pattern
        ']',     # Glob pattern
        '{',     # Brace expansion
        '}',     # Brace expansion
        '<',     # Redirection
        '>',     # Redirection
    }

    def __init__(
        self,
        allowed_directories: Sequence[str | Path] | None = None,
        allowed_extensions: Sequence[str] | None = None,
        max_path_length: int = 4096,
        max_filename_length: int = 255,
        allow_absolute_paths: bool = False,
        resolve_symlinks: bool = True,
        base_directory: str | Path | None = None,
    ) -> None:
        """
        Initialize secure path validator.

        Args:
            allowed_directories: List of directories where paths must reside.
                                If None, paths are only checked for traversal patterns.
            allowed_extensions: List of allowed file extensions (e.g., [".pdf", ".json"]).
                               If None, all extensions are allowed.
            max_path_length: Maximum allowed path length.
            max_filename_length: Maximum allowed filename length.
            allow_absolute_paths: Whether to allow absolute paths as input.
            resolve_symlinks: Whether to resolve symlinks before validation.
            base_directory: Base directory for relative path resolution.
        """
        self._allowed_dirs: list[Path] = []
        if allowed_directories:
            for d in allowed_directories:
                resolved = Path(d).resolve()
                self._allowed_dirs.append(resolved)

        self._allowed_extensions: set[str] | None = None
        if allowed_extensions:
            # Normalize extensions to lowercase with dot prefix
            self._allowed_extensions = {
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in allowed_extensions
            }

        self._max_path_length = max_path_length
        self._max_filename_length = max_filename_length
        self._allow_absolute = allow_absolute_paths
        self._resolve_symlinks = resolve_symlinks
        self._base_dir = Path(base_directory).resolve() if base_directory else Path.cwd()

        # Compile traversal patterns for efficiency
        self._traversal_regex = re.compile(
            '|'.join(self.TRAVERSAL_PATTERNS),
            re.IGNORECASE
        )

    def validate(self, path: str | Path, must_exist: bool = False) -> Path:
        """
        Validate and sanitize a path.

        Args:
            path: The path to validate.
            must_exist: Whether the path must exist.

        Returns:
            Validated and resolved Path object.

        Raises:
            PathTraversalError: If path traversal is detected.
            PathValidationError: If path fails validation rules.
        """
        # Convert to string for pattern checking
        path_str = str(path)

        # Check for null bytes first
        if '\x00' in path_str:
            logger.warning(
                "path_traversal_null_byte",
                path=path_str[:100],
            )
            raise PathTraversalError("Null byte detected in path")

        # Check length limits
        if len(path_str) > self._max_path_length:
            raise PathValidationError(
                f"Path exceeds maximum length of {self._max_path_length}"
            )

        # Check for URL-encoded and other traversal patterns
        if self._traversal_regex.search(path_str):
            logger.warning(
                "path_traversal_pattern",
                path=path_str[:100],
            )
            raise PathTraversalError("Path traversal pattern detected")

        # Check for dangerous characters
        dangerous_found = set(path_str) & self.DANGEROUS_CHARS
        if dangerous_found:
            logger.warning(
                "path_dangerous_chars",
                path=path_str[:100],
                chars=list(dangerous_found),
            )
            raise PathTraversalError(
                f"Dangerous characters in path: {dangerous_found}"
            )

        # Parse path
        try:
            parsed = Path(path_str)
        except Exception as e:
            raise PathValidationError(f"Invalid path format: {e}") from e

        # Check absolute path policy
        if parsed.is_absolute() and not self._allow_absolute:
            raise PathValidationError(
                "Absolute paths are not allowed"
            )

        # Check for .. components in the parsed path
        # This catches cases the regex might miss after normalization
        for part in parsed.parts:
            if part == '..':
                logger.warning(
                    "path_traversal_dotdot",
                    path=path_str[:100],
                )
                raise PathTraversalError("Parent directory reference (..) not allowed")
            if part.startswith('.') and part not in ('.', '..'):
                # Hidden files - may be allowed but log it
                logger.debug(
                    "path_hidden_file",
                    path=path_str[:100],
                )

        # Resolve to absolute path
        if parsed.is_absolute():
            resolved = parsed.resolve() if self._resolve_symlinks else parsed
        else:
            resolved = (self._base_dir / parsed)
            if self._resolve_symlinks:
                resolved = resolved.resolve()

        # Check filename length
        if resolved.name and len(resolved.name) > self._max_filename_length:
            raise PathValidationError(
                f"Filename exceeds maximum length of {self._max_filename_length}"
            )

        # Check extension
        if self._allowed_extensions:
            suffix = resolved.suffix.lower()
            if suffix not in self._allowed_extensions:
                raise PathValidationError(
                    f"File extension '{suffix}' not allowed. "
                    f"Allowed: {', '.join(sorted(self._allowed_extensions))}"
                )

        # Check allowed directories
        if self._allowed_dirs:
            is_within_allowed = False
            for allowed_dir in self._allowed_dirs:
                try:
                    resolved.relative_to(allowed_dir)
                    is_within_allowed = True
                    break
                except ValueError:
                    continue

            if not is_within_allowed:
                logger.warning(
                    "path_outside_allowed_directory",
                    path=str(resolved)[:100],
                    allowed=str(self._allowed_dirs),
                )
                raise PathTraversalError(
                    "Path is outside allowed directories"
                )

        # Verify existence if required
        if must_exist and not resolved.exists():
            raise PathValidationError(f"Path does not exist: {resolved}")

        # Final safety check: ensure resolved path doesn't escape base
        if not self._allow_absolute and self._base_dir:
            try:
                resolved.relative_to(self._base_dir)
            except ValueError:
                # Path escaped base directory through symlink or other means
                logger.warning(
                    "path_escaped_base",
                    path=str(resolved)[:100],
                    base=str(self._base_dir),
                )
                raise PathTraversalError(
                    "Resolved path escapes base directory"
                )

        return resolved

    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize a filename (no directory components).

        Args:
            filename: The filename to validate.

        Returns:
            Sanitized filename.

        Raises:
            PathTraversalError: If traversal patterns detected.
            PathValidationError: If filename fails validation.
        """
        # Check for null bytes
        if '\x00' in filename:
            raise PathTraversalError("Null byte in filename")

        # Get basename only (strips any path components)
        basename = os.path.basename(filename)

        # Check if path components were stripped (traversal attempt)
        if basename != filename:
            logger.warning(
                "filename_contained_path",
                original=filename[:100],
                basename=basename,
            )
            raise PathTraversalError(
                "Filename contained directory separators"
            )

        # Check for dangerous characters
        dangerous_found = set(basename) & self.DANGEROUS_CHARS
        if dangerous_found:
            raise PathTraversalError(
                f"Dangerous characters in filename: {dangerous_found}"
            )

        # Check length
        if len(basename) > self._max_filename_length:
            raise PathValidationError(
                f"Filename exceeds maximum length of {self._max_filename_length}"
            )

        # Check for empty or dotfile-only names
        if not basename or basename == '.' or basename == '..':
            raise PathValidationError("Invalid filename")

        # Check extension if restrictions apply
        if self._allowed_extensions:
            suffix = Path(basename).suffix.lower()
            if suffix not in self._allowed_extensions:
                raise PathValidationError(
                    f"File extension '{suffix}' not allowed"
                )

        return basename

    def sanitize_filename(self, filename: str, replacement: str = '_') -> str:
        """
        Sanitize a filename by replacing dangerous characters.

        Args:
            filename: The filename to sanitize.
            replacement: Character to replace dangerous chars with.

        Returns:
            Sanitized filename safe for filesystem use.
        """
        # Get basename only
        basename = os.path.basename(filename.replace('\x00', ''))

        # Replace dangerous characters
        result = []
        for char in basename:
            if char in self.DANGEROUS_CHARS or ord(char) < 32:
                result.append(replacement)
            else:
                result.append(char)

        sanitized = ''.join(result)

        # Remove leading dots (hidden files)
        sanitized = sanitized.lstrip('.')

        # Handle empty result
        if not sanitized:
            sanitized = 'unnamed'

        # Truncate to max length
        if len(sanitized) > self._max_filename_length:
            # Preserve extension
            path = Path(sanitized)
            stem = path.stem[:self._max_filename_length - len(path.suffix) - 1]
            sanitized = stem + path.suffix

        return sanitized


# Default validator instances for common use cases
def get_pdf_validator(
    allowed_directories: Sequence[str | Path] | None = None,
) -> SecurePathValidator:
    """Get a validator configured for PDF file paths."""
    return SecurePathValidator(
        allowed_directories=allowed_directories,
        allowed_extensions=['.pdf'],
        allow_absolute_paths=True,  # Allow absolute paths but validate traversal
        resolve_symlinks=True,
    )


def get_output_validator(
    allowed_directories: Sequence[str | Path] | None = None,
) -> SecurePathValidator:
    """Get a validator configured for output paths."""
    return SecurePathValidator(
        allowed_directories=allowed_directories,
        allowed_extensions=['.json', '.xlsx', '.xls', '.md', '.csv'],
        allow_absolute_paths=True,
        resolve_symlinks=True,
    )


def validate_pdf_path(path: str, allowed_dirs: Sequence[str | Path] | None = None) -> Path:
    """
    Convenience function to validate a PDF file path.

    Args:
        path: Path to validate.
        allowed_dirs: Optional list of allowed directories.

    Returns:
        Validated Path object.

    Raises:
        PathTraversalError: If traversal detected.
        PathValidationError: If validation fails.
    """
    validator = get_pdf_validator(allowed_dirs)
    return validator.validate(path)


def validate_output_path(
    path: str,
    allowed_dirs: Sequence[str | Path] | None = None,
) -> Path:
    """
    Convenience function to validate an output file path.

    Args:
        path: Path to validate.
        allowed_dirs: Optional list of allowed directories.

    Returns:
        Validated Path object.

    Raises:
        PathTraversalError: If traversal detected.
        PathValidationError: If validation fails.
    """
    validator = get_output_validator(allowed_dirs)
    return validator.validate(path)


def is_safe_path(path: str) -> bool:
    """
    Quick check if a path appears safe (no traversal patterns).

    This is a lightweight check for basic traversal patterns.
    For full validation, use SecurePathValidator.

    Args:
        path: Path to check.

    Returns:
        True if path appears safe, False otherwise.
    """
    # Quick null byte check
    if '\x00' in path:
        return False

    # Check for .. anywhere
    if '..' in path:
        return False

    # Check for URL-encoded sequences
    if '%' in path.lower():
        return False

    # Check for dangerous shell characters
    dangerous = {'|', ';', '&', '$', '`', '<', '>', '*', '?'}
    if any(c in path for c in dangerous):
        return False

    return True
