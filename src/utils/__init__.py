"""
Utility modules for the document extraction system.

Provides common utility functions for file handling, hashing,
date manipulation, and string operations.
"""

from src.utils.date_utils import (
    calculate_age,
    format_date,
    get_current_timestamp,
    is_valid_date,
    parse_date,
    parse_date_range,
)
from src.utils.file_utils import (
    FileLock,
    atomic_write,
    cleanup_temp_files,
    ensure_directory,
    get_file_hash,
    get_temp_path,
    safe_filename,
)
from src.utils.hash_utils import (
    compute_file_hash,
    compute_md5,
    compute_sha256,
    generate_unique_id,
    mask_sensitive_data,
)
from src.utils.string_utils import (
    clean_currency,
    extract_numbers,
    fuzzy_match,
    levenshtein_distance,
    normalize_name,
    normalize_whitespace,
    truncate_text,
)


__all__ = [
    # File utilities
    "ensure_directory",
    "get_file_hash",
    "safe_filename",
    "get_temp_path",
    "cleanup_temp_files",
    "atomic_write",
    "FileLock",
    # Hash utilities
    "compute_sha256",
    "compute_md5",
    "compute_file_hash",
    "generate_unique_id",
    "mask_sensitive_data",
    # Date utilities
    "parse_date",
    "format_date",
    "parse_date_range",
    "is_valid_date",
    "get_current_timestamp",
    "calculate_age",
    # String utilities
    "normalize_whitespace",
    "normalize_name",
    "extract_numbers",
    "clean_currency",
    "truncate_text",
    "levenshtein_distance",
    "fuzzy_match",
]
