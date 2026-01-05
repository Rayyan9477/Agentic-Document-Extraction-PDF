"""
String utility functions for document extraction.

Provides string manipulation, normalization, and matching functions
for processing extracted text from medical documents.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Collapses multiple spaces, tabs, newlines into single spaces.

    Args:
        text: Text to normalize.

    Returns:
        Text with normalized whitespace.

    Example:
        normalize_whitespace("Hello   World\\n\\n") -> "Hello World"
    """
    if not text:
        return ""

    # Replace all whitespace sequences with single space
    return " ".join(text.split())


def normalize_name(name: str) -> str:
    """
    Normalize a person's name for consistent formatting.

    Handles various name formats and standardizes to "LAST, FIRST MI".

    Args:
        name: Name in various formats.

    Returns:
        Normalized name string.

    Example:
        normalize_name("John Smith") -> "SMITH, JOHN"
        normalize_name("Smith, John A") -> "SMITH, JOHN A"
        normalize_name("DR. JOHN SMITH MD") -> "SMITH, JOHN"
    """
    if not name:
        return ""

    # Uppercase and normalize whitespace
    name = normalize_whitespace(name.upper())

    # Remove common prefixes/suffixes
    prefixes = ["DR.", "DR", "MR.", "MR", "MRS.", "MRS", "MS.", "MS", "MISS"]
    suffixes = [
        "MD",
        "M.D.",
        "DO",
        "D.O.",
        "NP",
        "N.P.",
        "PA",
        "P.A.",
        "RN",
        "R.N.",
        "PHD",
        "PH.D.",
        "JR",
        "JR.",
        "SR",
        "SR.",
        "II",
        "III",
        "IV",
    ]

    # Remove prefixes
    for prefix in prefixes:
        if name.startswith(prefix + " "):
            name = name[len(prefix) + 1 :]

    # Remove suffixes
    for suffix in suffixes:
        if name.endswith(" " + suffix):
            name = name[: -(len(suffix) + 1)]
        elif name.endswith(", " + suffix):
            name = name[: -(len(suffix) + 2)]

    name = name.strip()

    # Check if already in "LAST, FIRST" format
    if ", " in name:
        return name

    # Convert "FIRST LAST" to "LAST, FIRST"
    parts = name.split()

    if len(parts) >= 2:
        last = parts[-1]
        first = " ".join(parts[:-1])
        return f"{last}, {first}"

    return name


def extract_numbers(text: str) -> list[str]:
    """
    Extract all numbers from text.

    Args:
        text: Text containing numbers.

    Returns:
        List of number strings found.

    Example:
        extract_numbers("Patient has 3 visits, total $150.00")
        -> ["3", "150.00"]
    """
    if not text:
        return []

    # Match integers and decimals
    pattern = r"-?\d+(?:\.\d+)?"
    return re.findall(pattern, text)


def extract_integers(text: str) -> list[int]:
    """
    Extract all integers from text.

    Args:
        text: Text containing integers.

    Returns:
        List of integers found.
    """
    if not text:
        return []

    pattern = r"-?\d+"
    return [int(n) for n in re.findall(pattern, text)]


def clean_currency(value: str) -> Decimal | None:
    """
    Clean and parse a currency value.

    Handles various currency formats including symbols and commas.

    Args:
        value: Currency string to parse.

    Returns:
        Decimal value, or None if parsing fails.

    Example:
        clean_currency("$1,234.56") -> Decimal("1234.56")
        clean_currency("(500.00)") -> Decimal("-500.00")
        clean_currency("1234") -> Decimal("1234.00")
    """
    if not value:
        return None

    # Handle string conversion
    value = str(value).strip()

    # Check for negative in parentheses: ($500.00)
    is_negative = value.startswith("(") and value.endswith(")")
    if is_negative:
        value = value[1:-1]

    # Check for negative sign or CR indicator
    if value.endswith("CR") or value.endswith("-"):
        is_negative = True
        value = value.rstrip("CR").rstrip("-").strip()

    if value.startswith("-"):
        is_negative = True
        value = value[1:]

    # Remove currency symbols and formatting
    value = re.sub(r"[$£€¥]", "", value)
    value = value.replace(",", "")
    value = value.strip()

    if not value:
        return None

    try:
        result = Decimal(value)
        if is_negative:
            result = -result
        return result
    except InvalidOperation:
        return None


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: String to append when truncated.
        word_boundary: If True, truncate at word boundary.

    Returns:
        Truncated text.

    Example:
        truncate_text("Hello World", 8) -> "Hello..."
        truncate_text("Hello World", 8, word_boundary=False) -> "Hello..."
    """
    if not text or len(text) <= max_length:
        return text

    truncate_length = max_length - len(suffix)

    if truncate_length <= 0:
        return suffix[:max_length]

    truncated = text[:truncate_length]

    if word_boundary:
        # Find last space
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Number of single-character edits needed.

    Example:
        levenshtein_distance("kitten", "sitting") -> 3
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match(
    s1: str,
    s2: str,
    threshold: float = 0.8,
    case_sensitive: bool = False,
) -> bool:
    """
    Check if two strings match within a similarity threshold.

    Uses normalized Levenshtein distance for comparison.

    Args:
        s1: First string.
        s2: Second string.
        threshold: Minimum similarity ratio (0.0 to 1.0).
        case_sensitive: Whether to match case-sensitively.

    Returns:
        True if similarity >= threshold.

    Example:
        fuzzy_match("Smith", "Smyth") -> True (similarity ~0.8)
        fuzzy_match("John", "Jane") -> False
    """
    if not s1 or not s2:
        return s1 == s2

    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()

    max_len = max(len(s1), len(s2))

    if max_len == 0:
        return True

    distance = levenshtein_distance(s1, s2)
    similarity = 1 - (distance / max_len)

    return similarity >= threshold


def similarity_ratio(
    s1: str,
    s2: str,
    case_sensitive: bool = False,
) -> float:
    """
    Calculate similarity ratio between two strings.

    Args:
        s1: First string.
        s2: Second string.
        case_sensitive: Whether to match case-sensitively.

    Returns:
        Similarity ratio from 0.0 to 1.0.
    """
    if not s1 and not s2:
        return 1.0

    if not s1 or not s2:
        return 0.0

    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()

    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1, s2)

    return 1 - (distance / max_len)


def remove_diacritics(text: str) -> str:
    """
    Remove diacritical marks from text.

    Converts accented characters to their ASCII equivalents.

    Args:
        text: Text with potential diacritics.

    Returns:
        Text with diacritics removed.

    Example:
        remove_diacritics("José García") -> "Jose Garcia"
    """
    if not text:
        return ""

    # Normalize to decomposed form (separate character and diacritics)
    normalized = unicodedata.normalize("NFD", text)

    # Remove diacritical marks
    result = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    return result


def clean_ocr_text(text: str) -> str:
    """
    Clean common OCR errors and artifacts from text.

    Args:
        text: Raw OCR text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Common OCR substitutions (reserved for future use)
    # ocr_corrections = [
    #     (r"0", "O"),  # Zero to O (only in specific contexts)
    #     (r"l(?=[A-Z])", "I"),  # lowercase l before uppercase -> I
    #     (r"(?<=[a-z])1(?=[a-z])", "l"),  # 1 between lowercase -> l
    #     (r"\|", "I"),  # Pipe to I
    #     (r"rn", "m"),  # Common OCR error
    #     (r"vv", "w"),  # Double v to w
    # ]

    result = text

    # Remove control characters
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", result)

    # Normalize unicode quotation marks
    result = result.replace(""", '"').replace(""", '"')
    result = result.replace("'", "'").replace("'", "'")

    # Remove zero-width characters
    result = re.sub(r"[\u200b-\u200d\ufeff]", "", result)

    return result


def extract_between(
    text: str,
    start: str,
    end: str,
    inclusive: bool = False,
) -> str | None:
    """
    Extract text between two markers.

    Args:
        text: Text to search.
        start: Starting marker.
        end: Ending marker.
        inclusive: Whether to include markers in result.

    Returns:
        Text between markers, or None if not found.

    Example:
        extract_between("Name: John Smith, Age:", "Name: ", ", Age:")
        -> "John Smith"
    """
    if not text or not start or not end:
        return None

    start_idx = text.find(start)
    if start_idx == -1:
        return None

    if inclusive:
        search_start = start_idx
    else:
        search_start = start_idx + len(start)

    end_idx = text.find(end, search_start)
    if end_idx == -1:
        return None

    if inclusive:
        return text[start_idx : end_idx + len(end)]

    return text[search_start:end_idx]


def pad_string(
    text: str,
    length: int,
    pad_char: str = " ",
    align: str = "left",
) -> str:
    """
    Pad a string to a fixed length.

    Args:
        text: Text to pad.
        length: Desired total length.
        pad_char: Character to use for padding.
        align: Alignment ("left", "right", "center").

    Returns:
        Padded string.
    """
    if len(text) >= length:
        return text[:length]

    if align == "left":
        return text.ljust(length, pad_char)
    if align == "right":
        return text.rjust(length, pad_char)
    # center
    return text.center(length, pad_char)


def split_on_pattern(
    text: str,
    pattern: str,
    keep_delimiter: bool = False,
) -> list[str]:
    """
    Split text on a regex pattern.

    Args:
        text: Text to split.
        pattern: Regex pattern to split on.
        keep_delimiter: Whether to keep the delimiter in results.

    Returns:
        List of split segments.
    """
    if not text:
        return []

    if keep_delimiter:
        # Use lookahead to keep delimiter
        parts = re.split(f"(?={pattern})", text)
    else:
        parts = re.split(pattern, text)

    return [p.strip() for p in parts if p.strip()]


def is_empty_or_whitespace(text: str | None) -> bool:
    """
    Check if text is None, empty, or contains only whitespace.

    Args:
        text: Text to check.

    Returns:
        True if empty or whitespace only.
    """
    if text is None:
        return True

    return len(text.strip()) == 0


def safe_string(value: Any, default: str = "") -> str:
    """
    Safely convert any value to string.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        String representation of value.
    """
    if value is None:
        return default

    try:
        return str(value)
    except Exception:
        return default
