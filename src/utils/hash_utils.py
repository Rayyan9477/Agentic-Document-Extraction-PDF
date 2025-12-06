"""
Hash and encryption utility functions.

Provides secure hashing, unique ID generation, and data masking
functions for HIPAA-compliant document processing.
"""

import hashlib
import re
import secrets
import uuid
from pathlib import Path
from typing import Any


def compute_sha256(data: bytes | str) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Bytes or string to hash.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def compute_md5(data: bytes | str) -> str:
    """
    Compute MD5 hash of data.

    Note: MD5 is not cryptographically secure. Use for checksums only.

    Args:
        data: Bytes or string to hash.

    Returns:
        Hexadecimal MD5 hash string.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.md5(data).hexdigest()


def compute_file_hash(
    file_path: Path | str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Compute hash of a file.

    Args:
        file_path: Path to file.
        algorithm: Hash algorithm (sha256, sha384, sha512, md5).
        chunk_size: Size of chunks to read.

    Returns:
        Hexadecimal hash string.

    Raises:
        ValueError: If invalid algorithm specified.
        FileNotFoundError: If file does not exist.
    """
    algorithms = {
        "sha256": hashlib.sha256,
        "sha384": hashlib.sha384,
        "sha512": hashlib.sha512,
        "md5": hashlib.md5,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Invalid algorithm. Use: {list(algorithms.keys())}")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = algorithms[algorithm]()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def generate_unique_id(prefix: str = "", length: int = 32) -> str:
    """
    Generate a cryptographically secure unique ID.

    Args:
        prefix: Optional prefix for the ID.
        length: Length of the random portion (max 64).

    Returns:
        Unique identifier string.
    """
    length = min(length, 64)

    # Use secrets for cryptographic randomness
    random_hex = secrets.token_hex(length // 2)

    if prefix:
        return f"{prefix}_{random_hex}"

    return random_hex


def generate_uuid4() -> str:
    """
    Generate a random UUID4.

    Returns:
        UUID4 string in standard format.
    """
    return str(uuid.uuid4())


def generate_document_id(
    file_path: Path | str | None = None,
    content_hash: str | None = None,
) -> str:
    """
    Generate a unique document identifier.

    Uses content hash if provided, otherwise generates random ID.

    Args:
        file_path: Optional file path for context.
        content_hash: Optional content hash to incorporate.

    Returns:
        Unique document ID.
    """
    if content_hash:
        # Use content hash as base for deterministic ID
        return f"doc_{content_hash[:16]}"

    # Generate random ID
    return f"doc_{generate_unique_id(length=16)}"


# PHI masking patterns with replacement values
PHI_PATTERNS: list[tuple[str, str, str]] = [
    # SSN: 123-45-6789 or 123456789
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "***-**-****", "SSN"),
    # Phone: (123) 456-7890, 123-456-7890, 1234567890
    (r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "(***) ***-****", "PHONE"),
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "****@****.***", "EMAIL"),
    # Dates: MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD
    (
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        "**/**/****",
        "DATE",
    ),
    # Medicare ID (11 alphanumeric characters starting with 1)
    (r"\b1[A-Z0-9]{10}\b", "***********", "MEDICARE_ID"),
    # Medical Record Numbers (alphanumeric, 6-12 chars)
    (r"\bMRN[:\s]*[A-Z0-9]{6,12}\b", "MRN: ********", "MRN"),
    # NPI (10 digits starting with 1 or 2)
    (r"\b[12]\d{9}\b", "**********", "NPI"),
    # Credit card numbers (13-19 digits with optional separators)
    (r"\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b", "****-****-****-****", "CREDIT_CARD"),
    # Account numbers (generic pattern)
    (r"\b(?:Account|Acct)[:\s#]*[A-Z0-9]{6,15}\b", "Account: ********", "ACCOUNT"),
    # Member IDs (generic pattern)
    (r"\b(?:Member|Subscriber)[:\s#]*[A-Z0-9]{6,15}\b", "Member: ********", "MEMBER_ID"),
]


def mask_sensitive_data(
    text: str,
    mask_char: str = "*",
    patterns: list[tuple[str, str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Mask sensitive PHI data in text.

    Args:
        text: Text containing potential PHI.
        mask_char: Character to use for masking.
        patterns: Optional custom patterns. Defaults to PHI_PATTERNS.

    Returns:
        Tuple of (masked text, list of detected items).

    Example:
        masked, detected = mask_sensitive_data("SSN: 123-45-6789")
        # masked = "SSN: ***-**-****"
        # detected = [{"type": "SSN", "position": 5, "masked": "***-**-****"}]
    """
    if patterns is None:
        patterns = PHI_PATTERNS

    detected_items: list[dict[str, Any]] = []
    masked_text = text

    for pattern, replacement, item_type in patterns:
        regex = re.compile(pattern, re.IGNORECASE)

        for match in regex.finditer(text):
            detected_items.append(
                {
                    "type": item_type,
                    "position": match.start(),
                    "length": len(match.group()),
                    "masked": replacement,
                }
            )

        masked_text = regex.sub(replacement, masked_text)

    return masked_text, detected_items


def mask_name(name: str, preserve_initials: bool = True) -> str:
    """
    Mask a person's name.

    Args:
        name: Full name to mask.
        preserve_initials: Whether to keep first letter of each part.

    Returns:
        Masked name string.

    Example:
        mask_name("John Smith") -> "J*** S****"
        mask_name("John Smith", preserve_initials=False) -> "**** *****"
    """
    if not name:
        return name

    parts = name.split()
    masked_parts = []

    for part in parts:
        if len(part) <= 1:
            masked_parts.append("*" if not preserve_initials else part)
        elif preserve_initials:
            masked_parts.append(part[0] + "*" * (len(part) - 1))
        else:
            masked_parts.append("*" * len(part))

    return " ".join(masked_parts)


def hash_for_deduplication(
    content: str,
    normalize: bool = True,
) -> str:
    """
    Create hash for document deduplication.

    Normalizes content to handle minor variations.

    Args:
        content: Document content to hash.
        normalize: Whether to normalize whitespace.

    Returns:
        Hash string for deduplication comparison.
    """
    if normalize:
        # Normalize whitespace
        content = " ".join(content.split())
        # Lowercase
        content = content.lower()
        # Remove punctuation variations
        content = re.sub(r"[^\w\s]", "", content)

    return compute_sha256(content)


def create_hmac(
    data: bytes | str,
    key: bytes | str,
    algorithm: str = "sha256",
) -> str:
    """
    Create HMAC signature for data integrity verification.

    Args:
        data: Data to sign.
        key: Secret key for HMAC.
        algorithm: Hash algorithm for HMAC.

    Returns:
        Hexadecimal HMAC signature.
    """
    import hmac as hmac_module

    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(key, str):
        key = key.encode("utf-8")

    return hmac_module.new(key, data, algorithm).hexdigest()


def verify_hmac(
    data: bytes | str,
    key: bytes | str,
    signature: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify HMAC signature.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        data: Data that was signed.
        key: Secret key used for signing.
        signature: Signature to verify.
        algorithm: Hash algorithm used.

    Returns:
        True if signature is valid.
    """
    import hmac as hmac_module

    expected = create_hmac(data, key, algorithm)
    return hmac_module.compare_digest(expected, signature)
