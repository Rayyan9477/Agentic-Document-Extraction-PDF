"""
PHI masking primitive shared by all exporters.

This module exposes a single function, ``enforce_mask_phi``, used by
``src/export/*`` whenever a request sets ``mask_phi=True``. It is the
single source of truth for "what gets redacted in an export" so behaviour
stays consistent across JSON, Excel, Markdown, FHIR, and any future
formats.

The primitive operates on **already-extracted records** (i.e. dicts that
came out of the validator). It is a deterministic, regex-driven layer
that does not require the heavier ``openai/privacy-filter`` token
classifier (that lives in ``src/security/phi_redactor.py`` and is
opt-in via PHI mode). Both layers stack: PHI mode redacts strings *before*
storage, and ``mask_phi`` redacts again at export time as a defence-in-depth
guarantee for users who only pass the export flag.

Usage::

    from src.security.phi_mask import enforce_mask_phi

    record = {"patient_name": "John Doe", "ssn": "123-45-6789", "amount": 250.0}
    masked = enforce_mask_phi(record, phi_field_names={"patient_name", "ssn"})
    # -> {"patient_name": "[REDACTED]", "ssn": "[REDACTED]", "amount": 250.0}

The default ``PHI_FIELD_PATTERNS`` covers the field-name patterns we see
in CMS-1500 / UB-04 / EOB / Superbill schemas. Callers can override per
schema by passing an explicit ``phi_field_names`` set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


REDACTED_TOKEN: str = "[REDACTED]"

# Field-name fragments that imply PHI under HIPAA's 18 identifiers.
# Matched case-insensitively against schema field names.
PHI_FIELD_PATTERNS: tuple[str, ...] = (
    "patient",
    "subscriber",
    "member",
    "guarantor",
    "insured",
    "name",
    "first_name",
    "last_name",
    "middle_name",
    "dob",
    "birth",
    "ssn",
    "social_security",
    "mrn",
    "medical_record",
    "phone",
    "fax",
    "email",
    "address",
    "city",
    "state",
    "zip",
    "postal",
    "policy_number",
    "member_id",
    "account_number",
    "claim_number",
    "license",
    "vehicle",
    "fingerprint",
    "biometric",
    "photo",
    "device_id",
    "url",
    "ip_address",
)


def _is_phi_field_name(field_name: str, extra_patterns: Iterable[str] = ()) -> bool:
    """Return ``True`` if ``field_name`` looks like a PHI identifier.

    The match is case-insensitive substring against PHI_FIELD_PATTERNS plus
    any caller-supplied extras.
    """
    needle = field_name.lower()
    for pattern in (*PHI_FIELD_PATTERNS, *extra_patterns):
        if pattern.lower() in needle:
            return True
    return False


# Regex patterns that match PHI shapes inside string values, used as a
# value-level fallback when callers pass mask_phi without a field list.
PHI_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),             # US phone
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),  # email
    re.compile(r"\b\d{1,5}\s+[A-Za-z0-9 ]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b",
               re.IGNORECASE),                                # street address
    re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b"),  # date
)


def _redact_string_value(value: str) -> str:
    """Apply value-level PHI regexes; fully redact if any pattern matches."""
    for pattern in PHI_VALUE_PATTERNS:
        if pattern.search(value):
            return REDACTED_TOKEN
    return value


def enforce_mask_phi(
    record: dict[str, Any],
    *,
    phi_field_names: set[str] | None = None,
    extra_patterns: Iterable[str] = (),
    redact_values: bool = True,
) -> dict[str, Any]:
    """Return a deep-copied ``record`` with PHI fields redacted.

    Args:
        record: Extracted record (a flat or nested dict).
        phi_field_names: Explicit set of field names to always redact.
            If ``None``, fields are detected by name pattern.
        extra_patterns: Additional name-fragment patterns to treat as PHI
            (case-insensitive substring match).
        redact_values: Also scan string values for SSN / phone / email /
            address / date shapes and redact whole values that match.
            Defaults to ``True`` for defence-in-depth; set to ``False`` for
            schemas where every PHI field is already in ``phi_field_names``.

    Returns:
        A new dict with PHI fields replaced by ``[REDACTED]``. The input is
        never mutated.

    Notes:
        - Lists and nested dicts are walked recursively.
        - Non-string scalar values in PHI fields are still redacted (e.g.
          numeric account numbers).
        - This function is *only* a regex / name-pattern layer. For
          ML-grade redaction (token-classifier covering uncommon names,
          regional address formats, etc.), enable PHI mode via
          ``src.security.phi_redactor.PHIRedactor`` upstream.
    """
    return _walk(record, phi_field_names, tuple(extra_patterns), redact_values)


def _walk(
    obj: Any,
    phi_field_names: set[str] | None,
    extra_patterns: tuple[str, ...],
    redact_values: bool,
    parent_field: str | None = None,
) -> Any:
    if isinstance(obj, dict):
        return {
            key: _walk(value, phi_field_names, extra_patterns, redact_values, parent_field=key)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            _walk(item, phi_field_names, extra_patterns, redact_values, parent_field=parent_field)
            for item in obj
        ]

    # Scalar leaf. Decide whether to redact.
    field_name = parent_field or ""
    is_phi_field = (
        (phi_field_names is not None and field_name in phi_field_names)
        or (phi_field_names is None and _is_phi_field_name(field_name, extra_patterns))
    )

    if is_phi_field:
        return REDACTED_TOKEN if obj is not None else None

    if redact_values and isinstance(obj, str):
        return _redact_string_value(obj)

    return obj


__all__ = ["REDACTED_TOKEN", "PHI_FIELD_PATTERNS", "PHI_VALUE_PATTERNS", "enforce_mask_phi"]
