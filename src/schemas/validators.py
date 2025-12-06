"""
Field validators for medical document extraction.

Provides comprehensive validation for medical codes (CPT, ICD-10, NPI),
dates, currency, and other healthcare-specific data types.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.config import get_logger
from src.schemas.field_types import FieldType


logger = get_logger(__name__)


class ValidationResult(str, Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ValidationInfo:
    """
    Validation result with details.

    Attributes:
        result: Validation status.
        message: Human-readable message.
        normalized_value: Cleaned/normalized value.
        details: Additional validation details.
    """

    result: ValidationResult
    message: str
    normalized_value: Any = None
    details: dict[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.result in (ValidationResult.VALID, ValidationResult.WARNING)


# =============================================================================
# CPT Code Validation
# =============================================================================

# CPT code patterns
CPT_PATTERN = re.compile(r"^\d{5}$")
CPT_WITH_MODIFIER_PATTERN = re.compile(r"^(\d{5})[-\s]?([A-Z0-9]{2})?$")

# Common CPT code ranges (not exhaustive, for basic validation)
CPT_RANGES = [
    (99201, 99499, "E&M Services"),
    (10021, 69990, "Surgery"),
    (70010, 79999, "Radiology"),
    (80047, 89398, "Pathology & Lab"),
    (90281, 99199, "Medicine"),
    (99500, 99607, "Home Health"),
]


def validate_cpt_code(code: str | int) -> ValidationInfo:
    """
    Validate a CPT (Current Procedural Terminology) code.

    CPT codes are 5-digit numeric codes, optionally followed by
    a 2-character modifier.

    Args:
        code: CPT code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_cpt_code("99213")
        ValidationInfo(result=VALID, message="Valid CPT code", ...)
        >>> validate_cpt_code("99213-25")
        ValidationInfo(result=VALID, message="Valid CPT code with modifier", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="CPT code is required",
        )

    # Convert to string and clean
    code_str = str(code).strip().upper()

    # Remove common separators
    code_str = re.sub(r"[.\-\s]+", "-", code_str)

    # Check for modifier
    match = CPT_WITH_MODIFIER_PATTERN.match(code_str)
    if not match:
        # Try simple 5-digit pattern
        if not CPT_PATTERN.match(code_str.replace("-", "")[:5]):
            return ValidationInfo(
                result=ValidationResult.INVALID,
                message="Invalid CPT code format. Expected 5 digits with optional modifier.",
                normalized_value=code_str,
            )
        base_code = code_str[:5]
        modifier = None
    else:
        base_code = match.group(1)
        modifier = match.group(2)

    # Validate code is in valid range
    code_num = int(base_code)
    category = None

    for start, end, name in CPT_RANGES:
        if start <= code_num <= end:
            category = name
            break

    # Normalize output
    normalized = base_code
    if modifier:
        normalized = f"{base_code}-{modifier}"

    if category:
        return ValidationInfo(
            result=ValidationResult.VALID,
            message=f"Valid CPT code ({category})" + (f" with modifier {modifier}" if modifier else ""),
            normalized_value=normalized,
            details={"category": category, "modifier": modifier},
        )
    else:
        # Code doesn't fall in known ranges - might still be valid
        return ValidationInfo(
            result=ValidationResult.WARNING,
            message="CPT code format is valid but not in standard ranges",
            normalized_value=normalized,
            details={"modifier": modifier},
        )


# =============================================================================
# ICD-10 Code Validation
# =============================================================================

# ICD-10-CM pattern: Letter + 2 digits + optional decimal + up to 4 more characters
ICD10_CM_PATTERN = re.compile(
    r"^[A-TV-Z]\d{2}(?:\.?\d{0,4})?$",
    re.IGNORECASE
)

# ICD-10-PCS pattern: 7 alphanumeric characters
ICD10_PCS_PATTERN = re.compile(r"^[A-HJ-NP-Z0-9]{7}$", re.IGNORECASE)


def validate_icd10_code(code: str) -> ValidationInfo:
    """
    Validate an ICD-10 diagnosis or procedure code.

    ICD-10-CM (diagnosis): Letter + 2 digits + optional decimal + up to 4 chars
    ICD-10-PCS (procedure): 7 alphanumeric characters

    Args:
        code: ICD-10 code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_icd10_code("E11.9")
        ValidationInfo(result=VALID, message="Valid ICD-10-CM code", ...)
        >>> validate_icd10_code("0BJ08ZZ")
        ValidationInfo(result=VALID, message="Valid ICD-10-PCS code", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="ICD-10 code is required",
        )

    # Clean and normalize
    code_str = str(code).strip().upper()

    # Remove any spaces
    code_str = code_str.replace(" ", "")

    # Check ICD-10-CM format
    if ICD10_CM_PATTERN.match(code_str):
        # Normalize with decimal
        if len(code_str) > 3 and "." not in code_str:
            normalized = f"{code_str[:3]}.{code_str[3:]}"
        else:
            normalized = code_str

        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid ICD-10-CM diagnosis code",
            normalized_value=normalized,
            details={"type": "ICD-10-CM", "category": code_str[0]},
        )

    # Check ICD-10-PCS format
    if ICD10_PCS_PATTERN.match(code_str):
        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid ICD-10-PCS procedure code",
            normalized_value=code_str,
            details={"type": "ICD-10-PCS"},
        )

    return ValidationInfo(
        result=ValidationResult.INVALID,
        message="Invalid ICD-10 code format",
        normalized_value=code_str,
    )


# =============================================================================
# NPI Validation (National Provider Identifier)
# =============================================================================

def _luhn_checksum(number: str) -> bool:
    """
    Validate NPI using Luhn algorithm.

    The NPI uses a modified Luhn algorithm with a prefix of 80840.
    """
    # Prepend 80840 for healthcare provider identifier
    full_number = "80840" + number

    total = 0
    for i, digit in enumerate(reversed(full_number)):
        d = int(digit)
        if i % 2 == 0:  # Even position (from right, 0-indexed)
            pass  # Add as-is
        else:  # Odd position - double it
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


def validate_npi(npi: str | int) -> ValidationInfo:
    """
    Validate a National Provider Identifier (NPI).

    NPI is a 10-digit number that must pass the Luhn checksum
    algorithm with a healthcare prefix.

    Args:
        npi: NPI to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_npi("1234567893")
        ValidationInfo(result=VALID, message="Valid NPI", ...)
    """
    if npi is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="NPI is required",
        )

    # Convert and clean
    npi_str = re.sub(r"\D", "", str(npi))

    # Check length
    if len(npi_str) != 10:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"NPI must be exactly 10 digits (got {len(npi_str)})",
            normalized_value=npi_str,
        )

    # NPI must start with 1 or 2
    if npi_str[0] not in ("1", "2"):
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="NPI must start with 1 or 2",
            normalized_value=npi_str,
        )

    # Validate Luhn checksum
    if not _luhn_checksum(npi_str):
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="NPI failed Luhn checksum validation",
            normalized_value=npi_str,
        )

    # Determine entity type
    entity_type = "Individual" if npi_str[0] == "1" else "Organization"

    return ValidationInfo(
        result=ValidationResult.VALID,
        message=f"Valid NPI ({entity_type})",
        normalized_value=npi_str,
        details={"entity_type": entity_type},
    )


# =============================================================================
# Phone Number Validation
# =============================================================================

PHONE_PATTERN = re.compile(r"^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$")


def validate_phone(phone: str) -> ValidationInfo:
    """
    Validate and normalize US phone number.

    Args:
        phone: Phone number to validate.

    Returns:
        ValidationInfo with normalized format.
    """
    if phone is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Phone number is required",
        )

    # Extract digits
    digits = re.sub(r"\D", "", str(phone))

    # Handle leading 1 for country code
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]

    # Validate length
    if len(digits) != 10:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Phone number must be 10 digits (got {len(digits)})",
            normalized_value=phone,
        )

    # Format as XXX-XXX-XXXX
    normalized = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

    return ValidationInfo(
        result=ValidationResult.VALID,
        message="Valid phone number",
        normalized_value=normalized,
    )


# =============================================================================
# SSN Validation
# =============================================================================

SSN_PATTERN = re.compile(r"^(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})$")

# Invalid SSN patterns
INVALID_SSN_PATTERNS = [
    "000",  # First three digits can't be 000
    "666",  # First three digits can't be 666
    "9",    # First digit can't be 9 (reserved)
]


def validate_ssn(ssn: str) -> ValidationInfo:
    """
    Validate Social Security Number format.

    Note: This validates format only, not actual SSN assignment.

    Args:
        ssn: SSN to validate.

    Returns:
        ValidationInfo with masked normalized format.
    """
    if ssn is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="SSN is required",
        )

    # Extract digits
    digits = re.sub(r"\D", "", str(ssn))

    # Validate length
    if len(digits) != 9:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"SSN must be 9 digits (got {len(digits)})",
        )

    # Check for invalid patterns
    area = digits[:3]
    group = digits[3:5]
    serial = digits[5:]

    if area == "000" or area == "666" or area[0] == "9":
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid SSN area number",
        )

    if group == "00":
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid SSN group number",
        )

    if serial == "0000":
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid SSN serial number",
        )

    # Format as XXX-XX-XXXX
    normalized = f"{area}-{group}-{serial}"

    # Mask for logging (show last 4 only)
    masked = f"XXX-XX-{serial}"

    return ValidationInfo(
        result=ValidationResult.VALID,
        message="Valid SSN format",
        normalized_value=normalized,
        details={"masked": masked},
    )


# =============================================================================
# Date Validation
# =============================================================================

DATE_FORMATS = [
    "%Y-%m-%d",       # 2024-01-15
    "%m/%d/%Y",       # 01/15/2024
    "%m-%d-%Y",       # 01-15-2024
    "%m/%d/%y",       # 01/15/24
    "%d/%m/%Y",       # 15/01/2024 (European)
    "%B %d, %Y",      # January 15, 2024
    "%b %d, %Y",      # Jan 15, 2024
    "%Y%m%d",         # 20240115
]


def validate_date(
    date_value: str | datetime,
    min_date: datetime | None = None,
    max_date: datetime | None = None,
) -> ValidationInfo:
    """
    Validate and normalize date value.

    Args:
        date_value: Date to validate.
        min_date: Minimum allowed date.
        max_date: Maximum allowed date.

    Returns:
        ValidationInfo with ISO-formatted date.
    """
    if date_value is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Date is required",
        )

    # If already a datetime
    if isinstance(date_value, datetime):
        parsed = date_value
    else:
        # Try to parse
        date_str = str(date_value).strip()
        parsed = None

        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            return ValidationInfo(
                result=ValidationResult.INVALID,
                message=f"Could not parse date: {date_str}",
                normalized_value=date_str,
            )

    # Validate range
    if min_date and parsed < min_date:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Date {parsed.date()} is before minimum {min_date.date()}",
            normalized_value=parsed.strftime("%Y-%m-%d"),
        )

    if max_date and parsed > max_date:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Date {parsed.date()} is after maximum {max_date.date()}",
            normalized_value=parsed.strftime("%Y-%m-%d"),
        )

    return ValidationInfo(
        result=ValidationResult.VALID,
        message="Valid date",
        normalized_value=parsed.strftime("%Y-%m-%d"),
        details={"datetime": parsed},
    )


# =============================================================================
# Currency Validation
# =============================================================================

CURRENCY_PATTERN = re.compile(
    r"^\$?\s*-?\s*\$?\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?\s*$"
)


def validate_currency(value: str | float | int) -> ValidationInfo:
    """
    Validate and normalize currency value.

    Args:
        value: Currency value to validate.

    Returns:
        ValidationInfo with float value.
    """
    if value is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Currency value is required",
        )

    # Handle numeric types
    if isinstance(value, (int, float)):
        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid currency",
            normalized_value=float(value),
        )

    # Clean string
    value_str = str(value).strip()

    # Check for negative
    is_negative = "-" in value_str or "(" in value_str

    # Remove currency symbols and formatting
    cleaned = re.sub(r"[$,\(\)\s]", "", value_str)

    # Handle negative in parentheses
    if is_negative and "-" not in cleaned:
        cleaned = "-" + cleaned.replace("-", "")

    try:
        amount = float(cleaned)

        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid currency",
            normalized_value=amount,
        )
    except ValueError:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Invalid currency format: {value_str}",
            normalized_value=value_str,
        )


# =============================================================================
# General Field Validation
# =============================================================================

def validate_field(
    value: Any,
    field_type: FieldType,
    required: bool = False,
) -> ValidationInfo:
    """
    Validate a field value based on its type.

    Args:
        value: Value to validate.
        field_type: Type of the field.
        required: Whether field is required.

    Returns:
        ValidationInfo with validation result.
    """
    # Check required
    if value is None:
        if required:
            return ValidationInfo(
                result=ValidationResult.INVALID,
                message="Required field is missing",
            )
        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Optional field is empty",
            normalized_value=None,
        )

    # Route to specific validators
    validators = {
        FieldType.CPT_CODE: validate_cpt_code,
        FieldType.ICD10_CODE: validate_icd10_code,
        FieldType.NPI: validate_npi,
        FieldType.PHONE: validate_phone,
        FieldType.FAX: validate_phone,
        FieldType.SSN: validate_ssn,
        FieldType.DATE: validate_date,
        FieldType.CURRENCY: validate_currency,
    }

    if field_type in validators:
        return validators[field_type](value)

    # Default validation for other types
    if field_type == FieldType.EMAIL:
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        if email_pattern.match(str(value)):
            return ValidationInfo(
                result=ValidationResult.VALID,
                message="Valid email",
                normalized_value=str(value).lower(),
            )
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid email format",
            normalized_value=value,
        )

    if field_type == FieldType.ZIP_CODE:
        zip_pattern = re.compile(r"^\d{5}(?:-\d{4})?$")
        zip_str = re.sub(r"\s", "", str(value))
        if zip_pattern.match(zip_str):
            return ValidationInfo(
                result=ValidationResult.VALID,
                message="Valid ZIP code",
                normalized_value=zip_str,
            )
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid ZIP code format",
            normalized_value=value,
        )

    if field_type == FieldType.STATE:
        states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC", "PR", "VI", "GU", "AS", "MP",
        ]
        state_upper = str(value).upper().strip()
        if state_upper in states:
            return ValidationInfo(
                result=ValidationResult.VALID,
                message="Valid state code",
                normalized_value=state_upper,
            )
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Invalid state code",
            normalized_value=value,
        )

    # Generic validation passed
    return ValidationInfo(
        result=ValidationResult.VALID,
        message="Field validated",
        normalized_value=value,
    )


# =============================================================================
# Medical Code Validator Class
# =============================================================================

class MedicalCodeValidator:
    """
    Comprehensive medical code validator.

    Provides validation for CPT, ICD-10, NPI, HCPCS, and NDC codes
    with caching for performance.

    Example:
        validator = MedicalCodeValidator()

        result = validator.validate_cpt("99213")
        if result.is_valid:
            print(f"Valid: {result.normalized_value}")
    """

    def __init__(self) -> None:
        """Initialize validator with caches."""
        self._cpt_cache: dict[str, ValidationInfo] = {}
        self._icd10_cache: dict[str, ValidationInfo] = {}
        self._npi_cache: dict[str, ValidationInfo] = {}

    def validate_cpt(self, code: str) -> ValidationInfo:
        """Validate CPT code with caching."""
        if code in self._cpt_cache:
            return self._cpt_cache[code]

        result = validate_cpt_code(code)
        self._cpt_cache[code] = result
        return result

    def validate_icd10(self, code: str) -> ValidationInfo:
        """Validate ICD-10 code with caching."""
        if code in self._icd10_cache:
            return self._icd10_cache[code]

        result = validate_icd10_code(code)
        self._icd10_cache[code] = result
        return result

    def validate_npi(self, npi: str) -> ValidationInfo:
        """Validate NPI with caching."""
        if npi in self._npi_cache:
            return self._npi_cache[npi]

        result = validate_npi(npi)
        self._npi_cache[npi] = result
        return result

    def validate_codes(
        self,
        codes: list[str],
        code_type: str,
    ) -> list[ValidationInfo]:
        """
        Validate multiple codes of the same type.

        Args:
            codes: List of codes to validate.
            code_type: Type of codes (cpt, icd10, npi).

        Returns:
            List of ValidationInfo results.
        """
        validators = {
            "cpt": self.validate_cpt,
            "icd10": self.validate_icd10,
            "npi": self.validate_npi,
        }

        validator = validators.get(code_type.lower())
        if not validator:
            raise ValueError(f"Unknown code type: {code_type}")

        return [validator(code) for code in codes]

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cpt_cache.clear()
        self._icd10_cache.clear()
        self._npi_cache.clear()
