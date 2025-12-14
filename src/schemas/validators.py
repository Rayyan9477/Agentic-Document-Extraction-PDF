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
# HCPCS Code Validation (Healthcare Common Procedure Coding System)
# =============================================================================

# HCPCS Level I = CPT codes (handled by validate_cpt_code)
# HCPCS Level II = Letter (A-V) + 4 digits + optional modifier
HCPCS_LEVEL2_PATTERN = re.compile(
    r"^[A-V]\d{4}(?:-[A-Z0-9]{2})?$",
    re.IGNORECASE
)


def validate_hcpcs_code(code: str | int) -> ValidationInfo:
    """
    Validate a HCPCS (Healthcare Common Procedure Coding System) code.

    HCPCS Level II codes start with a letter (A-V) followed by 4 digits.
    Optional 2-character modifier separated by hyphen.

    Args:
        code: HCPCS code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_hcpcs_code("A4253")
        ValidationInfo(result=VALID, message="Valid HCPCS Level II code", ...)
        >>> validate_hcpcs_code("E0601-RR")
        ValidationInfo(result=VALID, message="Valid HCPCS Level II code with modifier", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="HCPCS code is required",
        )

    code_str = str(code).strip().upper()

    if not code_str:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="HCPCS code is empty",
        )

    # Check if it's a CPT code (HCPCS Level I)
    if code_str[0].isdigit():
        return validate_cpt_code(code_str)

    # Validate HCPCS Level II format
    if HCPCS_LEVEL2_PATTERN.match(code_str):
        has_modifier = "-" in code_str
        category = code_str[0]

        # HCPCS categories
        categories = {
            "A": "Transportation/Medical Supplies",
            "B": "Enteral/Parenteral Therapy",
            "C": "Outpatient PPS (Temporary)",
            "D": "Dental Procedures",
            "E": "Durable Medical Equipment",
            "G": "Procedures/Professional Services (Temporary)",
            "H": "Alcohol/Drug Abuse Services",
            "J": "Drugs Administered Other Than Oral",
            "K": "Durable Medical Equipment (Temporary)",
            "L": "Orthotic/Prosthetic Procedures",
            "M": "Medical Services",
            "P": "Pathology/Laboratory Services",
            "Q": "Miscellaneous Services (Temporary)",
            "R": "Diagnostic Radiology Services",
            "S": "Commercial Payers (Temporary)",
            "T": "Medicaid Services (State)",
            "V": "Vision/Hearing Services",
        }

        return ValidationInfo(
            result=ValidationResult.VALID,
            message=f"Valid HCPCS Level II code{' with modifier' if has_modifier else ''}",
            normalized_value=code_str,
            details={
                "level": "II",
                "category": category,
                "category_name": categories.get(category, "Unknown"),
                "has_modifier": has_modifier,
            },
        )

    return ValidationInfo(
        result=ValidationResult.INVALID,
        message="Invalid HCPCS code format (expected letter A-V + 4 digits)",
        normalized_value=code_str,
    )


# =============================================================================
# NDC Code Validation (National Drug Code)
# =============================================================================

# NDC formats: 4-4-2, 5-3-2, 5-4-1 (labeler-product-package)
# Also accepts with or without hyphens
NDC_PATTERN = re.compile(
    r"^(\d{4,5})-?(\d{3,4})-?(\d{1,2})$"
)


def validate_ndc_code(code: str | int) -> ValidationInfo:
    """
    Validate an NDC (National Drug Code).

    NDC is a unique 10- or 11-digit identifier for drugs in the US.
    Format: Labeler (4-5 digits) - Product (3-4 digits) - Package (1-2 digits)

    Args:
        code: NDC code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_ndc_code("0002-3227-01")
        ValidationInfo(result=VALID, message="Valid NDC code", ...)
        >>> validate_ndc_code("00023227-01")
        ValidationInfo(result=VALID, message="Valid NDC code", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="NDC code is required",
        )

    code_str = str(code).strip()

    if not code_str:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="NDC code is empty",
        )

    # Try to match NDC pattern
    match = NDC_PATTERN.match(code_str)
    if match:
        labeler, product, package = match.groups()

        # Validate total length (10 or 11 digits)
        total_digits = len(labeler) + len(product) + len(package)
        if total_digits not in (10, 11):
            return ValidationInfo(
                result=ValidationResult.INVALID,
                message=f"NDC must be 10-11 digits, got {total_digits}",
                normalized_value=code_str,
            )

        # Normalize to standard 11-digit format with hyphens
        # Pad to 5-4-2 format
        labeler_padded = labeler.zfill(5)
        product_padded = product.zfill(4)
        package_padded = package.zfill(2)
        normalized = f"{labeler_padded}-{product_padded}-{package_padded}"

        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid NDC code",
            normalized_value=normalized,
            details={
                "labeler": labeler_padded,
                "product": product_padded,
                "package": package_padded,
                "format": f"{len(labeler)}-{len(product)}-{len(package)}",
            },
        )

    # Try without hyphens (pure digits)
    digits_only = re.sub(r"[^0-9]", "", code_str)
    if len(digits_only) in (10, 11):
        # Assume 5-4-2 format for 11 digits, 4-4-2 for 10
        if len(digits_only) == 11:
            labeler, product, package = digits_only[:5], digits_only[5:9], digits_only[9:]
        else:
            labeler, product, package = digits_only[:4], digits_only[4:8], digits_only[8:]

        normalized = f"{labeler.zfill(5)}-{product.zfill(4)}-{package.zfill(2)}"

        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid NDC code",
            normalized_value=normalized,
            details={
                "labeler": labeler.zfill(5),
                "product": product.zfill(4),
                "package": package.zfill(2),
            },
        )

    return ValidationInfo(
        result=ValidationResult.INVALID,
        message="Invalid NDC format (expected 10-11 digits as X-X-X)",
        normalized_value=code_str,
    )


# =============================================================================
# Taxonomy Code Validation (Healthcare Provider Taxonomy)
# =============================================================================

# Taxonomy codes: 10 alphanumeric characters
# Format: Level 1 (2 chars) + Level 2 (4 chars) + Level 3 (1 char) + Version (2 chars) + Check (1 char)
TAXONOMY_PATTERN = re.compile(r"^[0-9]{2}[0-9A-Z]{8}X?$", re.IGNORECASE)


def validate_taxonomy_code(code: str) -> ValidationInfo:
    """
    Validate a Healthcare Provider Taxonomy Code.

    Taxonomy codes are 10-character alphanumeric codes that categorize
    healthcare provider types and specializations.

    Args:
        code: Taxonomy code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_taxonomy_code("207Q00000X")
        ValidationInfo(result=VALID, message="Valid taxonomy code", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Taxonomy code is required",
        )

    code_str = str(code).strip().upper()

    if not code_str:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="Taxonomy code is empty",
        )

    # Must be exactly 10 characters
    if len(code_str) != 10:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Taxonomy code must be 10 characters, got {len(code_str)}",
            normalized_value=code_str,
        )

    # Validate format
    if TAXONOMY_PATTERN.match(code_str):
        # Parse taxonomy levels
        provider_type = code_str[:2]  # Grouping (e.g., "20" = Allopathic)
        classification = code_str[2:6]  # Classification
        specialization = code_str[6:9]  # Specialization
        version = code_str[9]  # Version indicator (usually X)

        return ValidationInfo(
            result=ValidationResult.VALID,
            message="Valid healthcare provider taxonomy code",
            normalized_value=code_str,
            details={
                "provider_type_code": provider_type,
                "classification": classification,
                "specialization": specialization,
                "version": version,
            },
        )

    return ValidationInfo(
        result=ValidationResult.INVALID,
        message="Invalid taxonomy code format",
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
# CARC (Claim Adjustment Reason Code) Validation
# =============================================================================

# CARC codes are 1-3 digit codes with standardized categories
# Group codes: CO (Contractual Obligation), CR (Correction/Reversal),
# OA (Other Adjustment), PI (Payer Initiated), PR (Patient Responsibility)
CARC_GROUP_CODES = frozenset({"CO", "CR", "OA", "PI", "PR"})

# Common CARC codes with their descriptions (subset for validation reference)
# Full list maintained by X12: https://x12.org/codes/claim-adjustment-reason-codes
COMMON_CARC_CODES = {
    "1": "Deductible Amount",
    "2": "Coinsurance Amount",
    "3": "Co-payment Amount",
    "4": "The procedure code is inconsistent with the modifier used",
    "5": "The procedure code/type of bill is inconsistent with the place of service",
    "6": "The procedure/revenue code is inconsistent with the patient's age",
    "7": "The procedure/revenue code is inconsistent with the patient's gender",
    "8": "The procedure code is inconsistent with the provider type/specialty",
    "9": "The diagnosis is inconsistent with the patient's age",
    "10": "The diagnosis is inconsistent with the patient's gender",
    "11": "The diagnosis is inconsistent with the procedure",
    "12": "The diagnosis is inconsistent with the provider type",
    "13": "The date of death precedes the date of service",
    "14": "The date of birth follows the date of service",
    "15": "Authorization/pre-certification denied or not obtained in time",
    "16": "Claim/service lacks information",
    "17": "Claim/service has been reviewed and approved for payment",
    "18": "Duplicate claim/service",
    "19": "Claim/service denied based on original payer decision",
    "20": "Procedure/revenue code must be billed with an appropriate modifier",
    "21": "The procedure/revenue code is invalid in the context submitted",
    "22": "Service is not covered when performed this frequently",
    "23": "Impact of prior payer(s) adjudication",
    "24": "Charges are covered under a capitation agreement",
    "25": "Payment denied - mutually exclusive procedures",
    "26": "Expenses incurred prior to coverage",
    "27": "Expenses incurred after coverage terminated",
    "29": "The time limit for filing has expired",
    "30": "Not authorized for place of service",
    "31": "Patient cannot be identified as our insured",
    "32": "Our records indicate the patient is covered by a different payer",
    "33": "Claim includes services covered under a global fee",
    "35": "Lifetime benefit maximum has been reached",
    "39": "Services denied at the time authorization/pre-certification was requested",
    "40": "Charges do not meet qualifications for emergency/urgent care",
    "45": "Charge exceeds fee schedule/maximum allowable or contracted rate",
    "49": "This is a non-covered service because it is a routine/preventive exam",
    "50": "These are non-covered services because this is not deemed a 'medical necessity'",
    "51": "Pre-existing condition",
    "55": "Procedure/treatment has not been deemed 'proven to be effective'",
    "56": "Procedure/treatment has not been deemed 'safe and effective'",
    "58": "Treatment was deemed by the payer to have been rendered in an inappropriate setting",
    "59": "Processed based on multiple or concurrent procedure rules",
    "60": "De minimis adjustment",
    "66": "Blood Deductible",
    "89": "Denied. Not covered by this payer/contractor.",
    "90": "Ingredient cost adjustment",
    "94": "Processed in Excess of charges",
    "96": "Non-covered charge(s)",
    "97": "The benefit for this service is included in the payment/allowance for another service",
    "100": "Payment made to patient/insured/responsible party",
    "101": "Predetermination: anticipated payment upon completion of services",
    "102": "Major Medical Adjustment",
    "103": "Provider promotional discount",
    "104": "Managed care withholding",
    "105": "Tax withholding",
    "107": "Related/Covered by the allowance for a previous service/claim",
    "108": "Claim/service processed according to current rent reimbursement guidelines",
    "109": "Claim not covered by this payer/contractor. Claim must be submitted to the correct payer.",
    "110": "Billing data file incomplete or invalid",
    "111": "Not covered unless provider is certified/eligible to provide treatment",
    "112": "Service not furnished directly to patient or in manner as required",
    "114": "Procedure/service is not allowed on separate days",
    "115": "Procedure/modifier combination is invalid",
    "116": "The advance indemnification notice signed by patient did not comply",
    "117": "Transportation not from or to nearest appropriate facility",
    "118": "Benefit maximum for this time period has been reached",
    "119": "Benefit maximum for this period or occurrence has been reached",
    "120": "Service previously paid under patient's outpatient benefits",
    "121": "Indemnification adjustment",
    "122": "Psychiatric reduction",
    "125": "Submission/billing error(s)",
    "128": "Newborn's services are covered in the mother's Allowance",
    "129": "Prior processing information appears incorrect",
    "130": "Claim submission fee",
    "131": "Claim specific negotiated discount",
    "132": "Prearranged demonstration project adjustment",
    "133": "The disposition of this claim/service is pending further review",
    "134": "Technical fees removed from charges",
    "135": "Interim bill adjustment",
    "136": "Failure to follow plan network rules or guidelines",
    "137": "Regulatory Surcharges, Assessments, or Allowances",
    "138": "Appeal procedures not followed or time limits not met",
    "139": "Contracted funding agreement - Loss allocation",
    "140": "Patient/Insured health identification number and name do not match",
    "141": "Claim spans eligible and ineligible periods of coverage",
    "142": "Monthly Benefit Maximum has been reached",
    "143": "Portion of payment deferred",
    "144": "Incentive adjustment",
    "145": "Premium payment withholding",
    "146": "Diagnosis was invalid for the date(s) of service reported",
    "147": "Provider contracted/negotiated rate expired or not on file",
    "148": "Information from another provider was not provided",
    "149": "Lifetime Benefit Maximum has been reached for this service/benefit",
    "150": "Payment adjusted because the payer deems the information submitted does not support this level of service",
    "151": "Payment adjusted because the payer deems the information submitted does not support this many services",
    "152": "Payment adjusted because the payer deems the information submitted does not support this length of service",
    "153": "Payment adjusted because the payer deems the information submitted does not support this dosage",
    "154": "Payment adjusted because the payer deems the information submitted does not support this day's supply",
    "155": "Patient refused the service/procedure",
    "157": "Service/procedure denied because it is an investigational/experimental service",
    "158": "Claim/service not payable under the claim's Health Insurance Prospective Payment System",
    "159": "Payment adjusted because the payer deems the information submitted does not support this product",
    "160": "Injury/illness was the liability of the no-fault or liability carrier",
    "161": "Payment denied/reduced for failure to submit certification/recertification or other documentation",
    "163": "Attachment/other documentation referenced on the claim was not received",
    "164": "Attachment/other documentation referenced on the claim was not received in a timely manner",
    "166": "Alternate benefit has been provided",
    "167": "Payment/service denied based on legislative requirements",
    "169": "Additional payment for Extraordinary Circumstances has been issued",
    "170": "Payment is denied when performed/billed by this type of provider",
    "171": "Payment is denied when performed/billed by this type of provider in this type of facility",
    "172": "Payment is adjusted when performed/billed by a provider of this specialty",
    "173": "Service/equipment/drug is not covered under the patient's current benefit plan",
    "174": "Service was not prescribed by a physician",
    "175": "Prescription is incomplete",
    "176": "Prescription is not current",
    "177": "Patient has not met the required eligibility requirements",
    "178": "Patient has not met the required spend down requirements",
    "179": "Services not provided directly by the provider",
    "180": "Not prescribed by physician",
    "181": "Procedure code was invalid on the date of service",
    "182": "Secondary payment amount",
    "183": "Sales Tax",
    "184": "Procedure billed is not authorized per your Clinical Laboratory Improvement Amendment",
    "185": "Entity's Primary Specialty/Taxonomy is in an inactive status",
    "186": "Level of care change adjustment",
    "187": "Consumer Spending Account payments",
    "188": "Product/service is out of stock",
    "189": "Product/service is not for retail sale",
    "190": "Contracted funding agreement - Loss adjustment",
    "192": "A non-specific procedure code was billed when a specific code is available",
    "193": "Original payment decision maintained after review of grievance/appeal",
    "194": "Payment denied or reduced for anesthesia. Anesthesia not covered for this procedure",
    "195": "Previously reduced/waived co-payment/co-insurance for terminated Medicare+Choice patient",
    "197": "Payment denied because the insured was not determined to be in hospice care",
    "198": "Payment denied because the insured was determined to be in hospice care",
    "199": "Revenue code and Procedure code do not match",
    "200": "Expenses incurred during lapse in coverage",
    "201": "Workers' Compensation case adjudicated as non-compensable",
    "202": "Workers' Compensation case adjudicated as non-compensable - Employee's condition has no relation to work",
    "203": "Discontinued or reduced service",
    "204": "Service is not payable per managed care contract",
    "205": "Pharmacy Prescription is not on the formulary",
    "206": "National Drug Code (NDC) is not eligible for payment",
    "207": "Claim/service denied based on Part B - MAC guidelines",
    "208": "National Drug Code (NDC) is not eligible for payment because it is not effective",
    "209": "Per regulatory or other agreement, there is a reduced schedule or fee",
    "210": "Payment adjusted because pre-certification/authorization not received timely",
    "211": "National Drug Code is missing/incomplete/invalid",
    "212": "Administrative cost adjustment",
    "213": "Non-covered personal comfort or convenience services",
    "214": "This service has been/will be paid in full by a prior payer",
    "215": "Based on subrogation of a third party settlement",
    "216": "Authorization/pre-certification, or notification absent",
    "219": "Claim based on a payer-loss ratio policy",
    "222": "Exceeds the contracted maximum",
    "223": "Adjustment based on Coordination of Benefits",
    "224": "Patient Identification compromised by identity theft",
    "225": "Penalty or Interest payment by the payer",
    "226": "Patient responsible for costs per information",
    "227": "Patient responsible for costs per exclusion",
    "228": "Patient responsible for costs per policy provisions",
    "229": "Patient responsible for costs - deemed not medically necessary",
    "231": "Mutually exclusive procedures",
    "232": "Institutional patient to be billed by spouse's insurance plan",
    "233": "Institutional patient to be billed by hospice",
    "234": "Institutional patient to be billed per managed care contract",
    "235": "Charges covered under a Per Diem or Day Rate arrangement",
    "236": "Member's plan terminated upon reaching maximum",
    "237": "Member's plan terminated prior to service date",
    "238": "Fixed format claim processing amount",
    "239": "Claim Processing prior period or hold adjustment",
    "240": "The diagnosis is inconsistent with the type of bill",
    "241": "The diagnosis is inconsistent with the place of service",
    "242": "Services not provided by network/primary care providers",
    "243": "Services not authorized by network/primary care providers",
    "245": "Waiting period applicable",
    "246": "Denied services due to failure to verify primary coverage",
    "247": "Subscriber/insured is liable for services rendered by referral provider",
    "248": "Account paid in full by uninsured motorist recovery",
    "249": "Payment adjusted based on excessive charges or unreasonable fees",
    "250": "The attachment or document received does not support the information provided",
    "251": "Fees incurred exceed benefits",
    "252": "Fee adjustment for an allowed service",
    "253": "Sequestration - Loss of federal funding",
    "254": "Claim is being re-processed",
    "256": "Service not payable per managed care contract",
    "257": "Provider performance pay adjustment",
    "258": "Claim/service denied due to claim data issues",
    "259": "Attachment/documentation found to be legally or contractually prohibited",
    "260": "Penalty for violation of administrative rules",
    "261": "Additional services not payable",
    "262": "Multiple/concurrent claims share the allowance/deductible",
    "263": "Attachment/documentation found not to satisfy requirement",
    "264": "Care may be covered by another payer per coordination of benefits",
    "265": "Prior claim identified. No payment made",
    "266": "Service denied/reduced based on previously paid duplicate claim",
    "267": "Out-of-pocket cost share calculated per published schedule",
    "268": "Out-of-pocket cost share calculated per actual charge",
    "269": "Out-of-pocket cost share calculated per contracted amount",
    "270": "Claim received after coverage terminated",
    "271": "Prior contractual/regulatory agency limitation adjustment",
    "272": "Coverage/program guidelines were not met",
    "273": "Coverage/program guidelines were exceeded",
    "274": "Fee schedule/maximum allowance violation for procedure combination",
    "275": "Prior payer's coverage has lapsed",
    "276": "Services denied. Appeals rights exhausted",
    "277": "Payer did not receive requested proof of eligibility determination",
    "278": "Claim submission requirements not met",
    "279": "Services ordered are not consistent with the current certification/determination",
    "280": "Claim/case is currently being reviewed",
    "281": "Services not payable for this patient status code",
    "282": "Service/procedure denied due to pending quality investigation",
    "283": "Patient age/service inconsistency",
    "284": "Services denied due to failure to verify coverage",
    "285": "Appeal procedures not followed",
    "286": "Provider appeal procedures not followed",
    "287": "Out of network",
    "288": "Prior claim adjudicated differently",
    "289": "Facility did not have a signed provider agreement in effect on the date of service",
    "290": "Penalty for failure to follow established protocol/guideline",
    "291": "Payment denied based on Medical Provider Discount Program regulations",
    "292": "Failed to obtain pre-certification/pre-authorization/prior authorization",
    "293": "Service is not separately payable",
    "294": "Additional services not authorized by the prior claim",
    "295": "Patient not eligible for coverage on the date of service",
    "296": "No prior coverage verification received",
    "A0": "Patient refund amount",
    "A1": "Claim submitted with less than minimum standard number of anesthesia units",
    "A5": "Medicare Claim PPS Capital Cost Outlier Amount",
    "A6": "Prior hospitalization or 30 day transfer requirement not met",
    "A7": "Presumptive Payment Adjustment",
    "A8": "Ungroupable DRG",
    "B1": "Non-covered visit(s)",
    "B4": "Late Filing Penalty",
    "B5": "Coverage/program guidelines were not met or were exceeded",
    "B7": "Provider did not follow provider certification requirements for this service",
    "B8": "Alternative services were available and should have been utilized",
    "B9": "Services were provided outside of time-frame allowed by payer",
    "B10": "Allowed amount has been reduced because the provider may not bill the insured for denied services",
    "B11": "Value of same service delivered after diagnosis made exceeds value of service delivered before diagnosis",
    "B12": "Services not documented in patient's medical records",
    "B13": "Previously paid. Payment for this claim is included in a prior payment",
    "B14": "Payment denied or reduced based on patient safety or other quality criteria",
    "B15": "Service/procedure denied because it does not meet the threshold",
    "B16": "Services cannot be billed as 'New Patient'",
    "B20": "Procedure/Service not authorized/not covered for the date of service",
    "B22": "Payment adjusted because this procedure was included in a previous allowance",
    "B23": "Claim denied/reduced because this procedure was billed in excess of the maximum allowed",
    "P1": "State-mandated requirement",
    "P2": "Amount represents denial of a provider initiated discount",
    "P3": "Provider-loss adjustment",
    "P4": "Penalty for failure to submit this claim electronically",
    "P5": "Service was paid via Health Spending Account",
    "P6": "Previously approved treatment plan has been modified",
    "P7": "Authorization/certification reassignment not allowed",
    "P8": "Claim information was already adjudicated in a previous response",
    "P9": "Provider has already been paid for this service",
    "P10": "Patient ineligible for this service",
    "P11": "Claim payment is being processed through provider patient care reduction program",
    "P12": "Worker's compensation case has been settled",
    "P13": "Worker's compensation medical bill review reduction",
    "P14": "Pharmacy discount adjustment",
    "P15": "Value-Based Care Adjustment",
    "P16": "Payment withheld pending provider compliance with quality requirements",
    "P17": "Consumer Spending Account cost share",
    "P18": "Late charges adjustment",
    "P19": "Claim processed as attachment to prior claim",
    "P20": "Provider Contracted Managed Care adjustment",
    "P21": "Pharmacist Services Adjustment",
    "P22": "Claim paid through Electronic Funds Transfer (EFT)",
    "P23": "Fee Schedule Adjustment",
    "W1": "Worker's Compensation State Fee Schedule Adjustment",
    "W2": "Workers' Compensation Jurisdictional Guideline adjustment",
    "W3": "Workers' Compensation Medicare Set-Aside Arrangement (MSA) Adjustment",
    "W4": "Workers' Compensation Second Injury Fund adjustment",
    "Y1": "Claim was processed under a demonstration/pilot program (demonstration number)",
    "Y2": "Claim processed based on multiple payment policies/procedures",
    "Y3": "Claim processed based on the provider's contract with the payer",
}


def validate_carc_code(code: str) -> ValidationInfo:
    """
    Validate a CARC (Claim Adjustment Reason Code).

    CARC codes are alphanumeric codes (1-3 characters for the code portion,
    optionally prefixed with a group code like CO, PR, OA, PI, CR).
    They explain why a claim or service line was paid differently than billed.

    Format examples:
    - "45" (standalone code)
    - "CO-45" or "CO45" (with group code)
    - "PR3" (patient responsibility with code)

    Args:
        code: CARC code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_carc_code("45")
        ValidationInfo(result=VALID, message="Valid CARC code", ...)
        >>> validate_carc_code("CO-45")
        ValidationInfo(result=VALID, message="Valid CARC code with group CO", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="CARC code is required",
        )

    # Clean and normalize the code
    code_str = str(code).strip().upper()

    # Remove common separators
    code_str = re.sub(r"[-\s]+", "", code_str)

    # Pattern to match optional group code followed by 1-3 alphanumeric code
    carc_pattern = re.compile(r"^(CO|CR|OA|PI|PR)?([A-Z]?\d{1,3})$")
    match = carc_pattern.match(code_str)

    if not match:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Invalid CARC code format: {code}. Expected format: [CO|CR|OA|PI|PR]<code> or standalone code (1-3 alphanumeric)",
            normalized_value=code_str,
        )

    group_code = match.group(1)  # May be None
    adjustment_code = match.group(2)

    # Check if this is a known code
    description = COMMON_CARC_CODES.get(adjustment_code)

    # Normalize output format
    if group_code:
        normalized = f"{group_code}-{adjustment_code}"
    else:
        normalized = adjustment_code

    if description:
        return ValidationInfo(
            result=ValidationResult.VALID,
            message=f"Valid CARC code" + (f" ({group_code})" if group_code else "") + f": {description}",
            normalized_value=normalized,
            details={
                "group_code": group_code,
                "adjustment_code": adjustment_code,
                "description": description,
            },
        )
    else:
        # Code format is valid but not in our known list - may still be valid
        return ValidationInfo(
            result=ValidationResult.WARNING,
            message=f"CARC code format is valid but code {adjustment_code} is not in standard reference",
            normalized_value=normalized,
            details={
                "group_code": group_code,
                "adjustment_code": adjustment_code,
            },
        )


# =============================================================================
# RARC (Remittance Advice Remark Code) Validation
# =============================================================================

# RARC codes provide additional explanation for adjustments
# Alerts: Informational messages (MA codes)
# Modified: Provider actions (M codes)
# Supplemental: Additional detail (N codes)
RARC_PREFIXES = frozenset({"MA", "M", "N"})

# Common RARC codes with descriptions (subset for validation reference)
# Full list: https://x12.org/codes/remittance-advice-remark-codes
COMMON_RARC_CODES = {
    # Alert codes (MA)
    "MA01": "Alert: If you do not agree with this decision, you have the right to appeal",
    "MA02": "Alert: If you do not agree with this adjustment, you may appeal",
    "MA04": "Alert: Secondary coverage may exist. Contact the payer",
    "MA07": "Alert: The claim information has been forwarded to the appropriate state agency",
    "MA08": "Alert: If you do not agree with the approved amounts, you may appeal",
    "MA09": "Alert: Claim processed in accordance with a demonstration project or program",
    "MA10": "Alert: The patient's secondary insurer may cover this service",
    "MA12": "Alert: You may be subject to penalties if you bill the patient for this service",
    "MA13": "Alert: You may appeal this decision",
    "MA14": "Alert: The patient's record indicates that a primary payer may be responsible",
    "MA15": "Alert: Your claim has been separated to expedite payment",
    "MA17": "Alert: We did not receive your response to our development letter",
    "MA18": "Alert: We determined that the patient was enrolled in Medicare at the time of service",
    "MA19": "Alert: Information was received indicating the beneficiary may have primary coverage",
    "MA20": "Alert: A minimum of two Distinct Time service units are required for payment of add-on code",
    "MA21": "Alert: You may not appeal this decision",
    "MA23": "Alert: Information identifies this patient as covered by Medicare",
    "MA24": "Alert: Secondary Medicare coverage exists. Contact the payer",
    "MA25": "Alert: Appeal must be submitted with supporting documentation",
    "MA26": "Alert: The patient is covered by another primary payer",
    "MA27": "Alert: Resubmit this claim with the corrected patient information",
    "MA28": "Alert: This claim has been separated for processing",
    "MA29": "Alert: A mandatory National Drug Code (NDC) is required for this service",
    "MA30": "Alert: Missing/incomplete/invalid service date(s)",
    "MA31": "Alert: Missing/incomplete/invalid place of service",
    "MA32": "Alert: Missing/incomplete/invalid date of birth",
    "MA33": "Alert: This service was not recognized or inconsistent with the current coding guidelines",
    "MA34": "Alert: A supporting prior authorization/certification number is required",
    "MA35": "Alert: This service was not medically necessary according to the established criteria",
    "MA36": "Alert: Patient's eligibility information has been verified",
    "MA37": "Alert: Timely filing deadline has been reached",
    "MA39": "Alert: Our records indicate that the patient has other coverage that may be primary",
    "MA40": "Alert: The claim information is being forwarded to another payer for payment determination",
    "MA41": "Alert: Secondary EOB requested but no record of primary",
    "MA42": "Alert: Patient's eligibility could not be determined",
    "MA43": "Alert: Patient is responsible for the difference between the bill and Medicare approved",
    "MA44": "Alert: Claim has been adjusted based on a review of the current diagnosis codes",
    "MA45": "Alert: Service included in payment for another service/procedure",
    "MA46": "Alert: There are no remaining benefits for this service",
    "MA47": "Alert: Our records indicate the patient was enrolled in a Medicare HMO at the time",
    "MA48": "Alert: Missing/incomplete/invalid name of subscriber/insured",
    "MA49": "Alert: Missing/incomplete/invalid identifier of subscriber/insured",
    "MA50": "Alert: A referral/authorization is required for this service",
    "MA51": "Alert: This service was adjudicated by a utilization management organization",
    "MA52": "Alert: Resubmit with additional documentation",
    "MA53": "Alert: Missing/incomplete/invalid identifier of the responsible payer",
    "MA54": "Alert: Services are not covered by this plan",
    "MA55": "Alert: The patient has coverage through another carrier",
    "MA56": "Alert: Records indicate the patient is enrolled in a Medicare Advantage program",
    "MA57": "Alert: Payment made to the patient or subscriber",
    "MA58": "Alert: Services have been combined for payment purposes",
    "MA59": "Alert: Missing/incomplete/invalid patient identifier",
    "MA60": "Alert: Missing/incomplete/invalid insured's name",
    "MA61": "Alert: Missing/incomplete/invalid policyholder/subscriber name",
    "MA62": "Alert: Missing/incomplete/invalid referring provider identifier",
    "MA63": "Alert: Missing/incomplete/invalid supervising provider identifier",
    "MA64": "Alert: Missing/incomplete/invalid attending provider identifier",
    "MA65": "Alert: Date-of-service on this claim is inconsistent with our records",
    "MA66": "Alert: Missing/incomplete/invalid rendering provider identifier",
    "MA67": "Alert: This service was adjusted based on published coding guidelines",
    "MA68": "Alert: Missing/incomplete/invalid billing provider identifier",
    "MA69": "Alert: Missing/incomplete/invalid service provider primary identifier",
    "MA70": "Alert: Missing/incomplete/invalid patient's relationship to subscriber/insured",
    "MA71": "Alert: Contractual adjustment has been applied",
    "MA72": "Alert: Patient's coverage has been verified as inactive",
    "MA73": "Alert: Medical record(s) do not support submitted information",
    "MA74": "Alert: Missing/incomplete/invalid medical record documentation",
    "MA75": "Alert: Missing/incomplete/invalid medical record number",
    "MA76": "Alert: Missing/incomplete/invalid diagnosis code(s)",
    "MA77": "Alert: Missing/incomplete/invalid procedure code(s)",
    "MA78": "Alert: Missing/incomplete/invalid unit of service information",
    "MA79": "Alert: Missing/incomplete/invalid charge information",
    "MA80": "Alert: Service date is outside the period covered by this plan",
    "MA81": "Alert: Missing/incomplete/invalid prior authorization number",
    "MA82": "Alert: Prior authorization was denied for this service",
    "MA83": "Alert: Did not qualify for appeal because the appeal was not received timely",
    "MA84": "Alert: Appeal denied as prior authorization was not provided",
    "MA85": "Alert: Emergency/urgent care payment may be reduced",
    "MA86": "Alert: Service denied per Medicare guidelines",
    "MA87": "Alert: Claim denied based on response to medical record request",
    "MA88": "Alert: Claim denied. No authorization or contract",
    "MA89": "Alert: Service was not rendered by the provider described",
    "MA90": "Alert: Services denied by third party review organization",
    "MA91": "Alert: This service has been included in another service payment",
    "MA92": "Alert: This service does not qualify for separate reimbursement",
    "MA93": "Alert: Services rendered are non-covered per benefit plan",
    "MA94": "Alert: Services reported inconsistent with medical record documentation",
    "MA95": "Alert: Patient liability reduced for error in submitted claim",
    "MA96": "Alert: Patient has primary insurance coverage through another carrier",
    "MA97": "Alert: Patient not enrolled at time of service",
    "MA98": "Alert: Services reported not covered per benefit plan",
    "MA99": "Alert: Medical records submitted do not support the service(s) rendered",
    "MA100": "Alert: Missing/incomplete/invalid insurance identification number",
    "MA101": "Alert: Missing/incomplete/invalid patient account number",
    "MA102": "Alert: Missing/incomplete/invalid diagnosis pointer",
    "MA103": "Alert: Diagnosis code(s) do not support medical necessity",
    "MA104": "Alert: Modifier(s) do not support medical necessity",
    "MA105": "Alert: Missing/incomplete/invalid clinical lab improvement amendment number",
    "MA106": "Alert: The patient has Medicare coverage effective on this date of service",
    "MA107": "Alert: The patient was not enrolled in Medicare at the time of service",
    "MA108": "Alert: Services must be provided and billed by a participating provider",
    "MA109": "Alert: Services must be provided and billed by an approved provider",
    "MA110": "Alert: Services must be provided by an approved facility",
    "MA111": "Alert: Laboratory services must be provided by a certified lab",
    "MA112": "Alert: DME must be provided by a certified supplier",
    "MA113": "Alert: This service was reviewed and approved for payment",
    "MA114": "Alert: This service was reviewed and denied for payment",
    "MA115": "Alert: Appeal rights have been exhausted",
    "MA116": "Alert: Patient responsibility reduced due to provider write-off",
    "MA117": "Alert: Patient not responsible due to provider network status",
    "MA118": "Alert: Claim submitted after deadline - resubmit with documentation",
    "MA119": "Alert: Patient not eligible for this benefit",
    "MA120": "Alert: Service was denied because it was not medically necessary",
    "MA121": "Alert: Service was denied because it exceeds frequency limitations",
    "MA122": "Alert: Service was denied because prior authorization was not obtained",
    "MA123": "Alert: Resubmit with additional clinical documentation",
    "MA124": "Alert: Service exceeds maximum allowable by contract",
    "MA125": "Alert: Service code inconsistent with place of service",
    "MA126": "Alert: Primary payer payment is pending",
    "MA127": "Alert: Claim forwarded to correct payer for adjudication",
    "MA128": "Alert: Claim forwarded to state Medicaid for consideration",
    "MA130": "Alert: Your claim contains incomplete/invalid documentation",
    "MA131": "Alert: Medicare records do not reflect patient's current address",
    "MA132": "Alert: Provider's appeal rights have been exhausted",
    "MA133": "Alert: Medical records do not support level of service billed",
    "MA134": "Alert: Timely filing documentation is required for appeal",
    # Supplemental codes (N)
    "N1": "Alert: You may appeal this decision",
    "N2": "This service/report may be separately payable under certain circumstances",
    "N3": "Missing or incomplete/invalid prior medical records",
    "N4": "Missing or incomplete/invalid medical documentation",
    "N5": "Service requiring prior authorization was not authorized",
    "N6": "Additional documentation is required",
    "N7": "Unit(s) of service exceeds maximum allowed",
    "N8": "Service is unrelated to the condition for which hospitalization occurred",
    "N9": "Appeal procedures not followed or time limits not met",
    "N10": "Payment based on contractual amount",
    "N11": "Resubmit claim with supporting documentation",
    "N12": "Claim has been denied/reduced because the documentation does not support medical necessity",
    "N13": "Payment denied because authorization was not obtained",
    "N14": "Payment denied because prior certification was not obtained",
    "N15": "Missing or incomplete/invalid authorization number",
    "N16": "Missing or incomplete/invalid prior authorization",
    "N17": "Claim denied because documentation does not support the medical necessity",
    "N19": "Procedure code incidental to primary procedure",
    "N20": "Service not billable to this payer",
    "N21": "Missing or incomplete/invalid medical necessity documentation",
    "N22": "Missing or incomplete/invalid original claim information",
    "N23": "The time frame for filing a claim has expired",
    "N24": "Claim submitted more than once for the same service/date",
    "N25": "Claim denied as a duplicate of a claim previously processed",
    "N26": "Please resubmit claim with correct billing code(s)",
    "N27": "Claim denied/reduced because service is included in another procedure",
    "N28": "Claim denied. Claim lacks required documentation",
    "N29": "Service denied. Missing/incomplete/invalid clinical notes",
    "N30": "Claim denied due to failure to follow appeal procedures",
    "N31": "Claim denied/reduced due to the absence of prior authorization",
    "N32": "Claim denied because of diagnosis",
    "N33": "Claim denied because of a missing diagnosis code or pointer",
    "N34": "Claim denied because of a missing diagnosis",
    "N35": "Payment based on diagnosis",
    "N36": "Payment adjusted based on payer determined negotiated fee",
    "N37": "Service denied because it exceeds frequency limits",
    "N38": "Service denied because claim lacks supporting documentation",
    "N39": "Service denied because the service does not meet coverage guidelines",
    "N40": "Claim denied pending additional information",
    "N41": "Claim denied pending documentation",
    "N42": "No maximum allowable amount defined for this service/procedure",
    "N43": "Payment adjusted because charges exceed fee schedule",
    "N44": "Payment denied because this procedure/service is not covered under patient's plan",
    "N45": "Payment denied. Missing/incomplete/invalid attachment",
    "N46": "Service denied because it does not meet criteria",
    "N47": "These are non-covered services because this is not deemed a medical necessity",
    "N48": "Payment denied because documentation was not received timely",
    "N49": "Appeal denied. Missing/incomplete/invalid documentation",
    "N50": "Missing/incomplete/invalid medical records",
    "N51": "Service denied because electronic claim not received",
    "N52": "No additional payment allowed",
    "N53": "Payment adjusted based on carrier-determined fee schedule",
    "N54": "Service denied. Prior authorization was denied",
    "N55": "Service denied. Service exceeds limitations",
    "N56": "Service denied because of frequency limitations",
    "N57": "Documentation submitted does not support amount billed",
    "N58": "Service not separately payable",
    "N59": "Payment denied. Missing/incomplete/invalid clinical record",
    "N60": "Claim denied due to failure to submit documentation timely",
    "N61": "Resubmit claim with corrected prior authorization number",
    "N62": "Payment denied. This procedure code is not valid",
    "N63": "Missing/incomplete/invalid service dates",
    "N64": "Missing/incomplete/invalid charge amount",
    "N65": "This procedure code must be billed with modifiers",
    "N66": "Missing/incomplete/invalid NDC code",
    "N67": "Claim denied. Service not authorized for date of service",
    "N68": "Missing/incomplete/invalid supporting documentation",
    "N69": "Claim denied due to missing/incomplete/invalid clinical record",
    "N70": "Payment denied. Missing/incomplete/invalid physician's order",
    "N71": "Resubmit this claim with medical records",
    "N72": "Claim denied for missing/incomplete/invalid clinical notes",
    "N73": "Service denied. Submitted documentation inadequate",
    "N74": "Claim denied pending medical records review",
    "N75": "Claim suspended pending receipt of additional documentation",
    "N76": "Missing/incomplete/invalid procedure information",
    "N77": "Missing/incomplete/invalid provider information",
    "N78": "Missing/incomplete/invalid patient information",
    "N79": "Missing/incomplete/invalid subscriber information",
    "N80": "Missing/incomplete/invalid insurance information",
    "N81": "Claim denied. Additional documentation is required",
    "N82": "Missing/incomplete/invalid supporting clinical documentation",
    "N83": "Service denied because service exceeds allowable",
    "N84": "Claim not submitted timely",
    "N85": "Claim submitted after deadline. Please provide documentation",
    "N86": "Claim denied. Missing/incomplete/invalid supporting documentation",
    "N87": "Payment denied. Missing/incomplete prior payment",
    "N88": "Payment denied because claim was submitted to the wrong payer",
    "N89": "Claim submitted does not match our records",
    "N90": "Service denied because the diagnosis code does not support",
    "N91": "Services inconsistent with medical history",
    "N92": "This service was denied based on medical review",
    "N93": "Payment was denied based on diagnosis coding",
    "N94": "Claim requires the unique physician identifier",
    "N95": "This service must be submitted with required documentation",
    "N96": "Claim denied because service was previously processed",
    "N97": "Claim denied because the documentation does not support",
    "N98": "Claim denied. Claim submission was not received",
    "N99": "Documentation submitted does not support",
    "N100": "Claim denied. Provider must submit all charges on one claim",
    "N101": "Please resubmit with patient demographic information",
    "N102": "Payment denied. This service was already paid under a previous claim",
    "N103": "Claim denied pending payer review",
    "N104": "Please submit corrected claim with additional documentation",
    "N105": "Service denied. Additional information is required",
    "N106": "Payment denied because required documentation was not received",
    "N107": "Claim denied pending Medicare review",
    "N108": "Service denied. Supporting documentation required",
    "N109": "Claim denied. Clinical documentation required",
    "N110": "Missing/incomplete/invalid prior authorization information",
    "N111": "Resubmission of denied claim requires appeals process",
    "N112": "Claim denied based on policy guidelines",
    "N113": "Service not covered under subscriber's current plan",
    "N114": "Service exceeds maximum benefit allowed",
    "N115": "Service excluded from coverage per plan terms",
    "N116": "Payment has been adjusted based on UCR fee schedule",
    "N117": "Claim requires supporting clinical documentation for review",
    "N118": "Service denied. Prior authorization documentation required",
    "N119": "Additional clinical information required for medical necessity review",
    "N120": "Claim denied. Supporting documentation must be submitted with resubmission",
    "N121": "Service denied. Documentation does not support frequency billed",
    "N122": "Payment denied. Claim requires additional supporting documents",
    # Modified codes (M)
    "M1": "Missing/incomplete/invalid X-ray film(s) or image(s)",
    "M2": "Service denied because the certification was not timely signed",
    "M3": "Missing/incomplete/invalid attending physician information",
    "M4": "Missing/incomplete/invalid referring provider identifier",
    "M5": "Missing/incomplete/invalid rendering provider information",
    "M6": "Missing/incomplete/invalid billing provider identifier",
    "M7": "Missing/incomplete/invalid place of service",
    "M8": "Missing/incomplete/invalid provider signature",
    "M9": "Missing/incomplete/invalid patient's name",
    "M10": "Missing/incomplete/invalid patient's address",
    "M11": "Missing/incomplete/invalid patient's date of birth",
    "M12": "Missing/incomplete/invalid patient's sex/gender",
    "M13": "Missing/incomplete/invalid patient relationship to subscriber",
    "M14": "Missing/incomplete/invalid subscriber's name",
    "M15": "Missing/incomplete/invalid claim information",
    "M16": "Claim denied. Records do not support documentation submitted",
    "M17": "Missing/incomplete/invalid condition information",
    "M18": "Missing/incomplete/invalid diagnosis coding information",
    "M19": "Missing/incomplete/invalid service unit information",
    "M20": "Missing/incomplete/invalid National Drug Code (NDC)",
    "M21": "Missing/incomplete/invalid documentation to support billed service",
    "M22": "Missing/incomplete/invalid revenue code for inpatient claim",
    "M23": "Missing/incomplete/invalid NDC unit of measurement",
    "M24": "Missing/incomplete/invalid NDC quantity",
    "M25": "Missing/incomplete/invalid number of anesthesia units",
    "M26": "Missing/incomplete/invalid patient's weight",
    "M27": "Missing/incomplete/invalid treatment authorization number",
    "M28": "Claim denied. Prior approval number invalid",
    "M29": "Missing/incomplete/invalid place of service description",
    "M30": "Missing/incomplete/invalid anesthesia physical status",
    "M31": "Missing/incomplete/invalid prescription number",
    "M32": "Missing/incomplete/invalid prescriber identifier",
    "M33": "Missing/incomplete/invalid dispenser identifier",
    "M34": "Missing/incomplete/invalid pharmacy prescription information",
    "M35": "Missing/incomplete/invalid accident information",
    "M36": "Missing/incomplete/invalid patient's insurance information",
    "M37": "Missing/incomplete/invalid other insurance information",
    "M38": "Missing/incomplete/invalid subscriber/insured identifier",
    "M39": "Missing/incomplete/invalid group number",
    "M40": "Missing/incomplete/invalid insured's date of birth",
    "M41": "Missing/incomplete/invalid insured's sex/gender",
    "M42": "Missing/incomplete/invalid facility information",
    "M43": "Missing/incomplete/invalid service facility location",
    "M44": "Missing/incomplete/invalid service facility NPI",
    "M45": "Missing/incomplete/invalid hospice number",
    "M46": "Missing/incomplete/invalid ambulance pickup location",
    "M47": "Missing/incomplete/invalid ambulance dropoff location",
    "M48": "Missing/incomplete/invalid ambulance certification",
    "M49": "Missing/incomplete/invalid patient's condition information",
    "M50": "Missing/incomplete/invalid supply information",
    "M51": "Missing/incomplete/invalid surgical procedure information",
    "M52": "Missing/incomplete/invalid medical necessity clinical information",
    "M53": "Missing/incomplete/invalid hemoglobin or hematocrit value",
    "M54": "Missing/incomplete/invalid principal diagnosis",
    "M55": "Missing/incomplete/invalid secondary diagnosis",
    "M56": "Missing/incomplete/invalid procedure code",
    "M57": "Missing/incomplete/invalid procedure date",
    "M58": "Missing/incomplete/invalid revenue code",
    "M59": "Missing/incomplete/invalid type of bill",
    "M60": "Missing/incomplete/invalid statement covers period",
    "M61": "Missing/incomplete/invalid occurrence information",
    "M62": "Missing/incomplete/invalid patient discharge status",
    "M63": "Missing/incomplete/invalid admission source",
    "M64": "Missing/incomplete/invalid admission type",
    "M65": "Missing/incomplete/invalid point of origin for admission",
    "M66": "Missing/incomplete/invalid emergency services indicator",
    "M67": "Missing/incomplete/invalid admit diagnosis",
    "M68": "Missing/incomplete/invalid principal procedure information",
    "M69": "Missing/incomplete/invalid other procedure information",
    "M70": "Missing/incomplete/invalid attending provider information",
    "M71": "Missing/incomplete/invalid operating physician information",
    "M72": "Missing/incomplete/invalid other provider information",
    "M73": "Missing/incomplete/invalid claim note information",
    "M74": "Missing/incomplete/invalid prior approval certificate",
    "M75": "Missing/incomplete/invalid homebound status information",
    "M76": "Missing/incomplete/invalid DMERC CMN/DIF information",
    "M77": "Missing/incomplete/invalid lab results",
    "M78": "Missing/incomplete/invalid oxygen certification information",
    "M79": "Missing/incomplete/invalid functional limitations documentation",
    "M80": "Missing/incomplete/invalid reason for not billing Medicare first",
    "M81": "Missing/incomplete/invalid provider taxonomy code",
    "M82": "Missing/incomplete/invalid service authorization exception code",
    "M83": "Missing/incomplete/invalid date of prior service",
    "M84": "Missing/incomplete/invalid date of last menstrual period",
    "M85": "Missing/incomplete/invalid pregnancy indicator",
    "M86": "Missing/incomplete/invalid newborn information",
    "M87": "Missing/incomplete/invalid COB information",
    "M88": "Missing/incomplete/invalid other subscriber information",
    "M89": "Missing/incomplete/invalid related claim information",
    "M90": "Missing/incomplete/invalid delayed reason code",
    "M91": "Missing/incomplete/invalid demonstration identifier",
    "M92": "Missing/incomplete/invalid mammography certification number",
    "M93": "Missing/incomplete/invalid CLIA number",
    "M94": "Missing/incomplete/invalid investigational device exemption",
    "M95": "Missing/incomplete/invalid claim frequency code",
    "M96": "Missing/incomplete/invalid patient paid amount",
    "M97": "Missing/incomplete/invalid provider characteristic information",
    "M98": "Missing/incomplete/invalid contract type code",
    "M99": "Missing/incomplete/invalid Medicaid-issued provider identifier",
    "M100": "Missing/incomplete/invalid provider pin number",
    "M101": "Missing/incomplete/invalid provider site address",
    "M102": "Missing/incomplete/invalid claim level adjustment",
    "M103": "Missing/incomplete/invalid service level adjustment",
    "M104": "Missing/incomplete/invalid claim information",
}


def validate_rarc_code(code: str) -> ValidationInfo:
    """
    Validate a RARC (Remittance Advice Remark Code).

    RARC codes provide additional explanation for claim adjustments.
    They supplement CARC codes with more specific information.

    Format examples:
    - "MA01" (Alert code - Medicare specific)
    - "N1" (Supplemental code)
    - "M1" (Modified code)

    Code categories:
    - MA codes: Medicare Alert codes (informational)
    - M codes: Modified codes (provider action needed)
    - N codes: Supplemental codes (additional detail)

    Args:
        code: RARC code to validate.

    Returns:
        ValidationInfo with result and details.

    Example:
        >>> validate_rarc_code("MA01")
        ValidationInfo(result=VALID, message="Valid RARC code (Alert)", ...)
        >>> validate_rarc_code("N1")
        ValidationInfo(result=VALID, message="Valid RARC code (Supplemental)", ...)
    """
    if code is None:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message="RARC code is required",
        )

    # Clean and normalize the code
    code_str = str(code).strip().upper()

    # Remove any separators
    code_str = re.sub(r"[-\s]+", "", code_str)

    # Pattern: MA followed by digits, or M/N followed by digits
    rarc_pattern = re.compile(r"^(MA|M|N)(\d{1,3})$")
    match = rarc_pattern.match(code_str)

    if not match:
        return ValidationInfo(
            result=ValidationResult.INVALID,
            message=f"Invalid RARC code format: {code}. Expected format: MA<num>, M<num>, or N<num>",
            normalized_value=code_str,
        )

    prefix = match.group(1)
    number = match.group(2)
    normalized = f"{prefix}{number}"

    # Determine category
    if prefix == "MA":
        category = "Alert"
    elif prefix == "M":
        category = "Modified"
    else:  # N
        category = "Supplemental"

    # Check if this is a known code
    description = COMMON_RARC_CODES.get(normalized)

    if description:
        return ValidationInfo(
            result=ValidationResult.VALID,
            message=f"Valid RARC code ({category}): {description}",
            normalized_value=normalized,
            details={
                "prefix": prefix,
                "number": number,
                "category": category,
                "description": description,
            },
        )
    else:
        # Code format is valid but not in our known list - may still be valid
        return ValidationInfo(
            result=ValidationResult.WARNING,
            message=f"RARC code format is valid but code {normalized} is not in standard reference",
            normalized_value=normalized,
            details={
                "prefix": prefix,
                "number": number,
                "category": category,
            },
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
        FieldType.HCPCS_CODE: validate_hcpcs_code,
        FieldType.NDC_CODE: validate_ndc_code,
        FieldType.TAXONOMY_CODE: validate_taxonomy_code,
        FieldType.CARC_CODE: validate_carc_code,
        FieldType.RARC_CODE: validate_rarc_code,
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

    Provides validation for CPT, ICD-10, NPI, HCPCS, NDC, CARC, and RARC codes
    with caching for performance.

    Example:
        validator = MedicalCodeValidator()

        result = validator.validate_cpt("99213")
        if result.is_valid:
            print(f"Valid: {result.normalized_value}")

        # Validate CARC/RARC codes from EOB
        carc_result = validator.validate_carc("CO-45")
        rarc_result = validator.validate_rarc("MA01")
    """

    def __init__(self) -> None:
        """Initialize validator with caches."""
        self._cpt_cache: dict[str, ValidationInfo] = {}
        self._icd10_cache: dict[str, ValidationInfo] = {}
        self._npi_cache: dict[str, ValidationInfo] = {}
        self._carc_cache: dict[str, ValidationInfo] = {}
        self._rarc_cache: dict[str, ValidationInfo] = {}

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

    def validate_carc(self, code: str) -> ValidationInfo:
        """
        Validate CARC (Claim Adjustment Reason Code) with caching.

        CARC codes explain why a claim was paid differently than billed.
        Common examples: CO-45 (exceeds fee schedule), PR-1 (deductible).

        Args:
            code: CARC code to validate (e.g., "45", "CO-45", "PR1").

        Returns:
            ValidationInfo with result and code details.
        """
        if code in self._carc_cache:
            return self._carc_cache[code]

        result = validate_carc_code(code)
        self._carc_cache[code] = result
        return result

    def validate_rarc(self, code: str) -> ValidationInfo:
        """
        Validate RARC (Remittance Advice Remark Code) with caching.

        RARC codes provide additional explanation for claim adjustments.
        Categories: MA (Alert), M (Modified), N (Supplemental).

        Args:
            code: RARC code to validate (e.g., "MA01", "N1", "M15").

        Returns:
            ValidationInfo with result and code details.
        """
        if code in self._rarc_cache:
            return self._rarc_cache[code]

        result = validate_rarc_code(code)
        self._rarc_cache[code] = result
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
            code_type: Type of codes (cpt, icd10, npi, carc, rarc).

        Returns:
            List of ValidationInfo results.
        """
        validators = {
            "cpt": self.validate_cpt,
            "icd10": self.validate_icd10,
            "npi": self.validate_npi,
            "carc": self.validate_carc,
            "rarc": self.validate_rarc,
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
        self._carc_cache.clear()
        self._rarc_cache.clear()
