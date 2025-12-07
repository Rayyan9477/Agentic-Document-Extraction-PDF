"""
Medical code validation for document extraction.

Provides comprehensive validation for healthcare-specific codes:
- CPT (Current Procedural Terminology) codes
- ICD-10-CM/PCS diagnosis and procedure codes
- NPI (National Provider Identifier)
- HCPCS codes
- NDC (National Drug Code)

Integrates with the existing schemas.validators module while providing
additional batch validation, code relationship checking, and detailed
reporting capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

from src.config import get_logger
from src.schemas.validators import (
    validate_cpt_code,
    validate_icd10_code,
    validate_npi,
    MedicalCodeValidator as SchemaValidator,
    ValidationInfo,
    ValidationResult as ValidatorResult,
)


logger = get_logger(__name__)


class CodeType(str, Enum):
    """Types of medical codes."""

    CPT = "cpt"
    ICD10_CM = "icd10_cm"
    ICD10_PCS = "icd10_pcs"
    NPI = "npi"
    HCPCS = "hcpcs"
    NDC = "ndc"
    PLACE_OF_SERVICE = "pos"
    TYPE_OF_SERVICE = "tos"
    MODIFIER = "modifier"


class CodeValidationStatus(str, Enum):
    """Status of code validation."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CodeValidationDetail:
    """
    Detailed validation result for a single code.

    Attributes:
        code: The code that was validated.
        code_type: Type of medical code.
        status: Validation status.
        message: Human-readable validation message.
        normalized_code: Normalized/formatted code.
        details: Additional validation details.
        confidence: Confidence in validation 0.0-1.0.
    """

    code: str
    code_type: CodeType
    status: CodeValidationStatus
    message: str
    normalized_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "code_type": self.code_type.value,
            "status": self.status.value,
            "message": self.message,
            "normalized_code": self.normalized_code,
            "details": dict(self.details) if self.details else {},
            "confidence": self.confidence,
        }

    @property
    def is_valid(self) -> bool:
        """Check if code is valid."""
        return self.status in (CodeValidationStatus.VALID, CodeValidationStatus.WARNING)


@dataclass(slots=True)
class MedicalCodeValidationResult:
    """
    Complete validation result for all medical codes in an extraction.

    Attributes:
        validations: List of individual code validations.
        valid_codes: Codes that passed validation.
        invalid_codes: Codes that failed validation.
        warning_codes: Codes with warnings.
        by_type: Validations grouped by code type.
        overall_valid: Whether all codes are valid.
        validation_rate: Percentage of codes that passed.
    """

    validations: list[CodeValidationDetail] = field(default_factory=list)
    valid_codes: list[str] = field(default_factory=list)
    invalid_codes: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    by_type: dict[str, list[CodeValidationDetail]] = field(default_factory=dict)
    overall_valid: bool = True
    validation_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validations": [v.to_dict() for v in self.validations],
            "valid_codes": self.valid_codes,
            "invalid_codes": self.invalid_codes,
            "warning_codes": self.warning_codes,
            "by_type": {
                k: [v.to_dict() for v in vals]
                for k, vals in self.by_type.items()
            },
            "overall_valid": self.overall_valid,
            "validation_rate": self.validation_rate,
        }


class MedicalCodeValidationEngine:
    """
    Comprehensive medical code validation engine.

    Validates medical codes from extraction results with support for:
    - Batch validation of multiple codes
    - Code type detection
    - Relationship validation (e.g., CPT-modifier pairs)
    - Detailed reporting

    Example:
        engine = MedicalCodeValidationEngine()
        result = engine.validate_all(extracted_data)

        if not result.overall_valid:
            for code in result.invalid_codes:
                print(f"Invalid: {code}")
    """

    # HCPCS code pattern (5 alphanumeric)
    HCPCS_PATTERN = r"^[A-Z][0-9]{4}$"

    # NDC patterns (various formats)
    NDC_PATTERNS = [
        r"^\d{5}-\d{4}-\d{2}$",
        r"^\d{5}-\d{3}-\d{2}$",
        r"^\d{4}-\d{4}-\d{2}$",
        r"^\d{11}$",
    ]

    # Place of Service codes
    VALID_POS_CODES = {
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "31", "32", "33", "34",
        "41", "42", "49", "50", "51", "52", "53", "54", "55", "56",
        "57", "58", "60", "61", "62", "65", "71", "72", "81", "99",
    }

    # CPT Modifier codes (common ones)
    VALID_MODIFIERS = {
        "22", "23", "24", "25", "26", "27", "32", "33", "47", "50",
        "51", "52", "53", "54", "55", "56", "57", "58", "59", "62",
        "63", "66", "73", "74", "76", "77", "78", "79", "80", "81",
        "82", "90", "91", "92", "93", "95", "96", "97", "99",
        "AA", "AD", "AM", "AS", "AT", "AU", "AX", "AY", "AZ",
        "E1", "E2", "E3", "E4", "FA", "F1", "F2", "F3", "F4", "F5",
        "F6", "F7", "F8", "F9", "GA", "GC", "GE", "GG", "GH", "GJ",
        "GK", "GL", "GM", "GN", "GO", "GP", "GQ", "GR", "GS", "GT",
        "GU", "GV", "GW", "GX", "GY", "GZ", "HA", "HB", "HC", "HD",
        "HE", "HF", "HG", "HH", "HI", "HJ", "HK", "HL", "HM", "HN",
        "HO", "HP", "HQ", "HR", "HS", "HT", "HU", "HV", "HW", "HX",
        "HY", "HZ", "JA", "JB", "JC", "JD", "JE", "JF", "JG", "JW",
        "K0", "K1", "K2", "K3", "K4", "KA", "KB", "KC", "KD", "KE",
        "KF", "KG", "KH", "KI", "KJ", "KK", "KL", "KM", "KN", "KO",
        "KP", "KQ", "KR", "KS", "KT", "KU", "KV", "KW", "KX", "KY",
        "KZ", "LC", "LD", "LR", "LS", "LT", "P1", "P2", "P3", "P4",
        "P5", "P6", "PA", "PB", "PC", "PD", "PI", "PL", "PM", "PN",
        "PO", "PS", "PT", "Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6",
        "QA", "QB", "QC", "QD", "QE", "QF", "QG", "QH", "QJ", "QK",
        "QL", "QM", "QN", "QP", "QQ", "QR", "QS", "QT", "QW", "QX",
        "QY", "QZ", "RA", "RB", "RC", "RD", "RE", "RI", "RR", "RT",
        "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SJ", "SK",
        "SL", "SM", "SN", "SQ", "SS", "ST", "SU", "SV", "SW", "SY",
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA",
        "TB", "TC", "TD", "TE", "TF", "TG", "TH", "TJ", "TK", "TL",
        "TM", "TN", "TP", "TQ", "TR", "TS", "TT", "TU", "TV", "TW",
        "UA", "UB", "UC", "UD", "UE", "UF", "UG", "UH", "UI", "UJ",
        "UK", "UN", "UP", "UQ", "UR", "US", "VP", "XE", "XP", "XS",
        "XU", "ZA", "ZB", "ZC",
    }

    # Class-level validation cache (shared across instances for efficiency)
    _validation_cache: dict[tuple[str, str], CodeValidationDetail] = {}
    _cache_max_size: int = 1000

    def __init__(self, cache_enabled: bool = True) -> None:
        """
        Initialize the validation engine.

        Args:
            cache_enabled: Whether to cache validation results.
        """
        self.cache_enabled = cache_enabled
        self._schema_validator = SchemaValidator()

        # Compile patterns
        import re
        self._hcpcs_pattern = re.compile(self.HCPCS_PATTERN)
        self._ndc_patterns = [re.compile(p) for p in self.NDC_PATTERNS]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the validation cache."""
        cls._validation_cache.clear()

    def _get_cached_validation(
        self, code: str, code_type: CodeType
    ) -> CodeValidationDetail | None:
        """Get cached validation result if available."""
        if not self.cache_enabled:
            return None
        cache_key = (code, code_type.value)
        return self._validation_cache.get(cache_key)

    def _cache_validation(
        self, code: str, code_type: CodeType, result: CodeValidationDetail
    ) -> None:
        """Cache a validation result."""
        if not self.cache_enabled:
            return
        # Limit cache size to prevent memory issues
        if len(self._validation_cache) >= self._cache_max_size:
            # Remove oldest entries (first 10%)
            keys_to_remove = list(self._validation_cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                self._validation_cache.pop(key, None)
        cache_key = (code, code_type.value)
        self._validation_cache[cache_key] = result

    def validate_all(
        self,
        extracted_data: dict[str, Any],
        code_field_mapping: dict[str, CodeType] | None = None,
    ) -> MedicalCodeValidationResult:
        """
        Validate all medical codes in extracted data.

        Args:
            extracted_data: Dictionary of field names to values.
            code_field_mapping: Mapping of field names to code types.

        Returns:
            MedicalCodeValidationResult with all validation details.
        """
        if code_field_mapping is None:
            code_field_mapping = self._infer_code_types(extracted_data)

        result = MedicalCodeValidationResult()

        for field_name, code_type in code_field_mapping.items():
            value = extracted_data.get(field_name)
            if value is None:
                continue

            # Handle list of codes
            if isinstance(value, list):
                for code in value:
                    if code:
                        detail = self.validate_code(str(code), code_type)
                        result.validations.append(detail)
            else:
                detail = self.validate_code(str(value), code_type)
                result.validations.append(detail)

        # Build result
        self._build_result(result)

        logger.debug(
            f"Medical code validation: "
            f"total={len(result.validations)}, "
            f"valid={len(result.valid_codes)}, "
            f"invalid={len(result.invalid_codes)}"
        )

        return result

    def validate_code(
        self,
        code: str,
        code_type: CodeType,
    ) -> CodeValidationDetail:
        """
        Validate a single medical code with caching.

        Args:
            code: The code to validate.
            code_type: Type of the code.

        Returns:
            CodeValidationDetail with validation result.
        """
        # Check cache first
        cached = self._get_cached_validation(code, code_type)
        if cached is not None:
            return cached

        validators = {
            CodeType.CPT: self._validate_cpt,
            CodeType.ICD10_CM: self._validate_icd10,
            CodeType.ICD10_PCS: self._validate_icd10,
            CodeType.NPI: self._validate_npi,
            CodeType.HCPCS: self._validate_hcpcs,
            CodeType.NDC: self._validate_ndc,
            CodeType.PLACE_OF_SERVICE: self._validate_pos,
            CodeType.MODIFIER: self._validate_modifier,
        }

        validator = validators.get(code_type)
        if validator:
            result = validator(code)
            self._cache_validation(code, code_type, result)
            return result

        # Unknown type (not cached)
        return CodeValidationDetail(
            code=code,
            code_type=code_type,
            status=CodeValidationStatus.UNKNOWN,
            message=f"Unknown code type: {code_type.value}",
            confidence=0.5,
        )

    def validate_code_pair(
        self,
        cpt_code: str,
        modifier: str,
    ) -> tuple[bool, str]:
        """
        Validate CPT code and modifier pair.

        Args:
            cpt_code: The CPT code.
            modifier: The modifier.

        Returns:
            Tuple of (is_valid, message).
        """
        cpt_result = self._validate_cpt(cpt_code)
        mod_result = self._validate_modifier(modifier)

        if not cpt_result.is_valid:
            return False, f"Invalid CPT code: {cpt_result.message}"

        if not mod_result.is_valid:
            return False, f"Invalid modifier: {mod_result.message}"

        # Both valid
        return True, "Valid CPT-modifier pair"

    def _validate_cpt(self, code: str) -> CodeValidationDetail:
        """Validate CPT code using schema validator."""
        info = self._schema_validator.validate_cpt(code)
        return self._convert_validation_info(code, CodeType.CPT, info)

    def _validate_icd10(self, code: str) -> CodeValidationDetail:
        """Validate ICD-10 code using schema validator."""
        info = self._schema_validator.validate_icd10(code)

        # Determine specific type
        code_type = CodeType.ICD10_CM
        if info.details and info.details.get("type") == "ICD-10-PCS":
            code_type = CodeType.ICD10_PCS

        return self._convert_validation_info(code, code_type, info)

    def _validate_npi(self, code: str) -> CodeValidationDetail:
        """Validate NPI using schema validator."""
        info = self._schema_validator.validate_npi(code)
        return self._convert_validation_info(code, CodeType.NPI, info)

    def _validate_hcpcs(self, code: str) -> CodeValidationDetail:
        """Validate HCPCS code."""
        code_clean = code.strip().upper()

        if self._hcpcs_pattern.match(code_clean):
            return CodeValidationDetail(
                code=code,
                code_type=CodeType.HCPCS,
                status=CodeValidationStatus.VALID,
                message="Valid HCPCS code format",
                normalized_code=code_clean,
                confidence=0.90,
            )

        return CodeValidationDetail(
            code=code,
            code_type=CodeType.HCPCS,
            status=CodeValidationStatus.INVALID,
            message="Invalid HCPCS code format. Expected: letter + 4 digits",
            confidence=0.95,
        )

    def _validate_ndc(self, code: str) -> CodeValidationDetail:
        """Validate NDC (National Drug Code)."""
        code_clean = code.strip()

        for pattern in self._ndc_patterns:
            if pattern.match(code_clean):
                return CodeValidationDetail(
                    code=code,
                    code_type=CodeType.NDC,
                    status=CodeValidationStatus.VALID,
                    message="Valid NDC format",
                    normalized_code=code_clean,
                    confidence=0.90,
                )

        return CodeValidationDetail(
            code=code,
            code_type=CodeType.NDC,
            status=CodeValidationStatus.INVALID,
            message="Invalid NDC format",
            confidence=0.90,
        )

    def _validate_pos(self, code: str) -> CodeValidationDetail:
        """Validate Place of Service code."""
        code_clean = code.strip().zfill(2)

        if code_clean in self.VALID_POS_CODES:
            return CodeValidationDetail(
                code=code,
                code_type=CodeType.PLACE_OF_SERVICE,
                status=CodeValidationStatus.VALID,
                message="Valid Place of Service code",
                normalized_code=code_clean,
                confidence=1.0,
            )

        return CodeValidationDetail(
            code=code,
            code_type=CodeType.PLACE_OF_SERVICE,
            status=CodeValidationStatus.INVALID,
            message=f"Invalid Place of Service code: {code}",
            confidence=0.95,
        )

    def _validate_modifier(self, code: str) -> CodeValidationDetail:
        """Validate CPT modifier."""
        code_clean = code.strip().upper()

        if code_clean in self.VALID_MODIFIERS:
            return CodeValidationDetail(
                code=code,
                code_type=CodeType.MODIFIER,
                status=CodeValidationStatus.VALID,
                message="Valid modifier",
                normalized_code=code_clean,
                confidence=1.0,
            )

        # Check format even if not in known list
        import re
        if re.match(r"^[A-Z0-9]{2}$", code_clean):
            return CodeValidationDetail(
                code=code,
                code_type=CodeType.MODIFIER,
                status=CodeValidationStatus.WARNING,
                message="Modifier format valid but not in standard list",
                normalized_code=code_clean,
                confidence=0.70,
            )

        return CodeValidationDetail(
            code=code,
            code_type=CodeType.MODIFIER,
            status=CodeValidationStatus.INVALID,
            message=f"Invalid modifier format: {code}",
            confidence=0.95,
        )

    def _convert_validation_info(
        self,
        code: str,
        code_type: CodeType,
        info: ValidationInfo,
    ) -> CodeValidationDetail:
        """Convert ValidationInfo to CodeValidationDetail."""
        status_map = {
            ValidatorResult.VALID: CodeValidationStatus.VALID,
            ValidatorResult.INVALID: CodeValidationStatus.INVALID,
            ValidatorResult.WARNING: CodeValidationStatus.WARNING,
            ValidatorResult.UNKNOWN: CodeValidationStatus.UNKNOWN,
        }

        return CodeValidationDetail(
            code=code,
            code_type=code_type,
            status=status_map.get(info.result, CodeValidationStatus.UNKNOWN),
            message=info.message,
            normalized_code=info.normalized_value,
            details=info.details or {},
            confidence=1.0 if info.is_valid else 0.9,
        )

    def _infer_code_types(
        self,
        extracted_data: dict[str, Any],
    ) -> dict[str, CodeType]:
        """Infer code types from field names."""
        mapping: dict[str, CodeType] = {}

        for field_name in extracted_data.keys():
            lower_name = field_name.lower()

            if "cpt" in lower_name or "hcpcs" in lower_name:
                if "hcpcs" in lower_name:
                    mapping[field_name] = CodeType.HCPCS
                else:
                    mapping[field_name] = CodeType.CPT
            elif "icd" in lower_name or "diagnosis" in lower_name:
                mapping[field_name] = CodeType.ICD10_CM
            elif "npi" in lower_name:
                mapping[field_name] = CodeType.NPI
            elif "ndc" in lower_name:
                mapping[field_name] = CodeType.NDC
            elif "place_of_service" in lower_name or "pos" == lower_name:
                mapping[field_name] = CodeType.PLACE_OF_SERVICE
            elif "modifier" in lower_name:
                mapping[field_name] = CodeType.MODIFIER

        return mapping

    def _build_result(
        self,
        result: MedicalCodeValidationResult,
    ) -> None:
        """Build result aggregations."""
        for detail in result.validations:
            # Add to appropriate list
            if detail.status == CodeValidationStatus.VALID:
                result.valid_codes.append(detail.code)
            elif detail.status == CodeValidationStatus.INVALID:
                result.invalid_codes.append(detail.code)
                result.overall_valid = False
            elif detail.status == CodeValidationStatus.WARNING:
                result.warning_codes.append(detail.code)

            # Group by type
            type_key = detail.code_type.value
            if type_key not in result.by_type:
                result.by_type[type_key] = []
            result.by_type[type_key].append(detail)

        # Calculate validation rate
        total = len(result.validations)
        if total > 0:
            passed = len(result.valid_codes) + len(result.warning_codes)
            result.validation_rate = passed / total


def validate_medical_codes(
    extracted_data: dict[str, Any],
    code_field_mapping: dict[str, str] | None = None,
) -> MedicalCodeValidationResult:
    """
    Validate all medical codes in extracted data.

    Convenience function for one-off validation.

    Args:
        extracted_data: Dictionary of field names to values.
        code_field_mapping: Mapping of field names to code type strings.

    Returns:
        MedicalCodeValidationResult with all validation details.

    Example:
        result = validate_medical_codes({
            "cpt_code": "99213",
            "diagnosis_code": "E11.9",
            "billing_npi": "1234567893",
        })

        if not result.overall_valid:
            print(f"Invalid codes: {result.invalid_codes}")
    """
    engine = MedicalCodeValidationEngine()

    # Convert string types to enum if provided
    type_mapping: dict[str, CodeType] | None = None
    if code_field_mapping:
        type_mapping = {}
        for field_name, type_str in code_field_mapping.items():
            try:
                type_mapping[field_name] = CodeType(type_str.lower())
            except ValueError:
                logger.warning(f"Unknown code type: {type_str}")

    return engine.validate_all(extracted_data, type_mapping)
