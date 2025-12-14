"""
Field type definitions for document extraction schemas.

Provides comprehensive field type enumeration, field definitions,
and cross-field validation rules.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FieldType(str, Enum):
    """
    Supported field types for extraction.

    Each type has associated validation rules and formatting.
    """

    # Basic types
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"

    # Medical codes
    CPT_CODE = "cpt_code"
    ICD10_CODE = "icd10_code"
    NPI = "npi"
    HCPCS_CODE = "hcpcs_code"
    NDC_CODE = "ndc_code"
    TAXONOMY_CODE = "taxonomy_code"
    CARC_CODE = "carc_code"  # Claim Adjustment Reason Code
    RARC_CODE = "rarc_code"  # Remittance Advice Remark Code

    # Financial
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

    # Contact
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    STATE = "state"

    # Identity
    SSN = "ssn"
    MEMBER_ID = "member_id"
    GROUP_NUMBER = "group_number"
    POLICY_NUMBER = "policy_number"
    CLAIM_NUMBER = "claim_number"
    ACCOUNT_NUMBER = "account_number"

    # Document-specific
    NAME = "name"
    SIGNATURE = "signature"
    CHECKBOX = "checkbox"

    # Complex types
    LIST = "list"
    TABLE = "table"
    OBJECT = "object"

    @property
    def is_medical_code(self) -> bool:
        """Check if this is a medical code type."""
        return self in (
            FieldType.CPT_CODE,
            FieldType.ICD10_CODE,
            FieldType.NPI,
            FieldType.HCPCS_CODE,
            FieldType.NDC_CODE,
            FieldType.TAXONOMY_CODE,
            FieldType.CARC_CODE,
            FieldType.RARC_CODE,
        )

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric type."""
        return self in (
            FieldType.INTEGER,
            FieldType.FLOAT,
            FieldType.CURRENCY,
            FieldType.PERCENTAGE,
        )

    @property
    def is_identifier(self) -> bool:
        """Check if this is an identifier type."""
        return self in (
            FieldType.SSN,
            FieldType.MEMBER_ID,
            FieldType.GROUP_NUMBER,
            FieldType.POLICY_NUMBER,
            FieldType.CLAIM_NUMBER,
            FieldType.ACCOUNT_NUMBER,
            FieldType.NPI,
        )

    @property
    def requires_phi_protection(self) -> bool:
        """Check if this field type requires PHI protection."""
        return self in (
            FieldType.SSN,
            FieldType.NAME,
            FieldType.DATE,
            FieldType.PHONE,
            FieldType.EMAIL,
            FieldType.ADDRESS,
            FieldType.MEMBER_ID,
        )


class RuleOperator(str, Enum):
    """Operators for cross-field validation rules."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES_REGEX = "matches_regex"
    SUM_EQUALS = "sum_equals"
    DATE_BEFORE = "date_before"
    DATE_AFTER = "date_after"
    REQUIRES = "requires"
    REQUIRES_IF = "requires_if"


@dataclass(slots=True)
class CrossFieldRule:
    """
    Cross-field validation rule.

    Defines relationships between fields for validation.

    Attributes:
        source_field: Field that triggers the rule.
        target_field: Field to validate against.
        operator: Comparison operator.
        value: Optional static value for comparison.
        error_message: Custom error message.
        severity: Rule severity (error, warning, info).
    """

    source_field: str
    target_field: str
    operator: RuleOperator
    value: Any = None
    error_message: str | None = None
    severity: str = "error"

    def get_error_message(self) -> str:
        """Get error message for rule violation."""
        if self.error_message:
            return self.error_message

        messages = {
            RuleOperator.EQUALS: f"{self.source_field} must equal {self.target_field}",
            RuleOperator.NOT_EQUALS: f"{self.source_field} must not equal {self.target_field}",
            RuleOperator.GREATER_THAN: f"{self.source_field} must be greater than {self.target_field}",
            RuleOperator.LESS_THAN: f"{self.source_field} must be less than {self.target_field}",
            RuleOperator.GREATER_EQUAL: f"{self.source_field} must be >= {self.target_field}",
            RuleOperator.LESS_EQUAL: f"{self.source_field} must be <= {self.target_field}",
            RuleOperator.DATE_BEFORE: f"{self.source_field} must be before {self.target_field}",
            RuleOperator.DATE_AFTER: f"{self.source_field} must be after {self.target_field}",
            RuleOperator.SUM_EQUALS: f"Sum of fields must equal {self.target_field}",
            RuleOperator.REQUIRES: f"{self.source_field} requires {self.target_field}",
            RuleOperator.REQUIRES_IF: f"{self.target_field} is required when {self.source_field} has value",
        }

        return messages.get(
            self.operator,
            f"Cross-field validation failed: {self.source_field} -> {self.target_field}",
        )


@dataclass(slots=True)
class FieldDefinition:
    """
    Complete field definition for extraction.

    Defines all properties of an extractable field including
    type, validation, and extraction hints.

    Attributes:
        name: Field identifier (snake_case).
        display_name: Human-readable field name.
        field_type: Data type of the field.
        description: Detailed description for VLM prompting.
        required: Whether field is required.
        default: Default value if not found.
        pattern: Regex pattern for validation.
        examples: Example values for VLM prompting.
        aliases: Alternative names that may appear in documents.
        location_hint: Hint about where field typically appears.
        validation_func: Custom validation function.
        transform_func: Transform function for extracted value.
        min_value: Minimum value for numeric fields.
        max_value: Maximum value for numeric fields.
        min_length: Minimum string length.
        max_length: Maximum string length.
        allowed_values: List of allowed values (enum).
        list_item_type: Type of items for LIST fields.
        nested_schema: Schema name for OBJECT fields.
        confidence_threshold: Minimum confidence for acceptance.
    """

    name: str
    display_name: str
    field_type: FieldType
    description: str = ""
    required: bool = False
    default: Any = None
    pattern: str | None = None
    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    location_hint: str | None = None
    validation_func: Callable[[Any], bool] | None = None
    transform_func: Callable[[Any], Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    allowed_values: list[Any] | None = None
    list_item_type: FieldType | None = None
    nested_schema: str | None = None
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate field definition."""
        if not self.name:
            raise ValueError("Field name is required")

        if not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ValueError(
                f"Field name must be snake_case: {self.name}"
            )

        if self.field_type == FieldType.LIST and not self.list_item_type:
            raise ValueError("LIST fields must specify list_item_type")

        if self.field_type == FieldType.OBJECT and not self.nested_schema:
            raise ValueError("OBJECT fields must specify nested_schema")

    @property
    def compiled_pattern(self) -> re.Pattern[str] | None:
        """Get compiled regex pattern."""
        if self.pattern:
            return re.compile(self.pattern)
        return None

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a value against field definition.

        Args:
            value: Value to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check required
        if self.required and value is None:
            return False, f"{self.display_name} is required"

        # Skip validation for None values
        if value is None:
            return True, None

        # Type-specific validation
        if self.field_type in (FieldType.INTEGER,):
            if not isinstance(value, int):
                try:
                    int(value)
                except (ValueError, TypeError):
                    return False, f"{self.display_name} must be an integer"

        elif self.field_type in (FieldType.FLOAT, FieldType.CURRENCY, FieldType.PERCENTAGE):
            if not isinstance(value, (int, float)):
                try:
                    float(str(value).replace("$", "").replace(",", "").replace("%", ""))
                except (ValueError, TypeError):
                    return False, f"{self.display_name} must be a number"

        # Check pattern
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                return False, f"{self.display_name} does not match expected format"

        # Check allowed values
        if self.allowed_values and value not in self.allowed_values:
            return False, f"{self.display_name} must be one of: {self.allowed_values}"

        # Check min/max for numeric
        if self.min_value is not None:
            try:
                if float(value) < self.min_value:
                    return False, f"{self.display_name} must be >= {self.min_value}"
            except (ValueError, TypeError):
                pass

        if self.max_value is not None:
            try:
                if float(value) > self.max_value:
                    return False, f"{self.display_name} must be <= {self.max_value}"
            except (ValueError, TypeError):
                pass

        # Check string length
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                return False, f"{self.display_name} must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"{self.display_name} must be at most {self.max_length} characters"

        # Custom validation
        if self.validation_func:
            try:
                if not self.validation_func(value):
                    return False, f"{self.display_name} failed custom validation"
            except Exception as e:
                return False, f"{self.display_name} validation error: {e}"

        return True, None

    def transform(self, value: Any) -> Any:
        """
        Transform extracted value.

        Args:
            value: Value to transform.

        Returns:
            Transformed value.
        """
        if value is None:
            return self.default

        if self.transform_func:
            return self.transform_func(value)

        # Default transformations by type
        if self.field_type == FieldType.CURRENCY:
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = value.replace("$", "").replace(",", "").strip()
                try:
                    return float(cleaned)
                except ValueError:
                    return value

        elif self.field_type == FieldType.PERCENTAGE:
            if isinstance(value, str):
                cleaned = value.replace("%", "").strip()
                try:
                    return float(cleaned)
                except ValueError:
                    return value

        elif self.field_type == FieldType.PHONE:
            if isinstance(value, str):
                # Normalize phone format
                digits = re.sub(r"\D", "", value)
                if len(digits) == 10:
                    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
                elif len(digits) == 11 and digits[0] == "1":
                    return f"{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
                return value

        elif self.field_type == FieldType.SSN:
            if isinstance(value, str):
                digits = re.sub(r"\D", "", value)
                if len(digits) == 9:
                    return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
                return value

        return value

    def to_prompt_description(self) -> str:
        """
        Generate description for VLM prompting.

        Returns:
            Field description for extraction prompt.
        """
        parts = [self.description or self.display_name]

        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples)}")

        if self.location_hint:
            parts.append(f"Location: {self.location_hint}")

        if self.pattern:
            parts.append(f"Format: {self.pattern}")

        if self.allowed_values:
            parts.append(f"Valid values: {', '.join(str(v) for v in self.allowed_values)}")

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "field_type": self.field_type.value,
            "description": self.description,
            "required": self.required,
            "pattern": self.pattern,
            "examples": self.examples,
            "aliases": self.aliases,
            "location_hint": self.location_hint,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "allowed_values": self.allowed_values,
            "confidence_threshold": self.confidence_threshold,
        }
