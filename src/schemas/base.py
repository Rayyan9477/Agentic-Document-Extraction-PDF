"""
Base schema definitions for document extraction.

Provides the core schema infrastructure including document schemas,
extraction results, confidence tracking, and schema registry.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType


class DocumentType(str, Enum):
    """Supported document types for extraction."""

    # Healthcare billing forms
    CMS_1500 = "cms_1500"
    UB_04 = "ub_04"
    SUPERBILL = "superbill"
    EOB = "eob"

    # Insurance forms
    CLAIM_FORM = "claim_form"
    PRIOR_AUTH = "prior_auth"

    # Medical records
    MEDICAL_RECORD = "medical_record"
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"

    # Administrative
    PATIENT_INTAKE = "patient_intake"
    CONSENT_FORM = "consent_form"

    # Generic
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level categorization."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Get confidence level from numeric score."""
        if score >= 0.85:
            return cls.HIGH
        if score >= 0.70:
            return cls.MEDIUM
        if score >= 0.50:
            return cls.LOW
        return cls.VERY_LOW


@dataclass(slots=True)
class FieldConfidence:
    """
    Confidence information for an extracted field.

    Attributes:
        field_name: Name of the field.
        value: Extracted value.
        confidence_score: Confidence score 0-1.
        pass1_value: Value from first extraction pass.
        pass2_value: Value from second extraction pass.
        passes_match: Whether both passes produced same value.
        location: Where in document field was found.
        validation_passed: Whether field passed validation.
        validation_message: Validation error/warning message.
        source: Source of the value (pass1, pass2, merged).
    """

    field_name: str
    value: Any
    confidence_score: float
    pass1_value: Any = None
    pass2_value: Any = None
    passes_match: bool = True
    location: str | None = None
    validation_passed: bool = True
    validation_message: str | None = None
    source: str = "pass1"

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        return ConfidenceLevel.from_score(self.confidence_score)

    @property
    def needs_review(self) -> bool:
        """Check if field needs human review."""
        return not self.passes_match or self.confidence_score < 0.50 or not self.validation_passed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "passes_match": self.passes_match,
            "location": self.location,
            "validation_passed": self.validation_passed,
            "validation_message": self.validation_message,
            "needs_review": self.needs_review,
            "source": self.source,
        }


@dataclass(slots=True)
class ExtractionField:
    """
    Single extracted field with metadata.

    Attributes:
        name: Field name.
        value: Extracted value.
        confidence: Confidence information.
        raw_value: Original value before transformation.
        is_inferred: Whether value was inferred.
    """

    name: str
    value: Any
    confidence: FieldConfidence
    raw_value: Any = None
    is_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "raw_value": self.raw_value,
            "is_inferred": self.is_inferred,
            "confidence": self.confidence.to_dict(),
        }


@dataclass(slots=True)
class ExtractionResult:
    """
    Complete extraction result for a document.

    Attributes:
        document_id: Unique document identifier.
        document_type: Detected document type.
        schema_name: Schema used for extraction.
        fields: Dictionary of extracted fields.
        confidence_scores: Per-field confidence information.
        overall_confidence: Aggregate confidence score.
        page_results: Results by page for multi-page documents.
        validation_errors: List of validation errors.
        validation_warnings: List of validation warnings.
        hallucination_flags: Fields flagged for potential hallucination.
        extraction_time_ms: Processing time in milliseconds.
        vlm_calls: Number of VLM calls made.
        pass1_raw: Raw output from pass 1.
        pass2_raw: Raw output from pass 2.
        created_at: Extraction timestamp.
        metadata: Additional metadata.
    """

    document_id: str
    document_type: DocumentType
    schema_name: str
    fields: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, FieldConfidence] = field(default_factory=dict)
    overall_confidence: float = 0.0
    page_results: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    extraction_time_ms: int = 0
    vlm_calls: int = 0
    pass1_raw: dict[str, Any] | None = None
    pass2_raw: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if extraction passed validation."""
        return len(self.validation_errors) == 0

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get overall confidence level."""
        return ConfidenceLevel.from_score(self.overall_confidence)

    @property
    def needs_review(self) -> bool:
        """Check if result needs human review."""
        return (
            self.overall_confidence < 0.50
            or len(self.hallucination_flags) > 0
            or any(conf.needs_review for conf in self.confidence_scores.values())
        )

    @property
    def field_count(self) -> int:
        """Get number of extracted fields."""
        return len(self.fields)

    @property
    def valid_field_count(self) -> int:
        """Get number of fields that passed validation."""
        return sum(1 for conf in self.confidence_scores.values() if conf.validation_passed)

    def get_field(self, name: str) -> Any:
        """Get field value by name."""
        return self.fields.get(name)

    def get_confidence(self, name: str) -> FieldConfidence | None:
        """Get field confidence by name."""
        return self.confidence_scores.get(name)

    def calculate_overall_confidence(self) -> float:
        """Calculate aggregate confidence score."""
        if not self.confidence_scores:
            return 0.0

        total_score = sum(conf.confidence_score for conf in self.confidence_scores.values())
        return total_score / len(self.confidence_scores)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "schema_name": self.schema_name,
            "fields": self.fields,
            "confidence_scores": {k: v.to_dict() for k, v in self.confidence_scores.items()},
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "is_valid": self.is_valid,
            "needs_review": self.needs_review,
            "field_count": self.field_count,
            "valid_field_count": self.valid_field_count,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "hallucination_flags": self.hallucination_flags,
            "extraction_time_ms": self.extraction_time_ms,
            "vlm_calls": self.vlm_calls,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Convert to flat dictionary for export.

        Flattens nested structures for Excel/CSV export.
        """
        flat = {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "overall_confidence": self.overall_confidence,
            "is_valid": self.is_valid,
            "needs_review": self.needs_review,
            "extraction_time_ms": self.extraction_time_ms,
        }

        # Add all fields
        for name, value in self.fields.items():
            if isinstance(value, (dict, list)):
                flat[name] = str(value)
            else:
                flat[name] = value

            # Add confidence
            if name in self.confidence_scores:
                flat[f"{name}_confidence"] = self.confidence_scores[name].confidence_score

        return flat


@dataclass
class DocumentSchema:
    """
    Schema definition for a document type.

    Defines the structure, fields, and validation rules
    for a specific type of document.

    Attributes:
        name: Schema identifier.
        display_name: Human-readable name.
        document_type: Associated document type.
        description: Schema description.
        version: Schema version.
        fields: List of field definitions.
        cross_field_rules: Cross-field validation rules.
        required_sections: Required document sections.
        classification_hints: Hints for document classification.
        extraction_prompt_template: Custom extraction prompt.
    """

    name: str
    display_name: str
    document_type: DocumentType
    description: str = ""
    version: str = "1.0.0"
    fields: list[FieldDefinition] = field(default_factory=list)
    cross_field_rules: list[CrossFieldRule] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)
    classification_hints: list[str] = field(default_factory=list)
    extraction_prompt_template: str | None = None

    _field_map: dict[str, FieldDefinition] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build field lookup map."""
        self._field_map = {f.name: f for f in self.fields}

    def get_field(self, name: str) -> FieldDefinition | None:
        """Get field definition by name."""
        return self._field_map.get(name)

    def get_required_fields(self) -> list[FieldDefinition]:
        """Get list of required fields."""
        return [f for f in self.fields if f.required]

    def get_fields_by_type(self, field_type: FieldType) -> list[FieldDefinition]:
        """Get fields of a specific type."""
        return [f for f in self.fields if f.field_type == field_type]

    def get_phi_fields(self) -> list[FieldDefinition]:
        """Get fields that contain PHI."""
        return [f for f in self.fields if f.field_type.requires_phi_protection]

    def validate_result(self, result: dict[str, Any]) -> tuple[list[str], list[str]]:
        """
        Validate extraction result against schema.

        Args:
            result: Dictionary of extracted values.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check required fields
        for field_def in self.get_required_fields():
            if field_def.name not in result or result[field_def.name] is None:
                errors.append(f"Required field missing: {field_def.display_name}")

        # Validate each field
        for name, value in result.items():
            field_def = self.get_field(name)
            if field_def:
                is_valid, message = field_def.validate(value)
                if not is_valid:
                    if field_def.required:
                        errors.append(message or f"Invalid value for {name}")
                    else:
                        warnings.append(message or f"Invalid value for {name}")

        # Apply cross-field rules
        for rule in self.cross_field_rules:
            rule_passed = self._check_cross_field_rule(rule, result)
            if not rule_passed:
                if rule.severity == "error":
                    errors.append(rule.get_error_message())
                else:
                    warnings.append(rule.get_error_message())

        return errors, warnings

    def _check_cross_field_rule(
        self,
        rule: CrossFieldRule,
        result: dict[str, Any],
    ) -> bool:
        """Check a single cross-field rule."""
        from src.schemas.field_types import RuleOperator

        source_value = result.get(rule.source_field)
        target_value = result.get(rule.target_field)

        # Skip if source is None
        if source_value is None:
            return True

        if rule.operator == RuleOperator.REQUIRES:
            return target_value is not None

        if rule.operator == RuleOperator.REQUIRES_IF:
            if source_value:
                return target_value is not None
            return True

        if rule.operator == RuleOperator.EQUALS:
            return source_value == target_value

        if rule.operator == RuleOperator.NOT_EQUALS:
            return source_value != target_value

        if rule.operator in (
            RuleOperator.GREATER_THAN,
            RuleOperator.LESS_THAN,
            RuleOperator.GREATER_EQUAL,
            RuleOperator.LESS_EQUAL,
        ):
            try:
                s = float(source_value) if source_value else 0
                t = float(target_value) if target_value else 0

                if rule.operator == RuleOperator.GREATER_THAN:
                    return s > t
                if rule.operator == RuleOperator.LESS_THAN:
                    return s < t
                if rule.operator == RuleOperator.GREATER_EQUAL:
                    return s >= t
                if rule.operator == RuleOperator.LESS_EQUAL:
                    return s <= t
            except (ValueError, TypeError):
                return False

        if rule.operator in (RuleOperator.DATE_BEFORE, RuleOperator.DATE_AFTER):
            from datetime import datetime

            try:
                # Parse dates (support multiple formats)
                def parse_date(d: Any) -> datetime | None:
                    if isinstance(d, datetime):
                        return d
                    if isinstance(d, str):
                        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y"):
                            try:
                                return datetime.strptime(d, fmt)
                            except ValueError:
                                continue
                    return None

                s_date = parse_date(source_value)
                t_date = parse_date(target_value)

                if s_date and t_date:
                    if rule.operator == RuleOperator.DATE_BEFORE:
                        return s_date < t_date
                    return s_date > t_date
            except Exception:
                return False

        return True

    def generate_extraction_prompt(self) -> str:
        """
        Generate extraction prompt for VLM.

        Returns:
            Extraction prompt string.
        """
        if self.extraction_prompt_template:
            return self.extraction_prompt_template

        # Build field descriptions
        field_descriptions = []
        for field_def in self.fields:
            desc = f"- {field_def.name}: {field_def.to_prompt_description()}"
            if field_def.required:
                desc += " [REQUIRED]"
            field_descriptions.append(desc)

        prompt = f"""Extract the following fields from this {self.display_name} document:

{chr(10).join(field_descriptions)}

IMPORTANT RULES:
1. Only extract values that are CLEARLY VISIBLE in the document
2. If a field is unclear, blurry, or not visible, use null
3. Do NOT guess or infer values
4. Do NOT use placeholder or default values
5. Include confidence (0.0-1.0) for each field
6. Describe the location where each value was found

Return the result as a JSON object with this structure:
{{
    "fields": {{
        "field_name": {{
            "value": "extracted value or null",
            "confidence": 0.0-1.0,
            "location": "where in document"
        }}
    }}
}}"""

        return prompt

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "document_type": self.document_type.value,
            "description": self.description,
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "cross_field_rules": [
                {
                    "source": r.source_field,
                    "target": r.target_field,
                    "operator": r.operator.value,
                }
                for r in self.cross_field_rules
            ],
            "required_sections": self.required_sections,
            "classification_hints": self.classification_hints,
        }


class SchemaRegistry:
    """
    Registry for document schemas.

    Provides centralized schema management and lookup.

    Example:
        registry = SchemaRegistry()
        registry.register(superbill_schema)

        schema = registry.get("superbill")
        schema = registry.get_by_document_type(DocumentType.SUPERBILL)
    """

    _instance: ClassVar["SchemaRegistry | None"] = None
    _schemas: dict[str, DocumentSchema]

    def __new__(cls) -> "SchemaRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._schemas = {}
        return cls._instance

    def register(self, schema: DocumentSchema) -> None:
        """
        Register a schema.

        Args:
            schema: Schema to register.
        """
        self._schemas[schema.name] = schema

    def get(self, name: str) -> DocumentSchema | None:
        """
        Get schema by name.

        Args:
            name: Schema name.

        Returns:
            DocumentSchema or None.
        """
        return self._schemas.get(name)

    def get_by_document_type(self, doc_type: DocumentType) -> DocumentSchema | None:
        """
        Get schema by document type.

        Args:
            doc_type: Document type.

        Returns:
            First matching schema or None.
        """
        for schema in self._schemas.values():
            if schema.document_type == doc_type:
                return schema
        return None

    def list_schemas(self) -> list[DocumentSchema]:
        """Get list of all registered schemas."""
        return list(self._schemas.values())

    def list_schema_names(self) -> list[str]:
        """Get list of registered schema names."""
        return list(self._schemas.keys())

    def list_by_document_type(self, doc_type: DocumentType) -> list[DocumentSchema]:
        """Get all schemas for a document type."""
        return [s for s in self._schemas.values() if s.document_type == doc_type]

    def get_by_type(self, doc_type: DocumentType) -> DocumentSchema:
        """
        Get schema by document type.

        Args:
            doc_type: Document type to find.

        Returns:
            DocumentSchema for the type.

        Raises:
            ValueError: If no schema found for type.
        """
        schema = self.get_by_document_type(doc_type)
        if schema is None:
            raise ValueError(f"No schema registered for document type: {doc_type.value}")
        return schema

    def unregister(self, name: str) -> bool:
        """
        Unregister a schema.

        Args:
            name: Schema name.

        Returns:
            True if schema was removed.
        """
        if name in self._schemas:
            del self._schemas[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered schemas."""
        self._schemas.clear()

    def get_schema_count(self) -> int:
        """Get number of registered schemas."""
        return len(self._schemas)

    def has_schema(self, name: str) -> bool:
        """Check if schema is registered."""
        return name in self._schemas
