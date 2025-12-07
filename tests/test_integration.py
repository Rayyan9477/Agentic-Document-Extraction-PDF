"""
Integration tests for Phase 1 and Phase 2 components.

Tests the full integration of:
- Schema building (zero-shot, fluent API, nested)
- Custom schema support in agents
- Cross-field validation
- Pipeline state management
- Agent coordination
"""

import pytest
from typing import Any
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import FieldType, FieldDefinition, CrossFieldRule, RuleOperator
from src.schemas.schema_builder import (
    FieldBuilder,
    SchemaBuilder,
    RuleBuilder,
    NestedSchemaBuilder,
    generate_zero_shot_schema,
    create_field,
    create_schema,
    clone_schema,
)
from src.schemas.nested_schemas import (
    NestedSchema,
    NestedSchemaRegistry,
    get_nested_schema,
    CMS1500_SERVICE_LINE_SCHEMA,
)
from src.schemas.validators import (
    validate_cpt_code,
    validate_icd10_code,
    validate_npi,
    validate_phone,
    validate_ssn,
    validate_date,
    validate_currency,
)
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    FieldMetadata,
    PageExtraction,
    ValidationResult,
    ConfidenceLevel,
    create_initial_state,
    update_state,
    set_status,
    add_error,
    add_warning,
    complete_extraction,
    request_human_review,
    request_retry,
    serialize_state,
    deserialize_state,
)


# =============================================================================
# Schema Builder Tests
# =============================================================================

class TestZeroShotSchemaGeneration:
    """Tests for zero-shot schema generation."""

    def test_basic_field_names(self) -> None:
        """Test schema generation from basic field names."""
        schema = generate_zero_shot_schema(
            name="test_schema",
            field_names=["patient_name", "date_of_birth", "total_amount"],
            register=False,
        )

        assert schema.name == "test_schema"
        assert len(schema.fields) == 3

        # Verify type inference
        field_types = {f.name: f.field_type for f in schema.fields}
        assert field_types["patient_name"] == FieldType.NAME
        assert field_types["date_of_birth"] == FieldType.DATE
        assert field_types["total_amount"] == FieldType.CURRENCY

    def test_medical_code_inference(self) -> None:
        """Test type inference for medical codes."""
        schema = generate_zero_shot_schema(
            name="medical_schema",
            field_names=["cpt_code", "diagnosis_code", "provider_npi"],
            register=False,
        )

        field_types = {f.name: f.field_type for f in schema.fields}
        assert field_types["cpt_code"] == FieldType.CPT_CODE
        assert field_types["diagnosis_code"] == FieldType.ICD10_CODE
        assert field_types["provider_npi"] == FieldType.NPI

    def test_contact_info_inference(self) -> None:
        """Test type inference for contact information."""
        schema = generate_zero_shot_schema(
            name="contact_schema",
            field_names=["phone_number", "email_address", "zip_code", "state"],
            register=False,
        )

        field_types = {f.name: f.field_type for f in schema.fields}
        assert field_types["phone_number"] == FieldType.PHONE
        assert field_types["email_address"] == FieldType.EMAIL
        assert field_types["zip_code"] == FieldType.ZIP_CODE
        assert field_types["state"] == FieldType.STATE


class TestFluentSchemaBuilder:
    """Tests for fluent schema builder API."""

    def test_basic_schema_building(self) -> None:
        """Test basic schema creation with fluent API."""
        schema = (
            SchemaBuilder("invoice", DocumentType.CUSTOM)
            .display_name("Invoice Document")
            .description("Standard business invoice")
            .field(
                FieldBuilder("invoice_number")
                .type(FieldType.STRING)
                .required()
                .pattern(r"^INV-\d{6}$")
            )
            .field(
                FieldBuilder("total_amount")
                .type(FieldType.CURRENCY)
                .min_value(0.01)
            )
            .build()
        )

        assert schema.name == "invoice"
        assert schema.display_name == "Invoice Document"
        assert len(schema.fields) == 2

        # Verify invoice_number field
        inv_field = next(f for f in schema.fields if f.name == "invoice_number")
        assert inv_field.required is True
        assert inv_field.pattern == r"^INV-\d{6}$"

    def test_cross_field_rules(self) -> None:
        """Test schema with cross-field validation rules."""
        schema = (
            SchemaBuilder("claim", DocumentType.CMS_1500)
            .field(FieldBuilder("service_date").type(FieldType.DATE).required())
            .field(FieldBuilder("billing_date").type(FieldType.DATE).required())
            .rule(
                RuleBuilder("service_date", "billing_date")
                .date_before()
                .error("Service date must be before billing date")
            )
            .build()
        )

        assert len(schema.cross_field_rules) == 1
        rule = schema.cross_field_rules[0]
        assert rule.source_field == "service_date"
        assert rule.target_field == "billing_date"
        assert rule.operator == RuleOperator.DATE_BEFORE

    def test_field_with_all_options(self) -> None:
        """Test field builder with all options set."""
        field = (
            FieldBuilder("complex_field")
            .type(FieldType.STRING)
            .display_name("Complex Field")
            .description("A field with all options")
            .required()
            .location_hint("Top right corner")
            .examples(["Example 1", "Example 2"])
            .pattern(r"^[A-Z]+$")
            .allowed_values(["A", "B", "C"])
            .min_length(1)
            .max_length(10)
            .build()
        )

        assert field.name == "complex_field"
        assert field.required is True
        assert field.location_hint == "Top right corner"
        assert len(field.examples) == 2
        assert field.min_length == 1
        assert field.max_length == 10


class TestNestedSchemas:
    """Tests for nested schema support."""

    def test_nested_schema_registry(self) -> None:
        """Test nested schema registration and lookup."""
        registry = NestedSchemaRegistry()

        # Pre-registered schemas should be available
        cms_line = registry.get("cms1500_service_line")
        assert cms_line is not None
        assert cms_line.name == "cms1500_service_line"

    def test_nested_schema_validation(self) -> None:
        """Test nested schema validation."""
        schema = get_nested_schema("cms1500_service_line")
        assert schema is not None

        # Valid service line
        valid_line = {
            "line_number": 1,
            "date_from": "01/15/2024",
            "place_of_service": "11",
            "cpt_hcpcs": "99213",
            "diagnosis_pointer": "A",
            "charges": 150.00,
            "units": 1,
        }

        is_valid, errors = schema.validate(valid_line)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid - missing required field
        invalid_line = {
            "line_number": 1,
            "date_from": "01/15/2024",
            # Missing place_of_service, cpt_hcpcs, etc.
        }

        is_valid, errors = schema.validate(invalid_line)
        assert is_valid is False
        assert len(errors) > 0

    def test_nested_schema_builder(self) -> None:
        """Test building custom nested schemas."""
        nested = (
            NestedSchemaBuilder("custom_line_item")
            .display_name("Custom Line Item")
            .description("A custom line item for testing")
            .field(
                FieldBuilder("item_code")
                .type(FieldType.STRING)
                .required()
            )
            .field(
                FieldBuilder("quantity")
                .type(FieldType.INTEGER)
                .min_value(1)
            )
            .field(
                FieldBuilder("unit_price")
                .type(FieldType.CURRENCY)
                .required()
            )
            .build()
        )

        assert nested.name == "custom_line_item"
        assert len(nested.fields) == 3


# =============================================================================
# Validator Tests
# =============================================================================

class TestMedicalCodeValidation:
    """Tests for medical code validators."""

    def test_valid_cpt_codes(self) -> None:
        """Test valid CPT codes."""
        assert validate_cpt_code("99213").is_valid is True
        assert validate_cpt_code("99214").is_valid is True
        assert validate_cpt_code("99215").is_valid is True
        assert validate_cpt_code("99232").is_valid is True

    def test_invalid_cpt_codes(self) -> None:
        """Test invalid CPT codes."""
        assert validate_cpt_code("1234").is_valid is False  # Too short
        assert validate_cpt_code("12").is_valid is False  # Too short
        assert validate_cpt_code("ABCDE").is_valid is False  # Not numeric

    def test_valid_icd10_codes(self) -> None:
        """Test valid ICD-10 codes."""
        assert validate_icd10_code("A00.0").is_valid is True
        assert validate_icd10_code("Z99.89").is_valid is True
        assert validate_icd10_code("M54.5").is_valid is True

    def test_invalid_icd10_codes(self) -> None:
        """Test invalid ICD-10 codes."""
        assert validate_icd10_code("123.45").is_valid is False  # Must start with letter
        assert validate_icd10_code("A").is_valid is False  # Too short

    def test_valid_npi(self) -> None:
        """Test valid NPI numbers (Luhn algorithm)."""
        # 1234567893 passes Luhn check for NPI
        assert validate_npi("1234567893").is_valid is True
        # Common test NPI
        assert validate_npi("1234567893").is_valid is True

    def test_invalid_npi(self) -> None:
        """Test invalid NPI numbers."""
        assert validate_npi("1234567890").is_valid is False  # Fails Luhn check
        assert validate_npi("123456789").is_valid is False  # Wrong length

    def test_phone_validation(self) -> None:
        """Test phone number validation."""
        assert validate_phone("555-123-4567").is_valid is True
        assert validate_phone("(555) 123-4567").is_valid is True
        assert validate_phone("5551234567").is_valid is True
        assert validate_phone("123").is_valid is False  # Too short

    def test_ssn_validation(self) -> None:
        """Test SSN validation."""
        assert validate_ssn("123-45-6789").is_valid is True
        assert validate_ssn("123456789").is_valid is True
        assert validate_ssn("12345678").is_valid is False  # Too short


# =============================================================================
# Pipeline State Tests
# =============================================================================

class TestPipelineState:
    """Tests for pipeline state management."""

    def test_create_initial_state(self) -> None:
        """Test initial state creation."""
        state = create_initial_state(
            pdf_path="/path/to/doc.pdf",
            pdf_hash="abc123",
            max_retries=3,
        )

        assert state["pdf_path"] == "/path/to/doc.pdf"
        assert state["pdf_hash"] == "abc123"
        assert state["max_retries"] == 3
        assert state["status"] == ExtractionStatus.PENDING.value
        assert state["retry_count"] == 0
        assert state["errors"] == []
        assert state["warnings"] == []

    def test_update_state(self) -> None:
        """Test state updates."""
        state = create_initial_state("/path/to/doc.pdf")

        updated = update_state(state, {
            "document_type": "CMS-1500",
            "selected_schema_name": "cms1500_schema",
        })

        assert updated["document_type"] == "CMS-1500"
        assert updated["selected_schema_name"] == "cms1500_schema"
        # Original values should be preserved
        assert updated["pdf_path"] == "/path/to/doc.pdf"

    def test_status_transitions(self) -> None:
        """Test status transitions."""
        state = create_initial_state("/path/to/doc.pdf")

        state = set_status(state, ExtractionStatus.ANALYZING, "classifying")
        assert state["status"] == ExtractionStatus.ANALYZING.value

        state = set_status(state, ExtractionStatus.EXTRACTING, "extracting")
        assert state["status"] == ExtractionStatus.EXTRACTING.value

        state = set_status(state, ExtractionStatus.VALIDATING, "validating")
        assert state["status"] == ExtractionStatus.VALIDATING.value

    def test_error_and_warning_handling(self) -> None:
        """Test error and warning accumulation."""
        state = create_initial_state("/path/to/doc.pdf")

        state = add_error(state, "Error 1")
        state = add_error(state, "Error 2")
        state = add_warning(state, "Warning 1")

        assert len(state["errors"]) == 2
        assert len(state["warnings"]) == 1
        assert "Error 1" in state["errors"]
        assert "Warning 1" in state["warnings"]

    def test_complete_extraction(self) -> None:
        """Test extraction completion."""
        state = create_initial_state("/path/to/doc.pdf")
        state = update_state(state, {
            "merged_extraction": {"patient_name": {"value": "John Doe"}},
        })

        completed = complete_extraction(
            state,
            final_output={"patient_name": "John Doe"},
            overall_confidence=0.95,
        )

        assert completed["status"] == ExtractionStatus.COMPLETED.value
        assert completed["overall_confidence"] == 0.95
        assert completed["final_output"]["patient_name"] == "John Doe"

    def test_request_human_review(self) -> None:
        """Test human review request."""
        state = create_initial_state("/path/to/doc.pdf")

        reviewed = request_human_review(state, "Low confidence detected")

        assert reviewed["status"] == ExtractionStatus.HUMAN_REVIEW.value
        assert reviewed["human_review_reason"] == "Low confidence detected"

    def test_request_retry(self) -> None:
        """Test retry request."""
        state = create_initial_state("/path/to/doc.pdf", max_retries=2)

        retried = request_retry(state, "Extraction quality below threshold")

        assert retried["status"] == ExtractionStatus.RETRYING.value
        assert retried["retry_count"] == 1

    def test_state_serialization(self) -> None:
        """Test state serialization/deserialization."""
        state = create_initial_state("/path/to/doc.pdf")
        state = update_state(state, {
            "document_type": "CMS-1500",
            "overall_confidence": 0.85,
        })

        serialized = serialize_state(state)
        deserialized = deserialize_state(serialized)

        assert deserialized["pdf_path"] == state["pdf_path"]
        assert deserialized["document_type"] == state["document_type"]
        assert deserialized["overall_confidence"] == state["overall_confidence"]


class TestCustomSchemaIntegration:
    """Tests for custom schema integration with agents."""

    def test_custom_schema_definition(self) -> None:
        """Test custom schema definition structure."""
        custom_schema = {
            "name": "custom_invoice",
            "description": "Custom invoice extraction schema",
            "fields": [
                {
                    "name": "invoice_id",
                    "display_name": "Invoice ID",
                    "type": "STRING",
                    "required": True,
                    "pattern": r"^INV-\d+$",
                },
                {
                    "name": "total_amount",
                    "display_name": "Total Amount",
                    "type": "CURRENCY",
                    "required": True,
                    "min_value": 0.01,
                },
                {
                    "name": "invoice_date",
                    "display_name": "Invoice Date",
                    "type": "DATE",
                    "required": True,
                },
                {
                    "name": "due_date",
                    "display_name": "Due Date",
                    "type": "DATE",
                    "required": False,
                },
            ],
            "rules": [
                {
                    "source_field": "invoice_date",
                    "target_field": "due_date",
                    "operator": "DATE_BEFORE",
                    "error_message": "Invoice date must be before due date",
                },
            ],
        }

        # Build schema from definition
        from src.schemas.schema_builder import SchemaBuilder, FieldBuilder, RuleBuilder
        from src.schemas.field_types import RuleOperator

        builder = SchemaBuilder(
            name=custom_schema["name"],
            document_type=DocumentType.CUSTOM,
        ).description(custom_schema["description"])

        for field_def in custom_schema["fields"]:
            field_type = FieldType[field_def["type"]]
            field_builder = (
                FieldBuilder(field_def["name"])
                .display_name(field_def["display_name"])
                .type(field_type)
                .required(field_def.get("required", False))
            )

            if field_def.get("pattern"):
                field_builder.pattern(field_def["pattern"])
            if field_def.get("min_value") is not None:
                field_builder.min_value(field_def["min_value"])

            builder.field(field_builder)

        for rule_def in custom_schema["rules"]:
            operator = RuleOperator[rule_def["operator"]]
            rule_builder = RuleBuilder(
                rule_def["source_field"],
                rule_def["target_field"],
            )

            if operator == RuleOperator.DATE_BEFORE:
                rule_builder.date_before()

            if rule_def.get("error_message"):
                rule_builder.error(rule_def["error_message"])

            builder.rule(rule_builder)

        schema = builder.build()

        assert schema.name == "custom_invoice"
        assert len(schema.fields) == 4
        assert len(schema.cross_field_rules) == 1

    def test_state_with_custom_schema(self) -> None:
        """Test pipeline state with custom schema."""
        custom_schema = {
            "name": "custom_report",
            "fields": [
                {"name": "report_id", "type": "STRING", "required": True},
                {"name": "report_date", "type": "DATE", "required": True},
            ],
        }

        state = create_initial_state(
            pdf_path="/path/to/report.pdf",
            custom_schema=custom_schema,
        )

        assert state["custom_schema"] is not None
        assert state["custom_schema"]["name"] == "custom_report"
        assert len(state["custom_schema"]["fields"]) == 2


# =============================================================================
# Field Metadata Tests
# =============================================================================

class TestFieldMetadata:
    """Tests for field metadata handling."""

    def test_field_metadata_creation(self) -> None:
        """Test FieldMetadata creation."""
        metadata = FieldMetadata(
            field_name="patient_name",
            value="John Doe",
            confidence=0.95,
            pass1_value="John Doe",
            pass2_value="John Doe",
            passes_agree=True,
            source_page=1,
        )

        assert metadata.field_name == "patient_name"
        assert metadata.value == "John Doe"
        assert metadata.confidence == 0.95
        assert metadata.passes_agree is True

    def test_field_metadata_disagreement(self) -> None:
        """Test FieldMetadata with pass disagreement."""
        metadata = FieldMetadata(
            field_name="amount",
            value="100.00",
            confidence=0.6,
            pass1_value="100.00",
            pass2_value="150.00",
            passes_agree=False,
            source_page=1,
        )

        assert metadata.passes_agree is False
        assert metadata.confidence == 0.6  # Lower due to disagreement


class TestValidationResult:
    """Tests for validation result handling."""

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=True,
            overall_confidence=0.92,
            confidence_level=ConfidenceLevel.HIGH,
        )

        assert result.is_valid is True
        assert result.overall_confidence == 0.92
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_validation_with_errors(self) -> None:
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            overall_confidence=0.45,
            confidence_level=ConfidenceLevel.LOW,
            errors=["Invalid CPT code", "Missing required field"],
            hallucination_flags=["patient_name"],
            requires_human_review=True,  # Low confidence with errors requires human review
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.hallucination_flags) == 1
        assert result.requires_human_review is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
