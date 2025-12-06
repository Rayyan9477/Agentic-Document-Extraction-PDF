#!/usr/bin/env python
"""Quick integration tests for Phase 1 and Phase 2 components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.field_types import FieldType, FieldDefinition, CrossFieldRule, RuleOperator
from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.schema_builder import (
    FieldBuilder, SchemaBuilder, RuleBuilder, NestedSchemaBuilder,
    generate_zero_shot_schema, create_field, create_schema
)
from src.schemas.nested_schemas import (
    NestedSchema, NestedSchemaRegistry, get_nested_schema,
    CMS1500_SERVICE_LINE_SCHEMA
)
from src.schemas.validators import (
    validate_cpt_code, validate_icd10_code, validate_npi,
    validate_phone, validate_ssn, validate_date, validate_currency,
    ValidationResult as VResult
)
from src.pipeline.state import (
    ExtractionState, ExtractionStatus, FieldMetadata, PageExtraction,
    ValidationResult, ConfidenceLevel, create_initial_state, update_state,
    set_status, add_error, add_warning, complete_extraction, request_human_review,
    request_retry, serialize_state, deserialize_state
)


def main():
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)

    print("\n[1] Testing zero-shot schema generation...")
    schema = generate_zero_shot_schema(
        name="test_schema",
        field_names=["patient_name", "date_of_birth", "total_amount", "cpt_code", "npi"],
        register=False,
    )
    assert schema.name == "test_schema"
    assert len(schema.fields) == 5
    field_types = {f.name: f.field_type for f in schema.fields}
    assert field_types["patient_name"] == FieldType.NAME
    assert field_types["date_of_birth"] == FieldType.DATE
    assert field_types["total_amount"] == FieldType.CURRENCY
    assert field_types["cpt_code"] == FieldType.CPT_CODE
    assert field_types["npi"] == FieldType.NPI
    print("    PASSED: Zero-shot schema generation with type inference")

    print("\n[2] Testing fluent schema builder...")
    custom_schema = (
        SchemaBuilder("invoice", DocumentType.CUSTOM)
        .display_name("Invoice")
        .description("Invoice schema")
        .field(
            FieldBuilder("invoice_number")
            .type(FieldType.STRING)
            .required()
            .pattern(r"^INV-\d+$")
        )
        .field(
            FieldBuilder("amount")
            .type(FieldType.CURRENCY)
            .min_value(0.01)
        )
        .rule(
            RuleBuilder("invoice_date", "due_date")
            .date_before()
            .error("Invoice date must be before due date")
        )
        .build()
    )
    assert custom_schema.name == "invoice"
    assert len(custom_schema.fields) == 2
    assert len(custom_schema.cross_field_rules) == 1
    print("    PASSED: Fluent schema builder with fields and rules")

    print("\n[3] Testing nested schemas...")
    cms_line = get_nested_schema("cms1500_service_line")
    assert cms_line is not None
    assert cms_line.name == "cms1500_service_line"
    valid_line = {
        "line_number": 1,
        "date_from": "01/15/2024",
        "place_of_service": "11",
        "cpt_hcpcs": "99213",
        "diagnosis_pointer": "A",
        "charges": 150.00,
        "units": 1,
    }
    is_valid, errors = cms_line.validate(valid_line)
    assert is_valid, f"Expected valid but got errors: {errors}"
    print("    PASSED: Nested schema registration and validation")

    print("\n[4] Testing medical code validators...")
    # Validators return ValidationInfo objects with result attribute
    assert validate_cpt_code("99213").result == VResult.VALID
    assert validate_cpt_code("1234").result == VResult.INVALID
    assert validate_icd10_code("A00.0").result == VResult.VALID
    assert validate_icd10_code("123.45").result == VResult.INVALID
    assert validate_npi("1234567893").result == VResult.VALID
    assert validate_phone("555-123-4567").result == VResult.VALID
    assert validate_ssn("123-45-6789").result == VResult.VALID
    print("    PASSED: Medical code validators (CPT, ICD-10, NPI, Phone, SSN)")

    print("\n[5] Testing pipeline state management...")
    state = create_initial_state(
        pdf_path="/path/to/doc.pdf",
        pdf_hash="abc123",
        max_retries=3,
    )
    assert state["pdf_path"] == "/path/to/doc.pdf"
    assert state["status"] == ExtractionStatus.PENDING.value

    state = update_state(state, {"document_type": "CMS-1500"})
    assert state["document_type"] == "CMS-1500"

    state = set_status(state, ExtractionStatus.ANALYZING, "classifying")
    assert state["status"] == ExtractionStatus.ANALYZING.value

    state = add_error(state, "Test error")
    state = add_warning(state, "Test warning")
    assert len(state["errors"]) == 1
    assert len(state["warnings"]) == 1
    print("    PASSED: State creation, updates, status transitions, error handling")

    print("\n[6] Testing extraction completion states...")
    completed = complete_extraction(state, final_output={"field": "value"}, overall_confidence=0.92)
    assert completed["status"] == ExtractionStatus.COMPLETED.value
    assert completed["overall_confidence"] == 0.92

    retry_state = create_initial_state("/path/to/doc.pdf")
    retried = request_retry(retry_state, "Low quality")
    assert retried["status"] == ExtractionStatus.RETRYING.value
    assert retried["retry_count"] == 1

    review_state = create_initial_state("/path/to/doc.pdf")
    reviewed = request_human_review(review_state, "Manual review needed")
    assert reviewed["status"] == ExtractionStatus.HUMAN_REVIEW.value
    print("    PASSED: Completion, retry, and human review state transitions")

    print("\n[7] Testing state serialization...")
    state = create_initial_state("/test.pdf", custom_schema={"name": "test", "fields": []})
    serialized = serialize_state(state)
    deserialized = deserialize_state(serialized)
    assert deserialized["pdf_path"] == state["pdf_path"]
    assert deserialized["custom_schema"] == state["custom_schema"]
    print("    PASSED: State serialization/deserialization")

    print("\n[8] Testing custom schema in state...")
    custom = {
        "name": "custom_invoice",
        "description": "Custom invoice schema",
        "fields": [
            {"name": "invoice_id", "type": "STRING", "required": True},
            {"name": "total", "type": "CURRENCY", "min_value": 0.01},
        ],
        "rules": [
            {"source_field": "date", "target_field": "due_date", "operator": "DATE_BEFORE"},
        ],
    }
    state_with_custom = create_initial_state("/invoice.pdf", custom_schema=custom)
    assert state_with_custom["custom_schema"]["name"] == "custom_invoice"
    assert len(state_with_custom["custom_schema"]["fields"]) == 2
    print("    PASSED: Custom schema integration with pipeline state")

    print("\n[9] Testing FieldMetadata and ValidationResult...")
    metadata = FieldMetadata(
        field_name="test_field",
        value="test_value",
        confidence=0.95,
        pass1_value="test_value",
        pass2_value="test_value",
        passes_agree=True,
        source_page=1,
    )
    assert metadata.passes_agree is True
    assert metadata.confidence == 0.95

    result = ValidationResult(
        is_valid=False,
        overall_confidence=0.45,
        confidence_level=ConfidenceLevel.LOW,
        errors=["Error 1"],
        hallucination_flags=["field1"],
    )
    assert result.is_valid is False
    assert result.confidence_level == ConfidenceLevel.LOW
    print("    PASSED: FieldMetadata and ValidationResult dataclasses")

    print("\n" + "=" * 60)
    print("ALL 9 INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
