"""
Unit tests for Phase 2 multi-stage validation improvements to multi_record.py.

Tests cover:
- Constructor Phase 2 configuration
- Validation extraction method
- Self-correction for low-confidence fields
- Pipeline integration (extract → validate → correct)
- Feature flag gating
- Backward compatibility (Phase 2 disabled)
"""

from unittest.mock import MagicMock, call, patch

import pytest

from src.extraction.multi_record import (
    ExtractedRecord,
    MultiRecordExtractor,
    RecordBoundary,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_boundary(record_id=1, primary_id="Smith, John"):
    """Create a test RecordBoundary."""
    return RecordBoundary(
        record_id=record_id,
        primary_identifier=primary_id,
        bounding_box={"top": 0.0, "left": 0.0, "bottom": 0.5, "right": 1.0},
        visual_separator="line",
        entity_type="patient",
    )


def _make_schema():
    """Create a test schema."""
    return {
        "fields": [
            {"field_name": "patient_name", "field_type": "text", "description": "Name"},
            {"field_name": "patient_dob", "field_type": "date", "description": "Date of birth"},
            {"field_name": "patient_id", "field_type": "text", "description": "Patient ID"},
            {"field_name": "diagnosis", "field_type": "text", "description": "Diagnosis"},
        ]
    }


def _make_record(record_id=1, confidence=0.92):
    """Create a test ExtractedRecord."""
    return ExtractedRecord(
        record_id=record_id,
        page_number=1,
        primary_identifier="Smith, John",
        entity_type="patient",
        fields={
            "patient_name": "Smith, John",
            "patient_dob": "01/15/1980",
            "patient_id": "MRN12345",
            "diagnosis": "Hypertension",
        },
        confidence=confidence,
        extraction_time_ms=500,
    )


# ──────────────────────────────────────────────────────────────
# Constructor Tests
# ──────────────────────────────────────────────────────────────

class TestPhase2Constructor:
    """Test Phase 2 constructor parameters."""

    def test_default_phase2_disabled(self):
        """Phase 2 features should be disabled by default."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(client=mock_client)

        assert extractor._enable_validation is False
        assert extractor._enable_self_correction is False
        assert extractor._confidence_threshold == 0.85

    def test_enable_validation(self):
        """Validation can be enabled via constructor."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
        )
        assert extractor._enable_validation is True

    def test_enable_self_correction(self):
        """Self-correction can be enabled via constructor."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_self_correction=True,
        )
        assert extractor._enable_self_correction is True

    def test_custom_confidence_threshold(self):
        """Confidence threshold can be customized."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            confidence_threshold=0.90,
        )
        assert extractor._confidence_threshold == 0.90


# ──────────────────────────────────────────────────────────────
# Validation Tests
# ──────────────────────────────────────────────────────────────

class TestValidateExtraction:
    """Test the _validate_extraction() method."""

    def test_validation_all_correct(self):
        """When all fields are correct, overall_valid should be True."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "validation_summary": {
                "record_identifier": "Smith, John",
                "total_fields_checked": 4,
                "correct_count": 4,
                "incorrect_count": 0,
                "uncertain_count": 0,
            },
            "field_validations": [
                {
                    "field_name": "patient_name",
                    "original_value": "Smith, John",
                    "actual_value_in_image": "Smith, John",
                    "is_correct": True,
                    "corrected_value": None,
                    "confidence": 0.98,
                    "issue": None,
                },
                {
                    "field_name": "patient_dob",
                    "original_value": "01/15/1980",
                    "actual_value_in_image": "01/15/1980",
                    "is_correct": True,
                    "corrected_value": None,
                    "confidence": 0.95,
                    "issue": None,
                },
                {
                    "field_name": "patient_id",
                    "original_value": "MRN12345",
                    "actual_value_in_image": "MRN12345",
                    "is_correct": True,
                    "corrected_value": None,
                    "confidence": 0.97,
                    "issue": None,
                },
                {
                    "field_name": "diagnosis",
                    "original_value": "Hypertension",
                    "actual_value_in_image": "Hypertension",
                    "is_correct": True,
                    "corrected_value": None,
                    "confidence": 0.93,
                    "issue": None,
                },
            ],
            "overall_confidence": 0.96,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
            confidence_threshold=0.85,
        )

        result = extractor._validate_extraction(
            page_data_uri="data:image/png;base64,fake",
            record=_make_record(),
            boundary=_make_boundary(),
            schema=_make_schema(),
        )

        assert result["overall_valid"] is True
        assert len(result["fields_needing_correction"]) == 0

    def test_validation_detects_incorrect_fields(self):
        """When fields are incorrect, they should be flagged for correction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "validation_summary": {
                "total_fields_checked": 4,
                "correct_count": 2,
                "incorrect_count": 2,
            },
            "field_validations": [
                {
                    "field_name": "patient_name",
                    "is_correct": True,
                    "confidence": 0.98,
                    "issue": None,
                },
                {
                    "field_name": "patient_dob",
                    "is_correct": False,
                    "corrected_value": "01/15/1981",
                    "confidence": 0.90,
                    "issue": "Year digit misread",
                },
                {
                    "field_name": "patient_id",
                    "is_correct": False,
                    "corrected_value": "MRN12346",
                    "confidence": 0.88,
                    "issue": "Last digit incorrect",
                },
                {
                    "field_name": "diagnosis",
                    "is_correct": True,
                    "confidence": 0.91,
                    "issue": None,
                },
            ],
            "overall_confidence": 0.82,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
        )

        result = extractor._validate_extraction(
            page_data_uri="data:image/png;base64,fake",
            record=_make_record(),
            boundary=_make_boundary(),
            schema=_make_schema(),
        )

        assert result["overall_valid"] is False
        assert "patient_dob" in result["fields_needing_correction"]
        assert "patient_id" in result["fields_needing_correction"]
        assert "patient_name" not in result["fields_needing_correction"]

    def test_validation_flags_low_confidence_even_if_correct(self):
        """Fields marked correct but below confidence threshold should still be flagged."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "field_validations": [
                {
                    "field_name": "patient_name",
                    "is_correct": True,
                    "confidence": 0.60,  # Below threshold
                    "issue": None,
                },
                {
                    "field_name": "patient_dob",
                    "is_correct": True,
                    "confidence": 0.95,
                    "issue": None,
                },
            ],
            "overall_confidence": 0.78,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
            confidence_threshold=0.85,
        )

        result = extractor._validate_extraction(
            page_data_uri="data:image/png;base64,fake",
            record=_make_record(),
            boundary=_make_boundary(),
            schema=_make_schema(),
        )

        # patient_name is_correct=True but confidence=0.60 < 0.85 threshold
        assert "patient_name" in result["fields_needing_correction"]
        assert "patient_dob" not in result["fields_needing_correction"]

    def test_validation_uses_system_prompt(self):
        """Validation VLM call should include the system grounding prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "field_validations": [],
            "overall_confidence": 0.9,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
        )

        extractor._validate_extraction(
            page_data_uri="data:image/png;base64,fake",
            record=_make_record(),
            boundary=_make_boundary(),
            schema=_make_schema(),
        )

        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]
        assert request.system_prompt is not None
        assert "GROUNDING RULES" in request.system_prompt.upper()


# ──────────────────────────────────────────────────────────────
# Self-Correction Tests
# ──────────────────────────────────────────────────────────────

class TestCorrectLowConfidenceFields:
    """Test the _correct_low_confidence_fields() method."""

    def test_correction_applies_high_confidence_values(self):
        """Corrected values above threshold should replace original values."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "corrected_fields": {
                "patient_dob": {
                    "value": "01/15/1981",
                    "confidence": 0.95,
                    "method": "character-by-character",
                    "differs_from_original": True,
                },
                "patient_id": {
                    "value": "MRN12346",
                    "confidence": 0.92,
                    "method": "character-by-character",
                    "differs_from_original": True,
                },
            },
            "correction_summary": {
                "total_corrected": 2,
                "total_confirmed": 0,
                "overall_confidence": 0.94,
            },
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_self_correction=True,
            confidence_threshold=0.85,
        )

        record = _make_record()
        validation_result = {
            "field_validations": [
                {"field_name": "patient_dob", "issue": "Year digit misread"},
                {"field_name": "patient_id", "issue": "Last digit incorrect"},
            ],
        }

        corrected = extractor._correct_low_confidence_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            fields_to_correct=["patient_dob", "patient_id"],
            validation_result=validation_result,
            schema=_make_schema(),
        )

        assert corrected.fields["patient_dob"] == "01/15/1981"
        assert corrected.fields["patient_id"] == "MRN12346"

    def test_correction_rejects_low_confidence(self):
        """Corrected values below threshold should NOT replace originals."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "corrected_fields": {
                "patient_dob": {
                    "value": "01/15/1981",
                    "confidence": 0.50,  # Below threshold
                    "method": "guessed",
                    "differs_from_original": True,
                },
            },
            "correction_summary": {
                "total_corrected": 0,
                "total_confirmed": 0,
                "overall_confidence": 0.50,
            },
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_self_correction=True,
            confidence_threshold=0.85,
        )

        record = _make_record()
        original_dob = record.fields["patient_dob"]

        corrected = extractor._correct_low_confidence_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            fields_to_correct=["patient_dob"],
            validation_result={"field_validations": []},
            schema=_make_schema(),
        )

        # Value should NOT have changed
        assert corrected.fields["patient_dob"] == original_dob

    def test_correction_no_op_when_empty_fields(self):
        """When no fields need correction, method returns record unchanged."""
        mock_client = MagicMock()

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_self_correction=True,
        )

        record = _make_record()
        original_fields = dict(record.fields)

        corrected = extractor._correct_low_confidence_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            fields_to_correct=[],  # Empty list
            validation_result={"field_validations": []},
            schema=_make_schema(),
        )

        # No VLM call should have been made
        mock_client.send_vision_request.assert_not_called()
        assert corrected.fields == original_fields

    def test_correction_updates_confidence(self):
        """Record confidence should be updated if correction improves it."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "corrected_fields": {
                "patient_dob": {
                    "value": "01/15/1981",
                    "confidence": 0.97,
                    "differs_from_original": True,
                },
            },
            "correction_summary": {
                "total_corrected": 1,
                "overall_confidence": 0.96,
            },
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(
            client=mock_client,
            confidence_threshold=0.85,
        )

        record = _make_record(confidence=0.80)

        corrected = extractor._correct_low_confidence_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            fields_to_correct=["patient_dob"],
            validation_result={"field_validations": []},
            schema=_make_schema(),
        )

        # Confidence should have increased from 0.80 to 0.96
        assert corrected.confidence == 0.96


# ──────────────────────────────────────────────────────────────
# Pipeline Integration Tests
# ──────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Test the extract → validate → correct pipeline flow."""

    def test_pipeline_skips_validation_when_disabled(self):
        """With validation disabled, only extraction VLM calls are made."""
        mock_client = MagicMock()

        # Mock responses for: detect_type, generate_schema, detect_boundaries, extract_record
        mock_responses = [
            # detect_document_type
            MagicMock(has_json=True, parsed_json={
                "document_type": "test_doc",
                "entity_type": "patient",
                "primary_identifier_field": "name",
                "confidence": 0.95,
            }),
            # generate_schema
            MagicMock(has_json=True, parsed_json={
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [{"field_name": "name", "field_type": "text", "description": "Name"}],
                "total_field_count": 1,
            }),
            # detect_record_boundaries (1 record)
            MagicMock(has_json=True, parsed_json={
                "total_records": 1,
                "records": [{
                    "record_id": 1,
                    "primary_identifier": "Smith",
                    "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    "visual_separator": "line",
                }],
            }),
            # extract_single_record
            MagicMock(has_json=True, parsed_json={
                "record_id": 1,
                "primary_identifier": "Smith",
                "fields": {"name": "Smith"},
                "confidence": 0.90,
            }),
        ]
        mock_client.send_vision_request.side_effect = mock_responses

        # Phase 2 DISABLED
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=False,
            enable_self_correction=False,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # Should have exactly 4 VLM calls (no validation, no correction)
        assert mock_client.send_vision_request.call_count == 4
        assert result.total_records == 1

    def test_pipeline_runs_validation_when_enabled(self):
        """With validation enabled, extra VLM calls are made for validation."""
        mock_client = MagicMock()

        mock_responses = [
            # detect_document_type
            MagicMock(has_json=True, parsed_json={
                "document_type": "test_doc",
                "entity_type": "patient",
                "primary_identifier_field": "name",
                "confidence": 0.95,
            }),
            # generate_schema
            MagicMock(has_json=True, parsed_json={
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [{"field_name": "name", "field_type": "text", "description": "Name"}],
                "total_field_count": 1,
            }),
            # detect_record_boundaries
            MagicMock(has_json=True, parsed_json={
                "total_records": 1,
                "records": [{
                    "record_id": 1,
                    "primary_identifier": "Smith",
                    "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    "visual_separator": "line",
                }],
            }),
            # extract_single_record
            MagicMock(has_json=True, parsed_json={
                "record_id": 1,
                "primary_identifier": "Smith",
                "fields": {"name": "Smith"},
                "confidence": 0.90,
            }),
            # _validate_extraction (Phase 2)
            MagicMock(has_json=True, parsed_json={
                "field_validations": [
                    {"field_name": "name", "is_correct": True, "confidence": 0.95},
                ],
                "overall_confidence": 0.95,
            }),
        ]
        mock_client.send_vision_request.side_effect = mock_responses

        # Validation ON, correction OFF
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
            enable_self_correction=False,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 extraction calls + 1 validation call = 5
        assert mock_client.send_vision_request.call_count == 5
        assert result.total_records == 1

    def test_pipeline_runs_correction_when_validation_finds_issues(self):
        """With both enabled, correction runs when validation finds issues."""
        mock_client = MagicMock()

        mock_responses = [
            # detect_document_type
            MagicMock(has_json=True, parsed_json={
                "document_type": "test_doc",
                "entity_type": "patient",
                "primary_identifier_field": "name",
                "confidence": 0.95,
            }),
            # generate_schema
            MagicMock(has_json=True, parsed_json={
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [{"field_name": "name", "field_type": "text", "description": "Name"}],
                "total_field_count": 1,
            }),
            # detect_record_boundaries
            MagicMock(has_json=True, parsed_json={
                "total_records": 1,
                "records": [{
                    "record_id": 1,
                    "primary_identifier": "Smlth",
                    "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    "visual_separator": "line",
                }],
            }),
            # extract_single_record (incorrect 'name')
            MagicMock(has_json=True, parsed_json={
                "record_id": 1,
                "primary_identifier": "Smlth",
                "fields": {"name": "Smlth"},  # Wrong
                "confidence": 0.75,
            }),
            # _validate_extraction -> flags 'name' as incorrect
            MagicMock(has_json=True, parsed_json={
                "field_validations": [
                    {
                        "field_name": "name",
                        "is_correct": False,
                        "corrected_value": "Smith",
                        "confidence": 0.92,
                        "issue": "Character misread: 'l' should be 'i'",
                    },
                ],
                "overall_confidence": 0.82,
            }),
            # _correct_low_confidence_fields -> corrects 'name'
            MagicMock(has_json=True, parsed_json={
                "corrected_fields": {
                    "name": {
                        "value": "Smith",
                        "confidence": 0.96,
                        "method": "character-by-character",
                        "differs_from_original": True,
                    },
                },
                "correction_summary": {
                    "total_corrected": 1,
                    "overall_confidence": 0.96,
                },
            }),
        ]
        mock_client.send_vision_request.side_effect = mock_responses

        # Both Phase 2 features ON
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
            enable_self_correction=True,
            confidence_threshold=0.85,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 extraction + 1 validation + 1 correction = 6
        assert mock_client.send_vision_request.call_count == 6
        assert result.total_records == 1

        # The record should have corrected value
        assert result.records[0].fields["name"] == "Smith"

    def test_pipeline_no_correction_when_validation_passes(self):
        """Correction should be skipped when validation finds no issues."""
        mock_client = MagicMock()

        mock_responses = [
            # detect_document_type
            MagicMock(has_json=True, parsed_json={
                "document_type": "test_doc",
                "entity_type": "patient",
                "primary_identifier_field": "name",
                "confidence": 0.95,
            }),
            # generate_schema
            MagicMock(has_json=True, parsed_json={
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [{"field_name": "name", "field_type": "text", "description": "Name"}],
                "total_field_count": 1,
            }),
            # detect_record_boundaries
            MagicMock(has_json=True, parsed_json={
                "total_records": 1,
                "records": [{
                    "record_id": 1,
                    "primary_identifier": "Smith",
                    "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    "visual_separator": "line",
                }],
            }),
            # extract_single_record
            MagicMock(has_json=True, parsed_json={
                "record_id": 1,
                "primary_identifier": "Smith",
                "fields": {"name": "Smith"},
                "confidence": 0.95,
            }),
            # _validate_extraction -> all correct
            MagicMock(has_json=True, parsed_json={
                "field_validations": [
                    {"field_name": "name", "is_correct": True, "confidence": 0.98},
                ],
                "overall_confidence": 0.98,
            }),
            # NO correction call expected!
        ]
        mock_client.send_vision_request.side_effect = mock_responses

        # Both enabled, but correction won't run if validation passes
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=True,
            enable_self_correction=True,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 extraction + 1 validation = 5 (no correction needed)
        assert mock_client.send_vision_request.call_count == 5


# ──────────────────────────────────────────────────────────────
# Feature Flag Gating Tests
# ──────────────────────────────────────────────────────────────

class TestFeatureFlagGating:
    """Test that Phase 2 methods are only called when flags are set."""

    @patch.object(MultiRecordExtractor, "_validate_extraction")
    @patch.object(MultiRecordExtractor, "_correct_low_confidence_fields")
    def test_no_validation_when_flag_off(self, mock_correct, mock_validate):
        """_validate_extraction should NOT be called when flag is off."""
        mock_client = MagicMock()

        mock_responses = [
            MagicMock(has_json=True, parsed_json={
                "document_type": "t", "entity_type": "p", "primary_identifier_field": "n",
                "confidence": 0.9,
            }),
            MagicMock(has_json=True, parsed_json={
                "schema_id": "t", "entity_type": "p", "fields": [], "total_field_count": 0,
            }),
            MagicMock(has_json=True, parsed_json={
                "total_records": 1, "records": [
                    {"record_id": 1, "primary_identifier": "X", "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1}},
                ],
            }),
            MagicMock(has_json=True, parsed_json={
                "record_id": 1, "primary_identifier": "X", "fields": {}, "confidence": 0.9,
            }),
        ]
        mock_client.send_vision_request.side_effect = mock_responses

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=False,  # OFF
            enable_self_correction=False,
        )

        extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
        )

        mock_validate.assert_not_called()
        mock_correct.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
