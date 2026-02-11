"""
Unit tests for Phase 1 zero-shot accuracy improvements to multi_record.py.

Tests cover:
- ExtractedRecord dataclass
- System prompt generation
- Adaptive temperature calculation
- Chain-of-thought prompt integration
- Exponential backoff retry logic
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.multi_record import (
    ExtractedRecord,
    MultiRecordExtractor,
    RecordBoundary,
)


class TestExtractedRecord:
    """Test ExtractedRecord dataclass."""

    def test_extracted_record_basic(self):
        """Test creating ExtractedRecord with all fields."""
        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="John Smith",
            entity_type="patient",
            fields={"name": "John Smith", "age": 45},
            confidence=0.92,
            extraction_time_ms=500,
        )

        assert record.record_id == 1
        assert record.page_number == 1
        assert record.primary_identifier == "John Smith"
        assert record.entity_type == "patient"
        assert record.fields == {"name": "John Smith", "age": 45}
        assert record.confidence == 0.92
        assert record.extraction_time_ms == 500

    def test_fields_dict_access(self):
        """Test that fields dict is directly accessible."""
        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields={"field1": "value1"},
            confidence=0.9,
            extraction_time_ms=100,
        )

        assert record.fields["field1"] == "value1"
        assert isinstance(record.fields, dict)


class TestGroundingSystemPrompt:
    """Test system prompt generation."""

    def test_grounding_prompt_exists(self):
        """Test that grounding system prompt is generated."""
        extractor = MultiRecordExtractor()
        prompt = extractor._build_grounding_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_grounding_prompt_contains_rules(self):
        """Test that system prompt contains key grounding rules."""
        extractor = MultiRecordExtractor()
        prompt = extractor._build_grounding_system_prompt()

        # Check for critical grounding phrases from shared grounding_rules module
        assert "only extract" in prompt.lower()
        assert "clearly visible" in prompt.lower() or "clearly see" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "null" in prompt.lower()
        assert "no guessing" in prompt.lower() or "never guess" in prompt.lower()

    def test_grounding_prompt_reasoning_process(self):
        """Test that prompt includes reasoning guidance."""
        extractor = MultiRecordExtractor()
        prompt = extractor._build_grounding_system_prompt()

        # Should guide VLM through reasoning process
        assert (
            "describe" in prompt.lower() or "identify" in prompt.lower()
            or "reasoning" in prompt.lower()
        )


class TestFieldTypeValidation:
    """Test post-extraction field type validation and coercion."""

    def test_coerce_number_from_currency(self):
        """Test that currency strings are coerced to float."""
        extractor = MultiRecordExtractor()
        assert extractor._coerce_number("$1,234.56") == 1234.56

    def test_coerce_number_plain(self):
        """Test that plain numbers pass through."""
        extractor = MultiRecordExtractor()
        assert extractor._coerce_number(42) == 42
        assert extractor._coerce_number(3.14) == 3.14

    def test_coerce_number_invalid_returns_original(self):
        """Test that non-numeric strings are preserved."""
        extractor = MultiRecordExtractor()
        assert extractor._coerce_number("not a number") == "not a number"

    def test_normalize_date_us_format(self):
        """Test that US date formats are normalized to ISO."""
        extractor = MultiRecordExtractor()
        result = extractor._normalize_date("01/15/2024")
        assert result == "2024-01-15"

    def test_normalize_date_invalid_returns_original(self):
        """Test that unparseable dates are preserved."""
        extractor = MultiRecordExtractor()
        assert extractor._normalize_date("not a date") == "not a date"

    def test_coerce_boolean_strings(self):
        """Test boolean coercion from various string forms."""
        extractor = MultiRecordExtractor()
        assert extractor._coerce_boolean("yes") is True
        assert extractor._coerce_boolean("no") is False
        assert extractor._coerce_boolean("true") is True
        assert extractor._coerce_boolean("false") is False
        assert extractor._coerce_boolean("checked") is True

    def test_validate_field_types_integration(self):
        """Test _validate_field_types applies coercion to a record."""
        extractor = MultiRecordExtractor()
        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields={
                "name": "Smith",
                "total_charge": "$150.00",
                "date_of_birth": "03/15/1985",
                "is_active": "yes",
            },
            confidence=0.9,
            extraction_time_ms=100,
        )
        schema = {
            "fields": [
                {"field_name": "name", "field_type": "text"},
                {"field_name": "total_charge", "field_type": "number"},
                {"field_name": "date_of_birth", "field_type": "date"},
                {"field_name": "is_active", "field_type": "boolean"},
            ]
        }
        result = extractor._validate_field_types(record, schema)
        assert result.fields["name"] == "Smith"  # text unchanged
        assert result.fields["total_charge"] == 150.0
        assert result.fields["date_of_birth"] == "1985-03-15"
        assert result.fields["is_active"] is True

    def test_validate_field_types_null_preserved(self):
        """Test that null values are not coerced."""
        extractor = MultiRecordExtractor()
        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields={"amount": None},
            confidence=0.9,
            extraction_time_ms=100,
        )
        schema = {"fields": [{"field_name": "amount", "field_type": "number"}]}
        result = extractor._validate_field_types(record, schema)
        assert result.fields["amount"] is None


class TestAdaptiveTemperature:
    """Test adaptive temperature calculation."""

    def test_id_fields_deterministic(self):
        """Test that ID fields use temperature 0.0."""
        extractor = MultiRecordExtractor()

        id_keywords = ["id", "number", "code", "ssn", "mrn"]
        for keyword in id_keywords:
            temp = extractor._get_adaptive_temperature(
                field_type="text",
                field_name=f"patient_{keyword}",
                retry_count=0,
            )
            assert temp == 0.0, f"Field '{keyword}' should use temperature 0.0"

    def test_date_fields_low_temperature(self):
        """Test that date fields use low temperature."""
        extractor = MultiRecordExtractor()

        temp = extractor._get_adaptive_temperature(
            field_type="date",
            field_name="date_of_birth",
            retry_count=0,
        )
        assert temp == 0.05

    def test_amount_fields_low_temperature(self):
        """Test that amount/charge fields use low temperature."""
        extractor = MultiRecordExtractor()

        amount_keywords = ["amount", "charge", "total", "balance"]
        for keyword in amount_keywords:
            temp = extractor._get_adaptive_temperature(
                field_type="text",
                field_name=keyword,
                retry_count=0,
            )
            assert temp == 0.03, f"Field '{keyword}' should use temperature 0.03"

    def test_description_fields_higher_temperature(self):
        """Test that description fields use higher temperature."""
        extractor = MultiRecordExtractor()

        desc_keywords = ["description", "note", "comment"]
        for keyword in desc_keywords:
            temp = extractor._get_adaptive_temperature(
                field_type="text",
                field_name=keyword,
                retry_count=0,
            )
            assert temp == 0.15, f"Field '{keyword}' should use temperature 0.15"

    def test_base_temperature(self):
        """Test base temperature for generic fields."""
        extractor = MultiRecordExtractor()

        temp = extractor._get_adaptive_temperature(
            field_type="text",
            field_name="generic_field",
            retry_count=0,
        )
        assert temp == 0.1

    def test_retry_temperature_escalation(self):
        """Test that temperature increases with retry count."""
        extractor = MultiRecordExtractor()

        temp_0 = extractor._get_adaptive_temperature(
            field_type="text",
            field_name="field",
            retry_count=0,
        )
        temp_1 = extractor._get_adaptive_temperature(
            field_type="text",
            field_name="field",
            retry_count=1,
        )
        temp_2 = extractor._get_adaptive_temperature(
            field_type="text",
            field_name="field",
            retry_count=2,
        )

        assert temp_1 > temp_0
        assert temp_2 > temp_1
        assert temp_2 <= 0.3  # Max temperature cap


class TestExponentialBackoff:
    """Test exponential backoff retry logic."""

    @patch("src.extraction.multi_record.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Test that retry delays follow exponential backoff."""
        # Create mock client
        mock_client = MagicMock()

        # Mock VLM client to fail 3 times then succeed
        mock_response = MagicMock()
        mock_response.has_json = False
        mock_response.content = '{"result": "success"}'

        call_count = [0]

        def failing_then_success(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Simulated failure")
            mock_response.has_json = True
            mock_response.parsed_json = {"result": "success"}
            return mock_response

        mock_client.send_vision_request = failing_then_success

        extractor = MultiRecordExtractor(client=mock_client)

        # Call _send_vision_json which should retry
        result = extractor._send_vision_json(
            image_data="data:image/png;base64,fake",
            prompt="test",
            max_retries=3,
        )

        # Verify exponential backoff: 2^0=1s, 2^1=2s
        assert mock_sleep.call_count == 2
        sleep_delays = [call.args[0] for call in mock_sleep.call_args_list]

        # Delays should be: 1s (2^0), 2s (2^1)
        assert sleep_delays[0] == 1
        assert sleep_delays[1] == 2

    @patch("src.extraction.multi_record.time.sleep")
    def test_max_backoff_delay(self, mock_sleep):
        """Test that backoff delay caps at 10 seconds."""
        # Create mock client
        mock_client = MagicMock()

        call_count = [0]

        def always_fail(*args, **kwargs):
            call_count[0] += 1
            raise Exception("Always fails")

        mock_client.send_vision_request = always_fail

        extractor = MultiRecordExtractor(client=mock_client)

        # Attempt with 5 retries (should fail and raise)
        with pytest.raises(RuntimeError):
            extractor._send_vision_json(
                image_data="data:image/png;base64,fake",
                prompt="test",
                max_retries=5,
            )

        # Check that delay caps at 10 seconds
        sleep_delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16 (capped at 10)
        assert sleep_delays == [1, 2, 4, 8]

    @patch("src.extraction.multi_record.time.sleep")
    def test_json_parse_error_reformulation(self, mock_sleep):
        """Test that JSON parse errors trigger prompt reformulation."""
        # Create mock client
        mock_client = MagicMock()

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.has_json = False
        mock_response.content = "This is not valid JSON"

        call_count = [0]
        reformulated = [False]

        def check_reformulation(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call - return invalid JSON
                return mock_response
            else:
                # Second call - check if prompt was reformulated
                request = args[0]
                if "CRITICAL: Return ONLY valid JSON" in request.prompt:
                    reformulated[0] = True
                # Now return valid JSON
                mock_response.content = '{"success": true}'
                return mock_response

        mock_client.send_vision_request = check_reformulation

        extractor = MultiRecordExtractor(client=mock_client)

        extractor._send_vision_json(
            image_data="data:image/png;base64,fake",
            prompt="original prompt",
            max_retries=3,
        )

        assert reformulated[0], "Prompt should be reformulated after JSON parse error"


class TestChainOfThoughtPrompts:
    """Test that CoT prompts include step-by-step reasoning instructions."""

    def test_detect_document_type_cot(self):
        """Test detect_document_type uses step-by-step prompting."""
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "document_type": "medical_superbill",
            "entity_type": "patient",
            "primary_identifier_field": "patient_name",
            "confidence": 0.95,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        result = extractor.detect_document_type("data:image/png;base64,fake")

        # Verify step-by-step reasoning is in the prompt
        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]
        assert "step by step" in request.prompt.lower()

        # Verify system prompt is used
        assert request.system_prompt is not None
        assert len(request.system_prompt) > 0

        # Verify result fields
        assert "document_type" in result
        assert "entity_type" in result
        assert result["document_type"] == "medical_superbill"

    def test_generate_schema_cot(self):
        """Test generate_schema uses step-by-step prompting."""
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "schema_id": "test",
            "entity_type": "patient",
            "fields": [],
            "total_field_count": 0,
            "confidence": 0.9,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        result = extractor.generate_schema(
            "data:image/png;base64,fake",
            "medical_superbill",
            "patient",
        )

        # Verify step-by-step reasoning is in the prompt
        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]
        assert "step by step" in request.prompt.lower()

        assert "schema_id" in result
        assert "fields" in result


class TestOutputFormat:
    """Test that extraction methods return correct output structure."""

    def test_extract_single_record_output(self):
        """Test extract_single_record returns correct ExtractedRecord."""
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "record_id": 1,
            "primary_identifier": "Smith, John",
            "fields": {"name": "Smith, John", "age": 45},
            "confidence": 0.92,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)

        boundary = RecordBoundary(
            record_id=1,
            primary_identifier="Smith, John",
            bounding_box={"top": 0.0, "left": 0.0, "bottom": 0.5, "right": 1.0},
            visual_separator="line",
            entity_type="patient",
        )

        schema = {
            "fields": [
                {
                    "field_name": "name",
                    "field_type": "text",
                    "description": "Patient name",
                }
            ]
        }

        record = extractor.extract_single_record(
            "data:image/png;base64,fake",
            boundary,
            schema,
            page_number=1,
        )

        # Old code expects these fields
        assert hasattr(record, "record_id")
        assert hasattr(record, "page_number")
        assert hasattr(record, "primary_identifier")
        assert hasattr(record, "entity_type")
        assert hasattr(record, "fields")
        assert hasattr(record, "confidence")
        assert hasattr(record, "extraction_time_ms")

        # Fields dict should be directly accessible
        assert isinstance(record.fields, dict)
        assert record.fields["name"] == "Smith, John"


class TestCalibratedConfidence:
    """Test multi-factor confidence calibration."""

    def _make_record(self, confidence=0.9, fields=None):
        return ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields=fields or {"name": "Smith", "age": 45, "id": "MRN123"},
            confidence=confidence,
            extraction_time_ms=100,
        )

    def _make_schema(self):
        return {
            "fields": [
                {"field_name": "name", "field_type": "text"},
                {"field_name": "age", "field_type": "number"},
                {"field_name": "id", "field_type": "text"},
            ]
        }

    def test_baseline_no_validation_no_consensus(self):
        """Without Phase 2/3, calibration uses raw confidence + completeness."""
        extractor = MultiRecordExtractor()
        record = self._make_record(confidence=0.9)
        schema = self._make_schema()

        result = extractor._calibrate_confidence(record, schema)

        # 0.40*0.9 + 0.25*1.0 + 0.20*1.0 + 0.15*1.0 = 0.36 + 0.25 + 0.20 + 0.15 = 0.96
        assert result == 0.96

    def test_low_completeness_reduces_confidence(self):
        """Missing fields should reduce calibrated confidence."""
        extractor = MultiRecordExtractor()
        record = self._make_record(
            confidence=0.9,
            fields={"name": "Smith", "age": None, "id": None},
        )
        schema = self._make_schema()

        result = extractor._calibrate_confidence(record, schema)

        # completeness = 1/3 = 0.333
        # 0.40*0.9 + 0.25*1.0 + 0.20*0.333 + 0.15*1.0 = 0.36 + 0.25 + 0.067 + 0.15 = 0.827
        assert 0.82 <= result <= 0.83

    def test_validation_failures_reduce_confidence(self):
        """Validation failures should reduce calibrated confidence."""
        extractor = MultiRecordExtractor()
        record = self._make_record(confidence=0.9)
        schema = self._make_schema()

        validation_result = {
            "field_validations": [
                {"field": "name", "is_correct": True},
                {"field": "age", "is_correct": False},
                {"field": "id", "is_correct": True},
            ]
        }

        result = extractor._calibrate_confidence(
            record, schema, validation_result=validation_result,
        )

        # val_score = 2/3 = 0.667
        # 0.40*0.9 + 0.25*0.667 + 0.20*1.0 + 0.15*1.0 = 0.36 + 0.167 + 0.20 + 0.15 = 0.877
        assert 0.87 <= result <= 0.88

    def test_consensus_disagreement_reduces_confidence(self):
        """Consensus disagreements should reduce calibrated confidence."""
        extractor = MultiRecordExtractor()
        record = self._make_record(confidence=0.9)
        schema = self._make_schema()

        result = extractor._calibrate_confidence(
            record, schema, consensus_agreed=1, consensus_total=3,
        )

        # consensus_score = max(0.7, 1/3) = 0.7
        # 0.40*0.9 + 0.25*1.0 + 0.20*1.0 + 0.15*0.7 = 0.36 + 0.25 + 0.20 + 0.105 = 0.915
        assert result == 0.915

    def test_all_factors_combined(self):
        """Test with all signals contributing."""
        extractor = MultiRecordExtractor()
        record = self._make_record(
            confidence=0.8,
            fields={"name": "Smith", "age": None, "id": "MRN123"},
        )
        schema = self._make_schema()

        validation_result = {
            "field_validations": [
                {"field": "name", "is_correct": True},
                {"field": "id", "is_correct": False},
            ]
        }

        result = extractor._calibrate_confidence(
            record, schema,
            validation_result=validation_result,
            consensus_agreed=1,
            consensus_total=2,
        )

        # raw=0.8, completeness=2/3=0.667, val_score=1/2=0.5, consensus=max(0.7,0.5)=0.7
        # 0.40*0.8 + 0.25*0.5 + 0.20*0.667 + 0.15*0.7 = 0.32 + 0.125 + 0.133 + 0.105 = 0.683
        assert 0.68 <= result <= 0.69

    def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        extractor = MultiRecordExtractor()
        record = self._make_record(confidence=1.0)
        schema = self._make_schema()

        result = extractor._calibrate_confidence(record, schema)

        assert result <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
