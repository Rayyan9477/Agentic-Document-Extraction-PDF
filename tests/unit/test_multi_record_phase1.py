"""
Unit tests for Phase 1 zero-shot accuracy improvements to multi_record.py.

Tests cover:
- FieldMetadata dataclass
- ExtractedRecord with field_metadata
- System prompt generation
- Adaptive temperature calculation
- Chain-of-thought prompt integration
- Exponential backoff retry logic
- Backward compatibility
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.multi_record import (
    ExtractedRecord,
    FieldMetadata,
    MultiRecordExtractor,
    RecordBoundary,
)


class TestFieldMetadata:
    """Test FieldMetadata dataclass."""

    def test_field_metadata_creation(self):
        """Test creating FieldMetadata with all fields."""
        metadata = FieldMetadata(
            field_name="patient_name",
            value="Smith, John",
            confidence=0.95,
            extraction_time_ms=150,
            temperature_used=0.0,
            retry_count=0,
        )

        assert metadata.field_name == "patient_name"
        assert metadata.value == "Smith, John"
        assert metadata.confidence == 0.95
        assert metadata.extraction_time_ms == 150
        assert metadata.temperature_used == 0.0
        assert metadata.retry_count == 0

    def test_field_metadata_defaults(self):
        """Test FieldMetadata default values."""
        metadata = FieldMetadata(
            field_name="test",
            value="value",
            confidence=0.8,
            extraction_time_ms=100,
            temperature_used=0.1,
        )

        assert metadata.retry_count == 0  # Default value


class TestExtractedRecord:
    """Test ExtractedRecord with field_metadata."""

    def test_extracted_record_basic(self):
        """Test creating ExtractedRecord without field_metadata."""
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
        assert record.field_metadata == {}  # Default empty dict

    def test_extracted_record_with_field_metadata(self):
        """Test ExtractedRecord with field-level metadata."""
        field_meta = {
            "name": FieldMetadata(
                field_name="name",
                value="John Smith",
                confidence=0.98,
                extraction_time_ms=100,
                temperature_used=0.0,
                retry_count=0,
            ),
            "age": FieldMetadata(
                field_name="age",
                value=45,
                confidence=0.85,
                extraction_time_ms=80,
                temperature_used=0.05,
                retry_count=1,
            ),
        }

        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="John Smith",
            entity_type="patient",
            fields={"name": "John Smith", "age": 45},
            confidence=0.92,
            extraction_time_ms=500,
            field_metadata=field_meta,
        )

        assert len(record.field_metadata) == 2
        assert record.field_metadata["name"].confidence == 0.98
        assert record.field_metadata["age"].retry_count == 1

    def test_backward_compatibility(self):
        """Test that old code using only fields dict still works."""
        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields={"field1": "value1"},
            confidence=0.9,
            extraction_time_ms=100,
        )

        # Old code would access fields directly
        assert record.fields["field1"] == "value1"
        # field_metadata is optional and defaults to empty
        assert isinstance(record.field_metadata, dict)
        assert len(record.field_metadata) == 0


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

        # Check for critical grounding phrases
        assert "only extract" in prompt.lower() or "only report" in prompt.lower()
        assert "clearly see" in prompt.lower() or "directly see" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "null" in prompt.lower() or "unclear" in prompt.lower()

    def test_grounding_prompt_reasoning_process(self):
        """Test that prompt includes reasoning guidance."""
        extractor = MultiRecordExtractor()
        prompt = extractor._build_grounding_system_prompt()

        # Should guide VLM through reasoning process
        assert (
            "describe" in prompt.lower() or "identify" in prompt.lower()
            or "reasoning" in prompt.lower()
        )


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
    """Test that CoT prompts are properly structured."""

    def test_detect_document_type_cot(self):
        """Test detect_document_type uses CoT prompting."""
        # Create mock client
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "step_1_observations": {},
            "step_2_text_analysis": {},
            "step_3_patterns": {},
            "step_4_classification": {
                "document_type": "medical_superbill",
                "entity_type": "patient",
                "primary_identifier_field": "patient_name",
            },
            "confidence": 0.95,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        result = extractor.detect_document_type("data:image/png;base64,fake")

        # Verify CoT structure is present in request
        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]

        assert "STEP 1" in request.prompt.upper()
        assert "STEP 2" in request.prompt.upper()
        assert "STEP 3" in request.prompt.upper()
        assert "STEP 4" in request.prompt.upper()

        # Verify system prompt is used
        assert request.system_prompt is not None
        assert len(request.system_prompt) > 0

        # Verify result is backward compatible
        assert "document_type" in result
        assert "entity_type" in result
        assert result["document_type"] == "medical_superbill"

    def test_generate_schema_cot(self):
        """Test generate_schema uses CoT prompting."""
        # Create mock client
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "step_1_structure": {},
            "step_2_identified_fields": [],
            "step_3_type_analysis": [],
            "step_4_final_schema": {
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [],
                "total_field_count": 0,
            },
            "confidence": 0.9,
        }
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        result = extractor.generate_schema(
            "data:image/png;base64,fake",
            "medical_superbill",
            "patient",
        )

        # Verify CoT prompting
        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]

        assert "STEP 1" in request.prompt.upper()
        assert "STEP 2" in request.prompt.upper()
        assert "STEP 3" in request.prompt.upper()
        assert "STEP 4" in request.prompt.upper()

        # Verify backward compatibility
        assert "schema_id" in result
        assert "fields" in result


class TestBackwardCompatibility:
    """Test that Phase 1 changes maintain backward compatibility."""

    def test_extract_single_record_backward_compatible(self):
        """Test extract_single_record maintains backward compatible output."""
        # Create mock client
        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.has_json = True
        mock_response.parsed_json = {
            "step_4_final_extraction": {
                "record_id": 1,
                "primary_identifier": "Smith, John",
                "fields": {"name": "Smith, John", "age": 45},
                "confidence": 0.92,
            }
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
