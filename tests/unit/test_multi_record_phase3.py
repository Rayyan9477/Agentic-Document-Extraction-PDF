"""
Unit tests for Phase 3 consensus & ensemble improvements to multi_record.py.

Tests cover:
- Constructor Phase 3 configuration
- Critical field identification
- Dual-pass consensus extraction
- Tie-breaker disagreement resolution
- Pipeline integration (extract → validate → correct → consensus)
- Feature flag gating
"""

from unittest.mock import MagicMock, patch

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
    """Create a test schema with critical and non-critical fields."""
    return {
        "fields": [
            {"field_name": "patient_name", "field_type": "text", "description": "Name"},
            {"field_name": "patient_dob", "field_type": "date", "description": "Date of birth"},
            {"field_name": "patient_id", "field_type": "text", "description": "Patient ID"},
            {"field_name": "total_charge", "field_type": "number", "description": "Total charge"},
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
            "total_charge": "$150.00",
            "diagnosis": "Hypertension",
        },
        confidence=confidence,
        extraction_time_ms=500,
    )


def _base_pipeline_responses():
    """Return the 4 base mock responses for the extraction pipeline."""
    return [
        # detect_document_type
        MagicMock(has_json=True, parsed_json={
            "step_4_classification": {
                "document_type": "test_doc",
                "entity_type": "patient",
                "primary_identifier_field": "name",
            },
            "confidence": 0.95,
        }),
        # generate_schema
        MagicMock(has_json=True, parsed_json={
            "step_4_final_schema": {
                "schema_id": "test",
                "entity_type": "patient",
                "fields": [
                    {"field_name": "patient_id", "field_type": "text", "description": "ID"},
                    {"field_name": "patient_name", "field_type": "text", "description": "Name"},
                    {"field_name": "total_charge", "field_type": "number", "description": "Charge"},
                    {"field_name": "diagnosis", "field_type": "text", "description": "Diagnosis"},
                ],
                "total_field_count": 4,
            },
        }),
        # detect_record_boundaries (1 record)
        MagicMock(has_json=True, parsed_json={
            "step_4_boundaries": {
                "total_records": 1,
                "records": [{
                    "record_id": 1,
                    "primary_identifier": "Smith",
                    "bounding_box": {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    "visual_separator": "line",
                }],
            },
        }),
        # extract_single_record
        MagicMock(has_json=True, parsed_json={
            "step_4_final_extraction": {
                "record_id": 1,
                "primary_identifier": "Smith",
                "fields": {
                    "patient_id": "MRN12345",
                    "patient_name": "Smith",
                    "total_charge": "$150.00",
                    "diagnosis": "Hypertension",
                },
                "confidence": 0.90,
            },
        }),
    ]


# ──────────────────────────────────────────────────────────────
# Constructor Tests
# ──────────────────────────────────────────────────────────────

class TestPhase3Constructor:
    """Test Phase 3 constructor parameters."""

    def test_default_consensus_disabled(self):
        """Consensus should be disabled by default."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(client=mock_client)
        assert extractor._enable_consensus is False

    def test_enable_consensus(self):
        """Consensus can be enabled via constructor."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )
        assert extractor._enable_consensus is True

    def test_default_critical_keywords(self):
        """Default critical field keywords should be populated."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(client=mock_client)
        assert "id" in extractor._critical_field_keywords
        assert "date" in extractor._critical_field_keywords
        assert "amount" in extractor._critical_field_keywords
        assert "total" in extractor._critical_field_keywords
        assert "dob" in extractor._critical_field_keywords

    def test_custom_critical_keywords(self):
        """Custom critical field keywords override defaults."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            critical_field_keywords=["invoice_number", "total"],
        )
        assert extractor._critical_field_keywords == ["invoice_number", "total"]
        assert "id" not in extractor._critical_field_keywords


# ──────────────────────────────────────────────────────────────
# Critical Field Identification Tests
# ──────────────────────────────────────────────────────────────

class TestIdentifyCriticalFields:
    """Test _identify_critical_fields() method."""

    def test_identifies_id_fields(self):
        """Fields containing 'id' keyword should be critical."""
        extractor = MultiRecordExtractor(client=MagicMock())
        schema = {"fields": [
            {"field_name": "patient_id", "field_type": "text"},
            {"field_name": "diagnosis", "field_type": "text"},
        ]}
        critical = extractor._identify_critical_fields(schema)
        assert "patient_id" in critical
        assert "diagnosis" not in critical

    def test_identifies_date_fields(self):
        """Fields containing 'date' or 'dob' should be critical."""
        extractor = MultiRecordExtractor(client=MagicMock())
        schema = {"fields": [
            {"field_name": "patient_dob", "field_type": "date"},
            {"field_name": "service_date", "field_type": "date"},
            {"field_name": "findings", "field_type": "text"},
        ]}
        critical = extractor._identify_critical_fields(schema)
        assert "patient_dob" in critical
        assert "service_date" in critical
        assert "findings" not in critical

    def test_identifies_amount_fields(self):
        """Fields containing 'amount', 'charge', 'total', 'balance' should be critical."""
        extractor = MultiRecordExtractor(client=MagicMock())
        schema = {"fields": [
            {"field_name": "total_charge", "field_type": "number"},
            {"field_name": "balance_due", "field_type": "number"},
            {"field_name": "notes", "field_type": "text"},
        ]}
        critical = extractor._identify_critical_fields(schema)
        assert "total_charge" in critical
        assert "balance_due" in critical
        assert "notes" not in critical

    def test_no_critical_fields(self):
        """Schema with no critical fields returns empty list."""
        extractor = MultiRecordExtractor(client=MagicMock())
        schema = {"fields": [
            {"field_name": "diagnosis", "field_type": "text"},
            {"field_name": "findings", "field_type": "text"},
        ]}
        critical = extractor._identify_critical_fields(schema)
        assert critical == []

    def test_custom_keywords_used(self):
        """Custom keywords are used for identification."""
        extractor = MultiRecordExtractor(
            client=MagicMock(),
            critical_field_keywords=["custom"],
        )
        schema = {"fields": [
            {"field_name": "custom_field", "field_type": "text"},
            {"field_name": "patient_id", "field_type": "text"},
        ]}
        critical = extractor._identify_critical_fields(schema)
        assert "custom_field" in critical
        # 'id' not in custom keywords, so patient_id is NOT critical
        assert "patient_id" not in critical


# ──────────────────────────────────────────────────────────────
# Consensus Extraction Tests
# ──────────────────────────────────────────────────────────────

class TestConsensusExtractCriticalFields:
    """Test _consensus_extract_critical_fields() method."""

    def test_both_passes_agree(self):
        """When both passes agree, the value is used and no tie-breaker runs."""
        mock_client = MagicMock()

        # Pass 1 and Pass 2 return identical values
        pass1 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.95},
            }
        })
        pass2 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.93},
            }
        })
        mock_client.send_vision_request.side_effect = [pass1, pass2]

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )
        record = _make_record()

        result = extractor._consensus_extract_critical_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            schema=_make_schema(),
            critical_fields=["patient_id"],
        )

        assert result.fields["patient_id"] == "MRN12345"
        # Exactly 2 VLM calls (pass 1 + pass 2, no tie-breaker)
        assert mock_client.send_vision_request.call_count == 2

    def test_disagreement_triggers_tiebreaker(self):
        """When passes disagree, a tie-breaker VLM call resolves the conflict."""
        mock_client = MagicMock()

        pass1 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.90},
            }
        })
        pass2 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12346", "confidence": 0.88},
            }
        })
        tiebreaker = MagicMock(has_json=True, parsed_json={
            "value": "MRN12345",
            "confidence": 0.97,
            "reasoning": "Last digit is clearly 5",
        })
        mock_client.send_vision_request.side_effect = [pass1, pass2, tiebreaker]

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )
        record = _make_record()

        result = extractor._consensus_extract_critical_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            schema=_make_schema(),
            critical_fields=["patient_id"],
        )

        assert result.fields["patient_id"] == "MRN12345"
        # 2 passes + 1 tie-breaker = 3
        assert mock_client.send_vision_request.call_count == 3

    def test_empty_critical_fields_no_vlm_calls(self):
        """When no critical fields, return record unchanged with zero VLM calls."""
        mock_client = MagicMock()
        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )
        record = _make_record()
        original_fields = dict(record.fields)

        result = extractor._consensus_extract_critical_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            schema=_make_schema(),
            critical_fields=[],
        )

        mock_client.send_vision_request.assert_not_called()
        assert result.fields == original_fields

    def test_mixed_agreement_and_disagreement(self):
        """Only disagreeing fields trigger a tie-breaker."""
        mock_client = MagicMock()

        # patient_id agrees, patient_dob disagrees
        pass1 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.95},
                "patient_dob": {"value": "01/15/1980", "confidence": 0.90},
            }
        })
        pass2 = MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.93},
                "patient_dob": {"value": "01/15/1981", "confidence": 0.88},
            }
        })
        tiebreaker = MagicMock(has_json=True, parsed_json={
            "value": "01/15/1980",
            "confidence": 0.96,
            "reasoning": "Year is clearly 1980",
        })
        mock_client.send_vision_request.side_effect = [pass1, pass2, tiebreaker]

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )
        record = _make_record()

        result = extractor._consensus_extract_critical_fields(
            page_data_uri="data:image/png;base64,fake",
            record=record,
            boundary=_make_boundary(),
            schema=_make_schema(),
            critical_fields=["patient_id", "patient_dob"],
        )

        assert result.fields["patient_id"] == "MRN12345"
        assert result.fields["patient_dob"] == "01/15/1980"
        # 2 passes + 1 tie-breaker (only for patient_dob) = 3
        assert mock_client.send_vision_request.call_count == 3


# ──────────────────────────────────────────────────────────────
# Disagreement Resolution Tests
# ──────────────────────────────────────────────────────────────

class TestResolveDisagreement:
    """Test _resolve_disagreement() method."""

    def test_returns_resolved_value(self):
        """Tie-breaker should return a resolved value with confidence."""
        mock_client = MagicMock()
        mock_response = MagicMock(has_json=True, parsed_json={
            "value": "MRN12345",
            "confidence": 0.97,
            "reasoning": "Character is clearly 5 not 6",
        })
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        result = extractor._resolve_disagreement(
            page_data_uri="data:image/png;base64,fake",
            boundary=_make_boundary(),
            field_name="patient_id",
            value_1="MRN12345",
            value_2="MRN12346",
            record=_make_record(),
        )

        assert result["value"] == "MRN12345"
        assert result["confidence"] == 0.97
        assert "reasoning" in result

    def test_uses_deterministic_temperature(self):
        """Tie-breaker should use temperature 0.0."""
        mock_client = MagicMock()
        mock_response = MagicMock(has_json=True, parsed_json={
            "value": "test",
            "confidence": 0.9,
            "reasoning": "clear",
        })
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        extractor._resolve_disagreement(
            page_data_uri="data:image/png;base64,fake",
            boundary=_make_boundary(),
            field_name="patient_id",
            value_1="A",
            value_2="B",
            record=_make_record(),
        )

        # Inspect the VisionRequest passed to send_vision_request
        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]
        assert request.temperature == 0.0

    def test_prompt_contains_both_values(self):
        """Prompt should contain both disagreeing values."""
        mock_client = MagicMock()
        mock_response = MagicMock(has_json=True, parsed_json={
            "value": "MRN12345",
            "confidence": 0.9,
            "reasoning": "clear",
        })
        mock_client.send_vision_request.return_value = mock_response

        extractor = MultiRecordExtractor(client=mock_client)
        extractor._resolve_disagreement(
            page_data_uri="data:image/png;base64,fake",
            boundary=_make_boundary(),
            field_name="patient_id",
            value_1="MRN12345",
            value_2="MRN12346",
            record=_make_record(),
        )

        call_args = mock_client.send_vision_request.call_args
        request = call_args[0][0]
        assert "MRN12345" in request.prompt
        assert "MRN12346" in request.prompt
        assert "patient_id" in request.prompt


# ──────────────────────────────────────────────────────────────
# Pipeline Integration Tests
# ──────────────────────────────────────────────────────────────

class TestPhase3PipelineIntegration:
    """Test full pipeline with Phase 3 enabled."""

    def test_pipeline_skips_consensus_when_disabled(self):
        """With consensus disabled, no consensus VLM calls are made."""
        mock_client = MagicMock()
        mock_client.send_vision_request.side_effect = _base_pipeline_responses()

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=False,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 base VLM calls only
        assert mock_client.send_vision_request.call_count == 4
        assert result.total_records == 1

    def test_pipeline_runs_consensus_when_enabled(self):
        """With consensus enabled, extra VLM calls are made for consensus."""
        mock_client = MagicMock()

        responses = _base_pipeline_responses()
        # Consensus pass 1 (for patient_id and total_charge - both critical)
        responses.append(MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.95},
                "total_charge": {"value": "$150.00", "confidence": 0.93},
            }
        }))
        # Consensus pass 2
        responses.append(MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.94},
                "total_charge": {"value": "$150.00", "confidence": 0.92},
            }
        }))

        mock_client.send_vision_request.side_effect = responses

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=True,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 base + 2 consensus passes = 6
        assert mock_client.send_vision_request.call_count == 6
        assert result.total_records == 1

    def test_pipeline_consensus_without_validation(self):
        """Consensus can run independently without Phase 2."""
        mock_client = MagicMock()

        responses = _base_pipeline_responses()
        # Consensus pass 1
        responses.append(MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.95},
                "total_charge": {"value": "$150.00", "confidence": 0.93},
            }
        }))
        # Consensus pass 2
        responses.append(MagicMock(has_json=True, parsed_json={
            "critical_fields": {
                "patient_id": {"value": "MRN12345", "confidence": 0.94},
                "total_charge": {"value": "$150.00", "confidence": 0.92},
            }
        }))

        mock_client.send_vision_request.side_effect = responses

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_validation=False,
            enable_self_correction=False,
            enable_consensus=True,
        )

        result = extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        # 4 base + 2 consensus = 6 (no validation or correction calls)
        assert mock_client.send_vision_request.call_count == 6
        assert result.total_records == 1


# ──────────────────────────────────────────────────────────────
# Feature Flag Gating Tests
# ──────────────────────────────────────────────────────────────

class TestPhase3FeatureFlagGating:
    """Test that Phase 3 methods only run when flags are set."""

    @patch.object(MultiRecordExtractor, "_consensus_extract_critical_fields")
    @patch.object(MultiRecordExtractor, "_identify_critical_fields")
    def test_no_consensus_when_flag_off(self, mock_identify, mock_consensus):
        """Neither identification nor consensus should be called when flag is off."""
        mock_client = MagicMock()
        mock_client.send_vision_request.side_effect = _base_pipeline_responses()

        extractor = MultiRecordExtractor(
            client=mock_client,
            enable_consensus=False,
        )

        extractor.extract_document(
            page_images=[{"data_uri": "data:image/png;base64,fake", "page_number": 1}],
            pdf_path="test.pdf",
        )

        mock_identify.assert_not_called()
        mock_consensus.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
