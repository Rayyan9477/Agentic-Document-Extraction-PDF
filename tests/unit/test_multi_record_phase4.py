"""
Unit tests for Phase 4 intelligent extraction improvements.

Tests cover:
- C3: Schema prompt separation (_build_extraction_system_prompt)
- C2: Synthetic few-shot examples (synthetic_examples.py + integration)
- C1: Schema decomposition (_split_schema_for_extraction, chunked extraction)
- Config plumbing: new flags in extraction_config
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.multi_record import (
    ExtractedRecord,
    MultiRecordExtractor,
    RecordBoundary,
)
from src.prompts.synthetic_examples import (
    SYNTHETIC_EXAMPLES,
    _DEFAULT_EXAMPLE,
    get_synthetic_example,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_schema(field_count: int, prefix: str = "field") -> dict:
    """Build a schema with N fields for testing decomposition."""
    fields = []
    for i in range(field_count):
        name = f"{prefix}_{i}"
        fields.append({
            "field_name": name,
            "field_type": "text",
            "description": f"Description for {name}",
            "required": i < 3,
        })
    return {"schema_id": "adaptive_test_doc", "entity_type": "item", "fields": fields}


def _make_schema_with_critical(
    critical_names: list[str],
    other_count: int,
) -> dict:
    """Build a schema with explicit critical fields + N non-critical fields."""
    fields = []
    for name in critical_names:
        fields.append({
            "field_name": name,
            "field_type": "text",
            "description": f"Critical field: {name}",
            "required": True,
        })
    for i in range(other_count):
        fields.append({
            "field_name": f"other_{i}",
            "field_type": "text",
            "description": f"Other field {i}",
            "required": False,
        })
    return {"schema_id": "adaptive_test_doc", "entity_type": "item", "fields": fields}


def _make_boundary(record_id: int = 1) -> RecordBoundary:
    return RecordBoundary(
        record_id=record_id,
        primary_identifier="TEST-001",
        bounding_box={"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 1.0},
        visual_separator="horizontal_line",
        entity_type="item",
    )


# ── C3: Schema Prompt Separation ────────────────────────────────

class TestExtractionSystemPrompt:
    """Test _build_extraction_system_prompt (C3: prompt separation)."""

    def test_includes_grounding_rules(self):
        """System prompt should include base grounding rules."""
        extractor = MultiRecordExtractor()
        schema = _make_schema(3)
        prompt = extractor._build_extraction_system_prompt(schema)

        # Grounding rules from shared module
        assert "null" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_includes_output_schema(self):
        """System prompt should include OUTPUT SCHEMA section."""
        extractor = MultiRecordExtractor()
        schema = _make_schema(3)
        prompt = extractor._build_extraction_system_prompt(schema)

        assert "OUTPUT SCHEMA" in prompt
        assert "record_id" in prompt
        assert "primary_identifier" in prompt
        assert "confidence" in prompt

    def test_includes_field_types(self):
        """System prompt should declare field names and types."""
        extractor = MultiRecordExtractor()
        schema = {
            "fields": [
                {"field_name": "patient_name", "field_type": "text", "required": True},
                {"field_name": "total_charge", "field_type": "number", "required": False},
                {"field_name": "service_date", "field_type": "date", "required": True},
            ]
        }
        prompt = extractor._build_extraction_system_prompt(schema)

        assert '"patient_name": text (required)' in prompt
        assert '"total_charge": number (optional)' in prompt
        assert '"service_date": date (required)' in prompt

    def test_includes_structural_rules(self):
        """System prompt should include structural formatting rules."""
        extractor = MultiRecordExtractor()
        schema = _make_schema(2)
        prompt = extractor._build_extraction_system_prompt(schema)

        assert "STRUCTURAL RULES" in prompt
        assert "null" in prompt
        assert "ISO format" in prompt or "YYYY-MM-DD" in prompt


# ── C2: Synthetic Few-Shot Examples ─────────────────────────────

class TestSyntheticExamples:
    """Test synthetic_examples.py module (C2)."""

    def test_medical_superbill_patient_exists(self):
        """Known doc/entity type should return a specific example."""
        example = get_synthetic_example("medical_superbill", "patient")
        assert "EXAMPLE OUTPUT" in example
        assert "record_id" in example
        assert "DOE, JANE" in example

    def test_invoice_line_item_exists(self):
        example = get_synthetic_example("invoice", "line_item")
        assert "Widget Assembly" in example

    def test_invoice_invoice_exists(self):
        example = get_synthetic_example("invoice", "invoice")
        assert "Acme Corp" in example

    def test_employee_roster_exists(self):
        example = get_synthetic_example("employee_roster", "employee")
        assert "EMP-1001" in example

    def test_insurance_claim_exists(self):
        example = get_synthetic_example("insurance_claim", "claim")
        assert "CLM-2024-5678" in example

    def test_unknown_doc_type_returns_default(self):
        """Unknown document type should return generic default example."""
        example = get_synthetic_example("unknown_type", "unknown_entity")
        assert example == _DEFAULT_EXAMPLE
        assert "EXAMPLE OUTPUT" in example
        assert "identifier_value" in example

    def test_known_doc_unknown_entity_returns_default(self):
        """Known doc type but unknown entity should fall back to default."""
        example = get_synthetic_example("invoice", "nonexistent_entity")
        assert example == _DEFAULT_EXAMPLE

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        example = get_synthetic_example("MEDICAL_SUPERBILL", "PATIENT")
        assert "DOE, JANE" in example

    def test_default_example_has_null_guidance(self):
        """Default example should guide on null usage."""
        assert "null" in _DEFAULT_EXAMPLE
        assert "Never guess" in _DEFAULT_EXAMPLE


class TestSyntheticExamplesIntegration:
    """Test synthetic examples integration in extract_single_record."""

    def test_example_injected_when_enabled(self):
        """When enable_synthetic_examples=True, prompt should contain example."""
        extractor = MultiRecordExtractor(enable_synthetic_examples=True)
        schema = {
            "schema_id": "adaptive_medical_superbill",
            "entity_type": "patient",
            "fields": [
                {"field_name": "patient_name", "field_type": "text", "description": "Name"},
            ],
        }
        boundary = RecordBoundary(
            record_id=1,
            primary_identifier="DOE, JOHN",
            bounding_box={"top": 0.0, "bottom": 0.5},
            visual_separator="line",
            entity_type="patient",
        )

        # Mock the VLM call to capture the prompt
        with patch.object(extractor, "_send_vision_json") as mock_vlm:
            mock_vlm.return_value = {
                "record_id": 1,
                "primary_identifier": "DOE, JOHN",
                "fields": {"patient_name": "DOE, JOHN"},
                "confidence": 0.9,
            }
            extractor.extract_single_record("data:image/png;base64,abc", boundary, schema, 1)

            prompt = mock_vlm.call_args[1].get("prompt") or mock_vlm.call_args[0][1]
            assert "EXAMPLE OUTPUT" in prompt

    def test_example_not_injected_when_disabled(self):
        """When enable_synthetic_examples=False, prompt should NOT contain example."""
        extractor = MultiRecordExtractor(enable_synthetic_examples=False)
        schema = {
            "schema_id": "adaptive_medical_superbill",
            "entity_type": "patient",
            "fields": [
                {"field_name": "patient_name", "field_type": "text", "description": "Name"},
            ],
        }
        boundary = RecordBoundary(
            record_id=1,
            primary_identifier="DOE, JOHN",
            bounding_box={"top": 0.0, "bottom": 0.5},
            visual_separator="line",
            entity_type="patient",
        )

        with patch.object(extractor, "_send_vision_json") as mock_vlm:
            mock_vlm.return_value = {
                "record_id": 1,
                "primary_identifier": "DOE, JOHN",
                "fields": {"patient_name": "DOE, JOHN"},
                "confidence": 0.9,
            }
            extractor.extract_single_record("data:image/png;base64,abc", boundary, schema, 1)

            prompt = mock_vlm.call_args[1].get("prompt") or mock_vlm.call_args[0][1]
            assert "EXAMPLE OUTPUT" not in prompt


# ── C1: Schema Decomposition ────────────────────────────────────

class TestSplitSchemaForExtraction:
    """Test _split_schema_for_extraction (C1: task decomposition)."""

    def test_small_schema_no_split(self):
        """Schema with <=max_fields should return single chunk."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(8)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 1
        assert len(chunks[0]["fields"]) == 8

    def test_exact_limit_no_split(self):
        """Schema with exactly max_fields should not split."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(10)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 1
        assert len(chunks[0]["fields"]) == 10

    def test_large_schema_splits(self):
        """Schema with >max_fields should split into multiple chunks."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(15)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 2
        total_fields = sum(len(c["fields"]) for c in chunks)
        assert total_fields == 15

    def test_critical_fields_in_first_chunk(self):
        """Critical fields (id, date, amount) should be in the first chunk."""
        extractor = MultiRecordExtractor(max_fields_per_call=5)
        schema = _make_schema_with_critical(
            critical_names=["patient_id", "service_date", "total_amount"],
            other_count=10,
        )
        chunks = extractor._split_schema_for_extraction(schema)

        first_chunk_names = {f["field_name"] for f in chunks[0]["fields"]}
        assert "patient_id" in first_chunk_names
        assert "service_date" in first_chunk_names
        assert "total_amount" in first_chunk_names

    def test_balanced_chunk_sizes(self):
        """Chunks should be reasonably balanced (not 10+1)."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(11)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 2
        sizes = [len(c["fields"]) for c in chunks]
        # First chunk gets up to max, second gets remainder
        assert sizes[0] == 10
        assert sizes[1] == 1

    def test_25_fields_splits_into_3_chunks(self):
        """25 fields with max=10 should produce 3 chunks."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(25)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 3
        total = sum(len(c["fields"]) for c in chunks)
        assert total == 25

    def test_disabled_returns_single_chunk(self):
        """With enable_schema_decomposition=False, always returns single chunk."""
        extractor = MultiRecordExtractor(
            max_fields_per_call=5,
            enable_schema_decomposition=False,
        )
        schema = _make_schema(20)
        chunks = extractor._split_schema_for_extraction(schema)

        assert len(chunks) == 1
        assert len(chunks[0]["fields"]) == 20

    def test_preserves_schema_metadata(self):
        """Each chunk should preserve schema metadata (schema_id, entity_type)."""
        extractor = MultiRecordExtractor(max_fields_per_call=5)
        schema = _make_schema(12)
        schema["schema_id"] = "adaptive_invoice"
        schema["entity_type"] = "line_item"

        chunks = extractor._split_schema_for_extraction(schema)
        for chunk in chunks:
            assert chunk["schema_id"] == "adaptive_invoice"
            assert chunk["entity_type"] == "line_item"


class TestChunkedExtraction:
    """Test chunked extraction in extract_single_record (C1)."""

    def test_single_chunk_single_vlm_call(self):
        """Small schema should make exactly 1 VLM call."""
        extractor = MultiRecordExtractor(max_fields_per_call=10)
        schema = _make_schema(5)
        boundary = _make_boundary()

        with patch.object(extractor, "_send_vision_json") as mock_vlm:
            mock_vlm.return_value = {
                "record_id": 1,
                "primary_identifier": "TEST-001",
                "fields": {f"field_{i}": f"val_{i}" for i in range(5)},
                "confidence": 0.92,
            }
            record = extractor.extract_single_record(
                "data:image/png;base64,abc", boundary, schema, 1,
            )

            assert mock_vlm.call_count == 1
            assert len(record.fields) == 5
            assert record.confidence == 0.92

    def test_multi_chunk_multiple_vlm_calls(self):
        """Large schema should make multiple VLM calls and merge fields."""
        extractor = MultiRecordExtractor(max_fields_per_call=5)
        schema = _make_schema(12)
        boundary = _make_boundary()

        call_count = [0]

        def mock_vlm_side_effect(*args, **kwargs):
            call_count[0] += 1
            prompt = kwargs.get("prompt") or args[1]
            # Return fields based on which chunk is being extracted
            fields = {}
            for i in range(12):
                name = f"field_{i}"
                if name in prompt:
                    fields[name] = f"value_{i}"
            return {
                "record_id": 1,
                "primary_identifier": "TEST-001",
                "fields": fields,
                "confidence": 0.90 if call_count[0] == 1 else 0.85,
            }

        with patch.object(extractor, "_send_vision_json", side_effect=mock_vlm_side_effect):
            record = extractor.extract_single_record(
                "data:image/png;base64,abc", boundary, schema, 1,
            )

            # Should have made multiple VLM calls (3 chunks: 5+5+2)
            assert call_count[0] == 3
            # Confidence should be min across chunks
            assert record.confidence == 0.85

    def test_multi_chunk_min_confidence(self):
        """Multi-chunk extraction should use minimum confidence across chunks."""
        extractor = MultiRecordExtractor(max_fields_per_call=5)
        schema = _make_schema(11)
        boundary = _make_boundary()

        confidences = [0.95, 0.80, 0.90]
        call_idx = [0]

        def mock_side_effect(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return {
                "record_id": 1,
                "primary_identifier": "TEST-001",
                "fields": {f"chunk{idx}_f": "v"},
                "confidence": confidences[idx] if idx < len(confidences) else 0.9,
            }

        with patch.object(extractor, "_send_vision_json", side_effect=mock_side_effect):
            record = extractor.extract_single_record(
                "data:image/png;base64,abc", boundary, schema, 1,
            )

            assert record.confidence == 0.80

    def test_multi_chunk_merges_all_fields(self):
        """All fields from all chunks should be merged into the final record."""
        extractor = MultiRecordExtractor(max_fields_per_call=3)
        schema = _make_schema(7)
        boundary = _make_boundary()

        call_idx = [0]

        def mock_side_effect(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            # Each chunk returns its specific fields
            if idx == 0:
                fields = {"field_0": "a", "field_1": "b", "field_2": "c"}
            elif idx == 1:
                fields = {"field_3": "d", "field_4": "e", "field_5": "f"}
            else:
                fields = {"field_6": "g"}
            return {
                "record_id": 1,
                "primary_identifier": "TEST-001",
                "fields": fields,
                "confidence": 0.9,
            }

        with patch.object(extractor, "_send_vision_json", side_effect=mock_side_effect):
            record = extractor.extract_single_record(
                "data:image/png;base64,abc", boundary, schema, 1,
            )

            assert len(record.fields) == 7
            for i in range(7):
                assert f"field_{i}" in record.fields

    def test_batch_note_in_multi_chunk_prompts(self):
        """Multi-chunk extraction should include BATCH N of M in prompts."""
        extractor = MultiRecordExtractor(max_fields_per_call=5)
        schema = _make_schema(11)
        boundary = _make_boundary()

        prompts_captured = []

        def mock_side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt") or args[1]
            prompts_captured.append(prompt)
            return {
                "record_id": 1,
                "primary_identifier": "TEST-001",
                "fields": {"f": "v"},
                "confidence": 0.9,
            }

        with patch.object(extractor, "_send_vision_json", side_effect=mock_side_effect):
            extractor.extract_single_record(
                "data:image/png;base64,abc", boundary, schema, 1,
            )

            # 3 chunks (5+5+1), each prompt should have BATCH note
            assert len(prompts_captured) == 3
            assert "BATCH 1 of 3" in prompts_captured[0]
            assert "BATCH 2 of 3" in prompts_captured[1]
            assert "BATCH 3 of 3" in prompts_captured[2]


# ── Config Plumbing ─────────────────────────────────────────────

class TestConfigPlumbing:
    """Test that new Phase 4 config flags are loaded correctly."""

    def test_extraction_config_has_new_flags(self):
        """get_extraction_config should return the 3 new Phase 4 flags."""
        from src.config.extraction_config import reload_extraction_config

        cfg = reload_extraction_config()
        assert "max_fields_per_extraction_call" in cfg
        assert "enable_schema_decomposition" in cfg
        assert "enable_synthetic_few_shot_examples" in cfg

    def test_config_defaults(self):
        """New flags should have sensible defaults if missing from config.json."""
        from src.config.extraction_config import _load_raw, get_extraction_config

        with patch.object(
            __import__("src.config.extraction_config", fromlist=["_load_raw"]),
            "_load_raw",
            return_value={},
        ):
            # Clear cache and reload
            from src.config.extraction_config import _load_raw as real_load
            real_load.cache_clear()

            cfg = get_extraction_config()
            assert cfg["max_fields_per_extraction_call"] == 10
            assert cfg["enable_schema_decomposition"] is True
            assert cfg["enable_synthetic_few_shot_examples"] is False

            # Restore cache
            real_load.cache_clear()

    def test_constructor_accepts_new_params(self):
        """MultiRecordExtractor should accept all Phase 4 constructor params."""
        extractor = MultiRecordExtractor(
            max_fields_per_call=8,
            enable_schema_decomposition=False,
            enable_synthetic_examples=True,
        )
        assert extractor._max_fields_per_call == 8
        assert extractor._enable_schema_decomposition is False
        assert extractor._enable_synthetic_examples is True
