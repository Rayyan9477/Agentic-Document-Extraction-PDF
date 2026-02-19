"""Tests for SchemaGeneratorAgent — VLM-driven adaptive schema generation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.agents.base import AgentError
from src.agents.schema_generator import (
    SchemaGeneratorAgent,
    SchemaGenerationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(parsed_json: dict | None = None):
    """Create a MagicMock LMStudioClient for schema generation."""
    from src.client.lm_client import VisionResponse
    import json

    if parsed_json is None:
        parsed_json = {
            "schema_id": "adaptive_test123",
            "document_type_description": "Medical claim form",
            "field_groups": [
                {
                    "group_name": "patient_info",
                    "field_names": ["patient_name", "patient_dob"],
                    "group_type": "patient_demographics",
                    "extraction_strategy": "form_fields",
                }
            ],
            "fields": [
                {
                    "field_name": "patient_name",
                    "display_name": "Patient Name",
                    "field_type": "text",
                    "required": True,
                },
                {
                    "field_name": "patient_dob",
                    "display_name": "Date of Birth",
                    "field_type": "date",
                    "required": True,
                },
            ],
            "total_field_count": 2,
            "overall_strategy": "hybrid",
            "component_strategies": {"tables": "row_by_row"},
            "suggested_validations": {"patient_dob": ["Must be valid date"]},
            "cross_field_relationships": [],
            "high_confidence_fields": ["patient_name"],
            "optional_fields": [],
            "vlm_reasoning": "Detected form with patient info",
            "schema_confidence": 0.88,
        }

    content = json.dumps(parsed_json)
    client = MagicMock()
    client.send_vision_request.return_value = VisionResponse(
        content=content,
        parsed_json=parsed_json,
        model="qwen3-vl",
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        latency_ms=350,
    )
    return client


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "analyzing",
        "current_step": "schema_generation",
        "page_images": [
            {"page_number": 1, "data_uri": "data:image/png;base64,abc123"},
        ],
        "layout_analyses": [
            {
                "page_number": 1,
                "layout_type": "form",
                "layout_confidence": 0.9,
                "column_count": 2,
                "estimated_field_count": 30,
                "visual_marks": [],
                "density_estimate": "moderate",
                "vlm_observations": "Standard form",
                "extraction_difficulty": "moderate",
                "has_pre_printed_structure": True,
                "has_handwritten_content": False,
                "reading_order": "top-to-bottom",
            }
        ],
        "component_maps": [
            {
                "page_number": 1,
                "tables": [{"table_id": "t1", "row_count": 5, "column_count": 3}],
                "forms": [
                    {"field_id": "f1", "field_type": "text_filled", "label_text": "Name"},
                ],
                "key_value_pairs": [
                    {"pair_id": "kv1", "key_text": "DOB", "value_type_hint": "date"},
                ],
                "visual_marks": [],
                "has_tabular_data": True,
                "has_form_fields": True,
                "has_checkboxes": False,
                "has_signatures": False,
                "has_handwriting": False,
                "suggested_extraction_strategies": {"tables": "row_by_row"},
            }
        ],
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "total_vlm_calls": 2,
        "total_processing_time_ms": 400,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestSchemaGeneratorInit
# ---------------------------------------------------------------------------

class TestSchemaGeneratorInit:
    """Tests for SchemaGeneratorAgent initialization."""

    def test_default_name(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        assert agent.name == "schema_generator"

    def test_is_base_agent(self) -> None:
        from src.agents.base import BaseAgent
        agent = SchemaGeneratorAgent(client=MagicMock())
        assert isinstance(agent, BaseAgent)


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------

class TestProcess:
    """Tests for process() pipeline entry point."""

    def test_generates_adaptive_schema(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert "adaptive_schema" in result
        assert result["adaptive_schema"] is not None

    def test_schema_has_fields(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        schema = result["adaptive_schema"]
        assert "fields" in schema
        assert len(schema["fields"]) >= 1

    def test_updates_vlm_calls(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert result["total_vlm_calls"] >= 3  # 2 prior + 1 schema gen

    def test_no_layout_raises(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        state = _make_state(layout_analyses=[], component_maps=[])
        with pytest.raises(SchemaGenerationError, match="Missing layout"):
            agent.process(state)

    def test_no_page_images_raises(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        state = _make_state(page_images=[])
        with pytest.raises(SchemaGenerationError, match="No page images"):
            agent.process(state)


# ---------------------------------------------------------------------------
# TestGenerateSchema
# ---------------------------------------------------------------------------

class TestGenerateSchema:
    """Tests for _generate_schema internal method."""

    def test_returns_schema_dict(self) -> None:
        agent = SchemaGeneratorAgent(client=_mock_client())
        schema = agent._generate_schema(
            "data:image/png;base64,abc",
            {"layout_type": "form"},
            {"tables": [], "forms": []},
            1,
        )
        assert isinstance(schema, dict)
        assert "fields" in schema

    def test_fallback_on_vlm_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM down")
        agent = SchemaGeneratorAgent(client=client)
        schema = agent._generate_schema(
            "data:image/png;base64,abc",
            {"layout_type": "form"},
            {
                "tables": [],
                "forms": [{"label_text": "Name", "field_type": "text", "field_id": "f1"}],
                "key_value_pairs": [],
            },
            1,
        )
        assert schema["schema_confidence"] == 0.3
        assert "fallback" in schema["schema_id"]


# ---------------------------------------------------------------------------
# TestParseSchemaResponse
# ---------------------------------------------------------------------------

class TestParseSchemaResponse:
    """Tests for _parse_schema_response."""

    def test_adds_timing(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        result = agent._parse_schema_response({"fields": []}, 500)
        assert result["generation_time_ms"] == 500

    def test_preserves_schema_id(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        result = agent._parse_schema_response({"schema_id": "my_schema"}, 100)
        assert result["schema_id"] == "my_schema"

    def test_generates_schema_id_if_missing(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        result = agent._parse_schema_response({}, 100)
        assert result["schema_id"].startswith("adaptive_")

    def test_fills_defaults(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        result = agent._parse_schema_response({}, 0)
        assert result["overall_strategy"] == "adaptive"
        assert result["schema_confidence"] == 0.5
        assert result["fields"] == []


# ---------------------------------------------------------------------------
# TestCreateFallbackSchema
# ---------------------------------------------------------------------------

class TestCreateFallbackSchema:
    """Tests for _create_fallback_schema."""

    def test_creates_schema_from_components(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        components = {
            "forms": [
                {"label_text": "Patient Name", "field_type": "text", "field_id": "f1"},
                {"label_text": "Gender", "field_type": "checkbox", "field_id": "f2"},
            ],
            "key_value_pairs": [
                {"key_text": "Date of Birth", "value_type_hint": "date", "pair_id": "kv1"},
            ],
        }
        schema = agent._create_fallback_schema(None, components)
        assert len(schema["fields"]) == 3
        assert schema["schema_confidence"] == 0.3
        assert "fallback" in schema["schema_id"]

    def test_empty_components(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        schema = agent._create_fallback_schema(None, {"forms": [], "key_value_pairs": []})
        assert len(schema["fields"]) == 0

    def test_none_components(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        schema = agent._create_fallback_schema(None, None)
        assert len(schema["fields"]) == 0

    def test_checkbox_becomes_boolean(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        components = {
            "forms": [
                {"label_text": "Is Active", "field_type": "checkbox", "field_id": "f1"},
            ],
            "key_value_pairs": [],
        }
        schema = agent._create_fallback_schema(None, components)
        assert schema["fields"][0]["field_type"] == "boolean"


# ---------------------------------------------------------------------------
# TestBuildSchemaGenerationPrompt
# ---------------------------------------------------------------------------

class TestBuildSchemaGenerationPrompt:
    """Tests for prompt builder."""

    def test_without_context(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        prompt = agent._build_schema_generation_prompt(None, None, 1)
        assert "Adaptive Extraction Schema" in prompt

    def test_with_layout_context(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        layout = {
            "layout_type": "tabular_form",
            "column_count": 3,
            "density_estimate": "dense",
            "estimated_field_count": 50,
            "reading_order": "top-to-bottom",
            "has_pre_printed_structure": True,
            "has_handwritten_content": False,
            "visual_marks": [{"mark_type": "tick"}],
            "vlm_observations": "Complex medical form",
            "extraction_difficulty": "challenging",
        }
        prompt = agent._build_schema_generation_prompt(layout, None, 2)
        assert "tabular_form" in prompt
        assert "challenging" in prompt
        assert "Total Pages: 2" in prompt

    def test_with_component_context(self) -> None:
        agent = SchemaGeneratorAgent(client=MagicMock())
        components = {
            "tables": [
                {"row_count": 10, "column_count": 5, "description": "Line items",
                 "column_labels": ["A", "B", "C"]},
            ],
            "forms": [
                {"field_type": "text_filled", "label_text": "Name"},
                {"field_type": "checkbox", "label_text": "Active"},
            ],
            "key_value_pairs": [
                {"key_text": "DOB", "value_type_hint": "date"},
            ],
            "visual_marks": [],
            "has_tabular_data": True,
            "has_form_fields": True,
            "has_checkboxes": True,
            "has_signatures": False,
            "has_handwriting": False,
            "suggested_extraction_strategies": {"tables": "row_by_row"},
        }
        prompt = agent._build_schema_generation_prompt(None, components, 1)
        assert "Line items" in prompt
        assert "Name" in prompt


# ---------------------------------------------------------------------------
# TestSchemaGenerationError
# ---------------------------------------------------------------------------

class TestSchemaGenerationError:
    """Tests for SchemaGenerationError."""

    def test_is_agent_error(self) -> None:
        err = SchemaGenerationError("test", agent_name="sg")
        assert isinstance(err, AgentError)

    def test_message(self) -> None:
        err = SchemaGenerationError("Generation failed", agent_name="sg")
        assert "Generation failed" in str(err)
