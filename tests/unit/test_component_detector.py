"""Tests for ComponentDetectorAgent — VLM-native component detection."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.agents.base import AgentError
from src.agents.component_detector import (
    ComponentDetectorAgent,
    ComponentDetectionError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(parsed_json: dict | None = None):
    """Create a MagicMock LMStudioClient for component detection."""
    from src.client.lm_client import VisionResponse
    import json

    if parsed_json is None:
        parsed_json = {
            "tables": [
                {
                    "table_id": "table_1",
                    "row_count": 5,
                    "column_count": 3,
                    "has_header_row": True,
                    "column_labels": ["Date", "Service", "Amount"],
                    "description": "Service line items",
                }
            ],
            "forms": [
                {
                    "field_id": "field_patient_name",
                    "field_type": "text_filled",
                    "label_text": "Patient Name",
                    "is_filled": True,
                    "confidence": 0.90,
                },
                {
                    "field_id": "field_gender_m",
                    "field_type": "checkbox",
                    "label_text": "Male",
                    "is_filled": True,
                    "confidence": 0.85,
                },
            ],
            "key_value_pairs": [
                {
                    "pair_id": "kv_1",
                    "key_text": "Date of Birth",
                    "value_type_hint": "date",
                    "confidence": 0.88,
                }
            ],
            "visual_marks": [
                {"mark_type": "checkbox_checked", "confidence": 0.92}
            ],
            "special_elements": [],
            "has_tabular_data": True,
            "has_form_fields": True,
            "has_narrative_text": False,
            "has_checkboxes": True,
            "has_signatures": False,
            "has_handwriting": False,
            "component_count": 4,
            "extraction_order": ["patient_info", "services"],
            "challenging_regions": [],
            "suggested_extraction_strategies": {"tables": "row_by_row"},
            "vlm_notes": "Standard claim form",
        }

    content = json.dumps(parsed_json)
    client = MagicMock()
    client.send_vision_request.return_value = VisionResponse(
        content=content,
        parsed_json=parsed_json,
        model="qwen3-vl",
        usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
        latency_ms=250,
    )
    return client


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "analyzing",
        "current_step": "component_detection",
        "page_images": [
            {"page_number": 1, "data_uri": "data:image/png;base64,abc123"},
        ],
        "layout_analyses": [
            {
                "page_number": 1,
                "layout_type": "form",
                "estimated_field_count": 30,
                "visual_marks": [],
                "density_estimate": "moderate",
                "extraction_difficulty": "moderate",
                "vlm_observations": "Standard form",
                "reading_order": "top-to-bottom",
            }
        ],
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "total_vlm_calls": 1,
        "total_processing_time_ms": 200,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestComponentDetectorInit
# ---------------------------------------------------------------------------

class TestComponentDetectorInit:
    """Tests for ComponentDetectorAgent initialization."""

    def test_default_name(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        assert agent.name == "component_detector"

    def test_is_base_agent(self) -> None:
        from src.agents.base import BaseAgent
        agent = ComponentDetectorAgent(client=MagicMock())
        assert isinstance(agent, BaseAgent)


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------

class TestProcess:
    """Tests for process() pipeline entry point."""

    def test_single_page_detection(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert "component_maps" in result
        assert len(result["component_maps"]) == 1

    def test_multi_page_detection(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state(
            page_images=[
                {"page_number": 1, "data_uri": "data:image/png;base64,p1"},
                {"page_number": 2, "data_uri": "data:image/png;base64,p2"},
            ],
            layout_analyses=[
                {"page_number": 1, "layout_type": "form"},
                {"page_number": 2, "layout_type": "table"},
            ],
        )
        result = agent.process(state)
        assert len(result["component_maps"]) == 2

    def test_updates_vlm_calls(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert result["total_vlm_calls"] >= 2  # 1 from layout + 1 from component

    def test_no_images_raises(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        state = _make_state(page_images=[])
        with pytest.raises(ComponentDetectionError, match="No page images"):
            agent.process(state)

    def test_skips_page_without_data_uri(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state(page_images=[
            {"page_number": 1, "data_uri": None},
            {"page_number": 2, "data_uri": "data:image/png;base64,ok"},
        ])
        result = agent.process(state)
        assert len(result["component_maps"]) == 1

    def test_works_without_layout_analyses(self) -> None:
        """Should work even when no layout analysis is provided."""
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state(layout_analyses=[])
        result = agent.process(state)
        assert len(result["component_maps"]) == 1

    def test_component_map_structure(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        comp = result["component_maps"][0]
        assert "tables" in comp
        assert "forms" in comp
        assert "key_value_pairs" in comp
        assert "visual_marks" in comp
        assert "page_number" in comp


# ---------------------------------------------------------------------------
# TestDetectPageComponents
# ---------------------------------------------------------------------------

class TestDetectPageComponents:
    """Tests for _detect_page_components internal method."""

    def test_returns_dict(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        comp = agent._detect_page_components("data:image/png;base64,abc", 1, None)
        assert isinstance(comp, dict)
        assert comp["page_number"] == 1

    def test_with_layout_context(self) -> None:
        agent = ComponentDetectorAgent(client=_mock_client())
        layout = {"layout_type": "form", "estimated_field_count": 20}
        comp = agent._detect_page_components("data:image/png;base64,abc", 1, layout)
        assert comp["page_number"] == 1

    def test_fallback_on_vlm_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM error")
        agent = ComponentDetectorAgent(client=client)
        comp = agent._detect_page_components("data:image/png;base64,abc", 1, None)
        assert comp["component_count"] == 0
        assert comp["tables"] == []


# ---------------------------------------------------------------------------
# TestParseComponentResponse
# ---------------------------------------------------------------------------

class TestParseComponentResponse:
    """Tests for _parse_component_response."""

    def test_adds_page_number_and_timing(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        result = agent._parse_component_response(
            {"tables": [{"id": "t1"}], "forms": []}, 3, 456
        )
        assert result["page_number"] == 3
        assert result["detection_time_ms"] == 456

    def test_fills_defaults(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        result = agent._parse_component_response({}, 1, 0)
        assert result["tables"] == []
        assert result["forms"] == []
        assert result["has_tabular_data"] is False
        assert result["component_count"] == 0

    def test_calculates_component_count(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        result = agent._parse_component_response(
            {"tables": [{"id": "t1"}], "forms": [{"id": "f1"}, {"id": "f2"}],
             "key_value_pairs": [{"id": "kv1"}]},
            1, 0,
        )
        assert result["component_count"] == 4  # 1 table + 2 forms + 1 kv


# ---------------------------------------------------------------------------
# TestBuildComponentDetectionPrompt
# ---------------------------------------------------------------------------

class TestBuildComponentDetectionPrompt:
    """Tests for prompt builder."""

    def test_without_layout(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        prompt = agent._build_component_detection_prompt(None)
        assert "Component Detection" in prompt

    def test_with_layout_context(self) -> None:
        agent = ComponentDetectorAgent(client=MagicMock())
        layout = {
            "layout_type": "table",
            "reading_order": "left-to-right",
            "estimated_field_count": 45,
            "visual_marks": [{"mark_type": "tick"}],
            "density_estimate": "dense",
            "extraction_difficulty": "challenging",
            "vlm_observations": "Complex table layout",
        }
        prompt = agent._build_component_detection_prompt(layout)
        assert "table" in prompt.lower()
        assert "Layout Context" in prompt


# ---------------------------------------------------------------------------
# TestComponentDetectionError
# ---------------------------------------------------------------------------

class TestComponentDetectionError:
    """Tests for ComponentDetectionError."""

    def test_is_agent_error(self) -> None:
        err = ComponentDetectionError("test", agent_name="cd")
        assert isinstance(err, AgentError)

    def test_message(self) -> None:
        err = ComponentDetectionError("Detection failed", agent_name="cd")
        assert "Detection failed" in str(err)
