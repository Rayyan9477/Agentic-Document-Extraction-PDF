"""Tests for LayoutAgent — VLM-native visual layout understanding."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import AgentError
from src.agents.layout_agent import LayoutAgent, LayoutAnalysisError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(parsed_json: dict | None = None):
    """Create a MagicMock LMStudioClient with configurable VLM response."""
    from src.client.lm_client import VisionResponse
    import json

    if parsed_json is None:
        parsed_json = {
            "layout_type": "form",
            "layout_confidence": 0.92,
            "regions": [{"region_id": "header", "region_type": "header"}],
            "column_count": 2,
            "reading_order": "top-to-bottom",
            "visual_separators": ["horizontal_lines"],
            "density_estimate": "moderate",
            "estimated_field_count": 30,
            "has_pre_printed_structure": True,
            "has_handwritten_content": False,
            "alignment_style": "grid-aligned",
            "spacing_quality": "normal",
            "visual_marks": [
                {"mark_type": "checkbox_checked", "confidence": 0.9},
            ],
            "vlm_observations": "Standard form layout",
            "extraction_difficulty": "moderate",
            "recommended_strategy": "form_extraction",
        }

    content = json.dumps(parsed_json)
    client = MagicMock()
    client.send_vision_request.return_value = VisionResponse(
        content=content,
        parsed_json=parsed_json,
        model="qwen3-vl",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=200,
    )
    return client


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "analyzing",
        "current_step": "layout_analysis",
        "page_images": [
            {"page_number": 1, "data_uri": "data:image/png;base64,abc123"},
        ],
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "total_vlm_calls": 0,
        "total_processing_time_ms": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestLayoutAgentInit
# ---------------------------------------------------------------------------

class TestLayoutAgentInit:
    """Tests for LayoutAgent initialization."""

    def test_default_name(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        assert agent.name == "layout"

    def test_custom_client(self) -> None:
        client = MagicMock()
        agent = LayoutAgent(client=client)
        assert agent._client is client

    def test_is_base_agent(self) -> None:
        from src.agents.base import BaseAgent
        agent = LayoutAgent(client=MagicMock())
        assert isinstance(agent, BaseAgent)


# ---------------------------------------------------------------------------
# TestLayoutAgentProcess
# ---------------------------------------------------------------------------

class TestLayoutAgentProcess:
    """Tests for process() pipeline entry point."""

    def test_single_page_analysis(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert "layout_analyses" in result
        assert len(result["layout_analyses"]) == 1

    def test_multi_page_analysis(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state(page_images=[
            {"page_number": 1, "data_uri": "data:image/png;base64,page1"},
            {"page_number": 2, "data_uri": "data:image/png;base64,page2"},
        ])
        result = agent.process(state)
        assert len(result["layout_analyses"]) == 2

    def test_updates_vlm_calls_counter(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert result["total_vlm_calls"] >= 1

    def test_updates_processing_time(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        assert result["total_processing_time_ms"] >= 0

    def test_no_images_raises(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        state = _make_state(page_images=[])
        with pytest.raises(LayoutAnalysisError, match="No page images"):
            agent.process(state)

    def test_skips_page_without_data_uri(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state(page_images=[
            {"page_number": 1, "data_uri": None},
            {"page_number": 2, "data_uri": "data:image/png;base64,ok"},
        ])
        result = agent.process(state)
        assert len(result["layout_analyses"]) == 1

    def test_layout_analysis_structure(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        state = _make_state()
        result = agent.process(state)
        layout = result["layout_analyses"][0]
        assert "layout_type" in layout
        assert "page_number" in layout
        assert "regions" in layout
        assert "visual_marks" in layout


# ---------------------------------------------------------------------------
# TestAnalyzePageLayout
# ---------------------------------------------------------------------------

class TestAnalyzePageLayout:
    """Tests for _analyze_page_layout internal method."""

    def test_returns_layout_dict(self) -> None:
        agent = LayoutAgent(client=_mock_client())
        layout = agent._analyze_page_layout("data:image/png;base64,abc", 1)
        assert isinstance(layout, dict)
        assert layout["page_number"] == 1

    def test_sets_defaults_for_missing_keys(self) -> None:
        """If VLM returns sparse JSON, defaults are filled in."""
        agent = LayoutAgent(client=_mock_client({"layout_type": "table"}))
        layout = agent._analyze_page_layout("data:image/png;base64,abc", 3)
        assert layout["layout_type"] == "table"
        assert layout["page_number"] == 3
        assert layout["column_count"] == 1  # default
        assert layout["visual_marks"] == []  # default

    def test_fallback_on_vlm_error(self) -> None:
        """If VLM fails, return minimal fallback layout."""
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM down")
        agent = LayoutAgent(client=client)
        layout = agent._analyze_page_layout("data:image/png;base64,abc", 1)
        assert layout["layout_type"] == "unknown"
        assert layout["layout_confidence"] == 0.0
        assert layout["recommended_strategy"] == "fallback"


# ---------------------------------------------------------------------------
# TestParseLayoutResponse
# ---------------------------------------------------------------------------

class TestParseLayoutResponse:
    """Tests for _parse_layout_response."""

    def test_adds_page_number(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        result = agent._parse_layout_response({"layout_type": "form"}, 5, 123)
        assert result["page_number"] == 5
        assert result["analysis_time_ms"] == 123

    def test_preserves_existing_fields(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        vlm = {"layout_type": "table", "column_count": 4, "regions": [{"id": "r1"}]}
        result = agent._parse_layout_response(vlm, 1, 50)
        assert result["layout_type"] == "table"
        assert result["column_count"] == 4
        assert len(result["regions"]) == 1

    def test_fills_defaults(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        result = agent._parse_layout_response({}, 1, 0)
        assert result["layout_type"] == "mixed"
        assert result["reading_order"] == "top-to-bottom"
        assert result["density_estimate"] == "moderate"


# ---------------------------------------------------------------------------
# TestBuildLayoutAnalysisPrompt
# ---------------------------------------------------------------------------

class TestBuildLayoutAnalysisPrompt:
    """Tests for prompt builder."""

    def test_returns_string(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        prompt = agent._build_layout_analysis_prompt()
        assert isinstance(prompt, str)

    def test_mentions_visual_marks(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        prompt = agent._build_layout_analysis_prompt()
        assert "visual" in prompt.lower() or "Visual" in prompt

    def test_mentions_layout_classification(self) -> None:
        agent = LayoutAgent(client=MagicMock())
        prompt = agent._build_layout_analysis_prompt()
        assert "layout" in prompt.lower()


# ---------------------------------------------------------------------------
# TestLayoutAnalysisError
# ---------------------------------------------------------------------------

class TestLayoutAnalysisError:
    """Tests for LayoutAnalysisError."""

    def test_is_agent_error(self) -> None:
        err = LayoutAnalysisError("test", agent_name="layout")
        assert isinstance(err, AgentError)

    def test_message(self) -> None:
        err = LayoutAnalysisError("Layout failed", agent_name="layout")
        assert "Layout failed" in str(err)
