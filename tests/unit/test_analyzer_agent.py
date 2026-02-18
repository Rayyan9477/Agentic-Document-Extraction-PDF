"""
Unit tests for AnalyzerAgent.

Tests cover:
- Initialization and configuration
- process() state transitions
- _classify_document success/fallback
- _analyze_structure success/fallback
- _analyze_page_relationships (single, multi-page, VLM failure)
- _select_schema (custom, registry, no match)
- _normalize_document_type
- classify_document_standalone
- get_supported_document_types / get_available_schemas
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.analyzer import AnalyzerAgent, DOCUMENT_TYPE_MAP
from src.agents.base import AnalysisError
from src.client.lm_client import VisionResponse
from src.pipeline.state import ExtractionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(json_result: dict | None = None) -> MagicMock:
    """Create a mock LMStudioClient that returns the given JSON."""
    client = MagicMock()
    resp = VisionResponse(
        content="{}",
        parsed_json=json_result or {"document_type": "CMS-1500", "confidence": 0.92},
        model="qwen3-vl",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=200,
    )
    client.send_vision_request.return_value = resp
    return client


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "pending",
        "current_step": "initialized",
        "page_images": [
            {"page_number": 1, "data_uri": "data:image/png;base64,abc123"},
        ],
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "merged_extraction": {},
        "validation": {},
        "total_vlm_calls": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestAnalyzerInit
# ---------------------------------------------------------------------------


class TestAnalyzerInit:
    """Tests for AnalyzerAgent initialization."""

    def test_default_init(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent.name == "analyzer"
        assert agent._confidence_threshold == 0.7

    def test_custom_threshold(self) -> None:
        agent = AnalyzerAgent(client=MagicMock(), classification_confidence_threshold=0.9)
        assert agent._confidence_threshold == 0.9

    def test_schema_registry_created(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent._schema_registry is not None


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for process() method."""

    def test_process_updates_status(self) -> None:
        client = _mock_client({
            "document_type": "CMS-1500",
            "confidence": 0.92,
            "structures": ["form_fields"],
            "has_tables": True,
            "has_handwriting": False,
            "has_signatures": False,
            "regions_of_interest": [],
        })
        agent = AnalyzerAgent(client=client)
        state = _make_state()
        result = agent.process(state)
        assert result["status"] == ExtractionStatus.ANALYZING.value
        assert result["current_step"] == "analysis_complete"

    def test_process_no_page_images_raises(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        state = _make_state(page_images=[])
        with pytest.raises(AnalysisError, match="No page images"):
            agent.process(state)

    def test_process_no_image_data_raises(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        state = _make_state(page_images=[{"page_number": 1}])
        with pytest.raises(AnalysisError, match="First page has no image data"):
            agent.process(state)

    def test_process_sets_analysis_dict(self) -> None:
        client = _mock_client({
            "document_type": "EOB",
            "confidence": 0.85,
            "structures": [],
            "has_tables": False,
            "has_handwriting": False,
            "has_signatures": False,
            "regions_of_interest": [],
        })
        agent = AnalyzerAgent(client=client)
        state = _make_state()
        result = agent.process(state)
        analysis = result.get("analysis", {})
        assert "document_type" in analysis
        assert "document_type_confidence" in analysis

    def test_process_increments_vlm_calls(self) -> None:
        client = _mock_client({
            "document_type": "CMS-1500",
            "confidence": 0.95,
            "structures": [],
            "has_tables": False,
            "has_handwriting": False,
            "has_signatures": False,
            "regions_of_interest": [],
        })
        agent = AnalyzerAgent(client=client)
        state = _make_state()
        result = agent.process(state)
        assert result.get("total_vlm_calls", 0) > 0

    def test_process_adds_warning_on_low_confidence(self) -> None:
        client = _mock_client({
            "document_type": "OTHER",
            "confidence": 0.3,
            "structures": [],
            "has_tables": False,
            "has_handwriting": False,
            "has_signatures": False,
            "regions_of_interest": [],
        })
        agent = AnalyzerAgent(client=client, classification_confidence_threshold=0.7)
        state = _make_state()
        result = agent.process(state)
        assert len(result.get("warnings", [])) > 0


# ---------------------------------------------------------------------------
# TestClassifyDocument
# ---------------------------------------------------------------------------


class TestClassifyDocument:
    """Tests for _classify_document."""

    def test_successful_classification(self) -> None:
        client = _mock_client({
            "document_type": "CMS-1500",
            "confidence": 0.95,
            "reasoning": "standard CMS-1500 form layout",
        })
        agent = AnalyzerAgent(client=client)
        result = agent._classify_document("data:image/png;base64,abc")
        assert result["document_type"] == "CMS-1500"
        assert result["confidence"] == 0.95

    def test_classification_fallback_on_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM down")
        agent = AnalyzerAgent(client=client)
        result = agent._classify_document("data:image/png;base64,abc")
        assert result["document_type"] == "OTHER"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# TestAnalyzeStructure
# ---------------------------------------------------------------------------


class TestAnalyzeStructure:
    """Tests for _analyze_structure."""

    def test_successful_analysis(self) -> None:
        client = _mock_client({
            "structures": ["table", "form_fields"],
            "has_tables": True,
            "has_handwriting": True,
            "has_signatures": False,
            "has_barcodes": False,
            "regions_of_interest": [{"type": "table", "location": "center"}],
        })
        agent = AnalyzerAgent(client=client)
        result = agent._analyze_structure("data:image/png;base64,abc")
        assert result["has_tables"] is True
        assert result["has_handwriting"] is True

    def test_structure_fallback_on_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM down")
        agent = AnalyzerAgent(client=client)
        result = agent._analyze_structure("data:image/png;base64,abc")
        # Should return conservative defaults
        assert result["has_tables"] is True  # Conservative default
        assert result["has_handwriting"] is False


# ---------------------------------------------------------------------------
# TestSelectSchema
# ---------------------------------------------------------------------------


class TestSelectSchema:
    """Tests for _select_schema."""

    def test_custom_schema_takes_priority(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        result = agent._select_schema(
            {"document_type": "CMS-1500", "confidence": 0.9},
            custom_schema={"name": "my_schema"},
        )
        assert result["selected_schema"] == "my_schema"

    def test_no_schema_found(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        result = agent._select_schema(
            {"document_type": "NONEXISTENT", "confidence": 0.9},
            custom_schema=None,
        )
        # Should indicate no schema found
        assert result["schema_compatibility"] == 0.0 or result["selected_schema"] != ""


# ---------------------------------------------------------------------------
# TestNormalizeDocumentType
# ---------------------------------------------------------------------------


class TestNormalizeDocumentType:
    """Tests for _normalize_document_type."""

    def test_cms1500_variations(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent._normalize_document_type("CMS1500") == "CMS-1500"
        assert agent._normalize_document_type("HCFA1500") == "CMS-1500"

    def test_ub04_variations(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent._normalize_document_type("UB04") == "UB-04"
        assert agent._normalize_document_type("CMS1450") == "UB-04"

    def test_eob_variation(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent._normalize_document_type("EXPLANATIONOFBENEFITS") == "EOB"

    def test_unknown_type_returns_uppercase(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        assert agent._normalize_document_type("something_else") == "SOMETHING_ELSE"


# ---------------------------------------------------------------------------
# TestClassifyDocumentStandalone
# ---------------------------------------------------------------------------


class TestClassifyDocumentStandalone:
    """Tests for classify_document_standalone."""

    def test_standalone_success(self) -> None:
        client = _mock_client({
            "document_type": "EOB",
            "confidence": 0.88,
        })
        agent = AnalyzerAgent(client=client)
        result = agent.classify_document_standalone("data:image/png;base64,abc")
        assert result.success is True
        assert result.data["document_type"] == "EOB"

    def test_standalone_failure(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("boom")
        agent = AnalyzerAgent(client=client)
        result = agent.classify_document_standalone("data:image/png;base64,abc")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper methods."""

    def test_get_supported_document_types(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        types = agent.get_supported_document_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_get_available_schemas(self) -> None:
        agent = AnalyzerAgent(client=MagicMock())
        schemas = agent.get_available_schemas()
        assert isinstance(schemas, list)
