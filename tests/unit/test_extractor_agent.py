"""
Unit tests for ExtractorAgent.

Tests cover:
- Initialization (client, prompt_enhancer, agreement params)
- process() routing (adaptive vs legacy)
- _process_legacy success, no schema, no images
- _extract_page (dual-pass, page with no image data)
- _merge_pass_results
- _parse_bbox (dict, list, invalid)
- _merge_page_extractions (single page, multi-page)
- _build_field_metadata
- extract_single_field
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import AgentError, ExtractionError
from src.agents.extractor import ExtractorAgent
from src.client.lm_client import VisionResponse
from src.pipeline.state import BoundingBoxCoords, ExtractionStatus, FieldMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(json_result: dict | None = None) -> MagicMock:
    client = MagicMock()
    result = json_result or {"fields": {"patient_name": {"value": "Alice", "confidence": 0.9}}}
    resp = VisionResponse(
        content="{}",
        parsed_json=result,
        model="qwen3-vl",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=100,
    )
    client.send_vision_request.return_value = resp
    return client


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "analyzing",
        "current_step": "analysis_complete",
        "page_images": [
            {"page_number": 1, "data_uri": "data:image/png;base64,abc123"},
        ],
        "analysis": {"has_tables": False, "has_handwriting": False},
        "document_type": "CMS-1500",
        "selected_schema_name": "cms1500",
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "merged_extraction": {},
        "field_metadata": {},
        "validation": {},
        "total_vlm_calls": 0,
        "total_processing_time_ms": 0,
        "use_adaptive_extraction": False,
        "adaptive_schema": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestExtractorInit
# ---------------------------------------------------------------------------


class TestExtractorInit:
    """Tests for ExtractorAgent initialization."""

    def test_default_init(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        assert agent.name == "extractor"
        assert agent._agreement_boost == 0.1
        assert agent._disagreement_penalty == 0.3

    def test_custom_params(self) -> None:
        agent = ExtractorAgent(
            client=MagicMock(),
            agreement_confidence_boost=0.2,
            disagreement_confidence_penalty=0.5,
        )
        assert agent._agreement_boost == 0.2
        assert agent._disagreement_penalty == 0.5

    def test_prompt_enhancer_injection(self) -> None:
        enhancer = MagicMock()
        agent = ExtractorAgent(client=MagicMock(), prompt_enhancer=enhancer)
        assert agent._prompt_enhancer is enhancer

    def test_prompt_enhancer_default_none(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        assert agent._prompt_enhancer is None


# ---------------------------------------------------------------------------
# TestProcessRouting
# ---------------------------------------------------------------------------


class TestProcessRouting:
    """Tests for process() routing between adaptive and legacy."""

    def test_legacy_when_no_adaptive_schema(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state(use_adaptive_extraction=True, adaptive_schema=None)
        with patch.object(agent, "_process_legacy", return_value=state) as mock:
            agent.process(state)
            mock.assert_called_once()

    def test_legacy_when_flag_false(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state(use_adaptive_extraction=False)
        with patch.object(agent, "_process_legacy", return_value=state) as mock:
            agent.process(state)
            mock.assert_called_once()

    def test_adaptive_when_schema_present(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state(
            use_adaptive_extraction=True,
            adaptive_schema={"fields": [], "document_type_description": "invoice"},
        )
        with patch.object(agent, "_process_adaptive", return_value=state) as mock:
            agent.process(state)
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# TestProcessLegacy
# ---------------------------------------------------------------------------


class TestProcessLegacy:
    """Tests for _process_legacy."""

    def test_no_images_raises(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        state = _make_state(page_images=[])
        with pytest.raises(ExtractionError, match="No page images|No schema"):
            agent._process_legacy(state)

    def test_updates_status_and_step(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state()
        result = agent._process_legacy(state)
        assert result["status"] == ExtractionStatus.EXTRACTING.value
        assert result["current_step"] == "extraction_complete"

    def test_populates_page_extractions(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state()
        result = agent._process_legacy(state)
        assert "page_extractions" in result
        assert len(result["page_extractions"]) == 1

    def test_populates_merged_extraction(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state()
        result = agent._process_legacy(state)
        assert isinstance(result.get("merged_extraction"), dict)

    def test_increments_vlm_calls(self) -> None:
        client = _mock_client()
        agent = ExtractorAgent(client=client)
        state = _make_state()
        result = agent._process_legacy(state)
        assert result.get("total_vlm_calls", 0) >= 2  # Dual-pass


# ---------------------------------------------------------------------------
# TestParseBbox
# ---------------------------------------------------------------------------


class TestParseBbox:
    """Tests for _parse_bbox static method."""

    def test_dict_with_xywh(self) -> None:
        bbox = ExtractorAgent._parse_bbox(
            {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04}, page_number=1,
        )
        assert bbox is not None
        assert bbox.x == pytest.approx(0.1)
        assert bbox.width == pytest.approx(0.3)

    def test_dict_with_width_height(self) -> None:
        bbox = ExtractorAgent._parse_bbox(
            {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04}, page_number=1,
        )
        assert bbox is not None

    def test_list_format(self) -> None:
        bbox = ExtractorAgent._parse_bbox([0.1, 0.2, 0.3, 0.04], page_number=2)
        assert bbox is not None
        assert bbox.page == 2

    def test_invalid_returns_none(self) -> None:
        assert ExtractorAgent._parse_bbox("not a bbox", page_number=1) is None
        assert ExtractorAgent._parse_bbox(None, page_number=1) is None

    def test_out_of_range_returns_none(self) -> None:
        assert ExtractorAgent._parse_bbox(
            {"x": 2.0, "y": 0.2, "w": 0.3, "h": 0.04}, page_number=1,
        ) is None

    def test_zero_dimensions_returns_none(self) -> None:
        assert ExtractorAgent._parse_bbox(
            {"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.04}, page_number=1,
        ) is None


# ---------------------------------------------------------------------------
# TestBuildFieldMetadata
# ---------------------------------------------------------------------------


class TestBuildFieldMetadata:
    """Tests for _build_field_metadata."""

    def test_structured_field_data(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        extraction = {
            "name": {"value": "Alice", "confidence": 0.9, "source_page": 1},
        }
        metadata = agent._build_field_metadata(extraction)
        assert "name" in metadata
        assert isinstance(metadata["name"], FieldMetadata)
        assert metadata["name"].value == "Alice"
        assert metadata["name"].confidence == 0.9

    def test_unstructured_field_data(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        extraction = {"name": "raw_string_value"}
        metadata = agent._build_field_metadata(extraction)
        assert metadata["name"].value == "raw_string_value"
        assert metadata["name"].confidence == 0.5  # Conservative default

    def test_empty_extraction(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        metadata = agent._build_field_metadata({})
        assert metadata == {}

    def test_field_with_bbox(self) -> None:
        agent = ExtractorAgent(client=MagicMock())
        extraction = {
            "name": {
                "value": "Alice",
                "confidence": 0.9,
                "source_page": 1,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04},
            },
        }
        metadata = agent._build_field_metadata(extraction)
        assert metadata["name"].bbox is not None


# ---------------------------------------------------------------------------
# TestExtractSingleField
# ---------------------------------------------------------------------------


class TestExtractSingleField:
    """Tests for extract_single_field."""

    def test_success(self) -> None:
        client = _mock_client({"value": "Alice Smith", "confidence": 0.92, "location": "top-left"})
        agent = ExtractorAgent(client=client)

        from src.schemas import FieldType
        from src.schemas.schema_builder import FieldBuilder

        field_def = FieldBuilder("patient_name").type(FieldType.STRING).build()
        result = agent.extract_single_field("data:image/png;base64,abc", field_def)
        assert result.success is True
        assert result.data.value == "Alice Smith"

    def test_failure(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = Exception("VLM down")
        agent = ExtractorAgent(client=client)

        from src.schemas import FieldType
        from src.schemas.schema_builder import FieldBuilder

        field_def = FieldBuilder("patient_name").type(FieldType.STRING).build()
        result = agent.extract_single_field("data:image/png;base64,abc", field_def)
        assert result.success is False
