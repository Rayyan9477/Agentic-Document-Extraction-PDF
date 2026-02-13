"""
Unit tests for Phase 2A: Document Splitter Agent.

Tests SplitterAgent, DocumentSegment, boundary classification,
segment building, and state integration.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.splitter import (
    BOUNDARY_DETECTION_SYSTEM_PROMPT,
    CLASSIFICATION_BATCH_SIZE,
    DocumentSegment,
    SplitterAgent,
)
from src.pipeline.state import ExtractionState, create_initial_state, update_state


# ──────────────────────────────────────────────────────────────────
# Helper fixtures
# ──────────────────────────────────────────────────────────────────


def _make_page_image(page_number: int) -> dict[str, Any]:
    """Create a minimal page image dict for testing."""
    return {
        "page_number": page_number,
        "data_uri": f"data:image/png;base64,fakepage{page_number}",
        "base64_encoded": f"fakepage{page_number}",
        "width": 2550,
        "height": 3300,
    }


def _make_state_with_pages(n_pages: int) -> ExtractionState:
    """Create a state with N fake page images."""
    pages = [_make_page_image(i + 1) for i in range(n_pages)]
    state = create_initial_state(pdf_path="/tmp/test.pdf")
    return update_state(state, {"page_images": pages})


@pytest.fixture
def splitter() -> SplitterAgent:
    """Create a SplitterAgent with a mock client."""
    mock_client = MagicMock()
    return SplitterAgent(client=mock_client)


@pytest.fixture
def mock_vlm_single_doc():
    """Mock VLM that classifies all pages as single document."""
    def _responder(pages_data):
        return {
            "pages": [
                {
                    "page_number": i + 1,
                    "is_new_document": (i == 0),
                    "document_type": "CMS-1500",
                    "confidence": 0.95,
                    "reason": "First page" if i == 0 else "Continuation",
                }
                for i in range(len(pages_data))
            ]
        }
    return _responder


@pytest.fixture
def mock_vlm_multi_doc():
    """Mock VLM that classifies pages as two documents (split at page 3)."""
    def _responder(batch, offset):
        pages = []
        for i in range(len(batch)):
            page_num = offset + i + 1
            is_new = (page_num == 1) or (page_num == 4)
            doc_type = "CMS-1500" if page_num <= 3 else "EOB"
            pages.append({
                "page_number": page_num,
                "is_new_document": is_new,
                "document_type": doc_type,
                "confidence": 0.90,
                "reason": "New document boundary" if is_new else "Continuation",
            })
        return {"pages": pages}
    return _responder


# ──────────────────────────────────────────────────────────────────
# DocumentSegment Tests
# ──────────────────────────────────────────────────────────────────


class TestDocumentSegment:
    """Tests for DocumentSegment dataclass."""

    def test_basic_creation(self):
        seg = DocumentSegment(
            start_page=1, end_page=3, document_type="CMS-1500", confidence=0.95,
        )
        assert seg.start_page == 1
        assert seg.end_page == 3
        assert seg.document_type == "CMS-1500"
        assert seg.confidence == 0.95
        assert seg.page_count == 3

    def test_single_page_segment(self):
        seg = DocumentSegment(
            start_page=5, end_page=5, document_type="EOB", confidence=0.85,
        )
        assert seg.page_count == 1

    def test_page_count_auto_calculated(self):
        seg = DocumentSegment(
            start_page=10, end_page=20, document_type="unknown", confidence=0.5,
        )
        assert seg.page_count == 11

    def test_to_dict(self):
        seg = DocumentSegment(
            start_page=1, end_page=5, document_type="CMS-1500",
            confidence=0.92, title="Claim Form",
        )
        d = seg.to_dict()
        assert d["start_page"] == 1
        assert d["end_page"] == 5
        assert d["document_type"] == "CMS-1500"
        assert d["confidence"] == 0.92
        assert d["page_count"] == 5
        assert d["title"] == "Claim Form"

    def test_to_dict_default_title(self):
        seg = DocumentSegment(
            start_page=1, end_page=1, document_type="unknown", confidence=0.5,
        )
        assert seg.to_dict()["title"] == ""


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Empty / Single Page
# ──────────────────────────────────────────────────────────────────


class TestSplitterEmptyAndSinglePage:
    """Tests for empty and single-page documents."""

    def test_empty_pages_returns_empty_segments(self, splitter: SplitterAgent):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        state = update_state(state, {"page_images": []})

        result = splitter.process(state)

        assert result["document_segments"] == []
        assert result["is_multi_document"] is False
        assert result["active_segment_index"] == 0

    def test_single_page_never_multi_document(self, splitter: SplitterAgent):
        state = _make_state_with_pages(1)

        result = splitter.process(state)

        assert result["is_multi_document"] is False
        segments = result["document_segments"]
        assert len(segments) == 1
        assert segments[0]["start_page"] == 1
        assert segments[0]["end_page"] == 1
        assert segments[0]["confidence"] == 1.0
        assert result["active_segment_index"] == 0


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Classification Logic
# ──────────────────────────────────────────────────────────────────


class TestSplitterClassification:
    """Tests for page classification and boundary detection."""

    def test_classify_all_pages_processes_in_batches(self, splitter: SplitterAgent):
        """Verify pages are batched correctly."""
        pages = [_make_page_image(i + 1) for i in range(12)]

        # Track batch calls
        batch_calls = []
        original_classify_batch = splitter._classify_batch

        def tracking_classify_batch(batch, offset):
            batch_calls.append({"size": len(batch), "offset": offset})
            # Return default classifications
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset == 0 and i == 0),
                    "document_type": "unknown",
                    "confidence": 0.8,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = tracking_classify_batch

        result = splitter._classify_all_pages(pages)

        assert len(result) == 12
        # With batch size 5: batches at offsets 0, 5, 10
        assert len(batch_calls) == 3
        assert batch_calls[0]["offset"] == 0
        assert batch_calls[0]["size"] == 5
        assert batch_calls[1]["offset"] == 5
        assert batch_calls[1]["size"] == 5
        assert batch_calls[2]["offset"] == 10
        assert batch_calls[2]["size"] == 2

    def test_classify_batch_vlm_success(self, splitter: SplitterAgent):
        """VLM returns valid classifications."""
        batch = [_make_page_image(1), _make_page_image(2)]

        splitter.send_vision_request_with_json = MagicMock(return_value={
            "pages": [
                {
                    "page_number": 1,
                    "is_new_document": True,
                    "document_type": "CMS-1500",
                    "confidence": 0.95,
                    "reason": "CMS-1500 header detected",
                },
                {
                    "page_number": 2,
                    "is_new_document": False,
                    "document_type": "CMS-1500",
                    "confidence": 0.90,
                    "reason": "Continuation of claim form",
                },
            ]
        })

        result = splitter._classify_batch(batch, offset=0)

        assert len(result) == 2
        assert result[0]["is_new_document"] is True
        assert result[0]["document_type"] == "CMS-1500"
        assert result[1]["is_new_document"] is False

    def test_classify_batch_vlm_partial_response(self, splitter: SplitterAgent):
        """VLM returns fewer pages than sent — rest default to continuation."""
        batch = [_make_page_image(i + 1) for i in range(3)]

        splitter.send_vision_request_with_json = MagicMock(return_value={
            "pages": [
                {
                    "page_number": 1,
                    "is_new_document": True,
                    "document_type": "EOB",
                    "confidence": 0.88,
                },
            ]
        })

        result = splitter._classify_batch(batch, offset=0)

        assert len(result) == 3
        assert result[0]["is_new_document"] is True
        # Pages 2 and 3 default to continuation
        assert result[1]["is_new_document"] is False
        assert result[1]["confidence"] == 0.5
        assert result[2]["is_new_document"] is False

    def test_classify_batch_vlm_failure_fallback(self, splitter: SplitterAgent):
        """VLM failure triggers heuristic fallback."""
        batch = [_make_page_image(i + 1) for i in range(3)]

        splitter.send_vision_request_with_json = MagicMock(
            side_effect=Exception("VLM unavailable")
        )

        # First batch (offset=0): page 1 is new, rest continuation
        result = splitter._classify_batch(batch, offset=0)

        assert len(result) == 3
        assert result[0]["is_new_document"] is True
        assert result[0]["confidence"] == 0.3
        assert "fallback" in result[0]["reason"].lower()
        assert result[1]["is_new_document"] is False
        assert result[2]["is_new_document"] is False

    def test_classify_batch_vlm_failure_non_first_batch(self, splitter: SplitterAgent):
        """Non-first batch fallback: all pages are continuation."""
        batch = [_make_page_image(6), _make_page_image(7)]

        splitter.send_vision_request_with_json = MagicMock(
            side_effect=Exception("VLM timeout")
        )

        result = splitter._classify_batch(batch, offset=5)

        assert len(result) == 2
        # Neither is first page of entire doc, so both are continuation
        assert result[0]["is_new_document"] is False
        assert result[1]["is_new_document"] is False

    def test_classify_batch_empty_batch(self, splitter: SplitterAgent):
        result = splitter._classify_batch([], offset=0)
        assert result == []


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Segment Building
# ──────────────────────────────────────────────────────────────────


class TestSplitterBuildSegments:
    """Tests for _build_segments logic."""

    def test_single_document_one_segment(self, splitter: SplitterAgent):
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "CMS-1500", "confidence": 0.95},
            {"page_number": 2, "is_new_document": False, "document_type": "CMS-1500", "confidence": 0.90},
            {"page_number": 3, "is_new_document": False, "document_type": "CMS-1500", "confidence": 0.88},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 1
        assert segments[0].start_page == 1
        assert segments[0].end_page == 3
        assert segments[0].page_count == 3
        assert segments[0].document_type == "CMS-1500"
        # Average confidence: (0.95 + 0.90 + 0.88) / 3
        assert segments[0].confidence == pytest.approx(0.91, abs=0.01)

    def test_two_documents(self, splitter: SplitterAgent):
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "CMS-1500", "confidence": 0.95},
            {"page_number": 2, "is_new_document": False, "document_type": "CMS-1500", "confidence": 0.90},
            {"page_number": 3, "is_new_document": True, "document_type": "EOB", "confidence": 0.88},
            {"page_number": 4, "is_new_document": False, "document_type": "EOB", "confidence": 0.85},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 2
        assert segments[0].start_page == 1
        assert segments[0].end_page == 2
        assert segments[0].document_type == "CMS-1500"
        assert segments[1].start_page == 3
        assert segments[1].end_page == 4
        assert segments[1].document_type == "EOB"

    def test_three_documents(self, splitter: SplitterAgent):
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "A", "confidence": 0.9},
            {"page_number": 2, "is_new_document": True, "document_type": "B", "confidence": 0.8},
            {"page_number": 3, "is_new_document": True, "document_type": "C", "confidence": 0.7},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 3
        assert all(s.page_count == 1 for s in segments)
        assert [s.document_type for s in segments] == ["A", "B", "C"]

    def test_single_page_classification(self, splitter: SplitterAgent):
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "unknown", "confidence": 0.5},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 1
        assert segments[0].start_page == 1
        assert segments[0].end_page == 1

    def test_empty_classifications(self, splitter: SplitterAgent):
        segments = splitter._build_segments([])
        assert segments == []

    def test_confidence_averaging(self, splitter: SplitterAgent):
        """Confidence is averaged across pages within a segment."""
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "X", "confidence": 1.0},
            {"page_number": 2, "is_new_document": False, "document_type": "X", "confidence": 0.5},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 1
        assert segments[0].confidence == pytest.approx(0.75)

    def test_missing_confidence_defaults_to_half(self, splitter: SplitterAgent):
        classifications = [
            {"page_number": 1, "is_new_document": True, "document_type": "X"},
            {"page_number": 2, "is_new_document": False},
        ]

        segments = splitter._build_segments(classifications)

        assert len(segments) == 1
        assert segments[0].confidence == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Full Process Integration
# ──────────────────────────────────────────────────────────────────


class TestSplitterProcessIntegration:
    """Tests for the full process() method with mocked VLM."""

    def test_multi_page_single_document(self, splitter: SplitterAgent):
        """All pages classified as single document."""
        state = _make_state_with_pages(5)

        # Mock VLM to return single document classification
        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset == 0 and i == 0),
                    "document_type": "CMS-1500",
                    "confidence": 0.92,
                    "reason": "Continuation",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["is_multi_document"] is False
        segments = result["document_segments"]
        assert len(segments) == 1
        assert segments[0]["start_page"] == 1
        assert segments[0]["end_page"] == 5
        assert segments[0]["page_count"] == 5

    def test_multi_page_two_documents(self, splitter: SplitterAgent):
        """Pages classified as two documents (split at page 4)."""
        state = _make_state_with_pages(6)

        def mock_classify(batch, offset):
            results = []
            for i in range(len(batch)):
                page_num = offset + i + 1
                is_new = (page_num == 1) or (page_num == 4)
                results.append({
                    "page_number": page_num,
                    "is_new_document": is_new,
                    "document_type": "CMS-1500" if page_num <= 3 else "EOB",
                    "confidence": 0.90,
                    "reason": "",
                })
            return results

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["is_multi_document"] is True
        segments = result["document_segments"]
        assert len(segments) == 2
        assert segments[0]["start_page"] == 1
        assert segments[0]["end_page"] == 3
        assert segments[0]["document_type"] == "CMS-1500"
        assert segments[1]["start_page"] == 4
        assert segments[1]["end_page"] == 6
        assert segments[1]["document_type"] == "EOB"

    def test_state_preserves_existing_fields(self, splitter: SplitterAgent):
        """Splitter should not clobber existing state fields."""
        state = _make_state_with_pages(2)
        state = update_state(state, {"pdf_path": "/my/doc.pdf", "processing_id": "test123"})

        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset == 0 and i == 0),
                    "document_type": "unknown",
                    "confidence": 0.7,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["pdf_path"] == "/my/doc.pdf"
        assert result["processing_id"] == "test123"

    def test_active_segment_index_starts_at_zero(self, splitter: SplitterAgent):
        state = _make_state_with_pages(3)

        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": True,
                    "document_type": "unknown",
                    "confidence": 0.8,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["active_segment_index"] == 0


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — get_segment_pages
# ──────────────────────────────────────────────────────────────────


class TestGetSegmentPages:
    """Tests for the get_segment_pages utility."""

    def test_get_pages_with_segment_dict(self, splitter: SplitterAgent):
        pages = [_make_page_image(i + 1) for i in range(6)]
        segment = {"start_page": 2, "end_page": 4}

        result = splitter.get_segment_pages(pages, segment)

        assert len(result) == 3
        assert [p["page_number"] for p in result] == [2, 3, 4]

    def test_get_pages_with_segment_object(self, splitter: SplitterAgent):
        pages = [_make_page_image(i + 1) for i in range(6)]
        segment = DocumentSegment(
            start_page=1, end_page=3, document_type="CMS-1500", confidence=0.9,
        )

        result = splitter.get_segment_pages(pages, segment)

        assert len(result) == 3
        assert [p["page_number"] for p in result] == [1, 2, 3]

    def test_get_pages_single_page_segment(self, splitter: SplitterAgent):
        pages = [_make_page_image(i + 1) for i in range(5)]
        segment = {"start_page": 3, "end_page": 3}

        result = splitter.get_segment_pages(pages, segment)

        assert len(result) == 1
        assert result[0]["page_number"] == 3

    def test_get_pages_last_segment(self, splitter: SplitterAgent):
        pages = [_make_page_image(i + 1) for i in range(5)]
        segment = {"start_page": 4, "end_page": 5}

        result = splitter.get_segment_pages(pages, segment)

        assert len(result) == 2
        assert [p["page_number"] for p in result] == [4, 5]

    def test_get_pages_missing_page_numbers(self, splitter: SplitterAgent):
        """Pages without page_number attribute are skipped."""
        pages = [
            {"data_uri": "x"},  # No page_number
            _make_page_image(2),
            _make_page_image(3),
        ]
        segment = {"start_page": 1, "end_page": 3}

        result = splitter.get_segment_pages(pages, segment)

        # Only pages 2 and 3 have page_number in range
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Prompt Content
# ──────────────────────────────────────────────────────────────────


class TestSplitterPrompts:
    """Tests for prompt content and system prompt."""

    def test_system_prompt_mentions_json(self):
        assert "JSON" in BOUNDARY_DETECTION_SYSTEM_PROMPT

    def test_system_prompt_mentions_boundary_indicators(self):
        prompt = BOUNDARY_DETECTION_SYSTEM_PROMPT
        assert "Page 1 of N" in prompt or "resetting" in prompt
        assert "CMS-1500" in prompt or "form types" in prompt

    def test_batch_size_is_reasonable(self):
        assert 2 <= CLASSIFICATION_BATCH_SIZE <= 10


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — State Field Existence
# ──────────────────────────────────────────────────────────────────


class TestSplitterStateFields:
    """Verify document splitter fields exist in ExtractionState."""

    def test_initial_state_has_splitting_fields(self):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        assert state["document_segments"] == []
        assert state["is_multi_document"] is False
        assert state["active_segment_index"] == 0

    def test_update_state_preserves_splitting_fields(self):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        state = update_state(state, {
            "document_segments": [{"start_page": 1, "end_page": 5}],
            "is_multi_document": True,
            "active_segment_index": 1,
        })

        assert len(state["document_segments"]) == 1
        assert state["is_multi_document"] is True
        assert state["active_segment_index"] == 1


# ──────────────────────────────────────────────────────────────────
# SplitterAgent — Edge Cases
# ──────────────────────────────────────────────────────────────────


class TestSplitterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_document_batch_processing(self, splitter: SplitterAgent):
        """100-page document should process in batches without error."""
        state = _make_state_with_pages(100)

        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset + i) % 25 == 0,  # New doc every 25 pages
                    "document_type": f"type_{(offset + i) // 25}",
                    "confidence": 0.85,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["is_multi_document"] is True
        segments = result["document_segments"]
        assert len(segments) == 4  # Pages 1-25, 26-50, 51-75, 76-100
        assert segments[0]["page_count"] == 25
        assert segments[3]["end_page"] == 100

    def test_every_page_is_new_document(self, splitter: SplitterAgent):
        """Each page is a separate document."""
        state = _make_state_with_pages(5)

        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": True,
                    "document_type": f"doc_{offset + i + 1}",
                    "confidence": 0.80,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["is_multi_document"] is True
        segments = result["document_segments"]
        assert len(segments) == 5
        assert all(s["page_count"] == 1 for s in segments)

    def test_vlm_returns_empty_pages_list(self, splitter: SplitterAgent):
        """VLM returns empty pages list — defaults should handle it."""
        batch = [_make_page_image(1), _make_page_image(2)]

        splitter.send_vision_request_with_json = MagicMock(return_value={"pages": []})

        result = splitter._classify_batch(batch, offset=0)

        # Should produce default classifications
        assert len(result) == 2
        assert result[0]["confidence"] == 0.5
        assert result[1]["confidence"] == 0.5

    def test_two_pages_no_split(self, splitter: SplitterAgent):
        """Two-page document that is not split."""
        state = _make_state_with_pages(2)

        def mock_classify(batch, offset):
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset == 0 and i == 0),
                    "document_type": "unknown",
                    "confidence": 0.7,
                    "reason": "",
                }
                for i in range(len(batch))
            ]

        splitter._classify_batch = mock_classify

        result = splitter.process(state)

        assert result["is_multi_document"] is False
        assert len(result["document_segments"]) == 1
