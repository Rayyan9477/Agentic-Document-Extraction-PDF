"""
Integration tests for API document processing and webhook flows.

Tests the FastAPI routes for document upload/processing and webhook
subscription/delivery with mocked pipeline components.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes.documents import _build_process_response, _map_confidence_level
from src.api.models import ConfidenceLevelEnum, TaskStatusEnum
from src.pipeline.state import ExtractionStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_extraction_state() -> dict[str, Any]:
    """A completed extraction state for building API responses."""
    return {
        "processing_id": "api-test-001",
        "pdf_path": "/data/test.pdf",
        "pdf_hash": "abc123",
        "document_type": "EOB",
        "selected_schema_name": "eob_v1",
        "status": ExtractionStatus.COMPLETED.value,
        "start_time": "2024-06-01T12:00:00Z",
        "end_time": "2024-06-01T12:01:00Z",
        "total_processing_time_ms": 60000,
        "total_vlm_calls": 6,
        "retry_count": 0,
        "overall_confidence": 0.91,
        "requires_human_review": False,
        "human_review_reason": "",
        "page_images": [b"p1"],
        "merged_extraction": {
            "claim_number": {"value": "CLM-123456", "confidence": 0.97},
            "patient_name": {"value": "Alice Brown", "confidence": 0.93},
            "amount_paid": {"value": "850.00", "confidence": 0.88},
        },
        "field_metadata": {
            "claim_number": {"confidence": 0.97, "passes_agree": True, "validation_passed": True},
            "patient_name": {"confidence": 0.93, "passes_agree": True, "validation_passed": True},
            "amount_paid": {"confidence": 0.88, "passes_agree": True, "validation_passed": True},
        },
        "validation": {
            "is_valid": True,
            "field_validations": {},
            "cross_field_validations": [],
            "hallucination_flags": [],
            "warnings": [],
            "errors": [],
        },
        "errors": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# _map_confidence_level helper
# ---------------------------------------------------------------------------


class TestMapConfidenceLevel:

    def test_high(self):
        assert _map_confidence_level(0.90) == ConfidenceLevelEnum.HIGH

    def test_medium(self):
        assert _map_confidence_level(0.60) == ConfidenceLevelEnum.MEDIUM

    def test_low(self):
        assert _map_confidence_level(0.30) == ConfidenceLevelEnum.LOW

    def test_boundary_high(self):
        assert _map_confidence_level(0.85) == ConfidenceLevelEnum.HIGH

    def test_boundary_medium(self):
        assert _map_confidence_level(0.50) == ConfidenceLevelEnum.MEDIUM


# ---------------------------------------------------------------------------
# _build_process_response
# ---------------------------------------------------------------------------


class TestBuildProcessResponse:

    def test_builds_response_from_completed_state(self, mock_extraction_state):
        resp = _build_process_response(mock_extraction_state, "/output/result.json")

        assert resp.processing_id == "api-test-001"
        assert resp.status == TaskStatusEnum.COMPLETED
        assert resp.overall_confidence == 0.91
        assert resp.output_path == "/output/result.json"
        assert "claim_number" in resp.data
        assert resp.data["claim_number"] == "CLM-123456"

    def test_field_metadata_populated(self, mock_extraction_state):
        resp = _build_process_response(mock_extraction_state)
        assert "claim_number" in resp.field_metadata
        assert resp.field_metadata["claim_number"].confidence == 0.97

    def test_validation_attached(self, mock_extraction_state):
        resp = _build_process_response(mock_extraction_state)
        assert resp.validation is not None
        assert resp.validation.is_valid is True

    def test_metadata_populated(self, mock_extraction_state):
        resp = _build_process_response(mock_extraction_state)
        assert resp.metadata.document_type == "EOB"
        assert resp.metadata.page_count == 1
        assert resp.metadata.total_vlm_calls == 6

    def test_handles_failed_state(self):
        state = {
            "processing_id": "fail-001",
            "status": ExtractionStatus.FAILED.value,
            "merged_extraction": {},
            "field_metadata": {},
            "overall_confidence": 0.0,
            "errors": ["Pipeline exploded"],
            "warnings": [],
            "page_images": [],
        }
        resp = _build_process_response(state)
        assert resp.status == TaskStatusEnum.FAILED
        assert len(resp.errors) == 1

    def test_handles_plain_field_values(self):
        """merged_extraction may contain plain values (not dicts)."""
        state = {
            "processing_id": "plain-001",
            "status": "completed",
            "merged_extraction": {"name": "Alice"},
            "field_metadata": {},
            "overall_confidence": 0.8,
            "errors": [],
            "warnings": [],
            "page_images": [],
        }
        resp = _build_process_response(state)
        assert resp.data["name"] == "Alice"
