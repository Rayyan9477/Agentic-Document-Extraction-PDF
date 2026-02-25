"""
Integration tests for the PipelineRunner end-to-end flow.

Tests extraction through the PipelineRunner with mocked LM client,
verifying preprocessing, workflow execution, checkpointing, and
error recovery.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.runner import PipelineRunner, get_extraction_result
from src.pipeline.state import ExtractionStatus, create_initial_state, update_state


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_lm_client() -> MagicMock:
    """LMStudioClient mock that returns canned VLM responses."""
    client = MagicMock()
    client.analyze_image.return_value = MagicMock(
        content='{"document_type": "CMS-1500", "fields": {"patient_name": "Jane Smith"}}',
        parsed_json={"document_type": "CMS-1500", "fields": {"patient_name": "Jane Smith"}},
    )
    return client


# ---------------------------------------------------------------------------
# PipelineRunner init + helpers
# ---------------------------------------------------------------------------


class TestPipelineRunnerInit:

    def test_default_construction(self):
        with patch("src.pipeline.runner.LMStudioClient"):
            runner = PipelineRunner(enable_checkpointing=False)
            assert runner._max_retries == 2
            assert runner._dpi == 200

    def test_custom_params(self):
        client = MagicMock()
        runner = PipelineRunner(
            client=client,
            enable_checkpointing=False,
            max_retries=5,
            dpi=300,
            max_image_dimension=4096,
        )
        assert runner._max_retries == 5
        assert runner._dpi == 300
        assert runner._max_image_dimension == 4096

    def test_file_not_found_raises(self, mock_lm_client):
        runner = PipelineRunner(client=mock_lm_client, enable_checkpointing=False)
        with pytest.raises(FileNotFoundError):
            runner.extract_from_pdf("/nonexistent/path.pdf")


# ---------------------------------------------------------------------------
# get_extraction_result helper
# ---------------------------------------------------------------------------


class TestGetExtractionResult:

    def test_completed_state(self):
        state = {
            "status": ExtractionStatus.COMPLETED.value,
            "document_type": "CMS-1500",
            "selected_schema_name": "cms1500_v1",
            "merged_extraction": {
                "patient_name": {"value": "Jane Smith", "confidence": 0.95},
            },
            "overall_confidence": 0.92,
            "confidence_level": "high",
            "errors": [],
            "warnings": [],
            "processing_id": "test-001",
            "total_processing_ms": 1234,
        }
        result = get_extraction_result(state)
        assert result["success"] is True
        assert result["document_type"] == "CMS-1500"
        assert result["confidence"] == 0.92
        assert "patient_name" in result["fields"]

    def test_failed_state(self):
        state = {
            "status": ExtractionStatus.FAILED.value,
            "errors": ["Something went wrong"],
            "merged_extraction": {},
        }
        result = get_extraction_result(state)
        assert result["success"] is False
        assert len(result["errors"]) == 1

    def test_human_review_state(self):
        state = {
            "status": ExtractionStatus.HUMAN_REVIEW.value,
            "merged_extraction": {},
            "overall_confidence": 0.4,
        }
        result = get_extraction_result(state)
        assert result["requires_review"] is True
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Pipeline error recovery
# ---------------------------------------------------------------------------


class TestPipelineErrorRecovery:

    def test_apply_human_corrections(self, mock_lm_client):
        """Test that human corrections update merged_extraction."""
        runner = PipelineRunner(client=mock_lm_client, enable_checkpointing=False)

        state = create_initial_state("/fake.pdf")
        state = update_state(state, {
            "merged_extraction": {
                "patient_name": {"value": "Jan Smith", "confidence": 0.6},
            },
        })

        corrected = runner._apply_human_corrections(state, {
            "patient_name": "Jane Smith",
        })

        name_field = corrected["merged_extraction"]["patient_name"]
        assert name_field["value"] == "Jane Smith"
        assert name_field["confidence"] == 1.0
        assert name_field["human_corrected"] is True

    def test_apply_human_corrections_new_field(self, mock_lm_client):
        """Test adding a brand new field via corrections."""
        runner = PipelineRunner(client=mock_lm_client, enable_checkpointing=False)

        state = create_initial_state("/fake.pdf")
        state = update_state(state, {"merged_extraction": {}})

        corrected = runner._apply_human_corrections(state, {
            "new_field": "new_value",
        })

        assert corrected["merged_extraction"]["new_field"]["value"] == "new_value"
        assert corrected["merged_extraction"]["new_field"]["human_corrected"] is True


# ---------------------------------------------------------------------------
# Checkpoint status
# ---------------------------------------------------------------------------


class TestPipelineCheckpointing:

    def test_get_checkpoint_status_not_found(self, mock_lm_client):
        """Checkpoint status returns None for unknown thread."""
        runner = PipelineRunner(client=mock_lm_client, enable_checkpointing=False)

        with patch.object(runner, "_ensure_workflow_initialized"):
            runner._orchestrator = MagicMock()
            runner._orchestrator.get_checkpoint_state.return_value = None

            status = runner.get_checkpoint_status("unknown-thread")
            assert status is None

    def test_get_checkpoint_status_found(self, mock_lm_client):
        """Checkpoint status returns summary dict for known thread."""
        runner = PipelineRunner(client=mock_lm_client, enable_checkpointing=False)

        mock_state = {
            "processing_id": "p-123",
            "status": ExtractionStatus.EXTRACTING.value,
            "current_step": "extraction",
            "overall_confidence": 0.75,
            "retry_count": 1,
            "errors": [],
            "warnings": ["Low confidence"],
        }

        with patch.object(runner, "_ensure_workflow_initialized"):
            runner._orchestrator = MagicMock()
            runner._orchestrator.get_checkpoint_state.return_value = mock_state

            status = runner.get_checkpoint_status("thread-abc")

        assert status["processing_id"] == "p-123"
        assert status["status"] == ExtractionStatus.EXTRACTING.value
        assert status["retry_count"] == 1
