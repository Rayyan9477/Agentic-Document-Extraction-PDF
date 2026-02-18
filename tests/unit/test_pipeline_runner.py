"""
Unit tests for PipelineRunner.

Tests cover:
- Initialization and configuration
- extract_from_pdf validation
- extract_from_bytes workflow
- resume_extraction
- Checkpoint status retrieval
- Image enhancement configuration
- Lazy workflow initialization
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import OrchestrationError
from src.pipeline.runner import PipelineRunner
from src.pipeline.state import ExtractionStatus


# ---------------------------------------------------------------------------
# TestPipelineRunnerInit
# ---------------------------------------------------------------------------


class TestPipelineRunnerInit:
    """Tests for PipelineRunner initialization."""

    def test_default_init(self) -> None:
        runner = PipelineRunner()
        assert runner._enable_checkpointing is True
        assert runner._max_retries == 2
        assert runner._dpi == 200
        assert runner._max_image_dimension == 2048

    def test_custom_params(self) -> None:
        mock_client = MagicMock()
        runner = PipelineRunner(
            client=mock_client,
            enable_checkpointing=False,
            max_retries=5,
            dpi=300,
            max_image_dimension=4096,
        )
        assert runner._client is mock_client
        assert runner._enable_checkpointing is False
        assert runner._max_retries == 5
        assert runner._dpi == 300

    def test_workflow_not_initialized_at_start(self) -> None:
        runner = PipelineRunner()
        assert runner._orchestrator is None
        assert runner._compiled_workflow is None

    def test_enhancement_default(self) -> None:
        runner = PipelineRunner()
        # Should default based on settings
        assert isinstance(runner._enable_enhancement, bool)

    def test_enhancement_explicit_true(self) -> None:
        runner = PipelineRunner(enable_image_enhancement=True)
        assert runner._enable_enhancement is True

    def test_enhancement_explicit_false(self) -> None:
        runner = PipelineRunner(enable_image_enhancement=False)
        assert runner._enable_enhancement is False


# ---------------------------------------------------------------------------
# TestExtractFromPdf
# ---------------------------------------------------------------------------


class TestExtractFromPdf:
    """Tests for extract_from_pdf method."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        runner = PipelineRunner(enable_checkpointing=False)
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            runner.extract_from_pdf(tmp_path / "nonexistent.pdf")

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_calls_workflow(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        # Create a dummy PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy content")

        # Mock the workflow
        mock_orch = MagicMock()
        mock_orch.run_extraction.return_value = {
            "processing_id": "test-123",
            "status": ExtractionStatus.COMPLETED.value,
            "overall_confidence": 0.95,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner(enable_checkpointing=False)
        result = runner.extract_from_pdf(str(pdf_path))

        assert result["status"] == ExtractionStatus.COMPLETED.value
        mock_orch.run_extraction.assert_called_once()

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_generates_processing_id(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        mock_orch = MagicMock()
        mock_orch.run_extraction.return_value = {
            "processing_id": "generated-id",
            "status": ExtractionStatus.COMPLETED.value,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner(enable_checkpointing=False)
        runner.extract_from_pdf(str(pdf_path))

        call_args = mock_orch.run_extraction.call_args
        initial_state = call_args[1].get("initial_state") or call_args[0][0]
        assert initial_state.get("processing_id") is not None


# ---------------------------------------------------------------------------
# TestExtractFromBytes
# ---------------------------------------------------------------------------


class TestExtractFromBytes:
    """Tests for extract_from_bytes method."""

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_extract_from_bytes_calls_workflow(
        self, mock_create: MagicMock
    ) -> None:
        mock_orch = MagicMock()
        mock_orch.run_extraction.return_value = {
            "processing_id": "bytes-123",
            "status": ExtractionStatus.COMPLETED.value,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner(enable_checkpointing=False)

        # Mock the image conversion to avoid needing real PDF bytes
        with patch.object(runner, "_convert_pdf_bytes_to_images", return_value=[]):
            result = runner.extract_from_bytes(b"%PDF-1.4 dummy")

        assert result["status"] == ExtractionStatus.COMPLETED.value

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_extract_from_bytes_conversion_failure(
        self, mock_create: MagicMock
    ) -> None:
        mock_create.return_value = (MagicMock(), MagicMock())

        runner = PipelineRunner(enable_checkpointing=False)

        with patch.object(
            runner,
            "_convert_pdf_bytes_to_images",
            side_effect=RuntimeError("Bad PDF"),
        ):
            result = runner.extract_from_bytes(b"not-a-pdf")

        assert result["status"] == ExtractionStatus.FAILED.value


# ---------------------------------------------------------------------------
# TestResumeExtraction
# ---------------------------------------------------------------------------


class TestResumeExtraction:
    """Tests for resume_extraction method."""

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_resume_no_checkpoint_raises(self, mock_create: MagicMock) -> None:
        mock_orch = MagicMock()
        mock_orch.get_checkpoint_state.return_value = None
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner(enable_checkpointing=True)
        with pytest.raises(OrchestrationError, match="No checkpoint found"):
            runner.resume_extraction(thread_id="nonexistent-thread")

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_resume_with_checkpoint(self, mock_create: MagicMock) -> None:
        mock_orch = MagicMock()
        mock_orch.get_checkpoint_state.return_value = {
            "processing_id": "resume-test",
            "status": ExtractionStatus.HUMAN_REVIEW.value,
        }
        mock_orch.resume_extraction.return_value = {
            "processing_id": "resume-test",
            "status": ExtractionStatus.COMPLETED.value,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner(enable_checkpointing=True)
        result = runner.resume_extraction(thread_id="valid-thread")

        assert result["status"] == ExtractionStatus.COMPLETED.value
        mock_orch.resume_extraction.assert_called_once()


# ---------------------------------------------------------------------------
# TestCheckpointStatus
# ---------------------------------------------------------------------------


class TestCheckpointStatus:
    """Tests for checkpoint status retrieval."""

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_get_checkpoint_status_found(self, mock_create: MagicMock) -> None:
        mock_orch = MagicMock()
        mock_orch.get_checkpoint_state.return_value = {
            "status": ExtractionStatus.VALIDATING.value,
            "overall_confidence": 0.75,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner()
        status = runner.get_checkpoint_status("some-thread")
        assert status is not None

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_get_checkpoint_status_not_found(self, mock_create: MagicMock) -> None:
        mock_orch = MagicMock()
        mock_orch.get_checkpoint_state.return_value = None
        mock_create.return_value = (mock_orch, MagicMock())

        runner = PipelineRunner()
        status = runner.get_checkpoint_status("missing-thread")
        assert status is None


# ---------------------------------------------------------------------------
# TestLazyInit
# ---------------------------------------------------------------------------


class TestLazyInit:
    """Tests for lazy workflow initialization."""

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_workflow_initialized_on_first_call(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        mock_orch = MagicMock()
        mock_orch.run_extraction.return_value = {
            "status": ExtractionStatus.COMPLETED.value,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        runner = PipelineRunner(enable_checkpointing=False)
        assert runner._orchestrator is None

        runner.extract_from_pdf(str(pdf_path))
        assert runner._orchestrator is not None
        mock_create.assert_called_once()

    @patch("src.pipeline.runner.create_extraction_workflow")
    def test_workflow_not_reinitialized(
        self, mock_create: MagicMock, tmp_path: Path
    ) -> None:
        mock_orch = MagicMock()
        mock_orch.run_extraction.return_value = {
            "status": ExtractionStatus.COMPLETED.value,
        }
        mock_create.return_value = (mock_orch, MagicMock())

        pdf1 = tmp_path / "a.pdf"
        pdf1.write_bytes(b"%PDF-1.4 dummy")
        pdf2 = tmp_path / "b.pdf"
        pdf2.write_bytes(b"%PDF-1.4 dummy2")

        runner = PipelineRunner(enable_checkpointing=False)
        runner.extract_from_pdf(str(pdf1))
        runner.extract_from_pdf(str(pdf2))

        # Only created once (lazy singleton)
        mock_create.assert_called_once()
