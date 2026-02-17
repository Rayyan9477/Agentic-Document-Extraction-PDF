"""
Unit tests for the API module.

Tests FastAPI endpoints, request/response models,
and error handling.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.models import (
    BatchProcessRequest,
    ConfidenceLevelEnum,
    ExportFormatEnum,
    ProcessingPriority,
    ProcessRequest,
    ProcessResponse,
    TaskStatusEnum,
)


@pytest.fixture
def app() -> Any:
    """Create test FastAPI application."""
    return create_app()


@pytest.fixture
def client(app: Any) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestProcessRequest:
    """Test cases for ProcessRequest model."""

    def test_valid_request(self) -> None:
        """Test valid process request."""
        request = ProcessRequest(
            pdf_path="/test/sample.pdf",
            schema_name="cms1500",
            export_format=ExportFormatEnum.JSON,
        )

        assert request.pdf_path == "/test/sample.pdf"
        assert request.schema_name == "cms1500"
        assert request.export_format == ExportFormatEnum.JSON
        assert request.async_processing is False

    def test_default_values(self) -> None:
        """Test default values."""
        request = ProcessRequest(pdf_path="/test/sample.pdf")

        assert request.schema_name is None
        assert request.export_format == ExportFormatEnum.JSON
        assert request.mask_phi is False
        assert request.priority == ProcessingPriority.NORMAL

    def test_invalid_pdf_extension(self) -> None:
        """Test validation of PDF extension."""
        with pytest.raises(ValueError, match="pdf extension"):
            ProcessRequest(pdf_path="/test/sample.txt")

    def test_async_with_callback(self) -> None:
        """Test async request with callback URL."""
        request = ProcessRequest(
            pdf_path="/test/sample.pdf",
            async_processing=True,
            callback_url="https://example.com/callback",
        )

        assert request.async_processing is True
        assert request.callback_url == "https://example.com/callback"


class TestBatchProcessRequest:
    """Test cases for BatchProcessRequest model."""

    def test_valid_batch_request(self) -> None:
        """Test valid batch process request."""
        request = BatchProcessRequest(
            pdf_paths=["/test/doc1.pdf", "/test/doc2.pdf"],
            output_dir="/output",
        )

        assert len(request.pdf_paths) == 2
        assert request.output_dir == "/output"

    def test_empty_paths_rejected(self) -> None:
        """Test that empty paths list is rejected."""
        with pytest.raises(ValueError):
            BatchProcessRequest(
                pdf_paths=[],
                output_dir="/output",
            )

    def test_invalid_extension_in_batch(self) -> None:
        """Test validation of extensions in batch."""
        with pytest.raises(ValueError, match="pdf extension"):
            BatchProcessRequest(
                pdf_paths=["/test/doc1.pdf", "/test/doc2.txt"],
                output_dir="/output",
            )


class TestProcessResponse:
    """Test cases for ProcessResponse model."""

    def test_minimal_response(self) -> None:
        """Test minimal response."""
        response = ProcessResponse(
            processing_id="test-001",
            status=TaskStatusEnum.COMPLETED,
        )

        assert response.processing_id == "test-001"
        assert response.status == TaskStatusEnum.COMPLETED
        assert response.data == {}

    def test_full_response(self) -> None:
        """Test full response with all fields."""
        response = ProcessResponse(
            processing_id="test-002",
            status=TaskStatusEnum.COMPLETED,
            data={"field1": "value1"},
            overall_confidence=0.92,
            confidence_level=ConfidenceLevelEnum.HIGH,
        )

        assert response.overall_confidence == 0.92
        assert response.confidence_level == ConfidenceLevelEnum.HIGH


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_basic_health(self, client: TestClient) -> None:
        """Test basic health check."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "timestamp" in data

    def test_deep_health_check(self, client: TestClient) -> None:
        """Test deep health check."""
        response = client.get("/api/v1/health?deep=true")

        assert response.status_code == 200
        data = response.json()
        assert "components" in data

    def test_liveness_probe(self, client: TestClient) -> None:
        """Test liveness probe."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_probe(self, client: TestClient) -> None:
        """Test readiness probe."""
        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestDocumentEndpoints:
    """Test cases for document processing endpoints."""

    def test_process_document_file_not_found(self, client: TestClient) -> None:
        """Test processing with non-existent file."""
        response = client.post(
            "/api/v1/documents/process",
            json={
                "pdf_path": "/nonexistent/file.pdf",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("src.queue.tasks.process_document_task")
    def test_process_document_async(
        self,
        mock_task: MagicMock,
        client: TestClient,
    ) -> None:
        """Test asynchronous document processing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            temp_path = f.name

        try:
            mock_async_result = MagicMock()
            mock_async_result.id = "task-123"
            mock_task.delay.return_value = mock_async_result

            response = client.post(
                "/api/v1/documents/process",
                json={
                    "pdf_path": temp_path,
                    "async_processing": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "task-123"
            assert "status_url" in data

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_processing_result_not_found(self, client: TestClient) -> None:
        """Test getting non-existent result."""
        response = client.get("/api/v1/documents/nonexistent-id-test")

        assert response.status_code == 404


class TestTaskEndpoints:
    """Test cases for task management endpoints."""

    @patch("src.queue.tasks.get_task_status")
    def test_get_task_status(
        self,
        mock_get_status: MagicMock,
        client: TestClient,
    ) -> None:
        """Test getting task status."""
        mock_get_status.return_value = {
            "status": "PENDING",
            "ready": False,
            "successful": None,
        }

        response = client.get("/api/v1/tasks/task-123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-123"

    @patch("src.queue.tasks.cancel_task")
    def test_cancel_task(
        self,
        mock_cancel: MagicMock,
        client: TestClient,
    ) -> None:
        """Test canceling a task."""
        mock_cancel.return_value = {
            "cancelled": True,
            "reason": "",
        }

        response = client.delete("/api/v1/tasks/task-456")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-456"

    @patch("src.queue.worker.WorkerManager")
    def test_get_worker_status(
        self,
        mock_manager_class: MagicMock,
        client: TestClient,
    ) -> None:
        """Test getting worker status."""
        mock_manager = MagicMock()
        mock_manager.get_worker_status.return_value = {
            "status": "ok",
            "worker_count": 2,
            "workers": [],
            "registered_tasks": [],
        }
        mock_manager_class.return_value = mock_manager

        response = client.get("/api/v1/workers/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @patch("src.queue.worker.WorkerManager")
    def test_get_queue_stats(
        self,
        mock_manager_class: MagicMock,
        client: TestClient,
    ) -> None:
        """Test getting queue statistics."""
        mock_manager = MagicMock()
        mock_manager.get_queue_stats.return_value = {
            "status": "ok",
            "queues": {"document_processing": {"active": 0, "reserved": 0}},
        }
        mock_manager_class.return_value = mock_manager

        response = client.get("/api/v1/queues/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestSchemaEndpoints:
    """Test cases for schema management endpoints."""

    @patch("src.schemas.get_all_schemas")
    def test_list_schemas(
        self,
        mock_get_schemas: MagicMock,
        client: TestClient,
    ) -> None:
        """Test listing schemas."""
        # Mock returns a list of objects with .name attribute (like DocumentSchema)
        mock_schema = MagicMock()
        mock_schema.name = "cms1500"
        mock_schema.description = "CMS-1500 form"
        mock_schema.document_type = "CMS-1500"
        mock_schema.fields = [MagicMock(), MagicMock()]
        mock_schema.version = "1.0.0"
        mock_get_schemas.return_value = [mock_schema]

        response = client.get("/api/v1/schemas")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["schemas"]) == 1

    @patch("src.schemas.get_schema")
    def test_get_schema(
        self,
        mock_get_schema: MagicMock,
        client: TestClient,
    ) -> None:
        """Test getting a specific schema."""
        mock_get_schema.return_value = {
            "description": "CMS-1500 form",
            "fields": {"field1": {}},
        }

        response = client.get("/api/v1/schemas/cms1500")

        assert response.status_code == 200
        data = response.json()
        assert "fields" in data

    @patch("src.schemas.get_schema")
    def test_get_schema_not_found(
        self,
        mock_get_schema: MagicMock,
        client: TestClient,
    ) -> None:
        """Test getting non-existent schema."""
        mock_get_schema.return_value = None

        response = client.get("/api/v1/schemas/nonexistent")

        assert response.status_code == 404


class TestErrorHandling:
    """Test cases for error handling."""

    def test_validation_error(self, client: TestClient) -> None:
        """Test validation error response."""
        response = client.post(
            "/api/v1/documents/process",
            json={
                "pdf_path": "",  # Empty path
            },
        )

        assert response.status_code == 422  # Validation error

    def test_request_id_header(self, client: TestClient) -> None:
        """Test that request ID is returned in headers."""
        response = client.get("/api/v1/health")

        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers

    def test_custom_request_id(self, client: TestClient) -> None:
        """Test custom request ID is preserved."""
        response = client.get(
            "/api/v1/health",
            headers={"X-Request-ID": "custom-request-123"},
        )

        assert response.headers["X-Request-ID"] == "custom-request-123"


class TestRootEndpoint:
    """Test cases for root endpoint."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestEnumValues:
    """Test cases for enum values."""

    def test_export_format_values(self) -> None:
        """Test export format enum values."""
        assert ExportFormatEnum.JSON.value == "json"
        assert ExportFormatEnum.EXCEL.value == "excel"
        assert ExportFormatEnum.BOTH.value == "both"

    def test_processing_priority_values(self) -> None:
        """Test processing priority enum values."""
        assert ProcessingPriority.LOW.value == "low"
        assert ProcessingPriority.NORMAL.value == "normal"
        assert ProcessingPriority.HIGH.value == "high"

    def test_task_status_values(self) -> None:
        """Test task status enum values."""
        assert TaskStatusEnum.PENDING.value == "pending"
        assert TaskStatusEnum.COMPLETED.value == "completed"
        assert TaskStatusEnum.FAILED.value == "failed"

    def test_confidence_level_values(self) -> None:
        """Test confidence level enum values."""
        assert ConfidenceLevelEnum.HIGH.value == "high"
        assert ConfidenceLevelEnum.MEDIUM.value == "medium"
        assert ConfidenceLevelEnum.LOW.value == "low"
