"""
Unit tests for the queue module.

Tests Celery task definitions, worker management,
and task status handling.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.queue.celery_app import CeleryConfig, create_celery_app
from src.queue.tasks import (
    TaskResult,
    TaskStatus,
    get_task_status,
    cancel_task,
)
from src.queue.worker import (
    WorkerManager,
    WorkerConfig,
    WorkerState,
)


class TestCeleryConfig:
    """Test cases for CeleryConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CeleryConfig()

        assert config.broker_url == "redis://localhost:6379/0"
        assert config.result_backend == "redis://localhost:6379/1"
        assert config.task_serializer == "json"
        assert config.timezone == "UTC"
        assert config.task_track_started is True
        assert config.task_time_limit == 600
        assert config.worker_concurrency == 4

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CeleryConfig(
            broker_url="redis://custom:6380/0",
            worker_concurrency=8,
            task_time_limit=1200,
        )

        assert config.broker_url == "redis://custom:6380/0"
        assert config.worker_concurrency == 8
        assert config.task_time_limit == 1200

    def test_to_celery_config(self) -> None:
        """Test conversion to Celery config dict."""
        config = CeleryConfig()
        celery_dict = config.to_celery_config()

        assert isinstance(celery_dict, dict)
        assert celery_dict["broker_url"] == config.broker_url
        assert celery_dict["timezone"] == "UTC"
        assert celery_dict["task_track_started"] is True

    def test_task_routes(self) -> None:
        """Test default task routes."""
        config = CeleryConfig()

        assert "src.queue.tasks.process_document_task" in config.task_routes
        assert "src.queue.tasks.batch_process_task" in config.task_routes


class TestTaskResult:
    """Test cases for TaskResult."""

    def test_default_values(self) -> None:
        """Test default TaskResult values."""
        result = TaskResult(
            task_id="test-123",
            processing_id="proc-001",
            status=TaskStatus.PENDING,
        )

        assert result.task_id == "test-123"
        assert result.processing_id == "proc-001"
        assert result.status == TaskStatus.PENDING
        assert result.errors == []
        assert result.retry_count == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = TaskResult(
            task_id="test-123",
            processing_id="proc-001",
            status=TaskStatus.COMPLETED,
            overall_confidence=0.92,
            field_count=10,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["task_id"] == "test-123"
        assert result_dict["status"] == "completed"
        assert result_dict["overall_confidence"] == 0.92

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "task_id": "test-456",
            "processing_id": "proc-002",
            "status": "completed",
            "field_count": 15,
            "overall_confidence": 0.88,
        }

        result = TaskResult.from_dict(data)

        assert result.task_id == "test-456"
        assert result.status == TaskStatus.COMPLETED
        assert result.field_count == 15

    def test_all_status_values(self) -> None:
        """Test all status enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.STARTED.value == "started"
        assert TaskStatus.PROCESSING.value == "processing"
        assert TaskStatus.VALIDATING.value == "validating"
        assert TaskStatus.EXPORTING.value == "exporting"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.RETRYING.value == "retrying"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestWorkerConfig:
    """Test cases for WorkerConfig."""

    def test_default_config(self) -> None:
        """Test default worker configuration."""
        config = WorkerConfig()

        assert config.concurrency == 4
        assert len(config.queues) == 3
        assert "document_processing" in config.queues
        assert config.loglevel == "INFO"
        assert config.pool == "prefork"

    def test_custom_config(self) -> None:
        """Test custom worker configuration."""
        config = WorkerConfig(
            concurrency=8,
            queues=["custom_queue"],
            loglevel="DEBUG",
        )

        assert config.concurrency == 8
        assert config.queues == ["custom_queue"]
        assert config.loglevel == "DEBUG"

    def test_autoscale_config(self) -> None:
        """Test autoscale configuration."""
        config = WorkerConfig(
            autoscale=(2, 8),
        )

        assert config.autoscale == (2, 8)


class TestWorkerManager:
    """Test cases for WorkerManager."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        manager = WorkerManager()

        assert manager.config is not None
        assert manager.config.concurrency == 4

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = WorkerConfig(concurrency=8)
        manager = WorkerManager(config)

        assert manager.config.concurrency == 8

    def test_build_worker_command(self) -> None:
        """Test building worker command."""
        manager = WorkerManager()
        cmd = manager.build_worker_command()

        assert isinstance(cmd, list)
        assert "-m" in cmd
        assert "celery" in cmd
        assert "worker" in cmd
        assert "--concurrency" in cmd
        assert "4" in cmd

    def test_build_worker_command_with_autoscale(self) -> None:
        """Test worker command with autoscale."""
        config = WorkerConfig(autoscale=(2, 8))
        manager = WorkerManager(config)
        cmd = manager.build_worker_command()

        assert "--autoscale" in cmd
        assert "8,2" in cmd

    @patch("src.queue.worker.celery_app")
    def test_get_worker_status(self, mock_celery: MagicMock) -> None:
        """Test getting worker status."""
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {"worker1": []}
        mock_inspect.reserved.return_value = {}
        mock_inspect.scheduled.return_value = {}
        mock_inspect.stats.return_value = {
            "worker1": {"total": {}, "pool": {}, "broker": {}}
        }
        mock_inspect.registered.return_value = {"worker1": ["task1", "task2"]}
        mock_celery.control.inspect.return_value = mock_inspect

        manager = WorkerManager()
        status = manager.get_worker_status()

        assert status["status"] == "ok"
        assert status["worker_count"] == 1

    @patch("src.queue.worker.celery_app")
    def test_get_worker_status_no_workers(self, mock_celery: MagicMock) -> None:
        """Test worker status when no workers."""
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {}
        mock_inspect.reserved.return_value = {}
        mock_inspect.scheduled.return_value = {}
        mock_inspect.stats.return_value = {}
        mock_inspect.registered.return_value = {}
        mock_celery.control.inspect.return_value = mock_inspect

        manager = WorkerManager()
        status = manager.get_worker_status()

        assert status["status"] == "no_workers"
        assert status["worker_count"] == 0

    @patch("src.queue.worker.celery_app")
    def test_health_check_healthy(self, mock_celery: MagicMock) -> None:
        """Test health check with healthy workers."""
        mock_celery.control.ping.return_value = [
            {"worker1": {"ok": "pong"}},
            {"worker2": {"ok": "pong"}},
        ]

        manager = WorkerManager()
        health = manager.health_check()

        assert health["healthy"] is True
        assert health["count"] == 2
        assert "worker1" in health["workers"]

    @patch("src.queue.worker.celery_app")
    def test_health_check_no_workers(self, mock_celery: MagicMock) -> None:
        """Test health check with no workers."""
        mock_celery.control.ping.return_value = []

        manager = WorkerManager()
        health = manager.health_check()

        assert health["healthy"] is False
        assert "No workers responding" in health["reason"]

    @patch("src.queue.worker.celery_app")
    def test_scale_workers(self, mock_celery: MagicMock) -> None:
        """Test scaling workers."""
        manager = WorkerManager()
        result = manager.scale_workers(8)

        mock_celery.control.pool_resize.assert_called_once_with(8)
        assert result["status"] == "ok"

    @patch("src.queue.worker.celery_app")
    def test_scale_workers_invalid(self, mock_celery: MagicMock) -> None:
        """Test scaling with invalid concurrency."""
        manager = WorkerManager()
        result = manager.scale_workers(0)

        assert result["status"] == "error"
        mock_celery.control.pool_resize.assert_not_called()

    @patch("src.queue.worker.celery_app")
    def test_broadcast_shutdown(self, mock_celery: MagicMock) -> None:
        """Test broadcasting shutdown."""
        manager = WorkerManager()
        result = manager.broadcast_shutdown(graceful=True)

        mock_celery.control.broadcast.assert_called_once_with("shutdown")
        assert result["status"] == "ok"


class TestTaskStatusFunctions:
    """Test cases for task status functions."""

    @patch("celery.result.AsyncResult")
    @patch("src.queue.tasks.celery_app")
    def test_get_task_status_pending(
        self,
        mock_celery: MagicMock,
        mock_async_result: MagicMock,
    ) -> None:
        """Test getting status of pending task."""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.ready.return_value = False
        mock_result.info = None
        mock_async_result.return_value = mock_result

        status = get_task_status("test-task-123")

        assert status["task_id"] == "test-task-123"
        assert status["status"] == "PENDING"
        assert status["ready"] is False

    @patch("celery.result.AsyncResult")
    @patch("src.queue.tasks.celery_app")
    def test_get_task_status_completed(
        self,
        mock_celery: MagicMock,
        mock_async_result: MagicMock,
    ) -> None:
        """Test getting status of completed task."""
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        mock_result.get.return_value = {"data": "test"}
        mock_async_result.return_value = mock_result

        status = get_task_status("test-task-456")

        assert status["task_id"] == "test-task-456"
        assert status["ready"] is True
        assert status["successful"] is True
        assert status["result"] == {"data": "test"}

    @patch("celery.result.AsyncResult")
    @patch("src.queue.tasks.celery_app")
    def test_cancel_task_pending(
        self,
        mock_celery: MagicMock,
        mock_async_result: MagicMock,
    ) -> None:
        """Test canceling a pending task."""
        mock_result = MagicMock()
        mock_result.ready.return_value = False
        mock_async_result.return_value = mock_result

        result = cancel_task("test-task-789")

        mock_celery.control.revoke.assert_called_once_with(
            "test-task-789",
            terminate=False,
        )
        assert result["cancelled"] is True

    @patch("celery.result.AsyncResult")
    @patch("src.queue.tasks.celery_app")
    def test_cancel_task_already_completed(
        self,
        mock_celery: MagicMock,
        mock_async_result: MagicMock,
    ) -> None:
        """Test canceling an already completed task."""
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_async_result.return_value = mock_result

        result = cancel_task("test-task-completed")

        assert result["cancelled"] is False
        assert "already completed" in result["reason"]


class TestWorkerState:
    """Test cases for WorkerState enum."""

    def test_state_values(self) -> None:
        """Test worker state enum values."""
        assert WorkerState.STARTING.value == "starting"
        assert WorkerState.RUNNING.value == "running"
        assert WorkerState.STOPPING.value == "stopping"
        assert WorkerState.STOPPED.value == "stopped"
        assert WorkerState.ERROR.value == "error"


class TestCeleryAppCreation:
    """Test cases for Celery app creation."""

    @patch("src.queue.celery_app.get_settings")
    def test_create_app_default(self, mock_settings: MagicMock) -> None:
        """Test creating Celery app with defaults."""
        mock_settings.side_effect = Exception("No settings")

        app = create_celery_app()

        assert app is not None
        assert app.main == "pdf_extraction"

    @patch("src.queue.celery_app.get_settings")
    def test_create_app_custom_config(self, mock_settings: MagicMock) -> None:
        """Test creating Celery app with custom config."""
        mock_settings.side_effect = Exception("No settings")

        config = CeleryConfig(
            broker_url="redis://custom:6380/0",
            worker_concurrency=16,
        )
        app = create_celery_app(config)

        assert app is not None
        assert app.conf.broker_url == "redis://custom:6380/0"


class TestProcessDocumentTask:
    """Test cases for process_document_task."""

    def test_task_result_serialization(self) -> None:
        """Test TaskResult can be serialized and deserialized."""
        result = TaskResult(
            task_id="test-task-001",
            processing_id="proc-001",
            status=TaskStatus.COMPLETED,
            pdf_path="/test/doc.pdf",
            field_count=10,
            overall_confidence=0.92,
        )

        result_dict = result.to_dict()
        restored = TaskResult.from_dict(result_dict)

        assert restored.task_id == "test-task-001"
        assert restored.status == TaskStatus.COMPLETED
        assert restored.overall_confidence == 0.92

    def test_task_result_with_errors(self) -> None:
        """Test TaskResult with errors."""
        result = TaskResult(
            task_id="test-task-002",
            processing_id="",
            status=TaskStatus.FAILED,
            errors=["File not found", "Processing error"],
        )

        assert len(result.errors) == 2
        assert result.status == TaskStatus.FAILED

        result_dict = result.to_dict()
        assert result_dict["errors"] == ["File not found", "Processing error"]
