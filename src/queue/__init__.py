"""
Queue module for asynchronous document processing.

Provides Celery-based task queue for:
- Async PDF processing
- Task status tracking
- Retry handling with exponential backoff
- Priority queue management
"""

from src.queue.celery_app import (
    celery_app,
    CeleryConfig,
)
from src.queue.tasks import (
    process_document_task,
    batch_process_task,
    reprocess_failed_task,
    get_task_status,
    cancel_task,
    TaskResult,
    TaskStatus,
)
from src.queue.worker import (
    WorkerManager,
    WorkerConfig,
)


__all__ = [
    # Celery app
    "celery_app",
    "CeleryConfig",
    # Tasks
    "process_document_task",
    "batch_process_task",
    "reprocess_failed_task",
    "get_task_status",
    "cancel_task",
    "TaskResult",
    "TaskStatus",
    # Worker management
    "WorkerManager",
    "WorkerConfig",
]
