"""
Celery application configuration for document processing.

Configures Celery with Redis as broker and result backend,
with optimized settings for document processing workloads.
"""

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from celery import Celery

from src.config import get_logger, get_settings


logger = get_logger(__name__)


@dataclass(slots=True)
class CeleryConfig:
    """
    Celery configuration settings.

    Attributes:
        broker_url: Redis broker URL.
        result_backend: Redis result backend URL.
        task_serializer: Serialization format for tasks.
        result_serializer: Serialization format for results.
        accept_content: Accepted content types.
        timezone: Timezone for scheduled tasks.
        enable_utc: Use UTC for timestamps.
        task_track_started: Track task started state.
        task_time_limit: Hard time limit for tasks (seconds).
        task_soft_time_limit: Soft time limit for tasks (seconds).
        task_acks_late: Acknowledge tasks after execution.
        task_reject_on_worker_lost: Reject tasks if worker dies.
        worker_prefetch_multiplier: Tasks to prefetch per worker.
        worker_concurrency: Number of worker processes.
        result_expires: Result expiration time (seconds).
        task_default_queue: Default queue name.
        task_routes: Task routing configuration.
    """

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 600  # 10 minutes
    task_soft_time_limit: int = 540  # 9 minutes
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    worker_prefetch_multiplier: int = 1
    worker_concurrency: int = 4
    result_expires: int = 86400  # 24 hours
    task_default_queue: str = "document_processing"
    task_routes: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "src.queue.tasks.process_document_task": {"queue": "document_processing"},
            "src.queue.tasks.batch_process_task": {"queue": "batch_processing"},
            "src.queue.tasks.reprocess_failed_task": {"queue": "reprocessing"},
        }
    )

    def to_celery_config(self) -> dict[str, Any]:
        """Convert to Celery configuration dictionary."""
        return {
            "broker_url": self.broker_url,
            "result_backend": self.result_backend,
            "task_serializer": self.task_serializer,
            "result_serializer": self.result_serializer,
            "accept_content": self.accept_content,
            "timezone": self.timezone,
            "enable_utc": self.enable_utc,
            "task_track_started": self.task_track_started,
            "task_time_limit": self.task_time_limit,
            "task_soft_time_limit": self.task_soft_time_limit,
            "task_acks_late": self.task_acks_late,
            "task_reject_on_worker_lost": self.task_reject_on_worker_lost,
            "worker_prefetch_multiplier": self.worker_prefetch_multiplier,
            "worker_concurrency": self.worker_concurrency,
            "result_expires": self.result_expires,
            "task_default_queue": self.task_default_queue,
            "task_routes": self.task_routes,
        }


def create_celery_app(config: CeleryConfig | None = None) -> Celery:
    """
    Create and configure Celery application.

    Args:
        config: Celery configuration (uses defaults if not provided).

    Returns:
        Configured Celery application.
    """
    config = config or CeleryConfig()

    # Try to get Redis URL from settings
    try:
        settings = get_settings()
        if hasattr(settings, "redis_url") and settings.redis_url:
            config.broker_url = settings.redis_url
            # Parse Redis URL properly to set result backend to a different DB
            # This uses urllib.parse for robust URL handling instead of fragile string replace
            parsed = urlparse(settings.redis_url)
            # Extract current DB number from path (e.g., "/0" -> 0)
            current_db = 0
            if parsed.path and parsed.path.strip("/").isdigit():
                current_db = int(parsed.path.strip("/"))
            # Use next DB number for results (wrap at 15 for Redis default max)
            result_db = (current_db + 1) % 16
            # Rebuild URL with new DB path
            result_url = urlunparse(parsed._replace(path=f"/{result_db}"))
            config.result_backend = result_url
    except ImportError as e:
        logger.warning(
            "celery_settings_import_failed",
            error=str(e),
            message="Using default Celery configuration - settings module not available",
        )
    except AttributeError as e:
        logger.warning(
            "celery_settings_attribute_error",
            error=str(e),
            message="Settings missing redis_url attribute - using default configuration",
        )
    except Exception as e:
        logger.error(
            "celery_settings_config_error",
            error=str(e),
            error_type=type(e).__name__,
            message="Failed to load Redis settings for Celery - using default configuration",
        )

    app = Celery(
        "pdf_extraction",
        broker=config.broker_url,
        backend=config.result_backend,
        include=["src.queue.tasks"],
    )

    app.conf.update(config.to_celery_config())

    # Configure task queues
    # Each queue corresponds to a task type defined in task_routes above:
    # - document_processing: Single document extraction tasks
    # - batch_processing: Multi-document batch jobs
    # - reprocessing: Failed document retry tasks
    # - priority: Reserved for urgent documents (usage: task.apply_async(queue='priority'))
    app.conf.task_queues = {
        "document_processing": {
            "exchange": "document_processing",
            "routing_key": "document.#",
        },
        "batch_processing": {
            "exchange": "batch_processing",
            "routing_key": "batch.#",
        },
        "reprocessing": {
            "exchange": "reprocessing",
            "routing_key": "reprocess.#",
        },
        # Priority queue for urgent document processing.
        # Not used by default routing - must be explicitly specified:
        # Example: process_document_task.apply_async(args=[...], queue='priority')
        "priority": {
            "exchange": "priority",
            "routing_key": "priority.#",
        },
    }

    return app


# Create default Celery app instance
celery_app = create_celery_app()
