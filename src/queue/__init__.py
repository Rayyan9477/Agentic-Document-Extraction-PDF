"""
Queue module for asynchronous document processing.

Provides Celery-based task queue for:
- Async PDF processing
- Task status tracking
- Retry handling with exponential backoff
- Priority queue management
"""

import socket
from functools import lru_cache
from typing import Tuple
from urllib.parse import urlparse

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


# Cache Redis availability check for 5 seconds
_redis_check_cache: dict[str, Tuple[bool, float]] = {}


def is_redis_available(timeout: float = 1.0) -> bool:
    """
    Quick check if Redis is available.

    Uses socket connection with short timeout to avoid blocking.
    Results are cached for 5 seconds to reduce connection overhead.

    Args:
        timeout: Connection timeout in seconds (default 1.0).

    Returns:
        True if Redis is reachable, False otherwise.
    """
    import time

    cache_key = "redis_available"
    current_time = time.time()

    # Check cache (valid for 5 seconds)
    if cache_key in _redis_check_cache:
        result, cached_time = _redis_check_cache[cache_key]
        if current_time - cached_time < 5.0:
            return result

    try:
        # Parse Redis URL from Celery config
        broker_url = celery_app.conf.broker_url or "redis://localhost:6379/0"
        parsed = urlparse(broker_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379

        # Quick socket connection test
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port)) == 0
        sock.close()

        # Cache the result
        _redis_check_cache[cache_key] = (result, current_time)
        return result

    except Exception:
        _redis_check_cache[cache_key] = (False, current_time)
        return False


def get_queue_status() -> dict:
    """
    Get a quick overview of queue system status.

    Returns:
        Dictionary with queue system status.
    """
    redis_ok = is_redis_available()

    return {
        "redis_available": redis_ok,
        "async_processing_available": redis_ok,
        "recommendation": (
            "Use async_processing=true for background processing"
            if redis_ok else
            "Redis not available. Use async_processing=false for synchronous processing"
        ),
    }


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
    # Utilities
    "is_redis_available",
    "get_queue_status",
]
