"""
Queue management API routes.

Provides endpoints for queue statistics,
worker status, and queue management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/queue/stats",
    summary="Get queue statistics",
    description="Get statistics for all task queues.",
)
async def get_queue_stats(
    http_request: Request,
) -> list[dict[str, Any]]:
    """
    Get statistics for all task queues.

    Args:
        http_request: HTTP request object.

    Returns:
        List of queue statistics.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "queue_stats_request",
        request_id=request_id,
    )

    # Quick Redis availability check (avoids 5+ second timeout)
    from src.queue import is_redis_available
    if not is_redis_available():
        logger.debug(
            "queue_stats_redis_unavailable",
            request_id=request_id,
        )
        return [
            {"name": "default", "pending": 0, "active": 0, "completed": 0, "failed": 0},
            {"name": "high_priority", "pending": 0, "active": 0, "completed": 0, "failed": 0},
            {"name": "low_priority", "pending": 0, "active": 0, "completed": 0, "failed": 0},
        ]

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        stats = manager.get_queue_stats()

        queues = stats.get("queues", {})

        # Convert to list format expected by frontend
        result = []
        for name, data in queues.items():
            result.append({
                "name": name,
                "pending": data.get("pending", 0),
                "active": data.get("active", 0),
                "completed": data.get("completed", 0),
                "failed": data.get("failed", 0),
            })

        # Add default queues if empty
        if not result:
            result = [
                {"name": "default", "pending": 0, "active": 0, "completed": 0, "failed": 0},
                {"name": "high_priority", "pending": 0, "active": 0, "completed": 0, "failed": 0},
                {"name": "low_priority", "pending": 0, "active": 0, "completed": 0, "failed": 0},
            ]

        return result

    except Exception as e:
        logger.warning(
            "queue_stats_error",
            request_id=request_id,
            error=str(e),
        )
        # Return default empty stats on error
        return [
            {"name": "default", "pending": 0, "active": 0, "completed": 0, "failed": 0},
        ]


@router.get(
    "/queue/workers",
    summary="Get worker status",
    description="Get status of all Celery workers.",
)
async def get_workers(
    http_request: Request,
) -> list[dict[str, Any]]:
    """
    Get status of all Celery workers.

    Args:
        http_request: HTTP request object.

    Returns:
        List of worker status objects.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "queue_workers_request",
        request_id=request_id,
    )

    # Quick Redis availability check (avoids 5+ second timeout)
    from src.queue import is_redis_available
    if not is_redis_available():
        logger.debug(
            "queue_workers_redis_unavailable",
            request_id=request_id,
        )
        return []

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        status = manager.get_worker_status()

        workers = status.get("workers", [])

        # Format workers for frontend
        result = []
        for worker in workers:
            result.append({
                "id": worker.get("id", "unknown"),
                "name": worker.get("name", "unknown"),
                "status": worker.get("status", "unknown"),
                "active_tasks": worker.get("active_tasks", 0),
                "processed": worker.get("processed", 0),
                "last_heartbeat": worker.get("last_heartbeat"),
            })

        return result

    except Exception as e:
        logger.warning(
            "queue_workers_error",
            request_id=request_id,
            error=str(e),
        )
        return []


@router.post(
    "/queue/{queue_name}/purge",
    summary="Purge queue",
    description="Purge all messages from a queue.",
)
async def purge_queue(
    queue_name: str,
    http_request: Request,
) -> dict[str, Any]:
    """
    Purge all messages from a queue.

    Args:
        queue_name: Name of queue to purge.
        http_request: HTTP request object.

    Returns:
        Purge result.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "queue_purge_request",
        request_id=request_id,
        queue_name=queue_name,
    )

    # Quick Redis availability check (avoids 5+ second timeout)
    from src.queue import is_redis_available
    if not is_redis_available():
        raise HTTPException(
            status_code=503,
            detail="Redis is not available. Cannot purge queue.",
        )

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        result = manager.purge_queue(queue_name)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Unknown error"),
            )

        return {"purged": result.get("purged", 0)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "queue_purge_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge queue: {str(e)}",
        )
