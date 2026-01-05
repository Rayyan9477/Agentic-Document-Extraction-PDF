"""
Task management API routes.

Provides endpoints for task status checking,
cancellation, and worker management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.models import (
    QueueStatsResponse,
    TaskCancelResponse,
    TaskStatusResponse,
    WorkerStatusResponse,
)
from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/tasks/active",
    response_model=list[TaskStatusResponse],
    summary="List active tasks",
    description="Get a list of all active/pending tasks.",
)
async def list_active_tasks(
    http_request: Request,
) -> list[TaskStatusResponse]:
    """
    List all active and pending tasks.

    Args:
        http_request: HTTP request object.

    Returns:
        List of active task statuses.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "list_active_tasks_request",
        request_id=request_id,
    )

    # Quick Redis availability check (avoids 5+ second timeout)
    from src.queue import is_redis_available

    if not is_redis_available():
        logger.debug(
            "list_active_tasks_redis_unavailable",
            request_id=request_id,
        )
        return []  # Return empty list immediately if Redis not available

    try:
        from src.queue.celery_app import celery_app

        # Get active tasks from all workers with timeout
        inspect = celery_app.control.inspect(timeout=2.0)

        active_tasks = []

        # Get active (currently executing) tasks
        active = inspect.active() or {}
        for worker_tasks in active.values():
            for task in worker_tasks:
                active_tasks.append(
                    TaskStatusResponse(
                        task_id=task.get("id", ""),
                        status="STARTED",
                        ready=False,
                        successful=None,
                        progress=None,
                        result=None,
                        error=None,
                    )
                )

        # Get reserved (pending) tasks
        reserved = inspect.reserved() or {}
        for worker_tasks in reserved.values():
            for task in worker_tasks:
                active_tasks.append(
                    TaskStatusResponse(
                        task_id=task.get("id", ""),
                        status="PENDING",
                        ready=False,
                        successful=None,
                        progress=None,
                        result=None,
                        error=None,
                    )
                )

        return active_tasks

    except Exception as e:
        logger.error(
            "list_active_tasks_error",
            request_id=request_id,
            error=str(e),
        )
        # Return empty list on error instead of 500
        return []


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Get the status of an async processing task.",
)
async def get_task_status(
    task_id: str,
    http_request: Request,
) -> TaskStatusResponse:
    """
    Get the status of an async processing task.

    Args:
        task_id: Celery task ID.
        http_request: HTTP request object.

    Returns:
        Task status information.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "task_status_request",
        request_id=request_id,
        task_id=task_id,
    )

    try:
        from src.queue.tasks import get_task_status as get_status

        status_info = get_status(task_id)

        return TaskStatusResponse(
            task_id=task_id,
            status=status_info.get("status", "UNKNOWN"),
            ready=status_info.get("ready", False),
            successful=status_info.get("successful"),
            progress=status_info.get("progress"),
            result=status_info.get("result"),
            error=status_info.get("error"),
        )

    except Exception as e:
        logger.error(
            "task_status_error",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {e!s}",
        )


@router.delete(
    "/tasks/{task_id}",
    response_model=TaskCancelResponse,
    summary="Cancel task",
    description="Cancel a pending or running task.",
)
async def cancel_task(
    task_id: str,
    http_request: Request,
    terminate: bool = False,
) -> TaskCancelResponse:
    """
    Cancel a pending or running task.

    Args:
        task_id: Celery task ID.
        http_request: HTTP request object.
        terminate: Whether to terminate the worker process.

    Returns:
        Cancellation result.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "task_cancel_request",
        request_id=request_id,
        task_id=task_id,
        terminate=terminate,
    )

    try:
        from src.queue.tasks import cancel_task as do_cancel

        result = do_cancel(task_id, terminate=terminate)

        return TaskCancelResponse(
            task_id=task_id,
            cancelled=result.get("cancelled", False),
            reason=result.get("reason", ""),
        )

    except Exception as e:
        logger.error(
            "task_cancel_error",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {e!s}",
        )


@router.get(
    "/workers/status",
    response_model=WorkerStatusResponse,
    summary="Get worker status",
    description="Get the status of all Celery workers.",
)
async def get_worker_status(
    http_request: Request,
) -> WorkerStatusResponse:
    """
    Get the status of all Celery workers.

    Args:
        http_request: HTTP request object.

    Returns:
        Worker status information.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "worker_status_request",
        request_id=request_id,
    )

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        status = manager.get_worker_status()

        return WorkerStatusResponse(
            status=status.get("status", "unknown"),
            worker_count=status.get("worker_count", 0),
            workers=status.get("workers", []),
            registered_tasks=status.get("registered_tasks", []),
        )

    except Exception as e:
        logger.error(
            "worker_status_error",
            request_id=request_id,
            error=str(e),
        )
        return WorkerStatusResponse(
            status="error",
            worker_count=0,
            workers=[],
            registered_tasks=[],
        )


@router.get(
    "/queues/stats",
    response_model=QueueStatsResponse,
    summary="Get queue statistics",
    description="Get statistics for all task queues.",
)
async def get_queue_stats(
    http_request: Request,
) -> QueueStatsResponse:
    """
    Get statistics for all task queues.

    Args:
        http_request: HTTP request object.

    Returns:
        Queue statistics.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "queue_stats_request",
        request_id=request_id,
    )

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        stats = manager.get_queue_stats()

        return QueueStatsResponse(
            status=stats.get("status", "unknown"),
            queues=stats.get("queues", {}),
        )

    except Exception as e:
        logger.error(
            "queue_stats_error",
            request_id=request_id,
            error=str(e),
        )
        return QueueStatsResponse(
            status="error",
            queues={},
        )


@router.post(
    "/workers/scale",
    response_model=dict[str, Any],
    summary="Scale workers",
    description="Scale the number of worker processes.",
)
async def scale_workers(
    http_request: Request,
    concurrency: int,
) -> dict[str, Any]:
    """
    Scale the number of worker processes.

    Args:
        http_request: HTTP request object.
        concurrency: New concurrency level.

    Returns:
        Scale result.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "worker_scale_request",
        request_id=request_id,
        concurrency=concurrency,
    )

    if concurrency < 1 or concurrency > 32:
        raise HTTPException(
            status_code=400,
            detail="Concurrency must be between 1 and 32",
        )

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        result = manager.scale_workers(concurrency)

        return result

    except Exception as e:
        logger.error(
            "worker_scale_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scale workers: {e!s}",
        )


@router.post(
    "/queues/{queue_name}/purge",
    response_model=dict[str, Any],
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

    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        result = manager.purge_queue(queue_name)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Unknown error"),
            )

        return result

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
            detail=f"Failed to purge queue: {e!s}",
        )
