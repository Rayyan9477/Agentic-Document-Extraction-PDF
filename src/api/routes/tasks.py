"""
Task management API routes.

Provides endpoints for task status checking,
cancellation, and worker management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.models import (
    TaskStatusResponse,
    TaskCancelResponse,
    WorkerStatusResponse,
    QueueStatsResponse,
)
from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()


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
            detail=f"Failed to get task status: {str(e)}",
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
            detail=f"Failed to cancel task: {str(e)}",
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
            detail=f"Failed to scale workers: {str(e)}",
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
            detail=f"Failed to purge queue: {str(e)}",
        )
