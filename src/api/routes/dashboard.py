"""
Dashboard API routes.

Provides endpoints for dashboard metrics, activity,
and system overview data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request

from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()


def _get_system_metrics() -> dict[str, Any]:
    """Get system resource metrics."""
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
        }
    except ImportError:
        # psutil not installed - return zeros silently as this is optional
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
        }
    except PermissionError as e:
        logger.warning(
            "system_metrics_permission_error",
            error=str(e),
            message="Insufficient permissions to read system metrics",
        )
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
        }
    except Exception as e:
        logger.error(
            "system_metrics_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
        }


def _get_processing_metrics() -> dict[str, Any]:
    """Get document processing metrics."""
    # These would normally come from a database or metrics store
    return {
        "documents_processed_today": 0,
        "documents_processed_week": 0,
        "documents_processed_month": 0,
        "average_processing_time_ms": 0,
        "success_rate": 1.0,
        "error_rate": 0.0,
    }


def _get_queue_metrics() -> dict[str, Any]:
    """Get task queue metrics (queue disabled - synchronous mode)."""
    # Queue system is disabled - using synchronous processing
    return {
        "pending_tasks": 0,
        "active_tasks": 0,
        "worker_count": 0,
        "mode": "synchronous",
    }


@router.get(
    "/dashboard/metrics",
    summary="Get dashboard metrics",
    description="Get aggregated metrics for the dashboard.",
)
async def get_dashboard_metrics(
    http_request: Request,
) -> dict[str, Any]:
    """
    Get dashboard metrics.

    Args:
        http_request: HTTP request object.

    Returns:
        Dashboard metrics including system, processing, and queue stats.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "dashboard_metrics_request",
        request_id=request_id,
    )

    timestamp = datetime.now(UTC).isoformat()

    return {
        "system": _get_system_metrics(),
        "processing": _get_processing_metrics(),
        "queue": _get_queue_metrics(),
        "timestamp": timestamp,
    }


@router.get(
    "/dashboard/activity",
    summary="Get recent activity",
    description="Get recent processing activity.",
)
async def get_recent_activity(
    http_request: Request,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Get recent processing activity.

    Args:
        http_request: HTTP request object.
        limit: Maximum number of activities to return.

    Returns:
        List of recent activities.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "dashboard_activity_request",
        request_id=request_id,
        limit=limit,
    )

    # In a real implementation, this would query from database
    # For now, return empty list
    return []


@router.get(
    "/dashboard/summary",
    summary="Get dashboard summary",
    description="Get overall system summary for dashboard.",
)
async def get_dashboard_summary(
    http_request: Request,
) -> dict[str, Any]:
    """
    Get dashboard summary.

    Args:
        http_request: HTTP request object.

    Returns:
        Overall system summary.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "dashboard_summary_request",
        request_id=request_id,
    )

    timestamp = datetime.now(UTC).isoformat()

    # Check service health
    services = {
        "api": "healthy",
        "database": "healthy",
        "queue": "disabled",  # Queue system disabled - using synchronous processing
        "vlm": "unknown",
    }

    # Check LM Studio connectivity
    try:
        import urllib.request

        from src.config import get_settings

        settings = get_settings()
        lm_studio_url = str(settings.lm_studio.base_url)
        models_url = f"{lm_studio_url}/models"

        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    services["vlm"] = "healthy"
                else:
                    services["vlm"] = "degraded"
        except Exception:
            services["vlm"] = "unhealthy"
    except Exception:
        services["vlm"] = "unhealthy"

    return {
        "status": "operational",
        "services": services,
        "metrics": {
            **_get_system_metrics(),
            **_get_processing_metrics(),
        },
        "timestamp": timestamp,
    }
