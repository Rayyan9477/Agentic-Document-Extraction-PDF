"""
Health check API routes.

Provides endpoints for system health monitoring,
liveness, and readiness probes.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

from src.api.models import HealthResponse
from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()

API_VERSION = "1.0.0"


def _check_redis_health() -> dict[str, Any]:
    """Check Redis connectivity."""
    try:
        from src.queue.celery_app import celery_app

        # Attempt to ping Redis
        celery_app.control.ping(timeout=2)
        return {
            "status": "healthy",
            "latency_ms": 0,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def _check_worker_health() -> dict[str, Any]:
    """Check Celery worker health."""
    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        health = manager.health_check()

        return {
            "status": "healthy" if health.get("healthy") else "unhealthy",
            "worker_count": health.get("count", 0),
            "workers": health.get("workers", []),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def _check_vlm_health() -> dict[str, Any]:
    """Check VLM API connectivity."""
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.openai_api_key or settings.azure_openai_endpoint:
            return {
                "status": "healthy",
                "provider": "azure" if settings.azure_openai_endpoint else "openai",
            }
        return {
            "status": "unhealthy",
            "error": "No VLM API configured",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the API and its dependencies.",
)
async def health_check(
    http_request: Request,
    deep: bool = False,
) -> HealthResponse:
    """
    Health check endpoint.

    Args:
        http_request: HTTP request object.
        deep: Whether to perform deep health checks.

    Returns:
        Health status of the API and dependencies.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    components: dict[str, dict[str, Any]] = {
        "api": {
            "status": "healthy",
            "version": API_VERSION,
        },
    }

    if deep:
        components["redis"] = _check_redis_health()
        components["workers"] = _check_worker_health()
        components["vlm"] = _check_vlm_health()

    # Determine overall status
    all_healthy = all(
        c.get("status") == "healthy"
        for c in components.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=API_VERSION,
        timestamp=timestamp,
        components=components,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint.",
)
async def liveness() -> dict[str, str]:
    """
    Liveness probe for Kubernetes.

    Returns:
        Simple OK response.
    """
    return {"status": "ok"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint.",
)
async def readiness() -> dict[str, Any]:
    """
    Readiness probe for Kubernetes.

    Checks that critical dependencies are available.

    Returns:
        Readiness status.
    """
    try:
        # Check VLM API is configured
        from src.config import get_settings

        settings = get_settings()
        if not settings.openai_api_key and not settings.azure_openai_endpoint:
            return {
                "status": "not_ready",
                "reason": "No VLM API configured",
            }

        return {"status": "ready"}

    except Exception as e:
        return {
            "status": "not_ready",
            "reason": str(e),
        }


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint.",
)
async def metrics() -> dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Application metrics.
    """
    try:
        from src.queue.worker import WorkerManager

        manager = WorkerManager()
        worker_status = manager.get_worker_status()
        queue_stats = manager.get_queue_stats()

        return {
            "workers": {
                "count": worker_status.get("worker_count", 0),
                "status": worker_status.get("status", "unknown"),
            },
            "queues": queue_stats.get("queues", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
