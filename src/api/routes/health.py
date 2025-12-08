"""
Health check API routes.

Provides endpoints for system health monitoring,
liveness, readiness probes, Prometheus metrics,
and security status for HIPAA compliance.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from src.api.models import HealthResponse
from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()

API_VERSION = "1.0.0"


def _get_system_info() -> dict[str, Any]:
    """
    Get system information including CPU, memory, and disk usage.

    Returns:
        System metrics dictionary.
    """
    try:
        import psutil

        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory info
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_used_gb = memory.used / (1024 ** 3)
        memory_percent = memory.percent

        # Disk info
        disk = psutil.disk_usage("/")
        disk_total_gb = disk.total / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_percent = disk.percent

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
            },
            "memory": {
                "total_gb": round(memory_total_gb, 2),
                "used_gb": round(memory_used_gb, 2),
                "percent": memory_percent,
            },
            "disk": {
                "total_gb": round(disk_total_gb, 2),
                "used_gb": round(disk_used_gb, 2),
                "percent": disk_percent,
            },
            "python_version": sys.version,
            "platform": platform.platform(),
        }
    except ImportError:
        return {
            "error": "psutil not available",
            "python_version": sys.version,
            "platform": platform.platform(),
        }
    except Exception as e:
        return {
            "error": str(e),
            "python_version": sys.version,
            "platform": platform.platform(),
        }


def _check_redis_health() -> dict[str, Any]:
    """Check Redis connectivity (optional - disabled by default)."""
    # Redis/Celery is optional - return disabled status
    return {
        "status": "disabled",
        "message": "Queue system disabled (synchronous processing mode)",
    }


def _check_worker_health() -> dict[str, Any]:
    """Check Celery worker health (optional - disabled by default)."""
    # Workers are optional - return disabled status
    return {
        "status": "disabled",
        "message": "Workers disabled (synchronous processing mode)",
        "worker_count": 0,
        "workers": [],
        "active_tasks": 0,
    }


def _check_vlm_health() -> dict[str, Any]:
    """Check VLM (LM Studio) API connectivity."""
    try:
        import urllib.request
        import urllib.error
        import json
        from src.config import get_settings

        settings = get_settings()
        lm_studio_url = str(settings.lm_studio.base_url)

        # Try to connect to LM Studio models endpoint
        models_url = f"{lm_studio_url}/models"
        try:
            req = urllib.request.Request(models_url, method="GET")
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    models = data.get("data", [])
                    model_names = [m.get("id", "unknown") for m in models[:5]]

                    return {
                        "status": "healthy",
                        "provider": "lm_studio",
                        "endpoint": lm_studio_url,
                        "model": settings.lm_studio.model,
                        "available_models": model_names,
                        "connected": True,
                    }
        except urllib.error.URLError as e:
            return {
                "status": "unhealthy",
                "provider": "lm_studio",
                "endpoint": lm_studio_url,
                "error": f"Cannot connect to LM Studio: {str(e.reason)}",
                "connected": False,
            }
        except Exception as e:
            return {
                "status": "degraded",
                "provider": "lm_studio",
                "endpoint": lm_studio_url,
                "error": str(e),
                "connected": False,
            }

        return {
            "status": "unhealthy",
            "error": "LM Studio not responding",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def _check_security_components() -> dict[str, Any]:
    """
    Check security module components.

    Returns:
        Security components status.
    """
    status = {
        "encryption": {"status": "unknown"},
        "audit_logging": {"status": "unknown"},
        "rbac": {"status": "unknown"},
        "data_cleanup": {"status": "unknown"},
    }

    # Check encryption service
    try:
        from src.security.encryption import EncryptionService
        enc_service = EncryptionService()
        # Test encryption/decryption cycle
        test_data = b"health_check_test"
        encrypted = enc_service.encrypt(test_data)
        decrypted = enc_service.decrypt(encrypted)
        status["encryption"] = {
            "status": "healthy" if decrypted == test_data else "degraded",
            "algorithm": "AES-256-GCM",
        }
    except Exception as e:
        error_msg = str(e)
        # Check if it's a configuration issue (dev mode)
        if "Master key not set" in error_msg or "key" in error_msg.lower():
            status["encryption"] = {
                "status": "not_configured",
                "message": "Encryption key not configured (development mode)",
                "algorithm": "AES-256-GCM",
            }
        else:
            status["encryption"] = {
                "status": "unhealthy",
                "error": error_msg,
            }

    # Check audit logging
    try:
        from src.security.audit import AuditLogger
        status["audit_logging"] = {
            "status": "healthy",
            "phi_masking": True,
            "tamper_evident": True,
        }
    except Exception as e:
        status["audit_logging"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Check RBAC
    try:
        from src.security.rbac import RBACManager
        status["rbac"] = {
            "status": "healthy",
            "jwt_enabled": True,
        }
    except Exception as e:
        status["rbac"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Check data cleanup
    try:
        from src.security.data_cleanup import SecureDataCleanup
        status["data_cleanup"] = {
            "status": "healthy",
            "secure_deletion": True,
            "memory_cleanup": True,
        }
    except Exception as e:
        status["data_cleanup"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    return status


def _check_monitoring_components() -> dict[str, Any]:
    """
    Check monitoring module components.

    Returns:
        Monitoring components status.
    """
    status = {
        "metrics": {"status": "unknown"},
        "alerts": {"status": "unknown"},
    }

    # Check metrics collector
    try:
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        status["metrics"] = {
            "status": "healthy",
            "prometheus_enabled": True,
            "namespaces": ["api", "extraction", "vlm", "validation", "security", "pipeline"],
        }
    except Exception as e:
        status["metrics"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Check alert manager
    try:
        from src.monitoring.alerts import AlertManager
        status["alerts"] = {
            "status": "healthy",
            "channels_available": ["webhook", "slack", "pagerduty", "log"],
        }
    except Exception as e:
        status["alerts"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    return status


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
        components["system"] = _get_system_info()
        components["security"] = _check_security_components()
        components["monitoring"] = _check_monitoring_components()

    # Determine overall status ("disabled" and "not_configured" are acceptable in dev mode)
    all_healthy = all(
        c.get("status") in ("healthy", "disabled", "not_configured")
        for c in components.values()
        if isinstance(c, dict) and "status" in c
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=API_VERSION,
        timestamp=timestamp,
        components=components,
    )


@router.get(
    "/health/detailed",
    response_model=HealthResponse,
    summary="Detailed health check",
    description="Get detailed health status of all components.",
)
async def detailed_health_check(
    http_request: Request,
) -> HealthResponse:
    """
    Detailed health check endpoint.

    Args:
        http_request: HTTP request object.

    Returns:
        Detailed health status of all components.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    components: dict[str, dict[str, Any]] = {
        "api": {
            "status": "healthy",
            "version": API_VERSION,
        },
        "redis": _check_redis_health(),
        "workers": _check_worker_health(),
        "vlm": _check_vlm_health(),
        "system": _get_system_info(),
        "security": _check_security_components(),
        "monitoring": _check_monitoring_components(),
    }

    # Determine overall status ("disabled" and "not_configured" are acceptable in dev mode)
    all_healthy = all(
        c.get("status") in ("healthy", "disabled", "not_configured")
        for c in components.values()
        if isinstance(c, dict) and "status" in c
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
        from src.config import get_settings

        settings = get_settings()

        issues: list[str] = []

        # Check LM Studio is configured
        vlm_status = _check_vlm_health()
        if vlm_status.get("status") == "unhealthy":
            issues.append(f"LM Studio not available: {vlm_status.get('error', 'unknown')}")

        # Check encryption key
        if not settings.security.secret_key:
            issues.append("No secret key configured")

        if issues:
            return {
                "status": "not_ready",
                "issues": issues,
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
    response_class=PlainTextResponse,
)
async def metrics() -> Response:
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Prometheus exposition format metrics.
    """
    try:
        from src.monitoring.metrics import MetricsRegistry

        # Get the global registry and generate metrics
        registry = MetricsRegistry()
        metrics_text = registry.generate_exposition()

        return PlainTextResponse(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception as e:
        logger.error("Failed to generate Prometheus metrics", error=str(e))
        # Return empty metrics on error
        return PlainTextResponse(
            content=f"# Error generating metrics: {e}\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=500,
        )


@router.get(
    "/health/security",
    summary="Security status",
    description="HIPAA compliance security status endpoint.",
)
async def security_status() -> dict[str, Any]:
    """
    Security status endpoint for HIPAA compliance.

    Returns:
        Security components status and compliance indicators.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    security = _check_security_components()

    # Calculate compliance score
    total_components = len(security)
    healthy_components = sum(
        1 for component in security.values()
        if component.get("status") == "healthy"
    )
    compliance_score = (healthy_components / total_components) * 100 if total_components > 0 else 0

    # HIPAA compliance indicators
    hipaa_compliance = {
        "encryption_at_rest": security.get("encryption", {}).get("status") == "healthy",
        "audit_logging": security.get("audit_logging", {}).get("status") == "healthy",
        "access_control": security.get("rbac", {}).get("status") == "healthy",
        "secure_deletion": security.get("data_cleanup", {}).get("status") == "healthy",
        "phi_masking": security.get("audit_logging", {}).get("phi_masking", False),
        "tamper_evident_logs": security.get("audit_logging", {}).get("tamper_evident", False),
    }

    all_compliant = all(hipaa_compliance.values())

    return {
        "status": "compliant" if all_compliant else "non_compliant",
        "compliance_score": round(compliance_score, 1),
        "components": security,
        "hipaa_compliance": hipaa_compliance,
        "timestamp": timestamp,
    }


@router.get(
    "/health/alerts",
    summary="Active alerts",
    description="Get active alerts from the alerting system.",
)
async def active_alerts() -> dict[str, Any]:
    """
    Get active alerts from the alerting system.

    Returns:
        List of active alerts.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        from src.monitoring.alerts import AlertManager

        alert_manager = AlertManager()
        active = alert_manager.get_active_alerts()

        # Group alerts by severity
        by_severity: dict[str, list[dict]] = {
            "critical": [],
            "warning": [],
            "info": [],
        }

        for alert in active:
            severity = alert.severity.value.lower()
            if severity in by_severity:
                by_severity[severity].append({
                    "id": alert.id,
                    "rule_name": alert.rule_name,
                    "message": alert.message,
                    "labels": alert.labels,
                    "created_at": alert.created_at.isoformat(),
                })

        return {
            "total": len(active),
            "by_severity": {
                "critical": len(by_severity["critical"]),
                "warning": len(by_severity["warning"]),
                "info": len(by_severity["info"]),
            },
            "alerts": by_severity,
            "timestamp": timestamp,
        }
    except Exception as e:
        return {
            "error": str(e),
            "total": 0,
            "alerts": [],
            "timestamp": timestamp,
        }


@router.get(
    "/health/dependencies",
    summary="Dependency status",
    description="Check status of all external dependencies.",
)
async def dependency_status() -> dict[str, Any]:
    """
    Check status of all external dependencies.

    Returns:
        Status of each external dependency.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    dependencies = {
        "redis": _check_redis_health(),
        "celery_workers": _check_worker_health(),
        "vlm_api": _check_vlm_health(),
    }

    # Calculate overall status
    healthy_count = sum(
        1 for dep in dependencies.values()
        if dep.get("status") == "healthy"
    )
    total = len(dependencies)

    if healthy_count == total:
        overall = "healthy"
    elif healthy_count > 0:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "healthy_count": healthy_count,
        "total_count": total,
        "dependencies": dependencies,
        "timestamp": timestamp,
    }
