"""
Tests for the health check API routes in src/api/routes/health.py.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.health import (
    API_VERSION,
    _check_redis_health,
    _check_worker_health,
    _get_system_info,
    router,
)


app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestCheckRedisHealth:
    """Tests for _check_redis_health helper."""

    def test_returns_disabled_status(self):
        result = _check_redis_health()
        assert result["status"] == "disabled"
        assert "synchronous processing mode" in result["message"].lower()


class TestCheckWorkerHealth:
    """Tests for _check_worker_health helper."""

    def test_returns_disabled_status_with_expected_fields(self):
        result = _check_worker_health()
        assert result["status"] == "disabled"
        assert "synchronous processing mode" in result["message"].lower()
        assert result["worker_count"] == 0
        assert result["workers"] == []
        assert result["active_tasks"] == 0


class TestGetSystemInfo:
    """Tests for _get_system_info helper."""

    def test_returns_dict_with_python_version(self):
        result = _get_system_info()
        assert isinstance(result, dict)
        assert "python_version" in result


# ---------------------------------------------------------------------------
# Route tests via TestClient
# ---------------------------------------------------------------------------


class TestHealthLive:
    """Tests for GET /health/live."""

    def test_live_returns_ok(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_healthy_status(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_returns_correct_version(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == API_VERSION

    # WS-1: the public /health endpoint no longer accepts a `deep` flag — it
    # returns minimal info so unauthenticated callers cannot enumerate Redis,
    # workers, VLM, or HIPAA security state. Detailed information now lives on
    # /health/detailed (admin-only).


class TestHealthDetailed:
    """Tests for GET /health/detailed (admin-protected after WS-1)."""

    def test_detailed_requires_admin(self):
        # Without auth, the endpoint must reject with 401 (no token) or 403
        # (token without system:metrics). The TestClient sends no headers, so
        # the AuthorizationMiddleware's missing-auth path applies.
        response = client.get("/health/detailed")
        assert response.status_code in (401, 403)


class TestHealthReady:
    """Tests for GET /health/ready."""

    def test_ready_endpoint_returns_valid_response(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "not_ready")


class TestHealthSecurity:
    """Tests for GET /health/security (admin-protected after WS-1)."""

    def test_security_endpoint_requires_admin(self):
        response = client.get("/health/security")
        assert response.status_code in (401, 403)


class TestHealthDependencies:
    """Tests for GET /health/dependencies (admin-protected after WS-1)."""

    def test_dependencies_endpoint_requires_admin(self):
        response = client.get("/health/dependencies")
        assert response.status_code in (401, 403)


class TestMetrics:
    """Tests for GET /metrics."""

    def test_metrics_returns_plain_text(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        # Prometheus metrics are served as text/plain (possibly with charset)
        assert "text/plain" in content_type or "text/" in content_type
