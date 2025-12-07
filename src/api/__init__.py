"""
API module for document extraction service.

Provides FastAPI REST endpoints for:
- Document processing (sync and async)
- Task status tracking
- Health checks and metrics
- Schema management
- Security middleware
"""

from src.api.app import create_app, app
from src.api.middleware import (
    AuditMiddleware,
    AuthenticationMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RateLimiter,
    RateLimitConfig,
    SecurityHeadersMiddleware,
    get_current_user,
    require_permission,
)
from src.api.models import (
    ProcessRequest,
    ProcessResponse,
    BatchProcessRequest,
    BatchProcessResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
)


__all__ = [
    # App
    "create_app",
    "app",
    # Middleware
    "AuditMiddleware",
    "AuthenticationMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RateLimiter",
    "RateLimitConfig",
    "SecurityHeadersMiddleware",
    "get_current_user",
    "require_permission",
    # Models
    "ProcessRequest",
    "ProcessResponse",
    "BatchProcessRequest",
    "BatchProcessResponse",
    "TaskStatusResponse",
    "HealthResponse",
    "ErrorResponse",
]
