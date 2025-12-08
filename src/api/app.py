"""
FastAPI application for document extraction service.

Provides REST API with comprehensive middleware,
security features, monitoring, and OpenAPI documentation.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_logger, get_settings


logger = get_logger(__name__)


API_VERSION = "1.0.0"
API_TITLE = "PDF Document Extraction API"
API_DESCRIPTION = """
## Document Extraction API

Enterprise-grade API for extracting structured data from PDF documents.

### Features

- **Sync Processing**: Process documents synchronously for immediate results
- **Async Processing**: Queue documents for background processing
- **Batch Processing**: Process multiple documents in a single request
- **Multiple Export Formats**: JSON, Excel, or both
- **PHI Masking**: HIPAA-compliant data handling
- **Confidence Scoring**: Per-field confidence with thresholds
- **Validation**: Multi-layer anti-hallucination system

### Security Features

- **AES-256 Encryption**: HIPAA-compliant data encryption at rest
- **JWT Authentication**: Secure token-based authentication
- **RBAC**: Role-based access control with granular permissions
- **Audit Logging**: Tamper-evident audit logs for compliance
- **Rate Limiting**: Per-client and per-endpoint rate limits

### Monitoring

- **Prometheus Metrics**: Comprehensive application metrics
- **Health Checks**: Liveness, readiness, and deep health probes
- **Alerting**: Multi-channel alert notifications

### Authentication

API key or JWT authentication is required for all endpoints.
- Bearer token: `Authorization: Bearer <token>`
- API key: `X-API-Key: <key>`

### Rate Limiting

- Sync processing: 10 requests per minute
- Async processing: 100 requests per minute
- Batch processing: 5 requests per minute
- Export: 30 requests per minute

### Error Handling

All errors return a standard error response with:
- `error`: Error type
- `message`: Human-readable message
- `details`: Additional error details
- `request_id`: Request tracking ID
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Initializes and cleans up resources on startup/shutdown.
    """
    logger.info("api_startup", version=API_VERSION)

    # Initialize monitoring
    try:
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        app.state.metrics_collector = collector
        logger.info("metrics_initialized")
    except Exception as e:
        logger.warning("metrics_init_failed", error=str(e))

    # Initialize alert manager
    try:
        from src.monitoring.alerts import AlertManager, get_default_alert_rules
        alert_manager = AlertManager()
        for rule in get_default_alert_rules():
            alert_manager.add_rule(rule)
        app.state.alert_manager = alert_manager
        logger.info("alerts_initialized")
    except Exception as e:
        logger.warning("alerts_init_failed", error=str(e))

    yield

    # Cleanup on shutdown
    logger.info("api_shutdown")

    # Cleanup temporary files if needed
    try:
        from src.security.data_cleanup import TempFileManager
        temp_manager = TempFileManager()
        await temp_manager.cleanup_async()
        logger.info("temp_files_cleaned")
    except Exception as e:
        logger.warning("temp_cleanup_failed", error=str(e))


def create_app(
    enable_security: bool = True,
    enable_metrics: bool = True,
    enable_audit: bool = True,
    enable_rate_limiting: bool = True,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        enable_security: Enable security middleware (headers, auth).
        enable_metrics: Enable Prometheus metrics collection.
        enable_audit: Enable HIPAA-compliant audit logging.
        enable_rate_limiting: Enable request rate limiting.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS - require explicit origins for security
    cors_origins = getattr(settings, "cors_origins", None)
    if not cors_origins:
        cors_origins = ["http://localhost:3000", "http://localhost:8000"]
        logger.warning("cors_origins_not_configured", using_defaults=cors_origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    # Add security headers middleware
    if enable_security:
        try:
            from src.api.middleware import SecurityHeadersMiddleware
            app.add_middleware(SecurityHeadersMiddleware)
            logger.info("security_headers_middleware_enabled")
        except ImportError as e:
            logger.warning("security_headers_middleware_unavailable", error=str(e))

    # Add metrics middleware
    if enable_metrics:
        try:
            from src.api.middleware import MetricsMiddleware
            app.add_middleware(MetricsMiddleware)
            logger.info("metrics_middleware_enabled")
        except ImportError as e:
            logger.warning("metrics_middleware_unavailable", error=str(e))

    # Add audit middleware
    if enable_audit:
        try:
            from src.api.middleware import AuditMiddleware
            app.add_middleware(
                AuditMiddleware,
                log_dir="./logs/audit",
                mask_phi=True,
            )
            logger.info("audit_middleware_enabled")
        except ImportError as e:
            logger.warning("audit_middleware_unavailable", error=str(e))

    # Add rate limiting middleware
    if enable_rate_limiting:
        try:
            from src.api.middleware import RateLimitMiddleware
            app.add_middleware(
                RateLimitMiddleware,
                default_rpm=60,
                burst_size=10,
            )
            logger.info("rate_limit_middleware_enabled")
        except ImportError as e:
            logger.warning("rate_limit_middleware_unavailable", error=str(e))

    # Add authentication middleware (optional based on config)
    auth_enabled = getattr(settings, "auth_enabled", False)
    if enable_security and auth_enabled:
        try:
            from src.api.middleware import AuthenticationMiddleware
            from src.security.rbac import RBACManager

            rbac_manager = RBACManager(
                secret_key=settings.secret_key,
            )
            app.add_middleware(
                AuthenticationMiddleware,
                rbac_manager=rbac_manager,
            )
            app.state.rbac_manager = rbac_manager
            logger.info("authentication_middleware_enabled")
        except ImportError as e:
            logger.warning("authentication_middleware_unavailable", error=str(e))

    # Add custom request tracking middleware
    @app.middleware("http")
    async def request_middleware(request: Request, call_next: Any) -> Response:
        """Add request ID and timing to all requests."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Store request ID in state for access in routes
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "request_error",
                request_id=request_id,
                path=request.url.path,
                error=str(exc),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Add response headers
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response

    # Register exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        request: Request,
        exc: FileNotFoundError,
    ) -> JSONResponse:
        """Handle file not found errors."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=404,
            content={
                "error": "file_not_found",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request,
        exc: PermissionError,
    ) -> JSONResponse:
        """Handle permission errors."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=403,
            content={
                "error": "permission_denied",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Register routes
    from src.api.routes.documents import router as documents_router
    from src.api.routes.tasks import router as tasks_router
    from src.api.routes.health import router as health_router
    from src.api.routes.schemas import router as schemas_router
    from src.api.routes.dashboard import router as dashboard_router
    from src.api.routes.queue import router as queue_router

    app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
    app.include_router(tasks_router, prefix="/api/v1", tags=["Tasks"])
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(schemas_router, prefix="/api/v1", tags=["Schemas"])
    app.include_router(dashboard_router, prefix="/api/v1", tags=["Dashboard"])
    app.include_router(queue_router, prefix="/api/v1", tags=["Queue"])

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root endpoint redirect to docs."""
        return {
            "message": "PDF Document Extraction API",
            "version": API_VERSION,
            "docs": "/docs",
        }

    # Favicon endpoint (prevents 404 errors)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """Return empty favicon to prevent 404 errors."""
        # Return a minimal 1x1 transparent PNG
        import base64
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        return Response(content=png_data, media_type="image/png")

    return app


# Create default app instance
app = create_app()
