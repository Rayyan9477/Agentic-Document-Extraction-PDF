"""
FastAPI application for document extraction service.

Provides REST API with comprehensive middleware,
error handling, and OpenAPI documentation.
"""

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

### Authentication

API key authentication is required for all endpoints.
Include the API key in the `X-API-Key` header.

### Rate Limiting

- Sync processing: 10 requests per minute
- Async processing: 100 requests per minute
- Batch processing: 10 requests per minute

### Error Handling

All errors return a standard error response with:
- `error`: Error type
- `message`: Human-readable message
- `details`: Additional error details
- `request_id`: Request tracking ID
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("api_startup", version=API_VERSION)
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

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

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
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

    # Register routes
    from src.api.routes.documents import router as documents_router
    from src.api.routes.tasks import router as tasks_router
    from src.api.routes.health import router as health_router
    from src.api.routes.schemas import router as schemas_router

    app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
    app.include_router(tasks_router, prefix="/api/v1", tags=["Tasks"])
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(schemas_router, prefix="/api/v1", tags=["Schemas"])

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root endpoint redirect to docs."""
        return {
            "message": "PDF Document Extraction API",
            "version": API_VERSION,
            "docs": "/docs",
        }

    return app


# Create default app instance
app = create_app()
