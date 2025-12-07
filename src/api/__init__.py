"""
API module for document extraction service.

Provides FastAPI REST endpoints for:
- Document processing (sync and async)
- Task status tracking
- Health checks and metrics
- Schema management
"""

from src.api.app import create_app, app
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
    # Models
    "ProcessRequest",
    "ProcessResponse",
    "BatchProcessRequest",
    "BatchProcessResponse",
    "TaskStatusResponse",
    "HealthResponse",
    "ErrorResponse",
]
