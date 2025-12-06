"""
Client module for LM Studio VLM communication.

Provides robust client interfaces for communicating with the local
LM Studio server, including connection management, retry logic,
and health monitoring.
"""

from src.client.connection_manager import ConnectionManager, ConnectionState
from src.client.health_monitor import HealthMonitor, HealthStatus, ServerHealth
from src.client.lm_client import (
    LMStudioClient,
    VisionRequest,
    VisionResponse,
    LMClientError,
    LMConnectionError,
    LMTimeoutError,
    LMRateLimitError,
)


__all__ = [
    "LMStudioClient",
    "VisionRequest",
    "VisionResponse",
    "LMClientError",
    "LMConnectionError",
    "LMTimeoutError",
    "LMRateLimitError",
    "ConnectionManager",
    "ConnectionState",
    "HealthMonitor",
    "HealthStatus",
    "ServerHealth",
]
