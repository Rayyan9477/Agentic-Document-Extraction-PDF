"""
Client module for LM Studio VLM communication.

Provides robust client interfaces for communicating with the local
LM Studio server, including connection management, retry logic,
and health monitoring.
"""

from src.client.connection_manager import ConnectionManager, ConnectionState
from src.client.health_monitor import HealthMonitor, HealthStatus, ServerHealth
from src.client.lm_client import (
    LMClientError,
    LMConnectionError,
    LMRateLimitError,
    LMStudioClient,
    LMTimeoutError,
    VisionRequest,
    VisionResponse,
)
from src.client.model_router import (
    ModelConfig,
    ModelRouter,
    ModelTask,
    RoutingDecision,
    florence2_config,
    qwen3vl_config,
)


__all__ = [
    "ConnectionManager",
    "ConnectionState",
    "HealthMonitor",
    "HealthStatus",
    "LMClientError",
    "LMConnectionError",
    "LMRateLimitError",
    "LMStudioClient",
    "LMTimeoutError",
    "ModelConfig",
    "ModelRouter",
    "ModelTask",
    "RoutingDecision",
    "ServerHealth",
    "VisionRequest",
    "VisionResponse",
    "florence2_config",
    "qwen3vl_config",
]
