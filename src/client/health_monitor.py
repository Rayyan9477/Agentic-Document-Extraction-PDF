"""
Health monitoring for LM Studio server.

Provides continuous health monitoring, status reporting,
and alerting capabilities for the VLM backend.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import httpx

from src.config import get_logger, get_settings


logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ServerHealth:
    """
    LM Studio server health information.

    Attributes:
        status: Current health status.
        is_reachable: Whether server is reachable.
        response_time_ms: Last response time in milliseconds.
        model_loaded: Whether a model is loaded.
        model_name: Name of loaded model (if any).
        available_models: List of available models.
        gpu_memory_used_mb: GPU memory used (if available).
        gpu_memory_total_mb: Total GPU memory (if available).
        last_check_time: Time of last health check.
        consecutive_failures: Current consecutive failure count.
        uptime_seconds: Server uptime (if available).
        error_message: Error message if unhealthy.
    """

    status: HealthStatus = HealthStatus.UNKNOWN
    is_reachable: bool = False
    response_time_ms: int = 0
    model_loaded: bool = False
    model_name: str | None = None
    available_models: list[str] = field(default_factory=list)
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    last_check_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0
    uptime_seconds: float | None = None
    error_message: str | None = None

    @property
    def gpu_memory_percent(self) -> float | None:
        """Get GPU memory usage percentage."""
        if self.gpu_memory_total_mb and self.gpu_memory_used_mb:
            return (self.gpu_memory_used_mb / self.gpu_memory_total_mb) * 100
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "is_reachable": self.is_reachable,
            "response_time_ms": self.response_time_ms,
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "available_models": self.available_models,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "last_check_time": self.last_check_time.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "uptime_seconds": self.uptime_seconds,
            "error_message": self.error_message,
        }


class HealthMonitor:
    """
    Continuous health monitoring for LM Studio server.

    Provides background health checking, status reporting,
    and optional alert callbacks.

    Example:
        monitor = HealthMonitor()

        # Get current health
        health = monitor.check_health()
        if health.status == HealthStatus.HEALTHY:
            print(f"Server OK, model: {health.model_name}")

        # Start background monitoring
        monitor.start_background_monitoring(interval=30)

        # Register alert callback
        monitor.on_status_change(lambda old, new: print(f"Status: {old} -> {new}"))

        # Stop monitoring
        monitor.stop_background_monitoring()
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        """
        Initialize health monitor.

        Args:
            base_url: LM Studio server URL. Defaults to settings.
            timeout: Health check timeout in seconds.
            failure_threshold: Failures before marking unhealthy.
        """
        settings = get_settings()
        self._base_url = base_url or str(settings.lm_studio.base_url).rstrip("/v1")
        self._timeout = timeout
        self._failure_threshold = failure_threshold

        self._http_client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
        )

        # Async client created lazily to avoid event loop issues
        self._async_http_client: httpx.AsyncClient | None = None
        self._async_client_lock = asyncio.Lock()

        self._current_health = ServerHealth()
        self._consecutive_failures = 0
        self._lock = threading.Lock()

        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self._status_callbacks: list[Callable[[HealthStatus, HealthStatus], None]] = []
        self._health_callbacks: list[Callable[[ServerHealth], None]] = []

        logger.debug(
            "health_monitor_initialized",
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def current_health(self) -> ServerHealth:
        """Get current health status."""
        with self._lock:
            return self._current_health

    @property
    def is_healthy(self) -> bool:
        """Check if server is currently healthy."""
        return self.current_health.status == HealthStatus.HEALTHY

    def check_health(self) -> ServerHealth:
        """
        Perform a health check against the LM Studio server.

        Returns:
            ServerHealth with current status information.
        """
        start_time = time.perf_counter()
        health = ServerHealth(last_check_time=datetime.now(timezone.utc))

        try:
            # Check models endpoint
            response = self._http_client.get("/v1/models")
            response_time_ms = int((time.perf_counter() - start_time) * 1000)

            if response.status_code == 200:
                health.is_reachable = True
                health.response_time_ms = response_time_ms

                # Parse models response
                data = response.json()
                models = data.get("data", [])
                health.available_models = [m.get("id", "") for m in models]
                health.model_loaded = len(models) > 0

                if models:
                    health.model_name = models[0].get("id")

                # Determine status based on response time
                if response_time_ms < 1000:
                    health.status = HealthStatus.HEALTHY
                elif response_time_ms < 5000:
                    health.status = HealthStatus.DEGRADED
                else:
                    health.status = HealthStatus.DEGRADED
                    health.error_message = f"High latency: {response_time_ms}ms"

                self._consecutive_failures = 0

            else:
                health.is_reachable = True
                health.response_time_ms = response_time_ms
                health.status = HealthStatus.DEGRADED
                health.error_message = f"Unexpected status code: {response.status_code}"
                self._consecutive_failures += 1

        except httpx.ConnectError as e:
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Connection failed: {e}"
            self._consecutive_failures += 1

        except httpx.TimeoutException as e:
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Timeout: {e}"
            self._consecutive_failures += 1

        except Exception as e:
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Health check failed: {e}"
            self._consecutive_failures += 1

        health.consecutive_failures = self._consecutive_failures

        # Check if we've exceeded failure threshold
        if self._consecutive_failures >= self._failure_threshold:
            health.status = HealthStatus.UNHEALTHY

        # Update current health and trigger callbacks
        self._update_health(health)

        logger.debug(
            "health_check_complete",
            status=health.status.value,
            reachable=health.is_reachable,
            response_time_ms=health.response_time_ms,
            model_loaded=health.model_loaded,
        )

        return health

    def _update_health(self, new_health: ServerHealth) -> None:
        """Update current health and trigger callbacks."""
        with self._lock:
            old_status = self._current_health.status
            self._current_health = new_health

        # Trigger status change callbacks
        if old_status != new_health.status:
            for callback in self._status_callbacks:
                try:
                    callback(old_status, new_health.status)
                except Exception as e:
                    logger.error("status_callback_failed", error=str(e))

        # Trigger health callbacks
        for callback in self._health_callbacks:
            try:
                callback(new_health)
            except Exception as e:
                logger.error("health_callback_failed", error=str(e))

    def on_status_change(
        self,
        callback: Callable[[HealthStatus, HealthStatus], None],
    ) -> None:
        """
        Register callback for status changes.

        Args:
            callback: Function(old_status, new_status) to call on changes.
        """
        self._status_callbacks.append(callback)

    def on_health_check(
        self,
        callback: Callable[[ServerHealth], None],
    ) -> None:
        """
        Register callback for every health check.

        Args:
            callback: Function(health) to call after each check.
        """
        self._health_callbacks.append(callback)

    def start_background_monitoring(
        self,
        interval: float = 30.0,
    ) -> None:
        """
        Start background health monitoring.

        Args:
            interval: Seconds between health checks.
        """
        if self._monitoring_active:
            logger.warning("background_monitoring_already_active")
            return

        self._monitoring_active = True
        self._stop_event.clear()

        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.check_health()
                except Exception as e:
                    logger.error("background_health_check_failed", error=str(e))

                self._stop_event.wait(interval)

            logger.info("background_monitoring_stopped")

        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="HealthMonitor",
        )
        self._monitor_thread.start()

        logger.info("background_monitoring_started", interval=interval)

    def stop_background_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_http_client is None:
            async with self._async_client_lock:
                if self._async_http_client is None:
                    self._async_http_client = httpx.AsyncClient(
                        base_url=self._base_url,
                        timeout=self._timeout,
                    )
        return self._async_http_client

    async def check_health_async(self) -> ServerHealth:
        """
        Perform async health check using native async I/O.

        Uses httpx.AsyncClient for true non-blocking I/O instead of
        wrapping sync code in executor.

        Returns:
            ServerHealth with current status.
        """
        start_time = time.perf_counter()
        health = ServerHealth(last_check_time=datetime.now(timezone.utc))

        try:
            client = await self._get_async_client()
            response = await client.get("/v1/models")
            response_time_ms = int((time.perf_counter() - start_time) * 1000)

            if response.status_code == 200:
                health.is_reachable = True
                health.response_time_ms = response_time_ms

                # Parse models response
                data = response.json()
                models = data.get("data", [])
                health.available_models = [m.get("id", "") for m in models]
                health.model_loaded = len(models) > 0

                if models:
                    health.model_name = models[0].get("id")

                # Determine status based on response time
                if response_time_ms < 1000:
                    health.status = HealthStatus.HEALTHY
                elif response_time_ms < 5000:
                    health.status = HealthStatus.DEGRADED
                else:
                    health.status = HealthStatus.DEGRADED
                    health.error_message = f"High latency: {response_time_ms}ms"

                self._consecutive_failures = 0

            else:
                health.is_reachable = True
                health.response_time_ms = response_time_ms
                health.status = HealthStatus.DEGRADED
                health.error_message = f"Unexpected status code: {response.status_code}"
                self._consecutive_failures += 1

        except httpx.ConnectError as e:
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Connection failed: {e}"
            self._consecutive_failures += 1

        except httpx.TimeoutException as e:
            health.is_reachable = False
            health.status = HealthStatus.UNHEALTHY
            health.error_message = f"Timeout: {e}"
            self._consecutive_failures += 1

        except Exception as e:
            health.status = HealthStatus.UNKNOWN
            health.error_message = f"Unexpected error: {e}"
            self._consecutive_failures += 1

        # Update state
        with self._lock:
            old_status = self._current_health.status
            self._current_health = health

        # Fire callbacks
        if old_status != health.status:
            for callback in self._status_callbacks:
                try:
                    callback(old_status, health.status)
                except Exception as e:
                    logger.warning("status_callback_error", error=str(e))

        for callback in self._health_callbacks:
            try:
                callback(health)
            except Exception as e:
                logger.warning("health_callback_error", error=str(e))

        return health

    async def close_async(self) -> None:
        """Close async HTTP client."""
        if self._async_http_client is not None:
            await self._async_http_client.aclose()
            self._async_http_client = None

    async def monitor_continuous(
        self,
        interval: float = 30.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """
        Async continuous health monitoring with cancellation support.

        Can be stopped by:
        1. Passing a stop_event and setting it
        2. Cancelling the task (raises asyncio.CancelledError)

        Args:
            interval: Seconds between checks.
            stop_event: Optional event to signal monitoring should stop.

        Example:
            stop = asyncio.Event()
            task = asyncio.create_task(monitor.monitor_continuous(stop_event=stop))
            # Later...
            stop.set()  # Graceful stop
            await task
        """
        try:
            while True:
                # Check for stop signal
                if stop_event is not None and stop_event.is_set():
                    logger.info("monitor_continuous_stopped", reason="stop_event")
                    break

                await self.check_health_async()

                # Use wait_for with timeout instead of sleep for responsiveness
                if stop_event is not None:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=interval)
                        # If we get here, stop_event was set
                        logger.info("monitor_continuous_stopped", reason="stop_event")
                        break
                    except asyncio.TimeoutError:
                        # Normal timeout, continue monitoring
                        pass
                else:
                    await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("monitor_continuous_stopped", reason="cancelled")
            raise  # Re-raise to allow proper task cleanup

    def wait_for_healthy(
        self,
        timeout: float = 300.0,
        check_interval: float = 5.0,
    ) -> bool:
        """
        Wait for server to become healthy.

        Args:
            timeout: Maximum wait time in seconds.
            check_interval: Seconds between checks.

        Returns:
            True if server became healthy within timeout.
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            health = self.check_health()
            if health.status == HealthStatus.HEALTHY:
                return True
            time.sleep(check_interval)

        return False

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get comprehensive health summary.

        Returns:
            Dictionary with health information.
        """
        health = self.current_health

        return {
            "status": health.status.value,
            "is_healthy": self.is_healthy,
            "server_reachable": health.is_reachable,
            "model_loaded": health.model_loaded,
            "model_name": health.model_name,
            "response_time_ms": health.response_time_ms,
            "available_models": health.available_models,
            "consecutive_failures": health.consecutive_failures,
            "last_check": health.last_check_time.isoformat(),
            "error": health.error_message,
            "monitoring_active": self._monitoring_active,
        }

    def close(self) -> None:
        """Stop monitoring and close connections."""
        self.stop_background_monitoring()
        self._http_client.close()

    def __enter__(self) -> "HealthMonitor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
