"""
Connection management for LM Studio client.

Provides connection pooling, automatic reconnection, and
circuit breaker pattern for resilient VLM communication.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from src.config import get_logger


logger = get_logger(__name__)

T = TypeVar("T")


class ConnectionState(str, Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class ConnectionMetrics:
    """
    Connection health metrics.

    Attributes:
        total_requests: Total requests made.
        successful_requests: Successful requests.
        failed_requests: Failed requests.
        total_latency_ms: Cumulative latency.
        last_success_time: Time of last successful request.
        last_failure_time: Time of last failed request.
        consecutive_failures: Current consecutive failure count.
        circuit_trips: Number of circuit breaker trips.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: int = 0
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    consecutive_failures: int = 0
    circuit_trips: int = 0

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    def record_success(self, latency_ms: int) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = datetime.now(UTC)
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.now(UTC)
        self.consecutive_failures += 1

    def record_circuit_trip(self) -> None:
        """Record a circuit breaker trip."""
        self.circuit_trips += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "circuit_trips": self.circuit_trips,
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
        }


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern to prevent repeated
    requests to a failing service.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Requests fail immediately, service considered down
    - HALF_OPEN: Testing if service has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        with breaker:
            # Make request
            result = client.send_request()

        # Or check state
        if breaker.can_execute():
            result = client.send_request()
            breaker.record_success()
        else:
            raise CircuitOpenError()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
            recovery_timeout: Seconds before attempting recovery.
            half_open_max_calls: Max calls in half-open state.
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()

        logger.debug(
            "circuit_breaker_initialized",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def can_execute(self) -> bool:
        """
        Check if request can be executed.

        Returns:
            True if circuit allows execution.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            # OPEN
            return False

    def record_success(self) -> None:
        """Record successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Recovery successful, close circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("circuit_breaker_closed", reason="successful_recovery")

    def record_failure(self) -> None:
        """Record failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed, reopen circuit
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning("circuit_breaker_reopened", reason="recovery_failed")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "circuit_breaker_opened",
                        failure_count=self._failure_count,
                        threshold=self._failure_threshold,
                    )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info("circuit_breaker_reset")

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "circuit_breaker_half_open",
                    elapsed_seconds=elapsed,
                )

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry - check if execution allowed."""
        if not self.can_execute():
            raise CircuitOpenError("Circuit breaker is open")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - record success or failure."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""


class ConnectionManager:
    """
    Manages connections to LM Studio with resilience patterns.

    Provides:
    - Connection state tracking
    - Automatic reconnection
    - Circuit breaker protection
    - Health metrics collection

    Example:
        manager = ConnectionManager()

        # Execute with protection
        result = manager.execute(lambda: client.send_request())

        # Check status
        if manager.is_healthy:
            ...

        # Get metrics
        metrics = manager.get_metrics()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """
        Initialize connection manager.

        Args:
            failure_threshold: Failures before circuit opens.
            recovery_timeout: Seconds before recovery attempt.
            health_check_interval: Seconds between health checks.
        """
        self._state = ConnectionState.DISCONNECTED
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self._metrics = ConnectionMetrics()
        self._health_check_interval = health_check_interval
        self._lock = threading.Lock()

        self._health_check_func: Callable[[], bool] | None = None
        self._last_health_check: float = 0

        logger.debug(
            "connection_manager_initialized",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        with self._lock:
            if self._circuit_breaker.state == CircuitState.OPEN:
                return ConnectionState.CIRCUIT_OPEN
            return self._state

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.state in (
            ConnectionState.CONNECTED,
            ConnectionState.RECONNECTING,
        )

    def set_health_check(self, func: Callable[[], bool]) -> None:
        """
        Set health check function.

        Args:
            func: Function that returns True if healthy.
        """
        self._health_check_func = func

    def connect(self) -> bool:
        """
        Attempt to establish connection.

        Returns:
            True if connection successful.
        """
        with self._lock:
            self._state = ConnectionState.CONNECTING

        try:
            if self._health_check_func and self._health_check_func():
                with self._lock:
                    self._state = ConnectionState.CONNECTED
                logger.info("connection_established")
                return True
            with self._lock:
                self._state = ConnectionState.FAILED
            return False
        except Exception as e:
            with self._lock:
                self._state = ConnectionState.FAILED
            logger.error("connection_failed", error=str(e))
            return False

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute.

        Returns:
            Function result.

        Raises:
            CircuitOpenError: If circuit is open.
            Exception: If function fails.
        """
        if not self._circuit_breaker.can_execute():
            self._metrics.record_circuit_trip()
            raise CircuitOpenError("Circuit breaker is open, request blocked")

        start_time = time.perf_counter()

        try:
            result = func()
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._circuit_breaker.record_success()
            self._metrics.record_success(latency_ms)

            with self._lock:
                self._state = ConnectionState.CONNECTED

            return result

        except Exception:
            self._circuit_breaker.record_failure()
            self._metrics.record_failure()

            with self._lock:
                if self._metrics.consecutive_failures > 1:
                    self._state = ConnectionState.RECONNECTING

            raise

    async def execute_async(
        self,
        func: Callable[[], T] | Callable[[], Awaitable[T]],
    ) -> T:
        """
        Execute function with circuit breaker protection (async version).

        Supports both true async callables (coroutines) and sync callables.
        For async callables, executes directly with await.
        For sync callables, runs in executor to avoid blocking.

        Args:
            func: Async or sync function to execute.

        Returns:
            Function result.
        """
        # Check if function is async (returns coroutine)
        result = func()

        if asyncio.iscoroutine(result):
            # True async - await directly for proper async I/O
            try:
                return await result
            except Exception:
                with self._lock:
                    self._metrics.total_requests += 1
                    self._record_failure()
                raise
        else:
            # Sync function already called - return result
            # Note: For sync functions requiring circuit breaker protection,
            # use execute() or wrap in run_in_executor externally
            return result

    async def execute_sync_in_executor(self, func: Callable[[], T]) -> T:
        """
        Execute synchronous function in executor with circuit breaker.

        Use this for CPU-bound or blocking I/O operations that need
        to run without blocking the event loop.

        Args:
            func: Synchronous function to execute.

        Returns:
            Function result.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, func)

    def check_health(self) -> bool:
        """
        Perform health check if interval has passed.

        Returns:
            True if healthy or check not needed yet.
        """
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return self.is_healthy

        self._last_health_check = current_time

        if self._health_check_func:
            try:
                healthy = self._health_check_func()
                if healthy:
                    with self._lock:
                        if self._state != ConnectionState.CONNECTED:
                            self._state = ConnectionState.CONNECTED
                            logger.info("connection_recovered")
                    return True
                with self._lock:
                    self._state = ConnectionState.FAILED
                return False
            except Exception as e:
                with self._lock:
                    self._state = ConnectionState.FAILED
                logger.warning("health_check_failed", error=str(e))
                return False

        return self.is_healthy

    def get_metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset connection manager state."""
        with self._lock:
            self._state = ConnectionState.DISCONNECTED
            self._circuit_breaker.reset()
            self._metrics = ConnectionMetrics()
        logger.info("connection_manager_reset")

    def to_dict(self) -> dict[str, Any]:
        """Get full status as dictionary."""
        return {
            "state": self.state.value,
            "circuit_state": self._circuit_breaker.state.value,
            "is_healthy": self.is_healthy,
            "metrics": self._metrics.to_dict(),
        }
