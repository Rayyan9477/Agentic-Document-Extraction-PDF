"""V3 Phase 7 — VLM backend queue-depth control.

Production deployments need to bound concurrent VLM calls per
process: a thundering herd of extractions can OOM a single GPU
backend or drive its tail latency to unusable levels. We solve
this with a simple **per-process semaphore** that callers acquire
before issuing a VLM request and release after the response comes
back.

The semaphore is **opt-in**: agents that want queue-depth control
wrap their VLM call in ``with vlm_queue_slot():`` (or
``async with vlm_queue_slot_async():``); agents that don't, get the
legacy unbounded behaviour. Existing call sites are unchanged
unless they explicitly add the wrapper.

Configuration:
    settings.vlm.max_concurrent_requests = N (default 0 = unbounded)

The semaphore is process-wide — multi-worker deployments still
need a backing-service-side gate (vLLM's own
``--max-num-batched-tokens`` etc.) for cluster-level limits. This
just protects a single Python process from runaway concurrency.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


_SEM_LOCK = threading.Lock()
_SEM: threading.Semaphore | None = None
_ASEM: asyncio.Semaphore | None = None
_SEM_CAPACITY: int = 0


def configure(capacity: int) -> None:
    """Configure the process-wide semaphore.

    ``capacity == 0`` disables the gate (unbounded). Calling
    ``configure`` after the semaphores have been created replaces
    them — outstanding ``acquire`` holds remain valid on the *old*
    instance but new acquires use the new capacity. In practice we
    only ever call this once at boot, so this is mostly a test
    affordance.
    """
    global _SEM, _ASEM, _SEM_CAPACITY
    with _SEM_LOCK:
        _SEM_CAPACITY = max(0, int(capacity))
        if _SEM_CAPACITY <= 0:
            _SEM = None
            _ASEM = None
            return
        _SEM = threading.Semaphore(_SEM_CAPACITY)
        _ASEM = asyncio.Semaphore(_SEM_CAPACITY)
        logger.info("vlm_queue_depth_configured", capacity=_SEM_CAPACITY)


def reset() -> None:
    """Tear down semaphores. Tests only."""
    global _SEM, _ASEM, _SEM_CAPACITY
    with _SEM_LOCK:
        _SEM = None
        _ASEM = None
        _SEM_CAPACITY = 0


def is_active() -> bool:
    """Whether queue-depth gating is currently in effect."""
    return _SEM is not None


@contextmanager
def vlm_queue_slot(timeout: float | None = None) -> Any:
    """Acquire a queue slot for a synchronous VLM call.

    No-op when the gate is disabled. When active, blocks (or
    times out) until a slot is available, then releases on exit.
    """
    sem = _SEM
    if sem is None:
        yield None
        return
    acquired = sem.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError(
            f"vlm_queue_slot: failed to acquire slot within {timeout}s"
        )
    try:
        yield sem
    finally:
        sem.release()


@asynccontextmanager
async def vlm_queue_slot_async(timeout: float | None = None) -> Any:
    """Async variant of ``vlm_queue_slot``.

    Backends serving event-loop callers should use this so the
    surrounding loop stays responsive while waiting for a slot.
    """
    asem = _ASEM
    if asem is None:
        yield None
        return
    if timeout is None:
        await asem.acquire()
    else:
        try:
            await asyncio.wait_for(asem.acquire(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"vlm_queue_slot_async: failed to acquire slot within {timeout}s"
            ) from e
    try:
        yield asem
    finally:
        asem.release()


def configure_from_settings() -> None:
    """Pull the capacity off ``settings.vlm.max_concurrent_requests``.

    Safe to call multiple times — only re-applies when the value
    has changed. Returns silently if settings can't be loaded.
    """
    try:
        from src.config import get_settings

        settings = get_settings()
        cap = int(
            getattr(settings.vlm, "max_concurrent_requests", 0) or 0
        )
    except Exception:  # pragma: no cover - defensive
        return
    if cap == _SEM_CAPACITY:
        return
    configure(cap)
