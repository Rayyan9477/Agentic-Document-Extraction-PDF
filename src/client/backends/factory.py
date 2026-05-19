"""
Backend factory.

Reads ``settings.vlm.backend`` and constructs the appropriate
``VLMBackend`` instance. Caches the result so repeated ``get_backend()``
calls return the same instance per-process — matters because each
backend owns lazy ``LMStudioClient`` connection pools.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from src.client.backends.gemma_backend import GemmaBackend
from src.client.backends.lm_studio_backend import LMStudioBackend
from src.client.backends.protocol import VLMBackend
from src.client.backends.vllm_backend import VLLMBackend
from src.config import get_logger, get_settings


if TYPE_CHECKING:
    from src.config.settings import Settings


logger = get_logger(__name__)


# Process-wide cache. Keyed by backend name so a settings hot-reload
# (e.g. tests overriding ``VLM_BACKEND``) yields a fresh instance.
_cache_lock = threading.Lock()
_cache: dict[str, VLMBackend] = {}


def get_backend(settings: "Settings | None" = None) -> VLMBackend:
    """Return the configured backend, creating it on first access.

    Args:
        settings: Optional settings override. Defaults to the global
            ``get_settings()``. Tests can pass a custom Settings to
            switch backends without setting environment variables.

    Returns:
        A ``VLMBackend`` instance. Either ``LMStudioBackend`` or
        ``VLLMBackend`` depending on ``settings.vlm.backend``.
    """
    settings = settings or get_settings()
    backend_name = settings.vlm.backend

    with _cache_lock:
        cached = _cache.get(backend_name)
        if cached is not None:
            return cached

        if backend_name == "lm_studio":
            backend: VLMBackend = _build_lm_studio_backend(settings)
        elif backend_name == "vllm":
            backend = _build_vllm_backend(settings)
        elif backend_name == "gemma":
            backend = _build_gemma_backend(settings)
        else:
            raise ValueError(
                f"Unsupported VLM_BACKEND: {backend_name!r}. "
                f"Expected 'lm_studio', 'vllm', or 'gemma'."
            )

        logger.info(
            "vlm_backend_initialised",
            backend=backend_name,
            capabilities=backend.capabilities().to_dict(),
        )
        _cache[backend_name] = backend
        return backend


def reset_cache() -> None:
    """Drop the cached backend(s).

    Used by tests that mutate settings between cases. Closes each cached
    backend before forgetting it.
    """
    with _cache_lock:
        for backend in _cache.values():
            try:
                backend.close()
            except Exception:  # pragma: no cover - defensive
                pass
        _cache.clear()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_lm_studio_backend(settings: "Settings") -> LMStudioBackend:
    cfg = settings.vlm.lm_studio

    # ``settings.lm_studio`` (the legacy section) still drives the
    # default URL and model when the new ``vlm.lm_studio`` block leaves
    # them unset. This preserves zero-config behaviour for existing
    # deployments.
    legacy = settings.lm_studio
    primary_url = cfg.primary_url or str(legacy.base_url)
    primary_model = cfg.primary_model or legacy.model

    return LMStudioBackend(
        primary_url=primary_url,
        primary_model=primary_model,
        secondary_url=cfg.secondary_url,
        secondary_model=cfg.secondary_model,
        dual_mode=cfg.dual_mode,
        max_tokens=legacy.max_tokens,
        temperature=legacy.temperature,
        timeout=legacy.timeout,
        max_retries=legacy.max_retries,
        retry_min_wait=legacy.retry_min_wait,
        retry_max_wait=legacy.retry_max_wait,
    )


def _build_vllm_backend(settings: "Settings") -> VLLMBackend:
    cfg = settings.vlm.vllm
    legacy = settings.lm_studio  # share retry/timeout knobs

    if not cfg.primary_url or not cfg.primary_model:
        raise ValueError(
            "VLM_BACKEND=vllm requires VLLM_PRIMARY_URL and VLLM_PRIMARY_MODEL "
            "to be configured. See docs/MVP/EXTRACTION.md §1."
        )

    return VLLMBackend(
        primary_url=cfg.primary_url,
        primary_model=cfg.primary_model,
        secondary_url=cfg.secondary_url,
        secondary_model=cfg.secondary_model,
        guided_decoding_backend=cfg.guided_decoding_backend,
        max_tokens=legacy.max_tokens,
        temperature=legacy.temperature,
        timeout=legacy.timeout,
        max_retries=legacy.max_retries,
        retry_min_wait=legacy.retry_min_wait,
        retry_max_wait=legacy.retry_max_wait,
    )


def _build_gemma_backend(settings: "Settings") -> GemmaBackend:
    """Phase K — instantiate the Gemma 4 backend.

    Reads from ``settings.vlm.gemma`` (Phase K) — no fallback to the
    legacy LM Studio block because Gemma 4 ships on its own port.
    """
    cfg = settings.vlm.gemma

    if not cfg.primary_url or not cfg.primary_model:
        raise ValueError(
            "VLM_BACKEND=gemma requires GEMMA_PRIMARY_URL and GEMMA_PRIMARY_MODEL "
            "to be configured. Defaults assume LM Studio at "
            "http://localhost:1235/v1 serving gemma-4-26b-a4b-it."
        )

    return GemmaBackend(
        primary_url=cfg.primary_url,
        primary_model=cfg.primary_model,
        tool_call_timeout=cfg.tool_call_timeout,
        max_retries=cfg.max_retries,
        temperature=cfg.temperature,
        register_rcm_tools=cfg.register_rcm_tools,
        fail_open_on_health=cfg.fail_open_on_health,
    )
