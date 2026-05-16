"""
VLM backend abstraction (Phase 0).

Two equally first-class backends: LM Studio and vLLM. Selected at runtime
via ``settings.vlm.backend = "lm_studio" | "vllm"``. Both expose the same
``VLMBackend`` protocol so call-sites in ``BaseAgent.send_vision_request``
remain backend-agnostic.

The legacy ``LMStudioClient`` is preserved verbatim — ``LMStudioBackend``
adapts it. The new ``VLLMBackend`` talks to vLLM's OpenAI-compatible
endpoint with optional XGrammar guided-decoding.

See ``docs/MVP/EXTRACTION.md`` §1 and ``docs/MVP/MODULE_MAP.md``.
"""

from src.client.backends.factory import get_backend
from src.client.backends.lm_studio_backend import LMStudioBackend
from src.client.backends.protocol import (
    BackendCapabilities,
    BackendHealth,
    VLMBackend,
    VLMRole,
)
from src.client.backends.vllm_backend import VLLMBackend


__all__ = [
    "BackendCapabilities",
    "BackendHealth",
    "LMStudioBackend",
    "VLLMBackend",
    "VLMBackend",
    "VLMRole",
    "get_backend",
]
