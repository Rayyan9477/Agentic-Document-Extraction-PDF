"""
vLLM backend.

Talks to one or two vLLM OpenAI-compatible endpoints (typically
``http://localhost:8001`` for primary and ``8002`` for secondary).
Capability matrix (May 2026):

* ``supports_dual_vlm`` — true when both URLs are configured.
* ``supports_constrained_decoding`` — XGrammar via
  ``extra_body={"guided_json": ..., "guided_decoding_backend": "xgrammar"}``.
  Outlines is also selectable.
* ``supports_logprobs`` — yes, even on grammar-permitted tokens.
* ``supports_multi_image`` — yes (modern Qwen-VL / Gemma-VL handle it).
* ``supports_tensor_parallelism`` — yes (server-side ``--tensor-parallel-size``).

Implementation notes:

* Re-uses the ``LMStudioClient`` underneath because LM Studio and vLLM
  both expose the OpenAI ``v1/chat/completions`` surface and the
  legacy client already covers retry, JSON extraction, and async
  pooling. The vLLM-specific bit is the ``extra_body`` injection for
  constrained decoding (Phase 1 wires this; Phase 0 ships the plumbing).
* The ``schema`` parameter is accepted but not yet used here — Phase 1's
  ``constrained_decode()`` wrapper layers it on top of this backend.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.client.backends.protocol import (
    BackendCapabilities,
    BackendHealth,
    VLMBackend,
    VLMRole,
)
from src.client.lm_client import LMStudioClient
from src.config import get_logger


if TYPE_CHECKING:
    from src.client.lm_client import VisionRequest, VisionResponse


logger = get_logger(__name__)


_VALID_GUIDED_BACKENDS = ("xgrammar", "outlines")


class VLLMBackend:
    """vLLM OpenAI-compat backend."""

    name = "vllm"

    def __init__(
        self,
        primary_url: str,
        primary_model: str,
        *,
        secondary_url: str | None = None,
        secondary_model: str | None = None,
        guided_decoding_backend: str = "xgrammar",
        # Plumbing for the underlying LMStudioClient (re-used for OpenAI compat)
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_min_wait: int | None = None,
        retry_max_wait: int | None = None,
    ) -> None:
        if guided_decoding_backend not in _VALID_GUIDED_BACKENDS:
            raise ValueError(
                f"guided_decoding_backend must be one of {_VALID_GUIDED_BACKENDS}, "
                f"got {guided_decoding_backend!r}"
            )

        self._primary_url = primary_url
        self._primary_model = primary_model
        self._secondary_url = secondary_url
        self._secondary_model = secondary_model
        self._guided_decoding_backend = guided_decoding_backend

        self._client_kwargs: dict[str, Any] = {
            k: v
            for k, v in {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "max_retries": max_retries,
                "retry_min_wait": retry_min_wait,
                "retry_max_wait": retry_max_wait,
            }.items()
            if v is not None
        }

        self._clients: dict[VLMRole, LMStudioClient] = {}

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def capabilities(self) -> BackendCapabilities:
        is_dual = bool(self._secondary_url and self._secondary_model)
        notes: list[str] = []
        if not is_dual:
            notes.append("secondary endpoint not configured; dual-VLM disabled")
        return BackendCapabilities(
            name=self.name,
            supports_dual_vlm=is_dual,
            supports_constrained_decoding=True,
            supports_logprobs=True,
            supports_multi_image=True,
            supports_tensor_parallelism=True,
            notes=tuple(notes),
        )

    def resolve(self, role: VLMRole) -> tuple[str, str]:
        if role in (VLMRole.PRIMARY, VLMRole.LITE):
            return self._primary_url, self._primary_model

        if role in (VLMRole.SECONDARY, VLMRole.CRITIC):
            if self._secondary_url and self._secondary_model:
                return self._secondary_url, self._secondary_model
            logger.debug(
                "vllm_role_collapsed_to_primary",
                requested_role=role.value,
                reason="secondary endpoint not configured",
            )
            return self._primary_url, self._primary_model

        raise ValueError(f"Unknown VLMRole: {role!r}")

    def health(self) -> BackendHealth:
        roles_to_probe: list[VLMRole] = [VLMRole.PRIMARY]
        if self._secondary_url and self._secondary_model:
            roles_to_probe.append(VLMRole.SECONDARY)

        results: dict[VLMRole, dict[str, Any]] = {}
        all_healthy = True
        for role in roles_to_probe:
            url, model = self.resolve(role)
            client = self._get_client(role)
            t0 = time.perf_counter()
            try:
                healthy = client.is_healthy()
                detail: dict[str, Any] = {
                    "healthy": healthy,
                    "base_url": url,
                    "model": model,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            except Exception as exc:  # pragma: no cover - defensive
                healthy = False
                detail = {
                    "healthy": False,
                    "base_url": url,
                    "model": model,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "error": str(exc),
                }
            results[role] = detail
            all_healthy = all_healthy and healthy
        return BackendHealth(
            backend_name=self.name,
            overall_healthy=all_healthy,
            roles=results,
        )

    def send_vision_request(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        """Sync vision call.

        Phase 1: ``schema`` is converted to vLLM's ``extra_body`` shape:

            {"guided_json": <schema_dict>,
             "guided_decoding_backend": "xgrammar" | "outlines"}

        which makes the decoder produce only schema-valid tokens. We use
        ``extra_body`` (not ``response_format``) so the operator's choice
        of guided-decoding backend (XGrammar vs Outlines) is respected;
        ``response_format`` lets vLLM pick its default backend
        unconditionally.
        """
        client = self._get_client(role)
        _, model = self.resolve(role)
        extra_body = self._build_extra_body(schema)
        return client.send_vision_request(
            request,
            model=model,
            extra_body=extra_body,
        )

    async def send_vision_request_async(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        # See LMStudioBackend.send_vision_request_async — async client
        # path does not yet plumb extra_body through. All Phase 1 call
        # sites are sync; async constrained decoding lands when needed.
        client = self._get_client(role)
        return await client.send_vision_request_async(request)

    def _build_extra_body(
        self,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Convert a JSON-Schema dict into vLLM's ``extra_body``.

        Returns ``None`` when no schema is supplied so unconstrained
        calls bypass the guided-decoding pipeline entirely.
        """
        if schema is None:
            return None
        return {
            "guided_json": schema,
            "guided_decoding_backend": self._guided_decoding_backend,
        }

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._clients.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def guided_decoding_backend(self) -> str:
        """Exposed for ``constrained_decode()`` (Phase 1) to read."""
        return self._guided_decoding_backend

    def _get_client(self, role: VLMRole) -> LMStudioClient:
        if role not in self._clients:
            url, model = self.resolve(role)
            self._clients[role] = LMStudioClient(
                base_url=url,
                model=model,
                **self._client_kwargs,
            )
        return self._clients[role]


# Protocol conformance is checked at runtime in
# ``tests/unit/test_vllm_backend.py`` via ``isinstance(backend, VLMBackend)``.
