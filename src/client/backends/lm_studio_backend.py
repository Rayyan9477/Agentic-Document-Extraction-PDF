"""
LM Studio backend adapter.

Wraps the existing ``LMStudioClient`` (kept verbatim) and exposes the
``VLMBackend`` protocol. Role resolution reads new
``LMStudioBackendSettings`` (Phase 0.d) so a single LM Studio backend
can serve multiple roles either against one instance (degraded
``single_only`` / ``jit_swap`` modes) or two instances on different
ports (``dual_instance`` mode).

Capability matrix for LM Studio (May 2026):

* ``supports_dual_vlm`` — only when ``LM_STUDIO_DUAL_MODE=dual_instance``
* ``supports_constrained_decoding`` — yes via
  ``response_format={"type": "json_schema", ...}`` (LM Studio 0.3+).
* ``supports_logprobs`` — partial (top-k only)
* ``supports_multi_image`` — depends on the loaded model
* ``supports_tensor_parallelism`` — no

Phase 0 leaves ``schema=None`` everywhere; Phase 1 wires schemas in.
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


class LMStudioDualMode(str):
    """Allowed values for ``LMStudioBackendSettings.dual_mode``.

    These are documented strings, not an Enum — Pydantic settings prefers
    plain Literal/str unions. See ``settings.vlm.lm_studio.dual_mode``.

    * ``dual_instance`` — operator runs two LM Studio servers on different
      ports; primary and secondary roles each route to one. Real
      heterogeneous dual-VLM.
    * ``jit_swap`` — single LM Studio instance, model swaps on demand.
      Functionally serial; we log a loud warning. Reasonable for
      development; bad for production.
    * ``single_only`` — primary only. Secondary/Critic roles transparently
      collapse to primary with a logged warning. Lite-mode behaviour.
    """

    DUAL_INSTANCE = "dual_instance"
    JIT_SWAP = "jit_swap"
    SINGLE_ONLY = "single_only"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LMStudioBackend:
    """``VLMBackend`` adapter around the legacy ``LMStudioClient``.

    Stateless from the orchestrator's view: each role gets its own
    underlying ``LMStudioClient`` lazily. The legacy client owns
    retry/timeout/JSON-extraction policy; this class only adds the
    role → endpoint resolution.
    """

    name = "lm_studio"

    def __init__(
        self,
        primary_url: str,
        primary_model: str,
        *,
        secondary_url: str | None = None,
        secondary_model: str | None = None,
        dual_mode: str = LMStudioDualMode.SINGLE_ONLY,
        # Plumbing for the underlying LMStudioClient
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_min_wait: int | None = None,
        retry_max_wait: int | None = None,
    ) -> None:
        self._primary_url = primary_url
        self._primary_model = primary_model
        self._secondary_url = secondary_url
        self._secondary_model = secondary_model
        self._dual_mode = dual_mode

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

        # Lazy per-role clients. ``_clients[role]`` holds an LMStudioClient
        # bound to the role's resolved (url, model) pair.
        self._clients: dict[VLMRole, LMStudioClient] = {}

        # Loud one-time warning when running in a degraded dual-mode.
        if dual_mode in (LMStudioDualMode.JIT_SWAP, LMStudioDualMode.SINGLE_ONLY):
            logger.warning(
                "lm_studio_backend_degraded_mode",
                dual_mode=dual_mode,
                impact=(
                    "secondary/critic roles will collapse to primary; "
                    "heterogeneous dual-VLM is NOT active"
                ),
            )

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def capabilities(self) -> BackendCapabilities:
        is_real_dual = self._dual_mode == LMStudioDualMode.DUAL_INSTANCE and bool(
            self._secondary_url and self._secondary_model
        )
        notes: list[str] = []
        if not is_real_dual:
            notes.append(
                f"dual_mode={self._dual_mode}: secondary/critic collapse to primary"
            )
        return BackendCapabilities(
            name=self.name,
            supports_dual_vlm=is_real_dual,
            supports_constrained_decoding=True,  # via response_format=json_schema
            supports_logprobs=False,  # top-k only; treat as no
            supports_multi_image=True,  # most VLMs in LM Studio support it
            supports_tensor_parallelism=False,
            notes=tuple(notes),
        )

    def resolve(self, role: VLMRole) -> tuple[str, str]:
        """Map a ``VLMRole`` to ``(base_url, model_id)``.

        ``LITE`` always maps to primary. ``SECONDARY`` and ``CRITIC`` map
        to the secondary endpoint when configured *and* dual_mode is
        ``dual_instance``; otherwise they collapse to primary with a
        debug log.
        """
        if role in (VLMRole.PRIMARY, VLMRole.LITE):
            return self._primary_url, self._primary_model

        if role in (VLMRole.SECONDARY, VLMRole.CRITIC):
            if (
                self._dual_mode == LMStudioDualMode.DUAL_INSTANCE
                and self._secondary_url
                and self._secondary_model
            ):
                return self._secondary_url, self._secondary_model
            # Degraded — collapse silently after the constructor's loud
            # warning. Per-call debug log makes the collapse visible in
            # traces but doesn't spam the operator.
            logger.debug(
                "lm_studio_role_collapsed_to_primary",
                requested_role=role.value,
                dual_mode=self._dual_mode,
            )
            return self._primary_url, self._primary_model

        # Defensive — VLMRole is a closed enum; if a new variant is
        # added without updating this method, fail loudly.
        raise ValueError(f"Unknown VLMRole: {role!r}")

    def health(self) -> BackendHealth:
        """Probe each configured endpoint."""
        roles_to_probe: list[VLMRole] = [VLMRole.PRIMARY]
        if (
            self._dual_mode == LMStudioDualMode.DUAL_INSTANCE
            and self._secondary_url
            and self._secondary_model
        ):
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

        Phase 1: when ``schema`` is provided, the backend translates it to
        LM Studio's OpenAI-style ``response_format={"type":"json_schema",
        "json_schema": {...}}`` so the decoder cannot emit JSON that
        violates the schema. The schema name is ``"veridoc"`` because LM
        Studio requires a name field but does not key on its value.
        """
        client = self._get_client(role)
        _, model = self.resolve(role)
        response_format = self._build_response_format(schema)
        return client.send_vision_request(
            request,
            model=model,
            response_format=response_format,
        )

    async def send_vision_request_async(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        client = self._get_client(role)
        # The async client path in LMStudioClient does not yet plumb
        # response_format through; the existing prompt-level JSON
        # discipline + post-hoc parse continues to work. Schema-bound
        # ASYNC calls fall back to that unconstrained path until a
        # follow-up wires response_format into send_vision_request_async.
        # Sync path covers all current schema-bound call sites in
        # Phase 1, so this gap is not a blocker.
        return await client.send_vision_request_async(request)

    @staticmethod
    def _build_response_format(
        schema: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Convert a JSON-Schema dict into LM Studio's response_format.

        Returns ``None`` when ``schema`` is ``None`` so the legacy
        unconstrained path is preserved. The ``strict`` flag is
        intentionally left at its default (LM Studio enforces strictly
        when the schema is well-formed; ``additionalProperties`` may
        surface as an explicit constraint in the schema itself).
        """
        if schema is None:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "veridoc",
                "schema": schema,
            },
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

    def _get_client(self, role: VLMRole) -> LMStudioClient:
        """Lazily create a per-role ``LMStudioClient``.

        Each role gets its own client because LMStudioClient stores
        ``base_url`` + ``model`` as instance attributes; sharing a single
        client across roles would force per-call URL/model overrides
        which the legacy class wasn't built for.
        """
        if role not in self._clients:
            url, model = self.resolve(role)
            self._clients[role] = LMStudioClient(
                base_url=url,
                model=model,
                **self._client_kwargs,
            )
        return self._clients[role]


# Protocol conformance is checked at runtime in
# ``tests/unit/test_lm_studio_backend.py`` via
# ``isinstance(backend, VLMBackend)`` (Protocol is ``runtime_checkable``).
