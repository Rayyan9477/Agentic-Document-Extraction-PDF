"""
``VLMBackend`` protocol — the single interface every VLM backend implements.

A backend is a thin wrapper that knows how to:

* resolve a ``VLMRole`` (``primary`` / ``secondary`` / ``critic`` / ``lite``)
  to a concrete ``(base_url, model_id)`` pair,
* send a vision request and return a ``VisionResponse`` (re-using the
  existing dataclass from ``src.client.lm_client`` for compatibility with
  ``BaseAgent.send_vision_request``),
* report its capabilities (does it support real dual-VLM? does it preserve
  logprobs? can it bind a JSON schema at decode time?).

Backends do NOT own retry/timeout/JSON-extraction policy — that lives in
the existing ``LMStudioClient``. The vLLM backend ships its own equivalent.

This file deliberately exposes only types/protocol surface — the concrete
backends live in ``lm_studio_backend.py`` and ``vllm_backend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    # Avoid a circular import at module import time. ``VisionRequest``
    # and ``VisionResponse`` are stable, frozen dataclasses we re-use.
    from src.client.lm_client import VisionRequest, VisionResponse


# ---------------------------------------------------------------------------
# Role taxonomy
# ---------------------------------------------------------------------------


class VLMRole(str, Enum):
    """Logical role for a VLM call.

    The role is decoupled from the concrete model so the same agent code
    works against single-VLM (``lite``) or dual-VLM (``standard``/``hard``)
    deployments. The backend resolves role → URL+model.
    """

    #: Default extractor pass. EXTRACTOR-style prompting.
    PRIMARY = "primary"

    #: Auditor / second-opinion pass. AUDITOR-style prompting (Phase 2+).
    SECONDARY = "secondary"

    #: Independent verifier (Phase 3). Family-rotated against the consensus
    #: of primary/secondary so the Critic never shares family with the
    #: extraction majority.
    CRITIC = "critic"

    #: Single-VLM mode for resource-constrained installs. Resolves to
    #: whichever role the operator marked as the primary.
    LITE = "lite"


# ---------------------------------------------------------------------------
# Capability reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What a backend can and cannot do.

    Reported by ``VLMBackend.capabilities()``. Used by:

    * ``/health/full`` — exposes capability matrix to operators so the
      degraded-mode story is explicit, not silent.
    * The orchestrator — disables nodes that require unsupported
      capabilities (e.g. dual-VLM extraction won't run on a backend that
      reports ``supports_dual_vlm=False``).
    * The CI gate — skips ``pytest.mark.gpu`` paths that require a real
      backend.
    """

    name: str
    """Human-readable backend name (``lm_studio`` / ``vllm``)."""

    supports_dual_vlm: bool
    """True iff the backend can serve two distinct VLMs concurrently."""

    supports_constrained_decoding: bool
    """True iff the backend can bind a JSON schema at decode time."""

    supports_logprobs: bool
    """True iff per-token logprobs are returned. False = imputed confidences only."""

    supports_multi_image: bool
    """True iff a single request may include multiple image payloads."""

    supports_tensor_parallelism: bool
    """True iff the backend can serve a single model across multiple GPUs."""

    notes: tuple[str, ...] = field(default_factory=tuple)
    """Free-form notes for operators (degraded-mode warnings etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``/health`` payloads and audit logs."""
        return {
            "name": self.name,
            "supports_dual_vlm": self.supports_dual_vlm,
            "supports_constrained_decoding": self.supports_constrained_decoding,
            "supports_logprobs": self.supports_logprobs,
            "supports_multi_image": self.supports_multi_image,
            "supports_tensor_parallelism": self.supports_tensor_parallelism,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Health reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """Per-role health snapshot.

    Returned by ``VLMBackend.health()``. The ``roles`` mapping covers
    every role the backend has resolved an endpoint for; missing roles
    indicate "not configured", not "unhealthy".
    """

    backend_name: str
    overall_healthy: bool
    roles: dict[VLMRole, dict[str, Any]] = field(default_factory=dict)
    """Per-role detail: ``{healthy, base_url, model, latency_ms, error?}``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "overall_healthy": self.overall_healthy,
            "roles": {role.value: detail for role, detail in self.roles.items()},
        }


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VLMBackend(Protocol):
    """Common surface for every VLM backend.

    ``BaseAgent.send_vision_request`` calls through this protocol via
    ``ModelRouter`` (Phase 0.c extends the router with role-routing). The
    backend resolves the role at request time — call-sites do not see
    whether they're talking to LM Studio or vLLM.
    """

    @property
    def name(self) -> str:
        """Backend short name (``"lm_studio"`` / ``"vllm"``)."""

    def capabilities(self) -> BackendCapabilities:
        """Static capability declaration. Cheap to call."""

    def health(self) -> BackendHealth:
        """Probe each configured role's endpoint. May make network calls."""

    def resolve(self, role: VLMRole) -> tuple[str, str]:
        """Resolve a role to ``(base_url, model_id)``.

        Raises:
            ValueError: when the role is not configured for this backend.
        """

    def send_vision_request(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        """Synchronous vision call.

        Args:
            request: The ``VisionRequest`` (image + prompt + max_tokens etc.).
            role: Which logical role this call serves. The backend uses
                this to pick the URL + model.
            schema: Optional JSON Schema for constrained decoding. When
                supplied AND the backend reports
                ``supports_constrained_decoding=True``, the response is
                guaranteed schema-valid. Phase 1 wires this end-to-end;
                Phase 0 leaves ``schema=None`` everywhere.
        """

    async def send_vision_request_async(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        """Async variant. Backends MAY share an event-loop client pool."""

    def close(self) -> None:
        """Release HTTP/network resources."""
