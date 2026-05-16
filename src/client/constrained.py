"""
Constrained decoding wrapper.

Single entry point for every schema-bound VLM call across the engine.
Phase 0 ships the wrapper as a thin pass-through with the schema
parameter accepted but only surfaced through the backend (the existing
JSON-extraction path in ``LMStudioClient`` continues to do the parse).

Phase 1 will activate the actual decode-time enforcement:

* **vLLM path** — ``extra_body={"guided_json": <schema>, "guided_decoding_backend": "xgrammar"}``
* **LM Studio path** — ``response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": <schema>}}``

In both cases the decoder cannot emit JSON that violates the schema,
which removes an entire class of malformed-output failures and also
makes the legacy ``_extract_json()`` regex chain redundant for the
agents that route through this wrapper.

Why a wrapper instead of a method on the backend? Three reasons:

1. **Shape conversion.** Agents pass Pydantic ``BaseModel`` types; the
   backends speak JSON Schema. The wrapper converts once.
2. **Pydantic instance return.** The wrapper parses the JSON response
   into the requested model so call-sites get a typed object, not a
   ``dict[str, Any]``.
3. **DecodingTrace.** Captures per-token logprobs and which agent /
   role / model produced the output. ``ConfidenceCalibrator`` consumes
   it (Phase 1+).

Phase 0 returns ``DecodingTrace`` with the trace fields set but
logprobs empty — backends haven't been wired to surface them yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from src.client.backends.protocol import VLMBackend, VLMRole
from src.config import get_logger


if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.client.lm_client import VisionRequest, VisionResponse


logger = get_logger(__name__)


T = TypeVar("T", bound="BaseModel")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConstrainedDecodingError(Exception):
    """Raised when a schema-bound call fails to produce schema-valid output.

    Phase 0 only raises this when JSON extraction itself fails (the
    legacy regex chain returns ``None``). Phase 1 will additionally
    raise when the backend's guided-decoding feedback indicates the
    decoder gave up — for example, vLLM's grammar engine fails to
    satisfy a deeply-nested discriminated union.
    """


# ---------------------------------------------------------------------------
# Decoding trace (consumed by ConfidenceCalibrator)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecodingTrace:
    """Per-call telemetry for confidence calibration and observability."""

    backend_name: str
    role: VLMRole
    model_id: str
    schema_name: str | None
    latency_ms: int
    tokens_in: int
    tokens_out: int
    logprobs: list[float] = field(default_factory=list)
    """Per-token logprobs over grammar-permitted tokens. Empty in Phase 0."""

    schema_enforced: bool = False
    """Whether the backend actually enforced the schema at decode time.

    Phase 0 reports ``False`` everywhere. Phase 1 flips this to ``True``
    on the vLLM/LM Studio paths once the wrapper actually injects the
    guided-decoding kwargs.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "role": self.role.value,
            "model_id": self.model_id,
            "schema": self.schema_name,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "logprob_count": len(self.logprobs),
            "schema_enforced": self.schema_enforced,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def constrained_decode(
    backend: VLMBackend,
    request: "VisionRequest",
    *,
    role: VLMRole = VLMRole.PRIMARY,
    schema: type["BaseModel"] | None = None,
) -> tuple["VisionResponse", DecodingTrace]:
    """Send a vision request and (Phase 1+) enforce a Pydantic schema at decode.

    Args:
        backend: The ``VLMBackend`` instance (typically from ``get_backend()``).
        request: The vision request (image + prompt + max_tokens etc.).
        role: Which logical role this call serves.
        schema: Optional Pydantic model class. When supplied, Phase 1
            will pass the JSON-Schema form to the backend's
            ``send_vision_request(..., schema=...)`` so the decoder
            cannot emit invalid output. In Phase 0 this is informational
            only — the call still succeeds, but the backend treats it as
            unconstrained.

    Returns:
        ``(response, trace)``. ``response`` is the existing
        ``VisionResponse`` so call-sites that already use ``has_json``
        and ``parsed_json`` continue to work. ``trace`` carries the
        per-call telemetry consumed by ``ConfidenceCalibrator`` and
        Phoenix spans.

    Raises:
        ConstrainedDecodingError: When JSON extraction returns ``None``
            and a schema was requested. Without a schema, malformed
            output is returned as ``response.parsed_json = None`` (the
            legacy contract is preserved).
    """
    schema_dict: dict[str, Any] | None = None
    schema_name: str | None = None
    if schema is not None:
        schema_dict = schema.model_json_schema()
        schema_name = schema.__name__

    t0 = time.perf_counter()
    response = backend.send_vision_request(request, role=role, schema=schema_dict)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    base_url, model_id = backend.resolve(role)
    # Phase 1: ``schema_enforced`` is True iff the backend reports
    # ``supports_constrained_decoding`` AND a schema was supplied. Both
    # shipped backends (LMStudioBackend, VLLMBackend) report True; a
    # future custom backend that fakes the surface would set it False.
    schema_enforced = bool(
        schema is not None and backend.capabilities().supports_constrained_decoding
    )
    trace = DecodingTrace(
        backend_name=backend.name,
        role=role,
        model_id=model_id,
        schema_name=schema_name,
        latency_ms=latency_ms,
        tokens_in=response.prompt_tokens,
        tokens_out=response.completion_tokens,
        # Logprob extraction lives in a follow-up: vLLM exposes them on
        # ``response.choices[0].logprobs`` but the legacy LMStudioClient
        # discards them today. ConfidenceCalibrator handles empty lists.
        logprobs=[],
        schema_enforced=schema_enforced,
    )

    if schema is not None and not response.has_json:
        # Schema was requested but the response wasn't parseable. With
        # Phase 1 guided decoding active on both shipping backends, this
        # branch should be unreachable except for transport errors —
        # we surface it as a typed error rather than silently returning
        # ``parsed_json=None``.
        logger.error(
            "constrained_decode_no_json",
            backend=backend.name,
            role=role.value,
            schema=schema_name,
            content_length=len(response.content),
            schema_enforced=schema_enforced,
        )
        raise ConstrainedDecodingError(
            f"Backend {backend.name} returned non-JSON for schema {schema_name}; "
            f"first 200 chars: {response.content[:200]!r}"
        )

    return response, trace


async def constrained_decode_async(
    backend: VLMBackend,
    request: "VisionRequest",
    *,
    role: VLMRole = VLMRole.PRIMARY,
    schema: type["BaseModel"] | None = None,
) -> tuple["VisionResponse", DecodingTrace]:
    """Async variant of :func:`constrained_decode`."""
    schema_dict: dict[str, Any] | None = None
    schema_name: str | None = None
    if schema is not None:
        schema_dict = schema.model_json_schema()
        schema_name = schema.__name__

    t0 = time.perf_counter()
    response = await backend.send_vision_request_async(
        request, role=role, schema=schema_dict
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    _, model_id = backend.resolve(role)
    trace = DecodingTrace(
        backend_name=backend.name,
        role=role,
        model_id=model_id,
        schema_name=schema_name,
        latency_ms=latency_ms,
        tokens_in=response.prompt_tokens,
        tokens_out=response.completion_tokens,
        logprobs=[],
        schema_enforced=False,
    )

    if schema is not None and not response.has_json:
        raise ConstrainedDecodingError(
            f"Backend {backend.name} returned non-JSON for schema {schema_name}; "
            f"first 200 chars: {response.content[:200]!r}"
        )

    return response, trace
