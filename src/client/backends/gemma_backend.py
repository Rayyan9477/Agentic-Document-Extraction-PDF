"""
Gemma 4 backend adapter.

Targets ``lmstudio-community/gemma-4-26B-A4B-it-GGUF`` served by LM Studio
at a dedicated port (default ``localhost:1235``). Implements the
``VLMBackend`` protocol so the existing reconciler / critic / orchestrator
code runs unchanged when ``settings.vlm.backend = "gemma"``.

Two extraction paths sit side-by-side:

1. **Standard schema-bound extraction** (``send_vision_request``,
   ``send_vision_request_async``) — delegates to ``LMStudioClient`` with
   OpenAI-compat ``response_format={"type": "json_schema", ...}``. Gemma 4
   supports ``response_format`` natively in LM Studio 0.3+, so the entire
   six-layer validation pyramid keeps working without modification.

2. **Native function-calling for medical-code validators**
   (``call_with_tools``) — exposes the five tools defined in
   ``gemma_tools.py`` and lets Gemma 4 invoke them mid-extraction. This is
   the technical-innovation centrepiece of the function-calling demo.
   ``LMStudioClient.send_vision_request`` already accepts ``extra_body``
   which forwards arbitrary kwargs to the OpenAI client, so we route
   ``tools`` + ``tool_choice`` through there. The tool call results are
   then dispatched via ``gemma_tools.dispatch_tool_call``.

Role resolution:
    PRIMARY / SECONDARY / CRITIC / LITE all resolve to the same
    ``primary_url`` + ``primary_model`` by design. The orchestrator
    distinguishes Pass 1 / Pass 2 / Critic via prompt frame, not model
    identity. This keeps the operator runbook to one model load.

Capability matrix (May 2026):
    * ``supports_dual_vlm`` — False (single instance by design).
    * ``supports_constrained_decoding`` — True (via ``response_format``).
    * ``supports_logprobs`` — False (Gemma 4 GGUF + LM Studio not
      surfacing logprobs at submission time).
    * ``supports_multi_image`` — True (Gemma 4 vision supports it).
    * ``supports_tensor_parallelism`` — False (single instance).

Phase K does not split traffic between roles. A future evolution could
add ``secondary_url`` etc. if operators want a separate Gemma instance
for Pass 2 — the protocol surface already allows it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.client.backends.gemma_tools import VERIDOC_TOOLS, dispatch_tool_call
from src.client.backends.protocol import (
    BackendCapabilities,
    BackendHealth,
    VLMRole,
)
from src.client.lm_client import LMStudioClient
from src.config import get_logger


if TYPE_CHECKING:
    from src.client.lm_client import VisionRequest, VisionResponse


logger = get_logger(__name__)


# Schema-tool forcing — when ``schema`` is passed via the tool-use path,
# wrap it as a single tool with this name. The model is then forced to
# call it via ``tool_choice``, and we parse ``tool_call.input`` as the
# extraction result. Mirrors the Anthropic adapter pattern documented in
# the Phase 9 Bedrock plan.
SCHEMA_TOOL_NAME = "emit_structured_result"


class GemmaBackend:
    """``VLMBackend`` adapter for Gemma 4 via LM Studio.

    Delegates standard schema-bound calls to the legacy ``LMStudioClient``
    (same OpenAI-compat HTTP surface, same retry/timeout policy). Adds a
    tool-calling primitive used by the medical-code validator demo.
    """

    name = "gemma"

    def __init__(
        self,
        primary_url: str,
        primary_model: str,
        *,
        tool_call_timeout: int = 180,
        max_retries: int = 3,
        temperature: float = 0.1,
        register_rcm_tools: bool = True,
        fail_open_on_health: bool = False,
    ) -> None:
        self._primary_url = primary_url
        self._primary_model = primary_model
        self._tool_call_timeout = tool_call_timeout
        self._max_retries = max_retries
        self._temperature = temperature
        self._register_rcm_tools = register_rcm_tools
        self._fail_open_on_health = fail_open_on_health

        # Build kwargs once so ``_get_client`` can pass them lazily.
        self._client_kwargs: dict[str, Any] = {
            "max_retries": max_retries,
            "temperature": temperature,
            "timeout": tool_call_timeout,
        }

        # Lazy per-role client cache. All four roles collapse to one
        # endpoint but each role still gets its own client so test
        # patches on a single role don't bleed into the others.
        self._clients: dict[VLMRole, LMStudioClient] = {}

        logger.info(
            "gemma_backend_initialised",
            primary_url=primary_url,
            primary_model=primary_model,
            register_rcm_tools=register_rcm_tools,
        )

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def capabilities(self) -> BackendCapabilities:
        notes: list[str] = [
            "all roles (primary/secondary/critic/lite) share one endpoint",
            "Pass1/Pass2/Critic distinguished by prompt frame, not model identity",
        ]
        if self._register_rcm_tools:
            notes.append(
                f"{len(VERIDOC_TOOLS)} medical-code tools attached "
                f"to every healthcare-mode extraction request"
            )
        return BackendCapabilities(
            name=self.name,
            supports_dual_vlm=False,
            supports_constrained_decoding=True,  # via response_format=json_schema
            supports_logprobs=False,
            supports_multi_image=True,
            supports_tensor_parallelism=False,
            notes=tuple(notes),
        )

    def resolve(self, role: VLMRole) -> tuple[str, str]:
        """All roles map to the same endpoint by design (Phase K)."""
        if role not in {
            VLMRole.PRIMARY,
            VLMRole.SECONDARY,
            VLMRole.CRITIC,
            VLMRole.LITE,
        }:
            raise ValueError(f"Unknown VLMRole: {role!r}")
        return self._primary_url, self._primary_model

    def health(self) -> BackendHealth:
        """Probe the configured endpoint.

        Returns one health entry under the PRIMARY role (the only one
        that has a distinct endpoint). When ``fail_open_on_health`` is
        True (demo environments), the result is always healthy so the
        API can boot before LM Studio finishes warming up the model.
        """
        url, model = self.resolve(VLMRole.PRIMARY)
        client = self._get_client(VLMRole.PRIMARY)
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
        overall = healthy or self._fail_open_on_health
        return BackendHealth(
            backend_name=self.name,
            overall_healthy=overall,
            roles={VLMRole.PRIMARY: detail},
        )

    def send_vision_request(
        self,
        request: "VisionRequest",
        *,
        role: VLMRole = VLMRole.PRIMARY,
        schema: dict[str, Any] | None = None,
    ) -> "VisionResponse":
        """Sync vision call with optional schema-bound decoding.

        Phase K — same ``response_format=json_schema`` path as LM Studio
        because Gemma 4 GGUFs in LM Studio honour this OpenAI-compat
        construct. The native function-calling path is reached via
        ``call_with_tools`` instead, which the orchestrator invokes
        explicitly when the medical-RCM profile is active.
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
        """Async vision call.

        Mirrors ``LMStudioBackend.send_vision_request_async`` — the async
        path in the legacy client does not yet plumb ``response_format``,
        so schema-bound async calls fall back to prompt-level discipline.
        Sync path covers every Phase K schema-bound call site.
        """
        client = self._get_client(role)
        return await client.send_vision_request_async(request)

    def call_with_tools(
        self,
        request: "VisionRequest",
        *,
        tools: tuple[dict[str, Any], ...] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        role: VLMRole = VLMRole.PRIMARY,
    ) -> "VisionResponse":
        """Send a vision request with the function-calling tool registry attached.

        Used by the function-calling demo path. When ``tools`` is None and
        ``register_rcm_tools=True``, the five default medical-code tools
        from ``gemma_tools.VERIDOC_TOOLS`` are attached. ``tool_choice``
        accepts the OpenAI shape: ``"auto"`` lets the model decide,
        ``{"type": "function", "function": {"name": ...}}`` forces a
        single tool, ``"none"`` disables tool use.

        Returns the standard ``VisionResponse``. Tool-call payloads are
        embedded in the response message and parseable by callers via
        ``parse_tool_calls()``.
        """
        client = self._get_client(role)
        _, model = self.resolve(role)
        active_tools = tools if tools is not None else (
            VERIDOC_TOOLS if self._register_rcm_tools else ()
        )
        if not active_tools:
            return client.send_vision_request(request, model=model)
        extra_body: dict[str, Any] = {
            "tools": list(active_tools),
            "tool_choice": tool_choice,
        }
        return client.send_vision_request(
            request,
            model=model,
            extra_body=extra_body,
        )

    def parse_tool_calls(
        self,
        response: "VisionResponse",
    ) -> list[dict[str, Any]]:
        """Best-effort parse of tool_calls from a ``VisionResponse``.

        Returns a list of ``{name, arguments, call_id}`` dicts. Empty
        list if the response carries no tool calls. The arguments value
        is a parsed Python dict (the model emits a JSON string per the
        OpenAI spec; we json-decode here so callers never have to).

        Two parse paths:
            1. Structured: the underlying OpenAI client populated
               ``response.usage["tool_calls"]`` (LM Studio's preferred
               surface for tools).
            2. Fallback regex: scan ``response.content`` for
               ``<tool_call>...</tool_call>`` blocks emitted by Gemma's
               raw chat template when LM Studio's tool-parsing layer
               isn't engaged.
        """
        import json
        import re

        results: list[dict[str, Any]] = []

        # Path 1 — structured. ``VisionResponse.usage`` is a free-form
        # dict that LMStudioClient populates; if it carries tool_calls
        # we parse them first.
        usage = getattr(response, "usage", None) or {}
        structured = usage.get("tool_calls") if isinstance(usage, dict) else None
        if isinstance(structured, list):
            for entry in structured:
                if not isinstance(entry, dict):
                    continue
                fn = entry.get("function") or {}
                name = fn.get("name") or entry.get("name")
                raw_args = fn.get("arguments") or entry.get("arguments") or "{}"
                args = self._safe_json_loads(raw_args)
                results.append(
                    {
                        "name": name,
                        "arguments": args,
                        "call_id": entry.get("id") or entry.get("call_id"),
                    }
                )

        # Path 2 — fallback regex on content. Gemma 4's chat template
        # emits ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
        # when the GGUF's tool-parsing layer isn't engaged. We scan once.
        if not results and response.content:
            pattern = re.compile(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
            )
            for match in pattern.finditer(response.content):
                payload = self._safe_json_loads(match.group(1))
                if isinstance(payload, dict) and payload.get("name"):
                    results.append(
                        {
                            "name": payload.get("name"),
                            "arguments": payload.get("arguments") or {},
                            "call_id": None,
                        }
                    )

        return results

    def dispatch_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Dispatch parsed tool calls to their Python implementations.

        Returns one result per call: ``{name, arguments, result, call_id}``.
        Wraps ``gemma_tools.dispatch_tool_call`` so the same error-handling
        applies regardless of the parse path.
        """
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("name") or ""
            arguments = call.get("arguments") or {}
            result = dispatch_tool_call(name, arguments)
            results.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                    "call_id": call.get("call_id"),
                }
            )
        return results

    @staticmethod
    def tool_calls_to_provenance_stages(
        dispatched_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project dispatched tool calls into Provenance-compatible stages.

        The orchestrator's per-field provenance threading consumes
        ``{stage, agent, metadata}`` records. This helper converts the
        output of :meth:`dispatch_tool_calls` into that shape so a
        downstream caller can do::

            stages = backend.tool_calls_to_provenance_stages(
                backend.dispatch_tool_calls(backend.parse_tool_calls(response))
            )
            for s in stages:
                field.provenance = field.provenance.append_stage(
                    s["stage"], agent=s["agent"]
                )

        Returns one stage dict per call with ``stage="gemma_tool_call"``
        and a stable agent label ``"gemma:<tool_name>"``. The full call
        record (including ``result``) is preserved under ``metadata`` so
        the Source View timeline can render the tool input/output.

        Phase K — wires the technical-innovation centrepiece (native
        function calling) into the click-to-source provenance UI without
        requiring orchestrator changes today.
        """
        stages: list[dict[str, Any]] = []
        for call in dispatched_calls:
            name = call.get("name") or "unknown_tool"
            stages.append(
                {
                    "stage": "gemma_tool_call",
                    "agent": f"gemma:{name}",
                    "metadata": {
                        "tool_name": name,
                        "arguments": call.get("arguments") or {},
                        "result": call.get("result") or {},
                        "call_id": call.get("call_id"),
                    },
                }
            )
        return stages

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

    @staticmethod
    def _build_response_format(
        schema: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Convert a JSON-Schema dict into LM Studio's response_format.

        Identical wire shape to :class:`LMStudioBackend`. Gemma 4 GGUFs
        in LM Studio honour the OpenAI-compat ``json_schema`` field.
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

    @staticmethod
    def _safe_json_loads(payload: Any) -> Any:
        """Parse a JSON string; pass-through if already a dict; ``{}`` on failure."""
        import json

        if isinstance(payload, dict):
            return payload
        if not isinstance(payload, str):
            return {}
        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            return {}

    def _get_client(self, role: VLMRole) -> LMStudioClient:
        """Lazily create a per-role ``LMStudioClient``.

        All four roles resolve to the same endpoint by design, so the
        per-role cache is mostly a hedge against test isolation — patches
        applied to one role don't leak into another.
        """
        if role not in self._clients:
            url, model = self.resolve(role)
            self._clients[role] = LMStudioClient(
                base_url=url,
                model=model,
                **self._client_kwargs,
            )
        return self._clients[role]


__all__ = ["GemmaBackend", "SCHEMA_TOOL_NAME"]
