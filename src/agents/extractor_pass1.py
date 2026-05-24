"""
V3 Phase 2 — Pass 1 (EXTRACTOR) agent.

Runs against the **primary** VLM (Qwen 3.6 27B-VL by default). Frame
is EXTRACTOR-style: read the page, emit the schema fields fluently.
The companion ``ExtractorPass2Agent`` runs against a different model
family (Gemma 4 31B-VL) with AUDITOR framing, and the
``HeterogeneousReconciler`` fuses the two.

Both passes call ``send_vision_request_with_schema`` so JSON shape is
guaranteed at decode time. Pass 1's schema is **permissive** (a
``JSONObjectEnvelope``) — the EXTRACTOR's job is fluency, not strict
bbox-grounding. Pass 2's schema mandates bboxes; the reconciler's
bbox-overlap tiebreak depends on Pass 2 being the bbox-grounded side.

Today this agent is invoked only when
``settings.extraction.engine == "dual_vlm"``. The legacy ``Extractor``
agent (``src/agents/extractor.py``) continues to handle the
``engine == "legacy"`` default. No mutual import; the orchestrator
factory chooses one or the other based on the flag.
"""

from __future__ import annotations

import time
from typing import Any

from src.agents._constrained_envelopes import JSONObjectEnvelope
from src.agents.base import BaseAgent, ExtractionError
from src.client.backends.protocol import VLMRole
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    update_state,
)
from src.prompts.pass1_extractor import (
    build_pass1_system_prompt,
    build_pass1_user_prompt,
)


logger = get_logger(__name__)


class ExtractorPass1Agent(BaseAgent):
    """Pass 1 / EXTRACTOR — primary-VLM fluent extraction."""

    AGENT_NAME = "extractor_pass1"

    def __init__(
        self,
        client: LMStudioClient | None = None,
        model_router: Any | None = None,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            client=client,
            model_router=model_router,
        )
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # LangGraph entry point
    # ------------------------------------------------------------------

    def process(self, state: ExtractionState) -> ExtractionState:
        """Run Pass 1 on every page and stash results in state.

        Output shape matches ``ExtractionState`` Phase 2 fields:

        * ``state["pass1_result"]`` is a dict keyed by 1-based page
          number, each value is the raw VLM payload (a JSON object).
        * ``state["pass1_model_id"]`` records the resolved model
          identifier (best-effort; ``""`` if no model_router).
        * ``state["pass1_latency_ms"]`` accumulates wall-clock VLM time.

        The method does NOT populate ``merged_extraction`` — that's the
        reconciler's job. Pass 1 + Pass 2 each write their raw output;
        the reconciler fuses them.
        """
        page_images = state.get("page_images", [])
        if not page_images:
            raise ExtractionError(
                "Pass 1 invoked with no page images",
                agent_name=self.name,
                recoverable=False,
            )

        schema_name = state.get("selected_schema_name") or "generic"
        document_type = state.get("document_type", "UNKNOWN")
        modalities = list(state.get("modalities", []) or [])
        # V3 Phase 5: pass profile through to the prompt builder so
        # the extractor sees the same RCM / finance reminders the
        # reconciler will reconcile against.
        profile = state.get("profile") or None
        field_defs = self._resolve_field_defs(state)

        pass1_result: dict[int, dict[str, Any]] = {}
        total_latency_ms = 0
        model_id_seen = ""

        system_prompt = build_pass1_system_prompt()

        for page in page_images:
            page_number = page.get("page_number", 0) or 0
            image_data = page.get("data_uri") or page.get("base64_encoded", "")
            if not image_data:
                self._logger.warning(
                    "pass1_skip_blank_page",
                    page_number=page_number,
                )
                continue

            user_prompt = build_pass1_user_prompt(
                schema_fields=field_defs,
                document_type=document_type,
                page_number=page_number,
                page_count=len(page_images),
                modalities=modalities,
                profile=profile,
            )

            t0 = time.perf_counter()
            try:
                payload, trace = self.send_vision_request_with_schema(
                    image_data=image_data,
                    prompt=user_prompt,
                    schema=JSONObjectEnvelope,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=6000,
                    role=VLMRole.PRIMARY,
                )
            except Exception as exc:  # noqa: BLE001 - we log and continue
                latency_ms = int((time.perf_counter() - t0) * 1000)
                total_latency_ms += latency_ms
                self._logger.warning(
                    "pass1_page_failed",
                    page_number=page_number,
                    error=str(exc),
                    latency_ms=latency_ms,
                )
                pass1_result[page_number] = {
                    "_pass1_error": str(exc),
                    "fields": {},
                }
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            total_latency_ms += latency_ms
            model_id_seen = trace.model_id or model_id_seen
            pass1_result[page_number] = payload

            self._logger.debug(
                "pass1_page_complete",
                page_number=page_number,
                latency_ms=latency_ms,
                model_id=trace.model_id,
                schema_enforced=trace.schema_enforced,
                field_count=self._count_fields(payload),
            )

        self._logger.info(
            "pass1_complete",
            pages_processed=len(pass1_result),
            total_latency_ms=total_latency_ms,
            model_id=model_id_seen,
            schema=schema_name,
        )

        return update_state(
            state,
            {
                "pass1_result": pass1_result,
                "pass1_model_id": model_id_seen,
                "pass1_latency_ms": total_latency_ms,
                "extraction_engine": "dual_vlm",
                # NB: ``set_status`` returns the *full* ExtractionState;
                # spreading it here would clobber the freshly-written
                # ``pass1_result`` above (Python dict literals resolve
                # duplicate keys in source order, later-wins). Write the
                # status fragment directly instead.
                "status": ExtractionStatus.EXTRACTING.value,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_field_defs(self, state: ExtractionState) -> list[dict[str, Any]]:
        """Pull the active schema's field definitions from state.

        Today the legacy single-VLM extractor builds these on the fly;
        Phase 2 keeps that contract. When neither the adaptive schema
        nor the registered schema is available, returns an empty list
        and the prompt falls back to a generic instruction.
        """
        adaptive = state.get("adaptive_schema") or {}
        if adaptive and isinstance(adaptive, dict):
            fields = adaptive.get("fields") or []
            if isinstance(fields, list) and fields:
                return [f if isinstance(f, dict) else {} for f in fields]

        # Registry path
        try:
            from src.schemas import SchemaRegistry

            schema_name = state.get("selected_schema_name")
            if schema_name:
                schema = SchemaRegistry.get(schema_name)
                if schema is not None:
                    return [
                        f.to_dict() if hasattr(f, "to_dict") else dict(f)
                        for f in getattr(schema, "fields", [])
                    ]
        except Exception as exc:  # pragma: no cover - registry is robust
            self._logger.warning(
                "pass1_schema_resolution_failed",
                error=str(exc),
            )
        return []

    @staticmethod
    def _count_fields(payload: dict[str, Any]) -> int:
        """Best-effort field-count for telemetry (envelope shape varies)."""
        if not isinstance(payload, dict):
            return 0
        if "fields" in payload and isinstance(payload["fields"], dict):
            return len(payload["fields"])
        # Generic envelope where the model returns the fields at the top level.
        return sum(1 for k in payload if not k.startswith("_"))
