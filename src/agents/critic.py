"""
V3 Phase 3 — Critic agent.

Independent verifier that runs after the validator (or the dual-VLM
reconciler in dual_vlm mode). Family-rotates against the consensus of
Pass 1 / Pass 2 by binding to ``VLMRole.CRITIC``, which the configured
backend resolves to whichever endpoint is NOT the primary in dual-VLM
deployments. In legacy / Lite mode the role collapses to primary with
a logged warning — the Critic still runs but as a same-model
second-opinion (less powerful, still useful).

The Critic does NOT re-extract. It produces a ``CriticReport`` whose
``recommendation`` drives downstream routing:

* ``accept`` — proceed to confidence/route as usual.
* ``verify_bbox`` — orchestrator should run bbox-roundtrip on flagged
  concerns (Phase 3 wires this; Phase 2's ``HeterogeneousReconciler``
  also uses round-trip but earlier).
* ``retry`` — re-run extraction with concerns embedded.
* ``human_review`` — escalate via ``interrupt()``.

The agent is sync; the Critic is one VLM call per document (not per
page, unlike Pass 1 / Pass 2). For multi-page documents we audit
against the merged extraction once, using the first page as the
visual anchor — this matches the legacy validator's pattern and keeps
the Critic latency bounded at ~3-5s. Per-page Critic auditing is a
Phase 6 follow-up if eval shows it's needed.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents.base import BaseAgent
from src.client.backends.protocol import VLMRole
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import (
    ExtractionState,
    update_state,
)
from src.prompts.critic import (
    build_critic_system_prompt,
    build_critic_user_prompt,
)


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


CriticIssue = Literal[
    "supported",
    "not_visible",
    "contradicts_image",
    "ambiguous",
]
"""Per-field audit verdict.

``supported`` is technically valid in the schema but the prompt
instructs the Critic to omit it from concerns (the orchestrator
treats absence as agreement). We accept it for robustness when a
verbose Critic includes positive verdicts anyway.
"""


CriticSeverity = Literal["info", "warning", "error"]


CriticRecommendation = Literal[
    "accept",
    "verify_bbox",
    "retry",
    "human_review",
]


class CriticConcern(BaseModel):
    """One field's audit verdict + severity."""

    model_config = ConfigDict(extra="allow")

    field_path: str = Field(
        description=(
            "Dotted path to the audited field (e.g. ``patient_name`` "
            "or ``service_lines.0.cpt_code``)."
        ),
    )
    issue: CriticIssue = Field(
        description=(
            "Audit verdict. ``supported``=visible-and-correct, "
            "``not_visible``=plausible-but-not-shown, "
            "``contradicts_image``=visibly-wrong, ``ambiguous``=unclear."
        ),
    )
    severity: CriticSeverity = Field(
        default="warning",
        description="Triage severity. ``error`` blocks emission.",
    )
    observed_in_image: bool = Field(
        default=False,
        description="True iff the value (or any version of it) is on the page.",
    )
    recommended_bbox: list[float] | None = Field(
        default=None,
        description=(
            "Optional bbox where the Critic believes the correct value "
            "lives. Used by the bbox-roundtrip helper if the "
            "orchestrator routes to ``verify_bbox``. Normalised "
            "[x1, y1, x2, y2] in [0, 1]."
        ),
    )


class CriticReport(BaseModel):
    """Schema bound at decode time for the Critic VLM call.

    The decoder cannot emit anything outside this shape.
    ``trust_score`` is a single float for the whole extraction;
    per-field detail lives in ``concerns``.
    """

    model_config = ConfigDict(extra="allow")

    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence that the extraction is correct. "
            "1.0 = every field verified visible-and-correct; 0.0 = "
            "nothing in the extraction matches the image."
        ),
    )
    concerns: list[CriticConcern] = Field(
        default_factory=list,
        description=(
            "One row per audited field with severity ≥ info. The "
            "Critic should omit ``supported`` verdicts to keep the "
            "list focused on actionable concerns."
        ),
    )
    recommendation: CriticRecommendation = Field(
        description="Downstream routing directive.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CriticAgent(BaseAgent):
    """Independent verifier for extractions."""

    AGENT_NAME = "critic"

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
        """Audit ``state["merged_extraction"]`` against the page image.

        When the merged extraction is empty (no fields), short-circuit
        to ``human_review`` with trust_score 0.0 — nothing to verify.

        When the page-image source is missing (defensive), short-
        circuit to ``accept`` so the Critic never blocks the pipeline
        on missing inputs.
        """
        merged = state.get("merged_extraction") or {}
        page_images = state.get("page_images", []) or []
        modalities = list(state.get("modalities", []) or [])
        doc_type = state.get("document_type", "UNKNOWN")

        if not merged:
            return self._short_circuit(
                state,
                trust_score=0.0,
                recommendation="human_review",
                reason="merged_extraction_empty",
            )

        if not page_images:
            self._logger.warning(
                "critic_no_page_images_short_circuit",
                processing_id=state.get("processing_id"),
            )
            return self._short_circuit(
                state,
                trust_score=1.0,
                recommendation="accept",
                reason="no_page_image_to_audit",
            )

        first_page = page_images[0]
        image_data = first_page.get("data_uri") or first_page.get(
            "base64_encoded", ""
        )
        if not image_data:
            return self._short_circuit(
                state,
                trust_score=1.0,
                recommendation="accept",
                reason="page_image_blank",
            )

        system_prompt = build_critic_system_prompt()
        user_prompt = build_critic_user_prompt(
            extraction=merged,
            document_type=doc_type,
            page_number=1,
            page_count=len(page_images),
            modalities=modalities,
        )

        t0 = time.perf_counter()
        try:
            payload, trace = self.send_vision_request_with_schema(
                image_data=image_data,
                prompt=user_prompt,
                schema=CriticReport,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=2048,
                role=VLMRole.CRITIC,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._logger.warning(
                "critic_call_failed_falling_back_accept",
                error=str(exc),
                latency_ms=latency_ms,
            )
            # Critic failure is non-fatal: the legacy validator + the
            # reconciler already produced their signals. Recommend
            # ``accept`` so the pipeline continues; the failure is
            # logged and surfaces in the audit trail.
            return self._short_circuit(
                state,
                trust_score=0.5,
                recommendation="accept",
                reason=f"critic_call_failed: {exc}",
                latency_ms=latency_ms,
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # ``payload`` is a dict (the constrained-decode wrapper unboxes
        # the Pydantic model). Normalise into CriticReport-shape for
        # downstream consumers.
        report = self._normalise_report(payload)

        self._logger.info(
            "critic_audit_complete",
            trust_score=report.get("trust_score"),
            concerns_count=len(report.get("concerns", [])),
            recommendation=report.get("recommendation"),
            latency_ms=latency_ms,
            model_id=trace.model_id,
        )

        # V3 Phase 6 — emit canonical critic_disagreed event when the
        # Critic returns anything other than "accept". This is the
        # signal product analytics needs to track Critic catch-rate
        # over time without parsing audit logs.
        recommendation = report.get("recommendation", "accept")
        if recommendation != "accept":
            try:
                from src.monitoring.observability import (
                    EVENT_CRITIC_DISAGREED,
                    _read_trace_id_from_context,
                    get_dispatcher,
                )

                _obs = get_dispatcher()
                if _obs is not None and _obs.is_active:
                    _obs.emit_event(
                        EVENT_CRITIC_DISAGREED,
                        {
                            "processing_id": state.get("processing_id"),
                            "recommendation": recommendation,
                            "trust_score": report.get("trust_score"),
                            "concerns_count": len(report.get("concerns", [])),
                            "document_type": state.get("document_type"),
                            "profile": state.get("profile"),
                            "model_id": trace.model_id,
                            "trace_id": _read_trace_id_from_context(),
                        },
                    )
            except Exception:  # pragma: no cover - defensive
                pass

        return update_state(
            state,
            {
                "critic_report": report,
                "critic_recommendation": report.get("recommendation", "accept"),
                "critic_model_id": trace.model_id or "",
                "critic_latency_ms": latency_ms,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_report(payload: dict[str, Any]) -> dict[str, Any]:
        """Coerce the schema-bound payload into a defensive shape.

        The constrained decoder guarantees keys + types, but we still
        clamp the trust_score and validate the recommendation enum so
        a misbehaving backend can't poison the routing decision.
        """
        if not isinstance(payload, dict):
            return {
                "trust_score": 0.5,
                "concerns": [],
                "recommendation": "accept",
            }
        ts = payload.get("trust_score", 0.5)
        try:
            ts = max(0.0, min(1.0, float(ts)))
        except (TypeError, ValueError):
            ts = 0.5
        rec = payload.get("recommendation", "accept")
        if rec not in ("accept", "verify_bbox", "retry", "human_review"):
            rec = "accept"
        concerns = payload.get("concerns", [])
        if not isinstance(concerns, list):
            concerns = []
        return {
            "trust_score": ts,
            "concerns": concerns,
            "recommendation": rec,
        }

    @staticmethod
    def _short_circuit(
        state: ExtractionState,
        *,
        trust_score: float,
        recommendation: str,
        reason: str,
        latency_ms: int = 0,
    ) -> ExtractionState:
        report = {
            "trust_score": trust_score,
            "concerns": [],
            "recommendation": recommendation,
            "_short_circuit_reason": reason,
        }
        return update_state(
            state,
            {
                "critic_report": report,
                "critic_recommendation": recommendation,
                "critic_model_id": "",
                "critic_latency_ms": latency_ms,
            },
        )
