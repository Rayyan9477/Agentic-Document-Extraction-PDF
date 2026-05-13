"""
V3 Phase 3 — Critic combiner.

Fuses three orthogonal signals into the final raw confidence that the
existing ``ConfidenceCalibrator`` consumes:

1. **Dual-pass agreement** — ``reconciliation_metadata.agreement_rate``
   in dual_vlm mode; the legacy validator's ``overall_confidence`` in
   legacy mode. Captures *internal* agreement.
2. **Critic trust** — ``critic_report.trust_score``. Captures the
   *external* opinion of an independent verifier.
3. **Modality penalty** — derived from active modalities (``fax`` /
   ``handwritten`` / ``visual``). Captures known-degraded inputs.

The combiner is **pure** (no I/O, no calibrator dependency); the
output is a dict of components plus a ``raw_combined`` value. The
calibrator runs after the combiner so calibration and combination
remain decoupled — operators can swap calibrators without rewriting
combination logic.

Default weights: ``(0.50, 0.30, 0.20)`` per ``EXTRACTION.md`` §6 and
``settings.extraction.critic_combiner_weights``.
"""

from __future__ import annotations

from typing import Any

from src.config import get_logger, get_settings


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Modality penalty table
# ---------------------------------------------------------------------------


# Per-modality penalty in [0, 1]. Higher = worse input quality.
# When multiple modalities apply we take the max (worst-of) penalty.
_MODALITY_PENALTIES: dict[str, float] = {
    "fax": 0.7,
    "handwritten": 0.6,
    "visual": 0.4,
    # Printed / form / table are baseline (no penalty).
}


def _modality_penalty(modalities: list[str]) -> float:
    """Worst-of penalty across active modalities."""
    if not modalities:
        return 0.0
    return max(
        (_MODALITY_PENALTIES.get(m, 0.0) for m in modalities),
        default=0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combine_confidence(
    *,
    dual_pass_agreement: float,
    critic_trust: float,
    modalities: list[str] | None = None,
    weights: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Combine three signals into a raw confidence value.

    Args:
        dual_pass_agreement: ``reconciliation_metadata.agreement_rate``
            in dual_vlm mode, or the legacy ``overall_confidence`` in
            legacy mode. Clamped to [0, 1].
        critic_trust: ``critic_report.trust_score``. ``1.0`` when the
            Critic was skipped (so the term contributes its full
            weight without dragging confidence down).
        modalities: detected modality labels.
        weights: ``(w_dual, w_critic, w_modality)`` summing to 1.0.
            Defaults to ``settings.extraction.critic_combiner_weights``.

    Returns:
        Dict with keys ``dual_pass``, ``critic``, ``modality_penalty``,
        and ``raw_combined``. The calibrator consumes ``raw_combined``;
        the rest are surfaced for observability and audit.
    """
    if weights is None:
        weights = get_settings().extraction.critic_combiner_weights

    w_dual, w_critic, w_modality = weights
    dp = max(0.0, min(1.0, float(dual_pass_agreement)))
    ct = max(0.0, min(1.0, float(critic_trust)))
    pen = _modality_penalty(list(modalities or []))
    modality_term = max(0.0, 1.0 - pen)

    raw = w_dual * dp + w_critic * ct + w_modality * modality_term
    raw = max(0.0, min(1.0, raw))

    return {
        "dual_pass": dp,
        "critic": ct,
        "modality_penalty": pen,
        "raw_combined": raw,
    }


def apply_combiner_to_state(state: dict[str, Any]) -> dict[str, Any]:
    """Read state, compute components, return ``confidence_components``.

    Pure read; never mutates ``state``. The orchestrator calls this
    after the Critic node and writes the result via ``update_state``.
    """
    settings = get_settings()

    # Dual-pass agreement — prefer reconciliation_metadata when dual_vlm
    # ran; fall back to the legacy overall_confidence otherwise.
    recon_meta = state.get("reconciliation_metadata") or {}
    if isinstance(recon_meta, dict) and "agreement_rate" in recon_meta:
        dual = float(recon_meta.get("agreement_rate") or 0.0)
    else:
        dual = float(state.get("overall_confidence") or 0.0)

    # Critic trust — 1.0 when Critic was skipped or short-circuited
    # to "accept" with no concerns. We treat the absence of a critic
    # report as "no signal" rather than penalising the extraction.
    critic_report = state.get("critic_report") or {}
    if isinstance(critic_report, dict) and "trust_score" in critic_report:
        critic_trust = float(critic_report.get("trust_score") or 0.0)
    else:
        critic_trust = 1.0  # no Critic ⇒ no penalty term

    modalities = list(state.get("modalities", []) or [])
    components = combine_confidence(
        dual_pass_agreement=dual,
        critic_trust=critic_trust,
        modalities=modalities,
        weights=settings.extraction.critic_combiner_weights,
    )

    logger.debug(
        "critic_combiner_applied",
        dual_pass=components["dual_pass"],
        critic=components["critic"],
        modality_penalty=components["modality_penalty"],
        raw_combined=components["raw_combined"],
    )
    return components
