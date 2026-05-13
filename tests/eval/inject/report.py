"""V3 Phase 6 — injection-harness reporting.

Aggregates per-injection ``InjectionResult`` outcomes into a
catch-rate confusion matrix per (layer, injection_type). The
reporter does not run the layers itself — call sites supply
"verdicts" (caught vs missed) per layer per result. This keeps
the runner pure (no Critic dependency) and makes the matrix easy
to tabulate in CI.

Layers tracked:

* ``pattern_detector`` — pre-Critic catch (regex / spatial anomaly).
* ``validator`` — post-extraction validator pack hits.
* ``critic`` — Critic agent ``recommendation != "accept"``.
* ``bbox_roundtrip`` — secondary VLM crop-and-re-extract delta.

Reports are JSON-serialisable so a nightly CI job can drop them
straight into PostHog / a dashboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tests.eval.inject.runner import InjectionResult, InjectionType


# ---------------------------------------------------------------------------
# Verdict primitives
# ---------------------------------------------------------------------------


CAUGHT = "caught"
MISSED = "missed"
NOT_APPLICABLE = "not_applicable"  # layer skipped this row


# Layers in catch-rate priority order. The Critic is conventionally
# last because it runs after validator + pattern detector.
TRACKED_LAYERS: tuple[str, ...] = (
    "pattern_detector",
    "validator",
    "critic",
    "bbox_roundtrip",
)


# ---------------------------------------------------------------------------
# classify_caught
# ---------------------------------------------------------------------------


def classify_caught(
    *,
    critic_recommendation: str | None = None,
    validator_violations: list[str] | None = None,
    pattern_hits: list[str] | None = None,
    bbox_roundtrip_failed: bool | None = None,
) -> dict[str, str]:
    """Translate per-layer outputs into the canonical verdict
    vocabulary.

    Each layer is independent — a row can be ``caught`` by multiple
    layers (which is what the catch-rate-by-layer matrix wants).

    Returns a ``{layer: verdict}`` map covering every layer in
    ``TRACKED_LAYERS``. ``NOT_APPLICABLE`` is reserved for layers
    that didn't run (the caller passed ``None`` for that signal).
    """
    verdicts: dict[str, str] = {}

    if pattern_hits is None:
        verdicts["pattern_detector"] = NOT_APPLICABLE
    else:
        verdicts["pattern_detector"] = CAUGHT if pattern_hits else MISSED

    if validator_violations is None:
        verdicts["validator"] = NOT_APPLICABLE
    else:
        verdicts["validator"] = CAUGHT if validator_violations else MISSED

    if critic_recommendation is None:
        verdicts["critic"] = NOT_APPLICABLE
    else:
        verdicts["critic"] = CAUGHT if critic_recommendation != "accept" else MISSED

    if bbox_roundtrip_failed is None:
        verdicts["bbox_roundtrip"] = NOT_APPLICABLE
    else:
        verdicts["bbox_roundtrip"] = CAUGHT if bbox_roundtrip_failed else MISSED

    return verdicts


# ---------------------------------------------------------------------------
# Confusion matrix builder
# ---------------------------------------------------------------------------


def confusion_matrix(
    rows: list[tuple[InjectionResult, dict[str, str]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Build the (layer → injection_type → verdict_counts) matrix.

    Each row is ``(InjectionResult, verdict_map)`` where the verdict
    map comes from ``classify_caught``. The output structure is::

        {
          "pattern_detector": {
            "phantom_field": {"caught": 8, "missed": 2, "not_applicable": 0},
            "value_swap":    {"caught": 1, "missed": 9, "not_applicable": 0},
            ...
          },
          "validator": {...},
          ...
        }
    """
    matrix: dict[str, dict[str, dict[str, int]]] = {
        layer: {t.value: {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 0} for t in InjectionType}
        for layer in TRACKED_LAYERS
    }
    for result, verdicts in rows:
        type_key = result.injection_type.value
        for layer in TRACKED_LAYERS:
            verdict = verdicts.get(layer, NOT_APPLICABLE)
            matrix[layer][type_key][verdict] = (
                matrix[layer][type_key].get(verdict, 0) + 1
            )
    return matrix


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InjectionReport:
    """Aggregated catch-rate report across an injection run."""

    matrix: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    total_rows: int = 0
    config_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls,
        rows: list[tuple[InjectionResult, dict[str, str]]],
        *,
        config_summary: dict[str, Any] | None = None,
    ) -> InjectionReport:
        return cls(
            matrix=confusion_matrix(rows),
            total_rows=len(rows),
            config_summary=config_summary or {},
        )

    def catch_rate(self, layer: str, injection_type: str) -> float:
        """Catch rate (0..1) for a single (layer, injection_type) cell.

        ``not_applicable`` rows are excluded from the denominator;
        if every row is N/A for this cell, returns 0.0.
        """
        cell = self.matrix.get(layer, {}).get(injection_type)
        if not cell:
            return 0.0
        n = cell.get(CAUGHT, 0) + cell.get(MISSED, 0)
        if n == 0:
            return 0.0
        return cell.get(CAUGHT, 0) / n

    def per_layer_catch_rate(self) -> dict[str, dict[str, float]]:
        """Catch rate by (layer, injection_type) for the whole matrix."""
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for layer in self.matrix:
            for inj_type in self.matrix[layer]:
                out[layer][inj_type] = self.catch_rate(layer, inj_type)
        return dict(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "matrix": self.matrix,
            "catch_rate": self.per_layer_catch_rate(),
            "config_summary": self.config_summary,
        }
