"""V3 Phase 6 — generic round-trip smoke driver.

This is the < 90s eval driver wired into CI's ``eval-smoke`` job.
It runs a small fixed corpus through the pipeline, compares
extracted output to a known-good ground truth, and asserts the
field-fidelity metric stays above the configured floor.

The driver does not call a live VLM. It uses the recorded
``mock_pipeline_run`` fixture data so CI doesn't depend on a GPU
runner. The full corpus driver (Synthea, CUAD, FUNSD) lives in
sibling modules and is gated behind the ``slow`` / ``gpu`` markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Field-fidelity threshold for the smoke job. The plan documents
# 92% as the production bar on the synthea round-trip; smoke uses a
# generous 70% because the synthetic generic corpus is small and
# noise-free, but anything below 70% indicates a real regression.
SMOKE_FIELD_FIDELITY_FLOOR = 0.70


@dataclass(slots=True)
class FidelityResult:
    """Per-record / per-corpus fidelity score."""

    record_id: str
    fields_compared: int
    fields_correct: int

    @property
    def fidelity(self) -> float:
        if self.fields_compared == 0:
            return 0.0
        return self.fields_correct / self.fields_compared

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "fields_compared": self.fields_compared,
            "fields_correct": self.fields_correct,
            "fidelity": round(self.fidelity, 4),
        }


def compare_extraction(
    extracted: dict[str, Any],
    expected: dict[str, Any],
    *,
    record_id: str = "rec",
    case_insensitive: bool = True,
    strip_whitespace: bool = True,
) -> FidelityResult:
    """Compare two flat extractions and return per-field fidelity.

    Comparison rules (intentionally lenient, mirroring the
    ``tests/eval/diff/`` semantic-equivalence rules from the plan):

    * Strings are stripped + case-folded by default.
    * Numerics with a < 0.01 absolute delta count as equal.
    * Missing keys count as wrong (no credit for "didn't try").
    * Extra keys in extracted (phantoms) count as wrong.
    """
    keys = set(expected.keys()) | set(extracted.keys())
    correct = 0
    for k in keys:
        if k not in expected or k not in extracted:
            continue  # extra key on either side = automatic miss
        a = extracted[k]
        b = expected[k]
        if isinstance(a, str) and isinstance(b, str):
            a_norm = a.strip() if strip_whitespace else a
            b_norm = b.strip() if strip_whitespace else b
            if case_insensitive:
                a_norm, b_norm = a_norm.casefold(), b_norm.casefold()
            if a_norm == b_norm:
                correct += 1
                continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) < 0.01:
                correct += 1
                continue
        if a == b:
            correct += 1
    return FidelityResult(
        record_id=record_id,
        fields_compared=len(keys),
        fields_correct=correct,
    )


def aggregate_fidelity(results: list[FidelityResult]) -> float:
    """Mean fidelity across a list of per-record results."""
    if not results:
        return 0.0
    return sum(r.fidelity for r in results) / len(results)
