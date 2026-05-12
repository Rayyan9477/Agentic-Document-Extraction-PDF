"""V3 Phase 6 — hallucination-injection harness tests.

Coverage:

* Each of six ``InjectionType``s mutates the extraction in the
  documented way and is **deterministic** (same input → same output).
* No-op behaviour when the corpus doesn't carry the right shape.
* ``classify_caught`` returns canonical verdicts.
* ``confusion_matrix`` aggregates per layer × injection type.
* ``InjectionReport.catch_rate`` excludes ``not_applicable`` rows.
"""

from __future__ import annotations

import copy

import pytest

from tests.eval.inject import (
    CAUGHT,
    InjectionConfig,
    InjectionRunner,
    InjectionType,
    classify_caught,
    confusion_matrix,
)
from tests.eval.inject.report import (
    MISSED,
    NOT_APPLICABLE,
    TRACKED_LAYERS,
    InjectionReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> InjectionRunner:
    return InjectionRunner(config=InjectionConfig(rng_seed=42))


@pytest.fixture
def golden_extraction() -> dict:
    """A small, well-shaped extraction with a mix of types."""
    return {
        "patient_name": "Alice Anderson",
        "provider_name": "Dr. Bob Brown",
        "service_date": "2026-04-01",
        "total_charge": "$1,234.56",
        "diagnosis_code": "E11.9",
        "patient_id": "MRN-77441",
    }


@pytest.fixture
def golden_with_bbox() -> dict:
    """Extraction shape carrying provenance (FieldValue-ish)."""
    return {
        "patient_name": {
            "value": "Alice",
            "_provenance": {
                "page": 1,
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05},
            },
        },
        "total_charge": {
            "value": "$100.00",
            "_provenance": {
                "page": 1,
                "bbox": {"x": 0.5, "y": 0.7, "width": 0.2, "height": 0.04},
            },
        },
    }


# ---------------------------------------------------------------------------
# Per-injection mutation behaviour
# ---------------------------------------------------------------------------


class TestValueSwap:
    def test_swaps_two_fields(self, runner, golden_extraction) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.VALUE_SWAP
        )
        assert result.injection_type == InjectionType.VALUE_SWAP
        # Mutated has the same key-set but at least two values
        # have moved relative to the original.
        assert set(result.mutated_extraction.keys()) == set(golden_extraction.keys())
        differences = sum(
            1
            for k in golden_extraction
            if result.mutated_extraction[k] != golden_extraction[k]
        )
        assert differences == 2, "value_swap touches exactly two fields"

    def test_deterministic(self, golden_extraction) -> None:
        a = InjectionRunner(config=InjectionConfig(rng_seed=7)).run(
            golden_extraction, injection_type=InjectionType.VALUE_SWAP
        )
        b = InjectionRunner(config=InjectionConfig(rng_seed=7)).run(
            golden_extraction, injection_type=InjectionType.VALUE_SWAP
        )
        assert a.mutated_extraction == b.mutated_extraction


class TestAmountFake:
    def test_replaces_currency(self, runner, golden_extraction) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.AMOUNT_FAKE
        )
        assert result.field_name == "total_charge"
        assert result.injected_value != result.original_value

    def test_noop_when_no_currency(self, runner) -> None:
        result = runner.run(
            {"name": "Alice", "id": 7},
            injection_type=InjectionType.AMOUNT_FAKE,
        )
        assert result.field_name is None
        assert "no-op" in result.notes


class TestPhantomField:
    def test_adds_synthetic_field(self, runner, golden_extraction) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.PHANTOM_FIELD
        )
        assert "phantom_provider_id" in result.mutated_extraction
        assert result.mutated_extraction["phantom_provider_id"] == "PHANTOM-9999"

    def test_noop_when_phantom_already_exists(self, runner) -> None:
        starting = {"phantom_provider_id": "real-99"}
        result = runner.run(starting, injection_type=InjectionType.PHANTOM_FIELD)
        assert result.field_name is None


class TestBboxDrift:
    def test_drifts_bbox(self, runner, golden_with_bbox) -> None:
        result = runner.run(
            golden_with_bbox, injection_type=InjectionType.BBOX_DRIFT
        )
        assert result.field_name in golden_with_bbox
        # The mutated bbox has shifted.
        moved = result.mutated_extraction[result.field_name]
        original = golden_with_bbox[result.field_name]
        assert moved["_provenance"]["bbox"] != original["_provenance"]["bbox"]

    def test_noop_when_no_bbox(self, runner, golden_extraction) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.BBOX_DRIFT
        )
        # Plain string fields have no bbox.
        assert result.field_name is None


class TestFieldDrop:
    def test_drops_field(self, runner, golden_extraction) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.FIELD_DROP
        )
        assert result.field_name not in result.mutated_extraction
        # All other keys preserved.
        for k in golden_extraction:
            if k != result.field_name:
                assert k in result.mutated_extraction


class TestPlaceholderInject:
    def test_replaces_with_known_placeholder(
        self, runner, golden_extraction
    ) -> None:
        result = runner.run(
            golden_extraction, injection_type=InjectionType.PLACEHOLDER_INJECT
        )
        assert result.injected_value in InjectionConfig().placeholder_pool


# ---------------------------------------------------------------------------
# classify_caught
# ---------------------------------------------------------------------------


class TestClassifyCaught:
    def test_critic_caught(self) -> None:
        v = classify_caught(critic_recommendation="retry")
        assert v["critic"] == CAUGHT

    def test_critic_missed(self) -> None:
        v = classify_caught(critic_recommendation="accept")
        assert v["critic"] == MISSED

    def test_validator_caught_when_violations(self) -> None:
        v = classify_caught(validator_violations=["modifier conflict"])
        assert v["validator"] == CAUGHT

    def test_validator_missed_when_empty_list(self) -> None:
        v = classify_caught(validator_violations=[])
        assert v["validator"] == MISSED

    def test_pattern_detector_caught(self) -> None:
        v = classify_caught(pattern_hits=["SPATIAL_ANOMALY"])
        assert v["pattern_detector"] == CAUGHT

    def test_bbox_roundtrip_caught(self) -> None:
        v = classify_caught(bbox_roundtrip_failed=True)
        assert v["bbox_roundtrip"] == CAUGHT

    def test_not_applicable_when_signal_missing(self) -> None:
        v = classify_caught()
        for layer in TRACKED_LAYERS:
            assert v[layer] == NOT_APPLICABLE


# ---------------------------------------------------------------------------
# confusion_matrix + report
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_aggregates_counts(self, runner, golden_extraction) -> None:
        rows: list = []
        # 3 phantom-field rows: critic catches all 3.
        for i in range(3):
            r = runner.run(
                golden_extraction,
                injection_type=InjectionType.PHANTOM_FIELD,
                record_id=f"r{i}",
            )
            v = classify_caught(critic_recommendation="retry")
            rows.append((r, v))
        # 2 value-swap rows: critic misses both.
        for i in range(2):
            r = runner.run(
                golden_extraction,
                injection_type=InjectionType.VALUE_SWAP,
                record_id=f"r{i}",
            )
            v = classify_caught(critic_recommendation="accept")
            rows.append((r, v))
        m = confusion_matrix(rows)
        assert m["critic"]["phantom_field"][CAUGHT] == 3
        assert m["critic"]["value_swap"][MISSED] == 2

    def test_report_catch_rate(self) -> None:
        rep = InjectionReport(
            matrix={
                "critic": {
                    "phantom_field": {CAUGHT: 8, MISSED: 2, NOT_APPLICABLE: 0},
                    "value_swap": {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 5},
                    "bbox_drift": {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 0},
                    "field_drop": {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 0},
                    "amount_fake": {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 0},
                    "placeholder_inject": {CAUGHT: 0, MISSED: 0, NOT_APPLICABLE: 0},
                },
            }
        )
        assert rep.catch_rate("critic", "phantom_field") == pytest.approx(0.8)
        # not_applicable doesn't count.
        assert rep.catch_rate("critic", "value_swap") == 0.0

    def test_to_dict_round_trip(self, runner, golden_extraction) -> None:
        rows = []
        for i in range(2):
            r = runner.run(
                golden_extraction,
                injection_type=InjectionType.PHANTOM_FIELD,
                record_id=f"r{i}",
            )
            v = classify_caught(critic_recommendation="retry")
            rows.append((r, v))
        rep = InjectionReport.from_rows(
            rows, config_summary={"corpus": "smoke"}
        )
        d = rep.to_dict()
        assert d["total_rows"] == 2
        assert "matrix" in d
        assert "catch_rate" in d
        assert d["config_summary"]["corpus"] == "smoke"
