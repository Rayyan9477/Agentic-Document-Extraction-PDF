"""V3 Phase 6 — eval smoke driver."""

from __future__ import annotations

import pytest

from tests.eval.smoke import (
    SMOKE_FIELD_FIDELITY_FLOOR,
    FidelityResult,
    aggregate_fidelity,
    compare_extraction,
)


class TestCompareExtraction:
    def test_identical_yields_full_fidelity(self) -> None:
        a = {"x": "Alice", "y": 42}
        b = {"x": "Alice", "y": 42}
        result = compare_extraction(a, b)
        assert result.fidelity == 1.0

    def test_case_insensitive_matches(self) -> None:
        a = {"name": "alice"}
        b = {"name": "ALICE"}
        result = compare_extraction(a, b)
        assert result.fidelity == 1.0

    def test_strip_whitespace(self) -> None:
        a = {"name": "  Alice  "}
        b = {"name": "Alice"}
        result = compare_extraction(a, b)
        assert result.fidelity == 1.0

    def test_numeric_tolerance(self) -> None:
        a = {"amount": 100.001}
        b = {"amount": 100.000}
        result = compare_extraction(a, b)
        assert result.fidelity == 1.0

    def test_missing_key_counts_as_wrong(self) -> None:
        a = {"name": "Alice"}
        b = {"name": "Alice", "id": "X-1"}
        result = compare_extraction(a, b)
        assert result.fidelity < 1.0
        assert result.fields_correct == 1
        assert result.fields_compared == 2  # union has 2 keys

    def test_extra_phantom_key_counts_as_wrong(self) -> None:
        a = {"name": "Alice", "phantom": "X"}
        b = {"name": "Alice"}
        result = compare_extraction(a, b)
        # Phantom key counted in denominator but not numerator.
        assert result.fields_correct == 1
        assert result.fields_compared == 2


class TestAggregateFidelity:
    def test_mean(self) -> None:
        results = [
            FidelityResult(record_id="r1", fields_compared=10, fields_correct=10),
            FidelityResult(record_id="r2", fields_compared=10, fields_correct=8),
            FidelityResult(record_id="r3", fields_compared=10, fields_correct=6),
        ]
        assert aggregate_fidelity(results) == pytest.approx((1.0 + 0.8 + 0.6) / 3)

    def test_empty(self) -> None:
        assert aggregate_fidelity([]) == 0.0


class TestFidelityFloor:
    def test_floor_constant_published(self) -> None:
        # Document that the floor exists and is in (0, 1).
        assert 0.0 < SMOKE_FIELD_FIDELITY_FLOOR < 1.0
