"""
Unit tests for Phase 3A: Confidence Calibration.

Tests PlattCalibrator, IsotonicCalibrator, LinearCalibrator,
ConfidenceCalibrator (unified), CalibrationMetrics, and persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.validation.calibration import (
    BaseCalibrator,
    CalibrationMetrics,
    CalibrationPoint,
    CalibrationResult,
    ConfidenceCalibrator,
    IsotonicCalibrator,
    LinearCalibrator,
    PlattCalibrator,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _make_overconfident_points(n: int = 50) -> list[CalibrationPoint]:
    """
    Create calibration data where model is overconfident.

    Raw confidence ~0.9 but actual accuracy ~0.6
    """
    rng = np.random.RandomState(42)
    points = []
    for _ in range(n):
        raw = rng.uniform(0.7, 0.95)
        is_correct = rng.random() < 0.6  # Only 60% correct
        points.append(CalibrationPoint(raw_confidence=raw, is_correct=bool(is_correct)))
    return points


def _make_well_calibrated_points(n: int = 50) -> list[CalibrationPoint]:
    """Create calibration data that is already well-calibrated."""
    rng = np.random.RandomState(42)
    points = []
    for _ in range(n):
        raw = rng.uniform(0.3, 0.95)
        is_correct = rng.random() < raw  # Accuracy matches confidence
        points.append(CalibrationPoint(raw_confidence=raw, is_correct=bool(is_correct)))
    return points


def _make_mixed_points(n: int = 200) -> list[CalibrationPoint]:
    """Create a larger mixed dataset with varying confidence ranges."""
    rng = np.random.RandomState(42)
    points = []
    for _ in range(n):
        raw = rng.uniform(0.1, 0.99)
        # Sigmoid-like accuracy curve
        accuracy = 1.0 / (1.0 + np.exp(-10 * (raw - 0.5)))
        is_correct = rng.random() < accuracy
        points.append(CalibrationPoint(
            raw_confidence=raw,
            is_correct=bool(is_correct),
            field_name="test_field",
            document_type="test_doc",
        ))
    return points


# ──────────────────────────────────────────────────────────────────
# CalibrationPoint Tests
# ──────────────────────────────────────────────────────────────────


class TestCalibrationPoint:
    def test_basic_creation(self):
        p = CalibrationPoint(raw_confidence=0.9, is_correct=True)
        assert p.raw_confidence == 0.9
        assert p.is_correct is True
        assert p.field_name == ""
        assert p.document_type == ""

    def test_with_metadata(self):
        p = CalibrationPoint(
            raw_confidence=0.5,
            is_correct=False,
            field_name="patient_name",
            document_type="cms1500",
        )
        assert p.field_name == "patient_name"
        assert p.document_type == "cms1500"

    def test_frozen(self):
        p = CalibrationPoint(raw_confidence=0.8, is_correct=True)
        with pytest.raises(AttributeError):
            p.raw_confidence = 0.5  # type: ignore


# ──────────────────────────────────────────────────────────────────
# CalibrationResult Tests
# ──────────────────────────────────────────────────────────────────


class TestCalibrationResult:
    def test_adjustment_computed(self):
        r = CalibrationResult(
            raw_confidence=0.9,
            calibrated_confidence=0.7,
            calibration_method="platt",
        )
        assert r.adjustment == pytest.approx(-0.2, abs=0.001)

    def test_to_dict(self):
        r = CalibrationResult(
            raw_confidence=0.85,
            calibrated_confidence=0.78,
            calibration_method="isotonic",
        )
        d = r.to_dict()
        assert "raw_confidence" in d
        assert "calibrated_confidence" in d
        assert "calibration_method" in d
        assert "adjustment" in d
        assert d["calibration_method"] == "isotonic"

    def test_positive_adjustment(self):
        r = CalibrationResult(
            raw_confidence=0.5,
            calibrated_confidence=0.65,
            calibration_method="linear",
        )
        assert r.adjustment > 0


# ──────────────────────────────────────────────────────────────────
# CalibrationMetrics Tests
# ──────────────────────────────────────────────────────────────────


class TestCalibrationMetrics:
    def test_default_values(self):
        m = CalibrationMetrics()
        assert m.expected_calibration_error == 0.0
        assert m.brier_score == 0.0
        assert m.num_samples == 0

    def test_to_dict(self):
        m = CalibrationMetrics(
            expected_calibration_error=0.05,
            max_calibration_error=0.12,
            brier_score=0.15,
            num_samples=100,
        )
        d = m.to_dict()
        assert d["expected_calibration_error"] == 0.05
        assert d["num_samples"] == 100


# ──────────────────────────────────────────────────────────────────
# LinearCalibrator Tests
# ──────────────────────────────────────────────────────────────────


class TestLinearCalibrator:
    def test_always_fitted(self):
        cal = LinearCalibrator()
        assert cal.is_fitted is True
        assert cal.name == "linear"

    def test_default_conservative_bias(self):
        cal = LinearCalibrator()
        result = cal.calibrate(0.9)
        # Default slope=0.85, offset=0.05 → 0.85*0.9 + 0.05 = 0.815
        assert result.calibrated_confidence < 0.9
        assert result.calibration_method == "linear"

    def test_identity_calibration(self):
        cal = LinearCalibrator(slope=1.0, offset=0.0)
        result = cal.calibrate(0.75)
        assert result.calibrated_confidence == pytest.approx(0.75)

    def test_clamps_to_valid_range(self):
        cal = LinearCalibrator(slope=2.0, offset=0.0)
        result = cal.calibrate(0.8)
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_low_confidence_clamped(self):
        cal = LinearCalibrator(slope=1.0, offset=-0.5)
        result = cal.calibrate(0.3)
        assert result.calibrated_confidence >= 0.0

    def test_batch_calibration(self):
        cal = LinearCalibrator()
        results = cal.calibrate_batch({"field1": 0.9, "field2": 0.5})
        assert "field1" in results
        assert "field2" in results
        assert results["field1"].calibrated_confidence > results["field2"].calibrated_confidence

    def test_fit_with_data(self):
        cal = LinearCalibrator()
        points = _make_well_calibrated_points(20)
        cal.fit(points)
        # Should still be fitted after fit()
        assert cal.is_fitted


# ──────────────────────────────────────────────────────────────────
# PlattCalibrator Tests
# ──────────────────────────────────────────────────────────────────


class TestPlattCalibrator:
    def test_unfitted_returns_raw(self):
        cal = PlattCalibrator()
        assert cal.is_fitted is False
        result = cal.calibrate(0.8)
        assert result.calibrated_confidence == 0.8
        assert result.calibration_method == "platt_unfitted"

    def test_fit_with_sufficient_data(self):
        cal = PlattCalibrator()
        points = _make_overconfident_points(50)
        cal.fit(points)
        assert cal.is_fitted is True

    def test_insufficient_data_not_fitted(self):
        cal = PlattCalibrator()
        points = _make_overconfident_points(5)
        cal.fit(points)
        assert cal.is_fitted is False

    def test_overconfident_correction(self):
        cal = PlattCalibrator()
        points = _make_overconfident_points(50)
        cal.fit(points)

        # High raw confidence should be calibrated down
        result = cal.calibrate(0.9)
        assert result.calibrated_confidence < 0.9
        assert result.calibration_method == "platt"

    def test_output_bounded(self):
        cal = PlattCalibrator()
        points = _make_overconfident_points(50)
        cal.fit(points)

        for raw in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = cal.calibrate(raw)
            assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_single_class_not_fitted(self):
        """If all training data is correct (or all wrong), can't fit."""
        cal = PlattCalibrator()
        points = [CalibrationPoint(0.9, True) for _ in range(20)]
        cal.fit(points)
        assert cal.is_fitted is False


# ──────────────────────────────────────────────────────────────────
# IsotonicCalibrator Tests
# ──────────────────────────────────────────────────────────────────


class TestIsotonicCalibrator:
    def test_unfitted_returns_raw(self):
        cal = IsotonicCalibrator()
        assert cal.is_fitted is False
        result = cal.calibrate(0.8)
        assert result.calibrated_confidence == 0.8
        assert result.calibration_method == "isotonic_unfitted"

    def test_fit_with_sufficient_data(self):
        cal = IsotonicCalibrator()
        points = _make_mixed_points(200)
        cal.fit(points)
        assert cal.is_fitted is True

    def test_insufficient_data_not_fitted(self):
        cal = IsotonicCalibrator()
        points = _make_overconfident_points(10)
        cal.fit(points)
        assert cal.is_fitted is False

    def test_output_bounded(self):
        cal = IsotonicCalibrator()
        points = _make_mixed_points(200)
        cal.fit(points)

        for raw in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = cal.calibrate(raw)
            assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_monotonic_output(self):
        """Higher raw confidence should produce higher calibrated confidence."""
        cal = IsotonicCalibrator()
        points = _make_mixed_points(200)
        cal.fit(points)

        results = [cal.calibrate(r / 10).calibrated_confidence for r in range(1, 10)]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1] + 0.01  # Allow small float tolerance


# ──────────────────────────────────────────────────────────────────
# ConfidenceCalibrator (Unified) Tests
# ──────────────────────────────────────────────────────────────────


class TestConfidenceCalibrator:
    def test_default_uses_linear(self):
        cal = ConfidenceCalibrator()
        assert cal.active_method == "linear"
        assert cal.sample_count == 0

    def test_add_points(self):
        cal = ConfidenceCalibrator()
        cal.add_point(CalibrationPoint(0.9, True))
        assert cal.sample_count == 1

        cal.add_points([
            CalibrationPoint(0.8, False),
            CalibrationPoint(0.7, True),
        ])
        assert cal.sample_count == 3

    def test_fit_small_data_uses_linear(self):
        cal = ConfidenceCalibrator()
        cal.add_points(_make_overconfident_points(5))
        method = cal.fit()
        assert method == "linear"

    def test_fit_medium_data_uses_platt(self):
        cal = ConfidenceCalibrator()
        cal.add_points(_make_overconfident_points(50))
        method = cal.fit()
        assert method == "platt"

    def test_fit_large_data_uses_isotonic(self):
        cal = ConfidenceCalibrator()
        cal.add_points(_make_mixed_points(200))
        method = cal.fit()
        assert method == "isotonic"

    def test_calibrate_returns_result(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.85)
        assert isinstance(result, CalibrationResult)
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_calibrate_batch(self):
        cal = ConfidenceCalibrator()
        cal.add_points(_make_overconfident_points(50))
        cal.fit()

        results = cal.calibrate_batch({
            "patient_name": 0.95,
            "dob": 0.80,
            "charges": 0.60,
        })
        assert len(results) == 3
        assert all(isinstance(r, CalibrationResult) for r in results.values())

    def test_evaluate_empty(self):
        cal = ConfidenceCalibrator()
        metrics = cal.evaluate()
        assert metrics.num_samples == 0

    def test_evaluate_with_data(self):
        cal = ConfidenceCalibrator()
        points = _make_mixed_points(100)
        cal.add_points(points)
        cal.fit()

        metrics = cal.evaluate()
        assert metrics.num_samples == 100
        assert 0.0 <= metrics.expected_calibration_error <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0

    def test_evaluate_well_calibrated_has_low_ece(self):
        cal = ConfidenceCalibrator()
        points = _make_well_calibrated_points(200)
        cal.add_points(points)
        cal.fit()

        metrics = cal.evaluate()
        # ECE should be reasonable for well-calibrated data
        assert metrics.expected_calibration_error < 0.3


# ──────────────────────────────────────────────────────────────────
# Persistence Tests
# ──────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal_points.json"

            # Save
            cal1 = ConfidenceCalibrator(storage_path=path)
            cal1.add_points(_make_overconfident_points(20))
            cal1.fit()

            assert path.exists()

            # Load in new instance
            cal2 = ConfidenceCalibrator(storage_path=path)
            assert cal2.sample_count == 20

    def test_no_path_no_save(self):
        cal = ConfidenceCalibrator()
        cal.add_points(_make_overconfident_points(20))
        cal.fit()
        # Should not raise — just silently skips save

    def test_load_nonexistent_path(self):
        path = Path("/nonexistent/path/cal.json")
        cal = ConfidenceCalibrator(storage_path=path)
        assert cal.sample_count == 0

    def test_persistence_across_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal_points.json"

            cal = ConfidenceCalibrator(storage_path=path)
            cal.add_points(_make_overconfident_points(30))
            method1 = cal.fit()

            # Add more points and refit
            cal.add_points(_make_overconfident_points(30))
            method2 = cal.fit()

            # Reload should have all 60 points
            cal2 = ConfidenceCalibrator(storage_path=path)
            assert cal2.sample_count == 60


# ──────────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_calibrate_zero(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.0)
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_calibrate_one(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(1.0)
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_calibrate_negative_clamped(self):
        cal = LinearCalibrator(slope=1.0, offset=-0.5)
        result = cal.calibrate(0.1)
        assert result.calibrated_confidence >= 0.0

    def test_empty_batch(self):
        cal = ConfidenceCalibrator()
        results = cal.calibrate_batch({})
        assert results == {}

    def test_isotonic_threshold_constant(self):
        assert ConfidenceCalibrator.ISOTONIC_THRESHOLD == 100
        assert ConfidenceCalibrator.PLATT_THRESHOLD == 10
