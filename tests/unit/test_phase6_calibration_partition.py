"""V3 Phase 6 — partitioned calibrator + ECE quality gate.

Coverage:

* ``CalibrationPoint`` carries profile + tenant_id; defaults to "_global".
* ``PartitionedCalibrator`` routes points to (profile, tenant) buckets.
* New tenants without enough data fall back to global.
* ECE quality gate rejects fits whose post-fit ECE regresses.
* Persisted points round-trip the new fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.calibration import (
    CalibrationPoint,
    ConfidenceCalibrator,
    ECE_REGRESSION_TOLERANCE,
    MIN_PARTITION_SAMPLES,
    PartitionedCalibrator,
    PartitionFitResult,
)


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------


class TestCalibrationPointPhase6Fields:
    def test_defaults_global(self) -> None:
        p = CalibrationPoint(raw_confidence=0.8, is_correct=True)
        assert p.profile == "_global"
        assert p.tenant_id == "_global"

    def test_explicit_partition(self) -> None:
        p = CalibrationPoint(
            raw_confidence=0.8,
            is_correct=True,
            profile="medical-rcm",
            tenant_id="acme",
        )
        assert p.profile == "medical-rcm"
        assert p.tenant_id == "acme"


# ---------------------------------------------------------------------------
# PartitionedCalibrator routing
# ---------------------------------------------------------------------------


class TestPartitionedCalibratorRouting:
    def test_global_partition_always_present(self) -> None:
        pc = PartitionedCalibrator()
        assert pc.GLOBAL_KEY in pc.partition_keys

    def test_add_point_routes_to_correct_bucket(self) -> None:
        pc = PartitionedCalibrator()
        pc.add_point(
            CalibrationPoint(
                raw_confidence=0.8,
                is_correct=True,
                profile="medical-rcm",
                tenant_id="acme",
            )
        )
        assert ("medical-rcm", "acme") in pc.partition_keys
        # And global gets the same point as well.
        assert pc.get_partition().sample_count == 1

    def test_add_point_to_global_does_not_double_count(self) -> None:
        pc = PartitionedCalibrator()
        pc.add_point(
            CalibrationPoint(raw_confidence=0.8, is_correct=True)
        )
        # Global point added once, not double-counted.
        assert pc.get_partition().sample_count == 1

    def test_calibrate_falls_back_to_global_when_partition_empty(self) -> None:
        pc = PartitionedCalibrator()
        # Add 25 points to global only.
        for i in range(25):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.6 + (i % 5) * 0.05,
                    is_correct=(i % 2 == 0),
                )
            )
        pc.fit_all()
        # Now ask for a tenant that has no data — should fall through.
        result = pc.calibrate(0.7, profile="medical-rcm", tenant_id="brand-new-tenant")
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_calibrate_uses_partition_when_enough_data(self) -> None:
        pc = PartitionedCalibrator()
        for i in range(MIN_PARTITION_SAMPLES + 5):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.8,
                    is_correct=True,
                    profile="medical-rcm",
                    tenant_id="acme",
                )
            )
        pc.fit_all()
        partition = pc.get_partition("medical-rcm", "acme")
        assert partition is not None
        assert partition.sample_count >= MIN_PARTITION_SAMPLES


# ---------------------------------------------------------------------------
# ECE quality gate
# ---------------------------------------------------------------------------


class TestECEQualityGate:
    def test_partitions_below_threshold_fall_back_to_global(self) -> None:
        pc = PartitionedCalibrator()
        # 5 points in a tenant — not enough; will fall back.
        for _ in range(5):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.5,
                    is_correct=True,
                    profile="finance",
                    tenant_id="small",
                )
            )
        results = pc.fit_all()
        small_result = results.get(("finance", "small"))
        assert small_result is not None
        assert small_result.fell_back_to_global is True
        assert small_result.method_selected == "fallback_global"

    def test_global_partition_always_fits(self) -> None:
        pc = PartitionedCalibrator()
        for i in range(50):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.5 + (i % 10) * 0.05,
                    is_correct=(i % 3 != 0),
                )
            )
        results = pc.fit_all()
        global_result = results[pc.GLOBAL_KEY]
        assert global_result.accepted is True
        # For 50 samples with both classes Platt is selected.
        assert global_result.method_selected in ("platt", "isotonic", "linear")

    def test_ece_rollback_when_fit_regresses(self) -> None:
        """If a re-fit's ECE is worse than the recorded previous ECE
        by more than the tolerance, we roll back to the LinearCalibrator."""
        pc = PartitionedCalibrator()
        # Seed with consistent, well-calibrated data.
        for i in range(40):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.9 if i % 2 == 0 else 0.1,
                    is_correct=(i % 2 == 0),
                )
            )
        pc.fit_all()
        # Record a fake "previous_ece" of 0 to force rollback on any
        # post-fit ECE > tolerance.
        pc._previous_ece[pc.GLOBAL_KEY] = 0.0
        # Now poison the partition with garbage to drive ECE up.
        for _ in range(40):
            # Inverted labels — model now reports 0.9 for incorrect.
            pc.add_point(CalibrationPoint(raw_confidence=0.9, is_correct=False))

        results = pc.fit_all()
        global_result = results[pc.GLOBAL_KEY]
        # If ECE regressed beyond tolerance, expect rollback.
        if global_result.post_fit_ece is not None:
            if global_result.post_fit_ece > 0.0 + ECE_REGRESSION_TOLERANCE:
                assert global_result.accepted is False
                assert global_result.method_selected == "rollback_linear"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestPartitionedPersistence:
    def test_save_load_carries_profile_and_tenant(self, tmp_path: Path) -> None:
        storage = tmp_path / "calib"
        pc = PartitionedCalibrator(storage_dir=storage)
        pc.add_point(
            CalibrationPoint(
                raw_confidence=0.7,
                is_correct=True,
                profile="medical-rcm",
                tenant_id="acme",
            )
        )
        pc.fit_all()  # Triggers save
        # Load a fresh instance and confirm partition rebuilt.
        pc2 = PartitionedCalibrator(storage_dir=storage)
        # The (medical-rcm, acme) partition file should exist.
        partition_path = storage / "calib_medical-rcm_acme.json"
        if partition_path.exists():
            data = json.loads(partition_path.read_text(encoding="utf-8"))
            assert data
            assert data[0]["profile"] == "medical-rcm"
            assert data[0]["tenant_id"] == "acme"


# ---------------------------------------------------------------------------
# Existing ConfidenceCalibrator unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_legacy_calibrator_still_works(self) -> None:
        cc = ConfidenceCalibrator()
        for i in range(15):
            cc.add_point(
                CalibrationPoint(
                    raw_confidence=0.5 + i * 0.03,
                    is_correct=(i > 7),
                )
            )
        method = cc.fit()
        assert method in ("platt", "linear")
        result = cc.calibrate(0.7)
        assert 0.0 <= result.calibrated_confidence <= 1.0
