"""
Confidence calibration for extraction scores.

VLM-reported confidence scores are often poorly calibrated —
a model reporting 90% confidence may only be correct 75% of the time.
This module maps raw scores to empirically accurate probabilities.

Supports:
- Platt scaling (logistic regression — good for small calibration sets)
- Isotonic regression (non-parametric — good for 100+ samples)
- Fallback linear scaling (no training data needed)

Integration: called after ConfidenceScorer.calculate() in the validator agent.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_logger


logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """A single calibration training point.

    V3 Phase 6: ``profile`` and ``tenant_id`` are now first-class
    so the calibration store can partition by both axes. Older
    persisted points without these fields default to
    ``"_global"`` / ``"_global"`` and continue to load cleanly.
    """

    raw_confidence: float  # VLM-reported confidence
    is_correct: bool  # Ground truth: was the extraction actually correct?
    field_name: str = ""
    document_type: str = ""
    profile: str = "_global"
    tenant_id: str = "_global"


@dataclass(slots=True)
class CalibrationResult:
    """Result of calibrating a confidence score."""

    raw_confidence: float
    calibrated_confidence: float
    calibration_method: str
    adjustment: float = 0.0  # calibrated - raw

    def __post_init__(self) -> None:
        self.adjustment = self.calibrated_confidence - self.raw_confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "calibration_method": self.calibration_method,
            "adjustment": round(self.adjustment, 4),
        }


@dataclass(slots=True)
class CalibrationMetrics:
    """Metrics assessing calibration quality."""

    expected_calibration_error: float = 0.0  # ECE — lower is better
    max_calibration_error: float = 0.0  # MCE
    brier_score: float = 0.0  # Brier score — lower is better
    num_samples: int = 0
    num_bins: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "max_calibration_error": round(self.max_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
        }


# ──────────────────────────────────────────────────────────────────
# Abstract calibrator
# ──────────────────────────────────────────────────────────────────


class BaseCalibrator(ABC):
    """Abstract base for confidence calibrators."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, points: list[CalibrationPoint]) -> None:
        """Train the calibrator on historical data."""

    @abstractmethod
    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate a single raw confidence score."""

    def calibrate_batch(
        self, raw_scores: dict[str, float],
    ) -> dict[str, CalibrationResult]:
        """Calibrate multiple scores at once."""
        return {
            name: self.calibrate(score)
            for name, score in raw_scores.items()
        }


# ──────────────────────────────────────────────────────────────────
# Platt scaling (logistic regression)
# ──────────────────────────────────────────────────────────────────


class PlattCalibrator(BaseCalibrator):
    """
    Platt scaling calibrator using logistic regression.

    Maps raw confidence to calibrated probability via sigmoid:
        P(correct | score) = 1 / (1 + exp(A * score + B))

    Good for: small calibration sets (20-100 samples), smooth monotonic mapping.
    """

    MIN_SAMPLES = 10

    def __init__(self) -> None:
        super().__init__(name="platt")
        self._model = None

    def fit(self, points: list[CalibrationPoint]) -> None:
        """Fit Platt scaling on calibration data."""
        if len(points) < self.MIN_SAMPLES:
            logger.warning(
                "platt_insufficient_data",
                samples=len(points),
                min_required=self.MIN_SAMPLES,
            )
            return

        from sklearn.linear_model import LogisticRegression

        X = np.array([p.raw_confidence for p in points]).reshape(-1, 1)
        y = np.array([int(p.is_correct) for p in points])

        # Check if we have both classes
        if len(set(y)) < 2:
            logger.warning("platt_single_class", unique_labels=list(set(y)))
            return

        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._model.fit(X, y)
        self._fitted = True

        logger.info(
            "platt_fitted",
            samples=len(points),
            coef=float(self._model.coef_[0][0]),
            intercept=float(self._model.intercept_[0]),
        )

    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate using the fitted logistic model."""
        if not self._fitted or self._model is None:
            return CalibrationResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_method="platt_unfitted",
            )

        X = np.array([[raw_confidence]])
        calibrated = float(self._model.predict_proba(X)[0, 1])
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_method="platt",
        )


# ──────────────────────────────────────────────────────────────────
# Isotonic regression
# ──────────────────────────────────────────────────────────────────


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression calibrator.

    Non-parametric monotonic mapping — more flexible than Platt scaling
    but needs more data.

    Good for: larger calibration sets (100+ samples), non-linear calibration curves.
    """

    MIN_SAMPLES = 20

    def __init__(self) -> None:
        super().__init__(name="isotonic")
        self._model = None

    def fit(self, points: list[CalibrationPoint]) -> None:
        """Fit isotonic regression on calibration data."""
        if len(points) < self.MIN_SAMPLES:
            logger.warning(
                "isotonic_insufficient_data",
                samples=len(points),
                min_required=self.MIN_SAMPLES,
            )
            return

        from sklearn.isotonic import IsotonicRegression

        X = np.array([p.raw_confidence for p in points])
        y = np.array([float(p.is_correct) for p in points])

        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._model.fit(X, y)
        self._fitted = True

        logger.info("isotonic_fitted", samples=len(points))

    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate using the fitted isotonic model."""
        if not self._fitted or self._model is None:
            return CalibrationResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_method="isotonic_unfitted",
            )

        calibrated = float(self._model.predict(np.array([raw_confidence]))[0])
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_method="isotonic",
        )


# ──────────────────────────────────────────────────────────────────
# Linear fallback (no training data)
# ──────────────────────────────────────────────────────────────────


class LinearCalibrator(BaseCalibrator):
    """
    Simple linear calibrator with configurable slope/offset.

    Always fitted — no training data required. Applies a simple
    affine transform: calibrated = slope * raw + offset, clamped to [0, 1].

    Default: slight conservative bias (models tend to be overconfident).
    """

    def __init__(self, slope: float = 0.85, offset: float = 0.05) -> None:
        super().__init__(name="linear")
        self._slope = slope
        self._offset = offset
        self._fitted = True  # always ready

    def fit(self, points: list[CalibrationPoint]) -> None:
        """Linear calibrator doesn't need fitting, but accepts data to estimate."""
        if len(points) < 5:
            return

        raw = np.array([p.raw_confidence for p in points])
        correct = np.array([float(p.is_correct) for p in points])

        # Simple linear regression
        if np.std(raw) > 0:
            correlation = np.corrcoef(raw, correct)
            if correlation.shape == (2, 2):
                r = correlation[0, 1]
                self._slope = r * np.std(correct) / np.std(raw)
                self._offset = np.mean(correct) - self._slope * np.mean(raw)
                logger.info(
                    "linear_fitted",
                    slope=round(self._slope, 3),
                    offset=round(self._offset, 3),
                )

    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Apply linear transform."""
        calibrated = self._slope * raw_confidence + self._offset
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_method="linear",
        )


# ──────────────────────────────────────────────────────────────────
# Unified calibrator with auto-selection
# ──────────────────────────────────────────────────────────────────


class ConfidenceCalibrator:
    """
    Unified confidence calibrator with automatic method selection.

    Picks the best calibration method based on available training data:
    - 100+ samples → isotonic regression
    - 10-99 samples → Platt scaling
    - <10 samples → linear fallback

    Usage:
        calibrator = ConfidenceCalibrator()
        calibrator.add_point(CalibrationPoint(0.9, True))
        calibrator.add_point(CalibrationPoint(0.8, False))
        ...
        calibrator.fit()

        result = calibrator.calibrate(0.92)
        print(f"Calibrated: {result.calibrated_confidence}")
    """

    ISOTONIC_THRESHOLD = 100
    PLATT_THRESHOLD = 10

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._points: list[CalibrationPoint] = []
        self._platt = PlattCalibrator()
        self._isotonic = IsotonicCalibrator()
        self._linear = LinearCalibrator()
        self._active: BaseCalibrator = self._linear
        self._storage_path = Path(storage_path) if storage_path else None

        # Load stored points if available
        if self._storage_path and self._storage_path.exists():
            self._load_points()

    @property
    def active_method(self) -> str:
        """Name of the currently active calibration method."""
        return self._active.name

    @property
    def sample_count(self) -> int:
        """Number of calibration points collected."""
        return len(self._points)

    def add_point(self, point: CalibrationPoint) -> None:
        """Add a calibration training point."""
        self._points.append(point)

    def add_points(self, points: list[CalibrationPoint]) -> None:
        """Add multiple calibration points."""
        self._points.extend(points)

    def fit(self) -> str:
        """
        Fit the best available calibrator based on data size.

        Returns:
            Name of the method selected.
        """
        n = len(self._points)

        if n >= self.ISOTONIC_THRESHOLD:
            self._isotonic.fit(self._points)
            if self._isotonic.is_fitted:
                self._active = self._isotonic
                self._save_points()
                return "isotonic"

        if n >= self.PLATT_THRESHOLD:
            self._platt.fit(self._points)
            if self._platt.is_fitted:
                self._active = self._platt
                self._save_points()
                return "platt"

        # Fall back to linear (with optional fitting if we have data)
        if n >= 5:
            self._linear.fit(self._points)
        self._active = self._linear
        self._save_points()
        return "linear"

    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate a single raw confidence score."""
        return self._active.calibrate(raw_confidence)

    def calibrate_batch(
        self, raw_scores: dict[str, float],
    ) -> dict[str, CalibrationResult]:
        """Calibrate multiple scores at once."""
        return self._active.calibrate_batch(raw_scores)

    def evaluate(self, test_points: list[CalibrationPoint] | None = None) -> CalibrationMetrics:
        """
        Evaluate calibration quality using ECE, MCE, and Brier score.

        Args:
            test_points: Points to evaluate on (defaults to training data).

        Returns:
            CalibrationMetrics with quality measures.
        """
        points = test_points or self._points
        if not points:
            return CalibrationMetrics()

        raw_scores = np.array([p.raw_confidence for p in points])
        labels = np.array([float(p.is_correct) for p in points])

        # Get calibrated scores
        calibrated_scores = np.array([
            self.calibrate(float(r)).calibrated_confidence for r in raw_scores
        ])

        # Brier score
        brier = float(np.mean((calibrated_scores - labels) ** 2))

        # ECE and MCE
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (calibrated_scores > bin_boundaries[i]) & (calibrated_scores <= bin_boundaries[i + 1])
            bin_count = mask.sum()

            if bin_count > 0:
                bin_acc = labels[mask].mean()
                bin_conf = calibrated_scores[mask].mean()
                bin_error = abs(bin_acc - bin_conf)

                ece += (bin_count / len(points)) * bin_error
                mce = max(mce, bin_error)

        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            max_calibration_error=float(mce),
            brier_score=float(brier),
            num_samples=len(points),
            num_bins=n_bins,
        )

    def _save_points(self) -> None:
        """Persist calibration points to disk."""
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "raw_confidence": p.raw_confidence,
                "is_correct": p.is_correct,
                "field_name": p.field_name,
                "document_type": p.document_type,
                "profile": p.profile,
                "tenant_id": p.tenant_id,
            }
            for p in self._points
        ]
        self._storage_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def _load_points(self) -> None:
        """Load calibration points from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            self._points = [
                CalibrationPoint(
                    raw_confidence=d["raw_confidence"],
                    is_correct=d["is_correct"],
                    field_name=d.get("field_name", ""),
                    document_type=d.get("document_type", ""),
                    profile=d.get("profile", "_global"),
                    tenant_id=d.get("tenant_id", "_global"),
                )
                for d in data
            ]
            logger.info("calibration_points_loaded", count=len(self._points))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("calibration_load_failed", error=str(e))


# ──────────────────────────────────────────────────────────────────
# V3 Phase 6 — Partitioned calibrator (per-(profile, tenant_id))
# ──────────────────────────────────────────────────────────────────


# Quality-gate threshold: if a fresh fit's ECE on its own training
# data is *worse* than the previous fit's ECE by more than this
# margin, reject the new model and keep the previous one. The
# rationale lives in the EXECUTION_PLAN — calibration regressions
# are silent killers, so we bias toward stability.
ECE_REGRESSION_TOLERANCE = 0.02

# Below this sample count we don't even attempt a per-partition fit;
# the partition falls back to the global table. Tuned to the lowest
# Platt threshold above + some headroom so we don't fit on noise.
MIN_PARTITION_SAMPLES = 20


@dataclass(slots=True)
class PartitionFitResult:
    """Outcome of a single partition's fit attempt.

    Used by ``PartitionedCalibrator.fit_all()`` to give a structured
    diagnostic report — operators want to know which partitions
    rolled back, which fell through to the global table, and which
    succeeded fresh.
    """

    partition_key: tuple[str, str]
    samples: int
    method_selected: str
    accepted: bool
    pre_fit_ece: float | None = None
    post_fit_ece: float | None = None
    rollback_reason: str | None = None
    fell_back_to_global: bool = False


class PartitionedCalibrator:
    """Per-(profile, tenant_id) calibration with a global fallback.

    The store keeps one ``ConfidenceCalibrator`` per partition plus a
    ``"_global"`` calibrator that's the union of every point.
    ``calibrate(raw, profile=, tenant_id=)`` looks up the partition;
    if it has < ``MIN_PARTITION_SAMPLES`` it transparently delegates
    to the global calibrator. This means new tenants don't need their
    own calibration table on day 1 — they ride on the global table
    until they accumulate enough data.

    Quality gate: ``fit_all()`` re-fits every partition that has
    enough data, but rejects fits whose ECE on the training set is
    worse than the previous fit's ECE by more than
    ``ECE_REGRESSION_TOLERANCE``. The previous calibrator stays in
    place; the new fit is logged as a rollback. This protects
    against a noisy week's data poisoning a long-running model.

    The class is **additive**: it does NOT replace
    ``ConfidenceCalibrator``. Callers that don't care about
    partitioning keep using the original API; callers that do
    construct ``PartitionedCalibrator`` and pass profile/tenant
    explicitly.
    """

    GLOBAL_KEY: tuple[str, str] = ("_global", "_global")

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._partitions: dict[tuple[str, str], ConfidenceCalibrator] = {}
        self._previous_ece: dict[tuple[str, str], float] = {}
        self._last_fit_summary: dict[tuple[str, str], PartitionFitResult] = {}

        # Always create the global partition.
        self._partitions[self.GLOBAL_KEY] = ConfidenceCalibrator(
            storage_path=self._global_storage_path(),
        )

    # ----- Public API -----------------------------------------------

    def add_point(self, point: CalibrationPoint) -> None:
        """Route a point to its (profile, tenant) partition + global."""
        key = (point.profile or "_global", point.tenant_id or "_global")
        self._get_or_create(key).add_point(point)
        # Always also count it toward the global table so a brand-new
        # partition has something to ride on.
        if key != self.GLOBAL_KEY:
            self._partitions[self.GLOBAL_KEY].add_point(point)

    def add_points(self, points: list[CalibrationPoint]) -> None:
        for p in points:
            self.add_point(p)

    def calibrate(
        self,
        raw_confidence: float,
        *,
        profile: str = "_global",
        tenant_id: str = "_global",
    ) -> CalibrationResult:
        """Calibrate using the most-specific partition with enough data."""
        key = (profile or "_global", tenant_id or "_global")
        partition = self._partitions.get(key)
        if (
            partition is None
            or partition.sample_count < MIN_PARTITION_SAMPLES
        ):
            # Fall through to global.
            return self._partitions[self.GLOBAL_KEY].calibrate(raw_confidence)
        return partition.calibrate(raw_confidence)

    def fit_all(self) -> dict[tuple[str, str], PartitionFitResult]:
        """Re-fit every partition with the ECE quality gate.

        Returns a per-partition fit-result map. Use this for
        operational dashboards / weekly fit-result PRs.
        """
        results: dict[tuple[str, str], PartitionFitResult] = {}

        for key, partition in list(self._partitions.items()):
            samples = partition.sample_count

            if samples < MIN_PARTITION_SAMPLES and key != self.GLOBAL_KEY:
                # Per-partition still too small. Don't fit; mark for
                # global fallback. Drop the partition so we re-create
                # it on the next ``add_point`` (memory hygiene).
                results[key] = PartitionFitResult(
                    partition_key=key,
                    samples=samples,
                    method_selected="fallback_global",
                    accepted=True,
                    fell_back_to_global=True,
                )
                continue

            # Snapshot the pre-fit ECE on the current calibrator BEFORE
            # we re-fit — that's what we compare against post-fit.
            pre_metrics = partition.evaluate()
            pre_ece = pre_metrics.expected_calibration_error
            # V3 Phase 8 — first-fit detection. Skip the regression
            # gate when this partition has never been fitted before:
            # the ``pre_ece`` of the unfitted LinearCalibrator is a
            # baseline noise floor, not a previous calibration result.
            # Comparing the new fit against it spuriously rejects
            # legitimate first fits whose post-fit ECE is naturally
            # noisier than the constant linear fallback.
            is_first_fit = key not in self._previous_ece
            previous_ece = self._previous_ece.get(key)

            method = partition.fit()
            post_metrics = partition.evaluate()
            post_ece = post_metrics.expected_calibration_error

            # Quality gate: if the new fit is *worse* than what we
            # had before by more than the tolerance, we'd be making
            # the system less calibrated, not more. Reject.
            # First-fit always accepts unconditionally.
            if (
                not is_first_fit
                and previous_ece is not None
                and post_ece > previous_ece + ECE_REGRESSION_TOLERANCE
            ):
                # Rollback: re-instantiate the calibrator and re-fit
                # on the exact same point set using the LinearCalibrator
                # which is always-fitted and stable.
                logger.warning(
                    "calibration_rollback",
                    partition=str(key),
                    pre_ece=round(pre_ece, 4),
                    post_ece=round(post_ece, 4),
                    previous_ece=(
                        round(previous_ece, 4) if previous_ece is not None else None
                    ),
                    tolerance=ECE_REGRESSION_TOLERANCE,
                )
                # Replace with a fresh linear-only calibrator so we
                # don't carry the bad isotonic/Platt weights.
                fresh = ConfidenceCalibrator(
                    storage_path=self._partition_storage_path(key)
                )
                fresh.add_points(list(partition._points))  # noqa: SLF001
                # Force linear (no fit attempt above PLATT_THRESHOLD).
                fresh._active = fresh._linear  # noqa: SLF001
                self._partitions[key] = fresh

                results[key] = PartitionFitResult(
                    partition_key=key,
                    samples=samples,
                    method_selected="rollback_linear",
                    accepted=False,
                    pre_fit_ece=pre_ece,
                    post_fit_ece=post_ece,
                    rollback_reason=(
                        f"post-fit ECE {post_ece:.4f} > "
                        f"previous {previous_ece if previous_ece is not None else 0.0:.4f} + "
                        f"tolerance {ECE_REGRESSION_TOLERANCE}"
                    ),
                )
                continue

            # Accepted fit. Record the new previous_ece.
            self._previous_ece[key] = post_ece
            results[key] = PartitionFitResult(
                partition_key=key,
                samples=samples,
                method_selected=method,
                accepted=True,
                pre_fit_ece=pre_ece,
                post_fit_ece=post_ece,
            )
            logger.info(
                "calibration_fit_accepted",
                partition=str(key),
                method=method,
                samples=samples,
                pre_ece=round(pre_ece, 4),
                post_ece=round(post_ece, 4),
            )

        self._last_fit_summary = results
        return results

    @property
    def last_fit_summary(self) -> dict[tuple[str, str], PartitionFitResult]:
        return dict(self._last_fit_summary)

    @property
    def partition_keys(self) -> list[tuple[str, str]]:
        return list(self._partitions.keys())

    def get_partition(
        self,
        profile: str = "_global",
        tenant_id: str = "_global",
    ) -> ConfidenceCalibrator | None:
        """Direct access to a partition (tests / diagnostics)."""
        return self._partitions.get(
            (profile or "_global", tenant_id or "_global")
        )

    # ----- Internals ------------------------------------------------

    def _get_or_create(self, key: tuple[str, str]) -> ConfidenceCalibrator:
        partition = self._partitions.get(key)
        if partition is None:
            partition = ConfidenceCalibrator(
                storage_path=self._partition_storage_path(key)
            )
            self._partitions[key] = partition
        return partition

    def _partition_storage_path(self, key: tuple[str, str]) -> Path | None:
        if self._storage_dir is None:
            return None
        # File-system-safe directory name. Profile + tenant are
        # already constrained by config, but escape just in case.
        profile_safe = key[0].replace("/", "_").replace("\\", "_")
        tenant_safe = key[1].replace("/", "_").replace("\\", "_")
        return (
            self._storage_dir
            / f"calib_{profile_safe}_{tenant_safe}.json"
        )

    def _global_storage_path(self) -> Path | None:
        return self._partition_storage_path(self.GLOBAL_KEY)
