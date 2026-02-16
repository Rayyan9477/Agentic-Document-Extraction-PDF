"""
Regression detection for extraction accuracy.

Compares current benchmark results against stored baselines to
detect accuracy regressions at the field, document, and dataset level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.evaluation.benchmark import BenchmarkResult

logger = structlog.get_logger(__name__)


class RegressionSeverity:
    """Severity levels for regressions."""

    CRITICAL = "critical"  # F1 dropped > 10%
    WARNING = "warning"  # F1 dropped > 2%
    INFO = "info"  # Minor changes


@dataclass(slots=True)
class FieldRegression:
    """A regression detected on a specific field."""

    field_name: str
    baseline_f1: float
    current_f1: float
    delta: float
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "baseline_f1": round(self.baseline_f1, 4),
            "current_f1": round(self.current_f1, 4),
            "delta": round(self.delta, 4),
            "severity": self.severity,
        }


@dataclass(slots=True)
class RegressionReport:
    """Complete regression analysis report."""

    baseline_run_id: str
    current_run_id: str
    dataset_name: str
    has_regression: bool = False
    overall_f1_delta: float = 0.0
    overall_severity: str = RegressionSeverity.INFO
    field_regressions: list[FieldRegression] = field(default_factory=list)
    field_improvements: list[FieldRegression] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def regression_count(self) -> int:
        return len(self.field_regressions)

    @property
    def improvement_count(self) -> int:
        return len(self.field_improvements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "dataset_name": self.dataset_name,
            "has_regression": self.has_regression,
            "overall_f1_delta": round(self.overall_f1_delta, 4),
            "overall_severity": self.overall_severity,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "field_regressions": [r.to_dict() for r in self.field_regressions],
            "field_improvements": [r.to_dict() for r in self.field_improvements],
            "created_at": self.created_at,
        }

    @property
    def summary(self) -> str:
        if not self.has_regression:
            return f"No regressions detected (F1 delta: {self.overall_f1_delta:+.4f})"
        return (
            f"{self.regression_count} field regression(s) detected "
            f"[{self.overall_severity}] (F1 delta: {self.overall_f1_delta:+.4f})"
        )


# ──────────────────────────────────────────────────────────────────
# Regression Detector
# ──────────────────────────────────────────────────────────────────


class RegressionDetector:
    """
    Detects accuracy regressions between benchmark runs.

    Usage:
        detector = RegressionDetector(warning_threshold=0.02, critical_threshold=0.10)
        report = detector.compare(baseline_result, current_result)
        if report.has_regression:
            print(report.summary)
    """

    def __init__(
        self,
        warning_threshold: float = 0.02,
        critical_threshold: float = 0.10,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold

    def _classify_severity(self, delta: float) -> str:
        """Classify the severity of a negative F1 delta."""
        abs_delta = abs(delta)
        if abs_delta >= self._critical_threshold:
            return RegressionSeverity.CRITICAL
        if abs_delta >= self._warning_threshold:
            return RegressionSeverity.WARNING
        return RegressionSeverity.INFO

    def compare(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult,
    ) -> RegressionReport:
        """
        Compare current results against baseline and detect regressions.

        Args:
            baseline: Previous benchmark result (the baseline).
            current: Current benchmark result to check.

        Returns:
            RegressionReport with detailed field-level analysis.
        """
        ba = baseline.aggregate
        ca = current.aggregate
        overall_delta = ca.micro_f1 - ba.micro_f1

        report = RegressionReport(
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            dataset_name=current.dataset_name,
            overall_f1_delta=overall_delta,
        )

        # Per-field analysis
        baseline_f1 = ba.per_field_f1()
        current_f1 = ca.per_field_f1()
        all_fields = set(baseline_f1.keys()) | set(current_f1.keys())

        for field_name in sorted(all_fields):
            b_f1 = baseline_f1.get(field_name, 0.0)
            c_f1 = current_f1.get(field_name, 0.0)
            delta = c_f1 - b_f1

            if delta < -self._warning_threshold:
                severity = self._classify_severity(delta)
                report.field_regressions.append(
                    FieldRegression(
                        field_name=field_name,
                        baseline_f1=b_f1,
                        current_f1=c_f1,
                        delta=delta,
                        severity=severity,
                    )
                )
            elif delta > self._warning_threshold:
                report.field_improvements.append(
                    FieldRegression(
                        field_name=field_name,
                        baseline_f1=b_f1,
                        current_f1=c_f1,
                        delta=delta,
                        severity=RegressionSeverity.INFO,
                    )
                )

        # Determine overall regression status
        if report.field_regressions:
            report.has_regression = True
            severities = [r.severity for r in report.field_regressions]
            if RegressionSeverity.CRITICAL in severities:
                report.overall_severity = RegressionSeverity.CRITICAL
            elif RegressionSeverity.WARNING in severities:
                report.overall_severity = RegressionSeverity.WARNING
            else:
                report.overall_severity = RegressionSeverity.INFO

        return report


# ──────────────────────────────────────────────────────────────────
# Baseline Storage
# ──────────────────────────────────────────────────────────────────


def save_baseline(result: BenchmarkResult, path: str | Path) -> Path:
    """Save a benchmark result as a baseline for future regression checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("baseline_saved", path=str(path), run_id=result.run_id)
    return path


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load a baseline from disk (returns raw dict for flexibility)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
