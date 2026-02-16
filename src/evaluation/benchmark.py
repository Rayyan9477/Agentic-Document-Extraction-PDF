"""
Benchmark runner for document extraction evaluation.

Orchestrates running extractions against a golden dataset,
collecting metrics, and producing comparison reports.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

from src.evaluation.golden_dataset import GoldenDataset, GoldenSample
from src.evaluation.metrics import (
    AggregateMetrics,
    DocumentMetrics,
    MatchLevel,
    evaluate_document,
)

logger = structlog.get_logger(__name__)


# Type alias for an extraction function.
# Takes (sample_id, schema_name, source_file, metadata) → extracted fields dict.
ExtractorFn = Callable[[str, str, str, dict[str, Any]], dict[str, Any]]


class BenchmarkStatus(str, Enum):
    """Status of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str = "default"
    match_level: MatchLevel = MatchLevel.NORMALIZED
    field_match_levels: dict[str, MatchLevel] = field(default_factory=dict)
    filter_tags: list[str] = field(default_factory=list)
    filter_types: list[str] = field(default_factory=list)
    max_samples: int | None = None
    fail_on_regression: bool = False
    baseline_f1: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "match_level": self.match_level.value,
            "field_match_levels": {
                k: v.value for k, v in self.field_match_levels.items()
            },
            "filter_tags": self.filter_tags,
            "filter_types": self.filter_types,
            "max_samples": self.max_samples,
            "fail_on_regression": self.fail_on_regression,
            "baseline_f1": self.baseline_f1,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""

    run_id: str
    config: BenchmarkConfig
    dataset_name: str
    dataset_version: str
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    aggregate: AggregateMetrics = field(default_factory=AggregateMetrics)
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    errors: list[dict[str, str]] = field(default_factory=list)
    regression_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == BenchmarkStatus.COMPLETED and not self.regression_detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "status": self.status.value,
            "aggregate": self.aggregate.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors,
            "regression_detected": self.regression_detected,
            "success": self.success,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────
# Benchmark Runner
# ──────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """
    Runs extraction benchmarks against a golden dataset.

    Usage:
        runner = BenchmarkRunner(extractor_fn=my_extract)
        result = runner.run(dataset, config=BenchmarkConfig(name="nightly"))
    """

    def __init__(
        self,
        extractor_fn: ExtractorFn,
        run_id: str | None = None,
    ) -> None:
        self._extractor_fn = extractor_fn
        self._run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    def _select_samples(
        self,
        dataset: GoldenDataset,
        config: BenchmarkConfig,
    ) -> list[GoldenSample]:
        """Filter and select samples based on config."""
        samples = list(dataset.samples)

        if config.filter_types:
            samples = [s for s in samples if s.document_type in config.filter_types]

        if config.filter_tags:
            samples = [
                s
                for s in samples
                if any(t in s.tags for t in config.filter_tags)
            ]

        if config.max_samples and len(samples) > config.max_samples:
            samples = samples[: config.max_samples]

        return samples

    def _run_single(
        self,
        sample: GoldenSample,
        config: BenchmarkConfig,
    ) -> DocumentMetrics | None:
        """Run extraction on a single sample and evaluate."""
        try:
            start = time.monotonic()
            extracted = self._extractor_fn(
                sample.sample_id,
                sample.schema_name,
                sample.source_file,
                sample.metadata,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            return evaluate_document(
                document_id=sample.sample_id,
                schema_name=sample.schema_name,
                expected=sample.expected_fields,
                extracted=extracted,
                match_level=config.match_level,
                field_match_levels=config.field_match_levels,
                extraction_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(
                "benchmark_sample_failed",
                sample_id=sample.sample_id,
                error=str(e),
            )
            return None

    def run(
        self,
        dataset: GoldenDataset,
        config: BenchmarkConfig | None = None,
    ) -> BenchmarkResult:
        """
        Execute a full benchmark run.

        Args:
            dataset: Golden dataset to benchmark against.
            config: Benchmark configuration.

        Returns:
            BenchmarkResult with aggregate metrics.
        """
        config = config or BenchmarkConfig()
        result = BenchmarkResult(
            run_id=self._run_id,
            config=config,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            status=BenchmarkStatus.RUNNING,
            started_at=datetime.now(UTC).isoformat(),
        )

        samples = self._select_samples(dataset, config)
        logger.info(
            "benchmark_started",
            run_id=self._run_id,
            dataset=dataset.name,
            samples=len(samples),
        )

        doc_metrics: list[DocumentMetrics] = []
        start_time = time.monotonic()

        for sample in samples:
            dm = self._run_single(sample, config)
            if dm is not None:
                doc_metrics.append(dm)
            else:
                result.errors.append({
                    "sample_id": sample.sample_id,
                    "error": "Extraction failed",
                })

        result.duration_seconds = time.monotonic() - start_time
        result.completed_at = datetime.now(UTC).isoformat()

        result.aggregate = AggregateMetrics(
            document_metrics=doc_metrics,
            dataset_name=dataset.name,
        )

        # Check for regression
        if config.fail_on_regression and config.baseline_f1 is not None:
            if result.aggregate.micro_f1 < config.baseline_f1:
                result.regression_detected = True
                logger.warning(
                    "benchmark_regression_detected",
                    run_id=self._run_id,
                    micro_f1=result.aggregate.micro_f1,
                    baseline_f1=config.baseline_f1,
                )

        result.status = BenchmarkStatus.COMPLETED

        logger.info(
            "benchmark_completed",
            run_id=self._run_id,
            micro_f1=round(result.aggregate.micro_f1, 4),
            macro_f1=round(result.aggregate.macro_f1, 4),
            exact_match_rate=round(result.aggregate.exact_match_rate, 4),
            duration=round(result.duration_seconds, 3),
            errors=len(result.errors),
        )

        return result


# ──────────────────────────────────────────────────────────────────
# Comparison Helper
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class BenchmarkComparison:
    """Comparison between two benchmark results."""

    baseline: BenchmarkResult
    candidate: BenchmarkResult
    f1_delta: float = 0.0
    precision_delta: float = 0.0
    recall_delta: float = 0.0
    exact_match_delta: float = 0.0
    speed_delta_ms: float = 0.0
    per_field_f1_deltas: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline.run_id,
            "candidate_run_id": self.candidate.run_id,
            "f1_delta": round(self.f1_delta, 4),
            "precision_delta": round(self.precision_delta, 4),
            "recall_delta": round(self.recall_delta, 4),
            "exact_match_delta": round(self.exact_match_delta, 4),
            "speed_delta_ms": round(self.speed_delta_ms, 2),
            "per_field_f1_deltas": {
                k: round(v, 4) for k, v in self.per_field_f1_deltas.items()
            },
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def compare_runs(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    regression_threshold: float = 0.02,
) -> BenchmarkComparison:
    """
    Compare two benchmark runs and identify regressions/improvements.

    Args:
        baseline: The baseline (previous) benchmark result.
        candidate: The candidate (current) benchmark result.
        regression_threshold: Minimum F1 drop to flag as regression.

    Returns:
        BenchmarkComparison with deltas and flagged fields.
    """
    ba = baseline.aggregate
    ca = candidate.aggregate

    comparison = BenchmarkComparison(
        baseline=baseline,
        candidate=candidate,
        f1_delta=ca.micro_f1 - ba.micro_f1,
        precision_delta=ca.micro_precision - ba.micro_precision,
        recall_delta=ca.micro_recall - ba.micro_recall,
        exact_match_delta=ca.exact_match_rate - ba.exact_match_rate,
        speed_delta_ms=ca.mean_extraction_time_ms - ba.mean_extraction_time_ms,
    )

    # Per-field F1 comparison
    baseline_field_f1 = ba.per_field_f1()
    candidate_field_f1 = ca.per_field_f1()
    all_fields = set(baseline_field_f1.keys()) | set(candidate_field_f1.keys())

    for field_name in sorted(all_fields):
        b_f1 = baseline_field_f1.get(field_name, 0.0)
        c_f1 = candidate_field_f1.get(field_name, 0.0)
        delta = c_f1 - b_f1
        comparison.per_field_f1_deltas[field_name] = delta

        if delta < -regression_threshold:
            comparison.regressions.append(field_name)
        elif delta > regression_threshold:
            comparison.improvements.append(field_name)

    return comparison
