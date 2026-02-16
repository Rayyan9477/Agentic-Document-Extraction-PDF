"""
A/B testing framework for comparing extraction strategies.

Allows running two extraction approaches side-by-side on the same
golden dataset and producing a statistical comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

from src.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    ExtractorFn,
    compare_runs,
)
from src.evaluation.golden_dataset import GoldenDataset

logger = structlog.get_logger(__name__)


class ABOutcome(str, Enum):
    """Outcome of an A/B test."""

    A_WINS = "a_wins"
    B_WINS = "b_wins"
    NO_DIFFERENCE = "no_difference"
    INCONCLUSIVE = "inconclusive"


@dataclass(slots=True)
class ABTestConfig:
    """Configuration for an A/B test."""

    test_name: str
    significance_threshold: float = 0.02  # minimum F1 delta to declare winner
    benchmark_config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of an A/B test comparing two extractors."""

    test_name: str
    outcome: ABOutcome
    result_a: BenchmarkResult
    result_b: BenchmarkResult
    f1_delta: float = 0.0  # B - A
    precision_delta: float = 0.0
    recall_delta: float = 0.0
    speed_delta_ms: float = 0.0
    per_field_regressions: list[str] = field(default_factory=list)
    per_field_improvements: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "outcome": self.outcome.value,
            "f1_a": round(self.result_a.aggregate.micro_f1, 4),
            "f1_b": round(self.result_b.aggregate.micro_f1, 4),
            "f1_delta": round(self.f1_delta, 4),
            "precision_delta": round(self.precision_delta, 4),
            "recall_delta": round(self.recall_delta, 4),
            "speed_delta_ms": round(self.speed_delta_ms, 2),
            "per_field_regressions": self.per_field_regressions,
            "per_field_improvements": self.per_field_improvements,
            "a_run_id": self.result_a.run_id,
            "b_run_id": self.result_b.run_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        winner = {
            ABOutcome.A_WINS: "Variant A",
            ABOutcome.B_WINS: "Variant B",
            ABOutcome.NO_DIFFERENCE: "No significant difference",
            ABOutcome.INCONCLUSIVE: "Inconclusive",
        }[self.outcome]
        return (
            f"A/B Test '{self.test_name}': {winner} "
            f"(F1 delta: {self.f1_delta:+.4f})"
        )


# ──────────────────────────────────────────────────────────────────
# A/B Test Runner
# ──────────────────────────────────────────────────────────────────


class ABTestRunner:
    """
    Runs A/B tests comparing two extraction strategies.

    Usage:
        runner = ABTestRunner()
        result = runner.run(
            dataset=my_dataset,
            extractor_a=extract_v1,
            extractor_b=extract_v2,
            config=ABTestConfig(test_name="v1_vs_v2"),
        )
    """

    def run(
        self,
        dataset: GoldenDataset,
        extractor_a: ExtractorFn,
        extractor_b: ExtractorFn,
        config: ABTestConfig | None = None,
    ) -> ABTestResult:
        """
        Execute an A/B test.

        Args:
            dataset: Golden dataset to test on.
            extractor_a: First extraction strategy (baseline).
            extractor_b: Second extraction strategy (candidate).
            config: A/B test configuration.

        Returns:
            ABTestResult with comparison.
        """
        config = config or ABTestConfig(test_name="ab_test")
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        logger.info("ab_test_started", test_name=config.test_name)

        # Run variant A
        runner_a = BenchmarkRunner(
            extractor_fn=extractor_a,
            run_id=f"{ts}_A",
        )
        result_a = runner_a.run(dataset, config.benchmark_config)

        # Run variant B
        runner_b = BenchmarkRunner(
            extractor_fn=extractor_b,
            run_id=f"{ts}_B",
        )
        result_b = runner_b.run(dataset, config.benchmark_config)

        # Compare
        comparison = compare_runs(
            baseline=result_a,
            candidate=result_b,
            regression_threshold=config.significance_threshold,
        )

        # Determine outcome
        threshold = config.significance_threshold
        if comparison.f1_delta > threshold:
            outcome = ABOutcome.B_WINS
        elif comparison.f1_delta < -threshold:
            outcome = ABOutcome.A_WINS
        else:
            outcome = ABOutcome.NO_DIFFERENCE

        ab_result = ABTestResult(
            test_name=config.test_name,
            outcome=outcome,
            result_a=result_a,
            result_b=result_b,
            f1_delta=comparison.f1_delta,
            precision_delta=comparison.precision_delta,
            recall_delta=comparison.recall_delta,
            speed_delta_ms=comparison.speed_delta_ms,
            per_field_regressions=comparison.regressions,
            per_field_improvements=comparison.improvements,
            metadata=config.metadata,
        )

        logger.info(
            "ab_test_completed",
            test_name=config.test_name,
            outcome=outcome.value,
            f1_delta=round(comparison.f1_delta, 4),
        )

        return ab_result
