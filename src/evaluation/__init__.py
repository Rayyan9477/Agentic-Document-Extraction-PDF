"""
Evaluation and benchmarking module for document extraction.

Provides tools for measuring extraction accuracy against golden datasets,
running A/B tests between strategies, and detecting regressions.

Usage:
    from src.evaluation import (
        BenchmarkRunner, BenchmarkConfig, BenchmarkResult,
        GoldenDataset, GoldenSample, create_sample,
        evaluate_document, AggregateMetrics,
        ABTestRunner, ABTestResult,
        RegressionDetector, RegressionReport,
    )
"""

from src.evaluation.ab_testing import (
    ABOutcome,
    ABTestConfig,
    ABTestResult,
    ABTestRunner,
)
from src.evaluation.benchmark import (
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStatus,
    compare_runs,
)
from src.evaluation.golden_dataset import (
    GoldenDataset,
    GoldenSample,
    create_sample,
    load_dataset,
    save_dataset,
)
from src.evaluation.metrics import (
    AggregateMetrics,
    DocumentMetrics,
    FieldMatchResult,
    MatchLevel,
    compare_field,
    evaluate_document,
)
from src.evaluation.regression import (
    FieldRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    load_baseline,
    save_baseline,
)

__all__ = [
    # Metrics
    "MatchLevel",
    "FieldMatchResult",
    "DocumentMetrics",
    "AggregateMetrics",
    "compare_field",
    "evaluate_document",
    # Golden dataset
    "GoldenSample",
    "GoldenDataset",
    "create_sample",
    "save_dataset",
    "load_dataset",
    # Benchmark
    "BenchmarkStatus",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkComparison",
    "compare_runs",
    # A/B testing
    "ABOutcome",
    "ABTestConfig",
    "ABTestResult",
    "ABTestRunner",
    # Regression
    "RegressionSeverity",
    "FieldRegression",
    "RegressionReport",
    "RegressionDetector",
    "save_baseline",
    "load_baseline",
]
