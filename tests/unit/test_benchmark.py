"""
Unit tests for Phase 5A: Evaluation & Benchmarking Framework.

Tests metrics computation, golden dataset management, benchmark runner,
A/B testing, and regression detection.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.ab_testing import ABOutcome, ABTestConfig, ABTestResult, ABTestRunner
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


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _dummy_extractor(
    sample_id: str,
    schema_name: str,
    source_file: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """A simple extractor that returns predefined data for testing."""
    return metadata.get("mock_extracted", {})


def _perfect_extractor(
    sample_id: str,
    schema_name: str,
    source_file: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extractor that always returns the expected fields exactly."""
    return metadata.get("expected_fields", {})


def _bad_extractor(
    sample_id: str,
    schema_name: str,
    source_file: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extractor that always returns wrong data."""
    return {"wrong_field": "wrong_value"}


def _failing_extractor(
    sample_id: str,
    schema_name: str,
    source_file: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extractor that always raises."""
    raise RuntimeError("Extraction failed intentionally")


def _make_golden_dataset(n: int = 3) -> GoldenDataset:
    """Create a small golden dataset for testing."""
    samples = []
    for i in range(n):
        samples.append(
            GoldenSample(
                sample_id=f"sample_{i}",
                document_type="invoice",
                schema_name="invoice",
                expected_fields={
                    "vendor_name": f"Vendor {i}",
                    "total_amount": f"${i * 100}.00",
                    "invoice_number": f"INV-{i:04d}",
                },
                source_file=f"invoice_{i}.pdf",
                metadata={
                    "mock_extracted": {
                        "vendor_name": f"Vendor {i}",
                        "total_amount": f"${i * 100}.00",
                        "invoice_number": f"INV-{i:04d}",
                    },
                    "expected_fields": {
                        "vendor_name": f"Vendor {i}",
                        "total_amount": f"${i * 100}.00",
                        "invoice_number": f"INV-{i:04d}",
                    },
                },
                tags=["easy"] if i == 0 else ["medium"],
            )
        )
    return GoldenDataset(
        name="test_dataset",
        version="1.0.0",
        description="Test dataset",
        samples=samples,
    )


# ──────────────────────────────────────────────────────────────────
# Field Comparison Tests
# ──────────────────────────────────────────────────────────────────


class TestFieldComparison:
    """Tests for compare_field and FieldMatchResult."""

    def test_exact_match(self):
        result = compare_field("name", "Acme Corp", "Acme Corp", MatchLevel.EXACT)
        assert result.is_match is True
        assert result.similarity == 1.0

    def test_exact_mismatch(self):
        result = compare_field("name", "Acme Corp", "acme corp", MatchLevel.EXACT)
        assert result.is_match is False
        assert 0 < result.similarity < 1

    def test_normalized_match(self):
        result = compare_field("name", "  Acme Corp  ", "acme corp", MatchLevel.NORMALIZED)
        assert result.is_match is True
        assert result.similarity == 1.0

    def test_normalized_mismatch(self):
        result = compare_field("name", "Acme Corp", "Beta Inc", MatchLevel.NORMALIZED)
        assert result.is_match is False

    def test_fuzzy_match_above_threshold(self):
        result = compare_field(
            "name", "Acme Corporation", "Acme Corporaion", MatchLevel.FUZZY
        )
        assert result.is_match is True
        assert result.similarity >= 0.85

    def test_fuzzy_mismatch_below_threshold(self):
        result = compare_field("name", "Acme Corp", "XYZ Inc", MatchLevel.FUZZY)
        assert result.is_match is False

    def test_numeric_match(self):
        result = compare_field("amount", "$1,000.00", "1000.00", MatchLevel.NUMERIC)
        assert result.is_match is True
        assert result.similarity >= 0.99

    def test_numeric_mismatch(self):
        result = compare_field("amount", "$1,000.00", "$500.00", MatchLevel.NUMERIC)
        assert result.is_match is False

    def test_numeric_with_tolerance(self):
        result = compare_field("amount", "100.00", "100.005", MatchLevel.NUMERIC)
        assert result.is_match is True

    def test_both_none(self):
        result = compare_field("name", None, None)
        assert result.is_match is True

    def test_expected_none_extracted_present(self):
        result = compare_field("name", None, "Something")
        assert result.is_match is False
        assert result.error_message == "Unexpected value"

    def test_expected_present_extracted_none(self):
        result = compare_field("name", "Expected", None)
        assert result.is_match is False
        assert result.error_message == "Missing value"

    def test_is_present_property(self):
        r1 = compare_field("a", "x", "y")
        assert r1.is_present is True
        r2 = compare_field("a", "x", None)
        assert r2.is_present is False

    def test_is_expected_property(self):
        r1 = compare_field("a", "x", "y")
        assert r1.is_expected is True
        r2 = compare_field("a", None, "y")
        assert r2.is_expected is False

    def test_numeric_zero(self):
        result = compare_field("amount", "0", "0", MatchLevel.NUMERIC)
        assert result.is_match is True

    def test_numeric_non_parseable(self):
        result = compare_field("amount", "N/A", "N/A", MatchLevel.NUMERIC)
        assert result.is_match is True  # falls back to normalized


# ──────────────────────────────────────────────────────────────────
# Document Metrics Tests
# ──────────────────────────────────────────────────────────────────


class TestDocumentMetrics:
    """Tests for evaluate_document and DocumentMetrics."""

    def test_perfect_extraction(self):
        expected = {"name": "Acme", "amount": "$100", "date": "2025-01-01"}
        extracted = {"name": "Acme", "amount": "$100", "date": "2025-01-01"}
        dm = evaluate_document("doc1", "invoice", expected, extracted)
        assert dm.precision == 1.0
        assert dm.recall == 1.0
        assert dm.f1 == 1.0
        assert dm.exact_match is True

    def test_partial_extraction(self):
        expected = {"name": "Acme", "amount": "$100", "date": "2025-01-01"}
        extracted = {"name": "Acme", "amount": "$100"}
        dm = evaluate_document("doc1", "invoice", expected, extracted)
        assert dm.precision == 1.0  # all extracted fields are correct
        assert dm.recall == pytest.approx(2 / 3, rel=1e-3)
        assert dm.exact_match is False

    def test_wrong_extraction(self):
        expected = {"name": "Acme", "amount": "$100"}
        extracted = {"name": "Beta", "amount": "$200"}
        dm = evaluate_document("doc1", "invoice", expected, extracted)
        assert dm.correct_fields == 0
        assert dm.precision == 0.0
        assert dm.recall == 0.0
        assert dm.f1 == 0.0

    def test_extra_fields_extracted(self):
        expected = {"name": "Acme"}
        extracted = {"name": "Acme", "extra_field": "data"}
        dm = evaluate_document("doc1", "invoice", expected, extracted)
        assert dm.extracted_fields == 2
        assert dm.correct_fields == 1  # name matches
        assert dm.recall == 1.0
        assert dm.precision == pytest.approx(0.5, rel=1e-3)

    def test_empty_expected(self):
        dm = evaluate_document("doc1", "invoice", {}, {})
        assert dm.total_fields == 0
        assert dm.f1 == 0.0
        assert dm.exact_match is True

    def test_extraction_time_tracked(self):
        dm = evaluate_document("doc1", "inv", {"a": "1"}, {"a": "1"}, extraction_time_ms=42)
        assert dm.extraction_time_ms == 42

    def test_per_field_match_levels(self):
        expected = {"name": "Acme Corp", "amount": "$1,000.00"}
        extracted = {"name": "acme corp", "amount": "1000.00"}
        dm = evaluate_document(
            "doc1",
            "invoice",
            expected,
            extracted,
            match_level=MatchLevel.EXACT,
            field_match_levels={"amount": MatchLevel.NUMERIC},
        )
        name_result = next(r for r in dm.field_results if r.field_name == "name")
        amount_result = next(r for r in dm.field_results if r.field_name == "amount")
        assert name_result.is_match is False  # exact — case sensitive
        assert amount_result.is_match is True  # numeric — $1000 == 1000

    def test_to_dict(self):
        dm = evaluate_document("doc1", "inv", {"a": "1"}, {"a": "1"})
        d = dm.to_dict()
        assert d["document_id"] == "doc1"
        assert d["precision"] == 1.0
        assert "f1" in d

    def test_mean_similarity(self):
        expected = {"a": "hello", "b": "world"}
        extracted = {"a": "hello", "b": "world"}
        dm = evaluate_document("doc1", "test", expected, extracted)
        assert dm.mean_similarity == 1.0


# ──────────────────────────────────────────────────────────────────
# Aggregate Metrics Tests
# ──────────────────────────────────────────────────────────────────


class TestAggregateMetrics:
    """Tests for AggregateMetrics."""

    def _make_perfect_doc(self, doc_id: str) -> DocumentMetrics:
        return evaluate_document(
            doc_id, "inv", {"a": "1", "b": "2"}, {"a": "1", "b": "2"}
        )

    def _make_half_doc(self, doc_id: str) -> DocumentMetrics:
        return evaluate_document(
            doc_id, "inv", {"a": "1", "b": "2"}, {"a": "1", "b": "WRONG"}
        )

    def test_perfect_aggregate(self):
        agg = AggregateMetrics(
            document_metrics=[self._make_perfect_doc("d1"), self._make_perfect_doc("d2")],
            dataset_name="test",
        )
        assert agg.micro_f1 == 1.0
        assert agg.macro_f1 == 1.0
        assert agg.exact_match_rate == 1.0

    def test_mixed_aggregate(self):
        agg = AggregateMetrics(
            document_metrics=[self._make_perfect_doc("d1"), self._make_half_doc("d2")],
            dataset_name="test",
        )
        assert agg.micro_precision == pytest.approx(3 / 4, rel=1e-3)
        assert agg.micro_recall == pytest.approx(3 / 4, rel=1e-3)
        assert agg.exact_match_rate == 0.5

    def test_empty_aggregate(self):
        agg = AggregateMetrics()
        assert agg.document_count == 0
        assert agg.micro_f1 == 0.0
        assert agg.macro_f1 == 0.0
        assert agg.exact_match_rate == 0.0

    def test_per_field_f1(self):
        agg = AggregateMetrics(
            document_metrics=[self._make_perfect_doc("d1"), self._make_half_doc("d2")],
        )
        field_f1 = agg.per_field_f1()
        assert field_f1["a"] == 1.0
        assert field_f1["b"] == pytest.approx(0.5, abs=0.01)

    def test_to_dict(self):
        agg = AggregateMetrics(
            document_metrics=[self._make_perfect_doc("d1")],
            dataset_name="ds",
        )
        d = agg.to_dict()
        assert d["dataset_name"] == "ds"
        assert d["document_count"] == 1
        assert "per_field_f1" in d

    def test_mean_extraction_time(self):
        dm1 = evaluate_document("d1", "inv", {"a": "1"}, {"a": "1"}, extraction_time_ms=100)
        dm2 = evaluate_document("d2", "inv", {"a": "1"}, {"a": "1"}, extraction_time_ms=200)
        agg = AggregateMetrics(document_metrics=[dm1, dm2])
        assert agg.mean_extraction_time_ms == 150.0


# ──────────────────────────────────────────────────────────────────
# Golden Dataset Tests
# ──────────────────────────────────────────────────────────────────


class TestGoldenDataset:
    """Tests for GoldenSample and GoldenDataset."""

    def test_create_sample(self):
        sample = create_sample("s1", "invoice", "invoice", {"total": "$100"}, tags=["easy"])
        assert sample.sample_id == "s1"
        assert sample.field_count() == 1
        assert "easy" in sample.tags

    def test_sample_to_from_dict(self):
        sample = create_sample("s1", "w2", "w2", {"wages": "$75000"})
        d = sample.to_dict()
        restored = GoldenSample.from_dict(d)
        assert restored.sample_id == "s1"
        assert restored.expected_fields["wages"] == "$75000"

    def test_dataset_creation(self):
        ds = _make_golden_dataset(5)
        assert ds.sample_count == 5
        assert ds.name == "test_dataset"

    def test_dataset_document_types(self):
        ds = _make_golden_dataset()
        assert ds.document_types == ["invoice"]

    def test_dataset_filter_by_type(self):
        ds = _make_golden_dataset()
        filtered = ds.filter_by_type("invoice")
        assert len(filtered) == 3

    def test_dataset_filter_by_tag(self):
        ds = _make_golden_dataset()
        easy = ds.filter_by_tag("easy")
        assert len(easy) == 1
        assert easy[0].sample_id == "sample_0"

    def test_dataset_add_sample(self):
        ds = _make_golden_dataset(1)
        ds.add_sample(create_sample("new_sample", "w2", "w2", {"wages": "$50000"}))
        assert ds.sample_count == 2

    def test_dataset_add_duplicate_raises(self):
        ds = _make_golden_dataset(1)
        with pytest.raises(ValueError, match="already exists"):
            ds.add_sample(create_sample("sample_0", "w2", "w2", {}))

    def test_dataset_remove_sample(self):
        ds = _make_golden_dataset(2)
        assert ds.remove_sample("sample_0") is True
        assert ds.sample_count == 1
        assert ds.remove_sample("nonexistent") is False

    def test_dataset_get_sample(self):
        ds = _make_golden_dataset()
        assert ds.get_sample("sample_0") is not None
        assert ds.get_sample("nonexistent") is None

    def test_dataset_content_hash_deterministic(self):
        ds1 = _make_golden_dataset()
        ds2 = _make_golden_dataset()
        assert ds1.content_hash() == ds2.content_hash()

    def test_dataset_content_hash_changes(self):
        ds = _make_golden_dataset()
        h1 = ds.content_hash()
        ds.add_sample(create_sample("extra", "w2", "w2", {"wages": "1"}))
        h2 = ds.content_hash()
        assert h1 != h2

    def test_save_and_load(self, tmp_path):
        ds = _make_golden_dataset()
        path = tmp_path / "golden.json"
        save_dataset(ds, path)
        loaded = load_dataset(path)
        assert loaded.name == ds.name
        assert loaded.sample_count == ds.sample_count

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_dataset_to_from_dict(self):
        ds = _make_golden_dataset()
        d = ds.to_dict()
        restored = GoldenDataset.from_dict(d)
        assert restored.name == ds.name
        assert restored.sample_count == ds.sample_count

    def test_dataset_all_tags(self):
        ds = _make_golden_dataset()
        tags = ds.all_tags
        assert "easy" in tags
        assert "medium" in tags

    def test_filter_by_schema(self):
        ds = _make_golden_dataset()
        filtered = ds.filter_by_schema("invoice")
        assert len(filtered) == 3


# ──────────────────────────────────────────────────────────────────
# Benchmark Runner Tests
# ──────────────────────────────────────────────────────────────────


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner and BenchmarkConfig."""

    def test_perfect_benchmark(self):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor, run_id="test_run")
        result = runner.run(ds)
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.aggregate.micro_f1 == 1.0
        assert result.run_id == "test_run"
        assert len(result.errors) == 0

    def test_failing_extractor_captured(self):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=_failing_extractor, run_id="fail_run")
        result = runner.run(ds)
        assert result.status == BenchmarkStatus.COMPLETED
        assert len(result.errors) == 3
        assert result.aggregate.document_count == 0

    def test_config_filter_by_tag(self):
        ds = _make_golden_dataset()
        config = BenchmarkConfig(filter_tags=["easy"])
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor)
        result = runner.run(ds, config)
        assert result.aggregate.document_count == 1

    def test_config_filter_by_type(self):
        ds = _make_golden_dataset()
        config = BenchmarkConfig(filter_types=["w2"])  # no w2 in dataset
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor)
        result = runner.run(ds, config)
        assert result.aggregate.document_count == 0

    def test_config_max_samples(self):
        ds = _make_golden_dataset(10)
        config = BenchmarkConfig(max_samples=3)
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor)
        result = runner.run(ds, config)
        assert result.aggregate.document_count == 3

    def test_regression_detection(self):
        ds = _make_golden_dataset()
        config = BenchmarkConfig(
            fail_on_regression=True,
            baseline_f1=0.99,  # set baseline higher than what bad_extractor achieves
        )
        runner = BenchmarkRunner(extractor_fn=_bad_extractor)
        result = runner.run(ds, config)
        assert result.regression_detected is True
        assert result.success is False

    def test_no_regression(self):
        ds = _make_golden_dataset()
        config = BenchmarkConfig(fail_on_regression=True, baseline_f1=0.5)
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor)
        result = runner.run(ds, config)
        assert result.regression_detected is False
        assert result.success is True

    def test_result_to_dict(self):
        ds = _make_golden_dataset(1)
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor, run_id="dict_test")
        result = runner.run(ds)
        d = result.to_dict()
        assert d["run_id"] == "dict_test"
        assert d["status"] == "completed"
        assert "aggregate" in d

    def test_config_to_dict(self):
        config = BenchmarkConfig(
            name="nightly",
            match_level=MatchLevel.FUZZY,
            field_match_levels={"amount": MatchLevel.NUMERIC},
        )
        d = config.to_dict()
        assert d["name"] == "nightly"
        assert d["match_level"] == "fuzzy"
        assert d["field_match_levels"]["amount"] == "numeric"

    def test_duration_tracked(self):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor)
        result = runner.run(ds)
        assert result.duration_seconds >= 0
        assert result.started_at != ""
        assert result.completed_at != ""


# ──────────────────────────────────────────────────────────────────
# Compare Runs Tests
# ──────────────────────────────────────────────────────────────────


class TestCompareRuns:
    """Tests for compare_runs."""

    def _run_benchmark(self, extractor, run_id="run"):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=extractor, run_id=run_id)
        return runner.run(ds)

    def test_identical_runs(self):
        r1 = self._run_benchmark(_dummy_extractor, "baseline")
        r2 = self._run_benchmark(_dummy_extractor, "candidate")
        comp = compare_runs(r1, r2)
        assert comp.f1_delta == pytest.approx(0.0, abs=1e-6)
        assert len(comp.regressions) == 0
        assert len(comp.improvements) == 0

    def test_improvement_detected(self):
        r1 = self._run_benchmark(_bad_extractor, "baseline")
        r2 = self._run_benchmark(_dummy_extractor, "candidate")
        comp = compare_runs(r1, r2)
        assert comp.f1_delta > 0
        assert len(comp.improvements) > 0

    def test_regression_detected(self):
        r1 = self._run_benchmark(_dummy_extractor, "baseline")
        r2 = self._run_benchmark(_bad_extractor, "candidate")
        comp = compare_runs(r1, r2)
        assert comp.f1_delta < 0
        assert len(comp.regressions) > 0

    def test_comparison_to_dict(self):
        r1 = self._run_benchmark(_dummy_extractor, "b")
        r2 = self._run_benchmark(_dummy_extractor, "c")
        comp = compare_runs(r1, r2)
        d = comp.to_dict()
        assert "f1_delta" in d
        assert "regressions" in d


# ──────────────────────────────────────────────────────────────────
# A/B Testing Tests
# ──────────────────────────────────────────────────────────────────


class TestABTesting:
    """Tests for ABTestRunner and ABTestResult."""

    def test_ab_b_wins(self):
        ds = _make_golden_dataset()
        runner = ABTestRunner()
        result = runner.run(
            dataset=ds,
            extractor_a=_bad_extractor,
            extractor_b=_dummy_extractor,
            config=ABTestConfig(test_name="bad_vs_good"),
        )
        assert result.outcome == ABOutcome.B_WINS
        assert result.f1_delta > 0

    def test_ab_a_wins(self):
        ds = _make_golden_dataset()
        runner = ABTestRunner()
        result = runner.run(
            dataset=ds,
            extractor_a=_dummy_extractor,
            extractor_b=_bad_extractor,
            config=ABTestConfig(test_name="good_vs_bad"),
        )
        assert result.outcome == ABOutcome.A_WINS
        assert result.f1_delta < 0

    def test_ab_no_difference(self):
        ds = _make_golden_dataset()
        runner = ABTestRunner()
        result = runner.run(
            dataset=ds,
            extractor_a=_dummy_extractor,
            extractor_b=_dummy_extractor,
            config=ABTestConfig(test_name="same_vs_same"),
        )
        assert result.outcome == ABOutcome.NO_DIFFERENCE

    def test_ab_result_to_dict(self):
        ds = _make_golden_dataset()
        runner = ABTestRunner()
        result = runner.run(
            dataset=ds,
            extractor_a=_dummy_extractor,
            extractor_b=_dummy_extractor,
            config=ABTestConfig(test_name="dict_test"),
        )
        d = result.to_dict()
        assert d["test_name"] == "dict_test"
        assert "outcome" in d
        assert "f1_delta" in d

    def test_ab_summary(self):
        ds = _make_golden_dataset()
        runner = ABTestRunner()
        result = runner.run(
            dataset=ds,
            extractor_a=_dummy_extractor,
            extractor_b=_dummy_extractor,
        )
        assert "A/B Test" in result.summary


# ──────────────────────────────────────────────────────────────────
# Regression Detection Tests
# ──────────────────────────────────────────────────────────────────


class TestRegressionDetection:
    """Tests for RegressionDetector."""

    def _run(self, extractor, run_id="run"):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=extractor, run_id=run_id)
        return runner.run(ds)

    def test_no_regression(self):
        baseline = self._run(_dummy_extractor, "baseline")
        current = self._run(_dummy_extractor, "current")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        assert report.has_regression is False
        assert report.regression_count == 0

    def test_regression_detected(self):
        baseline = self._run(_dummy_extractor, "baseline")
        current = self._run(_bad_extractor, "current")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        assert report.has_regression is True
        assert report.regression_count > 0
        assert report.overall_severity in (
            RegressionSeverity.WARNING,
            RegressionSeverity.CRITICAL,
        )

    def test_critical_severity(self):
        baseline = self._run(_dummy_extractor, "baseline")
        current = self._run(_bad_extractor, "current")
        detector = RegressionDetector(warning_threshold=0.01, critical_threshold=0.05)
        report = detector.compare(baseline, current)
        assert report.overall_severity == RegressionSeverity.CRITICAL

    def test_improvements_tracked(self):
        baseline = self._run(_bad_extractor, "baseline")
        current = self._run(_dummy_extractor, "current")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        assert report.improvement_count > 0
        assert report.has_regression is False

    def test_report_to_dict(self):
        baseline = self._run(_dummy_extractor, "b")
        current = self._run(_dummy_extractor, "c")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        d = report.to_dict()
        assert "has_regression" in d
        assert "field_regressions" in d

    def test_report_summary_no_regression(self):
        baseline = self._run(_dummy_extractor, "b")
        current = self._run(_dummy_extractor, "c")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        assert "No regressions" in report.summary

    def test_report_summary_with_regression(self):
        baseline = self._run(_dummy_extractor, "b")
        current = self._run(_bad_extractor, "c")
        detector = RegressionDetector()
        report = detector.compare(baseline, current)
        assert "regression(s) detected" in report.summary


# ──────────────────────────────────────────────────────────────────
# Baseline Storage Tests
# ──────────────────────────────────────────────────────────────────


class TestBaselineStorage:
    """Tests for save_baseline and load_baseline."""

    def test_save_and_load(self, tmp_path):
        ds = _make_golden_dataset()
        runner = BenchmarkRunner(extractor_fn=_dummy_extractor, run_id="bl_test")
        result = runner.run(ds)
        path = tmp_path / "baseline.json"
        save_baseline(result, path)
        loaded = load_baseline(path)
        assert loaded["run_id"] == "bl_test"
        assert loaded["status"] == "completed"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")


# ──────────────────────────────────────────────────────────────────
# Module Exports Tests
# ──────────────────────────────────────────────────────────────────


class TestModuleExports:
    """Verify all evaluation module exports."""

    def test_metrics_exports(self):
        from src.evaluation import (
            AggregateMetrics,
            DocumentMetrics,
            FieldMatchResult,
            MatchLevel,
            compare_field,
            evaluate_document,
        )
        assert MatchLevel.EXACT is not None

    def test_golden_dataset_exports(self):
        from src.evaluation import (
            GoldenDataset,
            GoldenSample,
            create_sample,
            load_dataset,
            save_dataset,
        )
        assert GoldenDataset is not None

    def test_benchmark_exports(self):
        from src.evaluation import (
            BenchmarkConfig,
            BenchmarkResult,
            BenchmarkRunner,
            BenchmarkStatus,
            compare_runs,
        )
        assert BenchmarkStatus.COMPLETED is not None

    def test_ab_testing_exports(self):
        from src.evaluation import ABOutcome, ABTestConfig, ABTestResult, ABTestRunner
        assert ABOutcome.B_WINS is not None

    def test_regression_exports(self):
        from src.evaluation import (
            RegressionDetector,
            RegressionReport,
            RegressionSeverity,
            save_baseline,
            load_baseline,
        )
        assert RegressionSeverity.CRITICAL is not None
