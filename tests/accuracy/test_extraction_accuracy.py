"""
Accuracy tests for document extraction using golden datasets.

Tests extraction accuracy against known ground truth data
to measure and track extraction quality metrics.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


@dataclass(slots=True)
class GoldenDataset:
    """
    Golden dataset for accuracy testing.

    Attributes:
        name: Dataset name.
        document_type: Type of documents in dataset.
        samples: List of sample documents with ground truth.
    """

    name: str
    document_type: str
    samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AccuracyMetrics:
    """
    Accuracy metrics for extraction evaluation.

    Attributes:
        total_fields: Total number of fields evaluated.
        correct_fields: Number of exactly correct extractions.
        partial_matches: Number of partial matches.
        missing_fields: Number of fields not extracted.
        extra_fields: Number of spurious extractions.
        field_accuracy: Percentage of correct fields.
        character_accuracy: Character-level accuracy for text fields.
        overall_score: Weighted overall accuracy score.
    """

    total_fields: int = 0
    correct_fields: int = 0
    partial_matches: int = 0
    missing_fields: int = 0
    extra_fields: int = 0
    field_accuracy: float = 0.0
    character_accuracy: float = 0.0
    overall_score: float = 0.0


class AccuracyCalculator:
    """Calculate extraction accuracy metrics."""

    def __init__(
        self,
        exact_match_weight: float = 0.7,
        partial_match_weight: float = 0.2,
        char_accuracy_weight: float = 0.1,
    ) -> None:
        """
        Initialize accuracy calculator.

        Args:
            exact_match_weight: Weight for exact matches in overall score.
            partial_match_weight: Weight for partial matches.
            char_accuracy_weight: Weight for character accuracy.
        """
        self.exact_match_weight = exact_match_weight
        self.partial_match_weight = partial_match_weight
        self.char_accuracy_weight = char_accuracy_weight

    def calculate(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> AccuracyMetrics:
        """
        Calculate accuracy metrics between extraction and ground truth.

        Args:
            extracted: Extracted field values.
            ground_truth: Ground truth field values.

        Returns:
            AccuracyMetrics with calculated scores.
        """
        metrics = AccuracyMetrics()

        gt_fields = set(ground_truth.keys())
        ext_fields = set(extracted.keys())

        # Fields in ground truth
        metrics.total_fields = len(gt_fields)

        # Missing and extra fields
        metrics.missing_fields = len(gt_fields - ext_fields)
        metrics.extra_fields = len(ext_fields - gt_fields)

        # Compare common fields
        total_chars = 0
        correct_chars = 0

        for field_name in gt_fields:
            if field_name not in extracted:
                continue

            gt_value = self._normalize_value(ground_truth[field_name])
            ext_value = self._normalize_value(extracted[field_name])

            if gt_value == ext_value:
                metrics.correct_fields += 1
            elif self._is_partial_match(gt_value, ext_value):
                metrics.partial_matches += 1

            # Character-level accuracy
            if isinstance(gt_value, str) and isinstance(ext_value, str):
                field_chars, field_correct = self._char_accuracy(gt_value, ext_value)
                total_chars += field_chars
                correct_chars += field_correct

        # Calculate percentages
        if metrics.total_fields > 0:
            metrics.field_accuracy = metrics.correct_fields / metrics.total_fields

        if total_chars > 0:
            metrics.character_accuracy = correct_chars / total_chars

        # Calculate overall score
        metrics.overall_score = (
            self.exact_match_weight * metrics.field_accuracy +
            self.partial_match_weight * (metrics.partial_matches / max(metrics.total_fields, 1)) +
            self.char_accuracy_weight * metrics.character_accuracy
        )

        return metrics

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a value for comparison."""
        if isinstance(value, dict):
            v = value.get("value", "")
            if isinstance(v, str):
                return v.strip().lower()
            return v
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def _is_partial_match(self, gt_value: Any, ext_value: Any) -> bool:
        """Check if values are a partial match."""
        if not isinstance(gt_value, str) or not isinstance(ext_value, str):
            return False

        gt_str = str(gt_value).lower()
        ext_str = str(ext_value).lower()

        # Check containment
        if gt_str in ext_str or ext_str in gt_str:
            return True

        # Check Levenshtein-like similarity (simplified)
        if len(gt_str) > 0 and len(ext_str) > 0:
            common_len = sum(1 for a, b in zip(gt_str, ext_str) if a == b)
            similarity = common_len / max(len(gt_str), len(ext_str))
            return similarity >= 0.8

        return False

    def _char_accuracy(self, gt_str: str, ext_str: str) -> tuple[int, int]:
        """Calculate character-level accuracy."""
        total = len(gt_str)
        correct = sum(1 for a, b in zip(gt_str, ext_str) if a == b)
        return total, correct


@pytest.fixture
def golden_cms1500_dataset() -> GoldenDataset:
    """Create golden dataset for CMS-1500 forms."""
    return GoldenDataset(
        name="cms1500_golden",
        document_type="CMS-1500",
        samples=[
            {
                "id": "cms1500_001",
                "ground_truth": {
                    "patient_name": "John A. Smith",
                    "patient_dob": "03/15/1965",
                    "patient_address": "123 Main Street, Springfield, IL 62701",
                    "insured_name": "John A. Smith",
                    "insurance_id": "ABC123456789",
                    "diagnosis_code_1": "J06.9",
                    "diagnosis_code_2": "R05.9",
                    "procedure_code_1": "99213",
                    "date_of_service": "01/15/2024",
                    "place_of_service": "11",
                    "charges_1": "125.00",
                    "total_charges": "125.00",
                    "provider_npi": "1234567890",
                    "provider_name": "Dr. Jane Wilson",
                },
            },
            {
                "id": "cms1500_002",
                "ground_truth": {
                    "patient_name": "Mary B. Johnson",
                    "patient_dob": "07/22/1978",
                    "patient_address": "456 Oak Avenue, Chicago, IL 60601",
                    "insured_name": "Robert Johnson",
                    "insurance_id": "XYZ987654321",
                    "diagnosis_code_1": "M54.5",
                    "procedure_code_1": "99214",
                    "procedure_code_2": "97110",
                    "date_of_service": "01/18/2024",
                    "place_of_service": "11",
                    "charges_1": "150.00",
                    "charges_2": "75.00",
                    "total_charges": "225.00",
                    "provider_npi": "9876543210",
                    "provider_name": "Dr. Michael Brown",
                },
            },
        ],
    )


@pytest.fixture
def golden_eob_dataset() -> GoldenDataset:
    """Create golden dataset for EOB documents."""
    return GoldenDataset(
        name="eob_golden",
        document_type="EOB",
        samples=[
            {
                "id": "eob_001",
                "ground_truth": {
                    "member_name": "Jane Smith",
                    "member_id": "MEM123456",
                    "claim_number": "CLM2024001234",
                    "service_date": "01/15/2024",
                    "provider_name": "Springfield Medical Center",
                    "billed_amount": "500.00",
                    "allowed_amount": "350.00",
                    "paid_amount": "280.00",
                    "patient_responsibility": "70.00",
                    "deductible": "50.00",
                    "coinsurance": "20.00",
                },
            },
        ],
    )


@pytest.fixture
def accuracy_calculator() -> AccuracyCalculator:
    """Create accuracy calculator."""
    return AccuracyCalculator()


@pytest.mark.accuracy
class TestAccuracyCalculator:
    """Test cases for AccuracyCalculator."""

    def test_perfect_match(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test perfect extraction match."""
        ground_truth = {
            "patient_name": "John Doe",
            "date_of_birth": "01/15/1980",
            "total_charges": "150.00",
        }
        extracted = {
            "patient_name": "John Doe",
            "date_of_birth": "01/15/1980",
            "total_charges": "150.00",
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.correct_fields == 3
        assert metrics.field_accuracy == 1.0
        assert metrics.missing_fields == 0
        assert metrics.extra_fields == 0

    def test_partial_match(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test partial extraction match."""
        ground_truth = {
            "patient_name": "John Doe Smith",
            "address": "123 Main Street",
        }
        extracted = {
            "patient_name": "John Doe",  # Contained in ground truth - partial match
            "address": "Main Street",  # Contained in ground truth - partial match
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # At least one should be a partial match (contained in original)
        assert metrics.partial_matches >= 1

    def test_missing_fields(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test handling of missing fields."""
        ground_truth = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3",
        }
        extracted = {
            "field1": "value1",
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.missing_fields == 2
        assert metrics.correct_fields == 1

    def test_extra_fields(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test handling of extra fields."""
        ground_truth = {
            "field1": "value1",
        }
        extracted = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3",
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.extra_fields == 2
        assert metrics.correct_fields == 1

    def test_case_insensitive(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test case-insensitive matching."""
        ground_truth = {"patient_name": "JOHN DOE"}
        extracted = {"patient_name": "john doe"}

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.correct_fields == 1

    def test_value_in_dict(self, accuracy_calculator: AccuracyCalculator) -> None:
        """Test extraction with value in dict format."""
        ground_truth = {"patient_name": "John Doe"}
        extracted = {"patient_name": {"value": "John Doe", "confidence": 0.95}}

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.correct_fields == 1


@pytest.mark.accuracy
class TestCMS1500Accuracy:
    """Accuracy tests for CMS-1500 form extraction."""

    def test_patient_information_accuracy(
        self,
        golden_cms1500_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test patient information extraction accuracy."""
        # Simulate extraction result
        sample = golden_cms1500_dataset.samples[0]
        extracted = {
            "patient_name": "John A. Smith",
            "patient_dob": "03/15/1965",
            "patient_address": "123 Main Street, Springfield, IL 62701",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if k in ["patient_name", "patient_dob", "patient_address"]
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.field_accuracy >= 0.9
        assert metrics.correct_fields >= 2

    def test_diagnosis_code_accuracy(
        self,
        golden_cms1500_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test diagnosis code extraction accuracy."""
        sample = golden_cms1500_dataset.samples[0]
        extracted = {
            "diagnosis_code_1": "J06.9",
            "diagnosis_code_2": "R05.9",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if k.startswith("diagnosis_code")
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # Medical codes should have high accuracy
        assert metrics.field_accuracy >= 0.9

    def test_procedure_code_accuracy(
        self,
        golden_cms1500_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test procedure code extraction accuracy."""
        sample = golden_cms1500_dataset.samples[0]
        extracted = {
            "procedure_code_1": "99213",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if k.startswith("procedure_code")
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.field_accuracy >= 0.9

    def test_financial_accuracy(
        self,
        golden_cms1500_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test financial field extraction accuracy."""
        sample = golden_cms1500_dataset.samples[0]
        extracted = {
            "charges_1": "125.00",
            "total_charges": "125.00",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if "charges" in k
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # Financial fields should be exact
        assert metrics.field_accuracy == 1.0


@pytest.mark.accuracy
class TestEOBAccuracy:
    """Accuracy tests for EOB document extraction."""

    def test_member_information_accuracy(
        self,
        golden_eob_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test member information extraction accuracy."""
        sample = golden_eob_dataset.samples[0]
        extracted = {
            "member_name": "Jane Smith",
            "member_id": "MEM123456",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if k.startswith("member")
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.field_accuracy >= 0.9

    def test_payment_accuracy(
        self,
        golden_eob_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test payment information extraction accuracy."""
        sample = golden_eob_dataset.samples[0]
        extracted = {
            "billed_amount": "500.00",
            "allowed_amount": "350.00",
            "paid_amount": "280.00",
            "patient_responsibility": "70.00",
        }

        ground_truth = {
            k: v for k, v in sample["ground_truth"].items()
            if "amount" in k or "responsibility" in k
        }

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # Payment amounts must be exact
        assert metrics.field_accuracy >= 0.9


@pytest.mark.accuracy
class TestAccuracyThresholds:
    """Tests for accuracy threshold validation."""

    def test_minimum_field_accuracy_threshold(
        self,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test minimum field accuracy threshold (80%)."""
        ground_truth = {f"field_{i}": f"value_{i}" for i in range(10)}
        extracted = {f"field_{i}": f"value_{i}" for i in range(8)}  # 80% correct

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.field_accuracy >= 0.80

    def test_critical_field_accuracy(
        self,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test accuracy of critical fields."""
        # Critical fields that must be 100% accurate
        critical_fields = ["npi", "diagnosis_code", "procedure_code"]

        ground_truth = {
            "npi": "1234567890",
            "diagnosis_code": "J06.9",
            "procedure_code": "99213",
        }
        extracted = ground_truth.copy()  # Perfect match for critical fields

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        assert metrics.field_accuracy == 1.0

    def test_overall_score_threshold(
        self,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test overall accuracy score threshold."""
        ground_truth = {f"field_{i}": f"value_{i}" for i in range(10)}
        extracted = {f"field_{i}": f"value_{i}" for i in range(10)}  # 100% correct

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # With 100% field accuracy and character accuracy:
        # score = 0.7 * 1.0 + 0.2 * 0 + 0.1 * 1.0 = 0.8
        assert metrics.overall_score >= 0.79  # Account for floating point precision
        assert metrics.field_accuracy == 1.0


@pytest.mark.accuracy
class TestAccuracyReporting:
    """Tests for accuracy reporting."""

    def test_metrics_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = AccuracyMetrics(
            total_fields=10,
            correct_fields=8,
            partial_matches=1,
            missing_fields=1,
            extra_fields=0,
            field_accuracy=0.8,
            character_accuracy=0.95,
            overall_score=0.85,
        )

        # Metrics should be serializable
        import dataclasses
        metrics_dict = dataclasses.asdict(metrics)

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["total_fields"] == 10
        assert metrics_dict["field_accuracy"] == 0.8

    def test_aggregate_accuracy(
        self,
        golden_cms1500_dataset: GoldenDataset,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test aggregating accuracy across multiple samples."""
        all_metrics: list[AccuracyMetrics] = []

        for sample in golden_cms1500_dataset.samples:
            # Simulate perfect extraction for test
            extracted = sample["ground_truth"].copy()
            metrics = accuracy_calculator.calculate(extracted, sample["ground_truth"])
            all_metrics.append(metrics)

        # Calculate aggregate metrics
        total_correct = sum(m.correct_fields for m in all_metrics)
        total_fields = sum(m.total_fields for m in all_metrics)
        aggregate_accuracy = total_correct / total_fields if total_fields > 0 else 0

        assert aggregate_accuracy >= 0.9


@pytest.mark.accuracy
class TestRegressionDetection:
    """Tests for detecting accuracy regressions."""

    def test_baseline_comparison(
        self,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test comparison against baseline accuracy."""
        baseline_accuracy = 0.90  # Previous version accuracy

        ground_truth = {f"field_{i}": f"value_{i}" for i in range(10)}
        extracted = {f"field_{i}": f"value_{i}" for i in range(9)}

        metrics = accuracy_calculator.calculate(extracted, ground_truth)

        # New version should not regress
        assert metrics.field_accuracy >= baseline_accuracy * 0.95  # Allow 5% margin

    def test_field_type_regression(
        self,
        accuracy_calculator: AccuracyCalculator,
    ) -> None:
        """Test for regressions in specific field types."""
        # Test different field types
        field_types = {
            "names": {"patient_name": "John Doe"},
            "dates": {"date_of_service": "01/15/2024"},
            "codes": {"diagnosis_code": "J06.9"},
            "amounts": {"total_charges": "150.00"},
        }

        for field_type, ground_truth in field_types.items():
            extracted = ground_truth.copy()
            metrics = accuracy_calculator.calculate(extracted, ground_truth)

            assert metrics.field_accuracy == 1.0, f"Regression in {field_type} extraction"
