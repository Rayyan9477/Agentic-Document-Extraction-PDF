"""
Tests for src/validation/dual_pass.py — dual-pass extraction comparison.
"""

import pytest

from src.validation.dual_pass import (
    ComparisonResult,
    DualPassComparator,
    DualPassResult,
    FieldComparison,
    MergeStrategy,
    compare_extractions,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:

    def test_comparison_result_values(self):
        assert ComparisonResult.EXACT_MATCH == "exact_match"
        assert ComparisonResult.MISMATCH == "mismatch"
        assert ComparisonResult.BOTH_EMPTY == "both_empty"

    def test_merge_strategy_values(self):
        assert MergeStrategy.PREFER_PASS1 == "prefer_pass1"
        assert MergeStrategy.REQUIRE_AGREEMENT == "require_agreement"


# ---------------------------------------------------------------------------
# FieldComparison
# ---------------------------------------------------------------------------


class TestFieldComparison:

    def test_to_dict(self):
        fc = FieldComparison(
            field_name="name",
            pass1_value="John",
            pass2_value="John",
            result=ComparisonResult.EXACT_MATCH,
            similarity_score=1.0,
            merged_value="John",
            merge_confidence=0.95,
        )
        d = fc.to_dict()
        assert d["field_name"] == "name"
        assert d["result"] == "exact_match"

    def test_frozen(self):
        fc = FieldComparison(
            field_name="x",
            pass1_value="a",
            pass2_value="b",
            result=ComparisonResult.MISMATCH,
            similarity_score=0.0,
            merged_value="a",
            merge_confidence=0.3,
        )
        with pytest.raises(AttributeError):
            fc.field_name = "y"


# ---------------------------------------------------------------------------
# DualPassResult
# ---------------------------------------------------------------------------


class TestDualPassResult:

    def test_defaults(self):
        r = DualPassResult()
        assert r.overall_agreement_rate == 0.0
        assert r.requires_retry is False
        assert r.requires_human_review is False

    def test_to_dict(self):
        r = DualPassResult()
        d = r.to_dict()
        assert "merged_output" in d
        assert "mismatch_fields" in d


# ---------------------------------------------------------------------------
# DualPassComparator — exact matches
# ---------------------------------------------------------------------------


class TestExactMatch:

    def test_identical_strings(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John Smith"},
        )
        fc = result.field_comparisons["name"]
        assert fc.result == ComparisonResult.EXACT_MATCH
        assert fc.similarity_score >= 0.99
        assert result.overall_agreement_rate == 1.0

    def test_identical_numbers(self):
        comp = DualPassComparator()
        result = comp.compare({"amount": 150.0}, {"amount": 150.0})
        fc = result.field_comparisons["amount"]
        assert fc.result == ComparisonResult.EXACT_MATCH


# ---------------------------------------------------------------------------
# DualPassComparator — fuzzy matches
# ---------------------------------------------------------------------------


class TestFuzzyMatch:

    def test_normalized_strings_fuzzy(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "John  Smith"},
            {"name": "john smith"},
        )
        fc = result.field_comparisons["name"]
        # After normalization these should be very similar
        assert fc.result in (ComparisonResult.EXACT_MATCH, ComparisonResult.FUZZY_MATCH)
        assert fc.similarity_score >= 0.85

    def test_close_numbers_fuzzy(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"amount": 100.00},
            {"amount": 100.05},
        )
        fc = result.field_comparisons["amount"]
        assert fc.similarity_score >= 0.85


# ---------------------------------------------------------------------------
# DualPassComparator — mismatches
# ---------------------------------------------------------------------------


class TestMismatch:

    def test_completely_different_values(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "Alice Johnson"},
            {"name": "Bob Williams"},
        )
        fc = result.field_comparisons["name"]
        assert fc.result in (ComparisonResult.PARTIAL_MATCH, ComparisonResult.MISMATCH)
        assert fc.requires_review is True

    def test_mismatch_in_result(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "AAAA"},
            {"name": "ZZZZ"},
        )
        assert "name" in result.mismatch_fields or result.field_comparisons["name"].requires_review


# ---------------------------------------------------------------------------
# DualPassComparator — empty values
# ---------------------------------------------------------------------------


class TestEmptyValues:

    def test_both_empty(self):
        comp = DualPassComparator()
        result = comp.compare({"name": None}, {"name": None})
        fc = result.field_comparisons["name"]
        assert fc.result == ComparisonResult.BOTH_EMPTY

    def test_pass1_only(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "John"},
            {"name": None},
        )
        fc = result.field_comparisons["name"]
        assert fc.result == ComparisonResult.PASS1_ONLY
        assert fc.merged_value == "John"

    def test_pass2_only(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": None},
            {"name": "Jane"},
        )
        fc = result.field_comparisons["name"]
        assert fc.result == ComparisonResult.PASS2_ONLY
        assert fc.merged_value == "Jane"

    def test_empty_string_treated_as_empty(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "   "},
            {"name": "Jane"},
        )
        fc = result.field_comparisons["name"]
        assert fc.result == ComparisonResult.PASS2_ONLY

    def test_empty_list_treated_as_empty(self):
        comp = DualPassComparator()
        result = comp.compare({"codes": []}, {"codes": []})
        fc = result.field_comparisons["codes"]
        assert fc.result == ComparisonResult.BOTH_EMPTY


# ---------------------------------------------------------------------------
# DualPassComparator — union of fields
# ---------------------------------------------------------------------------


class TestFieldUnion:

    def test_fields_from_both_passes(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "John", "dob": "1990-01-01"},
            {"name": "John", "npi": "1234567890"},
        )
        assert "name" in result.field_comparisons
        assert "dob" in result.field_comparisons
        assert "npi" in result.field_comparisons


# ---------------------------------------------------------------------------
# DualPassComparator — merge strategies
# ---------------------------------------------------------------------------


class TestMergeStrategies:

    def test_prefer_pass1(self):
        comp = DualPassComparator(default_strategy=MergeStrategy.PREFER_PASS1)
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John S."},
        )
        fc = result.field_comparisons["name"]
        assert fc.merged_value == "John Smith"

    def test_prefer_pass2(self):
        comp = DualPassComparator(default_strategy=MergeStrategy.PREFER_PASS2)
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John S."},
        )
        fc = result.field_comparisons["name"]
        assert fc.merged_value == "John S."

    def test_prefer_longer(self):
        comp = DualPassComparator(default_strategy=MergeStrategy.PREFER_LONGER)
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "J. Smith"},
        )
        fc = result.field_comparisons["name"]
        assert fc.merged_value == "John Smith"

    def test_require_agreement_on_match(self):
        comp = DualPassComparator(default_strategy=MergeStrategy.REQUIRE_AGREEMENT)
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John Smith"},
        )
        fc = result.field_comparisons["name"]
        assert fc.merged_value == "John Smith"

    def test_require_agreement_on_mismatch(self):
        comp = DualPassComparator(default_strategy=MergeStrategy.REQUIRE_AGREEMENT)
        result = comp.compare(
            {"name": "Alice"},
            {"name": "Bob"},
        )
        fc = result.field_comparisons["name"]
        # Mismatch → merged_value should be None
        assert fc.merged_value is None

    def test_field_specific_strategy(self):
        comp = DualPassComparator(
            default_strategy=MergeStrategy.PREFER_HIGHER_CONFIDENCE,
            field_strategies={"name": MergeStrategy.PREFER_PASS2},
        )
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John S."},
            pass1_confidence={"name": 0.9},
            pass2_confidence={"name": 0.5},
        )
        fc = result.field_comparisons["name"]
        # Field override to PREFER_PASS2
        assert fc.merged_value == "John S."


# ---------------------------------------------------------------------------
# DualPassComparator — confidence from passes
# ---------------------------------------------------------------------------


class TestConfidenceFromPasses:

    def test_prefer_higher_confidence_default(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"name": "John Smith"},
            {"name": "John S."},
            pass1_confidence={"name": 0.95},
            pass2_confidence={"name": 0.60},
        )
        fc = result.field_comparisons["name"]
        # Default strategy PREFER_HIGHER_CONFIDENCE → pass1
        assert fc.merged_value == "John Smith"


# ---------------------------------------------------------------------------
# DualPassComparator — required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:

    def test_required_field_both_empty_flags_review(self):
        comp = DualPassComparator(required_fields=["npi"])
        result = comp.compare({"npi": None}, {"npi": None})
        fc = result.field_comparisons["npi"]
        assert fc.requires_review is True

    def test_required_field_low_confidence_triggers_review(self):
        comp = DualPassComparator(required_fields=["name"])
        result = comp.compare(
            {"name": "AAAA"},
            {"name": "ZZZZ"},
            pass1_confidence={"name": 0.2},
            pass2_confidence={"name": 0.2},
        )
        assert result.requires_human_review is True


# ---------------------------------------------------------------------------
# DualPassComparator — agreement rate thresholds
# ---------------------------------------------------------------------------


class TestAgreementRate:

    def test_high_agreement_no_retry(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"a": "x", "b": "y", "c": "z"},
            {"a": "x", "b": "y", "c": "z"},
        )
        assert result.overall_agreement_rate == 1.0
        assert result.requires_retry is False
        assert result.requires_human_review is False

    def test_low_agreement_triggers_retry_or_review(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"a": "AAA", "b": "BBB", "c": "CCC"},
            {"a": "XXX", "b": "YYY", "c": "ZZZ"},
        )
        # With all mismatches, agreement < 0.50 → human review
        assert result.requires_human_review is True or result.requires_retry is True


# ---------------------------------------------------------------------------
# DualPassComparator — numeric similarity
# ---------------------------------------------------------------------------


class TestNumericSimilarity:

    def test_equal_numbers(self):
        comp = DualPassComparator()
        result = comp.compare({"amt": 100}, {"amt": 100})
        assert result.field_comparisons["amt"].similarity_score == 1.0

    def test_close_numbers(self):
        comp = DualPassComparator()
        result = comp.compare({"amt": 100.0}, {"amt": 100.5})
        fc = result.field_comparisons["amt"]
        assert fc.similarity_score >= 0.85

    def test_very_different_numbers(self):
        comp = DualPassComparator()
        result = comp.compare({"amt": 10}, {"amt": 1000})
        fc = result.field_comparisons["amt"]
        assert fc.similarity_score < 0.50


# ---------------------------------------------------------------------------
# DualPassComparator — notes
# ---------------------------------------------------------------------------


class TestNotes:

    def test_exact_match_note(self):
        comp = DualPassComparator()
        result = comp.compare({"a": "hello"}, {"a": "hello"})
        assert "Exact match" in result.field_comparisons["a"].notes

    def test_confidence_difference_note(self):
        comp = DualPassComparator()
        result = comp.compare(
            {"a": "hello"},
            {"a": "world"},
            pass1_confidence={"a": 0.95},
            pass2_confidence={"a": 0.50},
        )
        assert "higher confidence" in result.field_comparisons["a"].notes


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestCompareExtractions:

    def test_basic_usage(self):
        result = compare_extractions(
            pass1_data={"patient_name": "John Smith"},
            pass2_data={"patient_name": "John Smith"},
        )
        assert isinstance(result, DualPassResult)
        assert result.overall_agreement_rate == 1.0

    def test_with_required_fields(self):
        result = compare_extractions(
            pass1_data={"name": None},
            pass2_data={"name": None},
            required_fields=["name"],
        )
        fc = result.field_comparisons["name"]
        assert fc.requires_review is True
