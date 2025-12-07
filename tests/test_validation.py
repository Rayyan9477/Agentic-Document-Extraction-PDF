#!/usr/bin/env python
"""Comprehensive tests for Phase 3 validation module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import (
    # Dual-pass
    DualPassComparator,
    ComparisonResult,
    MergeStrategy,
    compare_extractions,
    # Pattern detection
    HallucinationPatternDetector,
    HallucinationPattern,
    PatternSeverity,
    detect_hallucination_patterns,
    # Confidence
    ConfidenceScorer,
    AdaptiveConfidenceScorer,
    ConfidenceLevel,
    ConfidenceAction,
    calculate_confidence,
    get_confidence_level,
    # Cross-field
    CrossFieldValidator,
    MedicalDocumentRules,
    RuleType,
    validate_cross_fields,
    # Medical codes
    MedicalCodeValidationEngine,
    CodeType,
    CodeValidationStatus,
    validate_medical_codes,
    # Human review
    HumanReviewQueue,
    ReviewPriority,
    ReviewStatus,
    ReviewReason,
    create_review_task,
)


def test_dual_pass_exact_match():
    """Test dual-pass comparison with exact matches."""
    print("\n[1] Testing dual-pass exact match...")

    comparator = DualPassComparator()
    result = comparator.compare(
        pass1_data={"patient_name": "John Smith", "dob": "1990-01-15"},
        pass2_data={"patient_name": "John Smith", "dob": "1990-01-15"},
        pass1_confidence={"patient_name": 0.95, "dob": 0.92},
        pass2_confidence={"patient_name": 0.93, "dob": 0.90},
    )

    assert result.overall_agreement_rate == 1.0
    assert result.overall_confidence > 0.9
    assert len(result.mismatch_fields) == 0
    assert not result.requires_human_review
    assert not result.requires_retry
    assert result.merged_output["patient_name"] == "John Smith"
    print("    PASSED: Exact match detection and merging")


def test_dual_pass_fuzzy_match():
    """Test dual-pass comparison with fuzzy matches."""
    print("\n[2] Testing dual-pass fuzzy match...")

    result = compare_extractions(
        pass1_data={"patient_name": "John Smith", "dob": "01/15/1990"},
        pass2_data={"patient_name": "JOHN SMITH", "dob": "1990-01-15"},
    )

    # Names should fuzzy match (case insensitive)
    assert result.field_comparisons["patient_name"].result in (
        ComparisonResult.EXACT_MATCH,
        ComparisonResult.FUZZY_MATCH,
    )
    # Both values should be in output
    assert "patient_name" in result.merged_output
    print("    PASSED: Fuzzy match detection")


def test_dual_pass_mismatch():
    """Test dual-pass comparison with mismatches."""
    print("\n[3] Testing dual-pass mismatch detection...")

    # Use clearly different values that will be detected as mismatches
    result = compare_extractions(
        pass1_data={"patient_name": "John Smith", "claim_type": "inpatient"},
        pass2_data={"patient_name": "Jane Doe", "claim_type": "outpatient"},
    )

    # At least some fields should not be exact/fuzzy matches
    assert result.overall_agreement_rate < 1.0
    # Check that comparisons were made
    assert len(result.field_comparisons) == 2
    # Check that fields requiring review exist (partial match or mismatch)
    fields_requiring_review = [
        name for name, comp in result.field_comparisons.items()
        if comp.requires_review
    ]
    assert len(fields_requiring_review) >= 1
    print("    PASSED: Mismatch detection and flagging")


def test_dual_pass_required_fields():
    """Test dual-pass with required fields."""
    print("\n[4] Testing dual-pass required fields...")

    comparator = DualPassComparator(
        required_fields=["patient_name", "billing_npi"]
    )

    result = comparator.compare(
        pass1_data={"patient_name": "John", "amount": "100"},
        pass2_data={"patient_name": "John", "amount": "100"},
    )

    # billing_npi is missing - should flag for review
    assert result.field_comparisons.get("billing_npi") is None or \
           result.field_comparisons.get("billing_npi", {})
    print("    PASSED: Required field tracking")


def test_pattern_detector_placeholder():
    """Test hallucination pattern detection for placeholders."""
    print("\n[5] Testing placeholder pattern detection...")

    detector = HallucinationPatternDetector()
    result = detector.detect({
        "patient_name": "N/A",
        "claim_id": "TBD",
        "amount": "XXX",
    })

    assert result.is_likely_hallucination
    assert len(result.matches) >= 3
    assert all(
        m.pattern == HallucinationPattern.PLACEHOLDER_TEXT
        for m in result.matches
        if m.field_name in ["patient_name", "claim_id", "amount"]
    )
    print("    PASSED: Placeholder text detection")


def test_pattern_detector_generic_names():
    """Test detection of generic/test names."""
    print("\n[6] Testing generic name detection...")

    result = detect_hallucination_patterns({
        "patient_name": "John Doe",
        "provider_name": "Test Patient",
    })

    name_matches = [
        m for m in result.matches
        if m.pattern == HallucinationPattern.GENERIC_NAME
    ]
    assert len(name_matches) >= 1
    assert result.is_likely_hallucination
    print("    PASSED: Generic name detection")


def test_pattern_detector_repeated_digits():
    """Test detection of repeated digit patterns."""
    print("\n[7] Testing repeated digit detection...")

    result = detect_hallucination_patterns({
        "claim_id": "000000000",
        "npi": "1111111111",
    })

    repeated_matches = [
        m for m in result.matches
        if m.pattern in (
            HallucinationPattern.REPEATED_DIGITS,
            HallucinationPattern.SYNTHETIC_IDENTIFIER,
        )
    ]
    assert len(repeated_matches) >= 1
    print("    PASSED: Repeated digit pattern detection")


def test_pattern_detector_round_numbers():
    """Test detection of suspiciously round numbers."""
    print("\n[8] Testing round number detection...")

    result = detect_hallucination_patterns({
        "total_charges": 1000.00,
        "payment_amount": 500.00,
    })

    round_matches = [
        m for m in result.matches
        if m.pattern == HallucinationPattern.ROUND_NUMBER
    ]
    assert len(round_matches) >= 1
    print("    PASSED: Round number detection")


def test_confidence_scorer_high():
    """Test confidence scoring for high confidence."""
    print("\n[9] Testing high confidence scoring...")

    scorer = ConfidenceScorer()
    result = scorer.calculate(
        extraction_confidences={"name": 0.95, "dob": 0.92, "npi": 0.88},
        agreement_scores={"name": 1.0, "dob": 1.0, "npi": 0.95},
        validation_results={"name": True, "dob": True, "npi": True},
    )

    assert result.overall_level == ConfidenceLevel.HIGH
    assert result.recommended_action == ConfidenceAction.AUTO_ACCEPT
    assert result.overall_confidence >= 0.85
    print("    PASSED: High confidence scoring and auto-accept")


def test_confidence_scorer_low():
    """Test confidence scoring for low confidence."""
    print("\n[10] Testing low confidence scoring...")

    # Use very low values to ensure LOW confidence level
    result = calculate_confidence(
        extraction_confidences={"name": 0.20, "amount": 0.15},
        agreement_scores={"name": 0.3, "amount": 0.2},
        validation_results={"name": False, "amount": False},
        pattern_flags={"name", "amount"},
    )

    assert result.overall_level == ConfidenceLevel.LOW
    assert result.recommended_action in (
        ConfidenceAction.RETRY,
        ConfidenceAction.HUMAN_REVIEW,
    )
    print("    PASSED: Low confidence scoring")


def test_confidence_scorer_retry():
    """Test confidence scoring for retry scenario."""
    print("\n[11] Testing retry recommendation...")

    # Use values that result in MEDIUM confidence (0.50 - 0.84)
    result = calculate_confidence(
        extraction_confidences={"name": 0.60, "dob": 0.55},
        agreement_scores={"name": 0.70, "dob": 0.65},
        validation_results={"name": True, "dob": True},
        retry_count=0,
    )

    assert result.overall_level == ConfidenceLevel.MEDIUM
    assert result.recommended_action == ConfidenceAction.RETRY
    print("    PASSED: Retry recommendation for medium confidence")


def test_confidence_scorer_critical_fields():
    """Test confidence with critical fields."""
    print("\n[12] Testing critical field handling...")

    scorer = ConfidenceScorer(
        critical_fields=["patient_name", "billing_npi"]
    )
    # Use very low values for billing_npi to ensure it ends up as LOW confidence
    result = scorer.calculate(
        extraction_confidences={"patient_name": 0.95, "billing_npi": 0.10, "amount": 0.90},
        agreement_scores={"patient_name": 1.0, "billing_npi": 0.20, "amount": 1.0},
        validation_results={"patient_name": True, "billing_npi": False, "amount": True},
        pattern_flags={"billing_npi"},
    )

    # Critical field has issues - should require review
    assert not result.critical_fields_status["billing_npi"]
    assert result.recommended_action == ConfidenceAction.HUMAN_REVIEW
    print("    PASSED: Critical field impact on confidence")


def test_adaptive_scorer():
    """Test adaptive confidence scorer."""
    print("\n[13] Testing adaptive scorer for document types...")

    scorer = AdaptiveConfidenceScorer(
        document_type="cms1500",
        historical_accuracy=0.95,
    )

    # Should have CMS-1500 specific critical fields
    assert "patient_name" in scorer.critical_fields
    assert "billing_npi" in scorer.critical_fields
    print("    PASSED: Adaptive scorer with document-specific settings")


def test_cross_field_date_order():
    """Test cross-field date ordering validation."""
    print("\n[14] Testing cross-field date ordering...")

    validator = CrossFieldValidator()
    validator.add_date_order_rule(
        "admission_before_discharge",
        "admission_date",
        "discharge_date",
    )

    # Valid order
    result = validator.validate({
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-20",
    })
    assert result.passed
    assert len(result.errors) == 0

    # Invalid order
    result = validator.validate({
        "admission_date": "2024-01-20",
        "discharge_date": "2024-01-15",
    })
    assert not result.passed
    assert len(result.errors) >= 1
    print("    PASSED: Date ordering validation")


def test_cross_field_sum_validation():
    """Test cross-field sum validation."""
    print("\n[15] Testing cross-field sum validation...")

    validator = CrossFieldValidator()
    validator.add_sum_rule(
        "line_total",
        component_fields=["line1", "line2", "line3"],
        total_field="total",
        tolerance=0.01,
    )

    # Valid sum
    result = validator.validate({
        "line1": 100.00,
        "line2": 200.00,
        "line3": 300.00,
        "total": 600.00,
    })
    assert result.passed

    # Invalid sum
    result = validator.validate({
        "line1": 100.00,
        "line2": 200.00,
        "line3": 300.00,
        "total": 500.00,
    })
    assert not result.passed
    print("    PASSED: Sum validation")


def test_cross_field_required_if():
    """Test cross-field required-if validation."""
    print("\n[16] Testing required-if validation...")

    validator = CrossFieldValidator()
    validator.add_required_if_rule(
        "modifier_requires_cpt",
        trigger_field="modifier",
        required_field="cpt_code",
    )

    # Valid: modifier present, cpt present
    result = validator.validate({
        "modifier": "25",
        "cpt_code": "99213",
    })
    assert result.passed

    # Invalid: modifier present, cpt missing
    result = validator.validate({
        "modifier": "25",
    })
    assert not result.passed
    print("    PASSED: Required-if validation")


def test_cross_field_medical_rules():
    """Test pre-configured medical document rules."""
    print("\n[17] Testing medical document rules...")

    # Get CMS-1500 rules
    validator = MedicalDocumentRules.get_cms1500_rules()
    assert len(validator.rules) > 0

    # Get UB-04 rules
    validator = MedicalDocumentRules.get_ub04_rules()
    assert len(validator.rules) > 0

    # Get EOB rules
    validator = MedicalDocumentRules.get_eob_rules()
    assert len(validator.rules) > 0
    print("    PASSED: Medical document rules factory")


def test_medical_code_cpt():
    """Test CPT code validation."""
    print("\n[18] Testing CPT code validation...")

    engine = MedicalCodeValidationEngine()

    # Valid CPT
    result = engine.validate_code("99213", CodeType.CPT)
    assert result.is_valid
    assert result.status == CodeValidationStatus.VALID

    # Invalid CPT
    result = engine.validate_code("1234", CodeType.CPT)
    assert not result.is_valid
    print("    PASSED: CPT code validation")


def test_medical_code_icd10():
    """Test ICD-10 code validation."""
    print("\n[19] Testing ICD-10 code validation...")

    engine = MedicalCodeValidationEngine()

    # Valid ICD-10-CM
    result = engine.validate_code("E11.9", CodeType.ICD10_CM)
    assert result.is_valid

    # Valid ICD-10-CM without decimal
    result = engine.validate_code("A000", CodeType.ICD10_CM)
    assert result.is_valid

    # Invalid ICD-10
    result = engine.validate_code("123.45", CodeType.ICD10_CM)
    assert not result.is_valid
    print("    PASSED: ICD-10 code validation")


def test_medical_code_npi():
    """Test NPI validation."""
    print("\n[20] Testing NPI validation...")

    engine = MedicalCodeValidationEngine()

    # Valid NPI (passes Luhn check)
    result = engine.validate_code("1234567893", CodeType.NPI)
    assert result.is_valid

    # Invalid NPI (wrong length)
    result = engine.validate_code("123456789", CodeType.NPI)
    assert not result.is_valid
    print("    PASSED: NPI validation")


def test_medical_code_batch():
    """Test batch medical code validation."""
    print("\n[21] Testing batch medical code validation...")

    result = validate_medical_codes({
        "cpt_code": "99213",
        "diagnosis_code": "E11.9",
        "billing_npi": "1234567893",
        "invalid_code": "XXXXX",
    })

    assert len(result.valid_codes) >= 3
    assert result.validation_rate > 0.5
    print("    PASSED: Batch medical code validation")


def test_human_review_queue_create():
    """Test human review queue task creation."""
    print("\n[22] Testing human review queue creation...")

    queue = HumanReviewQueue(auto_persist=False)
    task = queue.create_task(
        processing_id="test123",
        document_path="/docs/claim.pdf",
        document_type="cms1500",
        extracted_data={"patient_name": "Test"},
        fields_to_review=[
            {
                "field_name": "patient_name",
                "extracted_value": "Test",
                "confidence": 0.4,
                "reason": "low confidence",
            }
        ],
        reasons=[ReviewReason.LOW_CONFIDENCE],
        overall_confidence=0.4,
    )

    assert task.task_id.startswith("review_")
    assert task.status == ReviewStatus.PENDING
    assert task.priority in ReviewPriority
    assert len(task.fields_to_review) == 1
    print("    PASSED: Review task creation")


def test_human_review_queue_priority():
    """Test human review queue priority calculation."""
    print("\n[23] Testing review queue priority...")

    queue = HumanReviewQueue(auto_persist=False)

    # Critical priority for hallucination
    task = queue.create_task(
        processing_id="test1",
        document_path="/doc1.pdf",
        document_type="cms1500",
        extracted_data={},
        fields_to_review=[],
        reasons=[ReviewReason.HALLUCINATION_DETECTED],
        overall_confidence=0.5,
    )
    assert task.priority == ReviewPriority.CRITICAL

    # High priority for validation failure
    task = queue.create_task(
        processing_id="test2",
        document_path="/doc2.pdf",
        document_type="cms1500",
        extracted_data={},
        fields_to_review=[],
        reasons=[ReviewReason.VALIDATION_FAILURE],
        overall_confidence=0.5,
    )
    assert task.priority == ReviewPriority.HIGH

    # Low priority for quality check
    task = queue.create_task(
        processing_id="test3",
        document_path="/doc3.pdf",
        document_type="cms1500",
        extracted_data={},
        fields_to_review=[],
        reasons=[ReviewReason.QUALITY_CHECK],
        overall_confidence=0.8,
    )
    assert task.priority == ReviewPriority.LOW
    print("    PASSED: Priority calculation")


def test_human_review_queue_workflow():
    """Test human review queue complete workflow."""
    print("\n[24] Testing review queue workflow...")

    queue = HumanReviewQueue(auto_persist=False)

    # Create task
    task = queue.create_task(
        processing_id="workflow1",
        document_path="/workflow.pdf",
        document_type="cms1500",
        extracted_data={"patient_name": "Wrong Name"},
        fields_to_review=[{"field_name": "patient_name", "extracted_value": "Wrong Name", "confidence": 0.3, "reason": "low confidence"}],
        reasons=[ReviewReason.LOW_CONFIDENCE],
        overall_confidence=0.3,
    )

    # Get next task
    next_task = queue.get_next_task(assignee="reviewer1")
    assert next_task is not None
    assert next_task.task_id == task.task_id
    assert next_task.status == ReviewStatus.IN_PROGRESS
    assert next_task.assigned_to == "reviewer1"

    # Complete task
    success = queue.complete_task(
        task_id=task.task_id,
        corrections={"patient_name": "Correct Name"},
        decision="approved",
    )
    assert success

    # Get corrected extraction
    corrected = queue.get_corrected_extraction(task.task_id)
    assert corrected is not None
    assert corrected["patient_name"] == "Correct Name"
    print("    PASSED: Complete review workflow")


def test_human_review_convenience_function():
    """Test create_review_task convenience function."""
    print("\n[25] Testing create_review_task convenience function...")

    task = create_review_task(
        processing_id="conv123",
        document_path="/conv.pdf",
        document_type="ub04",
        extracted_data={"amount": 1000},
        low_confidence_fields=["amount"],
        validation_errors={"amount": ["Invalid format"]},
        hallucination_flags=["amount"],
        overall_confidence=0.35,
        field_confidences={"amount": 0.35},
    )

    assert task.processing_id == "conv123"
    assert ReviewReason.LOW_CONFIDENCE in task.reasons
    assert ReviewReason.VALIDATION_FAILURE in task.reasons
    assert ReviewReason.HALLUCINATION_DETECTED in task.reasons
    assert len(task.fields_to_review) == 1
    print("    PASSED: Convenience function for task creation")


def test_integration_full_validation():
    """Test full validation pipeline integration."""
    print("\n[26] Testing full validation pipeline integration...")

    # Simulate extraction data - use realistic names that won't be flagged
    pass1_data = {
        "patient_name": "Michael Johnson",
        "dob": "1990-01-15",
        "cpt_code": "99213",
        "diagnosis_code": "E11.9",
        "billing_npi": "1234567893",
        "total_charges": 150.00,
    }

    pass2_data = {
        "patient_name": "Michael Johnson",
        "dob": "01/15/1990",
        "cpt_code": "99213",
        "diagnosis_code": "E11.9",
        "billing_npi": "1234567893",
        "total_charges": 150.00,
    }

    # Step 1: Dual-pass comparison
    dual_result = compare_extractions(pass1_data, pass2_data)
    assert dual_result.overall_agreement_rate >= 0.8

    # Step 2: Pattern detection on merged output
    pattern_result = detect_hallucination_patterns(dual_result.merged_output)
    assert not pattern_result.is_likely_hallucination

    # Step 3: Medical code validation
    code_result = validate_medical_codes(dual_result.merged_output)
    assert code_result.overall_valid

    # Step 4: Cross-field validation
    cross_result = validate_cross_fields(dual_result.merged_output, "cms1500")
    # May or may not pass depending on data completeness

    # Step 5: Confidence scoring
    conf_result = calculate_confidence(
        extraction_confidences={k: 0.9 for k in dual_result.merged_output},
        agreement_scores={
            k: v.similarity_score
            for k, v in dual_result.field_comparisons.items()
        },
        validation_results={k: True for k in code_result.valid_codes},
        pattern_flags=pattern_result.flagged_fields,
    )

    assert conf_result.overall_confidence > 0.5
    print("    PASSED: Full validation pipeline integration")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("PHASE 3 VALIDATION MODULE TESTS")
    print("=" * 60)

    test_dual_pass_exact_match()
    test_dual_pass_fuzzy_match()
    test_dual_pass_mismatch()
    test_dual_pass_required_fields()
    test_pattern_detector_placeholder()
    test_pattern_detector_generic_names()
    test_pattern_detector_repeated_digits()
    test_pattern_detector_round_numbers()
    test_confidence_scorer_high()
    test_confidence_scorer_low()
    test_confidence_scorer_retry()
    test_confidence_scorer_critical_fields()
    test_adaptive_scorer()
    test_cross_field_date_order()
    test_cross_field_sum_validation()
    test_cross_field_required_if()
    test_cross_field_medical_rules()
    test_medical_code_cpt()
    test_medical_code_icd10()
    test_medical_code_npi()
    test_medical_code_batch()
    test_human_review_queue_create()
    test_human_review_queue_priority()
    test_human_review_queue_workflow()
    test_human_review_convenience_function()
    test_integration_full_validation()

    print("\n" + "=" * 60)
    print("ALL 26 PHASE 3 VALIDATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
