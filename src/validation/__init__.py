"""
Validation module for anti-hallucination and data quality assurance.

Implements the 3-Layer Anti-Hallucination System:
- Layer 1: Prompt Engineering (handled by prompts module)
- Layer 2: Dual-Pass Extraction (dual_pass.py)
- Layer 3: Pattern + Rule Validation (pattern_detector.py, cross_field.py)

Additional components:
- confidence.py: Confidence scoring and thresholds
- medical_codes.py: Medical code validation (CPT, ICD-10, NPI)
- human_review.py: Human review queue management

Usage:
    from src.validation import (
        DualPassComparator,
        compare_extractions,
        HallucinationPatternDetector,
        detect_hallucination_patterns,
        ConfidenceScorer,
        calculate_confidence,
        CrossFieldValidator,
        validate_cross_fields,
        MedicalCodeValidationEngine,
        validate_medical_codes,
        HumanReviewQueue,
        create_review_task,
    )
"""

# Dual-pass comparison
from src.validation.dual_pass import (
    ComparisonResult,
    MergeStrategy,
    FieldComparison,
    DualPassResult,
    DualPassComparator,
    compare_extractions,
)

# Hallucination pattern detection
from src.validation.pattern_detector import (
    HallucinationPattern,
    PatternSeverity,
    PatternMatch,
    PatternDetectionResult,
    HallucinationPatternDetector,
    detect_hallucination_patterns,
)

# Confidence scoring
from src.validation.confidence import (
    ConfidenceLevel,
    ConfidenceAction,
    FieldConfidence,
    ExtractionConfidence,
    ConfidenceScorer,
    AdaptiveConfidenceScorer,
    calculate_confidence,
    get_confidence_level,
)

# Cross-field validation
from src.validation.cross_field import (
    RuleType,
    RuleSeverity,
    RuleViolation,
    CrossFieldResult,
    CrossFieldRule,
    CrossFieldValidator,
    MedicalDocumentRules,
    validate_cross_fields,
)

# Medical code validation
from src.validation.medical_codes import (
    CodeType,
    CodeValidationStatus,
    CodeValidationDetail,
    MedicalCodeValidationResult,
    MedicalCodeValidationEngine,
    validate_medical_codes,
)

# Human review queue
from src.validation.human_review import (
    ReviewPriority,
    ReviewStatus,
    ReviewReason,
    ReviewField,
    ReviewTask,
    HumanReviewQueue,
    create_review_task,
)


__all__ = [
    # Dual-pass comparison
    "ComparisonResult",
    "MergeStrategy",
    "FieldComparison",
    "DualPassResult",
    "DualPassComparator",
    "compare_extractions",
    # Hallucination pattern detection
    "HallucinationPattern",
    "PatternSeverity",
    "PatternMatch",
    "PatternDetectionResult",
    "HallucinationPatternDetector",
    "detect_hallucination_patterns",
    # Confidence scoring
    "ConfidenceLevel",
    "ConfidenceAction",
    "FieldConfidence",
    "ExtractionConfidence",
    "ConfidenceScorer",
    "AdaptiveConfidenceScorer",
    "calculate_confidence",
    "get_confidence_level",
    # Cross-field validation
    "RuleType",
    "RuleSeverity",
    "RuleViolation",
    "CrossFieldResult",
    "CrossFieldRule",
    "CrossFieldValidator",
    "MedicalDocumentRules",
    "validate_cross_fields",
    # Medical code validation
    "CodeType",
    "CodeValidationStatus",
    "CodeValidationDetail",
    "MedicalCodeValidationResult",
    "MedicalCodeValidationEngine",
    "validate_medical_codes",
    # Human review queue
    "ReviewPriority",
    "ReviewStatus",
    "ReviewReason",
    "ReviewField",
    "ReviewTask",
    "HumanReviewQueue",
    "create_review_task",
]
