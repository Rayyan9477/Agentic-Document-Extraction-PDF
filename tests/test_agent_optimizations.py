"""
Test scenarios for optimized agents.

These tests validate the improvements made to the extraction pipeline:
1. Chain-of-thought extraction reasoning
2. Anti-hallucination pattern detection
3. Enhanced dual-pass verification
4. Few-shot classification examples
5. Retry logic with exponential backoff
6. Shared utilities
"""

import pytest

from src.agents.utils import (
    RetryConfig,
    build_custom_schema,
    calculate_extraction_quality_score,
    identify_disagreement_fields,
    identify_low_confidence_fields,
    retry_with_backoff,
)
from src.prompts.classification import (
    CLASSIFICATION_EXAMPLES,
    build_classification_prompt,
)
from src.prompts.extraction import (
    EXTRACTION_ANTI_PATTERNS,
    EXTRACTION_REASONING_TEMPLATE,
    build_extraction_prompt,
    build_verification_prompt,
)
from src.prompts.grounding_rules import (
    CHAIN_OF_THOUGHT_TEMPLATE,
    CONSTITUTIONAL_CRITIQUE,
    FEW_SHOT_EXAMPLES,
    GROUNDING_RULES,
    SELF_VERIFICATION_CHECKPOINT,
    build_enhanced_system_prompt,
    build_grounded_system_prompt,
)
from src.prompts.validation import (
    CONFIDENCE_CALIBRATION_EXAMPLES,
    CONSTITUTIONAL_VALIDATION_PRINCIPLES,
    build_validation_prompt,
)


class TestGroundingRulesEnhancements:
    """Test enhancements to grounding rules prompts."""

    def test_grounding_rules_contains_core_principles(self):
        """Verify grounding rules contain all core principles."""
        assert "VISUAL GROUNDING" in GROUNDING_RULES
        assert "NO GUESSING" in GROUNDING_RULES
        assert "NO INFERENCE" in GROUNDING_RULES
        assert "NO DEFAULTS" in GROUNDING_RULES
        assert "CONFIDENCE SCORING" in GROUNDING_RULES
        assert "LOCATION DESCRIPTION" in GROUNDING_RULES
        assert "UNCERTAINTY HANDLING" in GROUNDING_RULES

    def test_chain_of_thought_template_structure(self):
        """Verify chain-of-thought template has proper steps."""
        assert "LOCATE" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "READ" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "VERIFY" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "CONFIDENCE" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "EXTRACT or SKIP" in CHAIN_OF_THOUGHT_TEMPLATE

    def test_self_verification_checkpoint(self):
        """Verify self-verification checklist is comprehensive."""
        assert "Visual Verification" in SELF_VERIFICATION_CHECKPOINT
        assert "Character Check" in SELF_VERIFICATION_CHECKPOINT
        assert "Hallucination Check" in SELF_VERIFICATION_CHECKPOINT
        assert "Null Check" in SELF_VERIFICATION_CHECKPOINT
        assert "Confidence Calibration" in SELF_VERIFICATION_CHECKPOINT

    def test_few_shot_examples_include_good_and_bad(self):
        """Verify few-shot examples include positive and negative cases."""
        assert "GOOD EXTRACTION EXAMPLE" in FEW_SHOT_EXAMPLES
        assert "GOOD NULL EXAMPLE" in FEW_SHOT_EXAMPLES
        assert "BAD EXTRACTION EXAMPLE" in FEW_SHOT_EXAMPLES
        assert "BAD PLACEHOLDER EXAMPLE" in FEW_SHOT_EXAMPLES
        assert "DO NOT DO THIS" in FEW_SHOT_EXAMPLES

    def test_constitutional_critique_questions(self):
        """Verify constitutional critique has key questions."""
        assert "visible" in CONSTITUTIONAL_CRITIQUE.lower()
        assert "overconfident" in CONSTITUTIONAL_CRITIQUE.lower()
        assert "null" in CONSTITUTIONAL_CRITIQUE.lower()
        assert (
            "suspicious" in CONSTITUTIONAL_CRITIQUE.lower()
            or "perfect" in CONSTITUTIONAL_CRITIQUE.lower()
        )

    def test_build_grounded_system_prompt_options(self):
        """Test that enhanced options work correctly."""
        # Basic prompt
        basic = build_grounded_system_prompt()
        assert "GROUNDING RULES" in basic
        assert "FORBIDDEN ACTIONS" in basic
        assert "CONFIDENCE SCORE GUIDELINES" in basic

        # Enhanced prompt with all options
        enhanced = build_grounded_system_prompt(
            include_chain_of_thought=True,
            include_few_shot_examples=True,
            include_self_verification=True,
            include_constitutional_critique=True,
        )
        assert "EXTRACTION REASONING PROTOCOL" in enhanced
        assert "EXTRACTION EXAMPLES" in enhanced
        assert "SELF-VERIFICATION CHECKLIST" in enhanced
        assert "SELF-CRITIQUE PROTOCOL" in enhanced

    def test_build_enhanced_system_prompt_for_first_pass(self):
        """Test enhanced prompt for first extraction pass."""
        prompt = build_enhanced_system_prompt(
            document_type="CMS-1500",
            is_verification_pass=False,
        )
        assert "GROUNDING RULES" in prompt
        assert "CMS-1500" in prompt
        # First pass should have examples but not constitutional critique
        assert "EXTRACTION EXAMPLES" in prompt

    def test_build_enhanced_system_prompt_for_verification_pass(self):
        """Test enhanced prompt for verification pass."""
        prompt = build_enhanced_system_prompt(
            document_type="UB-04",
            is_verification_pass=True,
        )
        assert "GROUNDING RULES" in prompt
        assert "UB-04" in prompt
        # Verification pass should have constitutional critique
        assert "SELF-CRITIQUE PROTOCOL" in prompt


class TestClassificationPromptEnhancements:
    """Test enhancements to classification prompts."""

    def test_classification_examples_coverage(self):
        """Verify classification examples cover all document types."""
        assert "CMS-1500" in CLASSIFICATION_EXAMPLES
        assert "UB-04" in CLASSIFICATION_EXAMPLES
        assert "EOB" in CLASSIFICATION_EXAMPLES
        assert "Superbill" in CLASSIFICATION_EXAMPLES or "SUPERBILL" in CLASSIFICATION_EXAMPLES
        assert "OTHER" in CLASSIFICATION_EXAMPLES

    def test_build_classification_prompt_with_examples(self):
        """Test classification prompt includes examples when requested."""
        with_examples = build_classification_prompt(
            include_examples=True,
            include_step_by_step=True,
        )
        assert "CLASSIFICATION EXAMPLES" in with_examples
        assert "STEP-BY-STEP CLASSIFICATION PROTOCOL" in with_examples

    def test_build_classification_prompt_without_examples(self):
        """Test classification prompt can exclude examples."""
        without_examples = build_classification_prompt(
            include_examples=False,
            include_step_by_step=False,
        )
        assert "CLASSIFICATION EXAMPLES" not in without_examples
        assert "STEP-BY-STEP CLASSIFICATION PROTOCOL" not in without_examples


class TestExtractionPromptEnhancements:
    """Test enhancements to extraction prompts."""

    def test_extraction_reasoning_template_structure(self):
        """Verify extraction reasoning has proper steps."""
        assert "LOCATE" in EXTRACTION_REASONING_TEMPLATE
        assert "READ" in EXTRACTION_REASONING_TEMPLATE
        assert "VALIDATE" in EXTRACTION_REASONING_TEMPLATE
        assert "SCORE" in EXTRACTION_REASONING_TEMPLATE.upper()

    def test_extraction_anti_patterns_coverage(self):
        """Verify anti-patterns cover common hallucination types."""
        assert "Calculate" in EXTRACTION_ANTI_PATTERNS or "infer" in EXTRACTION_ANTI_PATTERNS
        assert (
            "Fill in expected" in EXTRACTION_ANTI_PATTERNS
            or "expected patterns" in EXTRACTION_ANTI_PATTERNS
        )
        assert (
            "partial dates" in EXTRACTION_ANTI_PATTERNS.lower()
            or "Complete partial" in EXTRACTION_ANTI_PATTERNS
        )
        assert (
            "typical names" in EXTRACTION_ANTI_PATTERNS.lower()
            or "Assume typical" in EXTRACTION_ANTI_PATTERNS
        )

    def test_build_extraction_prompt_with_enhancements(self):
        """Test extraction prompt includes reasoning and anti-patterns."""
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "test_field", "display_name": "Test", "field_type": "string"}],
            document_type="CMS-1500",
            page_number=1,
            total_pages=1,
            is_first_pass=True,
            include_reasoning=True,
            include_anti_patterns=True,
        )
        assert "EXTRACTION REASONING PROTOCOL" in prompt
        assert "EXTRACTION ANTI-PATTERNS" in prompt

    def test_verification_prompt_is_differentiated(self):
        """Test that verification prompt is sufficiently different from first pass."""
        first_pass = build_extraction_prompt(
            schema_fields=[{"name": "test_field"}],
            document_type="CMS-1500",
            page_number=1,
            total_pages=1,
            is_first_pass=True,
        )

        verification = build_verification_prompt(
            schema_fields=[{"name": "test_field"}],
            document_type="CMS-1500",
            page_number=1,
            first_pass_results={},
        )

        # Verification should have skeptical language
        assert "SKEPTICAL" in verification
        assert "auditor" in verification.lower()
        assert "CHARACTER-BY-CHARACTER" in verification

        # Should have stricter confidence thresholds
        assert "0.90+" in verification or "0.90" in verification

        # Should emphasize null returns
        assert "null" in verification.lower()


class TestValidationPromptEnhancements:
    """Test enhancements to validation prompts."""

    def test_constitutional_principles_coverage(self):
        """Verify constitutional principles are comprehensive."""
        assert "VISUAL EVIDENCE" in CONSTITUTIONAL_VALIDATION_PRINCIPLES
        assert "CHARACTER FIDELITY" in CONSTITUTIONAL_VALIDATION_PRINCIPLES
        assert "CONFIDENCE HONESTY" in CONSTITUTIONAL_VALIDATION_PRINCIPLES
        assert "SKEPTICAL DEFAULT" in CONSTITUTIONAL_VALIDATION_PRINCIPLES
        assert "PATTERN AWARENESS" in CONSTITUTIONAL_VALIDATION_PRINCIPLES

    def test_confidence_calibration_examples_coverage(self):
        """Verify calibration examples cover confidence ranges."""
        assert "HIGH CONFIDENCE" in CONFIDENCE_CALIBRATION_EXAMPLES
        assert "MEDIUM CONFIDENCE" in CONFIDENCE_CALIBRATION_EXAMPLES
        assert "LOW CONFIDENCE" in CONFIDENCE_CALIBRATION_EXAMPLES
        assert "MUST RETURN NULL" in CONFIDENCE_CALIBRATION_EXAMPLES

    def test_build_validation_prompt_with_enhancements(self):
        """Test validation prompt includes constitutional principles."""
        prompt = build_validation_prompt(
            extracted_data={"test_field": {"value": "test", "confidence": 0.9}},
            document_type="CMS-1500",
            schema_rules=[],
            include_constitutional_principles=True,
            include_calibration_examples=True,
        )
        assert "CONSTITUTIONAL VALIDATION PRINCIPLES" in prompt
        assert "CONFIDENCE CALIBRATION EXAMPLES" in prompt
        assert "skeptical auditor" in prompt.lower()


class TestSharedUtilities:
    """Test shared utility functions."""

    def test_build_custom_schema_basic(self):
        """Test building a basic custom schema."""
        schema_def = {
            "name": "test_schema",
            "description": "Test schema",
            "fields": [
                {
                    "name": "patient_name",
                    "display_name": "Patient Name",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "dob",
                    "display_name": "Date of Birth",
                    "type": "date",
                    "required": False,
                },
            ],
        }

        schema = build_custom_schema(schema_def)

        assert schema.name == "test_schema"
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "patient_name"
        assert schema.fields[0].required is True

    def test_build_custom_schema_with_rules(self):
        """Test building schema with cross-field rules."""
        schema_def = {
            "name": "schema_with_rules",
            "fields": [
                {"name": "field1", "type": "string"},
                {"name": "field2", "type": "string"},
            ],
            "rules": [
                {
                    "source_field": "field1",
                    "target_field": "field2",
                    "operator": "requires",
                    "error_message": "field2 required when field1 present",
                },
            ],
        }

        schema = build_custom_schema(schema_def)

        assert len(schema.cross_field_rules) == 1

    def test_identify_low_confidence_fields(self):
        """Test identifying fields with low confidence."""
        field_metadata = {
            "field_high": {"value": "test", "confidence": 0.95},
            "field_medium": {"value": "test", "confidence": 0.75},
            "field_low": {"value": "test", "confidence": 0.5},
            "field_null": {"value": None, "confidence": 0.3},  # Should not be flagged (null value)
        }

        low_conf_fields = identify_low_confidence_fields(field_metadata, threshold=0.7)

        assert "field_low" in low_conf_fields
        assert "field_high" not in low_conf_fields
        assert "field_medium" not in low_conf_fields
        assert "field_null" not in low_conf_fields

    def test_identify_disagreement_fields(self):
        """Test identifying fields where passes disagreed."""
        field_metadata = {
            "field_agree": {"passes_agree": True},
            "field_disagree": {"passes_agree": False},
            "field_no_info": {},
        }

        disagreements = identify_disagreement_fields(field_metadata)

        assert "field_disagree" in disagreements
        assert "field_agree" not in disagreements
        assert "field_no_info" not in disagreements

    def test_calculate_extraction_quality_score(self):
        """Test quality score calculation."""
        # High quality extraction
        high_quality = calculate_extraction_quality_score(
            field_metadata={
                "field1": {"value": "test", "confidence": 0.95},
                "field2": {"value": "test", "confidence": 0.90},
            },
            hallucination_flags=[],
            validation_errors=[],
        )
        assert high_quality > 0.85

        # Low quality with hallucinations
        low_quality = calculate_extraction_quality_score(
            field_metadata={
                "field1": {"value": "test", "confidence": 0.6},
                "field2": {"value": "test", "confidence": 0.5},
            },
            hallucination_flags=["field1", "field2"],
            validation_errors=["Error 1", "Error 2"],
        )
        assert low_quality < high_quality


class TestRetryLogic:
    """Test retry with exponential backoff."""

    def test_retry_config_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=30000,
            exponential_base=2.0,
            jitter=False,
        )

        # First retry: 1000ms
        assert config.get_delay_ms(0) == 1000

        # Second retry: 2000ms
        assert config.get_delay_ms(1) == 2000

        # Third retry: 4000ms
        assert config.get_delay_ms(2) == 4000

    def test_retry_config_max_delay(self):
        """Test that delay is capped at max_delay_ms."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=5000,
            exponential_base=2.0,
            jitter=False,
        )

        # At attempt 10, would be 1024000ms but capped at 5000
        assert config.get_delay_ms(10) == 5000

    def test_retry_config_jitter(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter=True,
        )

        delays = [config.get_delay_ms(0) for _ in range(10)]

        # With jitter, not all delays should be the same
        assert len(set(delays)) > 1

    def test_retry_with_backoff_success_first_try(self):
        """Test that successful function doesn't retry."""
        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(success_func, RetryConfig(max_retries=3))

        assert result == "success"
        assert call_count == 1

    def test_retry_with_backoff_eventual_success(self):
        """Test retry until success."""
        call_count = 0

        def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        config = RetryConfig(max_retries=3, base_delay_ms=10)
        result = retry_with_backoff(eventual_success, config)

        assert result == "success"
        assert call_count == 3

    def test_retry_with_backoff_exhausted(self):
        """Test that exception is raised when retries exhausted."""

        def always_fail():
            raise ValueError("Persistent error")

        config = RetryConfig(max_retries=2, base_delay_ms=10)

        with pytest.raises(ValueError, match="Persistent error"):
            retry_with_backoff(always_fail, config)

    def test_retry_on_retry_callback(self):
        """Test that on_retry callback is called."""
        callback_calls = []

        def failing_func():
            if len(callback_calls) < 2:
                raise ValueError("Error")
            return "success"

        def on_retry(attempt, exception):
            callback_calls.append((attempt, str(exception)))

        config = RetryConfig(max_retries=3, base_delay_ms=10)
        retry_with_backoff(failing_func, config, on_retry=on_retry)

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 0  # First retry attempt
        assert callback_calls[1][0] == 1  # Second retry attempt


class TestIntegrationScenarios:
    """Integration test scenarios for the complete pipeline."""

    def test_anti_hallucination_prompt_chain(self):
        """Test that anti-hallucination prompts work together."""
        # Build complete system prompt for extraction
        system_prompt = build_enhanced_system_prompt("CMS-1500", is_verification_pass=False)

        # Build extraction prompt
        extraction_prompt = build_extraction_prompt(
            schema_fields=[{"name": "total_charges", "field_type": "currency"}],
            document_type="CMS-1500",
            page_number=1,
            total_pages=1,
            is_first_pass=True,
            include_reasoning=True,
            include_anti_patterns=True,
        )

        # Verify anti-hallucination coverage in combined prompts
        combined = system_prompt + extraction_prompt

        # Core anti-hallucination rules
        assert "null" in combined.lower()
        assert "confidence" in combined.lower()
        assert "visible" in combined.lower()
        assert "guess" in combined.lower()

        # Anti-patterns
        assert "calculate" in combined.lower()
        assert "infer" in combined.lower()

    def test_dual_pass_differentiation(self):
        """Test that dual-pass prompts are sufficiently different."""
        schema_fields = [{"name": "patient_name", "field_type": "string"}]

        pass1 = build_extraction_prompt(
            schema_fields=schema_fields,
            document_type="CMS-1500",
            page_number=1,
            total_pages=1,
            is_first_pass=True,
        )

        pass2 = build_verification_prompt(
            schema_fields=schema_fields,
            document_type="CMS-1500",
            page_number=1,
            first_pass_results={},
        )

        # Passes should have different focus
        assert "COMPLETENESS" in pass1.upper() or "Completeness" in pass1
        assert "SKEPTICAL" in pass2 or "VERIFICATION" in pass2

        # Pass 2 should be more strict
        verification_indicators = ["audit", "skeptic", "character-by-character", "doubt"]
        has_verification_language = any(
            indicator in pass2.lower() for indicator in verification_indicators
        )
        assert has_verification_language
