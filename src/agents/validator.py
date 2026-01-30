"""
Validator Agent for quality assurance and hallucination detection.

Responsible for:
- Schema validation against document type rules
- Hallucination pattern detection (via validation module)
- Medical code validation (CPT, ICD-10, NPI)
- Cross-field rule validation
- Final confidence score calculation

Integrates with Phase 3 validation module for comprehensive
anti-hallucination detection and validation.
"""

import re
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.agents.base import ValidationError as AgentValidationError
from src.agents.utils import (
    build_custom_schema,
)
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import (
    ConfidenceLevel,
    ExtractionState,
    ExtractionStatus,
    ValidationResult,
    complete_extraction,
    request_human_review,
    request_retry,
    serialize_validation_result,
    set_status,
    update_state,
)
from src.schemas import (
    CrossFieldRule,
    DocumentSchema,
    RuleOperator,
    SchemaRegistry,
    validate_cpt_code,
    validate_field,
    validate_icd10_code,
    validate_npi,
)
from src.validation import (
    ConfidenceAction,
    ConfidenceScorer,
    HallucinationPatternDetector,
    HumanReviewQueue,
    MedicalCodeValidationEngine,
    validate_cross_fields,
)


logger = get_logger(__name__)


# Hallucination detection patterns
PLACEHOLDER_PATTERNS = [
    re.compile(r"^N/?A$", re.IGNORECASE),
    re.compile(r"^TBD$", re.IGNORECASE),
    re.compile(r"^XXX+$", re.IGNORECASE),
    re.compile(r"^12345\d*$"),  # Sequential test numbers: 12345, 123456, etc.
    re.compile(r"^00000+$"),
    re.compile(r"^\*+$"),
    re.compile(r"^TEST$", re.IGNORECASE),
    re.compile(r"^SAMPLE$", re.IGNORECASE),
    re.compile(r"^EXAMPLE$", re.IGNORECASE),
    re.compile(r"^JOHN\s*DOE$", re.IGNORECASE),
    re.compile(r"^JANE\s*DOE$", re.IGNORECASE),
]

# Suspiciously generic round amounts — only flag amounts that are commonly
# hallucinated by VLMs (large round numbers), NOT legitimate small charges
ROUND_AMOUNT_PATTERN = re.compile(r"^\$?(?:100|500|1000|2000|2500|5000|10000)\.00$")

# Common hallucinated dates
SUSPICIOUS_DATES = [
    "01/01/2000",
    "01/01/1900",
    "12/31/9999",
    "00/00/0000",
]


class ValidatorAgent(BaseAgent):
    """
    Validation agent for quality assurance and hallucination detection.

    Implements Layer 3 of the 3-layer anti-hallucination system:
    - Pattern-based hallucination detection (via HallucinationPatternDetector)
    - Medical code validation (via MedicalCodeValidationEngine)
    - Cross-field rule validation (via CrossFieldValidator)
    - Schema compliance checking
    - Final confidence scoring (via ConfidenceScorer)

    Integrates with the Phase 3 validation module for comprehensive
    anti-hallucination detection with confidence-based routing.

    VLM Calls: 0-1 per document (optional verification)
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.50,
        enable_vlm_verification: bool = False,
        review_queue_path: str | None = None,
    ) -> None:
        """
        Initialize the Validator agent.

        Args:
            client: Optional pre-configured LM Studio client.
            high_confidence_threshold: Threshold for high confidence (auto-accept).
            low_confidence_threshold: Threshold for low confidence (human review).
            enable_vlm_verification: Whether to use VLM for additional verification.
            review_queue_path: Optional path for persisting human review queue.
        """
        super().__init__(name="validator", client=client)
        self._schema_registry = SchemaRegistry()
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold
        self._enable_vlm_verification = enable_vlm_verification

        # Initialize Phase 3 validation components
        self._pattern_detector = HallucinationPatternDetector()
        self._medical_validator = MedicalCodeValidationEngine()
        self._confidence_scorer = ConfidenceScorer()
        self._review_queue = HumanReviewQueue(
            queue_path=review_queue_path,
            auto_persist=review_queue_path is not None,
        )

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Validate extraction results and determine final disposition.

        This is the main entry point for the LangGraph workflow.

        Args:
            state: Current extraction state.

        Returns:
            Updated state with validation results and routing decision.
        """
        # Reset metrics to prevent accumulation across documents
        self.reset_metrics()

        start_time = self.log_operation_start(
            "validation",
            processing_id=state.get("processing_id", ""),
        )

        try:
            # Update status
            state = set_status(state, ExtractionStatus.VALIDATING, "validating")

            # Get extraction results
            merged_extraction = state.get("merged_extraction", {})
            field_metadata = state.get("field_metadata", {})

            if not merged_extraction:
                raise AgentValidationError(
                    "No extraction results to validate",
                    agent_name=self.name,
                    recoverable=False,
                )

            # Get schema for validation
            schema = self._get_schema(state)

            # Perform validation
            validation_result = self._validate_extraction(
                extraction=merged_extraction,
                field_metadata=field_metadata,
                schema=schema,
                document_type=state.get("document_type", "OTHER"),
                retry_count=state.get("retry_count", 0),
            )

            # Calculate processing time
            duration_ms = self.log_operation_complete(
                "validation",
                start_time,
                success=True,
                is_valid=validation_result.is_valid,
                overall_confidence=validation_result.overall_confidence,
            )

            validation_result.validation_time_ms = duration_ms

            # Update state with validation results
            state = update_state(
                state,
                {
                    "validation": serialize_validation_result(validation_result),
                    "overall_confidence": validation_result.overall_confidence,
                    "confidence_level": validation_result.confidence_level.value,
                },
            )

            # Determine routing based on confidence
            state = self._route_based_on_confidence(state, validation_result)

            return state

        except AgentValidationError:
            raise
        except Exception as e:
            self.log_operation_complete("validation", start_time, success=False)
            raise AgentValidationError(
                f"Validation failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

    def _get_schema(self, state: ExtractionState) -> DocumentSchema | None:
        """
        Get schema for validation.

        Args:
            state: Current extraction state.

        Returns:
            DocumentSchema or None if not found.
        """
        # Check for custom schema first
        custom_schema = state.get("custom_schema")
        if custom_schema:
            return self._build_custom_schema(custom_schema)

        # Get schema by name from registry
        schema_name = state.get("selected_schema_name", "")
        if schema_name:
            try:
                return self._schema_registry.get(schema_name)
            except ValueError:
                self._logger.warning("schema_not_found", schema_name=schema_name)

        return None

    def _build_custom_schema(self, schema_def: dict[str, Any]) -> DocumentSchema:
        """
        Build a DocumentSchema from a custom schema definition.

        Uses shared utility to eliminate code duplication.

        Args:
            schema_def: Custom schema definition dictionary.

        Returns:
            Constructed DocumentSchema.
        """
        return build_custom_schema(schema_def)

    def _validate_extraction(
        self,
        extraction: dict[str, Any],
        field_metadata: dict[str, Any],
        schema: DocumentSchema | None,
        document_type: str,
        retry_count: int = 0,
    ) -> ValidationResult:
        """
        Perform comprehensive validation of extraction results.

        Uses Phase 3 validation module for:
        - Hallucination pattern detection
        - Medical code validation
        - Cross-field validation
        - Confidence scoring

        Args:
            extraction: Merged extraction results.
            field_metadata: Field metadata from extraction.
            schema: Document schema (if available).
            document_type: Type of document.
            retry_count: Current retry attempt count for confidence scoring.

        Returns:
            ValidationResult with all validation details.
        """
        result = ValidationResult()

        # Extract values and confidences from extraction data
        values: dict[str, Any] = {}
        extraction_confidences: dict[str, float] = {}

        for field_name, field_data in extraction.items():
            if isinstance(field_data, dict):
                values[field_name] = field_data.get("value")
                extraction_confidences[field_name] = field_data.get("confidence", 0.5)
            else:
                values[field_name] = field_data
                extraction_confidences[field_name] = 0.5

        # Step 1: Hallucination pattern detection using validation module
        pattern_result = self._pattern_detector.detect(values, extraction_confidences)

        if pattern_result.is_likely_hallucination:
            result.hallucination_flags.extend(list(pattern_result.flagged_fields))
            for match in pattern_result.matches:
                result.warnings.append(f"{match.field_name}: {match.description}")

        # Step 2: Medical code validation using validation module
        code_result = self._medical_validator.validate_all(values)

        for code_detail in code_result.validations:
            if not code_detail.is_valid:
                result.errors.append(f"{code_detail.code}: {code_detail.message}")
            result.field_validations[code_detail.code] = code_detail.is_valid

        # Step 3: Schema-based field validation (existing logic)
        for field_name, value in values.items():
            if value is None:
                continue

            field_errors: list[str] = []

            # Legacy hallucination check (for backward compatibility)
            hallucination_flag = self._check_hallucination_patterns(
                field_name, value, extraction_confidences.get(field_name, 0.5)
            )
            if hallucination_flag and field_name not in result.hallucination_flags:
                result.hallucination_flags.append(field_name)
                result.warnings.append(f"{field_name}: {hallucination_flag}")

            # Schema validation
            if schema:
                field_def = self._get_field_definition(schema, field_name)
                if field_def:
                    is_valid, error = validate_field(value, field_def)
                    if not is_valid and error:
                        field_errors.append(error)

            result.field_validations[field_name] = len(field_errors) == 0

            if field_errors:
                result.errors.extend([f"{field_name}: {e}" for e in field_errors])

        # Step 4: Cross-field validation using validation module
        cross_result = validate_cross_fields(values, document_type.lower())

        if not cross_result.passed:
            for violation in cross_result.violations:
                result.cross_field_validations.append(violation.to_dict())
                result.errors.append(violation.message)

        # Legacy cross-field validation from schema
        if schema and schema.cross_field_rules:
            legacy_cross = self._validate_cross_field_rules(extraction, schema.cross_field_rules)
            for cf_result in legacy_cross:
                if not cf_result.get("passed", True):
                    result.cross_field_validations.append(cf_result)
                    result.errors.append(cf_result.get("message", "Cross-field validation failed"))

        # Step 5: Check for repetitive values
        repetition_warnings = self._check_repetitive_values(extraction)
        result.warnings.extend(repetition_warnings)

        # Step 6: Calculate confidence using validation module
        validation_results = {k: v for k, v in result.field_validations.items()}

        conf_result = self._confidence_scorer.calculate(
            extraction_confidences=extraction_confidences,
            validation_results=validation_results,
            pattern_flags=set(result.hallucination_flags),
            retry_count=retry_count,
        )

        result.overall_confidence = conf_result.overall_confidence

        # Map confidence level from validation module to pipeline state
        if conf_result.overall_level.value == "high":
            result.confidence_level = ConfidenceLevel.HIGH
        elif conf_result.overall_level.value == "medium":
            result.confidence_level = ConfidenceLevel.MEDIUM
        else:
            result.confidence_level = ConfidenceLevel.LOW

        # Determine validity
        result.is_valid = len(result.errors) == 0 and len(result.hallucination_flags) == 0

        # Determine routing based on confidence module recommendation
        if conf_result.recommended_action == ConfidenceAction.HUMAN_REVIEW:
            result.requires_human_review = True
        elif conf_result.recommended_action == ConfidenceAction.RETRY:
            result.requires_retry = True

        return result

    def _check_hallucination_patterns(
        self,
        field_name: str,
        value: Any,
        confidence: float,
    ) -> str | None:
        """
        Check for common hallucination patterns.

        Args:
            field_name: Name of the field.
            value: Field value.
            confidence: Reported confidence.

        Returns:
            Description of hallucination pattern if found, None otherwise.
        """
        if value is None:
            return None

        str_value = str(value).strip()

        # Check placeholder patterns
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.match(str_value):
                return f"Placeholder pattern detected: {str_value}"

        # Check suspicious round amounts for currency fields
        if "amount" in field_name.lower() or "charge" in field_name.lower():
            if ROUND_AMOUNT_PATTERN.match(str_value):
                # Round amounts with high confidence are suspicious
                if confidence > 0.9:
                    return f"Suspiciously round amount with high confidence: {str_value}"

        # Check suspicious dates
        if "date" in field_name.lower():
            if str_value in SUSPICIOUS_DATES:
                return f"Suspicious date detected: {str_value}"

        # High confidence with disagreement between passes is suspicious
        # (This is checked elsewhere via field_metadata)

        return None

    def _validate_medical_codes(
        self,
        field_name: str,
        value: Any,
    ) -> list[str]:
        """
        Validate medical codes (CPT, ICD-10, NPI).

        Args:
            field_name: Name of the field.
            value: Field value.

        Returns:
            List of validation errors.
        """
        if value is None:
            return []

        errors: list[str] = []
        str_value = str(value).strip()

        # CPT code validation
        if "cpt" in field_name.lower():
            if not validate_cpt_code(str_value):
                errors.append(f"Invalid CPT code format: {str_value}")

        # ICD-10 code validation
        if "icd" in field_name.lower() or "diagnosis" in field_name.lower():
            if not validate_icd10_code(str_value):
                errors.append(f"Invalid ICD-10 code format: {str_value}")

        # NPI validation
        if "npi" in field_name.lower():
            if not validate_npi(str_value):
                errors.append(f"Invalid NPI (Luhn check failed): {str_value}")

        return errors

    def _validate_cross_field_rules(
        self,
        extraction: dict[str, Any],
        rules: list[CrossFieldRule],
    ) -> list[dict[str, Any]]:
        """
        Validate cross-field rules.

        Args:
            extraction: Extraction results.
            rules: List of cross-field rules.

        Returns:
            List of validation results for each rule.
        """
        results: list[dict[str, Any]] = []

        for rule in rules:
            # Handle SUM_EQUALS specially - source_field is comma-separated list of fields to sum
            if rule.operator == RuleOperator.SUM_EQUALS:
                source_fields = [f.strip() for f in rule.source_field.split(",")]
                field_sum = 0.0
                missing_fields: list[str] = []

                for field_name in source_fields:
                    field_data = extraction.get(field_name, {})
                    value = field_data.get("value") if isinstance(field_data, dict) else field_data
                    if value is None:
                        missing_fields.append(field_name)
                    else:
                        try:
                            # Handle currency formatting (remove $, commas)
                            clean_val = str(value).replace("$", "").replace(",", "").strip()
                            field_sum += float(clean_val)
                        except (ValueError, TypeError):
                            missing_fields.append(field_name)

                target_data = extraction.get(rule.target_field, {})
                target_value = (
                    target_data.get("value") if isinstance(target_data, dict) else target_data
                )
                source_value = field_sum

                if missing_fields:
                    passed = False
                    status = "inconclusive"
                elif target_value is None:
                    passed = False
                    status = "skipped"
                else:
                    passed, status = self._evaluate_rule(rule, source_value, target_value)
            else:
                source_data = extraction.get(rule.source_field, {})
                target_data = extraction.get(rule.target_field, {})

                source_value = (
                    source_data.get("value") if isinstance(source_data, dict) else source_data
                )
                target_value = (
                    target_data.get("value") if isinstance(target_data, dict) else target_data
                )

                passed, status = self._evaluate_rule(rule, source_value, target_value)

            # Determine message based on status
            if status == "skipped":
                message = f"Skipped: missing value(s) for {rule.source_field}/{rule.target_field}"
            elif status == "inconclusive":
                message = "Validation inconclusive due to missing data"
            elif not passed:
                message = rule.get_error_message()
            else:
                message = "OK"

            results.append(
                {
                    "rule": f"{rule.source_field} {rule.operator.value} {rule.target_field}",
                    "passed": passed,
                    "status": status,  # "passed", "failed", "skipped", "inconclusive"
                    "message": message,
                    "source_value": source_value,
                    "target_value": target_value,
                }
            )

        return results

    def _evaluate_rule(
        self,
        rule: CrossFieldRule,
        source_value: Any,
        target_value: Any,
    ) -> tuple[bool, str]:
        """
        Evaluate a single cross-field rule.

        Args:
            rule: The rule to evaluate.
            source_value: Value of source field.
            target_value: Value of target field.

        Returns:
            Tuple of (passed: bool, status: str).
            Status can be: "passed", "failed", "skipped", "inconclusive"
        """
        # Handle REQUIRES operators specially - they check for presence
        if rule.operator == RuleOperator.REQUIRES:
            # If source has value, target must also have value
            if source_value is None:
                # Source is missing, so requirement doesn't apply
                return (True, "skipped")
            if target_value is None:
                # Source present but target missing - FAIL
                return (False, "failed")
            return (True, "passed")

        if rule.operator == RuleOperator.REQUIRES_IF:
            # If source matches rule.value, target must have value
            if source_value is None:
                return (True, "skipped")
            if source_value == rule.value:
                if target_value is None:
                    return (False, "failed")
                return (True, "passed")
            return (True, "passed")

        # For other operators, handle missing values
        if source_value is None and target_value is None:
            # Both missing - skip validation but flag as inconclusive
            return (True, "skipped")

        if source_value is None or target_value is None:
            # One value missing - mark as inconclusive (not a pass!)
            # This prevents masking extraction failures
            return (False, "inconclusive")

        try:
            if rule.operator == RuleOperator.EQUALS:
                result = source_value == target_value
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.NOT_EQUALS:
                result = source_value != target_value
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.GREATER_THAN:
                result = float(source_value) > float(target_value)
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.LESS_THAN:
                result = float(source_value) < float(target_value)
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.GREATER_EQUAL:
                result = float(source_value) >= float(target_value)
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.LESS_EQUAL:
                result = float(source_value) <= float(target_value)
                return (result, "passed" if result else "failed")

            if rule.operator == RuleOperator.DATE_BEFORE:
                from src.utils.date_utils import parse_date

                source_date = parse_date(str(source_value))
                target_date = parse_date(str(target_value))
                if source_date and target_date:
                    result = source_date < target_date
                    return (result, "passed" if result else "failed")
                return (False, "inconclusive")  # Can't validate if parsing fails

            if rule.operator == RuleOperator.DATE_AFTER:
                from src.utils.date_utils import parse_date

                source_date = parse_date(str(source_value))
                target_date = parse_date(str(target_value))
                if source_date and target_date:
                    result = source_date > target_date
                    return (result, "passed" if result else "failed")
                return (False, "inconclusive")

            if rule.operator == RuleOperator.SUM_EQUALS:
                # Source value is pre-calculated sum from _validate_cross_field_rules
                # Target value is the expected total
                try:
                    # Handle currency formatting in target value
                    clean_target = str(target_value).replace("$", "").replace(",", "").strip()
                    expected_sum = float(clean_target)
                    actual_sum = float(source_value)

                    # Allow small tolerance for floating point comparison (0.01 for currency)
                    tolerance = 0.01
                    result = abs(actual_sum - expected_sum) <= tolerance
                    return (result, "passed" if result else "failed")
                except (ValueError, TypeError):
                    return (False, "inconclusive")

        except (ValueError, TypeError) as e:
            # Conversion failed - mark as inconclusive
            self._logger.warning(
                "rule_evaluation_error",
                rule=str(rule.operator.value),
                error=str(e),
            )
            return (False, "inconclusive")

        return (True, "skipped")

    def _check_repetitive_values(
        self,
        extraction: dict[str, Any],
    ) -> list[str]:
        """
        Check for repetitive values across fields (hallucination indicator).

        Args:
            extraction: Extraction results.

        Returns:
            List of warnings about repetitive values.
        """
        warnings: list[str] = []
        value_counts: dict[str, list[str]] = {}

        for field_name, field_data in extraction.items():
            value = field_data.get("value") if isinstance(field_data, dict) else field_data

            if value is None:
                continue

            str_value = str(value).strip().lower()

            # Skip short values
            if len(str_value) < 3:
                continue

            if str_value not in value_counts:
                value_counts[str_value] = []
            value_counts[str_value].append(field_name)

        # Flag values that appear in 3+ fields
        for value, fields in value_counts.items():
            if len(fields) >= 3:
                warnings.append(
                    f"Repetitive value '{value}' found in {len(fields)} fields: "
                    f"{', '.join(fields[:5])}"
                )

        return warnings

    def _get_field_definition(
        self,
        schema: DocumentSchema,
        field_name: str,
    ) -> Any | None:
        """Get field definition from schema."""
        for field in schema.fields:
            if field.name == field_name:
                return field
        return None

    def _route_based_on_confidence(
        self,
        state: ExtractionState,
        validation: ValidationResult,
    ) -> ExtractionState:
        """
        Annotate state with validation recommendations for the orchestrator.

        IMPORTANT: This method sets recommendation flags and confidence data
        but does NOT set the final status (completed/retry/human_review).
        The orchestrator's _determine_route() is the single source of truth
        for routing decisions, using the confidence_level and flags set here.

        Args:
            state: Current state.
            validation: Validation results.

        Returns:
            Updated state with validation recommendations (not final status).
        """
        # Build recommendation reasons for logging/debugging
        recommendation_reasons: list[str] = []

        if validation.confidence_level == ConfidenceLevel.LOW:
            recommendation_reasons.append(
                f"Low confidence: {validation.overall_confidence:.2f}"
            )
        if validation.hallucination_flags:
            recommendation_reasons.append(
                f"Hallucination flags: {', '.join(validation.hallucination_flags[:3])}"
            )
        if validation.errors:
            recommendation_reasons.append(
                f"Validation errors: {len(validation.errors)}"
            )

        # Annotate state with validation results — let orchestrator decide routing
        return update_state(
            state,
            {
                "validation_is_valid": validation.is_valid,
                "validation_requires_retry": validation.requires_retry,
                "validation_requires_human_review": validation.requires_human_review,
                "validation_reasons": "; ".join(recommendation_reasons) if recommendation_reasons else "",
                "current_step": "validation_complete",
            },
        )

    def validate_field_standalone(
        self,
        field_name: str,
        value: Any,
        field_type: str = "string",
    ) -> AgentResult[dict[str, Any]]:
        """
        Validate a single field value standalone.

        Args:
            field_name: Name of field.
            value: Value to validate.
            field_type: Type of field.

        Returns:
            AgentResult with validation details.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check hallucination patterns
        hallucination = self._check_hallucination_patterns(field_name, value, 0.5)
        if hallucination:
            warnings.append(hallucination)

        # Check medical codes
        code_errors = self._validate_medical_codes(field_name, value)
        errors.extend(code_errors)

        return AgentResult.ok(
            data={
                "field_name": field_name,
                "value": value,
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
            },
            agent_name=self.name,
            operation="validate_field",
        )
