"""
Validator Agent for quality assurance and hallucination detection.

Responsible for:
- Schema validation against document type rules
- Hallucination pattern detection
- Medical code validation (CPT, ICD-10, NPI)
- Cross-field rule validation
- Final confidence score calculation
"""

import re
from typing import Any
import time

from src.agents.base import BaseAgent, AgentResult, ValidationError as AgentValidationError
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    ValidationResult,
    ConfidenceLevel,
    update_state,
    set_status,
    add_warning,
    complete_extraction,
    request_human_review,
    request_retry,
    serialize_validation_result,
    deserialize_field_metadata,
)
from src.schemas import (
    SchemaRegistry,
    DocumentSchema,
    validate_field,
    validate_cpt_code,
    validate_icd10_code,
    validate_npi,
    CrossFieldRule,
    RuleOperator,
)


logger = get_logger(__name__)


# Hallucination detection patterns
PLACEHOLDER_PATTERNS = [
    re.compile(r"^N/?A$", re.IGNORECASE),
    re.compile(r"^TBD$", re.IGNORECASE),
    re.compile(r"^XXX+$", re.IGNORECASE),
    re.compile(r"^12345*$"),
    re.compile(r"^00000+$"),
    re.compile(r"^\*+$"),
    re.compile(r"^TEST$", re.IGNORECASE),
    re.compile(r"^SAMPLE$", re.IGNORECASE),
    re.compile(r"^EXAMPLE$", re.IGNORECASE),
    re.compile(r"^JOHN\s*DOE$", re.IGNORECASE),
    re.compile(r"^JANE\s*DOE$", re.IGNORECASE),
]

# Suspiciously round amounts (potential hallucinations)
ROUND_AMOUNT_PATTERN = re.compile(r"^\$?\d+\.00$")

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
    - Pattern-based hallucination detection
    - Medical code validation
    - Cross-field rule validation
    - Schema compliance checking
    - Final confidence scoring

    VLM Calls: 0-1 per document (optional verification)
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.50,
        enable_vlm_verification: bool = False,
    ) -> None:
        """
        Initialize the Validator agent.

        Args:
            client: Optional pre-configured LM Studio client.
            high_confidence_threshold: Threshold for high confidence (auto-accept).
            low_confidence_threshold: Threshold for low confidence (human review).
            enable_vlm_verification: Whether to use VLM for additional verification.
        """
        super().__init__(name="validator", client=client)
        self._schema_registry = SchemaRegistry()
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold
        self._enable_vlm_verification = enable_vlm_verification

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Validate extraction results and determine final disposition.

        This is the main entry point for the LangGraph workflow.

        Args:
            state: Current extraction state.

        Returns:
            Updated state with validation results and routing decision.
        """
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

        Args:
            schema_def: Custom schema definition dictionary.

        Returns:
            Constructed DocumentSchema.
        """
        from src.schemas.schema_builder import SchemaBuilder, FieldBuilder, RuleBuilder
        from src.schemas import DocumentType, FieldType, RuleOperator

        builder = SchemaBuilder(
            name=schema_def.get("name", "custom_schema"),
            document_type=DocumentType.OTHER,
        )

        builder.description(schema_def.get("description", "Custom extraction schema"))

        # Build fields
        for field_def in schema_def.get("fields", []):
            field_type_str = field_def.get("type", "string").upper()
            try:
                field_type = FieldType[field_type_str]
            except KeyError:
                field_type = FieldType.STRING

            field_builder = (
                FieldBuilder(field_def.get("name", "field"))
                .display_name(field_def.get("display_name", field_def.get("name", "")))
                .type(field_type)
                .description(field_def.get("description", ""))
                .required(field_def.get("required", False))
            )

            if field_def.get("examples"):
                field_builder.examples(field_def["examples"])

            if field_def.get("pattern"):
                field_builder.pattern(field_def["pattern"])

            if field_def.get("location_hint"):
                field_builder.location_hint(field_def["location_hint"])

            if field_def.get("min_value") is not None:
                field_builder.min_value(field_def["min_value"])

            if field_def.get("max_value") is not None:
                field_builder.max_value(field_def["max_value"])

            if field_def.get("allowed_values"):
                field_builder.allowed_values(field_def["allowed_values"])

            builder.field(field_builder)

        # Build cross-field rules
        for rule_def in schema_def.get("rules", []):
            source = rule_def.get("source_field", "")
            target = rule_def.get("target_field", "")
            operator_str = rule_def.get("operator", "equals").upper()

            try:
                operator = RuleOperator[operator_str]
            except KeyError:
                operator = RuleOperator.EQUALS

            rule_builder = RuleBuilder(source, target)

            # Set operator using fluent API
            operator_method_map = {
                RuleOperator.EQUALS: rule_builder.equals,
                RuleOperator.NOT_EQUALS: rule_builder.not_equals,
                RuleOperator.GREATER_THAN: rule_builder.greater_than,
                RuleOperator.LESS_THAN: rule_builder.less_than,
                RuleOperator.GREATER_EQUAL: rule_builder.greater_or_equal,
                RuleOperator.LESS_EQUAL: rule_builder.less_or_equal,
                RuleOperator.DATE_BEFORE: rule_builder.date_before,
                RuleOperator.DATE_AFTER: rule_builder.date_after,
                RuleOperator.REQUIRES: rule_builder.requires,
                RuleOperator.REQUIRES_IF: rule_builder.requires_if,
            }

            if operator in operator_method_map:
                operator_method_map[operator]()

            if rule_def.get("error_message"):
                rule_builder.error(rule_def["error_message"])

            builder.rule(rule_builder)

        return builder.build()

    def _validate_extraction(
        self,
        extraction: dict[str, Any],
        field_metadata: dict[str, Any],
        schema: DocumentSchema | None,
        document_type: str,
    ) -> ValidationResult:
        """
        Perform comprehensive validation of extraction results.

        Args:
            extraction: Merged extraction results.
            field_metadata: Field metadata from extraction.
            schema: Document schema (if available).
            document_type: Type of document.

        Returns:
            ValidationResult with all validation details.
        """
        result = ValidationResult()
        total_confidence = 0.0
        field_count = 0

        # Validate each field
        for field_name, field_data in extraction.items():
            value = field_data.get("value") if isinstance(field_data, dict) else field_data
            confidence = (
                field_data.get("confidence", 0.5)
                if isinstance(field_data, dict)
                else 0.5
            )

            # Track confidence
            if value is not None:
                total_confidence += confidence
                field_count += 1

            # Field validation
            field_errors: list[str] = []
            field_warnings: list[str] = []

            # Check for hallucination patterns
            hallucination_flag = self._check_hallucination_patterns(
                field_name, value, confidence
            )
            if hallucination_flag:
                result.hallucination_flags.append(field_name)
                field_warnings.append(f"Potential hallucination detected: {hallucination_flag}")

            # Schema-based validation
            if schema:
                field_def = self._get_field_definition(schema, field_name)
                if field_def:
                    is_valid, error = validate_field(value, field_def)
                    if not is_valid and error:
                        field_errors.append(error)

            # Medical code validation
            code_errors = self._validate_medical_codes(field_name, value)
            field_errors.extend(code_errors)

            # Store field validation result
            result.field_validations[field_name] = len(field_errors) == 0

            if field_errors:
                result.errors.extend(
                    [f"{field_name}: {e}" for e in field_errors]
                )

            if field_warnings:
                result.warnings.extend(
                    [f"{field_name}: {w}" for w in field_warnings]
                )

        # Cross-field validation
        if schema and schema.cross_field_rules:
            cross_field_results = self._validate_cross_field_rules(
                extraction, schema.cross_field_rules
            )
            result.cross_field_validations = cross_field_results

            for cf_result in cross_field_results:
                if not cf_result.get("passed", True):
                    result.errors.append(cf_result.get("message", "Cross-field validation failed"))

        # Check for repetitive values (hallucination indicator)
        repetition_warnings = self._check_repetitive_values(extraction)
        result.warnings.extend(repetition_warnings)

        # Calculate overall confidence
        if field_count > 0:
            result.overall_confidence = total_confidence / field_count
        else:
            result.overall_confidence = 0.0

        # Determine confidence level
        if result.overall_confidence >= self._high_threshold:
            result.confidence_level = ConfidenceLevel.HIGH
        elif result.overall_confidence >= self._low_threshold:
            result.confidence_level = ConfidenceLevel.MEDIUM
        else:
            result.confidence_level = ConfidenceLevel.LOW

        # Determine overall validity
        result.is_valid = (
            len(result.errors) == 0 and
            len(result.hallucination_flags) == 0
        )

        # Determine if retry or review needed
        if result.confidence_level == ConfidenceLevel.LOW:
            result.requires_human_review = True
        elif result.confidence_level == ConfidenceLevel.MEDIUM and len(result.hallucination_flags) > 0:
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
            source_data = extraction.get(rule.source_field, {})
            target_data = extraction.get(rule.target_field, {})

            source_value = (
                source_data.get("value")
                if isinstance(source_data, dict)
                else source_data
            )
            target_value = (
                target_data.get("value")
                if isinstance(target_data, dict)
                else target_data
            )

            passed = self._evaluate_rule(rule, source_value, target_value)

            results.append({
                "rule": f"{rule.source_field} {rule.operator.value} {rule.target_field}",
                "passed": passed,
                "message": rule.get_error_message() if not passed else "OK",
                "source_value": source_value,
                "target_value": target_value,
            })

        return results

    def _evaluate_rule(
        self,
        rule: CrossFieldRule,
        source_value: Any,
        target_value: Any,
    ) -> bool:
        """
        Evaluate a single cross-field rule.

        Args:
            rule: The rule to evaluate.
            source_value: Value of source field.
            target_value: Value of target field.

        Returns:
            True if rule passes, False otherwise.
        """
        # Skip if either value is None
        if source_value is None or target_value is None:
            return True  # Can't validate if values missing

        try:
            if rule.operator == RuleOperator.EQUALS:
                return source_value == target_value

            elif rule.operator == RuleOperator.NOT_EQUALS:
                return source_value != target_value

            elif rule.operator == RuleOperator.GREATER_THAN:
                return float(source_value) > float(target_value)

            elif rule.operator == RuleOperator.LESS_THAN:
                return float(source_value) < float(target_value)

            elif rule.operator == RuleOperator.GREATER_EQUAL:
                return float(source_value) >= float(target_value)

            elif rule.operator == RuleOperator.LESS_EQUAL:
                return float(source_value) <= float(target_value)

            elif rule.operator == RuleOperator.DATE_BEFORE:
                from src.utils.date_utils import parse_date
                source_date = parse_date(str(source_value))
                target_date = parse_date(str(target_value))
                if source_date and target_date:
                    return source_date < target_date
                return True  # Can't validate if parsing fails

            elif rule.operator == RuleOperator.DATE_AFTER:
                from src.utils.date_utils import parse_date
                source_date = parse_date(str(source_value))
                target_date = parse_date(str(target_value))
                if source_date and target_date:
                    return source_date > target_date
                return True

            elif rule.operator == RuleOperator.REQUIRES:
                # If source has value, target must also have value
                return target_value is not None

            elif rule.operator == RuleOperator.REQUIRES_IF:
                # If source matches rule.value, target must have value
                if source_value == rule.value:
                    return target_value is not None
                return True

            elif rule.operator == RuleOperator.SUM_EQUALS:
                # Source is list of fields, target is expected sum
                # This would need special handling
                return True

        except (ValueError, TypeError):
            return True  # Can't validate if conversion fails

        return True

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
            value = (
                field_data.get("value")
                if isinstance(field_data, dict)
                else field_data
            )

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
        Route extraction based on confidence level.

        Args:
            state: Current state.
            validation: Validation results.

        Returns:
            Updated state with routing decision.
        """
        # High confidence: auto-accept
        if validation.confidence_level == ConfidenceLevel.HIGH and validation.is_valid:
            return complete_extraction(
                state,
                final_output=state.get("merged_extraction", {}),
                overall_confidence=validation.overall_confidence,
            )

        # Medium confidence with issues: retry
        if validation.requires_retry:
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 2)

            if retry_count < max_retries:
                return request_retry(state)
            else:
                return request_human_review(
                    state,
                    f"Maximum retries ({max_retries}) exceeded with confidence "
                    f"{validation.overall_confidence:.2f}",
                )

        # Low confidence or hallucinations: human review
        if validation.requires_human_review:
            reasons = []
            if validation.confidence_level == ConfidenceLevel.LOW:
                reasons.append(f"Low confidence: {validation.overall_confidence:.2f}")
            if validation.hallucination_flags:
                reasons.append(
                    f"Hallucination flags: {', '.join(validation.hallucination_flags[:3])}"
                )
            if validation.errors:
                reasons.append(f"Validation errors: {len(validation.errors)}")

            return request_human_review(state, "; ".join(reasons))

        # Default: complete with medium confidence
        return complete_extraction(
            state,
            final_output=state.get("merged_extraction", {}),
            overall_confidence=validation.overall_confidence,
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
        hallucination = self._check_hallucination_patterns(
            field_name, value, 0.5
        )
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
