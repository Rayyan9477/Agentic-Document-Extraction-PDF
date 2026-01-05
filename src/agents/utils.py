"""
Shared utilities for extraction agents.

Provides common functionality including:
- Custom schema building (extracted from duplicated code)
- Retry logic with exponential backoff
- Targeted re-extraction for low-confidence fields
"""

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.config import get_logger
from src.schemas import DocumentSchema, DocumentType, FieldType, RuleOperator
from src.schemas.schema_builder import FieldBuilder, RuleBuilder, SchemaBuilder


logger = get_logger(__name__)

T = TypeVar("T")


def build_custom_schema(schema_def: dict[str, Any]) -> DocumentSchema:
    """
    Build a DocumentSchema from a custom schema definition.

    This is a shared utility extracted from ExtractorAgent and ValidatorAgent
    to eliminate code duplication.

    Args:
        schema_def: Custom schema definition dictionary with fields and rules.

    Returns:
        Constructed DocumentSchema.
    """
    builder = SchemaBuilder(
        name=schema_def.get("name", "custom_schema"),
        document_type=DocumentType.CUSTOM,
    )

    builder.description(schema_def.get("description", "Custom extraction schema"))

    if schema_def.get("display_name"):
        builder.display_name(schema_def["display_name"])

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

        if field_def.get("nested_schema"):
            field_builder.nested(field_def["nested_schema"])

        if field_def.get("list_item_type"):
            list_type_str = field_def["list_item_type"].upper()
            try:
                list_type = FieldType[list_type_str]
                field_builder.list_of(list_type)
            except KeyError:
                pass

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

        if rule_def.get("value"):
            # For REQUIRES_IF operator
            rule_builder._rule.value = rule_def["value"]

        builder.rule(rule_builder)

    return builder.build()


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 30000,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay_ms: Initial delay between retries in milliseconds.
            max_delay_ms: Maximum delay between retries in milliseconds.
            exponential_base: Base for exponential backoff calculation.
            jitter: Whether to add random jitter to delays.
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay_ms(self, attempt: int) -> int:
        """
        Calculate delay for a given retry attempt.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in milliseconds.
        """
        delay = self.base_delay_ms * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay_ms)

        if self.jitter:
            # Add random jitter of ±25%
            jitter_factor = 0.75 + random.random() * 0.5
            delay = delay * jitter_factor

        return int(delay)


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    recoverable_exceptions: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Function to execute.
        config: Retry configuration. Defaults to 3 retries with 1s base delay.
        recoverable_exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback called on each retry with (attempt, exception).

    Returns:
        Result of the function.

    Raises:
        Last exception if all retries fail.
    """
    if config is None:
        config = RetryConfig()

    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except recoverable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay_ms = config.get_delay_ms(attempt)

                logger.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    max_retries=config.max_retries,
                    delay_ms=delay_ms,
                    error=str(e),
                )

                if on_retry:
                    on_retry(attempt, e)

                time.sleep(delay_ms / 1000.0)
            else:
                logger.error(
                    "retry_exhausted",
                    attempts=config.max_retries + 1,
                    error=str(e),
                )

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but no success")


def identify_low_confidence_fields(
    field_metadata: dict[str, Any],
    threshold: float = 0.7,
) -> list[str]:
    """
    Identify fields with confidence below threshold for targeted re-extraction.

    Args:
        field_metadata: Dictionary of field metadata with confidence scores.
        threshold: Confidence threshold below which fields are flagged.

    Returns:
        List of field names with low confidence.
    """
    low_confidence_fields = []

    for field_name, metadata in field_metadata.items():
        if isinstance(metadata, dict):
            confidence = metadata.get("confidence", 0.0)
            value = metadata.get("value")

            # Only flag fields that have a value but low confidence
            if value is not None and confidence < threshold:
                low_confidence_fields.append(field_name)

    return low_confidence_fields


def identify_disagreement_fields(
    field_metadata: dict[str, Any],
) -> list[str]:
    """
    Identify fields where dual-pass extraction disagreed.

    Args:
        field_metadata: Dictionary of field metadata with pass agreement info.

    Returns:
        List of field names where passes disagreed.
    """
    disagreement_fields = []

    for field_name, metadata in field_metadata.items():
        if isinstance(metadata, dict):
            passes_agree = metadata.get("passes_agree", True)

            if not passes_agree:
                disagreement_fields.append(field_name)

    return disagreement_fields


def calculate_extraction_quality_score(
    field_metadata: dict[str, Any],
    hallucination_flags: list[str],
    validation_errors: list[str],
) -> float:
    """
    Calculate an overall quality score for an extraction.

    Args:
        field_metadata: Dictionary of field metadata.
        hallucination_flags: List of fields flagged for potential hallucination.
        validation_errors: List of validation error messages.

    Returns:
        Quality score from 0.0 to 1.0.
    """
    if not field_metadata:
        return 0.0

    # Calculate average confidence
    total_confidence = 0.0
    field_count = 0

    for metadata in field_metadata.values():
        if isinstance(metadata, dict):
            confidence = metadata.get("confidence", 0.0)
            value = metadata.get("value")

            if value is not None:
                total_confidence += confidence
                field_count += 1

    avg_confidence = total_confidence / field_count if field_count > 0 else 0.0

    # Apply penalties
    hallucination_penalty = len(hallucination_flags) * 0.1
    error_penalty = len(validation_errors) * 0.05

    # Calculate final score
    quality_score = avg_confidence - hallucination_penalty - error_penalty
    return max(0.0, min(1.0, quality_score))
