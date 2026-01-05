"""
Cross-field validation rules for medical document extraction.

Implements validation logic that checks relationships between multiple fields:
- Date ordering (admission before discharge)
- Sum validation (line items equal total)
- Required dependencies (field B required if field A present)
- Mutual exclusivity (field A or B, not both)
- Format consistency (related fields use same format)

These rules catch logical inconsistencies that single-field validation cannot detect.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


class RuleType(str, Enum):
    """Types of cross-field validation rules."""

    DATE_ORDER = "date_order"
    SUM_VALIDATION = "sum_validation"
    NESTED_SUM_VALIDATION = "nested_sum_validation"
    REQUIRED_IF = "required_if"
    REQUIRED_UNLESS = "required_unless"
    MUTUAL_EXCLUSIVE = "mutual_exclusive"
    MUTUAL_REQUIRED = "mutual_required"
    FORMAT_MATCH = "format_match"
    VALUE_RANGE = "value_range"
    CUSTOM = "custom"


class RuleSeverity(str, Enum):
    """Severity of rule violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """
    A violation of a cross-field validation rule.

    Attributes:
        rule_name: Name identifying the rule.
        rule_type: Type of rule violated.
        severity: Severity of the violation.
        fields: Fields involved in the violation.
        message: Human-readable description.
        expected: Expected relationship or value.
        actual: Actual values found.
    """

    rule_name: str
    rule_type: RuleType
    severity: RuleSeverity
    fields: tuple[str, ...]
    message: str
    expected: str
    actual: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "severity": self.severity.value,
            "fields": list(self.fields),
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(slots=True)
class CrossFieldResult:
    """
    Complete result of cross-field validation.

    Attributes:
        violations: List of rule violations.
        errors: Violations with error severity.
        warnings: Violations with warning severity.
        passed: Whether validation passed (no errors).
        rules_checked: Number of rules checked.
        rules_passed: Number of rules that passed.
    """

    violations: list[RuleViolation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True
    rules_checked: int = 0
    rules_passed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "violations": [v.to_dict() for v in self.violations],
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
            "rules_checked": self.rules_checked,
            "rules_passed": self.rules_passed,
        }


@dataclass
class CrossFieldRule:
    """
    Definition of a cross-field validation rule.

    Attributes:
        name: Unique identifier for the rule.
        rule_type: Type of validation to perform.
        fields: Fields involved in the rule.
        severity: Severity if rule is violated.
        params: Additional parameters for the rule.
        message_template: Template for violation message.
        enabled: Whether rule is active.
    """

    name: str
    rule_type: RuleType
    fields: list[str]
    severity: RuleSeverity = RuleSeverity.ERROR
    params: dict[str, Any] = field(default_factory=dict)
    message_template: str = ""
    enabled: bool = True


class CrossFieldValidator:
    """
    Validates relationships between multiple fields.

    This validator checks logical consistency across fields that
    single-field validation cannot detect.

    Built-in rules for medical documents:
    - Date ordering (admission <= service <= discharge)
    - Charge summation (line items = total)
    - Code dependencies (modifier requires CPT)
    - Provider relationships (billing NPI for facility)

    Example:
        validator = CrossFieldValidator()
        validator.add_date_order_rule(
            "admission_to_discharge",
            "admission_date",
            "discharge_date",
        )
        result = validator.validate(extracted_data)

        if not result.passed:
            for error in result.errors:
                print(error)
    """

    # Date formats to try when parsing
    DATE_FORMATS = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%Y%m%d",
    ]

    def __init__(self, rules: list[CrossFieldRule] | None = None) -> None:
        """
        Initialize the validator with optional rules.

        Args:
            rules: Initial list of validation rules.
        """
        self.rules: list[CrossFieldRule] = rules or []
        self._custom_validators: dict[
            str, Callable[[dict[str, Any], CrossFieldRule], RuleViolation | None]
        ] = {}

    def add_rule(self, rule: CrossFieldRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def add_date_order_rule(
        self,
        name: str,
        earlier_field: str,
        later_field: str,
        allow_equal: bool = True,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring one date to be before another.

        Args:
            name: Rule identifier.
            earlier_field: Field that should contain earlier date.
            later_field: Field that should contain later date.
            allow_equal: Whether dates can be equal.
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.DATE_ORDER,
                fields=[earlier_field, later_field],
                severity=severity,
                params={"allow_equal": allow_equal},
                message_template="{earlier_field} must be before {later_field}",
            )
        )

    def add_sum_rule(
        self,
        name: str,
        component_fields: list[str],
        total_field: str,
        tolerance: float = 0.01,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring component fields to sum to total.

        Args:
            name: Rule identifier.
            component_fields: Fields that should sum together.
            total_field: Field containing expected total.
            tolerance: Allowed difference for floating point.
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.SUM_VALIDATION,
                fields=component_fields + [total_field],
                severity=severity,
                params={
                    "component_fields": component_fields,
                    "total_field": total_field,
                    "tolerance": tolerance,
                },
                message_template="Sum of components must equal {total_field}",
            )
        )

    def add_nested_sum_rule(
        self,
        name: str,
        array_field: str,
        item_field: str,
        total_field: str,
        tolerance: float = 0.01,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring sum of nested array field values to equal total.

        Use this for validating totals of line items, e.g., sum of service_lines[].total_charges
        should equal the document's total_charges.

        Args:
            name: Rule identifier.
            array_field: Name of the array field (e.g., "service_lines").
            item_field: Field within each array item to sum (e.g., "total_charges").
            total_field: Field containing expected total.
            tolerance: Allowed difference for floating point.
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.NESTED_SUM_VALIDATION,
                fields=[array_field, total_field],
                severity=severity,
                params={
                    "array_field": array_field,
                    "item_field": item_field,
                    "total_field": total_field,
                    "tolerance": tolerance,
                },
                message_template=f"Sum of {array_field}[].{item_field} must equal {{total_field}}",
            )
        )

    def add_required_if_rule(
        self,
        name: str,
        trigger_field: str,
        required_field: str,
        trigger_values: list[Any] | None = None,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring field B if field A has specific value.

        Args:
            name: Rule identifier.
            trigger_field: Field that triggers the requirement.
            required_field: Field that becomes required.
            trigger_values: Values that trigger requirement (any non-empty if None).
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.REQUIRED_IF,
                fields=[trigger_field, required_field],
                severity=severity,
                params={
                    "trigger_field": trigger_field,
                    "required_field": required_field,
                    "trigger_values": trigger_values,
                },
                message_template="{required_field} is required when {trigger_field} is present",
            )
        )

    def add_mutual_exclusive_rule(
        self,
        name: str,
        field_a: str,
        field_b: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule that only one of two fields can have a value.

        Args:
            name: Rule identifier.
            field_a: First mutually exclusive field.
            field_b: Second mutually exclusive field.
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.MUTUAL_EXCLUSIVE,
                fields=[field_a, field_b],
                severity=severity,
                message_template="Only one of {field_a} or {field_b} can have a value",
            )
        )

    def add_mutual_required_rule(
        self,
        name: str,
        fields: list[str],
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring all fields or none.

        Args:
            name: Rule identifier.
            fields: Fields that must all be present or all absent.
            severity: Severity of violation.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.MUTUAL_REQUIRED,
                fields=fields,
                severity=severity,
                message_template="All or none of these fields must be present: {fields}",
            )
        )

    def add_value_range_rule(
        self,
        name: str,
        value_field: str,
        min_field: str | None = None,
        max_field: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        severity: RuleSeverity = RuleSeverity.ERROR,
    ) -> None:
        """
        Add a rule requiring a value to be within a range.

        Args:
            name: Rule identifier.
            value_field: Field to check.
            min_field: Field containing minimum (optional).
            max_field: Field containing maximum (optional).
            min_value: Static minimum value (optional).
            max_value: Static maximum value (optional).
            severity: Severity of violation.
        """
        fields = [value_field]
        if min_field:
            fields.append(min_field)
        if max_field:
            fields.append(max_field)

        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.VALUE_RANGE,
                fields=fields,
                severity=severity,
                params={
                    "value_field": value_field,
                    "min_field": min_field,
                    "max_field": max_field,
                    "min_value": min_value,
                    "max_value": max_value,
                },
                message_template="{value_field} must be within allowed range",
            )
        )

    def add_custom_rule(
        self,
        name: str,
        fields: list[str],
        validator: Callable[[dict[str, Any], CrossFieldRule], RuleViolation | None],
        severity: RuleSeverity = RuleSeverity.ERROR,
        message_template: str = "",
    ) -> None:
        """
        Add a custom validation rule.

        Args:
            name: Rule identifier.
            fields: Fields involved in validation.
            validator: Function that validates and returns violation or None.
            severity: Severity of violation.
            message_template: Template for violation message.
        """
        self.rules.append(
            CrossFieldRule(
                name=name,
                rule_type=RuleType.CUSTOM,
                fields=fields,
                severity=severity,
                message_template=message_template,
            )
        )
        self._custom_validators[name] = validator

    def validate(self, data: dict[str, Any]) -> CrossFieldResult:
        """
        Validate data against all configured rules.

        Args:
            data: Dictionary of field names to values.

        Returns:
            CrossFieldResult with all violations found.
        """
        result = CrossFieldResult()

        for rule in self.rules:
            if not rule.enabled:
                continue

            result.rules_checked += 1
            violation = self._check_rule(rule, data)

            if violation:
                result.violations.append(violation)
                if violation.severity == RuleSeverity.ERROR:
                    result.errors.append(violation.message)
                    result.passed = False
                elif violation.severity == RuleSeverity.WARNING:
                    result.warnings.append(violation.message)
            else:
                result.rules_passed += 1

        logger.debug(
            f"Cross-field validation: "
            f"checked={result.rules_checked}, "
            f"passed={result.rules_passed}, "
            f"errors={len(result.errors)}"
        )

        return result

    def _check_rule(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """
        Check a single rule against the data.

        Args:
            rule: Rule to check.
            data: Data to validate.

        Returns:
            RuleViolation if rule is violated, None otherwise.
        """
        checkers = {
            RuleType.DATE_ORDER: self._check_date_order,
            RuleType.SUM_VALIDATION: self._check_sum,
            RuleType.NESTED_SUM_VALIDATION: self._check_nested_sum,
            RuleType.REQUIRED_IF: self._check_required_if,
            RuleType.REQUIRED_UNLESS: self._check_required_unless,
            RuleType.MUTUAL_EXCLUSIVE: self._check_mutual_exclusive,
            RuleType.MUTUAL_REQUIRED: self._check_mutual_required,
            RuleType.VALUE_RANGE: self._check_value_range,
            RuleType.CUSTOM: self._check_custom,
        }

        checker = checkers.get(rule.rule_type)
        if checker:
            return checker(rule, data)

        return None

    def _check_date_order(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check date ordering rule."""
        if len(rule.fields) < 2:
            return None

        earlier_field = rule.fields[0]
        later_field = rule.fields[1]
        allow_equal = rule.params.get("allow_equal", True)

        earlier_val = data.get(earlier_field)
        later_val = data.get(later_field)

        # Skip if either is empty
        if not earlier_val or not later_val:
            return None

        earlier_date = self._parse_date(earlier_val)
        later_date = self._parse_date(later_val)

        if earlier_date is None or later_date is None:
            return None

        is_valid = earlier_date < later_date if not allow_equal else earlier_date <= later_date

        if not is_valid:
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=(earlier_field, later_field),
                message=f"{earlier_field} ({earlier_date.date()}) must be before {later_field} ({later_date.date()})",
                expected=(
                    f"{earlier_field} <= {later_field}"
                    if allow_equal
                    else f"{earlier_field} < {later_field}"
                ),
                actual=f"{earlier_date.date()} vs {later_date.date()}",
            )

        return None

    def _check_sum(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check sum validation rule."""
        component_fields = rule.params.get("component_fields", [])
        total_field = rule.params.get("total_field")
        tolerance = rule.params.get("tolerance", 0.01)

        if not component_fields or not total_field:
            return None

        total_val = data.get(total_field)
        if total_val is None:
            return None

        total = self._to_float(total_val)
        if total is None:
            return None

        # Sum component values
        component_sum = 0.0
        for comp_field in component_fields:
            comp_val = data.get(comp_field)
            if comp_val is not None:
                comp_float = self._to_float(comp_val)
                if comp_float is not None:
                    component_sum += comp_float

        # Check within tolerance
        if abs(component_sum - total) > tolerance:
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=tuple(component_fields + [total_field]),
                message=f"Sum of {', '.join(component_fields)} ({component_sum:.2f}) does not equal {total_field} ({total:.2f})",
                expected=f"Sum = {total:.2f}",
                actual=f"Sum = {component_sum:.2f}",
            )

        return None

    def _check_nested_sum(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check nested array sum validation rule.

        Sums a field across all items in an array and compares to a total field.
        E.g., sum(service_lines[].total_charges) == total_charges
        """
        array_field = rule.params.get("array_field")
        item_field = rule.params.get("item_field")
        total_field = rule.params.get("total_field")
        tolerance = rule.params.get("tolerance", 0.01)

        if not array_field or not item_field or not total_field:
            return None

        # Get total value
        total_val = data.get(total_field)
        if total_val is None:
            return None

        total = self._to_float(total_val)
        if total is None:
            return None

        # Get array and sum item field values
        array_data = data.get(array_field)
        if not array_data or not isinstance(array_data, list):
            # No array data to validate - skip
            return None

        item_sum = 0.0
        valid_items = 0
        for item in array_data:
            if isinstance(item, dict):
                item_val = item.get(item_field)
                if item_val is not None:
                    item_float = self._to_float(item_val)
                    if item_float is not None:
                        item_sum += item_float
                        valid_items += 1

        # No valid items found - skip validation
        if valid_items == 0:
            return None

        # Check within tolerance
        if abs(item_sum - total) > tolerance:
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=(array_field, total_field),
                message=(
                    f"Sum of {array_field}[].{item_field} ({item_sum:.2f}) "
                    f"does not equal {total_field} ({total:.2f})"
                ),
                expected=f"Sum = {total:.2f}",
                actual=f"Sum = {item_sum:.2f} ({valid_items} items)",
            )

        return None

    def _check_required_if(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check required-if rule."""
        trigger_field = rule.params.get("trigger_field")
        required_field = rule.params.get("required_field")
        trigger_values = rule.params.get("trigger_values")

        if not trigger_field or not required_field:
            return None

        trigger_val = data.get(trigger_field)
        required_val = data.get(required_field)

        # Check if trigger condition is met
        trigger_met = False
        if trigger_values is not None:
            trigger_met = trigger_val in trigger_values
        else:
            trigger_met = not self._is_empty(trigger_val)

        # If trigger is met, required field must have value
        if trigger_met and self._is_empty(required_val):
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=(trigger_field, required_field),
                message=f"{required_field} is required when {trigger_field} is present",
                expected=f"{required_field} should have a value",
                actual=f"{required_field} is empty",
            )

        return None

    def _check_required_unless(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check required-unless rule (opposite of required-if)."""
        trigger_field = rule.params.get("trigger_field")
        required_field = rule.params.get("required_field")

        if not trigger_field or not required_field:
            return None

        trigger_val = data.get(trigger_field)
        required_val = data.get(required_field)

        # If trigger is empty, required field must have value
        if self._is_empty(trigger_val) and self._is_empty(required_val):
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=(trigger_field, required_field),
                message=f"{required_field} is required when {trigger_field} is empty",
                expected=f"{required_field} should have a value",
                actual="Both fields are empty",
            )

        return None

    def _check_mutual_exclusive(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check mutual exclusivity rule."""
        if len(rule.fields) < 2:
            return None

        field_a = rule.fields[0]
        field_b = rule.fields[1]

        val_a = data.get(field_a)
        val_b = data.get(field_b)

        # Both have values = violation
        if not self._is_empty(val_a) and not self._is_empty(val_b):
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=(field_a, field_b),
                message=f"Only one of {field_a} or {field_b} should have a value",
                expected="One field empty",
                actual="Both fields have values",
            )

        return None

    def _check_mutual_required(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check mutual requirement rule (all or none)."""
        if len(rule.fields) < 2:
            return None

        has_values = [not self._is_empty(data.get(f)) for f in rule.fields]

        # All or none
        if any(has_values) and not all(has_values):
            present = [f for f, v in zip(rule.fields, has_values, strict=False) if v]
            missing = [f for f, v in zip(rule.fields, has_values, strict=False) if not v]

            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=tuple(rule.fields),
                message=f"If any of these fields are present, all must be: {', '.join(rule.fields)}",
                expected="All or none",
                actual=f"Present: {present}, Missing: {missing}",
            )

        return None

    def _check_value_range(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check value range rule."""
        value_field = rule.params.get("value_field")
        if not value_field:
            return None

        value = data.get(value_field)
        if value is None:
            return None

        value_float = self._to_float(value)
        if value_float is None:
            return None

        # Get min/max from fields or static values
        min_val = rule.params.get("min_value")
        max_val = rule.params.get("max_value")

        min_field = rule.params.get("min_field")
        max_field = rule.params.get("max_field")

        if min_field:
            field_min = self._to_float(data.get(min_field))
            if field_min is not None:
                min_val = field_min

        if max_field:
            field_max = self._to_float(data.get(max_field))
            if field_max is not None:
                max_val = field_max

        # Check range
        if min_val is not None and value_float < min_val:
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=tuple(rule.fields),
                message=f"{value_field} ({value_float}) is below minimum ({min_val})",
                expected=f">= {min_val}",
                actual=str(value_float),
            )

        if max_val is not None and value_float > max_val:
            return RuleViolation(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                fields=tuple(rule.fields),
                message=f"{value_field} ({value_float}) exceeds maximum ({max_val})",
                expected=f"<= {max_val}",
                actual=str(value_float),
            )

        return None

    def _check_custom(
        self,
        rule: CrossFieldRule,
        data: dict[str, Any],
    ) -> RuleViolation | None:
        """Check custom validation rule."""
        validator = self._custom_validators.get(rule.name)
        if validator:
            return validator(data, rule)
        return None

    def _parse_date(self, value: Any) -> datetime | None:
        """Parse date value to datetime."""
        if isinstance(value, datetime):
            return value

        str_val = str(value).strip()
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(str_val, fmt)
            except ValueError:
                continue

        return None

    def _to_float(self, value: Any) -> float | None:
        """Convert value to float."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        try:
            import re

            # Remove currency symbols and formatting
            cleaned = re.sub(r"[$,\s]", "", str(value))
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _is_empty(self, value: Any) -> bool:
        """Check if value is empty."""
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False


class MedicalDocumentRules:
    """
    Pre-configured cross-field rules for medical documents.

    Provides standard rule sets for CMS-1500, UB-04, and EOB documents.
    """

    @staticmethod
    def get_cms1500_rules() -> CrossFieldValidator:
        """Get validator with CMS-1500 specific rules."""
        validator = CrossFieldValidator()

        # Date ordering
        validator.add_date_order_rule(
            "patient_dob_before_service",
            "patient_birth_date",
            "service_date_from",
            allow_equal=False,
        )
        validator.add_date_order_rule(
            "service_date_order",
            "service_date_from",
            "service_date_to",
            allow_equal=True,
        )
        validator.add_date_order_rule(
            "hospitalization_dates",
            "hospitalization_from",
            "hospitalization_to",
            allow_equal=True,
        )

        # Required dependencies
        validator.add_required_if_rule(
            "cpt_requires_diagnosis",
            "cpt_code",
            "diagnosis_pointer",
        )
        validator.add_required_if_rule(
            "modifier_requires_cpt",
            "modifier",
            "cpt_code",
        )

        # Sum validation for service lines
        validator.add_sum_rule(
            "line_charges_total",
            [
                "line_1_charges",
                "line_2_charges",
                "line_3_charges",
                "line_4_charges",
                "line_5_charges",
                "line_6_charges",
            ],
            "total_charges",
            tolerance=0.01,
        )

        return validator

    @staticmethod
    def get_ub04_rules() -> CrossFieldValidator:
        """Get validator with UB-04 specific rules."""
        validator = CrossFieldValidator()

        # Date ordering
        validator.add_date_order_rule(
            "admission_before_discharge",
            "admission_date",
            "discharge_date",
            allow_equal=True,
        )
        validator.add_date_order_rule(
            "statement_from_to",
            "statement_from_date",
            "statement_to_date",
            allow_equal=True,
        )

        # Required dependencies
        validator.add_mutual_required_rule(
            "occurrence_code_date",
            ["occurrence_code", "occurrence_date"],
        )
        validator.add_required_if_rule(
            "attending_npi_with_name",
            "attending_physician_name",
            "attending_physician_npi",
        )

        # Sum validation - service line charges should sum to total
        validator.add_nested_sum_rule(
            "revenue_totals",
            array_field="service_lines",
            item_field="total_charges",
            total_field="total_charges",
            tolerance=0.01,
        )

        # Non-covered charges validation
        validator.add_nested_sum_rule(
            "non_covered_totals",
            array_field="service_lines",
            item_field="non_covered_charges",
            total_field="total_non_covered_charges",
            tolerance=0.01,
        )

        # Value ranges
        validator.add_value_range_rule(
            "total_charges_positive",
            "total_charges",
            min_value=0.0,
        )

        return validator

    @staticmethod
    def get_eob_rules() -> CrossFieldValidator:
        """Get validator with EOB specific rules."""
        validator = CrossFieldValidator()

        # Date ordering
        validator.add_date_order_rule(
            "service_before_payment",
            "service_date",
            "payment_date",
            allow_equal=True,
        )

        # Sum validation
        validator.add_sum_rule(
            "payment_calculation",
            ["allowed_amount", "patient_responsibility"],
            "billed_amount",
            tolerance=5.0,  # Allow some flexibility for adjustments
        )

        # Required dependencies
        validator.add_required_if_rule(
            "denial_requires_reason",
            "denial_code",
            "denial_reason",
        )

        # Mutual requirements
        validator.add_mutual_required_rule(
            "adjustment_code_amount",
            ["adjustment_code", "adjustment_amount"],
        )

        return validator


def validate_cross_fields(
    data: dict[str, Any],
    document_type: str = "generic",
) -> CrossFieldResult:
    """
    Validate cross-field relationships in extracted data.

    Convenience function that applies document-specific rules.

    Args:
        data: Dictionary of extracted field values.
        document_type: Type of document (cms1500, ub04, eob, generic).

    Returns:
        CrossFieldResult with validation details.

    Example:
        result = validate_cross_fields(
            data={"admission_date": "2024-01-15", "discharge_date": "2024-01-10"},
            document_type="ub04",
        )
        if not result.passed:
            print(result.errors)
    """
    validator_map = {
        "cms1500": MedicalDocumentRules.get_cms1500_rules,
        "ub04": MedicalDocumentRules.get_ub04_rules,
        "eob": MedicalDocumentRules.get_eob_rules,
    }

    factory = validator_map.get(document_type.lower())
    if factory:
        validator = factory()
    else:
        validator = CrossFieldValidator()

    return validator.validate(data)
