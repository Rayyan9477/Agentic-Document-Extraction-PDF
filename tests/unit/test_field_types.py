"""
Tests for src/schemas/field_types.py — FieldType enum, CrossFieldRule, FieldDefinition.
"""

import pytest

from src.schemas.field_types import (
    CrossFieldRule,
    FieldDefinition,
    FieldType,
    RuleOperator,
)


# ---------------------------------------------------------------------------
# FieldType enum properties
# ---------------------------------------------------------------------------


class TestFieldTypeProperties:

    def test_is_medical_code_true(self):
        medical = [
            FieldType.CPT_CODE,
            FieldType.ICD10_CODE,
            FieldType.NPI,
            FieldType.HCPCS_CODE,
            FieldType.NDC_CODE,
            FieldType.TAXONOMY_CODE,
            FieldType.CARC_CODE,
            FieldType.RARC_CODE,
        ]
        for ft in medical:
            assert ft.is_medical_code is True, f"{ft} should be medical code"

    def test_is_medical_code_false(self):
        non_medical = [FieldType.STRING, FieldType.INTEGER, FieldType.CURRENCY, FieldType.DATE]
        for ft in non_medical:
            assert ft.is_medical_code is False, f"{ft} should not be medical code"

    def test_is_numeric_true(self):
        numeric = [FieldType.INTEGER, FieldType.FLOAT, FieldType.CURRENCY, FieldType.PERCENTAGE]
        for ft in numeric:
            assert ft.is_numeric is True, f"{ft} should be numeric"

    def test_is_numeric_false(self):
        assert FieldType.STRING.is_numeric is False
        assert FieldType.DATE.is_numeric is False

    def test_is_identifier_true(self):
        ids = [FieldType.SSN, FieldType.MEMBER_ID, FieldType.NPI, FieldType.EIN]
        for ft in ids:
            assert ft.is_identifier is True, f"{ft} should be identifier"

    def test_is_identifier_false(self):
        assert FieldType.STRING.is_identifier is False
        assert FieldType.CURRENCY.is_identifier is False

    def test_requires_phi_protection_true(self):
        phi = [FieldType.SSN, FieldType.NAME, FieldType.DATE, FieldType.PHONE, FieldType.EMAIL]
        for ft in phi:
            assert ft.requires_phi_protection is True, f"{ft} should require PHI protection"

    def test_requires_phi_protection_false(self):
        assert FieldType.CURRENCY.requires_phi_protection is False
        assert FieldType.CPT_CODE.requires_phi_protection is False

    def test_string_value(self):
        assert FieldType.STRING.value == "string"
        assert FieldType.CPT_CODE.value == "cpt_code"

    def test_enum_is_str(self):
        """FieldType inherits from str."""
        assert isinstance(FieldType.STRING, str)


# ---------------------------------------------------------------------------
# RuleOperator enum
# ---------------------------------------------------------------------------


class TestRuleOperator:

    def test_values(self):
        assert RuleOperator.EQUALS.value == "equals"
        assert RuleOperator.SUM_EQUALS.value == "sum_equals"
        assert RuleOperator.REQUIRES_IF.value == "requires_if"


# ---------------------------------------------------------------------------
# CrossFieldRule
# ---------------------------------------------------------------------------


class TestCrossFieldRule:

    def test_create(self):
        rule = CrossFieldRule(
            source_field="total_charges",
            target_field="sum_of_line_items",
            operator=RuleOperator.SUM_EQUALS,
        )
        assert rule.source_field == "total_charges"
        assert rule.severity == "error"

    def test_default_error_messages(self):
        rule = CrossFieldRule("a", "b", RuleOperator.EQUALS)
        assert "must equal" in rule.get_error_message()

    def test_custom_error_message(self):
        rule = CrossFieldRule("a", "b", RuleOperator.EQUALS, error_message="custom msg")
        assert rule.get_error_message() == "custom msg"

    def test_fallback_error_message(self):
        """Operators not in the messages dict get fallback message."""
        rule = CrossFieldRule("a", "b", RuleOperator.CONTAINS)
        msg = rule.get_error_message()
        assert "Cross-field validation failed" in msg

    def test_date_before_message(self):
        rule = CrossFieldRule("start", "end", RuleOperator.DATE_BEFORE)
        assert "before" in rule.get_error_message()


# ---------------------------------------------------------------------------
# FieldDefinition — construction + validation
# ---------------------------------------------------------------------------


class TestFieldDefinitionInit:

    def test_basic_creation(self):
        fd = FieldDefinition(
            name="patient_name",
            display_name="Patient Name",
            field_type=FieldType.NAME,
        )
        assert fd.name == "patient_name"
        assert fd.required is False
        assert fd.confidence_threshold == 0.5

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Field name is required"):
            FieldDefinition(name="", display_name="X", field_type=FieldType.STRING)

    def test_non_snake_case_raises(self):
        with pytest.raises(ValueError, match="snake_case"):
            FieldDefinition(name="PatientName", display_name="X", field_type=FieldType.STRING)

    def test_list_without_item_type_raises(self):
        with pytest.raises(ValueError, match="list_item_type"):
            FieldDefinition(name="items", display_name="Items", field_type=FieldType.LIST)

    def test_list_with_item_type_ok(self):
        fd = FieldDefinition(
            name="items",
            display_name="Items",
            field_type=FieldType.LIST,
            list_item_type=FieldType.STRING,
        )
        assert fd.list_item_type == FieldType.STRING

    def test_object_without_nested_schema_raises(self):
        with pytest.raises(ValueError, match="nested_schema"):
            FieldDefinition(name="addr", display_name="Address", field_type=FieldType.OBJECT)

    def test_object_with_nested_schema_ok(self):
        fd = FieldDefinition(
            name="addr",
            display_name="Address",
            field_type=FieldType.OBJECT,
            nested_schema="address_schema",
        )
        assert fd.nested_schema == "address_schema"


# ---------------------------------------------------------------------------
# FieldDefinition — validate()
# ---------------------------------------------------------------------------


class TestFieldDefinitionValidate:

    def test_required_none_fails(self):
        fd = FieldDefinition(
            name="ssn", display_name="SSN", field_type=FieldType.SSN, required=True,
        )
        ok, err = fd.validate(None)
        assert ok is False
        assert "required" in err.lower()

    def test_optional_none_passes(self):
        fd = FieldDefinition(name="ssn", display_name="SSN", field_type=FieldType.SSN)
        ok, err = fd.validate(None)
        assert ok is True

    def test_integer_valid(self):
        fd = FieldDefinition(name="count", display_name="Count", field_type=FieldType.INTEGER)
        ok, err = fd.validate(42)
        assert ok is True

    def test_integer_string_valid(self):
        fd = FieldDefinition(name="count", display_name="Count", field_type=FieldType.INTEGER)
        ok, err = fd.validate("42")
        assert ok is True

    def test_integer_invalid(self):
        fd = FieldDefinition(name="count", display_name="Count", field_type=FieldType.INTEGER)
        ok, err = fd.validate("abc")
        assert ok is False

    def test_float_currency_valid(self):
        fd = FieldDefinition(name="amount", display_name="Amount", field_type=FieldType.CURRENCY)
        ok, _ = fd.validate("$1,234.56")
        assert ok is True

    def test_pattern_match(self):
        fd = FieldDefinition(
            name="code", display_name="Code", field_type=FieldType.STRING,
            pattern=r"^[A-Z]{3}$",
        )
        ok, _ = fd.validate("ABC")
        assert ok is True

    def test_pattern_no_match(self):
        fd = FieldDefinition(
            name="code", display_name="Code", field_type=FieldType.STRING,
            pattern=r"^[A-Z]{3}$",
        )
        ok, err = fd.validate("abc")
        assert ok is False

    def test_allowed_values(self):
        fd = FieldDefinition(
            name="status", display_name="Status", field_type=FieldType.STRING,
            allowed_values=["active", "inactive"],
        )
        ok, _ = fd.validate("active")
        assert ok is True
        ok2, err2 = fd.validate("unknown")
        assert ok2 is False

    def test_min_max_value(self):
        fd = FieldDefinition(
            name="pct", display_name="Pct", field_type=FieldType.FLOAT,
            min_value=0.0, max_value=100.0,
        )
        ok, _ = fd.validate(50.0)
        assert ok is True
        ok2, _ = fd.validate(150.0)
        assert ok2 is False
        ok3, _ = fd.validate(-1.0)
        assert ok3 is False

    def test_min_max_length(self):
        fd = FieldDefinition(
            name="note", display_name="Note", field_type=FieldType.STRING,
            min_length=3, max_length=10,
        )
        ok, _ = fd.validate("hello")
        assert ok is True
        ok2, _ = fd.validate("hi")
        assert ok2 is False
        ok3, _ = fd.validate("x" * 20)
        assert ok3 is False

    def test_custom_validation_func(self):
        fd = FieldDefinition(
            name="even", display_name="Even", field_type=FieldType.INTEGER,
            validation_func=lambda v: v % 2 == 0,
        )
        ok, _ = fd.validate(4)
        assert ok is True
        ok2, _ = fd.validate(3)
        assert ok2 is False

    def test_custom_validation_exception(self):
        fd = FieldDefinition(
            name="bad", display_name="Bad", field_type=FieldType.STRING,
            validation_func=lambda v: 1 / 0,
        )
        ok, err = fd.validate("x")
        assert ok is False
        assert "validation error" in err.lower()


# ---------------------------------------------------------------------------
# FieldDefinition — transform()
# ---------------------------------------------------------------------------


class TestFieldDefinitionTransform:

    def test_none_returns_default(self):
        fd = FieldDefinition(
            name="x", display_name="X", field_type=FieldType.STRING, default="N/A",
        )
        assert fd.transform(None) == "N/A"

    def test_custom_transform(self):
        fd = FieldDefinition(
            name="upper", display_name="U", field_type=FieldType.STRING,
            transform_func=lambda v: v.upper(),
        )
        assert fd.transform("hello") == "HELLO"

    def test_currency_transform(self):
        fd = FieldDefinition(name="amt", display_name="Amt", field_type=FieldType.CURRENCY)
        result = fd.transform("$1,234.56")
        assert result == 1234.56

    def test_percentage_transform(self):
        fd = FieldDefinition(name="pct", display_name="Pct", field_type=FieldType.PERCENTAGE)
        assert fd.transform("85.5%") == 85.5

    def test_phone_transform_10_digits(self):
        fd = FieldDefinition(name="ph", display_name="Ph", field_type=FieldType.PHONE)
        assert fd.transform("(555) 123-4567") == "555-123-4567"

    def test_phone_transform_11_digits_country_code(self):
        fd = FieldDefinition(name="ph", display_name="Ph", field_type=FieldType.PHONE)
        assert fd.transform("15551234567") == "555-123-4567"

    def test_ssn_transform(self):
        fd = FieldDefinition(name="ssn", display_name="SSN", field_type=FieldType.SSN)
        assert fd.transform("123456789") == "123-45-6789"

    def test_non_transformable_passthrough(self):
        fd = FieldDefinition(name="txt", display_name="T", field_type=FieldType.STRING)
        assert fd.transform("hello") == "hello"


# ---------------------------------------------------------------------------
# FieldDefinition — to_prompt_description / to_dict / compiled_pattern
# ---------------------------------------------------------------------------


class TestFieldDefinitionHelpers:

    def test_to_prompt_description_basic(self):
        fd = FieldDefinition(
            name="name", display_name="Name", field_type=FieldType.NAME,
            description="Full patient name",
        )
        desc = fd.to_prompt_description()
        assert "Full patient name" in desc

    def test_to_prompt_description_with_examples(self):
        fd = FieldDefinition(
            name="npi", display_name="NPI", field_type=FieldType.NPI,
            description="Provider NPI",
            examples=["1234567893"],
        )
        desc = fd.to_prompt_description()
        assert "1234567893" in desc

    def test_to_prompt_description_with_location_hint(self):
        fd = FieldDefinition(
            name="npi", display_name="NPI", field_type=FieldType.NPI,
            location_hint="Top right corner",
        )
        desc = fd.to_prompt_description()
        assert "Top right corner" in desc

    def test_to_dict(self):
        fd = FieldDefinition(
            name="code", display_name="Code", field_type=FieldType.CPT_CODE,
            required=True, pattern=r"^\d{5}$",
        )
        d = fd.to_dict()
        assert d["name"] == "code"
        assert d["field_type"] == "cpt_code"
        assert d["required"] is True
        assert d["pattern"] == r"^\d{5}$"

    def test_compiled_pattern(self):
        fd = FieldDefinition(
            name="code", display_name="Code", field_type=FieldType.STRING,
            pattern=r"^[A-Z]+$",
        )
        pat = fd.compiled_pattern
        assert pat is not None
        assert pat.match("ABC")
        assert not pat.match("abc")

    def test_compiled_pattern_none(self):
        fd = FieldDefinition(name="x", display_name="X", field_type=FieldType.STRING)
        assert fd.compiled_pattern is None
