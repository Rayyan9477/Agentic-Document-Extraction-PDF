"""
Unit tests for Phase 4B: Finance-Specific Schemas.

Tests Invoice, W-2, Form 1099, and Bank Statement schemas,
including field definitions, cross-field rules, validation,
schema registry integration, and new FieldType values.
"""

from __future__ import annotations

import pytest

from src.schemas.bank_statement import (
    ACCOUNT_HOLDER_FIELDS,
    ACCOUNT_INFO_FIELDS,
    BALANCE_FIELDS,
    BANK_FIELDS,
    BANK_STATEMENT_CROSS_FIELD_RULES,
    BANK_STATEMENT_SCHEMA,
    FEE_FIELDS,
    INTEREST_FIELDS,
    TRANSACTION_SUMMARY_FIELDS,
)
from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import FieldDefinition, FieldType, RuleOperator
from src.schemas.form_1099 import (
    FORM_1099_CROSS_FIELD_RULES,
    FORM_1099_META_FIELDS,
    FORM_1099_SCHEMA,
    FORM_1099_STATE_FIELDS,
    INT_FIELDS,
    MISC_FIELDS,
    NEC_FIELDS,
    PAYER_FIELDS,
    RECIPIENT_FIELDS,
)
from src.schemas.invoice import (
    BUYER_FIELDS,
    INVOICE_CROSS_FIELD_RULES,
    INVOICE_ID_FIELDS,
    INVOICE_SCHEMA,
    PAYMENT_FIELDS,
    TOTAL_FIELDS,
    VENDOR_FIELDS,
)
from src.schemas.w2 import (
    ADDITIONAL_TAX_FIELDS,
    EMPLOYER_FIELDS,
    EMPLOYEE_FIELDS,
    STATE_LOCAL_FIELDS,
    W2_CROSS_FIELD_RULES,
    W2_META_FIELDS,
    W2_SCHEMA,
    WAGE_FIELDS,
)


# ──────────────────────────────────────────────────────────────────
# New FieldType Values
# ──────────────────────────────────────────────────────────────────


class TestFinanceFieldTypes:
    """Verify the new finance-related FieldType enum values."""

    def test_ein_exists(self):
        assert FieldType.EIN == "ein"

    def test_routing_number_exists(self):
        assert FieldType.ROUTING_NUMBER == "routing_number"

    def test_bank_account_exists(self):
        assert FieldType.BANK_ACCOUNT == "bank_account"

    def test_ein_is_identifier(self):
        assert FieldType.EIN.is_identifier is True

    def test_routing_number_is_identifier(self):
        assert FieldType.ROUTING_NUMBER.is_identifier is True

    def test_bank_account_is_identifier(self):
        assert FieldType.BANK_ACCOUNT.is_identifier is True

    def test_ein_not_medical_code(self):
        assert FieldType.EIN.is_medical_code is False

    def test_routing_number_not_numeric(self):
        assert FieldType.ROUTING_NUMBER.is_numeric is False


# ──────────────────────────────────────────────────────────────────
# New DocumentType Values
# ──────────────────────────────────────────────────────────────────


class TestFinanceDocumentTypes:
    """Verify finance document types exist in the enum."""

    def test_invoice_type(self):
        assert DocumentType.INVOICE == "invoice"

    def test_w2_type(self):
        assert DocumentType.W2 == "w2"

    def test_form_1099_type(self):
        assert DocumentType.FORM_1099 == "form_1099"

    def test_bank_statement_type(self):
        assert DocumentType.BANK_STATEMENT == "bank_statement"


# ──────────────────────────────────────────────────────────────────
# Invoice Schema Tests
# ──────────────────────────────────────────────────────────────────


class TestInvoiceSchema:
    """Tests for the Invoice schema."""

    def test_schema_identity(self):
        assert INVOICE_SCHEMA.name == "invoice"
        assert INVOICE_SCHEMA.display_name == "Invoice"
        assert INVOICE_SCHEMA.document_type == DocumentType.INVOICE
        assert INVOICE_SCHEMA.version == "1.0.0"

    def test_field_groups_combined(self):
        expected_count = (
            len(VENDOR_FIELDS)
            + len(BUYER_FIELDS)
            + len(INVOICE_ID_FIELDS)
            + len(TOTAL_FIELDS)
            + len(PAYMENT_FIELDS)
        )
        assert len(INVOICE_SCHEMA.fields) == expected_count

    def test_vendor_fields_count(self):
        assert len(VENDOR_FIELDS) == 5

    def test_buyer_fields_count(self):
        assert len(BUYER_FIELDS) == 3

    def test_invoice_id_fields_count(self):
        assert len(INVOICE_ID_FIELDS) == 5

    def test_total_fields_count(self):
        assert len(TOTAL_FIELDS) == 9

    def test_payment_fields_count(self):
        assert len(PAYMENT_FIELDS) == 4

    def test_required_fields(self):
        required = INVOICE_SCHEMA.get_required_fields()
        required_names = {f.name for f in required}
        assert "vendor_name" in required_names
        assert "buyer_name" in required_names
        assert "invoice_number" in required_names
        assert "invoice_date" in required_names
        assert "total_amount" in required_names

    def test_vendor_ein_field(self):
        field = INVOICE_SCHEMA.get_field("vendor_ein")
        assert field is not None
        assert field.field_type == FieldType.EIN
        assert field.pattern == r"^\d{2}-?\d{7}$"

    def test_bank_routing_field(self):
        field = INVOICE_SCHEMA.get_field("bank_routing_number")
        assert field is not None
        assert field.field_type == FieldType.ROUTING_NUMBER

    def test_bank_account_field(self):
        field = INVOICE_SCHEMA.get_field("bank_account_number")
        assert field is not None
        assert field.field_type == FieldType.BANK_ACCOUNT

    def test_currency_field_allowed_values(self):
        field = INVOICE_SCHEMA.get_field("currency")
        assert field is not None
        assert "USD" in field.allowed_values
        assert "EUR" in field.allowed_values

    def test_cross_field_rules(self):
        assert len(INVOICE_CROSS_FIELD_RULES) == 2
        rule_pairs = [(r.source_field, r.target_field) for r in INVOICE_CROSS_FIELD_RULES]
        assert ("invoice_date", "due_date") in rule_pairs
        assert ("subtotal", "total_amount") in rule_pairs

    def test_date_before_rule_operator(self):
        date_rule = next(
            r for r in INVOICE_CROSS_FIELD_RULES if r.source_field == "invoice_date"
        )
        assert date_rule.operator == RuleOperator.DATE_BEFORE
        assert date_rule.severity == "warning"

    def test_classification_hints(self):
        hints = INVOICE_SCHEMA.classification_hints
        assert "INVOICE" in hints
        assert "Amount Due" in hints
        assert "Bill To" in hints

    def test_required_sections(self):
        assert "vendor_info" in INVOICE_SCHEMA.required_sections
        assert "invoice_details" in INVOICE_SCHEMA.required_sections
        assert "totals" in INVOICE_SCHEMA.required_sections

    def test_validate_valid_result(self):
        result = {
            "vendor_name": "Acme Corp",
            "buyer_name": "John Smith",
            "invoice_number": "INV-001",
            "invoice_date": "01/15/2025",
            "total_amount": "$1,000.00",
        }
        errors, warnings = INVOICE_SCHEMA.validate_result(result)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        result = {"vendor_name": "Acme Corp"}
        errors, warnings = INVOICE_SCHEMA.validate_result(result)
        assert len(errors) > 0

    def test_validate_date_ordering(self):
        result = {
            "vendor_name": "Acme Corp",
            "buyer_name": "John Smith",
            "invoice_number": "INV-001",
            "invoice_date": "2025-03-15",
            "due_date": "2025-01-01",
            "total_amount": "1000",
        }
        errors, warnings = INVOICE_SCHEMA.validate_result(result)
        assert any("before" in w.lower() for w in warnings)

    def test_generate_extraction_prompt(self):
        prompt = INVOICE_SCHEMA.generate_extraction_prompt()
        assert "Invoice" in prompt
        assert "vendor_name" in prompt
        assert "REQUIRED" in prompt

    def test_to_dict(self):
        d = INVOICE_SCHEMA.to_dict()
        assert d["name"] == "invoice"
        assert d["document_type"] == "invoice"
        assert len(d["fields"]) == len(INVOICE_SCHEMA.fields)


# ──────────────────────────────────────────────────────────────────
# W-2 Schema Tests
# ──────────────────────────────────────────────────────────────────


class TestW2Schema:
    """Tests for the W-2 schema."""

    def test_schema_identity(self):
        assert W2_SCHEMA.name == "w2"
        assert W2_SCHEMA.display_name == "W-2 Wage and Tax Statement"
        assert W2_SCHEMA.document_type == DocumentType.W2

    def test_field_groups_combined(self):
        expected_count = (
            len(EMPLOYER_FIELDS)
            + len(EMPLOYEE_FIELDS)
            + len(WAGE_FIELDS)
            + len(ADDITIONAL_TAX_FIELDS)
            + len(STATE_LOCAL_FIELDS)
            + len(W2_META_FIELDS)
        )
        assert len(W2_SCHEMA.fields) == expected_count

    def test_employer_fields(self):
        assert len(EMPLOYER_FIELDS) == 4
        names = {f.name for f in EMPLOYER_FIELDS}
        assert "employer_name" in names
        assert "employer_ein" in names

    def test_employee_fields(self):
        assert len(EMPLOYEE_FIELDS) == 5
        names = {f.name for f in EMPLOYEE_FIELDS}
        assert "employee_ssn" in names
        assert "employee_first_name" in names
        assert "employee_last_name" in names

    def test_wage_fields(self):
        assert len(WAGE_FIELDS) == 8
        names = {f.name for f in WAGE_FIELDS}
        assert "wages_tips_other" in names
        assert "federal_income_tax" in names
        assert "social_security_wages" in names
        assert "medicare_wages" in names

    def test_box_12_fields(self):
        box12_fields = [f for f in ADDITIONAL_TAX_FIELDS if f.name.startswith("box_12")]
        assert len(box12_fields) == 8  # 4 codes + 4 amounts

    def test_checkbox_fields(self):
        checkbox_fields = [
            f for f in ADDITIONAL_TAX_FIELDS if f.field_type == FieldType.BOOLEAN
        ]
        checkbox_names = {f.name for f in checkbox_fields}
        assert "statutory_employee" in checkbox_names
        assert "retirement_plan" in checkbox_names
        assert "third_party_sick_pay" in checkbox_names

    def test_required_fields(self):
        required = W2_SCHEMA.get_required_fields()
        required_names = {f.name for f in required}
        assert "employer_name" in required_names
        assert "employer_ein" in required_names
        assert "employee_ssn" in required_names
        assert "wages_tips_other" in required_names
        assert "federal_income_tax" in required_names
        assert "tax_year" in required_names

    def test_ssn_field_type(self):
        field = W2_SCHEMA.get_field("employee_ssn")
        assert field is not None
        assert field.field_type == FieldType.SSN
        assert field.field_type.requires_phi_protection is True

    def test_ein_field_pattern(self):
        field = W2_SCHEMA.get_field("employer_ein")
        assert field is not None
        assert field.pattern == r"^\d{2}-?\d{7}$"

    def test_tax_year_bounds(self):
        field = W2_SCHEMA.get_field("tax_year")
        assert field is not None
        assert field.min_value == 2000
        assert field.max_value == 2030

    def test_cross_field_rules(self):
        assert len(W2_CROSS_FIELD_RULES) == 4
        sources = {r.source_field for r in W2_CROSS_FIELD_RULES}
        assert "social_security_tax" in sources
        assert "medicare_tax" in sources
        assert "federal_income_tax" in sources
        assert "state_income_tax" in sources

    def test_all_cross_rules_are_warnings(self):
        for rule in W2_CROSS_FIELD_RULES:
            assert rule.severity == "warning"

    def test_classification_hints(self):
        hints = W2_SCHEMA.classification_hints
        assert "W-2" in hints
        assert "Wage and Tax Statement" in hints
        assert "Federal income tax withheld" in hints

    def test_validate_valid_w2(self):
        result = {
            "employer_name": "Acme Corp",
            "employer_ein": "12-3456789",
            "employee_ssn": "123-45-6789",
            "employee_first_name": "John",
            "employee_last_name": "Smith",
            "wages_tips_other": "$75,000.00",
            "federal_income_tax": "$12,500.00",
            "tax_year": 2024,
        }
        errors, warnings = W2_SCHEMA.validate_result(result)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        result = {"employer_name": "Acme Corp"}
        errors, _ = W2_SCHEMA.validate_result(result)
        assert len(errors) > 0

    def test_state_local_fields(self):
        assert len(STATE_LOCAL_FIELDS) == 6
        names = {f.name for f in STATE_LOCAL_FIELDS}
        assert "state" in names
        assert "state_wages" in names
        assert "locality_name" in names

    def test_employee_suffix_allowed_values(self):
        field = W2_SCHEMA.get_field("employee_suffix")
        assert field is not None
        assert "Jr." in field.allowed_values
        assert "III" in field.allowed_values


# ──────────────────────────────────────────────────────────────────
# Form 1099 Schema Tests
# ──────────────────────────────────────────────────────────────────


class TestForm1099Schema:
    """Tests for the Form 1099 schema."""

    def test_schema_identity(self):
        assert FORM_1099_SCHEMA.name == "form_1099"
        assert FORM_1099_SCHEMA.display_name == "Form 1099"
        assert FORM_1099_SCHEMA.document_type == DocumentType.FORM_1099

    def test_field_groups_combined(self):
        expected_count = (
            len(PAYER_FIELDS)
            + len(RECIPIENT_FIELDS)
            + len(NEC_FIELDS)
            + len(MISC_FIELDS)
            + len(INT_FIELDS)
            + len(FORM_1099_META_FIELDS)
            + len(FORM_1099_STATE_FIELDS)
        )
        assert len(FORM_1099_SCHEMA.fields) == expected_count

    def test_payer_fields(self):
        assert len(PAYER_FIELDS) == 4
        names = {f.name for f in PAYER_FIELDS}
        assert "payer_name" in names
        assert "payer_tin" in names

    def test_recipient_fields(self):
        assert len(RECIPIENT_FIELDS) == 5
        names = {f.name for f in RECIPIENT_FIELDS}
        assert "recipient_name" in names
        assert "recipient_tin" in names

    def test_nec_fields(self):
        assert len(NEC_FIELDS) == 3
        names = {f.name for f in NEC_FIELDS}
        assert "nonemployee_compensation" in names

    def test_misc_fields(self):
        assert len(MISC_FIELDS) == 11
        names = {f.name for f in MISC_FIELDS}
        assert "rents" in names
        assert "royalties" in names
        assert "medical_healthcare_payments" in names

    def test_int_fields(self):
        assert len(INT_FIELDS) == 11
        names = {f.name for f in INT_FIELDS}
        assert "interest_income" in names
        assert "tax_exempt_interest" in names

    def test_form_variant_allowed_values(self):
        field = FORM_1099_SCHEMA.get_field("form_variant")
        assert field is not None
        assert "1099-NEC" in field.allowed_values
        assert "1099-MISC" in field.allowed_values
        assert "1099-INT" in field.allowed_values
        assert "1099-K" in field.allowed_values

    def test_required_fields(self):
        required = FORM_1099_SCHEMA.get_required_fields()
        required_names = {f.name for f in required}
        assert "payer_name" in required_names
        assert "payer_tin" in required_names
        assert "recipient_name" in required_names
        assert "recipient_tin" in required_names
        assert "tax_year" in required_names

    def test_corrected_boolean(self):
        field = FORM_1099_SCHEMA.get_field("corrected")
        assert field is not None
        assert field.field_type == FieldType.BOOLEAN

    def test_fatca_boolean(self):
        field = FORM_1099_SCHEMA.get_field("fatca_filing")
        assert field is not None
        assert field.field_type == FieldType.BOOLEAN

    def test_state_fields(self):
        assert len(FORM_1099_STATE_FIELDS) == 8
        names = {f.name for f in FORM_1099_STATE_FIELDS}
        assert "state_1" in names
        assert "state_2" in names
        assert "state_tax_withheld_1" in names
        assert "state_tax_withheld_2" in names

    def test_cross_field_rules(self):
        assert len(FORM_1099_CROSS_FIELD_RULES) == 2
        rule_pairs = [
            (r.source_field, r.target_field) for r in FORM_1099_CROSS_FIELD_RULES
        ]
        assert ("federal_tax_withheld_nec", "nonemployee_compensation") in rule_pairs
        assert ("state_tax_withheld_1", "state_income_1") in rule_pairs

    def test_classification_hints(self):
        hints = FORM_1099_SCHEMA.classification_hints
        assert "1099" in hints
        assert "1099-NEC" in hints
        assert "Interest Income" in hints

    def test_validate_valid_1099(self):
        result = {
            "payer_name": "ABC Corp",
            "payer_tin": "12-3456789",
            "recipient_name": "John Smith",
            "recipient_tin": "123-45-6789",
            "tax_year": 2024,
            "nonemployee_compensation": "$50,000.00",
        }
        errors, warnings = FORM_1099_SCHEMA.validate_result(result)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        result = {"payer_name": "ABC Corp"}
        errors, _ = FORM_1099_SCHEMA.validate_result(result)
        assert len(errors) > 0


# ──────────────────────────────────────────────────────────────────
# Bank Statement Schema Tests
# ──────────────────────────────────────────────────────────────────


class TestBankStatementSchema:
    """Tests for the Bank Statement schema."""

    def test_schema_identity(self):
        assert BANK_STATEMENT_SCHEMA.name == "bank_statement"
        assert BANK_STATEMENT_SCHEMA.display_name == "Bank Statement"
        assert BANK_STATEMENT_SCHEMA.document_type == DocumentType.BANK_STATEMENT

    def test_field_groups_combined(self):
        expected_count = (
            len(BANK_FIELDS)
            + len(ACCOUNT_HOLDER_FIELDS)
            + len(ACCOUNT_INFO_FIELDS)
            + len(BALANCE_FIELDS)
            + len(TRANSACTION_SUMMARY_FIELDS)
            + len(INTEREST_FIELDS)
            + len(FEE_FIELDS)
        )
        assert len(BANK_STATEMENT_SCHEMA.fields) == expected_count

    def test_bank_fields(self):
        assert len(BANK_FIELDS) == 4
        names = {f.name for f in BANK_FIELDS}
        assert "bank_name" in names
        assert "bank_routing_number" in names

    def test_account_holder_fields(self):
        assert len(ACCOUNT_HOLDER_FIELDS) == 2

    def test_account_info_fields(self):
        assert len(ACCOUNT_INFO_FIELDS) == 5
        names = {f.name for f in ACCOUNT_INFO_FIELDS}
        assert "account_number" in names
        assert "account_type" in names
        assert "statement_period_start" in names
        assert "statement_period_end" in names

    def test_balance_fields(self):
        assert len(BALANCE_FIELDS) == 4
        names = {f.name for f in BALANCE_FIELDS}
        assert "beginning_balance" in names
        assert "ending_balance" in names

    def test_transaction_summary_fields(self):
        assert len(TRANSACTION_SUMMARY_FIELDS) == 7
        names = {f.name for f in TRANSACTION_SUMMARY_FIELDS}
        assert "total_deposits" in names
        assert "total_withdrawals" in names
        assert "total_fees" in names

    def test_interest_fields(self):
        assert len(INTEREST_FIELDS) == 3
        names = {f.name for f in INTEREST_FIELDS}
        assert "interest_rate" in names
        assert "annual_percentage_yield" in names

    def test_fee_fields(self):
        assert len(FEE_FIELDS) == 4

    def test_required_fields(self):
        required = BANK_STATEMENT_SCHEMA.get_required_fields()
        required_names = {f.name for f in required}
        assert "bank_name" in required_names
        assert "account_holder_name" in required_names
        assert "account_number" in required_names
        assert "statement_period_start" in required_names
        assert "statement_period_end" in required_names
        assert "beginning_balance" in required_names
        assert "ending_balance" in required_names

    def test_routing_number_field(self):
        field = BANK_STATEMENT_SCHEMA.get_field("bank_routing_number")
        assert field is not None
        assert field.field_type == FieldType.ROUTING_NUMBER
        assert field.pattern == r"^\d{9}$"

    def test_account_number_field(self):
        field = BANK_STATEMENT_SCHEMA.get_field("account_number")
        assert field is not None
        assert field.field_type == FieldType.BANK_ACCOUNT

    def test_account_type_allowed_values(self):
        field = BANK_STATEMENT_SCHEMA.get_field("account_type")
        assert field is not None
        assert "Checking" in field.allowed_values
        assert "Savings" in field.allowed_values
        assert "Money Market" in field.allowed_values

    def test_integer_fields_have_min_value(self):
        for f in TRANSACTION_SUMMARY_FIELDS:
            if f.field_type == FieldType.INTEGER:
                assert f.min_value == 0, f"{f.name} should have min_value=0"

    def test_cross_field_rules(self):
        assert len(BANK_STATEMENT_CROSS_FIELD_RULES) == 2
        rule_pairs = [
            (r.source_field, r.target_field)
            for r in BANK_STATEMENT_CROSS_FIELD_RULES
        ]
        assert ("statement_period_start", "statement_period_end") in rule_pairs
        assert ("total_fees", "total_withdrawals") in rule_pairs

    def test_date_before_rule(self):
        date_rule = next(
            r
            for r in BANK_STATEMENT_CROSS_FIELD_RULES
            if r.source_field == "statement_period_start"
        )
        assert date_rule.operator == RuleOperator.DATE_BEFORE

    def test_classification_hints(self):
        hints = BANK_STATEMENT_SCHEMA.classification_hints
        assert "Bank Statement" in hints
        assert "Beginning Balance" in hints
        assert "Ending Balance" in hints

    def test_validate_valid_statement(self):
        result = {
            "bank_name": "Chase",
            "account_holder_name": "John Smith",
            "account_number": "****1234",
            "statement_period_start": "01/01/2025",
            "statement_period_end": "01/31/2025",
            "beginning_balance": "$5,000.00",
            "ending_balance": "$6,000.00",
        }
        errors, warnings = BANK_STATEMENT_SCHEMA.validate_result(result)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        result = {"bank_name": "Chase"}
        errors, _ = BANK_STATEMENT_SCHEMA.validate_result(result)
        assert len(errors) > 0

    def test_validate_date_ordering(self):
        result = {
            "bank_name": "Chase",
            "account_holder_name": "John Smith",
            "account_number": "****1234",
            "statement_period_start": "2025-03-15",
            "statement_period_end": "2025-01-01",
            "beginning_balance": "5000",
            "ending_balance": "6000",
        }
        errors, warnings = BANK_STATEMENT_SCHEMA.validate_result(result)
        assert any("before" in w.lower() for w in warnings)


# ──────────────────────────────────────────────────────────────────
# Schema Registry Integration
# ──────────────────────────────────────────────────────────────────


class TestSchemaRegistryIntegration:
    """Verify all finance schemas are auto-registered."""

    def test_invoice_registered(self):
        registry = SchemaRegistry()
        schema = registry.get("invoice")
        assert schema is not None
        assert schema.name == "invoice"

    def test_w2_registered(self):
        registry = SchemaRegistry()
        schema = registry.get("w2")
        assert schema is not None
        assert schema.name == "w2"

    def test_form_1099_registered(self):
        registry = SchemaRegistry()
        schema = registry.get("form_1099")
        assert schema is not None
        assert schema.name == "form_1099"

    def test_bank_statement_registered(self):
        registry = SchemaRegistry()
        schema = registry.get("bank_statement")
        assert schema is not None
        assert schema.name == "bank_statement"

    def test_lookup_by_document_type_invoice(self):
        registry = SchemaRegistry()
        schema = registry.get_by_document_type(DocumentType.INVOICE)
        assert schema is not None
        assert schema.name == "invoice"

    def test_lookup_by_document_type_w2(self):
        registry = SchemaRegistry()
        schema = registry.get_by_document_type(DocumentType.W2)
        assert schema is not None
        assert schema.name == "w2"

    def test_lookup_by_document_type_1099(self):
        registry = SchemaRegistry()
        schema = registry.get_by_document_type(DocumentType.FORM_1099)
        assert schema is not None
        assert schema.name == "form_1099"

    def test_lookup_by_document_type_bank_statement(self):
        registry = SchemaRegistry()
        schema = registry.get_by_document_type(DocumentType.BANK_STATEMENT)
        assert schema is not None
        assert schema.name == "bank_statement"


# ──────────────────────────────────────────────────────────────────
# Module Exports
# ──────────────────────────────────────────────────────────────────


class TestModuleExports:
    """Verify all finance schemas are accessible from the schemas package."""

    def test_invoice_import(self):
        from src.schemas import INVOICE_SCHEMA as imported

        assert imported is INVOICE_SCHEMA

    def test_w2_import(self):
        from src.schemas import W2_SCHEMA as imported

        assert imported is W2_SCHEMA

    def test_form_1099_import(self):
        from src.schemas import FORM_1099_SCHEMA as imported

        assert imported is FORM_1099_SCHEMA

    def test_bank_statement_import(self):
        from src.schemas import BANK_STATEMENT_SCHEMA as imported

        assert imported is BANK_STATEMENT_SCHEMA


# ──────────────────────────────────────────────────────────────────
# Cross-Schema Consistency
# ──────────────────────────────────────────────────────────────────


class TestCrossSchemaConsistency:
    """Verify consistency properties across all finance schemas."""

    SCHEMAS = [INVOICE_SCHEMA, W2_SCHEMA, FORM_1099_SCHEMA, BANK_STATEMENT_SCHEMA]

    def test_all_schemas_are_document_schema(self):
        for schema in self.SCHEMAS:
            assert isinstance(schema, DocumentSchema)

    def test_all_schemas_have_unique_names(self):
        names = [s.name for s in self.SCHEMAS]
        assert len(names) == len(set(names))

    def test_all_schemas_have_unique_document_types(self):
        types = [s.document_type for s in self.SCHEMAS]
        assert len(types) == len(set(types))

    def test_all_schemas_have_classification_hints(self):
        for schema in self.SCHEMAS:
            assert len(schema.classification_hints) >= 3, (
                f"{schema.name} should have at least 3 classification hints"
            )

    def test_all_schemas_have_required_sections(self):
        for schema in self.SCHEMAS:
            assert len(schema.required_sections) >= 2, (
                f"{schema.name} should have at least 2 required sections"
            )

    def test_all_schemas_have_required_fields(self):
        for schema in self.SCHEMAS:
            required = schema.get_required_fields()
            assert len(required) >= 2, (
                f"{schema.name} should have at least 2 required fields"
            )

    def test_all_field_names_are_snake_case(self):
        import re

        for schema in self.SCHEMAS:
            for field in schema.fields:
                assert re.match(r"^[a-z][a-z0-9_]*$", field.name), (
                    f"{schema.name}.{field.name} is not snake_case"
                )

    def test_all_schemas_generate_prompts(self):
        for schema in self.SCHEMAS:
            prompt = schema.generate_extraction_prompt()
            assert len(prompt) > 100
            assert "REQUIRED" in prompt

    def test_all_schemas_to_dict(self):
        for schema in self.SCHEMAS:
            d = schema.to_dict()
            assert d["name"] == schema.name
            assert "fields" in d
            assert len(d["fields"]) == len(schema.fields)

    def test_no_duplicate_field_names_within_schema(self):
        for schema in self.SCHEMAS:
            names = [f.name for f in schema.fields]
            assert len(names) == len(set(names)), (
                f"{schema.name} has duplicate field names"
            )
