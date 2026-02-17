"""
W-2 Wage and Tax Statement schema for document extraction.

Defines the complete schema for extracting data from IRS Form W-2,
including employer info, employee info, wages, and tax withholdings.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# Employer Fields
# =============================================================================

EMPLOYER_FIELDS = [
    FieldDefinition(
        name="employer_name",
        display_name="Employer Name",
        field_type=FieldType.STRING,
        description="Name of the employer",
        required=True,
        location_hint="Box c - Employer's name, address, and ZIP code",
        examples=["Acme Corporation", "Smith Manufacturing Inc"],
    ),
    FieldDefinition(
        name="employer_ein",
        display_name="Employer EIN",
        field_type=FieldType.EIN,
        description="Employer Identification Number",
        required=True,
        location_hint="Box b - Employer identification number (EIN)",
        pattern=r"^\d{2}-?\d{7}$",
        examples=["12-3456789", "123456789"],
    ),
    FieldDefinition(
        name="employer_address",
        display_name="Employer Address",
        field_type=FieldType.ADDRESS,
        description="Employer mailing address",
        required=False,
        location_hint="Box c - below employer name",
    ),
    FieldDefinition(
        name="employer_state_id",
        display_name="Employer State ID",
        field_type=FieldType.STRING,
        description="Employer state identification number",
        required=False,
        location_hint="Box 15 - Employer's state ID number",
    ),
]


# =============================================================================
# Employee Fields
# =============================================================================

EMPLOYEE_FIELDS = [
    FieldDefinition(
        name="employee_ssn",
        display_name="Employee SSN",
        field_type=FieldType.SSN,
        description="Employee Social Security Number",
        required=True,
        location_hint="Box a - Employee's social security number",
        pattern=r"^\d{3}-?\d{2}-?\d{4}$",
        examples=["123-45-6789", "123456789"],
    ),
    FieldDefinition(
        name="employee_first_name",
        display_name="Employee First Name",
        field_type=FieldType.NAME,
        description="Employee first name",
        required=True,
        location_hint="Box e - Employee's first name and initial",
    ),
    FieldDefinition(
        name="employee_last_name",
        display_name="Employee Last Name",
        field_type=FieldType.NAME,
        description="Employee last name",
        required=True,
        location_hint="Box e - Employee's last name",
    ),
    FieldDefinition(
        name="employee_suffix",
        display_name="Employee Suffix",
        field_type=FieldType.STRING,
        description="Employee name suffix (Jr., Sr., III, etc.)",
        required=False,
        location_hint="Box e - Suff.",
        allowed_values=["Jr.", "Sr.", "I", "II", "III", "IV"],
    ),
    FieldDefinition(
        name="employee_address",
        display_name="Employee Address",
        field_type=FieldType.ADDRESS,
        description="Employee mailing address",
        required=False,
        location_hint="Box f - Employee's address and ZIP code",
    ),
]


# =============================================================================
# Wage and Compensation Fields (Boxes 1-8)
# =============================================================================

WAGE_FIELDS = [
    FieldDefinition(
        name="wages_tips_other",
        display_name="Wages, Tips, Other Compensation",
        field_type=FieldType.CURRENCY,
        description="Total wages, tips, and other compensation",
        required=True,
        location_hint="Box 1",
        examples=["$75,000.00", "75000.00"],
    ),
    FieldDefinition(
        name="federal_income_tax",
        display_name="Federal Income Tax Withheld",
        field_type=FieldType.CURRENCY,
        description="Total federal income tax withheld",
        required=True,
        location_hint="Box 2",
        examples=["$12,500.00", "12500.00"],
    ),
    FieldDefinition(
        name="social_security_wages",
        display_name="Social Security Wages",
        field_type=FieldType.CURRENCY,
        description="Total Social Security wages",
        required=False,
        location_hint="Box 3",
        examples=["$75,000.00", "75000.00"],
    ),
    FieldDefinition(
        name="social_security_tax",
        display_name="Social Security Tax Withheld",
        field_type=FieldType.CURRENCY,
        description="Total Social Security tax withheld",
        required=False,
        location_hint="Box 4",
        examples=["$4,650.00", "4650.00"],
    ),
    FieldDefinition(
        name="medicare_wages",
        display_name="Medicare Wages and Tips",
        field_type=FieldType.CURRENCY,
        description="Total Medicare wages and tips",
        required=False,
        location_hint="Box 5",
        examples=["$75,000.00", "75000.00"],
    ),
    FieldDefinition(
        name="medicare_tax",
        display_name="Medicare Tax Withheld",
        field_type=FieldType.CURRENCY,
        description="Total Medicare tax withheld",
        required=False,
        location_hint="Box 6",
        examples=["$1,087.50", "1087.50"],
    ),
    FieldDefinition(
        name="social_security_tips",
        display_name="Social Security Tips",
        field_type=FieldType.CURRENCY,
        description="Total Social Security tips",
        required=False,
        location_hint="Box 7",
    ),
    FieldDefinition(
        name="allocated_tips",
        display_name="Allocated Tips",
        field_type=FieldType.CURRENCY,
        description="Total allocated tips",
        required=False,
        location_hint="Box 8",
    ),
]


# =============================================================================
# Additional Tax Fields (Boxes 9-14)
# =============================================================================

ADDITIONAL_TAX_FIELDS = [
    FieldDefinition(
        name="dependent_care_benefits",
        display_name="Dependent Care Benefits",
        field_type=FieldType.CURRENCY,
        description="Dependent care benefits",
        required=False,
        location_hint="Box 10",
    ),
    FieldDefinition(
        name="nonqualified_plans",
        display_name="Nonqualified Plans",
        field_type=FieldType.CURRENCY,
        description="Distributions from nonqualified plans",
        required=False,
        location_hint="Box 11",
    ),
    FieldDefinition(
        name="box_12a_code",
        display_name="Box 12a Code",
        field_type=FieldType.STRING,
        description="Box 12a code letter (e.g., D for 401k, DD for health coverage)",
        required=False,
        location_hint="Box 12a - Code",
        examples=["D", "DD", "W", "E", "C"],
    ),
    FieldDefinition(
        name="box_12a_amount",
        display_name="Box 12a Amount",
        field_type=FieldType.CURRENCY,
        description="Box 12a amount",
        required=False,
        location_hint="Box 12a - Amount",
    ),
    FieldDefinition(
        name="box_12b_code",
        display_name="Box 12b Code",
        field_type=FieldType.STRING,
        description="Box 12b code letter",
        required=False,
        location_hint="Box 12b - Code",
    ),
    FieldDefinition(
        name="box_12b_amount",
        display_name="Box 12b Amount",
        field_type=FieldType.CURRENCY,
        description="Box 12b amount",
        required=False,
        location_hint="Box 12b - Amount",
    ),
    FieldDefinition(
        name="box_12c_code",
        display_name="Box 12c Code",
        field_type=FieldType.STRING,
        description="Box 12c code letter",
        required=False,
        location_hint="Box 12c - Code",
    ),
    FieldDefinition(
        name="box_12c_amount",
        display_name="Box 12c Amount",
        field_type=FieldType.CURRENCY,
        description="Box 12c amount",
        required=False,
        location_hint="Box 12c - Amount",
    ),
    FieldDefinition(
        name="box_12d_code",
        display_name="Box 12d Code",
        field_type=FieldType.STRING,
        description="Box 12d code letter",
        required=False,
        location_hint="Box 12d - Code",
    ),
    FieldDefinition(
        name="box_12d_amount",
        display_name="Box 12d Amount",
        field_type=FieldType.CURRENCY,
        description="Box 12d amount",
        required=False,
        location_hint="Box 12d - Amount",
    ),
    FieldDefinition(
        name="statutory_employee",
        display_name="Statutory Employee",
        field_type=FieldType.BOOLEAN,
        description="Statutory employee checkbox",
        required=False,
        location_hint="Box 13 - Statutory employee",
    ),
    FieldDefinition(
        name="retirement_plan",
        display_name="Retirement Plan",
        field_type=FieldType.BOOLEAN,
        description="Retirement plan checkbox",
        required=False,
        location_hint="Box 13 - Retirement plan",
    ),
    FieldDefinition(
        name="third_party_sick_pay",
        display_name="Third-party Sick Pay",
        field_type=FieldType.BOOLEAN,
        description="Third-party sick pay checkbox",
        required=False,
        location_hint="Box 13 - Third-party sick pay",
    ),
    FieldDefinition(
        name="other_box_14",
        display_name="Other (Box 14)",
        field_type=FieldType.STRING,
        description="Other items reported in Box 14",
        required=False,
        location_hint="Box 14 - Other",
    ),
]


# =============================================================================
# State and Local Tax Fields (Boxes 15-20)
# =============================================================================

STATE_LOCAL_FIELDS = [
    FieldDefinition(
        name="state",
        display_name="State",
        field_type=FieldType.STATE,
        description="State abbreviation",
        required=False,
        location_hint="Box 15 - State",
        examples=["CA", "NY", "TX"],
    ),
    FieldDefinition(
        name="state_wages",
        display_name="State Wages, Tips, etc.",
        field_type=FieldType.CURRENCY,
        description="State wages, tips, and other compensation",
        required=False,
        location_hint="Box 16",
    ),
    FieldDefinition(
        name="state_income_tax",
        display_name="State Income Tax",
        field_type=FieldType.CURRENCY,
        description="State income tax withheld",
        required=False,
        location_hint="Box 17",
    ),
    FieldDefinition(
        name="local_wages",
        display_name="Local Wages, Tips, etc.",
        field_type=FieldType.CURRENCY,
        description="Local wages, tips, and other compensation",
        required=False,
        location_hint="Box 18",
    ),
    FieldDefinition(
        name="local_income_tax",
        display_name="Local Income Tax",
        field_type=FieldType.CURRENCY,
        description="Local income tax withheld",
        required=False,
        location_hint="Box 19",
    ),
    FieldDefinition(
        name="locality_name",
        display_name="Locality Name",
        field_type=FieldType.STRING,
        description="Name of the locality",
        required=False,
        location_hint="Box 20",
        examples=["New York City", "Philadelphia"],
    ),
]


# =============================================================================
# Document Metadata
# =============================================================================

W2_META_FIELDS = [
    FieldDefinition(
        name="tax_year",
        display_name="Tax Year",
        field_type=FieldType.INTEGER,
        description="Tax year for the W-2",
        required=True,
        location_hint="Top of form - prominently displayed",
        examples=["2024", "2025"],
        min_value=2000,
        max_value=2030,
    ),
    FieldDefinition(
        name="control_number",
        display_name="Control Number",
        field_type=FieldType.STRING,
        description="Employer-assigned control number",
        required=False,
        location_hint="Box d - Control number",
    ),
    FieldDefinition(
        name="form_type",
        display_name="Form Type",
        field_type=FieldType.STRING,
        description="W-2 form variant",
        required=False,
        allowed_values=["W-2", "W-2C", "W-2G"],
    ),
]


# =============================================================================
# Cross-Field Rules
# =============================================================================

W2_CROSS_FIELD_RULES = [
    CrossFieldRule(
        source_field="social_security_tax",
        target_field="social_security_wages",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Social Security tax should not exceed Social Security wages",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="medicare_tax",
        target_field="medicare_wages",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Medicare tax should not exceed Medicare wages",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="federal_income_tax",
        target_field="wages_tips_other",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Federal income tax withheld should not exceed wages",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="state_income_tax",
        target_field="state_wages",
        operator=RuleOperator.LESS_EQUAL,
        error_message="State income tax should not exceed state wages",
        severity="warning",
    ),
]


# =============================================================================
# W-2 Schema
# =============================================================================

W2_SCHEMA = DocumentSchema(
    name="w2",
    display_name="W-2 Wage and Tax Statement",
    document_type=DocumentType.W2,
    description="IRS Form W-2 Wage and Tax Statement",
    version="1.0.0",
    fields=(
        EMPLOYER_FIELDS
        + EMPLOYEE_FIELDS
        + WAGE_FIELDS
        + ADDITIONAL_TAX_FIELDS
        + STATE_LOCAL_FIELDS
        + W2_META_FIELDS
    ),
    cross_field_rules=W2_CROSS_FIELD_RULES,
    required_sections=["employer_info", "employee_info", "wages_taxes"],
    classification_hints=[
        "W-2",
        "Wage and Tax Statement",
        "Form W-2",
        "Employer identification number",
        "Social security wages",
        "Medicare wages",
        "Federal income tax withheld",
        "Employee's social security number",
    ],
)

# Auto-register
SchemaRegistry().register(W2_SCHEMA)
