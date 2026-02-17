"""
Form 1099 schema for document extraction.

Defines schemas for common 1099 variants: 1099-NEC (nonemployee compensation),
1099-MISC (miscellaneous income), and 1099-INT (interest income).
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# Payer Fields (common across all 1099 variants)
# =============================================================================

PAYER_FIELDS = [
    FieldDefinition(
        name="payer_name",
        display_name="Payer Name",
        field_type=FieldType.STRING,
        description="Name of the payer issuing the 1099",
        required=True,
        location_hint="Top left - PAYER'S name, street address, city or town",
        examples=["ABC Corporation", "First National Bank"],
    ),
    FieldDefinition(
        name="payer_address",
        display_name="Payer Address",
        field_type=FieldType.ADDRESS,
        description="Payer mailing address",
        required=False,
        location_hint="Below payer name",
    ),
    FieldDefinition(
        name="payer_tin",
        display_name="Payer TIN",
        field_type=FieldType.EIN,
        description="Payer's Taxpayer Identification Number (EIN or SSN)",
        required=True,
        location_hint="PAYER'S TIN field",
        pattern=r"^\d{2}-?\d{7}$",
        examples=["12-3456789", "123456789"],
    ),
    FieldDefinition(
        name="payer_phone",
        display_name="Payer Phone",
        field_type=FieldType.PHONE,
        description="Payer telephone number",
        required=False,
        location_hint="Below payer address",
    ),
]


# =============================================================================
# Recipient Fields (common across all 1099 variants)
# =============================================================================

RECIPIENT_FIELDS = [
    FieldDefinition(
        name="recipient_name",
        display_name="Recipient Name",
        field_type=FieldType.STRING,
        description="Name of the recipient",
        required=True,
        location_hint="RECIPIENT'S name field",
        examples=["John Smith", "Smith Consulting LLC"],
    ),
    FieldDefinition(
        name="recipient_tin",
        display_name="Recipient TIN",
        field_type=FieldType.SSN,
        description="Recipient's Taxpayer Identification Number (SSN or EIN)",
        required=True,
        location_hint="RECIPIENT'S TIN field",
        pattern=r"^\d{2,3}-?\d{2}-?\d{4,7}$",
        examples=["123-45-6789", "12-3456789"],
    ),
    FieldDefinition(
        name="recipient_address",
        display_name="Recipient Address",
        field_type=FieldType.ADDRESS,
        description="Recipient street address",
        required=False,
        location_hint="Street address field below recipient name",
    ),
    FieldDefinition(
        name="recipient_city_state_zip",
        display_name="Recipient City, State, ZIP",
        field_type=FieldType.STRING,
        description="Recipient city, state, and ZIP code",
        required=False,
        location_hint="City or town, state or province, country, and ZIP",
    ),
    FieldDefinition(
        name="recipient_account_number",
        display_name="Account Number",
        field_type=FieldType.ACCOUNT_NUMBER,
        description="Account number used by the payer",
        required=False,
        location_hint="Account number field (if applicable)",
    ),
]


# =============================================================================
# 1099-NEC Fields (Nonemployee Compensation)
# =============================================================================

NEC_FIELDS = [
    FieldDefinition(
        name="nonemployee_compensation",
        display_name="Nonemployee Compensation",
        field_type=FieldType.CURRENCY,
        description="Total nonemployee compensation",
        required=False,
        location_hint="Box 1 - Nonemployee compensation",
        examples=["$50,000.00", "50000.00"],
    ),
    FieldDefinition(
        name="payer_direct_sales",
        display_name="Payer Direct Sales",
        field_type=FieldType.BOOLEAN,
        description="Direct sales of $5,000 or more checkbox",
        required=False,
        location_hint="Box 2 - Payer made direct sales",
    ),
    FieldDefinition(
        name="federal_tax_withheld_nec",
        display_name="Federal Income Tax Withheld (NEC)",
        field_type=FieldType.CURRENCY,
        description="Federal income tax withheld",
        required=False,
        location_hint="Box 4 - Federal income tax withheld",
    ),
]


# =============================================================================
# 1099-MISC Fields (Miscellaneous Income)
# =============================================================================

MISC_FIELDS = [
    FieldDefinition(
        name="rents",
        display_name="Rents",
        field_type=FieldType.CURRENCY,
        description="Rental income received",
        required=False,
        location_hint="Box 1 - Rents",
    ),
    FieldDefinition(
        name="royalties",
        display_name="Royalties",
        field_type=FieldType.CURRENCY,
        description="Royalties income",
        required=False,
        location_hint="Box 2 - Royalties",
    ),
    FieldDefinition(
        name="other_income",
        display_name="Other Income",
        field_type=FieldType.CURRENCY,
        description="Other income amounts",
        required=False,
        location_hint="Box 3 - Other income",
    ),
    FieldDefinition(
        name="fishing_boat_proceeds",
        display_name="Fishing Boat Proceeds",
        field_type=FieldType.CURRENCY,
        description="Fishing boat proceeds",
        required=False,
        location_hint="Box 5 - Fishing boat proceeds",
    ),
    FieldDefinition(
        name="medical_healthcare_payments",
        display_name="Medical and Health Care Payments",
        field_type=FieldType.CURRENCY,
        description="Medical and health care payments",
        required=False,
        location_hint="Box 6 - Medical and health care payments",
    ),
    FieldDefinition(
        name="substitute_payments",
        display_name="Substitute Payments",
        field_type=FieldType.CURRENCY,
        description="Substitute payments in lieu of dividends or interest",
        required=False,
        location_hint="Box 8",
    ),
    FieldDefinition(
        name="crop_insurance_proceeds",
        display_name="Crop Insurance Proceeds",
        field_type=FieldType.CURRENCY,
        description="Crop insurance proceeds",
        required=False,
        location_hint="Box 9 - Crop insurance proceeds",
    ),
    FieldDefinition(
        name="gross_proceeds_attorney",
        display_name="Gross Proceeds to Attorney",
        field_type=FieldType.CURRENCY,
        description="Gross proceeds paid to an attorney",
        required=False,
        location_hint="Box 10 - Gross proceeds paid to an attorney",
    ),
    FieldDefinition(
        name="excess_golden_parachute",
        display_name="Excess Golden Parachute Payments",
        field_type=FieldType.CURRENCY,
        description="Excess golden parachute payments",
        required=False,
        location_hint="Box 13",
    ),
    FieldDefinition(
        name="nonqualified_deferred_comp",
        display_name="Nonqualified Deferred Compensation",
        field_type=FieldType.CURRENCY,
        description="Nonqualified deferred compensation",
        required=False,
        location_hint="Box 14",
    ),
    FieldDefinition(
        name="federal_tax_withheld_misc",
        display_name="Federal Income Tax Withheld (MISC)",
        field_type=FieldType.CURRENCY,
        description="Federal income tax withheld",
        required=False,
        location_hint="Box 4 - Federal income tax withheld",
    ),
]


# =============================================================================
# 1099-INT Fields (Interest Income)
# =============================================================================

INT_FIELDS = [
    FieldDefinition(
        name="interest_income",
        display_name="Interest Income",
        field_type=FieldType.CURRENCY,
        description="Total interest income",
        required=False,
        location_hint="Box 1 - Interest income",
        examples=["$1,250.00", "1250.00"],
    ),
    FieldDefinition(
        name="early_withdrawal_penalty",
        display_name="Early Withdrawal Penalty",
        field_type=FieldType.CURRENCY,
        description="Early withdrawal penalty",
        required=False,
        location_hint="Box 2 - Early withdrawal penalty",
    ),
    FieldDefinition(
        name="interest_on_us_savings_bonds",
        display_name="Interest on U.S. Savings Bonds",
        field_type=FieldType.CURRENCY,
        description="Interest on U.S. Savings Bonds and Treasury obligations",
        required=False,
        location_hint="Box 3",
    ),
    FieldDefinition(
        name="federal_tax_withheld_int",
        display_name="Federal Income Tax Withheld (INT)",
        field_type=FieldType.CURRENCY,
        description="Federal income tax withheld",
        required=False,
        location_hint="Box 4 - Federal income tax withheld",
    ),
    FieldDefinition(
        name="investment_expenses",
        display_name="Investment Expenses",
        field_type=FieldType.CURRENCY,
        description="Investment expenses",
        required=False,
        location_hint="Box 5",
    ),
    FieldDefinition(
        name="foreign_tax_paid",
        display_name="Foreign Tax Paid",
        field_type=FieldType.CURRENCY,
        description="Foreign tax paid",
        required=False,
        location_hint="Box 6",
    ),
    FieldDefinition(
        name="foreign_country",
        display_name="Foreign Country",
        field_type=FieldType.STRING,
        description="Foreign country or U.S. possession",
        required=False,
        location_hint="Box 7",
    ),
    FieldDefinition(
        name="tax_exempt_interest",
        display_name="Tax-Exempt Interest",
        field_type=FieldType.CURRENCY,
        description="Tax-exempt interest",
        required=False,
        location_hint="Box 8",
    ),
    FieldDefinition(
        name="private_activity_bond_interest",
        display_name="Private Activity Bond Interest",
        field_type=FieldType.CURRENCY,
        description="Specified private activity bond interest",
        required=False,
        location_hint="Box 9",
    ),
    FieldDefinition(
        name="market_discount",
        display_name="Market Discount",
        field_type=FieldType.CURRENCY,
        description="Market discount",
        required=False,
        location_hint="Box 10",
    ),
    FieldDefinition(
        name="bond_premium",
        display_name="Bond Premium",
        field_type=FieldType.CURRENCY,
        description="Bond premium",
        required=False,
        location_hint="Box 11",
    ),
]


# =============================================================================
# Common Metadata Fields
# =============================================================================

FORM_1099_META_FIELDS = [
    FieldDefinition(
        name="tax_year",
        display_name="Tax Year",
        field_type=FieldType.INTEGER,
        description="Tax year for the 1099",
        required=True,
        location_hint="Top of form - prominently displayed",
        examples=["2024", "2025"],
        min_value=2000,
        max_value=2030,
    ),
    FieldDefinition(
        name="form_variant",
        display_name="1099 Form Variant",
        field_type=FieldType.STRING,
        description="Which 1099 variant this is",
        required=False,
        location_hint="Form title area",
        allowed_values=[
            "1099-NEC",
            "1099-MISC",
            "1099-INT",
            "1099-DIV",
            "1099-R",
            "1099-G",
            "1099-K",
            "1099-S",
            "1099-B",
        ],
    ),
    FieldDefinition(
        name="corrected",
        display_name="Corrected",
        field_type=FieldType.BOOLEAN,
        description="Whether this is a corrected form",
        required=False,
        location_hint="CORRECTED checkbox at top",
    ),
    FieldDefinition(
        name="fatca_filing",
        display_name="FATCA Filing Requirement",
        field_type=FieldType.BOOLEAN,
        description="FATCA filing requirement checkbox",
        required=False,
        location_hint="FATCA filing requirement checkbox",
    ),
]


# =============================================================================
# State Tax Fields
# =============================================================================

FORM_1099_STATE_FIELDS = [
    FieldDefinition(
        name="state_1",
        display_name="State 1",
        field_type=FieldType.STATE,
        description="First state abbreviation",
        required=False,
        location_hint="State section - first entry",
    ),
    FieldDefinition(
        name="state_id_1",
        display_name="State Payer's ID 1",
        field_type=FieldType.STRING,
        description="First state payer identification number",
        required=False,
        location_hint="Payer's state no. - first entry",
    ),
    FieldDefinition(
        name="state_income_1",
        display_name="State Income 1",
        field_type=FieldType.CURRENCY,
        description="First state income amount",
        required=False,
        location_hint="State income - first entry",
    ),
    FieldDefinition(
        name="state_tax_withheld_1",
        display_name="State Tax Withheld 1",
        field_type=FieldType.CURRENCY,
        description="First state tax withheld",
        required=False,
        location_hint="State tax withheld - first entry",
    ),
    FieldDefinition(
        name="state_2",
        display_name="State 2",
        field_type=FieldType.STATE,
        description="Second state abbreviation",
        required=False,
        location_hint="State section - second entry",
    ),
    FieldDefinition(
        name="state_id_2",
        display_name="State Payer's ID 2",
        field_type=FieldType.STRING,
        description="Second state payer identification number",
        required=False,
        location_hint="Payer's state no. - second entry",
    ),
    FieldDefinition(
        name="state_income_2",
        display_name="State Income 2",
        field_type=FieldType.CURRENCY,
        description="Second state income amount",
        required=False,
        location_hint="State income - second entry",
    ),
    FieldDefinition(
        name="state_tax_withheld_2",
        display_name="State Tax Withheld 2",
        field_type=FieldType.CURRENCY,
        description="Second state tax withheld",
        required=False,
        location_hint="State tax withheld - second entry",
    ),
]


# =============================================================================
# Cross-Field Rules
# =============================================================================

FORM_1099_CROSS_FIELD_RULES = [
    CrossFieldRule(
        source_field="federal_tax_withheld_nec",
        target_field="nonemployee_compensation",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Federal tax withheld should not exceed nonemployee compensation",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="state_tax_withheld_1",
        target_field="state_income_1",
        operator=RuleOperator.LESS_EQUAL,
        error_message="State tax withheld should not exceed state income",
        severity="warning",
    ),
]


# =============================================================================
# Form 1099 Schema (combined — covers NEC, MISC, INT variants)
# =============================================================================

FORM_1099_SCHEMA = DocumentSchema(
    name="form_1099",
    display_name="Form 1099",
    document_type=DocumentType.FORM_1099,
    description="IRS Form 1099 series — NEC, MISC, INT, and other variants",
    version="1.0.0",
    fields=(
        PAYER_FIELDS
        + RECIPIENT_FIELDS
        + NEC_FIELDS
        + MISC_FIELDS
        + INT_FIELDS
        + FORM_1099_META_FIELDS
        + FORM_1099_STATE_FIELDS
    ),
    cross_field_rules=FORM_1099_CROSS_FIELD_RULES,
    required_sections=["payer_info", "recipient_info", "income"],
    classification_hints=[
        "1099",
        "Form 1099",
        "1099-NEC",
        "1099-MISC",
        "1099-INT",
        "Nonemployee Compensation",
        "PAYER'S TIN",
        "RECIPIENT'S TIN",
        "Miscellaneous Income",
        "Interest Income",
    ],
)

# Auto-register
SchemaRegistry().register(FORM_1099_SCHEMA)
