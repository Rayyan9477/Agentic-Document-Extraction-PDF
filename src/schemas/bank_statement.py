"""
Bank Statement schema for document extraction.

Defines the complete schema for extracting data from bank statements,
including account information, transaction summaries, and balances.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# Bank / Institution Fields
# =============================================================================

BANK_FIELDS = [
    FieldDefinition(
        name="bank_name",
        display_name="Bank / Institution Name",
        field_type=FieldType.STRING,
        description="Name of the bank or financial institution",
        required=True,
        location_hint="Top of document - header/logo area",
        examples=["Chase", "Bank of America", "Wells Fargo"],
    ),
    FieldDefinition(
        name="bank_address",
        display_name="Bank Address",
        field_type=FieldType.ADDRESS,
        description="Bank branch or mailing address",
        required=False,
        location_hint="Header area - below bank name",
    ),
    FieldDefinition(
        name="bank_phone",
        display_name="Bank Phone",
        field_type=FieldType.PHONE,
        description="Bank customer service phone number",
        required=False,
        location_hint="Header or footer area",
    ),
    FieldDefinition(
        name="bank_routing_number",
        display_name="Bank Routing Number",
        field_type=FieldType.ROUTING_NUMBER,
        description="ABA routing / transit number",
        required=False,
        pattern=r"^\d{9}$",
        examples=["021000021", "121042882"],
    ),
]


# =============================================================================
# Account Holder Fields
# =============================================================================

ACCOUNT_HOLDER_FIELDS = [
    FieldDefinition(
        name="account_holder_name",
        display_name="Account Holder Name",
        field_type=FieldType.NAME,
        description="Name of the account holder",
        required=True,
        location_hint="Upper portion - account holder section",
        examples=["John A. Smith", "Smith Family Trust"],
    ),
    FieldDefinition(
        name="account_holder_address",
        display_name="Account Holder Address",
        field_type=FieldType.ADDRESS,
        description="Account holder mailing address",
        required=False,
        location_hint="Below account holder name",
    ),
]


# =============================================================================
# Account Information Fields
# =============================================================================

ACCOUNT_INFO_FIELDS = [
    FieldDefinition(
        name="account_number",
        display_name="Account Number",
        field_type=FieldType.BANK_ACCOUNT,
        description="Bank account number (may be partially masked)",
        required=True,
        location_hint="Account information section",
        examples=["****1234", "123456789012"],
    ),
    FieldDefinition(
        name="account_type",
        display_name="Account Type",
        field_type=FieldType.STRING,
        description="Type of bank account",
        required=False,
        location_hint="Near account number",
        allowed_values=[
            "Checking",
            "Savings",
            "Money Market",
            "CD",
            "Business Checking",
            "Business Savings",
        ],
    ),
    FieldDefinition(
        name="statement_period_start",
        display_name="Statement Period Start",
        field_type=FieldType.DATE,
        description="Start date of the statement period",
        required=True,
        location_hint="Statement period section",
        examples=["01/01/2025", "2025-01-01", "January 1, 2025"],
    ),
    FieldDefinition(
        name="statement_period_end",
        display_name="Statement Period End",
        field_type=FieldType.DATE,
        description="End date of the statement period",
        required=True,
        location_hint="Statement period section",
        examples=["01/31/2025", "2025-01-31", "January 31, 2025"],
    ),
    FieldDefinition(
        name="currency",
        display_name="Currency",
        field_type=FieldType.STRING,
        description="Currency of the account",
        required=False,
        examples=["USD", "EUR", "GBP"],
        allowed_values=["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"],
    ),
]


# =============================================================================
# Balance Fields
# =============================================================================

BALANCE_FIELDS = [
    FieldDefinition(
        name="beginning_balance",
        display_name="Beginning Balance",
        field_type=FieldType.CURRENCY,
        description="Account balance at the start of the statement period",
        required=True,
        location_hint="Account summary section - beginning/opening balance",
        examples=["$5,432.10", "5432.10"],
    ),
    FieldDefinition(
        name="ending_balance",
        display_name="Ending Balance",
        field_type=FieldType.CURRENCY,
        description="Account balance at the end of the statement period",
        required=True,
        location_hint="Account summary section - ending/closing balance",
        examples=["$6,789.45", "6789.45"],
    ),
    FieldDefinition(
        name="average_daily_balance",
        display_name="Average Daily Balance",
        field_type=FieldType.CURRENCY,
        description="Average daily balance for the statement period",
        required=False,
        location_hint="Account summary or interest section",
    ),
    FieldDefinition(
        name="minimum_balance",
        display_name="Minimum Balance",
        field_type=FieldType.CURRENCY,
        description="Minimum balance during the statement period",
        required=False,
    ),
]


# =============================================================================
# Transaction Summary Fields
# =============================================================================

TRANSACTION_SUMMARY_FIELDS = [
    FieldDefinition(
        name="total_deposits",
        display_name="Total Deposits / Credits",
        field_type=FieldType.CURRENCY,
        description="Total deposits and credits for the period",
        required=False,
        location_hint="Account summary section",
        examples=["$12,500.00", "12500.00"],
    ),
    FieldDefinition(
        name="total_withdrawals",
        display_name="Total Withdrawals / Debits",
        field_type=FieldType.CURRENCY,
        description="Total withdrawals and debits for the period",
        required=False,
        location_hint="Account summary section",
        examples=["$11,142.65", "11142.65"],
    ),
    FieldDefinition(
        name="total_fees",
        display_name="Total Fees",
        field_type=FieldType.CURRENCY,
        description="Total fees charged during the period",
        required=False,
        location_hint="Fee summary section",
        examples=["$12.00", "0.00"],
    ),
    FieldDefinition(
        name="total_interest_earned",
        display_name="Total Interest Earned",
        field_type=FieldType.CURRENCY,
        description="Total interest earned during the period",
        required=False,
        location_hint="Interest section",
        examples=["$2.35", "0.00"],
    ),
    FieldDefinition(
        name="number_of_deposits",
        display_name="Number of Deposits",
        field_type=FieldType.INTEGER,
        description="Count of deposits during the period",
        required=False,
        min_value=0,
    ),
    FieldDefinition(
        name="number_of_withdrawals",
        display_name="Number of Withdrawals",
        field_type=FieldType.INTEGER,
        description="Count of withdrawals during the period",
        required=False,
        min_value=0,
    ),
    FieldDefinition(
        name="number_of_checks",
        display_name="Number of Checks",
        field_type=FieldType.INTEGER,
        description="Count of checks cleared during the period",
        required=False,
        min_value=0,
    ),
]


# =============================================================================
# Interest and APY Fields
# =============================================================================

INTEREST_FIELDS = [
    FieldDefinition(
        name="interest_rate",
        display_name="Interest Rate / APR",
        field_type=FieldType.PERCENTAGE,
        description="Annual percentage rate on the account",
        required=False,
        location_hint="Interest section",
        examples=["0.50%", "4.25%"],
    ),
    FieldDefinition(
        name="annual_percentage_yield",
        display_name="Annual Percentage Yield (APY)",
        field_type=FieldType.PERCENTAGE,
        description="Annual percentage yield",
        required=False,
        location_hint="Interest section",
        examples=["0.50%", "4.30%"],
    ),
    FieldDefinition(
        name="interest_ytd",
        display_name="Interest Earned YTD",
        field_type=FieldType.CURRENCY,
        description="Year-to-date interest earned",
        required=False,
        location_hint="Interest summary section",
    ),
]


# =============================================================================
# Overdraft / Fee Fields
# =============================================================================

FEE_FIELDS = [
    FieldDefinition(
        name="overdraft_protection",
        display_name="Overdraft Protection",
        field_type=FieldType.BOOLEAN,
        description="Whether overdraft protection is active",
        required=False,
    ),
    FieldDefinition(
        name="overdraft_limit",
        display_name="Overdraft Limit",
        field_type=FieldType.CURRENCY,
        description="Overdraft protection limit amount",
        required=False,
    ),
    FieldDefinition(
        name="monthly_service_fee",
        display_name="Monthly Service Fee",
        field_type=FieldType.CURRENCY,
        description="Monthly account service fee",
        required=False,
        examples=["$12.00", "0.00"],
    ),
    FieldDefinition(
        name="fees_ytd",
        display_name="Fees YTD",
        field_type=FieldType.CURRENCY,
        description="Year-to-date fees charged",
        required=False,
    ),
]


# =============================================================================
# Cross-Field Rules
# =============================================================================

BANK_STATEMENT_CROSS_FIELD_RULES = [
    CrossFieldRule(
        source_field="statement_period_start",
        target_field="statement_period_end",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Statement period start must be before end date",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="total_fees",
        target_field="total_withdrawals",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Total fees should not exceed total withdrawals",
        severity="warning",
    ),
]


# =============================================================================
# Bank Statement Schema
# =============================================================================

BANK_STATEMENT_SCHEMA = DocumentSchema(
    name="bank_statement",
    display_name="Bank Statement",
    document_type=DocumentType.BANK_STATEMENT,
    description="Monthly or periodic bank account statement",
    version="1.0.0",
    fields=(
        BANK_FIELDS
        + ACCOUNT_HOLDER_FIELDS
        + ACCOUNT_INFO_FIELDS
        + BALANCE_FIELDS
        + TRANSACTION_SUMMARY_FIELDS
        + INTEREST_FIELDS
        + FEE_FIELDS
    ),
    cross_field_rules=BANK_STATEMENT_CROSS_FIELD_RULES,
    required_sections=["bank_info", "account_info", "balances"],
    classification_hints=[
        "Bank Statement",
        "Account Statement",
        "Statement Period",
        "Beginning Balance",
        "Ending Balance",
        "Deposits and Credits",
        "Withdrawals and Debits",
        "Account Summary",
        "Checking Account",
        "Savings Account",
    ],
)

# Auto-register
SchemaRegistry().register(BANK_STATEMENT_SCHEMA)
