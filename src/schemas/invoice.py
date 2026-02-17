"""
Invoice schema for document extraction.

Defines the complete schema for extracting data from business invoices,
including vendor information, line items, totals, and payment terms.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# Vendor / Seller Fields
# =============================================================================

VENDOR_FIELDS = [
    FieldDefinition(
        name="vendor_name",
        display_name="Vendor / Seller Name",
        field_type=FieldType.STRING,
        description="Name of the company or individual issuing the invoice",
        required=True,
        location_hint="Top of document - header/letterhead",
        examples=["Acme Corporation", "Smith Consulting LLC"],
    ),
    FieldDefinition(
        name="vendor_address",
        display_name="Vendor Address",
        field_type=FieldType.ADDRESS,
        description="Mailing address of the vendor",
        required=False,
        location_hint="Header area - below vendor name",
    ),
    FieldDefinition(
        name="vendor_phone",
        display_name="Vendor Phone",
        field_type=FieldType.PHONE,
        description="Vendor contact phone number",
        required=False,
        location_hint="Header area",
    ),
    FieldDefinition(
        name="vendor_email",
        display_name="Vendor Email",
        field_type=FieldType.EMAIL,
        description="Vendor contact email address",
        required=False,
        location_hint="Header area",
    ),
    FieldDefinition(
        name="vendor_ein",
        display_name="Vendor EIN / Tax ID",
        field_type=FieldType.EIN,
        description="Employer Identification Number of the vendor",
        required=False,
        pattern=r"^\d{2}-?\d{7}$",
        examples=["12-3456789", "123456789"],
    ),
]


# =============================================================================
# Buyer / Bill-To Fields
# =============================================================================

BUYER_FIELDS = [
    FieldDefinition(
        name="buyer_name",
        display_name="Buyer / Bill-To Name",
        field_type=FieldType.STRING,
        description="Name of the buyer or bill-to party",
        required=True,
        location_hint="Upper portion - Bill To section",
        examples=["John Smith", "ABC Industries"],
    ),
    FieldDefinition(
        name="buyer_address",
        display_name="Buyer Address",
        field_type=FieldType.ADDRESS,
        description="Billing address of the buyer",
        required=False,
        location_hint="Bill To section",
    ),
    FieldDefinition(
        name="buyer_email",
        display_name="Buyer Email",
        field_type=FieldType.EMAIL,
        description="Buyer contact email",
        required=False,
    ),
]


# =============================================================================
# Invoice Identification Fields
# =============================================================================

INVOICE_ID_FIELDS = [
    FieldDefinition(
        name="invoice_number",
        display_name="Invoice Number",
        field_type=FieldType.STRING,
        description="Unique invoice identifier",
        required=True,
        location_hint="Upper right or prominent header position",
        examples=["INV-2025-0042", "10042", "A-12345"],
    ),
    FieldDefinition(
        name="invoice_date",
        display_name="Invoice Date",
        field_type=FieldType.DATE,
        description="Date the invoice was issued",
        required=True,
        location_hint="Near invoice number",
        examples=["01/15/2025", "2025-01-15", "January 15, 2025"],
    ),
    FieldDefinition(
        name="due_date",
        display_name="Due Date",
        field_type=FieldType.DATE,
        description="Payment due date",
        required=False,
        location_hint="Near invoice date or payment terms",
        examples=["02/14/2025", "2025-02-14"],
    ),
    FieldDefinition(
        name="purchase_order_number",
        display_name="Purchase Order Number",
        field_type=FieldType.STRING,
        description="Associated purchase order reference",
        required=False,
        location_hint="Near invoice number",
        examples=["PO-2025-001", "45678"],
    ),
    FieldDefinition(
        name="payment_terms",
        display_name="Payment Terms",
        field_type=FieldType.STRING,
        description="Payment terms (e.g., Net 30, Due on Receipt)",
        required=False,
        location_hint="Near due date or footer",
        examples=["Net 30", "Net 60", "Due on Receipt", "2/10 Net 30"],
    ),
]


# =============================================================================
# Financial Totals
# =============================================================================

TOTAL_FIELDS = [
    FieldDefinition(
        name="subtotal",
        display_name="Subtotal",
        field_type=FieldType.CURRENCY,
        description="Total before tax and discounts",
        required=False,
        location_hint="Bottom of line items section",
        examples=["$1,250.00", "1250.00"],
    ),
    FieldDefinition(
        name="tax_rate",
        display_name="Tax Rate",
        field_type=FieldType.PERCENTAGE,
        description="Sales tax rate applied",
        required=False,
        examples=["8.25%", "0%"],
    ),
    FieldDefinition(
        name="tax_amount",
        display_name="Tax Amount",
        field_type=FieldType.CURRENCY,
        description="Total tax amount",
        required=False,
        examples=["$103.13", "0.00"],
    ),
    FieldDefinition(
        name="discount_amount",
        display_name="Discount Amount",
        field_type=FieldType.CURRENCY,
        description="Total discount applied",
        required=False,
        examples=["$50.00", "0.00"],
    ),
    FieldDefinition(
        name="shipping_amount",
        display_name="Shipping / Freight",
        field_type=FieldType.CURRENCY,
        description="Shipping or freight charges",
        required=False,
        examples=["$15.00", "0.00"],
    ),
    FieldDefinition(
        name="total_amount",
        display_name="Total Amount Due",
        field_type=FieldType.CURRENCY,
        description="Grand total amount due",
        required=True,
        location_hint="Bottom of invoice - prominently displayed",
        examples=["$1,318.13", "1318.13"],
    ),
    FieldDefinition(
        name="amount_paid",
        display_name="Amount Paid",
        field_type=FieldType.CURRENCY,
        description="Amount already paid",
        required=False,
        examples=["$0.00", "500.00"],
    ),
    FieldDefinition(
        name="balance_due",
        display_name="Balance Due",
        field_type=FieldType.CURRENCY,
        description="Remaining balance after payments",
        required=False,
        location_hint="Bottom of invoice",
        examples=["$1,318.13", "818.13"],
    ),
    FieldDefinition(
        name="currency",
        display_name="Currency",
        field_type=FieldType.STRING,
        description="Currency of the invoice",
        required=False,
        examples=["USD", "EUR", "GBP"],
        allowed_values=["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"],
    ),
]


# =============================================================================
# Payment Information
# =============================================================================

PAYMENT_FIELDS = [
    FieldDefinition(
        name="payment_method",
        display_name="Payment Method",
        field_type=FieldType.STRING,
        description="Accepted payment method",
        required=False,
        location_hint="Footer or payment section",
        examples=["Check", "Wire Transfer", "ACH", "Credit Card"],
    ),
    FieldDefinition(
        name="bank_name",
        display_name="Bank Name",
        field_type=FieldType.STRING,
        description="Bank name for wire/ACH payment",
        required=False,
        location_hint="Payment details section",
    ),
    FieldDefinition(
        name="bank_routing_number",
        display_name="Bank Routing Number",
        field_type=FieldType.ROUTING_NUMBER,
        description="ABA routing number for wire/ACH",
        required=False,
        pattern=r"^\d{9}$",
        examples=["021000021", "121042882"],
    ),
    FieldDefinition(
        name="bank_account_number",
        display_name="Bank Account Number",
        field_type=FieldType.BANK_ACCOUNT,
        description="Bank account number for payment",
        required=False,
    ),
]


# =============================================================================
# Cross-Field Rules
# =============================================================================

INVOICE_CROSS_FIELD_RULES = [
    CrossFieldRule(
        source_field="invoice_date",
        target_field="due_date",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Invoice date must be before due date",
        severity="warning",
    ),
    CrossFieldRule(
        source_field="subtotal",
        target_field="total_amount",
        operator=RuleOperator.LESS_EQUAL,
        error_message="Subtotal should not exceed total amount",
        severity="warning",
    ),
]


# =============================================================================
# Invoice Schema
# =============================================================================

INVOICE_SCHEMA = DocumentSchema(
    name="invoice",
    display_name="Invoice",
    document_type=DocumentType.INVOICE,
    description="Business invoice for goods or services",
    version="1.0.0",
    fields=(
        VENDOR_FIELDS
        + BUYER_FIELDS
        + INVOICE_ID_FIELDS
        + TOTAL_FIELDS
        + PAYMENT_FIELDS
    ),
    cross_field_rules=INVOICE_CROSS_FIELD_RULES,
    required_sections=["vendor_info", "invoice_details", "totals"],
    classification_hints=[
        "INVOICE",
        "Invoice Number",
        "Bill To",
        "Amount Due",
        "Payment Terms",
        "Subtotal",
        "Tax",
        "Total",
        "Due Date",
    ],
)

# Auto-register
SchemaRegistry().register(INVOICE_SCHEMA)
