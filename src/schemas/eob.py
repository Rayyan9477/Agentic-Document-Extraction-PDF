"""
Explanation of Benefits (EOB) Schema.

Defines the complete schema for extracting data from insurance EOB documents,
which are statements explaining how medical claims were processed and paid.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# EOB Field Definitions
# =============================================================================

# Insurance Company Information
INSURER_FIELDS = [
    FieldDefinition(
        name="insurance_company_name",
        display_name="Insurance Company Name",
        field_type=FieldType.STRING,
        description="Name of the insurance company issuing the EOB",
        required=True,
        location_hint="Top of document - header/letterhead",
        examples=["Blue Cross Blue Shield", "Aetna", "United Healthcare", "Cigna"],
    ),
    FieldDefinition(
        name="insurance_company_address",
        display_name="Insurance Company Address",
        field_type=FieldType.ADDRESS,
        description="Mailing address of the insurance company",
        required=False,
        location_hint="Header area - below company name",
    ),
    FieldDefinition(
        name="insurance_company_phone",
        display_name="Insurance Company Phone",
        field_type=FieldType.PHONE,
        description="Customer service phone number",
        required=False,
        location_hint="Header or footer area",
    ),
    FieldDefinition(
        name="insurance_company_website",
        display_name="Insurance Company Website",
        field_type=FieldType.STRING,
        description="Insurance company website URL",
        required=False,
        location_hint="Header or footer area",
        pattern=r"^(https?://)?[\w\-]+(\.[\w\-]+)+(/[\w\-._~:/?#\[\]@!$&'()*+,;=]*)?$",
    ),
    FieldDefinition(
        name="plan_name",
        display_name="Plan Name",
        field_type=FieldType.STRING,
        description="Name of the insurance plan",
        required=False,
        location_hint="Near member information",
        examples=["PPO Gold", "HMO Basic", "High Deductible Health Plan"],
    ),
]

# Document Identification
DOCUMENT_FIELDS = [
    FieldDefinition(
        name="eob_date",
        display_name="EOB Date",
        field_type=FieldType.DATE,
        description="Date the EOB was issued",
        required=True,
        location_hint="Top of document - near header",
    ),
    FieldDefinition(
        name="claim_number",
        display_name="Claim Number",
        field_type=FieldType.STRING,
        description="Unique claim identifier assigned by insurance",
        required=True,
        location_hint="Prominent location - header or claim section",
        pattern=r"^[A-Za-z0-9\-]{5,30}$",
    ),
    FieldDefinition(
        name="eob_reference_number",
        display_name="EOB Reference Number",
        field_type=FieldType.STRING,
        description="EOB document reference number",
        required=False,
        location_hint="Header area",
    ),
    FieldDefinition(
        name="page_number",
        display_name="Page Number",
        field_type=FieldType.STRING,
        description="Current page of EOB (e.g., 'Page 1 of 3')",
        required=False,
        location_hint="Header or footer",
        pattern=r"^(Page\s*)?\d+\s*(of|/)\s*\d+$",
    ),
]

# Member/Patient Information
MEMBER_FIELDS = [
    FieldDefinition(
        name="member_name",
        display_name="Member Name",
        field_type=FieldType.NAME,
        description="Name of the insurance plan member (subscriber)",
        required=True,
        location_hint="Member information section",
        examples=["SMITH, JOHN A", "DOE, JANE"],
    ),
    FieldDefinition(
        name="member_id",
        display_name="Member ID",
        field_type=FieldType.MEMBER_ID,
        description="Member's insurance ID number",
        required=True,
        location_hint="Member information section",
    ),
    FieldDefinition(
        name="member_address",
        display_name="Member Address",
        field_type=FieldType.ADDRESS,
        description="Member's mailing address",
        required=False,
        location_hint="Member information section or mailing address area",
    ),
    FieldDefinition(
        name="member_city",
        display_name="Member City",
        field_type=FieldType.STRING,
        description="Member's city",
        required=False,
        location_hint="Member address line",
    ),
    FieldDefinition(
        name="member_state",
        display_name="Member State",
        field_type=FieldType.STATE,
        description="Member's state",
        required=False,
        location_hint="Member address line",
    ),
    FieldDefinition(
        name="member_zip",
        display_name="Member ZIP Code",
        field_type=FieldType.ZIP_CODE,
        description="Member's ZIP code",
        required=False,
        location_hint="Member address line",
    ),
    FieldDefinition(
        name="group_number",
        display_name="Group Number",
        field_type=FieldType.GROUP_NUMBER,
        description="Employer group number",
        required=False,
        location_hint="Member information section",
    ),
    FieldDefinition(
        name="group_name",
        display_name="Group Name",
        field_type=FieldType.STRING,
        description="Employer or group name",
        required=False,
        location_hint="Member information section",
    ),
]

# Patient Information (if different from member)
PATIENT_FIELDS = [
    FieldDefinition(
        name="patient_name",
        display_name="Patient Name",
        field_type=FieldType.NAME,
        description="Name of the patient who received services",
        required=True,
        location_hint="Patient section - may be same as member",
    ),
    FieldDefinition(
        name="patient_dob",
        display_name="Patient Date of Birth",
        field_type=FieldType.DATE,
        description="Patient's date of birth",
        required=False,
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="patient_relationship",
        display_name="Patient Relationship to Member",
        field_type=FieldType.STRING,
        description="Patient's relationship to the insurance member",
        required=False,
        allowed_values=["Self", "Spouse", "Child", "Dependent", "Other"],
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="patient_account_number",
        display_name="Patient Account Number",
        field_type=FieldType.ACCOUNT_NUMBER,
        description="Provider's account number for the patient",
        required=False,
        location_hint="Claim details section",
    ),
]

# Provider Information
PROVIDER_FIELDS = [
    FieldDefinition(
        name="provider_name",
        display_name="Provider Name",
        field_type=FieldType.STRING,
        description="Name of the healthcare provider or facility",
        required=True,
        location_hint="Provider/claim information section",
        examples=["DR. JOHN SMITH, MD", "GENERAL HOSPITAL", "ABC MEDICAL GROUP"],
    ),
    FieldDefinition(
        name="provider_address",
        display_name="Provider Address",
        field_type=FieldType.ADDRESS,
        description="Provider's address",
        required=False,
        location_hint="Provider information section",
    ),
    FieldDefinition(
        name="provider_npi",
        display_name="Provider NPI",
        field_type=FieldType.NPI,
        description="Provider's National Provider Identifier",
        required=False,
        location_hint="Provider information section",
    ),
    FieldDefinition(
        name="provider_tax_id",
        display_name="Provider Tax ID",
        field_type=FieldType.STRING,
        description="Provider's federal tax ID",
        required=False,
        location_hint="Provider information section",
        pattern=r"^\d{2}-?\d{7}$",
    ),
    FieldDefinition(
        name="network_status",
        display_name="Network Status",
        field_type=FieldType.STRING,
        description="Whether the provider is in-network or out-of-network",
        required=False,
        allowed_values=[
            "In-Network",
            "Out-of-Network",
            "In Network",
            "Out of Network",
            "Participating",
            "Non-Participating",
        ],
        location_hint="Provider or benefits section",
    ),
]

# Claim/Service Information
CLAIM_FIELDS = [
    FieldDefinition(
        name="date_of_service",
        display_name="Date of Service",
        field_type=FieldType.DATE,
        description="Date services were provided (or start date for range)",
        required=True,
        location_hint="Service details section",
    ),
    FieldDefinition(
        name="date_of_service_end",
        display_name="Date of Service End",
        field_type=FieldType.DATE,
        description="End date of services (for date ranges)",
        required=False,
        location_hint="Service details section",
    ),
    FieldDefinition(
        name="date_received",
        display_name="Date Claim Received",
        field_type=FieldType.DATE,
        description="Date the claim was received by insurance",
        required=False,
        location_hint="Claim processing section",
    ),
    FieldDefinition(
        name="date_processed",
        display_name="Date Claim Processed",
        field_type=FieldType.DATE,
        description="Date the claim was processed",
        required=False,
        location_hint="Claim processing section",
    ),
    FieldDefinition(
        name="claim_type",
        display_name="Claim Type",
        field_type=FieldType.STRING,
        description="Type of claim (Medical, Dental, Vision, Pharmacy)",
        required=False,
        allowed_values=["Medical", "Dental", "Vision", "Pharmacy", "Behavioral Health", "DME"],
        location_hint="Claim header section",
    ),
    FieldDefinition(
        name="place_of_service",
        display_name="Place of Service",
        field_type=FieldType.STRING,
        description="Where services were rendered",
        required=False,
        location_hint="Service details section",
        examples=["Office", "Outpatient Hospital", "Inpatient Hospital", "Emergency Room"],
    ),
]

# Service Line Items
SERVICE_LINE_FIELDS = [
    FieldDefinition(
        name="service_lines",
        display_name="Service Lines",
        field_type=FieldType.LIST,
        list_item_type=FieldType.OBJECT,
        nested_schema="eob_service_line",
        description="Detailed list of services with codes, charges, and payments",
        required=True,
        location_hint="Main body - service details table",
    ),
]

# Financial Summary - Charges
CHARGE_FIELDS = [
    FieldDefinition(
        name="total_billed",
        display_name="Total Amount Billed",
        field_type=FieldType.CURRENCY,
        description="Total amount billed by provider",
        required=True,
        location_hint="Summary section - billed/charged column",
        min_value=0,
    ),
    FieldDefinition(
        name="total_allowed",
        display_name="Total Allowed Amount",
        field_type=FieldType.CURRENCY,
        description="Total amount allowed/approved by insurance",
        required=True,
        location_hint="Summary section - allowed/approved column",
        min_value=0,
    ),
    FieldDefinition(
        name="total_not_covered",
        display_name="Total Not Covered",
        field_type=FieldType.CURRENCY,
        description="Total amount not covered by insurance",
        required=False,
        location_hint="Summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="total_discount",
        display_name="Total Discount/Savings",
        field_type=FieldType.CURRENCY,
        description="Total discount from network negotiated rates",
        required=False,
        location_hint="Summary section - savings/discount column",
        min_value=0,
    ),
]

# Financial Summary - Insurance Payment
PAYMENT_FIELDS = [
    FieldDefinition(
        name="plan_paid",
        display_name="Plan Paid Amount",
        field_type=FieldType.CURRENCY,
        description="Amount paid by the insurance plan",
        required=True,
        location_hint="Summary section - plan paid column",
        min_value=0,
    ),
    FieldDefinition(
        name="payment_date",
        display_name="Payment Date",
        field_type=FieldType.DATE,
        description="Date payment was issued",
        required=False,
        location_hint="Payment section",
    ),
    FieldDefinition(
        name="payment_method",
        display_name="Payment Method",
        field_type=FieldType.STRING,
        description="How payment was made (check, EFT)",
        required=False,
        allowed_values=["Check", "EFT", "Electronic Funds Transfer", "Direct Deposit"],
        location_hint="Payment section",
    ),
    FieldDefinition(
        name="check_number",
        display_name="Check/Payment Number",
        field_type=FieldType.STRING,
        description="Check or electronic payment reference number",
        required=False,
        location_hint="Payment section",
    ),
    FieldDefinition(
        name="paid_to",
        display_name="Paid To",
        field_type=FieldType.STRING,
        description="Recipient of payment (provider or member)",
        required=False,
        allowed_values=["Provider", "Member", "Subscriber"],
        location_hint="Payment section",
    ),
]

# Financial Summary - Patient Responsibility
PATIENT_RESPONSIBILITY_FIELDS = [
    FieldDefinition(
        name="deductible_applied",
        display_name="Deductible Applied",
        field_type=FieldType.CURRENCY,
        description="Amount applied to annual deductible",
        required=False,
        location_hint="Patient responsibility section",
        min_value=0,
    ),
    FieldDefinition(
        name="copay_amount",
        display_name="Copay Amount",
        field_type=FieldType.CURRENCY,
        description="Fixed copay amount owed by patient",
        required=False,
        location_hint="Patient responsibility section",
        min_value=0,
    ),
    FieldDefinition(
        name="coinsurance_amount",
        display_name="Coinsurance Amount",
        field_type=FieldType.CURRENCY,
        description="Percentage-based coinsurance owed by patient",
        required=False,
        location_hint="Patient responsibility section",
        min_value=0,
    ),
    FieldDefinition(
        name="other_patient_responsibility",
        display_name="Other Patient Responsibility",
        field_type=FieldType.CURRENCY,
        description="Other amounts patient is responsible for",
        required=False,
        location_hint="Patient responsibility section",
        min_value=0,
    ),
    FieldDefinition(
        name="total_patient_responsibility",
        display_name="Total Patient Responsibility",
        field_type=FieldType.CURRENCY,
        description="Total amount patient owes",
        required=True,
        location_hint="Summary section - you owe/patient responsibility",
        min_value=0,
    ),
]

# Benefit Accumulator Information
BENEFIT_ACCUMULATOR_FIELDS = [
    FieldDefinition(
        name="deductible_met_individual",
        display_name="Individual Deductible Met",
        field_type=FieldType.CURRENCY,
        description="Amount of individual deductible met so far",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="deductible_remaining_individual",
        display_name="Individual Deductible Remaining",
        field_type=FieldType.CURRENCY,
        description="Amount of individual deductible remaining",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="deductible_met_family",
        display_name="Family Deductible Met",
        field_type=FieldType.CURRENCY,
        description="Amount of family deductible met so far",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="deductible_remaining_family",
        display_name="Family Deductible Remaining",
        field_type=FieldType.CURRENCY,
        description="Amount of family deductible remaining",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="oop_max_met_individual",
        display_name="Individual OOP Maximum Met",
        field_type=FieldType.CURRENCY,
        description="Amount of individual out-of-pocket maximum met",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="oop_max_remaining_individual",
        display_name="Individual OOP Maximum Remaining",
        field_type=FieldType.CURRENCY,
        description="Amount of individual out-of-pocket maximum remaining",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="oop_max_met_family",
        display_name="Family OOP Maximum Met",
        field_type=FieldType.CURRENCY,
        description="Amount of family out-of-pocket maximum met",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="oop_max_remaining_family",
        display_name="Family OOP Maximum Remaining",
        field_type=FieldType.CURRENCY,
        description="Amount of family out-of-pocket maximum remaining",
        required=False,
        location_hint="Benefits summary section",
        min_value=0,
    ),
    FieldDefinition(
        name="benefit_year",
        display_name="Benefit Year",
        field_type=FieldType.STRING,
        description="Plan benefit year for accumulators",
        required=False,
        location_hint="Benefits summary section",
        pattern=r"^\d{4}$|^\d{2}/\d{2}/\d{4}\s*-\s*\d{2}/\d{2}/\d{4}$",
    ),
]

# Adjustments and Denials
ADJUSTMENT_FIELDS = [
    FieldDefinition(
        name="adjustments",
        display_name="Adjustments",
        field_type=FieldType.LIST,
        list_item_type=FieldType.OBJECT,
        nested_schema="eob_adjustment",
        description="List of adjustments applied to the claim",
        required=False,
        location_hint="Adjustments or remarks section",
    ),
    FieldDefinition(
        name="denial_reasons",
        display_name="Denial Reasons",
        field_type=FieldType.LIST,
        list_item_type=FieldType.STRING,
        description="List of reasons for any claim denials",
        required=False,
        location_hint="Remarks or denial section",
    ),
    FieldDefinition(
        name="claim_status",
        display_name="Claim Status",
        field_type=FieldType.STRING,
        description="Overall status of the claim",
        required=True,
        allowed_values=[
            "Paid",
            "Partially Paid",
            "Denied",
            "Pending",
            "In Process",
            "Processed",
            "Approved",
            "Adjusted",
        ],
        location_hint="Claim header or summary section",
    ),
    FieldDefinition(
        name="remark_codes",
        display_name="Remark Codes",
        field_type=FieldType.LIST,
        list_item_type=FieldType.STRING,
        description="Claim adjustment reason codes (CARC) or remark codes (RARC)",
        required=False,
        location_hint="Remarks section",
    ),
    FieldDefinition(
        name="remarks",
        display_name="Remarks/Notes",
        field_type=FieldType.STRING,
        description="Additional remarks or explanatory notes",
        required=False,
        location_hint="Remarks section or footer",
    ),
]

# Appeal Information
APPEAL_FIELDS = [
    FieldDefinition(
        name="appeal_deadline",
        display_name="Appeal Deadline",
        field_type=FieldType.DATE,
        description="Last date to file an appeal",
        required=False,
        location_hint="Appeal rights section or footer",
    ),
    FieldDefinition(
        name="appeal_instructions",
        display_name="Appeal Instructions",
        field_type=FieldType.STRING,
        description="Instructions on how to file an appeal",
        required=False,
        location_hint="Appeal rights section",
    ),
    FieldDefinition(
        name="appeal_address",
        display_name="Appeal Mailing Address",
        field_type=FieldType.ADDRESS,
        description="Address for submitting appeals",
        required=False,
        location_hint="Appeal rights section",
    ),
    FieldDefinition(
        name="appeal_phone",
        display_name="Appeal Phone Number",
        field_type=FieldType.PHONE,
        description="Phone number for appeal inquiries",
        required=False,
        location_hint="Appeal rights section",
    ),
]

# Coordination of Benefits
COB_FIELDS = [
    FieldDefinition(
        name="other_insurance_paid",
        display_name="Other Insurance Paid",
        field_type=FieldType.CURRENCY,
        description="Amount paid by other/primary insurance",
        required=False,
        location_hint="COB or payment section",
        min_value=0,
    ),
    FieldDefinition(
        name="other_insurance_name",
        display_name="Other Insurance Name",
        field_type=FieldType.STRING,
        description="Name of other insurance carrier",
        required=False,
        location_hint="COB section",
    ),
    FieldDefinition(
        name="cob_adjustment",
        display_name="COB Adjustment",
        field_type=FieldType.CURRENCY,
        description="Adjustment due to coordination of benefits",
        required=False,
        location_hint="COB or adjustment section",
        min_value=0,
    ),
]


# =============================================================================
# Cross-Field Validation Rules
# =============================================================================

EOB_CROSS_FIELD_RULES = [
    # Date of service must be before EOB date
    CrossFieldRule(
        source_field="date_of_service",
        target_field="eob_date",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Date of service must be before EOB date",
    ),
    # Date received must be on or after date of service
    CrossFieldRule(
        source_field="date_of_service",
        target_field="date_received",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Date of service must be before date claim received",
        severity="warning",
    ),
    # Total billed should be >= total allowed
    CrossFieldRule(
        source_field="total_billed",
        target_field="total_allowed",
        operator=RuleOperator.GREATER_EQUAL,
        error_message="Total billed should be greater than or equal to total allowed",
        severity="warning",
    ),
    # Plan paid + patient responsibility should approximate allowed amount
    CrossFieldRule(
        source_field="total_allowed",
        target_field="plan_paid",
        operator=RuleOperator.GREATER_EQUAL,
        error_message="Total allowed should be greater than or equal to plan paid",
        severity="warning",
    ),
    # If end date provided, must be >= start date
    CrossFieldRule(
        source_field="date_of_service",
        target_field="date_of_service_end",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Date of service start must be before or equal to end date",
    ),
    # Payment date should be on or after claim processed date
    CrossFieldRule(
        source_field="date_processed",
        target_field="payment_date",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Claim processed date should be before payment date",
        severity="warning",
    ),
    # If claim denied, patient responsibility should be 0 or not covered
    CrossFieldRule(
        source_field="claim_status",
        target_field="total_not_covered",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Denied claims should show not covered amount",
        severity="warning",
    ),
]


# =============================================================================
# EOB Schema Definition
# =============================================================================

EOB_SCHEMA = DocumentSchema(
    name="eob",
    display_name="Explanation of Benefits (EOB)",
    document_type=DocumentType.EOB,
    description="Statement from insurance company explaining how a medical claim was processed, what was paid, and patient responsibility",
    version="1.0.0",
    fields=(
        INSURER_FIELDS
        + DOCUMENT_FIELDS
        + MEMBER_FIELDS
        + PATIENT_FIELDS
        + PROVIDER_FIELDS
        + CLAIM_FIELDS
        + SERVICE_LINE_FIELDS
        + CHARGE_FIELDS
        + PAYMENT_FIELDS
        + PATIENT_RESPONSIBILITY_FIELDS
        + BENEFIT_ACCUMULATOR_FIELDS
        + ADJUSTMENT_FIELDS
        + APPEAL_FIELDS
        + COB_FIELDS
    ),
    cross_field_rules=EOB_CROSS_FIELD_RULES,
    required_sections=[
        "Insurance Company",
        "Member Information",
        "Patient Information",
        "Provider Information",
        "Claim Information",
        "Service Details",
        "Payment Summary",
        "Patient Responsibility",
    ],
    classification_hints=[
        "EXPLANATION OF BENEFITS",
        "EOB",
        "THIS IS NOT A BILL",
        "CLAIM SUMMARY",
        "BENEFITS STATEMENT",
        "HEALTH BENEFITS",
        "CLAIM NUMBER",
        "MEMBER ID",
        "AMOUNT BILLED",
        "AMOUNT ALLOWED",
        "PLAN PAID",
        "YOU OWE",
        "YOUR SHARE",
        "PATIENT RESPONSIBILITY",
        "DEDUCTIBLE",
        "COINSURANCE",
        "COPAY",
        "OUT-OF-POCKET",
        "NETWORK DISCOUNT",
        "CLAIM STATUS",
        "APPEAL RIGHTS",
    ],
)


def register_eob_schema() -> None:
    """Register EOB schema with the global registry."""
    registry = SchemaRegistry()
    registry.register(EOB_SCHEMA)


# Auto-register on import
register_eob_schema()
