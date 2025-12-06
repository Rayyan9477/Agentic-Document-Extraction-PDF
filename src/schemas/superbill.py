"""
Medical Superbill Schema.

Defines the complete schema for extracting data from medical superbills,
which are itemized forms listing services provided during a patient visit.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# Superbill Field Definitions
# =============================================================================

# Practice Information
PRACTICE_FIELDS = [
    FieldDefinition(
        name="practice_name",
        display_name="Practice Name",
        field_type=FieldType.STRING,
        description="Name of the medical practice or clinic",
        required=True,
        location_hint="Top of form - header area",
        examples=["Smith Family Medicine", "Downtown Medical Center"],
    ),
    FieldDefinition(
        name="practice_address",
        display_name="Practice Address",
        field_type=FieldType.ADDRESS,
        description="Street address of the practice",
        required=True,
        location_hint="Below practice name",
    ),
    FieldDefinition(
        name="practice_city",
        display_name="Practice City",
        field_type=FieldType.STRING,
        description="City of the practice",
        required=True,
        location_hint="Practice address line",
    ),
    FieldDefinition(
        name="practice_state",
        display_name="Practice State",
        field_type=FieldType.STATE,
        description="State of the practice (2-letter code)",
        required=True,
        location_hint="Practice address line",
        examples=["CA", "TX", "NY"],
    ),
    FieldDefinition(
        name="practice_zip",
        display_name="Practice ZIP Code",
        field_type=FieldType.ZIP_CODE,
        description="ZIP code of the practice",
        required=True,
        location_hint="Practice address line",
    ),
    FieldDefinition(
        name="practice_phone",
        display_name="Practice Phone",
        field_type=FieldType.PHONE,
        description="Practice phone number",
        required=False,
        location_hint="Near practice name or address",
    ),
    FieldDefinition(
        name="practice_fax",
        display_name="Practice Fax",
        field_type=FieldType.FAX,
        description="Practice fax number",
        required=False,
        location_hint="Near practice phone",
    ),
    FieldDefinition(
        name="practice_npi",
        display_name="Practice NPI",
        field_type=FieldType.NPI,
        description="Practice/Group National Provider Identifier",
        required=False,
        location_hint="Near practice name or bottom of form",
    ),
    FieldDefinition(
        name="practice_tax_id",
        display_name="Practice Tax ID",
        field_type=FieldType.STRING,
        description="Practice federal tax ID (EIN)",
        required=False,
        location_hint="Near practice NPI",
        pattern=r"^\d{2}-?\d{7}$",
    ),
]

# Provider Information
PROVIDER_FIELDS = [
    FieldDefinition(
        name="provider_name",
        display_name="Provider Name",
        field_type=FieldType.NAME,
        description="Name of the treating physician/provider",
        required=True,
        location_hint="Provider section or signature area",
        examples=["Dr. John Smith, MD", "Jane Doe, NP"],
    ),
    FieldDefinition(
        name="provider_npi",
        display_name="Provider NPI",
        field_type=FieldType.NPI,
        description="Individual provider's National Provider Identifier",
        required=True,
        location_hint="Near provider name",
    ),
    FieldDefinition(
        name="provider_specialty",
        display_name="Provider Specialty",
        field_type=FieldType.STRING,
        description="Provider's medical specialty",
        required=False,
        location_hint="Near provider name",
        examples=["Family Medicine", "Internal Medicine", "Cardiology"],
    ),
    FieldDefinition(
        name="provider_signature",
        display_name="Provider Signature",
        field_type=FieldType.SIGNATURE,
        description="Provider's signature on the superbill",
        required=False,
        location_hint="Bottom of form",
    ),
]

# Patient Information
PATIENT_FIELDS = [
    FieldDefinition(
        name="patient_name",
        display_name="Patient Name",
        field_type=FieldType.NAME,
        description="Patient's full name (Last, First MI)",
        required=True,
        location_hint="Patient information section",
        examples=["JOHNSON, MARY A", "SMITH, ROBERT"],
    ),
    FieldDefinition(
        name="patient_dob",
        display_name="Patient Date of Birth",
        field_type=FieldType.DATE,
        description="Patient's date of birth",
        required=True,
        location_hint="Near patient name",
        examples=["01/15/1965", "12/25/1980"],
    ),
    FieldDefinition(
        name="patient_sex",
        display_name="Patient Sex",
        field_type=FieldType.STRING,
        description="Patient's sex/gender",
        required=False,
        allowed_values=["Male", "Female", "M", "F"],
        location_hint="Near patient DOB",
    ),
    FieldDefinition(
        name="patient_address",
        display_name="Patient Address",
        field_type=FieldType.ADDRESS,
        description="Patient's street address",
        required=False,
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="patient_city",
        display_name="Patient City",
        field_type=FieldType.STRING,
        description="Patient's city",
        required=False,
        location_hint="Patient address line",
    ),
    FieldDefinition(
        name="patient_state",
        display_name="Patient State",
        field_type=FieldType.STATE,
        description="Patient's state",
        required=False,
        location_hint="Patient address line",
    ),
    FieldDefinition(
        name="patient_zip",
        display_name="Patient ZIP Code",
        field_type=FieldType.ZIP_CODE,
        description="Patient's ZIP code",
        required=False,
        location_hint="Patient address line",
    ),
    FieldDefinition(
        name="patient_phone",
        display_name="Patient Phone",
        field_type=FieldType.PHONE,
        description="Patient's phone number",
        required=False,
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="patient_account_number",
        display_name="Patient Account Number",
        field_type=FieldType.ACCOUNT_NUMBER,
        description="Patient's account number in the practice",
        required=False,
        location_hint="Near patient name",
    ),
    FieldDefinition(
        name="chart_number",
        display_name="Chart/Medical Record Number",
        field_type=FieldType.STRING,
        description="Patient's chart or medical record number",
        required=False,
        location_hint="Near patient name",
    ),
]

# Visit Information
VISIT_FIELDS = [
    FieldDefinition(
        name="date_of_service",
        display_name="Date of Service",
        field_type=FieldType.DATE,
        description="Date services were provided",
        required=True,
        location_hint="Top of form near patient info",
        examples=["01/15/2024", "12/05/2023"],
    ),
    FieldDefinition(
        name="visit_type",
        display_name="Visit Type",
        field_type=FieldType.STRING,
        description="Type of visit (New, Established, Follow-up)",
        required=False,
        allowed_values=["New Patient", "Established Patient", "Follow-up", "Consultation"],
        location_hint="Near service codes section",
    ),
    FieldDefinition(
        name="place_of_service",
        display_name="Place of Service",
        field_type=FieldType.STRING,
        description="Location where services were rendered",
        required=False,
        location_hint="Near date of service",
        examples=["Office", "Hospital", "Telehealth", "11", "22"],
    ),
    FieldDefinition(
        name="authorization_number",
        display_name="Prior Authorization Number",
        field_type=FieldType.STRING,
        description="Prior authorization or referral number",
        required=False,
        location_hint="Insurance or authorization section",
    ),
]

# Insurance Information
INSURANCE_FIELDS = [
    FieldDefinition(
        name="primary_insurance_name",
        display_name="Primary Insurance Name",
        field_type=FieldType.STRING,
        description="Name of primary insurance company",
        required=False,
        location_hint="Insurance section",
        examples=["Blue Cross Blue Shield", "Aetna", "United Healthcare"],
    ),
    FieldDefinition(
        name="primary_insurance_id",
        display_name="Primary Insurance ID",
        field_type=FieldType.MEMBER_ID,
        description="Primary insurance member/subscriber ID",
        required=False,
        location_hint="Near primary insurance name",
    ),
    FieldDefinition(
        name="primary_insurance_group",
        display_name="Primary Insurance Group",
        field_type=FieldType.GROUP_NUMBER,
        description="Primary insurance group number",
        required=False,
        location_hint="Near primary insurance ID",
    ),
    FieldDefinition(
        name="subscriber_name",
        display_name="Subscriber Name",
        field_type=FieldType.NAME,
        description="Name of insurance subscriber if different from patient",
        required=False,
        location_hint="Insurance section",
    ),
    FieldDefinition(
        name="subscriber_dob",
        display_name="Subscriber Date of Birth",
        field_type=FieldType.DATE,
        description="Subscriber's date of birth",
        required=False,
        location_hint="Near subscriber name",
    ),
    FieldDefinition(
        name="patient_relationship",
        display_name="Patient Relationship to Subscriber",
        field_type=FieldType.STRING,
        description="Patient's relationship to insurance subscriber",
        required=False,
        allowed_values=["Self", "Spouse", "Child", "Other"],
        location_hint="Insurance section",
    ),
    FieldDefinition(
        name="copay_amount",
        display_name="Copay Amount",
        field_type=FieldType.CURRENCY,
        description="Patient copay amount",
        required=False,
        location_hint="Payment section",
        min_value=0,
    ),
    FieldDefinition(
        name="copay_collected",
        display_name="Copay Collected",
        field_type=FieldType.BOOLEAN,
        description="Was copay collected at time of visit?",
        required=False,
        location_hint="Payment section",
    ),
]

# Diagnosis Codes
DIAGNOSIS_FIELDS = [
    FieldDefinition(
        name="diagnosis_1",
        display_name="Diagnosis Code 1 (Primary)",
        field_type=FieldType.ICD10_CODE,
        description="Primary diagnosis ICD-10 code",
        required=True,
        location_hint="Diagnosis section - first line",
        examples=["E11.9", "I10", "J06.9"],
    ),
    FieldDefinition(
        name="diagnosis_1_description",
        display_name="Diagnosis 1 Description",
        field_type=FieldType.STRING,
        description="Description of primary diagnosis",
        required=False,
        location_hint="Next to diagnosis code 1",
    ),
    FieldDefinition(
        name="diagnosis_2",
        display_name="Diagnosis Code 2",
        field_type=FieldType.ICD10_CODE,
        description="Secondary diagnosis ICD-10 code",
        required=False,
        location_hint="Diagnosis section - second line",
    ),
    FieldDefinition(
        name="diagnosis_2_description",
        display_name="Diagnosis 2 Description",
        field_type=FieldType.STRING,
        description="Description of secondary diagnosis",
        required=False,
        location_hint="Next to diagnosis code 2",
    ),
    FieldDefinition(
        name="diagnosis_3",
        display_name="Diagnosis Code 3",
        field_type=FieldType.ICD10_CODE,
        description="Third diagnosis ICD-10 code",
        required=False,
        location_hint="Diagnosis section - third line",
    ),
    FieldDefinition(
        name="diagnosis_3_description",
        display_name="Diagnosis 3 Description",
        field_type=FieldType.STRING,
        description="Description of third diagnosis",
        required=False,
        location_hint="Next to diagnosis code 3",
    ),
    FieldDefinition(
        name="diagnosis_4",
        display_name="Diagnosis Code 4",
        field_type=FieldType.ICD10_CODE,
        description="Fourth diagnosis ICD-10 code",
        required=False,
        location_hint="Diagnosis section - fourth line",
    ),
    FieldDefinition(
        name="diagnosis_4_description",
        display_name="Diagnosis 4 Description",
        field_type=FieldType.STRING,
        description="Description of fourth diagnosis",
        required=False,
        location_hint="Next to diagnosis code 4",
    ),
]

# Service/Procedure Codes (CPT)
SERVICE_FIELDS = [
    FieldDefinition(
        name="procedures",
        display_name="Procedures/Services",
        field_type=FieldType.LIST,
        list_item_type=FieldType.OBJECT,
        nested_schema="superbill_procedure",
        description="List of procedures/services performed with CPT codes",
        required=True,
        location_hint="Main service codes section - usually center of form",
    ),
]

# Financial Totals
FINANCIAL_FIELDS = [
    FieldDefinition(
        name="total_charges",
        display_name="Total Charges",
        field_type=FieldType.CURRENCY,
        description="Total charges for all services",
        required=True,
        location_hint="Bottom of form - totals section",
        min_value=0.01,
    ),
    FieldDefinition(
        name="payment_received",
        display_name="Payment Received",
        field_type=FieldType.CURRENCY,
        description="Payment received at time of service",
        required=False,
        location_hint="Payment section",
        min_value=0,
    ),
    FieldDefinition(
        name="payment_method",
        display_name="Payment Method",
        field_type=FieldType.STRING,
        description="Method of payment (Cash, Check, Credit Card)",
        required=False,
        allowed_values=["Cash", "Check", "Credit Card", "Debit Card", "None"],
        location_hint="Payment section",
    ),
    FieldDefinition(
        name="balance_due",
        display_name="Balance Due",
        field_type=FieldType.CURRENCY,
        description="Remaining balance due",
        required=False,
        location_hint="Bottom of form",
        min_value=0,
    ),
    FieldDefinition(
        name="next_appointment",
        display_name="Next Appointment",
        field_type=FieldType.DATE,
        description="Date of next scheduled appointment",
        required=False,
        location_hint="Bottom of form",
    ),
    FieldDefinition(
        name="return_visit",
        display_name="Return Visit Instructions",
        field_type=FieldType.STRING,
        description="Instructions for return visit (e.g., '2 weeks', 'PRN')",
        required=False,
        location_hint="Bottom of form",
        examples=["2 weeks", "1 month", "PRN", "As needed"],
    ),
]

# =============================================================================
# Cross-Field Validation Rules
# =============================================================================

SUPERBILL_CROSS_FIELD_RULES = [
    # Patient DOB must be before date of service
    CrossFieldRule(
        source_field="patient_dob",
        target_field="date_of_service",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Patient date of birth must be before date of service",
    ),
    # Total charges must be greater than 0
    CrossFieldRule(
        source_field="total_charges",
        target_field="payment_received",
        operator=RuleOperator.GREATER_EQUAL,
        error_message="Total charges should be greater than or equal to payment received",
        severity="warning",
    ),
    # If copay collected, copay amount should be provided
    CrossFieldRule(
        source_field="copay_collected",
        target_field="copay_amount",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Copay amount required when copay collected is marked",
    ),
    # Subscriber info required if patient is not self
    CrossFieldRule(
        source_field="patient_relationship",
        target_field="subscriber_name",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Subscriber name required when patient is not the subscriber",
    ),
]

# =============================================================================
# Superbill Schema Definition
# =============================================================================

SUPERBILL_SCHEMA = DocumentSchema(
    name="superbill",
    display_name="Medical Superbill",
    document_type=DocumentType.SUPERBILL,
    description="Itemized form listing services provided during a patient visit, including CPT codes, ICD-10 diagnoses, and charges",
    version="1.0.0",
    fields=(
        PRACTICE_FIELDS
        + PROVIDER_FIELDS
        + PATIENT_FIELDS
        + VISIT_FIELDS
        + INSURANCE_FIELDS
        + DIAGNOSIS_FIELDS
        + SERVICE_FIELDS
        + FINANCIAL_FIELDS
    ),
    cross_field_rules=SUPERBILL_CROSS_FIELD_RULES,
    required_sections=[
        "Practice Information",
        "Patient Information",
        "Date of Service",
        "Diagnosis Codes",
        "Procedures/Services",
        "Total Charges",
    ],
    classification_hints=[
        "SUPERBILL",
        "ENCOUNTER FORM",
        "CHARGE SLIP",
        "FEE TICKET",
        "ROUTING SLIP",
        "CPT CODE",
        "ICD-10",
        "DIAGNOSIS",
        "PROCEDURE",
        "OFFICE VISIT",
        "E/M CODE",
    ],
)


def register_superbill_schema() -> None:
    """Register Superbill schema with the global registry."""
    registry = SchemaRegistry()
    registry.register(SUPERBILL_SCHEMA)


# Auto-register on import
register_superbill_schema()
