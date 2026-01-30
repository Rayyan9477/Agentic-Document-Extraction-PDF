"""
CMS-1500 Healthcare Claim Form Schema.

Defines the complete schema for extracting data from CMS-1500
(Health Insurance Claim Form), the standard form used for
professional medical billing in the United States.
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator


# =============================================================================
# CMS-1500 Field Definitions
# =============================================================================

# Section 1: Insurance Information
INSURANCE_FIELDS = [
    FieldDefinition(
        name="insurance_type",
        display_name="Insurance Type",
        field_type=FieldType.STRING,
        description="Type of insurance program (Medicare, Medicaid, TRICARE, CHAMPVA, Group Health, FECA, Other)",
        required=True,
        allowed_values=[
            "Medicare",
            "Medicaid",
            "TRICARE",
            "CHAMPVA",
            "Group Health Plan",
            "FECA BLK LUNG",
            "Other",
        ],
        location_hint="Box 1 - top left checkboxes",
    ),
    FieldDefinition(
        name="insured_id_number",
        display_name="Insured's ID Number",
        field_type=FieldType.MEMBER_ID,
        description="Insurance member/subscriber ID number",
        required=True,
        location_hint="Box 1a - top right",
        examples=["ABC123456789", "12345678901"],
    ),
]

# Section 2: Patient Information
PATIENT_FIELDS = [
    FieldDefinition(
        name="patient_name",
        display_name="Patient Name",
        field_type=FieldType.NAME,
        description="Patient's full legal name (Last, First, Middle)",
        required=True,
        location_hint="Box 2",
        examples=["SMITH, JOHN A", "DOE, JANE MARIE"],
    ),
    FieldDefinition(
        name="patient_birth_date",
        display_name="Patient Birth Date",
        field_type=FieldType.DATE,
        description="Patient's date of birth (MM/DD/YYYY)",
        required=True,
        location_hint="Box 3 - left side",
        examples=["01/15/1980", "12/25/1955"],
    ),
    FieldDefinition(
        name="patient_sex",
        display_name="Patient Sex",
        field_type=FieldType.STRING,
        description="Patient's sex (Male or Female)",
        required=True,
        allowed_values=["Male", "Female", "M", "F"],
        location_hint="Box 3 - right side checkbox",
    ),
    FieldDefinition(
        name="patient_address",
        display_name="Patient Address",
        field_type=FieldType.ADDRESS,
        description="Patient's street address",
        required=True,
        location_hint="Box 5 - first line",
    ),
    FieldDefinition(
        name="patient_city",
        display_name="Patient City",
        field_type=FieldType.STRING,
        description="Patient's city",
        required=True,
        location_hint="Box 5 - second line left",
    ),
    FieldDefinition(
        name="patient_state",
        display_name="Patient State",
        field_type=FieldType.STATE,
        description="Patient's state (2-letter abbreviation)",
        required=True,
        location_hint="Box 5 - second line middle",
        examples=["CA", "NY", "TX"],
    ),
    FieldDefinition(
        name="patient_zip",
        display_name="Patient ZIP Code",
        field_type=FieldType.ZIP_CODE,
        description="Patient's ZIP code",
        required=True,
        location_hint="Box 5 - second line right",
        examples=["12345", "12345-6789"],
    ),
    FieldDefinition(
        name="patient_phone",
        display_name="Patient Phone",
        field_type=FieldType.PHONE,
        description="Patient's phone number",
        required=False,
        location_hint="Box 5 - phone number",
    ),
    FieldDefinition(
        name="patient_relationship",
        display_name="Patient Relationship to Insured",
        field_type=FieldType.STRING,
        description="Patient's relationship to the insured person",
        required=True,
        allowed_values=["Self", "Spouse", "Child", "Other"],
        location_hint="Box 6 - checkboxes",
    ),
]

# Section 3: Insured Information
INSURED_FIELDS = [
    FieldDefinition(
        name="insured_name",
        display_name="Insured's Name",
        field_type=FieldType.NAME,
        description="Insured/subscriber's full name (Last, First, Middle)",
        required=False,
        location_hint="Box 4",
    ),
    FieldDefinition(
        name="insured_address",
        display_name="Insured's Address",
        field_type=FieldType.ADDRESS,
        description="Insured's street address",
        required=False,
        location_hint="Box 7",
    ),
    FieldDefinition(
        name="insured_city",
        display_name="Insured City",
        field_type=FieldType.STRING,
        description="Insured's city",
        required=False,
        location_hint="Box 7 - city",
    ),
    FieldDefinition(
        name="insured_state",
        display_name="Insured State",
        field_type=FieldType.STATE,
        description="Insured's state",
        required=False,
        location_hint="Box 7 - state",
    ),
    FieldDefinition(
        name="insured_zip",
        display_name="Insured ZIP",
        field_type=FieldType.ZIP_CODE,
        description="Insured's ZIP code",
        required=False,
        location_hint="Box 7 - zip",
    ),
    FieldDefinition(
        name="insured_phone",
        display_name="Insured Phone",
        field_type=FieldType.PHONE,
        description="Insured's phone number",
        required=False,
        location_hint="Box 7 - phone",
    ),
    FieldDefinition(
        name="other_insured_name",
        display_name="Other Insured Name",
        field_type=FieldType.NAME,
        description="Other insured's name if secondary insurance",
        required=False,
        location_hint="Box 9",
    ),
    FieldDefinition(
        name="other_insured_policy_number",
        display_name="Other Insured Policy Number",
        field_type=FieldType.POLICY_NUMBER,
        description="Other insured's policy/group number",
        required=False,
        location_hint="Box 9a",
    ),
    FieldDefinition(
        name="insured_policy_group",
        display_name="Insured Policy/Group Number",
        field_type=FieldType.GROUP_NUMBER,
        description="Insured's policy or group number",
        required=False,
        location_hint="Box 11",
    ),
    FieldDefinition(
        name="insured_birth_date",
        display_name="Insured Birth Date",
        field_type=FieldType.DATE,
        description="Insured's date of birth",
        required=False,
        location_hint="Box 11a - left",
    ),
    FieldDefinition(
        name="insured_sex",
        display_name="Insured Sex",
        field_type=FieldType.STRING,
        description="Insured's sex",
        allowed_values=["Male", "Female", "M", "F"],
        required=False,
        location_hint="Box 11a - right",
    ),
    FieldDefinition(
        name="insured_employer",
        display_name="Insured Employer",
        field_type=FieldType.STRING,
        description="Insured's employer or school name",
        required=False,
        location_hint="Box 11b",
    ),
    FieldDefinition(
        name="insurance_plan_name",
        display_name="Insurance Plan Name",
        field_type=FieldType.STRING,
        description="Insurance plan or program name",
        required=False,
        location_hint="Box 11c",
    ),
    FieldDefinition(
        name="other_health_benefit_plan",
        display_name="Is There Another Health Benefit Plan?",
        field_type=FieldType.BOOLEAN,
        description="Is there another health benefit plan? Required for Coordination of Benefits",
        required=False,
        location_hint="Box 11d",
    ),
]

# Section 4: Condition Information
CONDITION_FIELDS = [
    FieldDefinition(
        name="condition_employment",
        display_name="Condition Related to Employment",
        field_type=FieldType.BOOLEAN,
        description="Is condition related to patient's employment?",
        required=False,
        location_hint="Box 10a",
    ),
    FieldDefinition(
        name="condition_auto_accident",
        display_name="Condition Related to Auto Accident",
        field_type=FieldType.BOOLEAN,
        description="Is condition related to an auto accident?",
        required=False,
        location_hint="Box 10b",
    ),
    FieldDefinition(
        name="auto_accident_state",
        display_name="Auto Accident State",
        field_type=FieldType.STATE,
        description="State where auto accident occurred",
        required=False,
        location_hint="Box 10b - state",
    ),
    FieldDefinition(
        name="condition_other_accident",
        display_name="Condition Related to Other Accident",
        field_type=FieldType.BOOLEAN,
        description="Is condition related to another type of accident?",
        required=False,
        location_hint="Box 10c",
    ),
    FieldDefinition(
        name="claim_codes",
        display_name="Claim Codes",
        field_type=FieldType.STRING,
        description="Condition codes designated by NUCC (e.g., W1, W2)",
        required=False,
        location_hint="Box 10d",
    ),
    FieldDefinition(
        name="illness_date",
        display_name="Date of Current Illness/Injury",
        field_type=FieldType.DATE,
        description="Date of current illness, injury, or pregnancy",
        required=False,
        location_hint="Box 14",
    ),
    FieldDefinition(
        name="similar_illness_date",
        display_name="Date of Similar Illness",
        field_type=FieldType.DATE,
        description="Date patient had same or similar illness",
        required=False,
        location_hint="Box 15",
    ),
    FieldDefinition(
        name="unable_to_work_from",
        display_name="Unable to Work From Date",
        field_type=FieldType.DATE,
        description="Date patient was unable to work from",
        required=False,
        location_hint="Box 16 - from",
    ),
    FieldDefinition(
        name="unable_to_work_to",
        display_name="Unable to Work To Date",
        field_type=FieldType.DATE,
        description="Date patient was unable to work to",
        required=False,
        location_hint="Box 16 - to",
    ),
]

# Section 5: Provider Information
PROVIDER_FIELDS = [
    FieldDefinition(
        name="referring_provider_name",
        display_name="Referring Provider Name",
        field_type=FieldType.NAME,
        description="Name of referring physician",
        required=False,
        location_hint="Box 17",
    ),
    FieldDefinition(
        name="referring_provider_npi",
        display_name="Referring Provider NPI",
        field_type=FieldType.NPI,
        description="Referring physician's NPI",
        required=False,
        location_hint="Box 17b",
    ),
    FieldDefinition(
        name="hospitalization_from",
        display_name="Hospitalization From Date",
        field_type=FieldType.DATE,
        description="Hospitalization dates from",
        required=False,
        location_hint="Box 18 - from",
    ),
    FieldDefinition(
        name="hospitalization_to",
        display_name="Hospitalization To Date",
        field_type=FieldType.DATE,
        description="Hospitalization dates to",
        required=False,
        location_hint="Box 18 - to",
    ),
    FieldDefinition(
        name="additional_claim_info",
        display_name="Additional Claim Information",
        field_type=FieldType.STRING,
        description="Additional claim information or reserved for local use",
        required=False,
        location_hint="Box 19",
    ),
    FieldDefinition(
        name="outside_lab",
        display_name="Outside Lab",
        field_type=FieldType.BOOLEAN,
        description="Were outside lab services performed?",
        required=False,
        location_hint="Box 20 - checkbox",
    ),
    FieldDefinition(
        name="outside_lab_charges",
        display_name="Outside Lab Charges",
        field_type=FieldType.CURRENCY,
        description="Charges for outside lab services",
        required=False,
        location_hint="Box 20 - charges",
    ),
    FieldDefinition(
        name="resubmission_code",
        display_name="Resubmission Code",
        field_type=FieldType.STRING,
        description="Frequency code for original/replacement/void claim (7=replacement, 8=void)",
        required=False,
        allowed_values=["1", "7", "8"],
        location_hint="Box 22 - left",
    ),
    FieldDefinition(
        name="original_reference_number",
        display_name="Original Reference Number",
        field_type=FieldType.STRING,
        description="Original claim reference number for resubmissions",
        required=False,
        location_hint="Box 22 - right",
    ),
    FieldDefinition(
        name="prior_authorization_number",
        display_name="Prior Authorization Number",
        field_type=FieldType.STRING,
        description="Prior authorization or referral number from payer",
        required=False,
        location_hint="Box 23",
        examples=["AUTH123456", "REF789012"],
    ),
]

# Section 6: Diagnosis Codes
DIAGNOSIS_FIELDS = [
    FieldDefinition(
        name="icd_indicator",
        display_name="ICD Indicator",
        field_type=FieldType.STRING,
        description="ICD version indicator: 9 for ICD-9, 0 for ICD-10",
        required=False,
        allowed_values=["9", "0"],
        location_hint="Box 21 - top right indicator",
    ),
    FieldDefinition(
        name="diagnosis_code_a",
        display_name="Diagnosis Code A",
        field_type=FieldType.ICD10_CODE,
        description="Primary diagnosis code (ICD-10-CM)",
        required=True,
        location_hint="Box 21 - line A",
        examples=["E11.9", "J06.9", "M54.5"],
    ),
    FieldDefinition(
        name="diagnosis_code_b",
        display_name="Diagnosis Code B",
        field_type=FieldType.ICD10_CODE,
        description="Secondary diagnosis code",
        required=False,
        location_hint="Box 21 - line B",
    ),
    FieldDefinition(
        name="diagnosis_code_c",
        display_name="Diagnosis Code C",
        field_type=FieldType.ICD10_CODE,
        description="Tertiary diagnosis code",
        required=False,
        location_hint="Box 21 - line C",
    ),
    FieldDefinition(
        name="diagnosis_code_d",
        display_name="Diagnosis Code D",
        field_type=FieldType.ICD10_CODE,
        description="Fourth diagnosis code",
        required=False,
        location_hint="Box 21 - line D",
    ),
    FieldDefinition(
        name="diagnosis_code_e",
        display_name="Diagnosis Code E",
        field_type=FieldType.ICD10_CODE,
        description="Fifth diagnosis code",
        required=False,
        location_hint="Box 21 - line E",
    ),
    FieldDefinition(
        name="diagnosis_code_f",
        display_name="Diagnosis Code F",
        field_type=FieldType.ICD10_CODE,
        description="Sixth diagnosis code",
        required=False,
        location_hint="Box 21 - line F",
    ),
    FieldDefinition(
        name="diagnosis_code_g",
        display_name="Diagnosis Code G",
        field_type=FieldType.ICD10_CODE,
        description="Seventh diagnosis code",
        required=False,
        location_hint="Box 21 - line G",
    ),
    FieldDefinition(
        name="diagnosis_code_h",
        display_name="Diagnosis Code H",
        field_type=FieldType.ICD10_CODE,
        description="Eighth diagnosis code",
        required=False,
        location_hint="Box 21 - line H",
    ),
    FieldDefinition(
        name="diagnosis_code_i",
        display_name="Diagnosis Code I",
        field_type=FieldType.ICD10_CODE,
        description="Ninth diagnosis code",
        required=False,
        location_hint="Box 21 - line I",
    ),
    FieldDefinition(
        name="diagnosis_code_j",
        display_name="Diagnosis Code J",
        field_type=FieldType.ICD10_CODE,
        description="Tenth diagnosis code",
        required=False,
        location_hint="Box 21 - line J",
    ),
    FieldDefinition(
        name="diagnosis_code_k",
        display_name="Diagnosis Code K",
        field_type=FieldType.ICD10_CODE,
        description="Eleventh diagnosis code",
        required=False,
        location_hint="Box 21 - line K",
    ),
    FieldDefinition(
        name="diagnosis_code_l",
        display_name="Diagnosis Code L",
        field_type=FieldType.ICD10_CODE,
        description="Twelfth diagnosis code",
        required=False,
        location_hint="Box 21 - line L",
    ),
]

# Section 7: Service Lines
SERVICE_LINE_FIELDS = [
    FieldDefinition(
        name="service_lines",
        display_name="Service Lines",
        field_type=FieldType.LIST,
        list_item_type=FieldType.OBJECT,
        nested_schema="cms1500_service_line",
        description="List of service lines (up to 6)",
        required=True,
        location_hint="Box 24 - lines 1-6",
    ),
]

# Section 8: Totals and Billing
BILLING_FIELDS = [
    FieldDefinition(
        name="federal_tax_id",
        display_name="Federal Tax ID",
        field_type=FieldType.STRING,
        description="Billing provider's federal tax ID (SSN or EIN)",
        required=True,
        location_hint="Box 25",
        pattern=r"^\d{2}-?\d{7}$|^\d{3}-?\d{2}-?\d{4}$",
    ),
    FieldDefinition(
        name="tax_id_type",
        display_name="Tax ID Type",
        field_type=FieldType.STRING,
        description="Type of tax ID (SSN or EIN)",
        allowed_values=["SSN", "EIN"],
        required=False,
        location_hint="Box 25 - checkbox",
    ),
    FieldDefinition(
        name="patient_account_number",
        display_name="Patient Account Number",
        field_type=FieldType.ACCOUNT_NUMBER,
        description="Patient's account number assigned by provider",
        required=False,
        location_hint="Box 26",
    ),
    FieldDefinition(
        name="accept_assignment",
        display_name="Accept Assignment",
        field_type=FieldType.BOOLEAN,
        description="Does provider accept assignment?",
        required=False,
        location_hint="Box 27 - checkbox",
    ),
    FieldDefinition(
        name="total_charge",
        display_name="Total Charge",
        field_type=FieldType.CURRENCY,
        description="Total charges for all service lines",
        required=True,
        location_hint="Box 28",
        min_value=0.01,
    ),
    FieldDefinition(
        name="amount_paid",
        display_name="Amount Paid",
        field_type=FieldType.CURRENCY,
        description="Amount already paid",
        required=False,
        location_hint="Box 29",
        min_value=0,
    ),
    FieldDefinition(
        name="balance_due",
        display_name="Balance Due",
        field_type=FieldType.CURRENCY,
        description="Balance due (if shown - reserved for NUCC use)",
        required=False,
        location_hint="Box 30",
    ),
]

# Section 9: Signatures and Provider Info
SIGNATURE_FIELDS = [
    FieldDefinition(
        name="patient_signature",
        display_name="Patient Signature",
        field_type=FieldType.SIGNATURE,
        description="Patient or authorized person's signature",
        required=False,
        location_hint="Box 12",
    ),
    FieldDefinition(
        name="patient_signature_date",
        display_name="Patient Signature Date",
        field_type=FieldType.DATE,
        description="Date patient signed",
        required=False,
        location_hint="Box 12 - date",
    ),
    FieldDefinition(
        name="insured_signature",
        display_name="Insured Signature",
        field_type=FieldType.SIGNATURE,
        description="Insured's or authorized person's signature",
        required=False,
        location_hint="Box 13",
    ),
    FieldDefinition(
        name="physician_signature",
        display_name="Physician Signature",
        field_type=FieldType.SIGNATURE,
        description="Physician/supplier signature",
        required=False,
        location_hint="Box 31",
    ),
    FieldDefinition(
        name="physician_signature_date",
        display_name="Physician Signature Date",
        field_type=FieldType.DATE,
        description="Date physician signed",
        required=False,
        location_hint="Box 31 - date",
    ),
]

# Section 10: Facility and Billing Provider
FACILITY_FIELDS = [
    FieldDefinition(
        name="service_facility_name",
        display_name="Service Facility Name",
        field_type=FieldType.STRING,
        description="Name of facility where services were rendered",
        required=False,
        location_hint="Box 32 - name",
    ),
    FieldDefinition(
        name="service_facility_address",
        display_name="Service Facility Address",
        field_type=FieldType.ADDRESS,
        description="Address of service facility",
        required=False,
        location_hint="Box 32 - address",
    ),
    FieldDefinition(
        name="service_facility_npi",
        display_name="Service Facility NPI",
        field_type=FieldType.NPI,
        description="Service facility's NPI",
        required=False,
        location_hint="Box 32a",
    ),
    FieldDefinition(
        name="billing_provider_name",
        display_name="Billing Provider Name",
        field_type=FieldType.STRING,
        description="Name of billing provider",
        required=True,
        location_hint="Box 33 - name",
    ),
    FieldDefinition(
        name="billing_provider_address",
        display_name="Billing Provider Address",
        field_type=FieldType.ADDRESS,
        description="Billing provider's address",
        required=True,
        location_hint="Box 33 - address",
    ),
    FieldDefinition(
        name="billing_provider_city",
        display_name="Billing Provider City",
        field_type=FieldType.STRING,
        description="Billing provider's city",
        required=True,
        location_hint="Box 33 - city",
    ),
    FieldDefinition(
        name="billing_provider_state",
        display_name="Billing Provider State",
        field_type=FieldType.STATE,
        description="Billing provider's state",
        required=True,
        location_hint="Box 33 - state",
    ),
    FieldDefinition(
        name="billing_provider_zip",
        display_name="Billing Provider ZIP",
        field_type=FieldType.ZIP_CODE,
        description="Billing provider's ZIP code",
        required=True,
        location_hint="Box 33 - zip",
    ),
    FieldDefinition(
        name="billing_provider_phone",
        display_name="Billing Provider Phone",
        field_type=FieldType.PHONE,
        description="Billing provider's phone number",
        required=False,
        location_hint="Box 33 - phone",
    ),
    FieldDefinition(
        name="billing_provider_npi",
        display_name="Billing Provider NPI",
        field_type=FieldType.NPI,
        description="Billing provider's NPI",
        required=True,
        location_hint="Box 33a",
    ),
]

# =============================================================================
# Cross-Field Validation Rules
# =============================================================================

CMS1500_CROSS_FIELD_RULES = [
    # Date validations
    CrossFieldRule(
        source_field="patient_birth_date",
        target_field="illness_date",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Patient birth date must be before illness date",
    ),
    CrossFieldRule(
        source_field="unable_to_work_from",
        target_field="unable_to_work_to",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Unable to work 'from' date must be before 'to' date",
    ),
    CrossFieldRule(
        source_field="hospitalization_from",
        target_field="hospitalization_to",
        operator=RuleOperator.DATE_BEFORE,
        error_message="Hospitalization 'from' date must be before 'to' date",
    ),
    # Required if patient is not self
    CrossFieldRule(
        source_field="patient_relationship",
        target_field="insured_name",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Insured name required when patient is not self",
    ),
    # Auto accident state required if auto accident
    CrossFieldRule(
        source_field="condition_auto_accident",
        target_field="auto_accident_state",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Auto accident state required when auto accident is marked",
    ),
    # Outside lab charges required if outside lab checked
    CrossFieldRule(
        source_field="outside_lab",
        target_field="outside_lab_charges",
        operator=RuleOperator.REQUIRES_IF,
        error_message="Outside lab charges required when outside lab is marked",
    ),
]

# =============================================================================
# CMS-1500 Schema Definition
# =============================================================================

CMS1500_SCHEMA = DocumentSchema(
    name="cms1500",
    display_name="CMS-1500 Health Insurance Claim Form",
    document_type=DocumentType.CMS_1500,
    description="Standard healthcare professional claim form (HCFA-1500) for submitting claims to insurance payers",
    version="02/12",
    fields=(
        INSURANCE_FIELDS
        + PATIENT_FIELDS
        + INSURED_FIELDS
        + CONDITION_FIELDS
        + PROVIDER_FIELDS
        + DIAGNOSIS_FIELDS
        + SERVICE_LINE_FIELDS
        + BILLING_FIELDS
        + SIGNATURE_FIELDS
        + FACILITY_FIELDS
    ),
    cross_field_rules=CMS1500_CROSS_FIELD_RULES,
    required_sections=[
        "Patient Information",
        "Insurance Information",
        "Diagnosis Codes",
        "Service Lines",
        "Billing Provider",
    ],
    classification_hints=[
        "CMS-1500",
        "HCFA-1500",
        "Health Insurance Claim Form",
        "PLEASE PRINT OR TYPE",
        "APPROVED OMB",
        "PATIENT AND INSURED INFORMATION",
        "PHYSICIAN OR SUPPLIER INFORMATION",
    ],
)


def register_cms1500_schema() -> None:
    """Register CMS-1500 schema with the global registry."""
    registry = SchemaRegistry()
    registry.register(CMS1500_SCHEMA)


# Auto-register on import
register_cms1500_schema()
