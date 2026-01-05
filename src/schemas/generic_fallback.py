"""
Generic Fallback Schema for Unknown Documents.

Provides a flexible schema for extracting data from documents that don't match
any known medical document types (CMS-1500, UB-04, Superbill, EOB).
"""

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import FieldDefinition, FieldType


# =============================================================================
# Generic Field Definitions
# =============================================================================

# Document Identification Fields
DOCUMENT_FIELDS = [
    FieldDefinition(
        name="document_title",
        display_name="Document Title",
        field_type=FieldType.STRING,
        description="Title or heading of the document",
        required=False,
        location_hint="Top of document, header area",
    ),
    FieldDefinition(
        name="document_date",
        display_name="Document Date",
        field_type=FieldType.DATE,
        description="Date the document was created or effective",
        required=False,
        location_hint="Near header or top of document",
    ),
    FieldDefinition(
        name="document_number",
        display_name="Document Number/ID",
        field_type=FieldType.STRING,
        description="Reference number, invoice number, or document identifier",
        required=False,
        location_hint="Header area or top corner",
    ),
]

# Entity/Organization Fields
ENTITY_FIELDS = [
    FieldDefinition(
        name="organization_name",
        display_name="Organization Name",
        field_type=FieldType.STRING,
        description="Name of the primary organization or entity",
        required=False,
        location_hint="Header or letterhead area",
    ),
    FieldDefinition(
        name="organization_address",
        display_name="Organization Address",
        field_type=FieldType.ADDRESS,
        description="Full address of the organization",
        required=False,
        location_hint="Below organization name",
    ),
    FieldDefinition(
        name="organization_phone",
        display_name="Organization Phone",
        field_type=FieldType.PHONE,
        description="Contact phone number",
        required=False,
        location_hint="Near address or header",
    ),
]

# Person Information Fields
PERSON_FIELDS = [
    FieldDefinition(
        name="person_name",
        display_name="Person Name",
        field_type=FieldType.NAME,
        description="Name of the primary person mentioned",
        required=False,
        location_hint="May appear in various locations",
    ),
    FieldDefinition(
        name="person_dob",
        display_name="Date of Birth",
        field_type=FieldType.DATE,
        description="Date of birth if present",
        required=False,
        location_hint="Near person name",
    ),
    FieldDefinition(
        name="person_id",
        display_name="Person ID/Account",
        field_type=FieldType.STRING,
        description="Account number, member ID, or identifier",
        required=False,
        location_hint="Near person name",
    ),
]

# Financial Fields
FINANCIAL_FIELDS = [
    FieldDefinition(
        name="total_amount",
        display_name="Total Amount",
        field_type=FieldType.CURRENCY,
        description="Total monetary amount",
        required=False,
        location_hint="Often at bottom or in summary section",
    ),
    FieldDefinition(
        name="subtotal",
        display_name="Subtotal",
        field_type=FieldType.CURRENCY,
        description="Subtotal before adjustments",
        required=False,
        location_hint="Near total amount",
    ),
    FieldDefinition(
        name="tax_amount",
        display_name="Tax Amount",
        field_type=FieldType.CURRENCY,
        description="Tax amount if applicable",
        required=False,
        location_hint="Near total amount",
    ),
]

# Healthcare-Specific Fields (for medical documents)
HEALTHCARE_FIELDS = [
    FieldDefinition(
        name="patient_name",
        display_name="Patient Name",
        field_type=FieldType.NAME,
        description="Patient's full name",
        required=False,
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="provider_name",
        display_name="Provider Name",
        field_type=FieldType.NAME,
        description="Healthcare provider name",
        required=False,
        location_hint="Provider section or signature area",
    ),
    FieldDefinition(
        name="service_date",
        display_name="Date of Service",
        field_type=FieldType.DATE,
        description="Date services were provided",
        required=False,
        location_hint="Service details section",
    ),
    FieldDefinition(
        name="diagnosis_codes",
        display_name="Diagnosis Codes",
        field_type=FieldType.STRING,  # Use STRING for arrays
        description="ICD-10 diagnosis codes if present (comma-separated)",
        required=False,
        location_hint="Diagnosis section",
    ),
    FieldDefinition(
        name="procedure_codes",
        display_name="Procedure Codes",
        field_type=FieldType.STRING,  # Use STRING for arrays
        description="CPT procedure codes if present (comma-separated)",
        required=False,
        location_hint="Services or procedures section",
    ),
]

# =============================================================================
# Enhanced Generic Schema
# =============================================================================

# Combine all field groups
ALL_GENERIC_FIELDS = (
    DOCUMENT_FIELDS + ENTITY_FIELDS + PERSON_FIELDS + FINANCIAL_FIELDS + HEALTHCARE_FIELDS
)

# Create the enhanced generic schema
ENHANCED_GENERIC_SCHEMA = DocumentSchema(
    name="enhanced_generic",
    display_name="Generic Document",
    document_type=DocumentType.UNKNOWN,
    description=(
        "Flexible schema for extracting common fields from unknown document types. "
        "Attempts to identify standard fields like dates, amounts, names, and "
        "identifiers that are common across many document formats."
    ),
    fields=ALL_GENERIC_FIELDS,
    cross_field_rules=[],  # No cross-field validation for generic
    required_sections=[],  # No required sections for flexibility
    version="1.0.0",
    classification_hints=["generic", "fallback", "unknown"],
)

# Auto-register the schema
_registry = SchemaRegistry()
_registry.register(ENHANCED_GENERIC_SCHEMA)
