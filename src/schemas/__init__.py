"""
Schema module for document extraction.

Provides schema definitions for various medical document types,
field type definitions, and validation rules.
"""

from src.schemas.base import (
    DocumentSchema,
    DocumentType,
    ExtractionField,
    ExtractionResult,
    FieldConfidence,
    SchemaRegistry,
)
from src.schemas.field_types import (
    FieldType,
    FieldDefinition,
    CrossFieldRule,
    RuleOperator,
)
from src.schemas.validators import (
    validate_cpt_code,
    validate_icd10_code,
    validate_npi,
    validate_phone,
    validate_ssn,
    validate_date,
    validate_currency,
    validate_field,
    MedicalCodeValidator,
)

# Import healthcare schemas to auto-register them
from src.schemas.cms1500 import CMS1500_SCHEMA
from src.schemas.superbill import SUPERBILL_SCHEMA
from src.schemas.ub04 import UB04_SCHEMA
from src.schemas.eob import EOB_SCHEMA


def get_schema(document_type: DocumentType) -> DocumentSchema:
    """
    Get schema for a document type.

    Args:
        document_type: Type of document to get schema for.

    Returns:
        DocumentSchema for the specified type.

    Raises:
        ValueError: If schema not found for document type.
    """
    registry = SchemaRegistry()
    return registry.get_by_type(document_type)


def get_all_schemas() -> list[DocumentSchema]:
    """
    Get all registered schemas.

    Returns:
        List of all registered DocumentSchema objects.
    """
    registry = SchemaRegistry()
    return registry.list_schemas()


__all__ = [
    # Base schema
    "DocumentSchema",
    "DocumentType",
    "ExtractionField",
    "ExtractionResult",
    "FieldConfidence",
    "SchemaRegistry",
    # Field types
    "FieldType",
    "FieldDefinition",
    "CrossFieldRule",
    "RuleOperator",
    # Validators
    "validate_cpt_code",
    "validate_icd10_code",
    "validate_npi",
    "validate_phone",
    "validate_ssn",
    "validate_date",
    "validate_currency",
    "validate_field",
    "MedicalCodeValidator",
    # Healthcare schemas
    "CMS1500_SCHEMA",
    "SUPERBILL_SCHEMA",
    "UB04_SCHEMA",
    "EOB_SCHEMA",
    # Helper functions
    "get_schema",
    "get_all_schemas",
]
