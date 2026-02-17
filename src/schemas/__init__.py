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

# Import healthcare schemas to auto-register them
from src.schemas.cms1500 import CMS1500_SCHEMA
from src.schemas.eob import EOB_SCHEMA
from src.schemas.field_types import (
    CrossFieldRule,
    FieldDefinition,
    FieldType,
    RuleOperator,
)

# Import finance schemas to auto-register them
from src.schemas.bank_statement import BANK_STATEMENT_SCHEMA
from src.schemas.form_1099 import FORM_1099_SCHEMA

# Import enhanced generic fallback for unknown documents
from src.schemas.generic_fallback import ENHANCED_GENERIC_SCHEMA

# Import nested schemas
from src.schemas.nested_schemas import (
    CMS1500_SERVICE_LINE_SCHEMA,
    EOB_ADJUSTMENT_SCHEMA,
    EOB_SERVICE_LINE_SCHEMA,
    SUPERBILL_PROCEDURE_SCHEMA,
    UB04_CODE_CODE_SCHEMA,
    UB04_SERVICE_LINE_SCHEMA,
    NestedSchema,
    NestedSchemaRegistry,
    get_nested_schema,
)

# Import schema builders for dynamic schema creation
from src.schemas.schema_builder import (
    FieldBuilder,
    NestedSchemaBuilder,
    RuleBuilder,
    SchemaBuilder,
    clone_schema,
    create_field,
    create_schema,
    generate_zero_shot_schema,
)
# Import invoice schema
from src.schemas.invoice import INVOICE_SCHEMA
from src.schemas.superbill import SUPERBILL_SCHEMA
from src.schemas.ub04 import UB04_SCHEMA

# Import W-2 schema
from src.schemas.w2 import W2_SCHEMA
from src.schemas.validators import (
    MedicalCodeValidator,
    validate_cpt_code,
    validate_currency,
    validate_date,
    validate_field,
    validate_icd10_code,
    validate_npi,
    validate_phone,
    validate_ssn,
)


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
    # Finance schemas
    "INVOICE_SCHEMA",
    "W2_SCHEMA",
    "FORM_1099_SCHEMA",
    "BANK_STATEMENT_SCHEMA",
    # Generic fallback for unknown documents
    "ENHANCED_GENERIC_SCHEMA",
    # Nested schemas
    "NestedSchema",
    "NestedSchemaRegistry",
    "CMS1500_SERVICE_LINE_SCHEMA",
    "SUPERBILL_PROCEDURE_SCHEMA",
    "UB04_SERVICE_LINE_SCHEMA",
    "UB04_CODE_CODE_SCHEMA",
    "EOB_SERVICE_LINE_SCHEMA",
    "EOB_ADJUSTMENT_SCHEMA",
    "get_nested_schema",
    # Schema builders (for zero-shot flexibility)
    "FieldBuilder",
    "RuleBuilder",
    "SchemaBuilder",
    "NestedSchemaBuilder",
    "create_field",
    "create_schema",
    "clone_schema",
    "generate_zero_shot_schema",
    # Helper functions
    "get_schema",
    "get_all_schemas",
]
