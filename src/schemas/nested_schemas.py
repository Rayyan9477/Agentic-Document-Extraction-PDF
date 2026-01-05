"""
Nested schema definitions for complex field types.

Defines the structure of nested objects used within main document schemas,
such as service lines, procedures, and adjustments.
"""

from dataclasses import dataclass, field
from typing import Any

from src.schemas.field_types import FieldDefinition, FieldType


@dataclass
class NestedSchema:
    """
    Schema definition for nested objects within documents.

    Attributes:
        name: Schema identifier (matches nested_schema in FieldDefinition).
        display_name: Human-readable name.
        description: Schema description.
        fields: List of field definitions.
    """

    name: str
    display_name: str
    description: str
    fields: list[FieldDefinition] = field(default_factory=list)

    _field_map: dict[str, FieldDefinition] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build field lookup map."""
        self._field_map = {f.name: f for f in self.fields}

    def get_field(self, name: str) -> FieldDefinition | None:
        """Get field definition by name."""
        return self._field_map.get(name)

    def get_required_fields(self) -> list[FieldDefinition]:
        """Get list of required fields."""
        return [f for f in self.fields if f.required]

    def validate(self, value: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a nested object value.

        Args:
            value: Dictionary of field values.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: list[str] = []

        # Check required fields
        for field_def in self.get_required_fields():
            if field_def.name not in value or value[field_def.name] is None:
                errors.append(f"Required field missing: {field_def.display_name}")

        # Validate each field
        for name, val in value.items():
            field_def = self.get_field(name)
            if field_def:
                is_valid, message = field_def.validate(val)
                if not is_valid and message:
                    errors.append(message)

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
        }

    def generate_prompt_description(self) -> str:
        """Generate prompt description for VLM extraction."""
        field_descs = []
        for f in self.fields:
            req = " [REQUIRED]" if f.required else ""
            field_descs.append(f"    - {f.name}: {f.description}{req}")

        return f"{self.display_name}:\n" + "\n".join(field_descs)


# =============================================================================
# CMS-1500 Service Line Schema
# =============================================================================

CMS1500_SERVICE_LINE_SCHEMA = NestedSchema(
    name="cms1500_service_line",
    display_name="CMS-1500 Service Line",
    description="Individual service line from CMS-1500 form (Box 24)",
    fields=[
        FieldDefinition(
            name="line_number",
            display_name="Line Number",
            field_type=FieldType.INTEGER,
            description="Service line number (1-6)",
            required=True,
            min_value=1,
            max_value=6,
        ),
        FieldDefinition(
            name="date_from",
            display_name="Date of Service From (24A)",
            field_type=FieldType.DATE,
            description="Start date of service",
            required=True,
            location_hint="Box 24A - From",
        ),
        FieldDefinition(
            name="date_to",
            display_name="Date of Service To (24A)",
            field_type=FieldType.DATE,
            description="End date of service (same as from for single day)",
            required=False,
            location_hint="Box 24A - To",
        ),
        FieldDefinition(
            name="place_of_service",
            display_name="Place of Service (24B)",
            field_type=FieldType.STRING,
            description="Place of service code (2 digits)",
            required=True,
            location_hint="Box 24B",
            pattern=r"^\d{2}$",
            examples=["11", "21", "22", "23"],
        ),
        FieldDefinition(
            name="emg",
            display_name="EMG (24C)",
            field_type=FieldType.BOOLEAN,
            description="Emergency indicator",
            required=False,
            location_hint="Box 24C",
        ),
        FieldDefinition(
            name="cpt_hcpcs",
            display_name="CPT/HCPCS Code (24D)",
            field_type=FieldType.CPT_CODE,
            description="Procedure code",
            required=True,
            location_hint="Box 24D",
            examples=["99213", "99214", "99215"],
        ),
        FieldDefinition(
            name="modifier_1",
            display_name="Modifier 1 (24D)",
            field_type=FieldType.STRING,
            description="First procedure modifier",
            required=False,
            location_hint="Box 24D - Modifier 1",
            pattern=r"^[A-Z0-9]{2}$",
        ),
        FieldDefinition(
            name="modifier_2",
            display_name="Modifier 2 (24D)",
            field_type=FieldType.STRING,
            description="Second procedure modifier",
            required=False,
            location_hint="Box 24D - Modifier 2",
            pattern=r"^[A-Z0-9]{2}$",
        ),
        FieldDefinition(
            name="modifier_3",
            display_name="Modifier 3 (24D)",
            field_type=FieldType.STRING,
            description="Third procedure modifier",
            required=False,
            location_hint="Box 24D - Modifier 3",
            pattern=r"^[A-Z0-9]{2}$",
        ),
        FieldDefinition(
            name="modifier_4",
            display_name="Modifier 4 (24D)",
            field_type=FieldType.STRING,
            description="Fourth procedure modifier",
            required=False,
            location_hint="Box 24D - Modifier 4",
            pattern=r"^[A-Z0-9]{2}$",
        ),
        FieldDefinition(
            name="diagnosis_pointer",
            display_name="Diagnosis Pointer (24E)",
            field_type=FieldType.STRING,
            description="Letters A-L pointing to diagnosis codes in Box 21",
            required=True,
            location_hint="Box 24E",
            pattern=r"^[A-L,\s]+$",
            examples=["A", "A,B", "A B C"],
        ),
        FieldDefinition(
            name="charges",
            display_name="Charges (24F)",
            field_type=FieldType.CURRENCY,
            description="Line item charges",
            required=True,
            location_hint="Box 24F",
            min_value=0.01,
        ),
        FieldDefinition(
            name="units",
            display_name="Days or Units (24G)",
            field_type=FieldType.INTEGER,
            description="Number of days or units",
            required=True,
            location_hint="Box 24G",
            min_value=1,
        ),
        FieldDefinition(
            name="epsdt_family_plan",
            display_name="EPSDT/Family Plan (24H)",
            field_type=FieldType.STRING,
            description="EPSDT or family planning indicator",
            required=False,
            location_hint="Box 24H",
        ),
        FieldDefinition(
            name="id_qualifier",
            display_name="ID Qualifier (24I)",
            field_type=FieldType.STRING,
            description="Rendering provider ID qualifier",
            required=False,
            location_hint="Box 24I",
        ),
        FieldDefinition(
            name="rendering_provider_id",
            display_name="Rendering Provider ID (24J)",
            field_type=FieldType.NPI,
            description="Rendering provider NPI or legacy ID",
            required=False,
            location_hint="Box 24J",
        ),
    ],
)


# =============================================================================
# Superbill Procedure Schema
# =============================================================================

SUPERBILL_PROCEDURE_SCHEMA = NestedSchema(
    name="superbill_procedure",
    display_name="Superbill Procedure/Service",
    description="Individual procedure or service from a medical superbill",
    fields=[
        FieldDefinition(
            name="cpt_code",
            display_name="CPT Code",
            field_type=FieldType.CPT_CODE,
            description="CPT procedure code",
            required=True,
            examples=["99213", "99214", "99215", "99203"],
        ),
        FieldDefinition(
            name="description",
            display_name="Procedure Description",
            field_type=FieldType.STRING,
            description="Description of the procedure/service",
            required=False,
            examples=["Office Visit - Established", "New Patient - Moderate"],
        ),
        FieldDefinition(
            name="modifier",
            display_name="Modifier",
            field_type=FieldType.STRING,
            description="CPT modifier code",
            required=False,
            pattern=r"^[A-Z0-9]{2}$",
        ),
        FieldDefinition(
            name="diagnosis_codes",
            display_name="Linked Diagnosis Codes",
            field_type=FieldType.LIST,
            list_item_type=FieldType.ICD10_CODE,
            description="ICD-10 codes linked to this procedure",
            required=False,
        ),
        FieldDefinition(
            name="units",
            display_name="Units",
            field_type=FieldType.INTEGER,
            description="Number of units/services",
            required=False,
            min_value=1,
        ),
        FieldDefinition(
            name="fee",
            display_name="Fee/Charge",
            field_type=FieldType.CURRENCY,
            description="Charge amount for this procedure",
            required=True,
            min_value=0,
        ),
        FieldDefinition(
            name="is_selected",
            display_name="Selected/Checked",
            field_type=FieldType.BOOLEAN,
            description="Whether this procedure was marked/selected on the form",
            required=False,
        ),
        FieldDefinition(
            name="category",
            display_name="Category",
            field_type=FieldType.STRING,
            description="Procedure category on the superbill",
            required=False,
            examples=["Office Visits", "Preventive", "Lab", "Procedures"],
        ),
    ],
)


# =============================================================================
# UB-04 Service Line Schema
# =============================================================================

UB04_SERVICE_LINE_SCHEMA = NestedSchema(
    name="ub04_service_line",
    display_name="UB-04 Service Line",
    description="Revenue code service line from UB-04 institutional claim",
    fields=[
        FieldDefinition(
            name="line_number",
            display_name="Line Number",
            field_type=FieldType.INTEGER,
            description="Service line number",
            required=True,
            min_value=1,
            max_value=22,
        ),
        FieldDefinition(
            name="revenue_code",
            display_name="Revenue Code (FL 42)",
            field_type=FieldType.STRING,
            description="4-digit revenue code",
            required=True,
            location_hint="Form Locator 42",
            pattern=r"^\d{4}$",
            examples=["0250", "0300", "0320", "0450"],
        ),
        FieldDefinition(
            name="revenue_description",
            display_name="Revenue Description (FL 43)",
            field_type=FieldType.STRING,
            description="Description of the revenue code/service",
            required=False,
            location_hint="Form Locator 43",
        ),
        FieldDefinition(
            name="hcpcs_rate",
            display_name="HCPCS/Rate/HIPPS Code (FL 44)",
            field_type=FieldType.STRING,
            description="HCPCS code, rate, or HIPPS code",
            required=False,
            location_hint="Form Locator 44",
        ),
        FieldDefinition(
            name="service_date",
            display_name="Service Date (FL 45)",
            field_type=FieldType.DATE,
            description="Date of service for this line",
            required=False,
            location_hint="Form Locator 45",
        ),
        FieldDefinition(
            name="units",
            display_name="Service Units (FL 46)",
            field_type=FieldType.INTEGER,
            description="Number of units/days/visits",
            required=True,
            location_hint="Form Locator 46",
            min_value=1,
        ),
        FieldDefinition(
            name="total_charges",
            display_name="Total Charges (FL 47)",
            field_type=FieldType.CURRENCY,
            description="Total charges for this line",
            required=True,
            location_hint="Form Locator 47",
            min_value=0,
        ),
        FieldDefinition(
            name="non_covered_charges",
            display_name="Non-Covered Charges (FL 48)",
            field_type=FieldType.CURRENCY,
            description="Non-covered charges for this line",
            required=False,
            location_hint="Form Locator 48",
            min_value=0,
        ),
    ],
)


# =============================================================================
# UB-04 Code-Code Field Schema (FL 81)
# =============================================================================

UB04_CODE_CODE_SCHEMA = NestedSchema(
    name="ub04_code_code",
    display_name="UB-04 Code-Code Field",
    description="Code-code qualifier and value pairs from FL 81",
    fields=[
        FieldDefinition(
            name="qualifier",
            display_name="Code Qualifier",
            field_type=FieldType.STRING,
            description="Two-character code qualifier",
            required=True,
            pattern=r"^[A-Z0-9]{2}$",
            examples=["A1", "A2", "A3", "B1"],
        ),
        FieldDefinition(
            name="code",
            display_name="Code Value",
            field_type=FieldType.STRING,
            description="Code value associated with qualifier",
            required=True,
        ),
    ],
)


# =============================================================================
# EOB Service Line Schema
# =============================================================================

EOB_SERVICE_LINE_SCHEMA = NestedSchema(
    name="eob_service_line",
    display_name="EOB Service Line",
    description="Individual service line from Explanation of Benefits",
    fields=[
        FieldDefinition(
            name="line_number",
            display_name="Line Number",
            field_type=FieldType.INTEGER,
            description="Service line number",
            required=False,
            min_value=1,
        ),
        FieldDefinition(
            name="date_of_service",
            display_name="Date of Service",
            field_type=FieldType.DATE,
            description="Date service was provided",
            required=True,
        ),
        FieldDefinition(
            name="cpt_code",
            display_name="Procedure/CPT Code",
            field_type=FieldType.CPT_CODE,
            description="CPT or HCPCS procedure code",
            required=False,
        ),
        FieldDefinition(
            name="description",
            display_name="Service Description",
            field_type=FieldType.STRING,
            description="Description of service provided",
            required=True,
        ),
        FieldDefinition(
            name="billed_amount",
            display_name="Amount Billed",
            field_type=FieldType.CURRENCY,
            description="Amount billed by provider",
            required=True,
            min_value=0,
        ),
        FieldDefinition(
            name="allowed_amount",
            display_name="Allowed Amount",
            field_type=FieldType.CURRENCY,
            description="Amount allowed by insurance",
            required=True,
            min_value=0,
        ),
        FieldDefinition(
            name="discount_amount",
            display_name="Discount/Savings",
            field_type=FieldType.CURRENCY,
            description="Network discount amount",
            required=False,
            min_value=0,
        ),
        FieldDefinition(
            name="deductible_amount",
            display_name="Deductible Applied",
            field_type=FieldType.CURRENCY,
            description="Amount applied to deductible",
            required=False,
            min_value=0,
        ),
        FieldDefinition(
            name="copay_amount",
            display_name="Copay",
            field_type=FieldType.CURRENCY,
            description="Copay amount",
            required=False,
            min_value=0,
        ),
        FieldDefinition(
            name="coinsurance_amount",
            display_name="Coinsurance",
            field_type=FieldType.CURRENCY,
            description="Coinsurance amount",
            required=False,
            min_value=0,
        ),
        FieldDefinition(
            name="plan_paid",
            display_name="Plan Paid",
            field_type=FieldType.CURRENCY,
            description="Amount paid by insurance plan",
            required=True,
            min_value=0,
        ),
        FieldDefinition(
            name="patient_responsibility",
            display_name="Patient Responsibility",
            field_type=FieldType.CURRENCY,
            description="Amount patient owes for this line",
            required=False,
            min_value=0,
        ),
        FieldDefinition(
            name="remark_code",
            display_name="Remark Code",
            field_type=FieldType.STRING,
            description="Claim adjustment or remark code",
            required=False,
        ),
        FieldDefinition(
            name="status",
            display_name="Line Status",
            field_type=FieldType.STRING,
            description="Status of this line (Paid, Denied, etc.)",
            required=False,
            allowed_values=["Paid", "Denied", "Partially Paid", "Pending", "Adjusted"],
        ),
    ],
)


# =============================================================================
# EOB Adjustment Schema
# =============================================================================

EOB_ADJUSTMENT_SCHEMA = NestedSchema(
    name="eob_adjustment",
    display_name="EOB Adjustment",
    description="Claim adjustment from Explanation of Benefits",
    fields=[
        FieldDefinition(
            name="adjustment_code",
            display_name="Adjustment Code",
            field_type=FieldType.STRING,
            description="CARC/RARC code or proprietary adjustment code",
            required=False,
            examples=["CO-45", "PR-1", "OA-23"],
        ),
        FieldDefinition(
            name="adjustment_reason",
            display_name="Adjustment Reason",
            field_type=FieldType.STRING,
            description="Description of the adjustment reason",
            required=True,
        ),
        FieldDefinition(
            name="adjustment_amount",
            display_name="Adjustment Amount",
            field_type=FieldType.CURRENCY,
            description="Dollar amount of adjustment",
            required=True,
        ),
        FieldDefinition(
            name="adjustment_type",
            display_name="Adjustment Type",
            field_type=FieldType.STRING,
            description="Type of adjustment (Contractual, Patient Responsibility, etc.)",
            required=False,
            allowed_values=[
                "Contractual Obligation",
                "Patient Responsibility",
                "Other Adjustment",
                "Prior Payment",
                "Correction",
            ],
        ),
        FieldDefinition(
            name="applies_to_line",
            display_name="Applies to Line",
            field_type=FieldType.INTEGER,
            description="Service line number this adjustment applies to",
            required=False,
        ),
    ],
)


# =============================================================================
# Nested Schema Registry
# =============================================================================


class NestedSchemaRegistry:
    """
    Registry for nested schemas.

    Provides centralized management and lookup for nested object schemas.
    """

    _instance: "NestedSchemaRegistry | None" = None
    _schemas: dict[str, NestedSchema]

    def __new__(cls) -> "NestedSchemaRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._schemas = {}
        return cls._instance

    def register(self, schema: NestedSchema) -> None:
        """Register a nested schema."""
        self._schemas[schema.name] = schema

    def get(self, name: str) -> NestedSchema | None:
        """Get nested schema by name."""
        return self._schemas.get(name)

    def list_schemas(self) -> list[NestedSchema]:
        """Get all registered nested schemas."""
        return list(self._schemas.values())

    def list_schema_names(self) -> list[str]:
        """Get list of registered schema names."""
        return list(self._schemas.keys())

    def has_schema(self, name: str) -> bool:
        """Check if schema is registered."""
        return name in self._schemas

    def validate_nested_value(
        self,
        schema_name: str,
        value: dict[str, Any] | list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """
        Validate a nested value against its schema.

        Args:
            schema_name: Name of the nested schema.
            value: Value to validate (single object or list).

        Returns:
            Tuple of (is_valid, error_messages).
        """
        schema = self.get(schema_name)
        if schema is None:
            return False, [f"Unknown nested schema: {schema_name}"]

        all_errors: list[str] = []

        if isinstance(value, list):
            for i, item in enumerate(value):
                is_valid, errors = schema.validate(item)
                for error in errors:
                    all_errors.append(f"Item {i + 1}: {error}")
        else:
            _, errors = schema.validate(value)
            all_errors.extend(errors)

        return len(all_errors) == 0, all_errors


def register_nested_schemas() -> None:
    """Register all nested schemas with the global registry."""
    registry = NestedSchemaRegistry()
    registry.register(CMS1500_SERVICE_LINE_SCHEMA)
    registry.register(SUPERBILL_PROCEDURE_SCHEMA)
    registry.register(UB04_SERVICE_LINE_SCHEMA)
    registry.register(UB04_CODE_CODE_SCHEMA)
    registry.register(EOB_SERVICE_LINE_SCHEMA)
    registry.register(EOB_ADJUSTMENT_SCHEMA)


# Auto-register on import
register_nested_schemas()


# =============================================================================
# Convenience function for getting nested schemas
# =============================================================================


def get_nested_schema(name: str) -> NestedSchema | None:
    """
    Get a nested schema by name.

    Args:
        name: Nested schema name.

    Returns:
        NestedSchema or None if not found.
    """
    registry = NestedSchemaRegistry()
    return registry.get(name)
