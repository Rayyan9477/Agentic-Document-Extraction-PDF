"""
Dynamic schema builder for flexible document extraction.

Provides a fluent API for building custom schemas at runtime,
enabling zero-shot learning and easy extension for new document types.
"""

from collections.abc import Callable
from typing import Any

from src.schemas.base import DocumentSchema, DocumentType, SchemaRegistry
from src.schemas.field_types import CrossFieldRule, FieldDefinition, FieldType, RuleOperator
from src.schemas.nested_schemas import NestedSchema, NestedSchemaRegistry


class FieldBuilder:
    """
    Fluent builder for creating field definitions.

    Example:
        field = (FieldBuilder("patient_name")
            .type(FieldType.NAME)
            .display_name("Patient Name")
            .description("Patient's full name")
            .required()
            .examples(["SMITH, JOHN", "DOE, JANE"])
            .build())
    """

    def __init__(self, name: str) -> None:
        """
        Initialize field builder.

        Args:
            name: Field name (snake_case identifier).
        """
        self._name = name
        self._display_name = name.replace("_", " ").title()
        self._field_type = FieldType.STRING
        self._description = ""
        self._required = False
        self._location_hint: str | None = None
        self._examples: list[str] = []
        self._pattern: str | None = None
        self._allowed_values: list[str] | None = None
        self._min_value: float | None = None
        self._max_value: float | None = None
        self._min_length: int | None = None
        self._max_length: int | None = None
        self._validation_func: Callable[[Any], bool] | None = None
        self._transformation_func: Callable[[Any], Any] | None = None
        self._list_item_type: FieldType | None = None
        self._nested_schema: str | None = None
        self._default_value: Any = None

    def type(self, field_type: FieldType) -> "FieldBuilder":
        """Set field type."""
        self._field_type = field_type
        return self

    def display_name(self, name: str) -> "FieldBuilder":
        """Set display name."""
        self._display_name = name
        return self

    def description(self, desc: str) -> "FieldBuilder":
        """Set field description."""
        self._description = desc
        return self

    def required(self, is_required: bool = True) -> "FieldBuilder":
        """Set required flag."""
        self._required = is_required
        return self

    def optional(self) -> "FieldBuilder":
        """Mark field as optional."""
        self._required = False
        return self

    def location_hint(self, hint: str) -> "FieldBuilder":
        """Set location hint for VLM."""
        self._location_hint = hint
        return self

    def examples(self, examples: list[str]) -> "FieldBuilder":
        """Set example values."""
        self._examples = examples
        return self

    def pattern(self, regex: str) -> "FieldBuilder":
        """Set regex validation pattern."""
        self._pattern = regex
        return self

    def allowed_values(self, values: list[str]) -> "FieldBuilder":
        """Set list of allowed values."""
        self._allowed_values = values
        return self

    def min_value(self, value: float) -> "FieldBuilder":
        """Set minimum numeric value."""
        self._min_value = value
        return self

    def max_value(self, value: float) -> "FieldBuilder":
        """Set maximum numeric value."""
        self._max_value = value
        return self

    def range(self, min_val: float, max_val: float) -> "FieldBuilder":
        """Set numeric value range."""
        self._min_value = min_val
        self._max_value = max_val
        return self

    def min_length(self, length: int) -> "FieldBuilder":
        """Set minimum string length."""
        self._min_length = length
        return self

    def max_length(self, length: int) -> "FieldBuilder":
        """Set maximum string length."""
        self._max_length = length
        return self

    def length_range(self, min_len: int, max_len: int) -> "FieldBuilder":
        """Set string length range."""
        self._min_length = min_len
        self._max_length = max_len
        return self

    def validator(self, func: Callable[[Any], bool]) -> "FieldBuilder":
        """Set custom validation function."""
        self._validation_func = func
        return self

    def transformer(self, func: Callable[[Any], Any]) -> "FieldBuilder":
        """Set value transformation function."""
        self._transformation_func = func
        return self

    def list_of(self, item_type: FieldType) -> "FieldBuilder":
        """Set as list field with item type."""
        self._field_type = FieldType.LIST
        self._list_item_type = item_type
        return self

    def nested(self, schema_name: str) -> "FieldBuilder":
        """Set nested schema for object/list fields."""
        self._nested_schema = schema_name
        return self

    def default(self, value: Any) -> "FieldBuilder":
        """Set default value."""
        self._default_value = value
        return self

    def build(self) -> FieldDefinition:
        """Build the field definition."""
        return FieldDefinition(
            name=self._name,
            display_name=self._display_name,
            field_type=self._field_type,
            description=self._description,
            required=self._required,
            location_hint=self._location_hint,
            examples=self._examples,
            pattern=self._pattern,
            allowed_values=self._allowed_values,
            min_value=self._min_value,
            max_value=self._max_value,
            min_length=self._min_length,
            max_length=self._max_length,
            validation_func=self._validation_func,
            transform_func=self._transformation_func,
            list_item_type=self._list_item_type,
            nested_schema=self._nested_schema,
            default=self._default_value,
        )


class RuleBuilder:
    """
    Fluent builder for creating cross-field validation rules.

    Example:
        rule = (RuleBuilder("date_of_birth", "date_of_service")
            .date_before()
            .error("DOB must be before service date")
            .build())
    """

    def __init__(self, source_field: str, target_field: str) -> None:
        """
        Initialize rule builder.

        Args:
            source_field: Source field name.
            target_field: Target field name.
        """
        self._source_field = source_field
        self._target_field = target_field
        self._operator = RuleOperator.EQUALS
        self._error_message = ""
        self._severity = "error"

    def equals(self) -> "RuleBuilder":
        """Source must equal target."""
        self._operator = RuleOperator.EQUALS
        return self

    def not_equals(self) -> "RuleBuilder":
        """Source must not equal target."""
        self._operator = RuleOperator.NOT_EQUALS
        return self

    def requires(self) -> "RuleBuilder":
        """Source requires target to be present."""
        self._operator = RuleOperator.REQUIRES
        return self

    def requires_if(self) -> "RuleBuilder":
        """If source has value, target is required."""
        self._operator = RuleOperator.REQUIRES_IF
        return self

    def greater_than(self) -> "RuleBuilder":
        """Source must be greater than target."""
        self._operator = RuleOperator.GREATER_THAN
        return self

    def less_than(self) -> "RuleBuilder":
        """Source must be less than target."""
        self._operator = RuleOperator.LESS_THAN
        return self

    def greater_or_equal(self) -> "RuleBuilder":
        """Source must be >= target."""
        self._operator = RuleOperator.GREATER_EQUAL
        return self

    def less_or_equal(self) -> "RuleBuilder":
        """Source must be <= target."""
        self._operator = RuleOperator.LESS_EQUAL
        return self

    def date_before(self) -> "RuleBuilder":
        """Source date must be before target date."""
        self._operator = RuleOperator.DATE_BEFORE
        return self

    def date_after(self) -> "RuleBuilder":
        """Source date must be after target date."""
        self._operator = RuleOperator.DATE_AFTER
        return self

    def error(self, message: str) -> "RuleBuilder":
        """Set error message."""
        self._error_message = message
        return self

    def warning(self) -> "RuleBuilder":
        """Set as warning instead of error."""
        self._severity = "warning"
        return self

    def build(self) -> CrossFieldRule:
        """Build the rule."""
        return CrossFieldRule(
            source_field=self._source_field,
            target_field=self._target_field,
            operator=self._operator,
            error_message=self._error_message,
            severity=self._severity,
        )


class SchemaBuilder:
    """
    Fluent builder for creating document schemas.

    Enables rapid creation of custom schemas for zero-shot learning
    and new document types without modifying core code.

    Example:
        schema = (SchemaBuilder("invoice", DocumentType.CUSTOM)
            .display_name("Invoice Document")
            .description("Standard business invoice")
            .field(FieldBuilder("invoice_number")
                .type(FieldType.STRING)
                .required()
                .pattern(r"^INV-\\d{6}$"))
            .field(FieldBuilder("total_amount")
                .type(FieldType.CURRENCY)
                .required()
                .min_value(0.01))
            .rule(RuleBuilder("invoice_date", "due_date")
                .date_before()
                .error("Invoice date must be before due date"))
            .classification_hints(["INVOICE", "BILL", "AMOUNT DUE"])
            .register()
            .build())
    """

    def __init__(
        self,
        name: str,
        document_type: DocumentType = DocumentType.CUSTOM,
    ) -> None:
        """
        Initialize schema builder.

        Args:
            name: Schema identifier (snake_case).
            document_type: Document type for classification.
        """
        self._name = name
        self._display_name = name.replace("_", " ").title()
        self._document_type = document_type
        self._description = ""
        self._version = "1.0.0"
        self._fields: list[FieldDefinition] = []
        self._rules: list[CrossFieldRule] = []
        self._required_sections: list[str] = []
        self._classification_hints: list[str] = []
        self._extraction_prompt: str | None = None
        self._auto_register = False

    def display_name(self, name: str) -> "SchemaBuilder":
        """Set display name."""
        self._display_name = name
        return self

    def description(self, desc: str) -> "SchemaBuilder":
        """Set schema description."""
        self._description = desc
        return self

    def version(self, version: str) -> "SchemaBuilder":
        """Set schema version."""
        self._version = version
        return self

    def field(self, field_or_builder: FieldDefinition | FieldBuilder) -> "SchemaBuilder":
        """
        Add a field to the schema.

        Args:
            field_or_builder: FieldDefinition or FieldBuilder instance.
        """
        if isinstance(field_or_builder, FieldBuilder):
            self._fields.append(field_or_builder.build())
        else:
            self._fields.append(field_or_builder)
        return self

    def fields(self, fields: list[FieldDefinition | FieldBuilder]) -> "SchemaBuilder":
        """Add multiple fields."""
        for f in fields:
            self.field(f)
        return self

    def rule(self, rule_or_builder: CrossFieldRule | RuleBuilder) -> "SchemaBuilder":
        """
        Add a cross-field validation rule.

        Args:
            rule_or_builder: CrossFieldRule or RuleBuilder instance.
        """
        if isinstance(rule_or_builder, RuleBuilder):
            self._rules.append(rule_or_builder.build())
        else:
            self._rules.append(rule_or_builder)
        return self

    def rules(self, rules: list[CrossFieldRule | RuleBuilder]) -> "SchemaBuilder":
        """Add multiple rules."""
        for r in rules:
            self.rule(r)
        return self

    def required_sections(self, sections: list[str]) -> "SchemaBuilder":
        """Set required document sections."""
        self._required_sections = sections
        return self

    def classification_hints(self, hints: list[str]) -> "SchemaBuilder":
        """Set classification hints for document detection."""
        self._classification_hints = hints
        return self

    def extraction_prompt(self, prompt: str) -> "SchemaBuilder":
        """Set custom extraction prompt template."""
        self._extraction_prompt = prompt
        return self

    def register(self) -> "SchemaBuilder":
        """Mark schema for auto-registration after build."""
        self._auto_register = True
        return self

    def build(self) -> DocumentSchema:
        """Build the document schema."""
        schema = DocumentSchema(
            name=self._name,
            display_name=self._display_name,
            document_type=self._document_type,
            description=self._description,
            version=self._version,
            fields=self._fields,
            cross_field_rules=self._rules,
            required_sections=self._required_sections,
            classification_hints=self._classification_hints,
            extraction_prompt_template=self._extraction_prompt,
        )

        if self._auto_register:
            registry = SchemaRegistry()
            registry.register(schema)

        return schema


class NestedSchemaBuilder:
    """
    Fluent builder for creating nested object schemas.

    Example:
        nested = (NestedSchemaBuilder("line_item")
            .display_name("Invoice Line Item")
            .field(FieldBuilder("description").type(FieldType.STRING).required())
            .field(FieldBuilder("quantity").type(FieldType.INTEGER).required())
            .field(FieldBuilder("unit_price").type(FieldType.CURRENCY).required())
            .register()
            .build())
    """

    def __init__(self, name: str) -> None:
        """
        Initialize nested schema builder.

        Args:
            name: Schema identifier (referenced by nested_schema in fields).
        """
        self._name = name
        self._display_name = name.replace("_", " ").title()
        self._description = ""
        self._fields: list[FieldDefinition] = []
        self._auto_register = False

    def display_name(self, name: str) -> "NestedSchemaBuilder":
        """Set display name."""
        self._display_name = name
        return self

    def description(self, desc: str) -> "NestedSchemaBuilder":
        """Set description."""
        self._description = desc
        return self

    def field(self, field_or_builder: FieldDefinition | FieldBuilder) -> "NestedSchemaBuilder":
        """Add a field to the nested schema."""
        if isinstance(field_or_builder, FieldBuilder):
            self._fields.append(field_or_builder.build())
        else:
            self._fields.append(field_or_builder)
        return self

    def fields(self, fields: list[FieldDefinition | FieldBuilder]) -> "NestedSchemaBuilder":
        """Add multiple fields."""
        for f in fields:
            self.field(f)
        return self

    def register(self) -> "NestedSchemaBuilder":
        """Mark for auto-registration after build."""
        self._auto_register = True
        return self

    def build(self) -> NestedSchema:
        """Build the nested schema."""
        schema = NestedSchema(
            name=self._name,
            display_name=self._display_name,
            description=self._description,
            fields=self._fields,
        )

        if self._auto_register:
            registry = NestedSchemaRegistry()
            registry.register(schema)

        return schema


# =============================================================================
# Convenience Functions for Quick Schema Creation
# =============================================================================


def create_field(
    name: str,
    field_type: FieldType,
    required: bool = False,
    description: str = "",
    **kwargs: Any,
) -> FieldDefinition:
    """
    Quick helper to create a field definition.

    Args:
        name: Field name.
        field_type: Field type.
        required: Whether field is required.
        description: Field description.
        **kwargs: Additional field attributes.

    Returns:
        FieldDefinition instance.
    """
    return FieldDefinition(
        name=name,
        display_name=name.replace("_", " ").title(),
        field_type=field_type,
        description=description,
        required=required,
        **kwargs,
    )


def create_schema(
    name: str,
    document_type: DocumentType,
    fields: list[FieldDefinition],
    display_name: str | None = None,
    description: str = "",
    rules: list[CrossFieldRule] | None = None,
    classification_hints: list[str] | None = None,
    register: bool = True,
) -> DocumentSchema:
    """
    Quick helper to create and register a document schema.

    Args:
        name: Schema name.
        document_type: Document type.
        fields: List of field definitions.
        display_name: Human-readable name.
        description: Schema description.
        rules: Cross-field validation rules.
        classification_hints: Document classification hints.
        register: Whether to auto-register the schema.

    Returns:
        DocumentSchema instance.
    """
    schema = DocumentSchema(
        name=name,
        display_name=display_name or name.replace("_", " ").title(),
        document_type=document_type,
        description=description,
        fields=fields,
        cross_field_rules=rules or [],
        classification_hints=classification_hints or [],
    )

    if register:
        registry = SchemaRegistry()
        registry.register(schema)

    return schema


def clone_schema(
    source_name: str,
    new_name: str,
    new_document_type: DocumentType | None = None,
    additional_fields: list[FieldDefinition] | None = None,
    remove_fields: list[str] | None = None,
    register: bool = True,
) -> DocumentSchema:
    """
    Clone an existing schema with modifications.

    Args:
        source_name: Name of schema to clone.
        new_name: Name for new schema.
        new_document_type: Optional new document type.
        additional_fields: Fields to add.
        remove_fields: Field names to remove.
        register: Whether to auto-register.

    Returns:
        New DocumentSchema instance.

    Raises:
        ValueError: If source schema not found.
    """
    registry = SchemaRegistry()
    source = registry.get(source_name)

    if source is None:
        raise ValueError(f"Source schema not found: {source_name}")

    # Clone fields, removing specified ones
    remove_set = set(remove_fields or [])
    fields = [f for f in source.fields if f.name not in remove_set]

    # Add additional fields
    if additional_fields:
        fields.extend(additional_fields)

    schema = DocumentSchema(
        name=new_name,
        display_name=f"{source.display_name} (Custom)",
        document_type=new_document_type or source.document_type,
        description=f"Customized version of {source.display_name}",
        version=source.version,
        fields=fields,
        cross_field_rules=source.cross_field_rules.copy(),
        required_sections=source.required_sections.copy(),
        classification_hints=source.classification_hints.copy(),
    )

    if register:
        registry.register(schema)

    return schema


def generate_zero_shot_schema(
    name: str,
    field_names: list[str],
    document_type: DocumentType = DocumentType.CUSTOM,
    infer_types: bool = True,
    register: bool = True,
) -> DocumentSchema:
    """
    Generate a basic schema from a list of field names.

    Useful for quick zero-shot extraction experiments.

    Args:
        name: Schema name.
        field_names: List of field names to extract.
        document_type: Document type.
        infer_types: Attempt to infer field types from names.
        register: Whether to auto-register.

    Returns:
        DocumentSchema with inferred field definitions.
    """
    fields: list[FieldDefinition] = []

    for field_name in field_names:
        field_type = FieldType.STRING

        if infer_types:
            # Infer type from common naming patterns
            lower_name = field_name.lower()

            if "date" in lower_name or "dob" in lower_name:
                field_type = FieldType.DATE
            elif (
                "amount" in lower_name
                or "total" in lower_name
                or "fee" in lower_name
                or "charge" in lower_name
            ):
                field_type = FieldType.CURRENCY
            elif "phone" in lower_name or "fax" in lower_name:
                field_type = FieldType.PHONE
            elif "email" in lower_name:
                field_type = FieldType.EMAIL
            elif "zip" in lower_name or "postal" in lower_name:
                field_type = FieldType.ZIP_CODE
            elif "state" in lower_name and "address" not in lower_name:
                field_type = FieldType.STATE
            elif "address" in lower_name:
                field_type = FieldType.ADDRESS
            elif "name" in lower_name:
                field_type = FieldType.NAME
            elif "npi" in lower_name:
                field_type = FieldType.NPI
            elif "ssn" in lower_name or "social" in lower_name:
                field_type = FieldType.SSN
            elif "cpt" in lower_name or "procedure" in lower_name:
                field_type = FieldType.CPT_CODE
            elif "icd" in lower_name or "diagnosis" in lower_name:
                field_type = FieldType.ICD10_CODE
            elif "count" in lower_name or "quantity" in lower_name or "units" in lower_name:
                field_type = FieldType.INTEGER

        fields.append(
            FieldDefinition(
                name=field_name,
                display_name=field_name.replace("_", " ").title(),
                field_type=field_type,
                description=f"Extracted {field_name.replace('_', ' ')} value",
                required=False,
            )
        )

    return create_schema(
        name=name,
        document_type=document_type,
        fields=fields,
        description=f"Zero-shot schema for {name} extraction",
        register=register,
    )
