"""
VLM-native layout analysis types for document structure understanding.

Defines TypedDicts for layout analysis, component detection, and adaptive
schema generation - the foundation of VLM-first document extraction.
"""

from typing import Any, Literal, TypedDict


class BoundingBox(TypedDict):
    """Normalized bounding box coordinates (0-1 range)."""
    
    x: float  # Left edge (0 = left side of page, 1 = right side)
    y: float  # Top edge (0 = top of page, 1 = bottom)
    width: float  # Width as fraction of page width
    height: float  # Height as fraction of page height


class Region(TypedDict):
    """Document region with location and description."""
    
    region_id: str
    region_type: Literal["header", "body", "footer", "sidebar", "margin", "table", "form", "signature_area"]
    bounding_box: BoundingBox
    description: str
    confidence: float


class VisualMark(TypedDict):
    """Visual marks like checkboxes, ticks, crosses, circles, etc."""
    
    mark_type: Literal[
        "checkbox_checked", "checkbox_unchecked", "checkbox_partial",
        "tick", "checkmark", "cross", "x_mark",
        "circle", "circled_item",
        "underline", "highlight",
        "arrow", "pointer",
        "stamp", "seal",
        "signature_present", "signature_placeholder",
        "initial", "handwritten_mark",
        "redaction", "obscured"
    ]
    location: BoundingBox
    confidence: float
    state: str | None  # For checkboxes: "checked", "unchecked", "partial"
    description: str  # VLM's natural description of what it sees


class LayoutAnalysis(TypedDict):
    """
    VLM-generated visual layout understanding.
    
    First stage output - pure structural analysis without content extraction.
    """
    
    page_number: int
    layout_type: Literal[
        "form", "table", "tabular_form", "narrative", "mixed",
        "invoice", "receipt", "letter", "report", "claim_form"
    ]
    layout_confidence: float
    
    # Spatial structure
    regions: list[Region]
    column_count: int
    reading_order: str  # VLM's description: "top-to-bottom, left-to-right"
    visual_separators: list[str]  # ["horizontal_lines", "boxes", "borders"]
    
    # Information density
    density_estimate: Literal["sparse", "moderate", "dense", "very_dense"]
    estimated_field_count: int
    has_pre_printed_structure: bool
    has_handwritten_content: bool
    
    # Visual characteristics
    alignment_style: str  # "grid-aligned", "flowing", "mixed"
    spacing_quality: Literal["tight", "normal", "spacious"]
    
    # VLM observations
    vlm_observations: str  # Free-form notes from VLM
    extraction_difficulty: Literal["easy", "moderate", "challenging", "very_difficult"]
    recommended_strategy: str  # VLM's suggestion for extraction approach
    
    analysis_time_ms: int


class TableStructure(TypedDict):
    """VLM-detected table with structure information."""
    
    table_id: str
    location: BoundingBox
    row_count: int
    column_count: int
    has_header_row: bool
    has_row_numbers: bool
    cell_borders_visible: bool
    cell_separation_method: Literal["borders", "spacing", "mixed"]
    column_labels: list[str] | None  # If header row present
    description: str  # VLM's description of table purpose
    complexity: Literal["simple", "moderate", "complex"]
    confidence: float


class FormField(TypedDict):
    """VLM-detected form field with metadata."""
    
    field_id: str
    field_type: Literal[
        "text_input", "text_filled", "text_empty",
        "checkbox", "radio_button", "dropdown",
        "date_field", "numeric_field",
        "signature_area", "multi_line_text",
        "barcode", "qr_code"
    ]
    location: BoundingBox
    label_text: str | None  # The field's label/prompt
    label_location: BoundingBox | None
    is_filled: bool
    visual_marks: list[VisualMark]  # Associated checkmarks, ticks, etc.
    confidence: float
    extraction_hint: str  # VLM's hint for how to extract this


class KeyValuePair(TypedDict):
    """VLM-detected key-value pair (label: value format)."""
    
    pair_id: str
    key_text: str
    key_location: BoundingBox
    value_location: BoundingBox
    separator_type: Literal["colon", "line", "spacing", "box", "none"]
    value_type_hint: Literal["text", "number", "date", "currency", "code", "unknown"]
    confidence: float


class SpecialElement(TypedDict):
    """Special visual elements like logos, stamps, barcodes."""
    
    element_id: str
    element_type: Literal[
        "logo", "stamp", "seal", "barcode", "qr_code",
        "watermark", "letterhead", "footer_text",
        "page_number", "date_stamp"
    ]
    location: BoundingBox
    description: str
    confidence: float


class ComponentMap(TypedDict):
    """
    VLM-generated map of all extractable components.
    
    Second stage output - identifies what can be extracted and how.
    """
    
    page_number: int
    
    # Structural components
    tables: list[TableStructure]
    forms: list[FormField]
    key_value_pairs: list[KeyValuePair]
    
    # Visual marks (human annotations)
    visual_marks: list[VisualMark]
    
    # Special elements
    special_elements: list[SpecialElement]
    
    # Content categorization
    has_tabular_data: bool
    has_form_fields: bool
    has_narrative_text: bool
    has_checkboxes: bool
    has_signatures: bool
    has_handwriting: bool
    
    # Extraction guidance
    component_count: int
    extraction_order: list[str]  # Recommended extraction sequence
    challenging_regions: list[dict[str, Any]]  # Regions that may be hard to extract
    
    # VLM recommendations
    suggested_extraction_strategies: dict[str, str]  # component_type -> strategy
    vlm_notes: str
    
    detection_time_ms: int


class FieldGroup(TypedDict):
    """Logical grouping of related fields."""
    
    group_name: str
    field_names: list[str]
    group_type: Literal["patient_demographics", "insurance", "diagnosis", "billing", "provider", "custom"]
    extraction_strategy: str  # How to extract this group
    component_ids: list[str]  # References to components in ComponentMap


class AdaptiveField(TypedDict):
    """VLM-proposed field for adaptive schema."""
    
    field_name: str
    display_name: str
    field_type: Literal["text", "number", "date", "boolean", "currency", "code", "list", "table", "object"]
    description: str
    source_component_id: str  # Which component this comes from
    location_hint: str
    required: bool
    confidence_threshold: float  # Suggested minimum confidence
    validation_hints: list[str]  # VLM suggestions for validation
    examples: list[str] | None


class AdaptiveSchema(TypedDict):
    """
    VLM-generated adaptive extraction schema.
    
    Third stage output - what to extract based on visual analysis.
    """
    
    schema_id: str
    document_type_description: str  # Descriptive, not prescriptive
    
    # Field definitions
    field_groups: list[FieldGroup]
    fields: list[AdaptiveField]
    total_field_count: int
    
    # Extraction strategy
    overall_strategy: Literal["form_extraction", "table_extraction", "hybrid", "document_parsing", "adaptive"]
    component_strategies: dict[str, str]  # component_id -> specific strategy
    
    # Validation guidance
    suggested_validations: dict[str, list[str]]  # field_name -> validation rules
    cross_field_relationships: list[dict[str, Any]]
    
    # Confidence requirements
    high_confidence_fields: list[str]  # Fields that MUST be accurate
    optional_fields: list[str]  # Fields that can be null
    
    # VLM reasoning
    vlm_reasoning: str  # Why VLM proposed this schema
    schema_confidence: float
    
    generation_time_ms: int


class StructuredExtraction(TypedDict):
    """
    Structure-aware extraction result (enhanced from current PageExtraction).
    
    Fourth stage output - extracted data with full spatial context.
    """
    
    page_number: int
    
    # Extracted data with spatial context
    extracted_fields: dict[str, Any]  # field_name -> value
    field_locations: dict[str, BoundingBox]  # field_name -> location
    field_confidences: dict[str, float]  # field_name -> confidence
    
    # Component-level results
    table_data: list[dict[str, Any]]  # Extracted table rows
    checkbox_states: dict[str, bool]  # checkbox_id -> checked/unchecked
    visual_mark_states: dict[str, str]  # mark_id -> state description
    
    # Extraction metadata
    extraction_strategy_used: str
    component_extraction_order: list[str]
    
    # Validation hints
    spatial_validation_passed: bool  # Values in expected locations?
    component_validation_passed: bool  # Match component types?
    
    # VLM observations
    extraction_notes: str
    uncertain_regions: list[str]  # Regions where VLM wasn't confident
    
    extraction_time_ms: int


# Helper functions for state updates

def create_empty_layout_analysis(page_number: int) -> LayoutAnalysis:
    """Create empty layout analysis for initialization."""
    return LayoutAnalysis(
        page_number=page_number,
        layout_type="mixed",
        layout_confidence=0.0,
        regions=[],
        column_count=1,
        reading_order="top-to-bottom",
        visual_separators=[],
        density_estimate="moderate",
        estimated_field_count=0,
        has_pre_printed_structure=False,
        has_handwritten_content=False,
        alignment_style="unknown",
        spacing_quality="normal",
        vlm_observations="",
        extraction_difficulty="moderate",
        recommended_strategy="",
        analysis_time_ms=0,
    )


def create_empty_component_map(page_number: int) -> ComponentMap:
    """Create empty component map for initialization."""
    return ComponentMap(
        page_number=page_number,
        tables=[],
        forms=[],
        key_value_pairs=[],
        visual_marks=[],
        special_elements=[],
        has_tabular_data=False,
        has_form_fields=False,
        has_narrative_text=False,
        has_checkboxes=False,
        has_signatures=False,
        has_handwriting=False,
        component_count=0,
        extraction_order=[],
        challenging_regions=[],
        suggested_extraction_strategies={},
        vlm_notes="",
        detection_time_ms=0,
    )


def create_empty_adaptive_schema() -> AdaptiveSchema:
    """Create empty adaptive schema for initialization."""
    import secrets
    
    return AdaptiveSchema(
        schema_id=f"adaptive_{secrets.token_hex(8)}",
        document_type_description="unknown",
        field_groups=[],
        fields=[],
        total_field_count=0,
        overall_strategy="adaptive",
        component_strategies={},
        suggested_validations={},
        cross_field_relationships=[],
        high_confidence_fields=[],
        optional_fields=[],
        vlm_reasoning="",
        schema_confidence=0.0,
        generation_time_ms=0,
    )
