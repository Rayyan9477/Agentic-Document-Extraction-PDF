"""
SchemaGeneratorAgent - VLM-driven adaptive schema generation.

Third stage of VLM-first extraction pipeline. Generates extraction schema
dynamically based on layout analysis and component detection - no hardcoded
schemas needed.
"""

import secrets
import time
from typing import Any

from src.agents.base import AgentError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.config import get_settings
from src.pipeline.state import ExtractionState
from src.prompts.grounding_rules import build_grounded_system_prompt


class SchemaGenerationError(AgentError):
    """Error during schema generation."""


class SchemaGeneratorAgent(BaseAgent):
    """
    VLM-powered adaptive schema generator.
    
    Analyzes document structure and generates extraction schema dynamically.
    No hardcoded document types or field definitions required.
    
    Key capabilities:
    - Proposes fields based on detected components
    - Suggests field types and validation rules
    - Creates logical field groupings
    - Recommends extraction strategies per component
    - Adapts to any document type
    
    Uses layout_analyses and component_maps from previous stages.
    
    VLM Utilization: ~20% of total pipeline capability
    Critical for: Zero-shot documents, adaptive extraction, unknown formats
    """
    
    def __init__(self, client=None):
        """Initialize schema generator agent."""
        super().__init__(name="schema_generator", client=client)
        self._settings = get_settings()
    
    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Generate adaptive schema based on document analysis.
        
        Args:
            state: Current extraction state with layout_analyses and component_maps.
        
        Returns:
            State updated with adaptive_schema.
        """
        start_time = time.time()
        
        try:
            self._logger.info(
                "schema_generation_started",
                processing_id=state.get("processing_id"),
            )
            
            layout_analyses = state.get("layout_analyses", [])
            component_maps = state.get("component_maps", [])
            
            if not layout_analyses or not component_maps:
                raise SchemaGenerationError(
                    "Missing layout analyses or component maps for schema generation",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            # Use first page for schema generation (assume consistent structure)
            # For multi-page documents, could aggregate across pages
            first_page_layout = layout_analyses[0] if layout_analyses else None
            first_page_components = component_maps[0] if component_maps else None
            
            # Get page image for VLM analysis
            page_images = state.get("page_images", [])
            if not page_images:
                raise SchemaGenerationError(
                    "No page images available for schema generation",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            first_page_image = page_images[0]
            image_data_uri = first_page_image.get("data_uri")
            
            # Generate adaptive schema
            adaptive_schema = self._generate_schema(
                image_data_uri=image_data_uri,
                layout=first_page_layout,
                components=first_page_components,
                total_pages=len(page_images),
            )
            
            # Update state
            state["adaptive_schema"] = adaptive_schema
            state["total_vlm_calls"] = state.get("total_vlm_calls", 0) + 1
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            state["total_processing_time_ms"] = state.get("total_processing_time_ms", 0) + elapsed_ms
            
            self._logger.info(
                "schema_generation_completed",
                field_count=adaptive_schema.get("total_field_count", 0),
                field_groups=len(adaptive_schema.get("field_groups", [])),
                confidence=adaptive_schema.get("schema_confidence", 0.0),
                time_ms=elapsed_ms,
            )
            
            return state
        
        except Exception as e:
            self._logger.error(
                "schema_generation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SchemaGenerationError(
                f"Schema generation failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e
    
    def _generate_schema(
        self,
        image_data_uri: str,
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
        total_pages: int,
    ) -> dict[str, Any]:
        """
        Generate adaptive schema using VLM analysis.
        
        Args:
            image_data_uri: Data URI of first page image.
            layout: Layout analysis from LayoutAgent.
            components: Component map from ComponentDetectorAgent.
            total_pages: Total page count.
        
        Returns:
            AdaptiveSchema dict with field definitions and strategies.
        """
        start_time = time.time()
        
        system_prompt = build_grounded_system_prompt(
            additional_context=(
                "You are generating an extraction schema based on document structure. "
                "Propose fields, types, and strategies dynamically - do NOT use predefined templates. "
                "Analyze what's actually present and suggest how to extract it."
            ),
            include_forbidden=True,
            include_confidence_scale=True,
        )
        
        user_prompt = self._build_schema_generation_prompt(
            layout=layout,
            components=components,
            total_pages=total_pages,
        )
        
        # Retry with backoff
        retry_config = RetryConfig(
            max_retries=self._settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=self._settings.agent.max_retry_delay_ms,
        )
        
        def make_vlm_call() -> dict[str, Any]:
            return self.send_vision_request_with_json(
                image_data=image_data_uri,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Slightly higher for creative schema generation
                max_tokens=5000,  # Schema generation needs detailed response
            )
        
        try:
            result = retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "schema_generation_retry",
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Parse and structure response
            return self._parse_schema_response(result, elapsed_ms)
        
        except Exception as e:
            self._logger.error(
                "schema_generation_vlm_failed",
                error=str(e),
            )
            # Return minimal fallback schema
            return self._create_fallback_schema(layout, components)
    
    def _build_schema_generation_prompt(
        self,
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
        total_pages: int,
    ) -> str:
        """Build schema generation prompt with context."""
        
        # Build context sections
        layout_context = ""
        if layout:
            layout_context = f"""
## Layout Analysis Context

**Document Structure:**
- Layout Type: {layout.get('layout_type', 'unknown')}
- Columns: {layout.get('column_count', 1)}
- Density: {layout.get('density_estimate', 'moderate')}
- Estimated Fields: {layout.get('estimated_field_count', 0)}
- Reading Order: {layout.get('reading_order', 'unknown')}
- Pre-printed Form: {layout.get('has_pre_printed_structure', False)}
- Handwriting Present: {layout.get('has_handwritten_content', False)}

**Visual Marks Found:** {len(layout.get('visual_marks', []))}
- These may indicate checkbox selections, signatures, stamps, etc.

**VLM Observations:**
{layout.get('vlm_observations', 'None')}

**Extraction Difficulty:** {layout.get('extraction_difficulty', 'moderate')}
"""
        
        component_context = ""
        if components:
            tables = components.get('tables', [])
            forms = components.get('forms', [])
            kv_pairs = components.get('key_value_pairs', [])
            visual_marks = components.get('visual_marks', [])
            
            component_context = f"""
## Component Detection Context

**Tables:** {len(tables)}
"""
            for idx, table in enumerate(tables[:3], 1):  # Show first 3
                component_context += f"""
  Table {idx}: {table.get('row_count', 0)} rows × {table.get('column_count', 0)} cols
    - {table.get('description', 'no description')}
    - Column Labels: {table.get('column_labels', [])}
"""
            
            component_context += f"""
**Form Fields:** {len(forms)}
"""
            # Group by field type
            field_types = {}
            for form in forms:
                ftype = form.get('field_type', 'unknown')
                field_types[ftype] = field_types.get(ftype, 0) + 1
            
            for ftype, count in sorted(field_types.items()):
                component_context += f"  - {ftype}: {count}\n"
            
            # Show sample form fields
            component_context += "\n  Sample Fields:\n"
            for form in forms[:10]:  # First 10
                label = form.get('label_text', 'unlabeled')
                ftype = form.get('field_type', 'unknown')
                component_context += f"    - \"{label}\" ({ftype})\n"
            
            component_context += f"""
**Key-Value Pairs:** {len(kv_pairs)}
"""
            for kv in kv_pairs[:10]:  # First 10
                key = kv.get('key_text', 'unknown')
                vtype = kv.get('value_type_hint', 'text')
                component_context += f"  - \"{key}\" → ({vtype})\n"
            
            component_context += f"""
**Visual Marks:** {len(visual_marks)}
  (Checkboxes, ticks, crosses, stamps, signatures, etc.)

**Content Summary:**
  - Tabular Data: {components.get('has_tabular_data', False)}
  - Form Fields: {components.get('has_form_fields', False)}
  - Checkboxes: {components.get('has_checkboxes', False)}
  - Signatures: {components.get('has_signatures', False)}
  - Handwriting: {components.get('has_handwriting', False)}

**Suggested Extraction Strategies:**
"""
            strategies = components.get('suggested_extraction_strategies', {})
            for comp_type, strategy in strategies.items():
                component_context += f"  - {comp_type}: {strategy}\n"
        
        return f"""# TASK: Generate Adaptive Extraction Schema

You have analyzed this document's layout and components. Now propose an extraction schema.

**CRITICAL: Be adaptive, not prescriptive. Design schema for THIS document, not a template.**

{layout_context}

{component_context}

## Document Info
- Total Pages: {total_pages}
- Multi-page extraction may require aggregation strategies

---

## Your Task: Propose Extraction Schema

### Step 1: Document Type Description

Provide a **descriptive** (not prescriptive) document type:
- What is this document? (e.g., "Medical billing superbill with patient info and service line items")
- What's its purpose? (e.g., "To bill insurance for medical services rendered")
- What makes it unique? (e.g., "Contains both patient demographics and itemized CPT codes")

**NOT**: "CMS-1500" (too prescriptive)
**YES**: "Health insurance claim form with provider and patient sections" (descriptive)

### Step 2: Identify Field Groups

Group related fields logically:

```json
{{
  "field_groups": [
    {{
      "group_name": "patient_demographics",
      "field_names": ["patient_name", "patient_dob", "patient_address", "..."],
      "group_type": "patient_demographics|insurance|diagnosis|billing|provider|custom",
      "extraction_strategy": "How to extract this group (form_fields, table_row, key_value_pairs)",
      "component_ids": ["field_1", "field_2", "kv_3"]
    }}
  ]
}}
```

**Group Types:**
- `patient_demographics` - Patient name, DOB, address, contact
- `insurance` - Insurance company, policy numbers, group numbers
- `diagnosis` - ICD codes, diagnosis descriptions
- `billing` - Service dates, CPT codes, amounts, totals
- `provider` - Provider name, NPI, facility info
- `custom` - Any other logical grouping

### Step 3: Define Fields

For each field, provide:

```json
{{
  "fields": [
    {{
      "field_name": "patient_name",
      "display_name": "Patient Name",
      "field_type": "text|number|date|boolean|currency|code|list|table|object",
      "description": "Full name of the patient",
      "source_component_id": "field_patient_name",
      "location_hint": "Top-left section, labeled 'Patient Name'",
      "required": true,
      "confidence_threshold": 0.85,
      "validation_hints": ["Should be non-empty", "Format: Last, First MI"],
      "examples": ["Smith, John A", "Doe, Jane"]
    }}
  ]
}}
```

**Field Types:**
- `text` - Free text (names, addresses, descriptions)
- `number` - Numeric values (quantities, IDs)
- `date` - Dates (DOB, service date, etc.)
- `boolean` - Yes/No, checkboxes
- `currency` - Money amounts ($100.00)
- `code` - Medical codes (ICD-10, CPT, NPI)
- `list` - Multiple values (diagnosis codes)
- `table` - Tabular data (service line items)
- `object` - Nested structure (address with street/city/state)

**Important:**
- Set `required: true` for critical fields that MUST be extracted
- Set higher `confidence_threshold` for important fields
- Provide `validation_hints` to help validation stage
- Include `examples` if you see patterns

### Step 4: Extraction Strategy

Define overall approach:

```json
{{
  "overall_strategy": "form_extraction|table_extraction|hybrid|document_parsing|adaptive",
  "component_strategies": {{
    "table_1": "Extract row-by-row with column mapping",
    "forms": "Field-by-field with label association",
    "checkboxes": "Visual state detection from marks"
  }}
}}
```

**Strategies:**
- `form_extraction` - Field-by-field from labeled form
- `table_extraction` - Row-based extraction from tables
- `hybrid` - Mix of form fields and tables
- `document_parsing` - Narrative text parsing
- `adaptive` - Dynamic approach per section

### Step 5: Validation Guidance

Suggest validations:

```json
{{
  "suggested_validations": {{
    "patient_dob": ["Must be valid date", "Must be in past", "Format: MM/DD/YYYY"],
    "npi": ["Must be 10 digits", "Check digit validation"],
    "total_charges": ["Must equal sum of line item charges"]
  }},
  "cross_field_relationships": [
    {{
      "fields": ["service_date", "patient_dob"],
      "rule": "service_date must be after patient_dob",
      "error_message": "Service date cannot be before patient birth date"
    }}
  ]
}}
```

### Step 6: Confidence and Critical Fields

Identify fields that MUST be accurate:

```json
{{
  "high_confidence_fields": ["patient_name", "patient_dob", "total_charges"],
  "optional_fields": ["patient_email", "secondary_insurance"]
}}
```

### Step 7: VLM Reasoning

Explain your schema design:

```json
{{
  "vlm_reasoning": "This appears to be a medical superbill with patient demographics in the header, a table of service line items in the body, and signature/totals at the bottom. The extraction should prioritize the service table structure while capturing patient info from form fields. Checkboxes indicate insurance type selection.",
  "schema_confidence": 0.92
}}
```

---

## Required Output Format

Return complete schema as JSON:

```json
{{
  "schema_id": "adaptive_xxxxxxxx",
  "document_type_description": "Descriptive explanation",
  
  "field_groups": [/* FieldGroup objects */],
  "fields": [/* AdaptiveField objects */],
  "total_field_count": 45,
  
  "overall_strategy": "hybrid",
  "component_strategies": {{}},
  
  "suggested_validations": {{}},
  "cross_field_relationships": [],
  
  "high_confidence_fields": [],
  "optional_fields": [],
  
  "vlm_reasoning": "Why this schema makes sense",
  "schema_confidence": 0.90
}}
```

## Critical Reminders

1. **ADAPT TO THIS DOCUMENT** - Don't use templates
2. **USE COMPONENT CONTEXT** - Reference detected tables, forms, fields
3. **BE SPECIFIC** - Clear field names, types, validation hints
4. **THINK EXTRACTION** - How will ExtractorAgent use this schema?
5. **PRIORITIZE CRITICAL FIELDS** - Mark important fields as required

Begin schema generation now. Analyze the document image and propose the best extraction schema."""
    
    def _parse_schema_response(
        self,
        vlm_response: dict[str, Any],
        elapsed_ms: int,
    ) -> dict[str, Any]:
        """
        Parse VLM response into AdaptiveSchema structure.
        
        Args:
            vlm_response: Raw response from VLM.
            elapsed_ms: Generation time.
        
        Returns:
            Properly structured AdaptiveSchema dict.
        """
        # Add schema_id and timing
        schema = dict(vlm_response)
        
        if "schema_id" not in schema:
            schema["schema_id"] = f"adaptive_{secrets.token_hex(8)}"
        
        schema["generation_time_ms"] = elapsed_ms
        
        # Ensure required fields with defaults
        schema.setdefault("document_type_description", "unknown")
        schema.setdefault("field_groups", [])
        schema.setdefault("fields", [])
        schema.setdefault("total_field_count", len(schema.get("fields", [])))
        schema.setdefault("overall_strategy", "adaptive")
        schema.setdefault("component_strategies", {})
        schema.setdefault("suggested_validations", {})
        schema.setdefault("cross_field_relationships", [])
        schema.setdefault("high_confidence_fields", [])
        schema.setdefault("optional_fields", [])
        schema.setdefault("vlm_reasoning", "")
        schema.setdefault("schema_confidence", 0.5)
        
        return schema
    
    def _create_fallback_schema(
        self,
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Create minimal fallback schema if VLM generation fails.
        
        Args:
            layout: Layout analysis.
            components: Component map.
        
        Returns:
            Minimal AdaptiveSchema dict.
        """
        # Create basic fields from component detection
        fields = []
        
        if components:
            # Add fields from detected forms
            for form in components.get("forms", [])[:20]:  # Limit to first 20
                label = form.get("label_text", "")
                if label:
                    field_type = "boolean" if "checkbox" in form.get("field_type", "") else "text"
                    fields.append({
                        "field_name": label.lower().replace(" ", "_"),
                        "display_name": label,
                        "field_type": field_type,
                        "description": f"Field: {label}",
                        "source_component_id": form.get("field_id", ""),
                        "location_hint": "",
                        "required": False,
                        "confidence_threshold": 0.7,
                        "validation_hints": [],
                        "examples": [],
                    })
            
            # Add fields from key-value pairs
            for kv in components.get("key_value_pairs", [])[:20]:  # Limit to first 20
                key = kv.get("key_text", "")
                if key:
                    vtype = kv.get("value_type_hint", "text")
                    fields.append({
                        "field_name": key.lower().replace(" ", "_"),
                        "display_name": key,
                        "field_type": vtype,
                        "description": f"Key-value: {key}",
                        "source_component_id": kv.get("pair_id", ""),
                        "location_hint": "",
                        "required": False,
                        "confidence_threshold": 0.7,
                        "validation_hints": [],
                        "examples": [],
                    })
        
        return {
            "schema_id": f"fallback_{secrets.token_hex(8)}",
            "document_type_description": "Unknown document type (fallback schema)",
            "field_groups": [{
                "group_name": "all_fields",
                "field_names": [f["field_name"] for f in fields],
                "group_type": "custom",
                "extraction_strategy": "best_effort",
                "component_ids": [],
            }],
            "fields": fields,
            "total_field_count": len(fields),
            "overall_strategy": "adaptive",
            "component_strategies": {},
            "suggested_validations": {},
            "cross_field_relationships": [],
            "high_confidence_fields": [],
            "optional_fields": [f["field_name"] for f in fields],
            "vlm_reasoning": "VLM schema generation failed, using fallback based on detected components",
            "schema_confidence": 0.3,
            "generation_time_ms": 0,
        }
