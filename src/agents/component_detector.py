"""
ComponentDetectorAgent - VLM-native component detection.

Second stage of VLM-first extraction pipeline. Detects tables, forms,
checkboxes, key-value pairs with precise structure information.
"""

import time
from typing import Any

from src.agents.base import AgentError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.config import get_settings
from src.pipeline.state import ExtractionState
from src.prompts.grounding_rules import build_grounded_system_prompt


class ComponentDetectionError(AgentError):
    """Error during component detection."""


class ComponentDetectorAgent(BaseAgent):
    """
    VLM-powered component detection agent.
    
    Identifies extractable components using layout context:
    - Tables with row/column structure
    - Form fields (text, checkbox, radio, date, etc.)
    - Key-value pairs
    - Checkboxes and their states
    - Signatures and special elements
    
    Uses layout analysis from LayoutAgent for context-aware detection.
    
    VLM Utilization: ~20% of total pipeline capability
    Critical for: Structured data, forms, checkboxes, table parsing
    """
    
    def __init__(self, client=None):
        """Initialize component detector agent."""
        super().__init__(name="component_detector", client=client)
        self._settings = get_settings()
    
    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Detect components for all pages.
        
        Args:
            state: Current extraction state with page_images and layout_analyses.
        
        Returns:
            State updated with component_maps list.
        """
        start_time = time.time()
        
        try:
            self._logger.info(
                "component_detection_started",
                page_count=len(state.get("page_images", [])),
                processing_id=state.get("processing_id"),
            )
            
            page_images = state.get("page_images", [])
            layout_analyses = state.get("layout_analyses", [])
            
            if not page_images:
                raise ComponentDetectionError(
                    "No page images available for component detection",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            component_maps = []
            
            for idx, page_data in enumerate(page_images):
                page_number = page_data.get("page_number", idx + 1)
                image_data_uri = page_data.get("data_uri")
                
                # Get corresponding layout analysis
                layout = None
                for la in layout_analyses:
                    if la.get("page_number") == page_number:
                        layout = la
                        break
                
                if not image_data_uri:
                    self._logger.warning(
                        "page_missing_image_data",
                        page_number=page_number,
                    )
                    continue
                
                # Detect components for this page
                components = self._detect_page_components(
                    image_data_uri, page_number, layout
                )
                component_maps.append(components)
                
                self._logger.info(
                    "page_components_detected",
                    page_number=page_number,
                    tables=len(components.get("tables", [])),
                    forms=len(components.get("forms", [])),
                    key_value_pairs=len(components.get("key_value_pairs", [])),
                    visual_marks=len(components.get("visual_marks", [])),
                    checkboxes=sum(1 for f in components.get("forms", []) 
                                  if f.get("field_type") == "checkbox"),
                )
            
            # Update state
            state["component_maps"] = component_maps
            state["total_vlm_calls"] = state.get("total_vlm_calls", 0) + len(component_maps)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            state["total_processing_time_ms"] = state.get("total_processing_time_ms", 0) + elapsed_ms
            
            self._logger.info(
                "component_detection_completed",
                pages_processed=len(component_maps),
                total_time_ms=elapsed_ms,
            )
            
            return state
        
        except Exception as e:
            self._logger.error(
                "component_detection_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ComponentDetectionError(
                f"Component detection failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e
    
    def _detect_page_components(
        self, 
        image_data_uri: str, 
        page_number: int,
        layout: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Detect components for a single page using VLM.
        
        Args:
            image_data_uri: Data URI of page image.
            page_number: Page number being analyzed.
            layout: Layout analysis from previous stage (context).
        
        Returns:
            ComponentMap dict with all detected components.
        """
        start_time = time.time()
        
        system_prompt = build_grounded_system_prompt(
            additional_context=(
                "You are detecting extractable components in a document. "
                "Identify tables, forms, checkboxes, key-value pairs with precise structure. "
                "Use layout context to improve detection accuracy."
            ),
            include_forbidden=True,
            include_confidence_scale=True,
        )
        
        user_prompt = self._build_component_detection_prompt(layout)
        
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
                temperature=0.1,
                max_tokens=4000,  # Component detection needs detailed response
            )
        
        try:
            result = retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "component_detection_retry",
                    page_number=page_number,
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Convert VLM response to ComponentMap structure
            return self._parse_component_response(result, page_number, elapsed_ms)
        
        except Exception as e:
            self._logger.error(
                "page_component_detection_failed",
                page_number=page_number,
                error=str(e),
            )
            # Return minimal component map on failure
            return {
                "page_number": page_number,
                "tables": [],
                "forms": [],
                "key_value_pairs": [],
                "visual_marks": [],
                "special_elements": [],
                "has_tabular_data": False,
                "has_form_fields": False,
                "has_narrative_text": False,
                "has_checkboxes": False,
                "has_signatures": False,
                "has_handwriting": False,
                "component_count": 0,
                "extraction_order": [],
                "challenging_regions": [],
                "suggested_extraction_strategies": {},
                "vlm_notes": f"Detection failed: {e}",
                "detection_time_ms": 0,
            }
    
    def _build_component_detection_prompt(self, layout: dict[str, Any] | None) -> str:
        """Build component detection prompt with layout context."""
        
        layout_context = ""
        if layout:
            layout_context = f"""
## Layout Context (from previous analysis)

Layout Type: {layout.get('layout_type', 'unknown')}
Reading Order: {layout.get('reading_order', 'unknown')}
Estimated Fields: {layout.get('estimated_field_count', 0)}
Visual Marks Found: {len(layout.get('visual_marks', []))}
Density: {layout.get('density_estimate', 'unknown')}
Difficulty: {layout.get('extraction_difficulty', 'unknown')}

VLM Observations: {layout.get('vlm_observations', 'none')}

Use this context to focus your component detection.
"""
        
        return f"""# TASK: Component Detection (Extractable Elements)

{layout_context}

## Goal
Identify all extractable components with precise structure information.
Focus on elements that contain data to be extracted.

## Component Type 1: Tables

Detect tabular data with this information:

```json
{{
  "table_id": "table_1",
  "location": {{"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.4}},
  "row_count": 10,
  "column_count": 5,
  "has_header_row": true,
  "has_row_numbers": false,
  "cell_borders_visible": true,
  "cell_separation_method": "borders|spacing|mixed",
  "column_labels": ["Date", "Service", "Code", "Amount", "Total"],
  "description": "Billing line items table",
  "complexity": "simple|moderate|complex",
  "confidence": 0.95
}}
```

Identify:
- Table boundaries (normalized coordinates 0-1)
- Row and column counts
- Header row and column labels
- Border style (visible borders, spacing-based, or mixed)
- Complexity (simple grid vs. merged cells)

## Component Type 2: Form Fields

Detect form input fields with metadata:

```json
{{
  "field_id": "field_patient_name",
  "field_type": "text_filled|text_empty|checkbox|radio_button|date_field|numeric_field|signature_area",
  "location": {{"x": 0.2, "y": 0.1, "width": 0.3, "height": 0.03}},
  "label_text": "Patient Name",
  "label_location": {{"x": 0.05, "y": 0.1, "width": 0.12, "height": 0.03}},
  "is_filled": true,
  "visual_marks": [/* any checkmarks/ticks in this field */],
  "confidence": 0.90,
  "extraction_hint": "Text field with handwritten name"
}}
```

Field Types:
- **text_input**: Empty text field
- **text_filled**: Filled text field
- **checkbox**: Checkbox (detect state: checked/unchecked/partial)
- **radio_button**: Radio button option
- **dropdown**: Dropdown indicator
- **date_field**: Date input field
- **numeric_field**: Number input field
- **signature_area**: Signature box
- **multi_line_text**: Multi-line text area
- **barcode/qr_code**: Machine-readable codes

For each field:
1. Detect label and its location
2. Detect input area and its location
3. Determine if filled or empty
4. Check for visual marks (checkmarks in checkboxes)
5. Provide extraction hint

### CRITICAL: Checkbox Detection

For checkboxes, pay special attention:
- Is there a box/square outline?
- What's inside: ✓ checkmark, ✗ cross, blank, X mark?
- State: "checked", "unchecked", "partial"
- Confidence in state detection

## Component Type 3: Key-Value Pairs

Detect label:value patterns:

```json
{{
  "pair_id": "kv_1",
  "key_text": "Patient DOB",
  "key_location": {{"x": 0.1, "y": 0.2, "width": 0.15, "height": 0.02}},
  "value_location": {{"x": 0.26, "y": 0.2, "width": 0.2, "height": 0.02}},
  "separator_type": "colon|line|spacing|box|none",
  "value_type_hint": "date|text|number|currency|code",
  "confidence": 0.88
}}
```

Look for patterns like:
- "Label: Value"
- "Label ________" (value on line)
- "Label [    ]" (value in box)
- Label followed by value spatially

## Component Type 4: Visual Marks (Refined from Layout)

Refine visual mark detection with component context:

```json
{{
  "mark_type": "checkbox_checked|tick|cross|stamp|signature_present|...",
  "location": {{"x": 0.5, "y": 0.3, "width": 0.02, "height": 0.02}},
  "confidence": 0.92,
  "state": "checked",
  "description": "Checkmark in 'Gender: Male' checkbox"
}}
```

Cross-reference with form fields:
- Which marks belong to which fields?
- Are marks inside checkboxes or standalone?

## Component Type 5: Special Elements

Detect non-textual elements:

```json
{{
  "element_id": "elem_1",
  "element_type": "logo|stamp|seal|barcode|qr_code|watermark",
  "location": {{"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.08}},
  "description": "Practice logo in top right",
  "confidence": 0.95
}}
```

## Component Categorization

After detecting components, categorize document content:

```json
{{
  "has_tabular_data": true,
  "has_form_fields": true,
  "has_narrative_text": false,
  "has_checkboxes": true,
  "has_signatures": true,
  "has_handwriting": false
}}
```

## Extraction Strategy Recommendations

Suggest strategies per component type:

```json
{{
  "suggested_extraction_strategies": {{
    "tables": "row_by_row_with_headers",
    "forms": "field_by_field_with_labels",
    "checkboxes": "visual_state_detection",
    "key_value_pairs": "spatial_association"
  }},
  "extraction_order": ["header_info", "patient_demographics", "service_table", "totals"],
  "challenging_regions": [
    {{"region": "bottom_signature", "challenge": "handwritten_signatures", "strategy": "confidence_threshold_0.6"}}
  ]
}}
```

## Required Output Format

```json
{{
  "tables": [/* array of TableStructure */],
  "forms": [/* array of FormField */],
  "key_value_pairs": [/* array of KeyValuePair */],
  "visual_marks": [/* array of VisualMark */],
  "special_elements": [/* array of SpecialElement */],
  
  "has_tabular_data": true,
  "has_form_fields": true,
  "has_narrative_text": false,
  "has_checkboxes": true,
  "has_signatures": false,
  "has_handwriting": false,
  
  "component_count": 45,
  "extraction_order": ["section1", "section2", "..."],
  "challenging_regions": [],
  "suggested_extraction_strategies": {{}},
  "vlm_notes": "Your observations about components"
}}
```

## Critical Reminders

1. **PRECISE LOCATIONS** - Use normalized 0-1 coordinates
2. **CHECKBOX STATES** - Carefully detect checked/unchecked/partial
3. **TABLE STRUCTURE** - Count rows and columns accurately
4. **LABEL-VALUE ASSOCIATION** - Link labels to their values
5. **EXTRACTION HINTS** - Help next stage know how to extract

Begin component detection now."""
    
    def _parse_component_response(
        self,
        vlm_response: dict[str, Any],
        page_number: int,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        """
        Parse VLM response into ComponentMap structure.
        
        Args:
            vlm_response: Raw response from VLM.
            page_number: Page number analyzed.
            elapsed_ms: Detection time.
        
        Returns:
            Properly structured ComponentMap dict.
        """
        # Add page_number and timing
        components = dict(vlm_response)
        components["page_number"] = page_number
        components["detection_time_ms"] = elapsed_ms
        
        # Ensure required fields with defaults
        components.setdefault("tables", [])
        components.setdefault("forms", [])
        components.setdefault("key_value_pairs", [])
        components.setdefault("visual_marks", [])
        components.setdefault("special_elements", [])
        components.setdefault("has_tabular_data", False)
        components.setdefault("has_form_fields", False)
        components.setdefault("has_narrative_text", False)
        components.setdefault("has_checkboxes", False)
        components.setdefault("has_signatures", False)
        components.setdefault("has_handwriting", False)
        components.setdefault("component_count", 0)
        components.setdefault("extraction_order", [])
        components.setdefault("challenging_regions", [])
        components.setdefault("suggested_extraction_strategies", {})
        components.setdefault("vlm_notes", "")
        
        # Calculate component count if not provided
        if components["component_count"] == 0:
            components["component_count"] = (
                len(components["tables"]) +
                len(components["forms"]) +
                len(components["key_value_pairs"])
            )
        
        return components
