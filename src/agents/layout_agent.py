"""
LayoutAgent - VLM-native visual layout understanding.

First stage of VLM-first extraction pipeline. Analyzes document visual
structure, detects visual marks (checkboxes, ticks, crosses), and provides
spatial understanding before content extraction.
"""

import time
from typing import Any

from src.agents.base import AgentError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.config import get_settings
from src.pipeline.layout_types import LayoutAnalysis, Region, VisualMark, BoundingBox
from src.pipeline.state import ExtractionState
from src.prompts.grounding_rules import build_grounded_system_prompt


class LayoutAnalysisError(AgentError):
    """Error during layout analysis."""


class LayoutAgent(BaseAgent):
    """
    VLM-powered layout understanding agent.
    
    Analyzes document visual structure without extracting content.
    Detects visual marks that humans use (checkboxes, ticks, crosses, circles).
    
    Key capabilities:
    - Identifies layout type (form, table, mixed, etc.)
    - Detects regions and spatial structure
    - Recognizes visual marks and their states
    - Determines reading order and information density
    - Provides extraction strategy recommendations
    
    VLM Utilization: ~15% of total pipeline capability
    Critical for: Zero-shot documents, complex layouts, visual annotations
    """
    
    def __init__(self, client=None):
        """Initialize layout agent."""
        super().__init__(name="layout", client=client)
        self._settings = get_settings()
    
    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Analyze visual layout for all pages.
        
        Args:
            state: Current extraction state with page_images.
        
        Returns:
            State updated with layout_analyses list.
        """
        start_time = time.time()
        
        try:
            self._logger.info(
                "layout_analysis_started",
                page_count=len(state.get("page_images", [])),
                processing_id=state.get("processing_id"),
            )
            
            page_images = state.get("page_images", [])
            if not page_images:
                raise LayoutAnalysisError(
                    "No page images available for layout analysis",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            layout_analyses = []
            
            for page_data in page_images:
                page_number = page_data.get("page_number", 1)
                image_data_uri = page_data.get("data_uri")
                
                if not image_data_uri:
                    self._logger.warning(
                        "page_missing_image_data",
                        page_number=page_number,
                    )
                    continue
                
                # Analyze layout for this page
                layout = self._analyze_page_layout(image_data_uri, page_number)
                layout_analyses.append(layout)
                
                self._logger.info(
                    "page_layout_analyzed",
                    page_number=page_number,
                    layout_type=layout["layout_type"],
                    visual_marks_found=len(layout.get("visual_marks", [])),
                    estimated_fields=layout.get("estimated_field_count", 0),
                )
            
            # Update state
            state["layout_analyses"] = layout_analyses
            state["total_vlm_calls"] = state.get("total_vlm_calls", 0) + len(layout_analyses)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            state["total_processing_time_ms"] = state.get("total_processing_time_ms", 0) + elapsed_ms
            
            self._logger.info(
                "layout_analysis_completed",
                pages_analyzed=len(layout_analyses),
                total_time_ms=elapsed_ms,
            )
            
            return state
        
        except Exception as e:
            self._logger.error(
                "layout_analysis_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise LayoutAnalysisError(
                f"Layout analysis failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e
    
    def _analyze_page_layout(self, image_data_uri: str, page_number: int) -> dict[str, Any]:
        """
        Analyze visual layout for a single page using VLM.
        
        Args:
            image_data_uri: Data URI of page image.
            page_number: Page number being analyzed.
        
        Returns:
            LayoutAnalysis dict with structure and visual marks.
        """
        start_time = time.time()
        
        system_prompt = build_grounded_system_prompt(
            additional_context=(
                "You are analyzing document visual structure. Focus on layout, "
                "spatial relationships, and visual marks (checkboxes, ticks, crosses). "
                "Do NOT extract content yet - only describe structure."
            ),
            include_forbidden=True,
            include_confidence_scale=True,
        )
        
        user_prompt = self._build_layout_analysis_prompt()
        
        # Retry with backoff for VLM calls
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
                max_tokens=3000,  # Layout analysis needs detailed response
            )
        
        try:
            result = retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "layout_analysis_retry",
                    page_number=page_number,
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Convert VLM response to LayoutAnalysis structure
            return self._parse_layout_response(result, page_number, elapsed_ms)
        
        except Exception as e:
            self._logger.error(
                "page_layout_analysis_failed",
                page_number=page_number,
                error=str(e),
            )
            # Return minimal layout analysis on failure
            return {
                "page_number": page_number,
                "layout_type": "unknown",
                "layout_confidence": 0.0,
                "regions": [],
                "column_count": 1,
                "reading_order": "top-to-bottom",
                "visual_separators": [],
                "density_estimate": "moderate",
                "estimated_field_count": 0,
                "has_pre_printed_structure": False,
                "has_handwritten_content": False,
                "alignment_style": "unknown",
                "spacing_quality": "normal",
                "vlm_observations": f"Analysis failed: {e}",
                "extraction_difficulty": "moderate",
                "recommended_strategy": "fallback",
                "visual_marks": [],
                "analysis_time_ms": 0,
            }
    
    def _build_layout_analysis_prompt(self) -> str:
        """Build comprehensive layout analysis prompt with visual mark detection."""
        return """# TASK: Visual Layout Analysis (No Content Extraction Yet)

You are analyzing a document's VISUAL STRUCTURE ONLY. Do not extract content - focus on layout and spatial organization.

## Stage 1: Overall Layout Classification

Identify the primary layout type:
- **form**: Pre-printed fields with labels and input areas
- **table**: Row and column grid structure
- **tabular_form**: Hybrid of form fields arranged in table format
- **narrative**: Flowing paragraphs of text
- **mixed**: Combination of multiple layouts
- **invoice**: Billing document with line items
- **receipt**: Transaction record
- **letter**: Correspondence format
- **report**: Structured document with sections
- **claim_form**: Insurance or reimbursement claim

Provide confidence score (0.0-1.0) for your classification.

## Stage 2: Spatial Structure Analysis

### Regions
Identify major regions using normalized coordinates (0.0-1.0):
- **header**: Top section (logo, document title, dates)
- **body**: Main content area
- **footer**: Bottom section (page numbers, disclaimers)
- **sidebar**: Side columns or margin information
- **table**: Tabular data region
- **form**: Form field section
- **signature_area**: Signature/stamp region

For each region, provide:
```json
{
  "region_id": "unique_id",
  "region_type": "header|body|footer|sidebar|table|form|signature_area",
  "bounding_box": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 0.15},
  "description": "Brief description of what's in this region",
  "confidence": 0.95
}
```

### Column Structure
- How many columns? (1, 2, 3, or more)
- Is it single-column flowing text or multi-column layout?

### Reading Order
Describe natural reading flow:
- "top-to-bottom, left-to-right"
- "left-column first, then right column"
- "table row-by-row"
- "form section-by-section"

### Visual Separators
What creates visual structure?
- "horizontal_lines" - rules separating sections
- "vertical_lines" - column dividers
- "boxes" - outlined regions
- "borders" - edges around content
- "whitespace" - spacing creating separation
- "shading" - background colors/patterns

## Stage 3: Visual Marks Detection (CRITICAL for Human Annotations)

**Humans use visual marks to indicate selections, confirmations, and annotations.**

Identify ALL visual marks with precise types:

### Checkbox Marks
- **checkbox_checked**: ☑ Checkbox with checkmark/tick inside
- **checkbox_unchecked**: ☐ Empty checkbox
- **checkbox_partial**: ☒ Checkbox with X or partial mark

### Tick/Check Marks
- **tick**: ✓ Standalone checkmark (not in box)
- **checkmark**: ✔ Bold checkmark
- **cross**: ✗ Cross mark indicating selection or cancellation
- **x_mark**: ✘ X mark (different from cross)

### Circle Marks
- **circle**: ○ Circle around text/number
- **circled_item**: Something circled for emphasis

### Emphasis Marks
- **underline**: Text underlined for emphasis
- **highlight**: Text highlighted/marked
- **arrow**: → Arrow pointing to something important
- **pointer**: Hand-drawn pointer or indicator

### Official Marks
- **stamp**: Official stamp or seal impression
- **seal**: Embossed or printed seal
- **signature_present**: Actual signature present
- **signature_placeholder**: Empty signature line
- **initial**: Initials in margin or field
- **handwritten_mark**: Any other handwritten annotation

### Redaction Marks
- **redaction**: Blacked-out or obscured content
- **obscured**: Partially hidden information

For each visual mark, provide:
```json
{
  "mark_type": "checkbox_checked|tick|cross|...",
  "location": {"x": 0.1, "y": 0.3, "width": 0.02, "height": 0.02},
  "confidence": 0.90,
  "state": "checked|unchecked|partial|null",
  "description": "Natural language: what you see"
}
```

**Examples of visual mark descriptions:**
- "Checkbox in 'Male' field has checkmark inside"
- "Tick mark next to 'Option A' indicating selection"
- "X mark crossed out in date field"
- "Red stamp reading 'APPROVED' in bottom right"
- "Handwritten initials 'JD' in margin"

## Stage 4: Information Density

### Density Estimate
- **sparse**: Wide spacing, few data points, mostly empty
- **moderate**: Balanced spacing, comfortable readability
- **dense**: Packed information, minimal whitespace
- **very_dense**: Crowded, challenging to parse visually

### Field Count Estimation
Approximately how many extractable data fields exist?
- Count form fields, table cells, key-value pairs
- Estimate, don't need exact count

### Pre-printed Structure
- Is this a pre-printed form with filled-in data? (true/false)
- Or is it generated/typed document? (false)

### Handwritten Content
- Does document contain handwritten portions? (true/false)
- Are handwritten parts filling in printed form?

## Stage 5: Visual Characteristics

### Alignment Style
- **grid-aligned**: Fields align to invisible grid
- **flowing**: Natural text flow without strict alignment
- **mixed**: Some aligned, some flowing

### Spacing Quality
- **tight**: Minimal spacing, content close together
- **normal**: Standard spacing
- **spacious**: Generous whitespace, comfortable reading

## Stage 6: Extraction Guidance

### VLM Observations
Provide free-form notes about what makes this document unique:
- "Header has 3-column layout for patient/provider/insurance info"
- "Multiple checkboxes on right side for selecting options"
- "Bottom section has signature areas with dates"
- "Handwritten notes in margins may be corrections"

### Extraction Difficulty
Rate difficulty of extracting from this layout:
- **easy**: Clear structure, printed text, good quality
- **moderate**: Some challenges but manageable
- **challenging**: Handwriting, poor quality, or complex layout
- **very_difficult**: Severe quality issues or extremely complex

### Recommended Strategy
Suggest extraction approach:
- "form_extraction" - Field-by-field extraction
- "table_extraction" - Row-based extraction
- "hybrid" - Mix of approaches
- "ocr_fallback" - May need OCR assistance
- "manual_review" - Human review recommended

## Required Output Format

Return JSON with this exact structure:

```json
{
  "layout_type": "form|table|tabular_form|...",
  "layout_confidence": 0.92,
  "regions": [/* array of region objects */],
  "column_count": 2,
  "reading_order": "description of flow",
  "visual_separators": ["horizontal_lines", "boxes"],
  "density_estimate": "moderate",
  "estimated_field_count": 45,
  "has_pre_printed_structure": true,
  "has_handwritten_content": false,
  "alignment_style": "grid-aligned",
  "spacing_quality": "normal",
  "visual_marks": [/* array of VisualMark objects */],
  "vlm_observations": "Detailed notes about layout",
  "extraction_difficulty": "moderate",
  "recommended_strategy": "form_extraction"
}
```

## Critical Reminders

1. **DO NOT extract content yet** - Only analyze structure
2. **BE SPECIFIC about visual marks** - Precise identification crucial
3. **USE NORMALIZED COORDINATES** - 0.0 to 1.0 range for x, y, width, height
4. **CONFIDENCE SCORES** - Honest assessment of layout certainty
5. **DESCRIBE VISUALLY** - What you SEE, not what you interpret

Begin layout analysis now."""
    
    def _parse_layout_response(
        self,
        vlm_response: dict[str, Any],
        page_number: int,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        """
        Parse VLM response into LayoutAnalysis structure.
        
        Args:
            vlm_response: Raw response from VLM.
            page_number: Page number analyzed.
            elapsed_ms: Analysis time.
        
        Returns:
            Properly structured LayoutAnalysis dict.
        """
        # Add page_number and timing
        layout = dict(vlm_response)
        layout["page_number"] = page_number
        layout["analysis_time_ms"] = elapsed_ms
        
        # Ensure required fields with defaults
        layout.setdefault("layout_type", "mixed")
        layout.setdefault("layout_confidence", 0.5)
        layout.setdefault("regions", [])
        layout.setdefault("column_count", 1)
        layout.setdefault("reading_order", "top-to-bottom")
        layout.setdefault("visual_separators", [])
        layout.setdefault("density_estimate", "moderate")
        layout.setdefault("estimated_field_count", 0)
        layout.setdefault("has_pre_printed_structure", False)
        layout.setdefault("has_handwritten_content", False)
        layout.setdefault("alignment_style", "unknown")
        layout.setdefault("spacing_quality", "normal")
        layout.setdefault("visual_marks", [])
        layout.setdefault("vlm_observations", "")
        layout.setdefault("extraction_difficulty", "moderate")
        layout.setdefault("recommended_strategy", "adaptive")
        
        return layout
