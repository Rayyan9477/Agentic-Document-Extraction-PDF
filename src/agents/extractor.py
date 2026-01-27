"""
Extractor Agent for dual-pass document data extraction.

Responsible for:
- Schema-driven field extraction
- Dual-pass extraction for verification
- Per-field confidence scoring
- Field-by-field comparison and merging
"""

import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent, ExtractionError
from src.agents.utils import (
    RetryConfig,
    build_custom_schema,
    retry_with_backoff,
)
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    FieldMetadata,
    PageExtraction,
    serialize_field_metadata,
    serialize_page_extraction,
    set_status,
    update_state,
)
from src.prompts.extraction import (
    build_extraction_prompt,
    build_verification_prompt,
)
from src.prompts.grounding_rules import (
    build_enhanced_system_prompt,
    build_grounded_system_prompt,
)
from src.schemas import DocumentSchema, FieldDefinition, SchemaRegistry
from src.validation import ComparisonResult, DualPassComparator


logger = get_logger(__name__)


class ExtractorAgent(BaseAgent):
    """
    Dual-pass extraction agent for document data extraction.

    Performs two extraction passes on each page:
    - Pass 1: Standard extraction focusing on completeness
    - Pass 2: Verification extraction focusing on accuracy

    Results are compared field-by-field to calculate confidence
    and flag potential discrepancies.

    VLM Calls: 2 per page (dual-pass)
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        agreement_confidence_boost: float = 0.1,
        disagreement_confidence_penalty: float = 0.3,
    ) -> None:
        """
        Initialize the Extractor agent.

        Args:
            client: Optional pre-configured LM Studio client.
            agreement_confidence_boost: Confidence boost when passes agree.
            disagreement_confidence_penalty: Confidence penalty when passes disagree.
        """
        super().__init__(name="extractor", client=client)
        self._schema_registry = SchemaRegistry()
        self._agreement_boost = agreement_confidence_boost
        self._disagreement_penalty = disagreement_confidence_penalty
        self._dual_pass_comparator = DualPassComparator()

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Extract data from all pages using dual-pass strategy.

        This is the main entry point for the LangGraph workflow.
        
        Routes to either adaptive (VLM-first) or legacy (schema-based) extraction.

        Args:
            state: Current extraction state.

        Returns:
            Updated state with extraction results.
        """
        # Reset metrics to prevent accumulation across documents
        self.reset_metrics()
        
        # Check if using VLM-first adaptive extraction
        use_adaptive = state.get("use_adaptive_extraction", False)
        has_adaptive_schema = state.get("adaptive_schema") is not None
        
        if use_adaptive and has_adaptive_schema:
            self._logger.info(
                "using_adaptive_extraction",
                processing_id=state.get("processing_id", ""),
            )
            return self._process_adaptive(state)
        else:
            self._logger.info(
                "using_legacy_extraction",
                processing_id=state.get("processing_id", ""),
            )
            return self._process_legacy(state)
    
    def _process_legacy(self, state: ExtractionState) -> ExtractionState:
        """
        Legacy extraction using hardcoded schemas (backward compatibility).
        
        Args:
            state: Current extraction state.
        
        Returns:
            Updated state with extraction results.
        """
        start_time = self.log_operation_start(
            "dual_pass_extraction",
            processing_id=state.get("processing_id", ""),
            page_count=len(state.get("page_images", [])),
        )

        try:
            # Update status
            state = set_status(state, ExtractionStatus.EXTRACTING, "extracting")

            # Get schema for extraction
            schema = self._get_schema(state)
            if not schema:
                raise ExtractionError(
                    "No schema available for extraction",
                    agent_name=self.name,
                    recoverable=False,
                )

            # Get page images
            page_images = state.get("page_images", [])
            if not page_images:
                raise ExtractionError(
                    "No page images available for extraction",
                    agent_name=self.name,
                    recoverable=False,
                )

            # Extract from each page
            page_extractions: list[dict[str, Any]] = []
            total_vlm_calls = 0

            for page_data in page_images:
                page_result = self._extract_page(
                    page_data=page_data,
                    schema=schema,
                    document_type=state.get("document_type", "OTHER"),
                    total_pages=len(page_images),
                )
                page_extractions.append(serialize_page_extraction(page_result))
                total_vlm_calls += page_result.vlm_calls

            # Merge results from all pages
            merged_extraction = self._merge_page_extractions(page_extractions, schema)

            # Build field metadata
            field_metadata = self._build_field_metadata(merged_extraction)

            # Calculate processing time
            duration_ms = self.log_operation_complete(
                "dual_pass_extraction",
                start_time,
                success=True,
                pages_extracted=len(page_extractions),
                vlm_calls=total_vlm_calls,
            )

            # Update state
            state = update_state(
                state,
                {
                    "page_extractions": page_extractions,
                    "merged_extraction": merged_extraction,
                    "field_metadata": {
                        k: serialize_field_metadata(v) for k, v in field_metadata.items()
                    },
                    "status": ExtractionStatus.EXTRACTING.value,
                    "current_step": "extraction_complete",
                    "total_vlm_calls": state.get("total_vlm_calls", 0) + total_vlm_calls,
                    "total_processing_time_ms": (
                        state.get("total_processing_time_ms", 0) + duration_ms
                    ),
                },
            )

            return state

        except ExtractionError:
            raise
        except Exception as e:
            self.log_operation_complete("dual_pass_extraction", start_time, success=False)
            raise ExtractionError(
                f"Extraction failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e
    
    def _process_adaptive(self, state: ExtractionState) -> ExtractionState:
        """
        Adaptive zero-shot extraction using VLM-first pipeline.
        
        Uses layout analysis, component detection, and adaptive schema
        for structure-aware extraction without hardcoded templates.
        
        Args:
            state: Current extraction state with VLM-first analysis.
        
        Returns:
            Updated state with extraction results.
        """
        start_time = self.log_operation_start(
            "adaptive_extraction",
            processing_id=state.get("processing_id", ""),
            page_count=len(state.get("page_images", [])),
        )
        
        try:
            # Update status
            state = set_status(state, ExtractionStatus.EXTRACTING, "adaptive_extracting")
            
            # Get VLM-first analysis
            adaptive_schema = state.get("adaptive_schema")
            layout_analyses = state.get("layout_analyses", [])
            component_maps = state.get("component_maps", [])
            
            if not adaptive_schema:
                raise ExtractionError(
                    "No adaptive schema available for extraction",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            # Get page images
            page_images = state.get("page_images", [])
            if not page_images:
                raise ExtractionError(
                    "No page images available for extraction",
                    agent_name=self.name,
                    recoverable=False,
                )
            
            self._logger.info(
                "adaptive_extraction_started",
                pages=len(page_images),
                fields=adaptive_schema.get("total_field_count", 0),
                strategy=adaptive_schema.get("overall_strategy", "unknown"),
            )
            
            # Extract from each page using adaptive strategy
            page_extractions: list[dict[str, Any]] = []
            total_vlm_calls = 0
            
            for idx, page_data in enumerate(page_images):
                page_number = page_data.get("page_number", idx + 1)
                
                # Get corresponding layout and components for this page
                layout = None
                components = None
                
                for la in layout_analyses:
                    if la.get("page_number") == page_number:
                        layout = la
                        break
                
                for cm in component_maps:
                    if cm.get("page_number") == page_number:
                        components = cm
                        break
                
                # Extract page with full context
                page_result = self._extract_page_adaptive(
                    page_data=page_data,
                    adaptive_schema=adaptive_schema,
                    layout=layout,
                    components=components,
                    page_number=page_number,
                    total_pages=len(page_images),
                )
                
                page_extractions.append(serialize_page_extraction(page_result))
                total_vlm_calls += page_result.vlm_calls
                
                self._logger.debug(
                    "page_extracted_adaptive",
                    page_number=page_number,
                    fields_extracted=len(page_result.merged_fields),
                    confidence=page_result.overall_confidence,
                )
            
            # Merge results from all pages
            merged_extraction = self._merge_page_extractions_adaptive(
                page_extractions, adaptive_schema
            )
            
            # Build field metadata
            field_metadata = self._build_field_metadata(merged_extraction)
            
            # Calculate processing time
            duration_ms = self.log_operation_complete(
                "adaptive_extraction",
                start_time,
                success=True,
                pages_extracted=len(page_extractions),
                vlm_calls=total_vlm_calls,
            )
            
            # Update state
            state = update_state(
                state,
                {
                    "page_extractions": page_extractions,
                    "merged_extraction": merged_extraction,
                    "field_metadata": {
                        k: serialize_field_metadata(v) for k, v in field_metadata.items()
                    },
                    "status": ExtractionStatus.EXTRACTING.value,
                    "current_step": "adaptive_extraction_complete",
                    "total_vlm_calls": state.get("total_vlm_calls", 0) + total_vlm_calls,
                    "total_processing_time_ms": (
                        state.get("total_processing_time_ms", 0) + duration_ms
                    ),
                },
            )
            
            self._logger.info(
                "adaptive_extraction_completed",
                total_fields=len(field_metadata),
                total_vlm_calls=total_vlm_calls,
                duration_ms=duration_ms,
            )
            
            return state
        
        except ExtractionError:
            raise
        except Exception as e:
            self.log_operation_complete("adaptive_extraction", start_time, success=False)
            raise ExtractionError(
                f"Adaptive extraction failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e
    
    def _extract_page_adaptive(
        self,
        page_data: dict[str, Any],
        adaptive_schema: dict[str, Any],
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
        page_number: int,
        total_pages: int,
    ) -> PageExtraction:
        """
        Extract data from a single page using adaptive schema and context.
        
        Performs dual-pass extraction with full layout and component context.
        
        Args:
            page_data: Page image data.
            adaptive_schema: VLM-generated adaptive schema.
            layout: Layout analysis for this page.
            components: Component map for this page.
            page_number: Current page number.
            total_pages: Total page count.
        
        Returns:
            PageExtraction with merged dual-pass results.
        """
        image_data_uri = page_data.get("data_uri")
        if not image_data_uri:
            raise ExtractionError(
                f"No image data for page {page_number}",
                agent_name=self.name,
                recoverable=False,
            )
        
        # Build field definitions from adaptive schema
        field_defs = adaptive_schema.get("fields", [])
        document_desc = adaptive_schema.get("document_type_description", "unknown document")
        
        # Pass 1: Structure-aware extraction (completeness focus)
        pass1_start = time.time()
        pass1_result = self._extract_page_pass_adaptive(
            image_data=image_data_uri,
            field_defs=field_defs,
            document_desc=document_desc,
            layout=layout,
            components=components,
            page_number=page_number,
            total_pages=total_pages,
            is_first_pass=True,
        )
        pass1_time = int((time.time() - pass1_start) * 1000)
        
        # Pass 2: Verification pass (accuracy focus)
        pass2_start = time.time()
        pass2_result = self._extract_page_pass_adaptive(
            image_data=image_data_uri,
            field_defs=field_defs,
            document_desc=document_desc,
            layout=layout,
            components=components,
            page_number=page_number,
            total_pages=total_pages,
            is_first_pass=False,
        )
        pass2_time = int((time.time() - pass2_start) * 1000)
        
        # Merge results using dual-pass comparator
        pass1_fields = pass1_result.get("fields", {})
        pass2_fields = pass2_result.get("fields", {})
        
        merged_fields = self._merge_pass_results(pass1_fields, pass2_fields, page_number)
        
        # Create PageExtraction
        page_extraction = PageExtraction(
            page_number=page_number,
            pass1_raw=pass1_result,
            pass2_raw=pass2_result,
            merged_fields=merged_fields,
            extraction_time_ms=pass1_time + pass2_time,
            vlm_calls=2,  # Dual-pass
            errors=[],
        )
        
        return page_extraction
    
    def _extract_page_pass_adaptive(
        self,
        image_data: str,
        field_defs: list[dict[str, Any]],
        document_desc: str,
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
        page_number: int,
        total_pages: int,
        is_first_pass: bool,
    ) -> dict[str, Any]:
        """
        Perform single extraction pass with full VLM context.
        
        Args:
            image_data: Base64-encoded image.
            field_defs: Adaptive field definitions.
            document_desc: Document type description.
            layout: Layout analysis context.
            components: Component map context.
            page_number: Current page number.
            total_pages: Total pages.
            is_first_pass: Whether this is pass 1 or 2.
        
        Returns:
            Extraction result dictionary.
        """
        # Build structure-aware system prompt
        system_prompt = self._build_adaptive_system_prompt(
            document_desc=document_desc,
            is_verification=not is_first_pass,
        )
        
        # Build structure-aware extraction prompt
        user_prompt = self._build_adaptive_extraction_prompt(
            field_defs=field_defs,
            document_desc=document_desc,
            layout=layout,
            components=components,
            page_number=page_number,
            total_pages=total_pages,
            is_first_pass=is_first_pass,
        )
        
        # Retry with backoff
        settings = get_settings()
        retry_config = RetryConfig(
            max_retries=settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=settings.agent.max_retry_delay_ms,
        )
        
        def make_vlm_call() -> dict[str, Any]:
            return self.send_vision_request_with_json(
                image_data=image_data,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=6000,  # Adaptive extraction may need more tokens
            )
        
        try:
            return retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "adaptive_extraction_retry",
                    page_number=page_number,
                    pass_number=1 if is_first_pass else 2,
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )
        except Exception as e:
            self._logger.warning(
                "adaptive_extraction_pass_failed",
                page_number=page_number,
                pass_number=1 if is_first_pass else 2,
                error=str(e),
            )
            return {"fields": {}, "error": str(e)}
    
    def _build_adaptive_system_prompt(
        self,
        document_desc: str,
        is_verification: bool,
    ) -> str:
        """Build system prompt for adaptive extraction."""
        mode = "VERIFICATION" if is_verification else "EXTRACTION"
        
        return f"""You are an expert document data extractor specializing in zero-shot extraction.

DOCUMENT TYPE: {document_desc}

MODE: {mode} Pass

You have full context about this document's structure:
- Layout analysis (regions, reading order, visual marks)
- Component detection (tables, forms, checkboxes, key-value pairs)
- Adaptive schema (fields proposed based on structure)

CRITICAL INSTRUCTIONS:

1. **Use Structural Context**: Leverage layout and component information to guide extraction
2. **Respect Component Types**: Extract tables row-by-row, forms field-by-field, checkboxes by state
3. **Visual Mark Detection**: Pay attention to checkmarks, ticks, crosses in checkboxes
4. **Spatial Validation**: Values should be in expected regions based on layout
5. **Confidence Scoring**: Be honest about uncertainty, especially for handwriting
6. **Anti-Hallucination**: Return null for unclear values, do NOT guess

{'VERIFICATION MODE: You are a skeptical auditor. Verify each value independently.' if is_verification else 'EXTRACTION MODE: Focus on completeness. Extract all visible values.'}

Return structured JSON with extracted fields, confidences, and locations."""
    
    def _build_adaptive_extraction_prompt(
        self,
        field_defs: list[dict[str, Any]],
        document_desc: str,
        layout: dict[str, Any] | None,
        components: dict[str, Any] | None,
        page_number: int,
        total_pages: int,
        is_first_pass: bool,
    ) -> str:
        """Build extraction prompt with full structural context."""
        
        # Build context sections
        layout_context = "No layout analysis available"
        if layout:
            visual_marks = layout.get("visual_marks", [])
            mark_summary = {}
            for mark in visual_marks:
                mtype = mark.get("mark_type", "unknown")
                mark_summary[mtype] = mark_summary.get(mtype, 0) + 1
            
            layout_context = f"""
**Layout Structure:**
- Type: {layout.get('layout_type', 'unknown')}
- Reading Order: {layout.get('reading_order', 'unknown')}
- Columns: {layout.get('column_count', 1)}
- Density: {layout.get('density_estimate', 'unknown')}
- Handwriting: {"Yes" if layout.get('has_handwritten_content') else "No"}

**Visual Marks Detected:** {len(visual_marks)}
"""
            for mtype, count in sorted(mark_summary.items()):
                layout_context += f"  - {mtype}: {count}\n"
        
        component_context = "No component analysis available"
        if components:
            tables = components.get("tables", [])
            forms = components.get("forms", [])
            checkboxes = [f for f in forms if "checkbox" in f.get("field_type", "")]
            
            component_context = f"""
**Components Detected:**
- Tables: {len(tables)}
- Form Fields: {len(forms)}
- Checkboxes: {len(checkboxes)}
- Key-Value Pairs: {len(components.get('key_value_pairs', []))}

**Extraction Strategy:** {components.get('suggested_extraction_strategies', {})}
"""
        
        # Build field instructions
        field_instructions = ""
        for field in field_defs[:30]:  # Limit to first 30 fields per page
            name = field.get("field_name", "unknown")
            display = field.get("display_name", name)
            ftype = field.get("field_type", "text")
            desc = field.get("description", "")
            required = field.get("required", False)
            location_hint = field.get("location_hint", "")
            
            req_marker = "**REQUIRED**" if required else "optional"
            
            field_instructions += f"""
### {display} (`{name}`) - {req_marker}
- Type: {ftype}
- Description: {desc}
- Location Hint: {location_hint}
- Component: {field.get('source_component_id', 'unknown')}
"""
        
        pass_instruction = "PASS 1: Extract all visible values. Focus on completeness." if is_first_pass else "PASS 2: Verify each value independently. Focus on accuracy. Be skeptical."
        
        return f"""# ADAPTIVE EXTRACTION - Page {page_number}/{total_pages}

{pass_instruction}

## Document Context

**Document Type:** {document_desc}

{layout_context}

{component_context}

## Fields to Extract

{field_instructions}

## Extraction Instructions

1. **Use Layout Context**: Pay attention to regions, reading order, and visual structure
2. **Component-Specific Strategies**:
   - Tables: Extract row-by-row, maintaining column structure
   - Forms: Match field labels to values spatially
   - Checkboxes: Detect visual marks (✓ ✗ ☑ ☐) to determine state
   - Key-Value Pairs: Associate labels with values based on separators

3. **Visual Mark Detection**:
   - Look for checkmarks, ticks, crosses in checkbox areas
   - Note stamps, signatures, handwritten annotations
   - Identify redactions or obscured content

4. **Confidence Scoring** (0.0-1.0):
   - 0.95+: Crystal clear, no doubt
   - 0.85-0.94: Clear but minor uncertainty
   - 0.70-0.84: Readable but needs verification
   - <0.70: Too uncertain → return null

5. **Spatial Validation**:
   - Values should be in expected regions
   - Check if location matches component bounding boxes
   - Verify reading order makes sense

## Required Output Format

```json
{{
  "page_number": {page_number},
  "extraction_pass": {1 if is_first_pass else 2},
  "fields": {{
    "field_name": {{
      "value": "extracted value or null",
      "confidence": 0.92,
      "location": "description of where found",
      "component_type": "table|form|checkbox|key_value",
      "visual_marks": ["any marks associated with this field"]
    }}
  }},
  "extraction_notes": "Observations about extraction quality",
  "uncertain_fields": ["list of fields with low confidence"],
  "component_extractions": {{
    "table_1": [/* extracted table rows */],
    "checkboxes": {{"checkbox_id": "checked|unchecked"}}
  }}
}}
```

## Critical Reminders

- **Return null for uncertain values** - Do NOT guess
- **Use component context** - Extract appropriately for each component type
- **Detect visual marks** - Checkboxes, stamps, signatures, ticks, crosses
- **Spatial awareness** - Values should match expected locations
- **Confidence honesty** - Be realistic about uncertainty

Begin adaptive extraction now."""
    
    def _merge_page_extractions_adaptive(
        self,
        page_extractions: list[dict[str, Any]],
        adaptive_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge extractions from multiple pages for adaptive schema.
        
        Args:
            page_extractions: List of per-page extraction results.
            adaptive_schema: Adaptive schema definition.
        
        Returns:
            Merged extraction dictionary.
        """
        # For now, use simple page-by-page aggregation
        # Could be enhanced to handle multi-page fields intelligently
        
        merged = {}
        fields = adaptive_schema.get("fields", [])
        
        for field in fields:
            field_name = field.get("field_name")
            values = []
            
            # Collect values from all pages
            for page_ext in page_extractions:
                merged_fields = page_ext.get("merged_fields", {})
                if field_name in merged_fields:
                    field_data = merged_fields[field_name]
                    value = field_data.get("value")
                    if value is not None:
                        values.append({
                            "value": value,
                            "confidence": field_data.get("confidence", 0.5),
                            "page": page_ext.get("page_number", 0),
                        })
            
            # For single-value fields, take highest confidence
            # For list fields, aggregate all values
            if values:
                if field.get("field_type") in ["list", "table"]:
                    merged[field_name] = [v["value"] for v in values]
                else:
                    # Take value with highest confidence
                    best = max(values, key=lambda v: v["confidence"])
                    merged[field_name] = best["value"]
        
        return merged

    def _get_schema(self, state: ExtractionState) -> DocumentSchema | None:
        """
        Get the schema for extraction.

        Args:
            state: Current extraction state.

        Returns:
            DocumentSchema or None if not found.
        """
        # Check for custom schema
        custom_schema = state.get("custom_schema")
        if custom_schema:
            # Build schema from custom definition
            return self._build_custom_schema(custom_schema)

        # Get schema by name
        schema_name = state.get("selected_schema_name", "")
        if schema_name:
            try:
                return self._schema_registry.get(schema_name)
            except ValueError:
                self._logger.warning("schema_not_found", schema_name=schema_name)

        # Return a generic fallback schema for unknown/other documents
        self._logger.info("using_generic_schema", reason="No specific schema found")
        return self._create_generic_schema()

    def _create_generic_schema(self) -> DocumentSchema:
        """
        Create a generic schema for documents without a specific schema.

        Returns:
            A generic DocumentSchema that extracts common document fields.
        """
        from src.schemas import DocumentType, FieldType
        from src.schemas.schema_builder import FieldBuilder, SchemaBuilder

        builder = SchemaBuilder(
            name="generic_document",
            document_type=DocumentType.CUSTOM,
        )

        builder.display_name("Generic Document")
        builder.description("Generic schema for extracting common document information")

        # Add common fields using fluent API
        schema = (
            builder.field(
                FieldBuilder("title")
                .display_name("Document Title")
                .type(FieldType.STRING)
                .description("Main title or heading of the document")
                .location_hint("top of page, header area")
            )
            .field(
                FieldBuilder("date")
                .display_name("Date")
                .type(FieldType.DATE)
                .description("Primary date on the document")
                .location_hint("header, top right, or near title")
            )
            .field(
                FieldBuilder("document_type")
                .display_name("Document Type")
                .type(FieldType.STRING)
                .description("Type or category of the document")
            )
            .field(
                FieldBuilder("summary")
                .display_name("Summary")
                .type(FieldType.STRING)
                .description("Brief summary of the document content")
            )
            .field(
                FieldBuilder("key_information")
                .display_name("Key Information")
                .type(FieldType.STRING)
                .description("Important facts, figures, or data from the document")
            )
            .build()
        )

        return schema

    def _build_custom_schema(self, schema_def: dict[str, Any]) -> DocumentSchema:
        """
        Build a DocumentSchema from a custom schema definition.

        Uses shared utility to eliminate code duplication.

        Args:
            schema_def: Custom schema definition dictionary.

        Returns:
            Constructed DocumentSchema.
        """
        return build_custom_schema(schema_def)

    def _extract_page(
        self,
        page_data: dict[str, Any],
        schema: DocumentSchema,
        document_type: str,
        total_pages: int,
    ) -> PageExtraction:
        """
        Extract data from a single page using dual-pass strategy.

        Args:
            page_data: Page image data.
            schema: Extraction schema.
            document_type: Type of document.
            total_pages: Total number of pages.

        Returns:
            PageExtraction with merged results.
        """
        page_number = page_data.get("page_number", 1)
        image_data = page_data.get("data_uri") or page_data.get("base64_encoded", "")

        if not image_data:
            return PageExtraction(
                page_number=page_number,
                errors=["No image data available for page"],
            )

        start_time = time.perf_counter()
        vlm_calls = 0

        try:
            # Convert schema fields to list of dicts for prompt
            field_defs = [f.to_dict() for f in schema.fields]

            # === PASS 1: Standard Extraction ===
            pass1_result = self._perform_extraction_pass(
                image_data=image_data,
                field_defs=field_defs,
                document_type=document_type,
                page_number=page_number,
                total_pages=total_pages,
                is_first_pass=True,
            )
            vlm_calls += 1

            # === PASS 2: Verification Extraction ===
            pass2_result = self._perform_extraction_pass(
                image_data=image_data,
                field_defs=field_defs,
                document_type=document_type,
                page_number=page_number,
                total_pages=total_pages,
                is_first_pass=False,
            )
            vlm_calls += 1

            # === MERGE RESULTS ===
            merged_fields = self._merge_pass_results(
                pass1_result.get("fields", {}),
                pass2_result.get("fields", {}),
                page_number,
            )

            extraction_time_ms = int((time.perf_counter() - start_time) * 1000)

            return PageExtraction(
                page_number=page_number,
                pass1_raw=pass1_result,
                pass2_raw=pass2_result,
                merged_fields=merged_fields,
                extraction_time_ms=extraction_time_ms,
                vlm_calls=vlm_calls,
                errors=[],
            )

        except Exception as e:
            self._logger.error(
                "page_extraction_failed",
                page_number=page_number,
                error=str(e),
            )
            return PageExtraction(
                page_number=page_number,
                errors=[f"Extraction failed: {e}"],
                vlm_calls=vlm_calls,
            )

    def _perform_extraction_pass(
        self,
        image_data: str,
        field_defs: list[dict[str, Any]],
        document_type: str,
        page_number: int,
        total_pages: int,
        is_first_pass: bool,
    ) -> dict[str, Any]:
        """
        Perform a single extraction pass with enhanced prompts and retry logic.

        Args:
            image_data: Base64-encoded image.
            field_defs: List of field definitions.
            document_type: Type of document.
            page_number: Current page number.
            total_pages: Total pages.
            is_first_pass: Whether this is pass 1 or 2.

        Returns:
            Extraction result dictionary.
        """
        # Build enhanced system prompt with chain-of-thought and anti-hallucination
        system_prompt = build_enhanced_system_prompt(
            document_type=document_type,
            is_verification_pass=not is_first_pass,
        )

        # Build extraction prompt with enhanced features
        if is_first_pass:
            prompt = build_extraction_prompt(
                schema_fields=field_defs,
                document_type=document_type,
                page_number=page_number,
                total_pages=total_pages,
                is_first_pass=True,
                include_reasoning=True,
                include_anti_patterns=True,
            )
        else:
            prompt = build_verification_prompt(
                schema_fields=field_defs,
                document_type=document_type,
                page_number=page_number,
                first_pass_results={},  # Don't show first pass to ensure independence
            )

        # Use retry with exponential backoff for VLM calls
        settings = get_settings()
        retry_config = RetryConfig(
            max_retries=settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=settings.agent.max_retry_delay_ms,
        )

        def make_vlm_call() -> dict[str, Any]:
            return self.send_vision_request_with_json(
                image_data=image_data,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

        try:
            return retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "extraction_pass_retry",
                    pass_number=1 if is_first_pass else 2,
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )
        except Exception as e:
            self._logger.warning(
                "extraction_pass_failed",
                pass_number=1 if is_first_pass else 2,
                error=str(e),
            )
            return {"fields": {}, "error": str(e)}

    def _merge_pass_results(
        self,
        pass1_fields: dict[str, Any],
        pass2_fields: dict[str, Any],
        page_number: int,
    ) -> dict[str, FieldMetadata]:
        """
        Merge results from both extraction passes using DualPassComparator.

        Uses sophisticated comparison algorithm from validation module for:
        - Fuzzy matching with similarity scoring
        - Confidence-weighted value selection
        - Agreement rate calculation

        Args:
            pass1_fields: Fields from pass 1.
            pass2_fields: Fields from pass 2.
            page_number: Current page number.

        Returns:
            Dictionary of merged FieldMetadata.
        """
        # Extract values and confidences from pass data
        pass1_values: dict[str, Any] = {}
        pass2_values: dict[str, Any] = {}
        pass1_conf: dict[str, float] = {}
        pass2_conf: dict[str, float] = {}
        locations: dict[str, str] = {}

        all_fields = set(pass1_fields.keys()) | set(pass2_fields.keys())

        for field_name in all_fields:
            p1 = pass1_fields.get(field_name, {})
            p2 = pass2_fields.get(field_name, {})

            # Extract values
            pass1_values[field_name] = p1.get("value") if isinstance(p1, dict) else p1
            pass2_values[field_name] = p2.get("value") if isinstance(p2, dict) else p2

            # Extract confidences
            pass1_conf[field_name] = p1.get("confidence", 0.5) if isinstance(p1, dict) else 0.5
            pass2_conf[field_name] = p2.get("confidence", 0.5) if isinstance(p2, dict) else 0.5

            # Extract locations
            loc1 = p1.get("location", "") if isinstance(p1, dict) else ""
            loc2 = p2.get("location", "") if isinstance(p2, dict) else ""
            locations[field_name] = loc1 or loc2

        # Use DualPassComparator for sophisticated merging
        comparison_result = self._dual_pass_comparator.compare(
            pass1_data=pass1_values,
            pass2_data=pass2_values,
            pass1_confidence=pass1_conf,
            pass2_confidence=pass2_conf,
        )

        # Convert comparison results to FieldMetadata
        merged: dict[str, FieldMetadata] = {}
        for field_name, field_comparison in comparison_result.field_comparisons.items():
            passes_agree = field_comparison.result in (
                ComparisonResult.EXACT_MATCH,
                ComparisonResult.FUZZY_MATCH,
            )

            merged[field_name] = FieldMetadata(
                field_name=field_name,
                value=field_comparison.merged_value,
                confidence=field_comparison.merge_confidence,
                pass1_value=field_comparison.pass1_value,
                pass2_value=field_comparison.pass2_value,
                passes_agree=passes_agree,
                location_hint=locations.get(field_name, ""),
                source_page=page_number,
            )

        return merged

    def _merge_page_extractions(
        self,
        page_extractions: list[dict[str, Any]],
        schema: DocumentSchema,
    ) -> dict[str, Any]:
        """
        Merge extractions from multiple pages into final result.

        For LIST/TABLE fields: Merges values across pages into arrays.
        For scalar fields: Keeps value with highest confidence.

        Args:
            page_extractions: List of serialized PageExtraction dicts.
            schema: Document schema.

        Returns:
            Merged extraction dictionary.
        """
        from src.schemas.field_types import FieldType

        merged: dict[str, Any] = {}

        # Build field type lookup from schema
        field_types: dict[str, FieldType] = {}
        if schema and hasattr(schema, "fields"):
            for field_def in schema.fields:
                field_types[field_def.name] = field_def.field_type

        def is_mergeable_type(field_name: str) -> bool:
            """Check if field should merge values instead of overwrite."""
            field_type = field_types.get(field_name)
            return field_type in (FieldType.LIST, FieldType.TABLE)

        # For single-page documents, use page 1 directly
        if len(page_extractions) == 1:
            page = page_extractions[0]
            for field_name, field_data in page.get("merged_fields", {}).items():
                merged[field_name] = {
                    "value": field_data.get("value"),
                    "confidence": field_data.get("confidence", 0.0),
                    "source_page": field_data.get("source_page", 1),
                }
            return merged

        # For multi-page documents, apply intelligent merging
        for page in page_extractions:
            page_number = page.get("page_number", 1)
            for field_name, field_data in page.get("merged_fields", {}).items():
                current_value = field_data.get("value")
                current_confidence = field_data.get("confidence", 0.0)

                if field_name not in merged:
                    # First occurrence - initialize
                    if is_mergeable_type(field_name):
                        # For list/table types, wrap in list if not already
                        value_list = (
                            current_value if isinstance(current_value, list) else [current_value]
                        )
                        merged[field_name] = {
                            "value": value_list,
                            "confidence": current_confidence,
                            "source_pages": [page_number],
                        }
                    else:
                        merged[field_name] = {
                            "value": current_value,
                            "confidence": current_confidence,
                            "source_page": page_number,
                        }
                # Field exists - merge or overwrite based on type
                elif is_mergeable_type(field_name):
                    # Merge list/table values
                    existing_value = merged[field_name].get("value", [])
                    if not isinstance(existing_value, list):
                        existing_value = [existing_value]

                    if current_value is not None:
                        if isinstance(current_value, list):
                            existing_value.extend(current_value)
                        else:
                            existing_value.append(current_value)

                    # Average confidence for merged values
                    existing_confidence = merged[field_name].get("confidence", 0.0)
                    avg_confidence = (existing_confidence + current_confidence) / 2

                    source_pages = merged[field_name].get("source_pages", [])
                    if page_number not in source_pages:
                        source_pages.append(page_number)

                    merged[field_name] = {
                        "value": existing_value,
                        "confidence": avg_confidence,
                        "source_pages": source_pages,
                    }
                else:
                    # For scalar types, keep value with higher confidence
                    existing_confidence = merged[field_name].get("confidence", 0.0)
                    if current_confidence > existing_confidence and current_value is not None:
                        merged[field_name] = {
                            "value": current_value,
                            "confidence": current_confidence,
                            "source_page": page_number,
                        }

        return merged

    def _build_field_metadata(
        self,
        merged_extraction: dict[str, Any],
    ) -> dict[str, FieldMetadata]:
        """
        Build FieldMetadata objects from merged extraction.

        Args:
            merged_extraction: Merged extraction dictionary.

        Returns:
            Dictionary of FieldMetadata objects.
        """
        metadata: dict[str, FieldMetadata] = {}

        for field_name, field_data in merged_extraction.items():
            # Handle both old format (dict with value/confidence) and new format (direct values)
            if isinstance(field_data, dict) and "value" in field_data:
                # Old format: {"value": x, "confidence": y, "source_page": z}
                metadata[field_name] = FieldMetadata(
                    field_name=field_name,
                    value=field_data.get("value"),
                    confidence=field_data.get("confidence", 0.0),
                    source_page=field_data.get("source_page", 1),
                )
            else:
                # New format: Direct value (string, list, etc.)
                metadata[field_name] = FieldMetadata(
                    field_name=field_name,
                    value=field_data,
                    confidence=0.85,  # Default confidence for adaptive extraction
                    source_page=1,
                )

        return metadata

    def extract_single_field(
        self,
        image_data: str,
        field_definition: FieldDefinition,
        document_type: str = "OTHER",
    ) -> AgentResult[FieldMetadata]:
        """
        Extract a single field from an image.

        Useful for targeted re-extraction of specific fields.

        Args:
            image_data: Base64-encoded image.
            field_definition: Definition of field to extract.
            document_type: Type of document.

        Returns:
            AgentResult with FieldMetadata.
        """
        from src.prompts.extraction import build_field_prompt

        start_time = self.log_operation_start(
            "single_field_extraction",
            field_name=field_definition.name,
        )

        try:
            system_prompt = build_grounded_system_prompt(
                include_confidence_scale=True,
            )

            prompt = build_field_prompt(
                field_definition=field_definition.to_dict(),
                document_type=document_type,
            )

            result = self.send_vision_request_with_json(
                image_data=image_data,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            metadata = FieldMetadata(
                field_name=field_definition.name,
                value=result.get("value"),
                confidence=result.get("confidence", 0.0),
                location_hint=result.get("location", ""),
            )

            duration_ms = self.log_operation_complete(
                "single_field_extraction",
                start_time,
                success=True,
            )

            return AgentResult.ok(
                data=metadata,
                agent_name=self.name,
                operation="extract_field",
                vlm_calls=1,
                processing_time_ms=duration_ms,
            )

        except Exception as e:
            self.log_operation_complete(
                "single_field_extraction",
                start_time,
                success=False,
            )
            return AgentResult.fail(
                error=str(e),
                agent_name=self.name,
                operation="extract_field",
            )
