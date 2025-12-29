"""
Extractor Agent for dual-pass document data extraction.

Responsible for:
- Schema-driven field extraction
- Dual-pass extraction for verification
- Per-field confidence scoring
- Field-by-field comparison and merging
"""

from typing import Any
import time

from src.agents.base import BaseAgent, AgentResult, ExtractionError
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    FieldMetadata,
    PageExtraction,
    update_state,
    set_status,
    add_warning,
    increment_vlm_calls,
    serialize_page_extraction,
    serialize_field_metadata,
)
from src.prompts.grounding_rules import (
    build_grounded_system_prompt,
    build_enhanced_system_prompt,
    build_hallucination_warning,
)
from src.prompts.extraction import (
    build_extraction_prompt,
    build_verification_prompt,
    build_table_extraction_prompt,
)
from src.agents.utils import (
    build_custom_schema,
    retry_with_backoff,
    RetryConfig,
    identify_low_confidence_fields,
)
from src.schemas import SchemaRegistry, DocumentSchema, FieldDefinition
from src.validation import DualPassComparator, ComparisonResult


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

        Args:
            state: Current extraction state.

        Returns:
            Updated state with extraction results.
        """
        # Reset metrics to prevent accumulation across documents
        self.reset_metrics()

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
                        k: serialize_field_metadata(v)
                        for k, v in field_metadata.items()
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
        from src.schemas.schema_builder import SchemaBuilder, FieldBuilder
        from src.schemas import DocumentType, FieldType

        builder = SchemaBuilder(
            name="generic_document",
            document_type=DocumentType.CUSTOM,
        )

        builder.display_name("Generic Document")
        builder.description("Generic schema for extracting common document information")

        # Add common fields using fluent API
        schema = (builder
            .field(FieldBuilder("title")
                .display_name("Document Title")
                .type(FieldType.STRING)
                .description("Main title or heading of the document")
                .location_hint("top of page, header area"))
            .field(FieldBuilder("date")
                .display_name("Date")
                .type(FieldType.DATE)
                .description("Primary date on the document")
                .location_hint("header, top right, or near title"))
            .field(FieldBuilder("document_type")
                .display_name("Document Type")
                .type(FieldType.STRING)
                .description("Type or category of the document"))
            .field(FieldBuilder("summary")
                .display_name("Summary")
                .type(FieldType.STRING)
                .description("Brief summary of the document content"))
            .field(FieldBuilder("key_information")
                .display_name("Key Information")
                .type(FieldType.STRING)
                .description("Important facts, figures, or data from the document"))
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
        retry_config = RetryConfig(
            max_retries=2,
            base_delay_ms=500,
            max_delay_ms=5000,
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
                        value_list = current_value if isinstance(current_value, list) else [current_value]
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
                else:
                    # Field exists - merge or overwrite based on type
                    if is_mergeable_type(field_name):
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
            metadata[field_name] = FieldMetadata(
                field_name=field_name,
                value=field_data.get("value"),
                confidence=field_data.get("confidence", 0.0),
                source_page=field_data.get("source_page", 1),
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
