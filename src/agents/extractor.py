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
    build_hallucination_warning,
)
from src.prompts.extraction import (
    build_extraction_prompt,
    build_verification_prompt,
    build_table_extraction_prompt,
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

        return None

    def _build_custom_schema(self, schema_def: dict[str, Any]) -> DocumentSchema:
        """
        Build a DocumentSchema from a custom schema definition.

        Args:
            schema_def: Custom schema definition dictionary.

        Returns:
            Constructed DocumentSchema.
        """
        from src.schemas.schema_builder import SchemaBuilder, FieldBuilder, RuleBuilder
        from src.schemas import DocumentType, FieldType, RuleOperator

        builder = SchemaBuilder(
            name=schema_def.get("name", "custom_schema"),
            document_type=DocumentType.OTHER,
        )

        builder.description(schema_def.get("description", "Custom extraction schema"))

        # Build fields
        for field_def in schema_def.get("fields", []):
            field_type_str = field_def.get("type", "string").upper()
            try:
                field_type = FieldType[field_type_str]
            except KeyError:
                field_type = FieldType.STRING

            field_builder = (
                FieldBuilder(field_def.get("name", "field"))
                .display_name(field_def.get("display_name", field_def.get("name", "")))
                .type(field_type)
                .description(field_def.get("description", ""))
                .required(field_def.get("required", False))
            )

            if field_def.get("examples"):
                field_builder.examples(field_def["examples"])

            if field_def.get("pattern"):
                field_builder.pattern(field_def["pattern"])

            if field_def.get("location_hint"):
                field_builder.location_hint(field_def["location_hint"])

            if field_def.get("min_value") is not None:
                field_builder.min_value(field_def["min_value"])

            if field_def.get("max_value") is not None:
                field_builder.max_value(field_def["max_value"])

            if field_def.get("allowed_values"):
                field_builder.allowed_values(field_def["allowed_values"])

            if field_def.get("nested_schema"):
                field_builder.nested(field_def["nested_schema"])

            if field_def.get("list_item_type"):
                list_type_str = field_def["list_item_type"].upper()
                try:
                    list_type = FieldType[list_type_str]
                    field_builder.list_of(list_type)
                except KeyError:
                    pass

            builder.field(field_builder)

        # Build cross-field rules
        for rule_def in schema_def.get("rules", []):
            source = rule_def.get("source_field", "")
            target = rule_def.get("target_field", "")
            operator_str = rule_def.get("operator", "equals").upper()

            try:
                operator = RuleOperator[operator_str]
            except KeyError:
                operator = RuleOperator.EQUALS

            rule_builder = RuleBuilder(source, target)

            # Set operator using fluent API
            operator_method_map = {
                RuleOperator.EQUALS: rule_builder.equals,
                RuleOperator.NOT_EQUALS: rule_builder.not_equals,
                RuleOperator.GREATER_THAN: rule_builder.greater_than,
                RuleOperator.LESS_THAN: rule_builder.less_than,
                RuleOperator.GREATER_EQUAL: rule_builder.greater_or_equal,
                RuleOperator.LESS_EQUAL: rule_builder.less_or_equal,
                RuleOperator.DATE_BEFORE: rule_builder.date_before,
                RuleOperator.DATE_AFTER: rule_builder.date_after,
                RuleOperator.REQUIRES: rule_builder.requires,
                RuleOperator.REQUIRES_IF: rule_builder.requires_if,
            }

            if operator in operator_method_map:
                operator_method_map[operator]()

            if rule_def.get("error_message"):
                rule_builder.error(rule_def["error_message"])

            builder.rule(rule_builder)

        return builder.build()

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
        Perform a single extraction pass.

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
        # Build system prompt with grounding rules
        system_prompt = build_grounded_system_prompt(
            additional_context=build_hallucination_warning(document_type),
            include_forbidden=True,
            include_confidence_scale=True,
        )

        # Build extraction prompt
        if is_first_pass:
            prompt = build_extraction_prompt(
                schema_fields=field_defs,
                document_type=document_type,
                page_number=page_number,
                total_pages=total_pages,
                is_first_pass=True,
            )
        else:
            prompt = build_verification_prompt(
                schema_fields=field_defs,
                document_type=document_type,
                page_number=page_number,
                first_pass_results={},  # Don't show first pass to ensure independence
            )

        try:
            return self.send_vision_request_with_json(
                image_data=image_data,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
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

        Args:
            page_extractions: List of serialized PageExtraction dicts.
            schema: Document schema.

        Returns:
            Merged extraction dictionary.
        """
        merged: dict[str, Any] = {}

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

        # For multi-page, merge based on confidence
        for page in page_extractions:
            for field_name, field_data in page.get("merged_fields", {}).items():
                current_value = field_data.get("value")
                current_confidence = field_data.get("confidence", 0.0)

                if field_name not in merged:
                    merged[field_name] = {
                        "value": current_value,
                        "confidence": current_confidence,
                        "source_page": page.get("page_number", 1),
                    }
                else:
                    # Keep value with higher confidence
                    existing_confidence = merged[field_name].get("confidence", 0.0)
                    if current_confidence > existing_confidence and current_value is not None:
                        merged[field_name] = {
                            "value": current_value,
                            "confidence": current_confidence,
                            "source_page": page.get("page_number", 1),
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
