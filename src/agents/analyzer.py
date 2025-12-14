"""
Analyzer Agent for document classification and schema selection.

Responsible for:
- Classifying document type (CMS-1500, UB-04, EOB, Superbill, Other)
- Detecting document structure (tables, forms, handwriting)
- Analyzing page relationships for multi-page documents
- Selecting appropriate extraction schema
"""

from typing import Any

from src.agents.base import BaseAgent, AgentResult, AnalysisError
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    DocumentAnalysis,
    update_state,
    set_status,
    add_warning,
)
from src.prompts.grounding_rules import build_grounded_system_prompt
from src.prompts.classification import (
    build_classification_prompt,
    build_structure_analysis_prompt,
    build_page_relationship_prompt,
    build_schema_selection_prompt,
    DOCUMENT_TYPE_DESCRIPTIONS,
)
from src.schemas import SchemaRegistry, DocumentType


logger = get_logger(__name__)


# Mapping from classification output to DocumentType enum
DOCUMENT_TYPE_MAP = {
    "CMS-1500": DocumentType.CMS_1500,
    "CMS1500": DocumentType.CMS_1500,
    "HCFA-1500": DocumentType.CMS_1500,
    "UB-04": DocumentType.UB_04,
    "UB04": DocumentType.UB_04,
    "CMS-1450": DocumentType.UB_04,
    "EOB": DocumentType.EOB,
    "EXPLANATION OF BENEFITS": DocumentType.EOB,
    "SUPERBILL": DocumentType.SUPERBILL,
    "ENCOUNTER FORM": DocumentType.SUPERBILL,
    "OTHER": DocumentType.UNKNOWN,
    "UNKNOWN": DocumentType.UNKNOWN,
}


class AnalyzerAgent(BaseAgent):
    """
    Document analysis agent for classification and schema selection.

    Performs initial document analysis to determine:
    - Document type (CMS-1500, UB-04, EOB, Superbill, Other)
    - Document structure (tables, forms, handwriting areas)
    - Page relationships for multi-page documents
    - Appropriate extraction schema

    VLM Calls: 1 per document (first page classification)
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        classification_confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the Analyzer agent.

        Args:
            client: Optional pre-configured LM Studio client.
            classification_confidence_threshold: Minimum confidence for classification.
        """
        super().__init__(name="analyzer", client=client)
        self._confidence_threshold = classification_confidence_threshold
        self._schema_registry = SchemaRegistry()

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Analyze document and update state with classification results.

        This is the main entry point for the LangGraph workflow.

        Args:
            state: Current extraction state.

        Returns:
            Updated state with analysis results.
        """
        # Reset metrics to prevent accumulation across documents
        self.reset_metrics()

        start_time = self.log_operation_start(
            "document_analysis",
            processing_id=state.get("processing_id", ""),
            page_count=len(state.get("page_images", [])),
        )

        try:
            # Update status
            state = set_status(state, ExtractionStatus.ANALYZING, "classifying")

            # Get first page for classification
            page_images = state.get("page_images", [])
            if not page_images:
                raise AnalysisError(
                    "No page images available for analysis",
                    agent_name=self.name,
                    recoverable=False,
                )

            first_page = page_images[0]
            image_data = first_page.get("data_uri") or first_page.get("base64_encoded", "")

            if not image_data:
                raise AnalysisError(
                    "First page has no image data",
                    agent_name=self.name,
                    recoverable=False,
                )

            # Perform classification
            classification_result = self._classify_document(image_data)

            # Perform structure analysis
            structure_result = self._analyze_structure(image_data)

            # Analyze page relationships if multi-page
            page_relationships = {}
            if len(page_images) > 1:
                page_relationships = self._analyze_page_relationships(
                    page_images,
                    classification_result.get("document_type", "OTHER"),
                )

            # Select schema
            schema_result = self._select_schema(
                classification_result,
                state.get("custom_schema"),
            )

            # Build analysis result
            analysis: DocumentAnalysis = {
                "document_type": classification_result.get("document_type", "OTHER"),
                "document_type_confidence": classification_result.get("confidence", 0.0),
                "schema_name": schema_result.get("selected_schema", ""),
                "detected_structures": structure_result.get("structures", []),
                "has_tables": structure_result.get("has_tables", False),
                "has_handwriting": structure_result.get("has_handwriting", False),
                "has_signatures": structure_result.get("has_signatures", False),
                "page_relationships": page_relationships,
                "regions_of_interest": structure_result.get("regions_of_interest", []),
                "analysis_time_ms": 0,  # Will be updated below
            }

            # Calculate processing time
            duration_ms = self.log_operation_complete(
                "document_analysis",
                start_time,
                success=True,
                document_type=analysis["document_type"],
                confidence=analysis["document_type_confidence"],
            )

            analysis["analysis_time_ms"] = duration_ms

            # Update state with results
            state = update_state(
                state,
                {
                    "analysis": analysis,
                    "document_type": analysis["document_type"],
                    "selected_schema_name": analysis["schema_name"],
                    "status": ExtractionStatus.ANALYZING.value,
                    "current_step": "analysis_complete",
                    "total_vlm_calls": state.get("total_vlm_calls", 0) + self._vlm_calls,
                },
            )

            # Add warning if low confidence
            if analysis["document_type_confidence"] < self._confidence_threshold:
                state = add_warning(
                    state,
                    f"Document classification confidence ({analysis['document_type_confidence']:.2f}) "
                    f"below threshold ({self._confidence_threshold})",
                )

            return state

        except AnalysisError:
            raise
        except Exception as e:
            self.log_operation_complete("document_analysis", start_time, success=False)
            raise AnalysisError(
                f"Document analysis failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

    def _classify_document(self, image_data: str) -> dict[str, Any]:
        """
        Classify the document type using VLM.

        Args:
            image_data: Base64-encoded image or data URI.

        Returns:
            Classification result with document_type and confidence.
        """
        self._logger.debug("classifying_document")

        system_prompt = build_grounded_system_prompt(
            additional_context=(
                "You are classifying a medical/healthcare document. "
                "Focus on identifying the document type based on visual layout and structure."
            ),
            include_forbidden=False,
            include_confidence_scale=True,
        )

        classification_prompt = build_classification_prompt(
            include_confidence=True,
            include_reasoning=True,
        )

        try:
            result = self.send_vision_request_with_json(
                image_data=image_data,
                prompt=classification_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

            # Normalize document type
            raw_type = result.get("document_type", "OTHER").upper().strip()
            normalized_type = self._normalize_document_type(raw_type)

            return {
                "document_type": normalized_type,
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "key_features": result.get("key_features_found", []),
                "alternate_types": result.get("alternate_types", []),
            }

        except Exception as e:
            self._logger.warning(
                "classification_fallback",
                error=str(e),
            )
            # Return default classification on failure
            return {
                "document_type": "OTHER",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {e}",
                "key_features": [],
                "alternate_types": [],
            }

    def _analyze_structure(self, image_data: str) -> dict[str, Any]:
        """
        Analyze document structure using VLM.

        Detects structural elements in the document including:
        - Tables and their locations
        - Form fields and checkboxes
        - Handwritten vs printed text
        - Signature areas
        - Headers/footers
        - Barcodes/QR codes

        Args:
            image_data: Base64-encoded image or data URI.

        Returns:
            Structure analysis result with detected elements.
        """
        self._logger.debug("analyzing_document_structure")

        system_prompt = build_grounded_system_prompt(
            additional_context=(
                "You are analyzing the visual structure of a document. "
                "Focus on identifying structural elements like tables, form fields, "
                "handwritten areas, signatures, and regions of interest. "
                "Be precise about what you can see."
            ),
            include_forbidden=False,
            include_confidence_scale=False,
        )

        structure_prompt = build_structure_analysis_prompt()

        try:
            result = self.send_vision_request_with_json(
                image_data=image_data,
                prompt=structure_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

            # Normalize and validate the result
            structures = result.get("structures", [])
            if not isinstance(structures, list):
                structures = []

            # Ensure boolean values
            has_tables = bool(result.get("has_tables", False))
            has_handwriting = bool(result.get("has_handwriting", False))
            has_signatures = bool(result.get("has_signatures", False))
            has_barcodes = bool(result.get("has_barcodes", False))

            # Get regions of interest with validation
            regions = result.get("regions_of_interest", [])
            if not isinstance(regions, list):
                regions = []

            # Get table count if available
            table_count = result.get("table_count", 1 if has_tables else 0)
            if not isinstance(table_count, int):
                table_count = 0

            return {
                "structures": structures,
                "has_tables": has_tables,
                "table_count": table_count,
                "has_handwriting": has_handwriting,
                "has_signatures": has_signatures,
                "has_barcodes": has_barcodes,
                "regions_of_interest": regions,
                "detected_fields": result.get("detected_fields", []),
                "layout_type": result.get("layout_type", "form"),  # form, letter, report
                "text_density": result.get("text_density", "medium"),  # low, medium, high
            }

        except Exception as e:
            self._logger.warning(
                "structure_analysis_fallback",
                error=str(e),
            )
            # Return conservative defaults on failure
            return {
                "structures": ["form_fields", "text_blocks"],
                "has_tables": True,
                "table_count": 1,
                "has_handwriting": False,
                "has_signatures": False,
                "has_barcodes": False,
                "regions_of_interest": [],
                "detected_fields": [],
                "layout_type": "form",
                "text_density": "medium",
            }

    def _analyze_page_relationships(
        self,
        page_images: list[dict[str, Any]],
        document_type: str,
    ) -> dict[int, str]:
        """
        Analyze relationships between pages in multi-page documents.

        Args:
            page_images: List of page image data.
            document_type: Classified document type.

        Returns:
            Mapping of page number to relationship description.
        """
        relationships = {}
        total_pages = len(page_images)

        for i, page in enumerate(page_images, start=1):
            if i == 1:
                relationships[i] = "primary"
            elif i == total_pages:
                relationships[i] = "final"
            else:
                relationships[i] = "continuation"

        return relationships

    def _select_schema(
        self,
        classification: dict[str, Any],
        custom_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Select appropriate extraction schema based on classification.

        Args:
            classification: Document classification result.
            custom_schema: Optional user-provided custom schema.

        Returns:
            Schema selection result.
        """
        # If custom schema provided, use it
        if custom_schema:
            return {
                "selected_schema": custom_schema.get("name", "custom_schema"),
                "selection_reason": "Custom schema provided by user",
                "schema_compatibility": 1.0,
            }

        # Map document type to schema
        doc_type_str = classification.get("document_type", "OTHER")
        doc_type = DOCUMENT_TYPE_MAP.get(doc_type_str, DocumentType.UNKNOWN)

        # Get schema from registry
        try:
            schema = self._schema_registry.get_by_type(doc_type)
            return {
                "selected_schema": schema.name,
                "selection_reason": f"Matched schema for {doc_type.value}",
                "schema_compatibility": classification.get("confidence", 0.8),
            }
        except ValueError:
            # No schema found for type
            self._logger.warning(
                "no_schema_for_type",
                document_type=doc_type_str,
            )
            return {
                "selected_schema": "",
                "selection_reason": f"No schema registered for {doc_type_str}",
                "schema_compatibility": 0.0,
            }

    def _normalize_document_type(self, raw_type: str) -> str:
        """
        Normalize document type string to standard format.

        Args:
            raw_type: Raw document type from VLM.

        Returns:
            Normalized document type string.
        """
        # Clean up the type string
        cleaned = raw_type.upper().strip()
        cleaned = cleaned.replace("-", "").replace("_", "").replace(" ", "")

        # Map common variations
        type_map = {
            "CMS1500": "CMS-1500",
            "HCFA1500": "CMS-1500",
            "UB04": "UB-04",
            "CMS1450": "UB-04",
            "EXPLANATIONOFBENEFITS": "EOB",
            "ENCOUNTERFORM": "SUPERBILL",
        }

        return type_map.get(cleaned, raw_type.upper())

    def classify_document_standalone(
        self,
        image_data: str,
    ) -> AgentResult[dict[str, Any]]:
        """
        Classify a document without full pipeline processing.

        Useful for quick classification without extraction.

        Args:
            image_data: Base64-encoded image or data URI.

        Returns:
            AgentResult with classification data.
        """
        start_time = self.log_operation_start("standalone_classification")

        try:
            result = self._classify_document(image_data)

            duration_ms = self.log_operation_complete(
                "standalone_classification",
                start_time,
                success=True,
                document_type=result.get("document_type"),
            )

            return AgentResult.ok(
                data=result,
                agent_name=self.name,
                operation="classify",
                vlm_calls=self._vlm_calls,
                processing_time_ms=duration_ms,
            )

        except Exception as e:
            self.log_operation_complete(
                "standalone_classification",
                start_time,
                success=False,
            )
            return AgentResult.fail(
                error=str(e),
                agent_name=self.name,
                operation="classify",
            )

    def get_supported_document_types(self) -> list[str]:
        """Get list of supported document types."""
        return list(DOCUMENT_TYPE_DESCRIPTIONS.keys())

    def get_available_schemas(self) -> list[str]:
        """Get list of available extraction schemas."""
        return self._schema_registry.list_schema_names()
