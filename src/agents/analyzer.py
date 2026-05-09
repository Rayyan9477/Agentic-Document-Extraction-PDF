"""
Analyzer Agent for document classification and schema selection.

Responsible for:
- Classifying document type (CMS-1500, UB-04, EOB, Superbill, Other)
- Detecting document structure (tables, forms, handwriting)
- Analyzing page relationships for multi-page documents
- Selecting appropriate extraction schema
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.base import AgentResult, AnalysisError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import (
    DocumentAnalysis,
    ExtractionState,
    ExtractionStatus,
    add_warning,
    set_status,
    update_state,
)
from src.profiles import detect_profile, get_profile
from src.prompts.classification import (
    DOCUMENT_TYPE_DESCRIPTIONS,
    build_classification_prompt,
    build_page_relationship_prompt,
    build_structure_analysis_prompt,
)
from src.prompts.grounding_rules import (
    build_grounded_system_prompt,
)
from src.schemas import DocumentType, SchemaRegistry


# ---------------------------------------------------------------------------
# V3 Phase 1 — schemas for constrained-decode calls
# ---------------------------------------------------------------------------


class DocumentClassification(BaseModel):
    """Schema bound at decode time for the classification VLM call.

    Field shapes mirror the legacy unconstrained response so the
    surrounding code in ``_classify_document`` continues to work
    unchanged. ``document_type`` accepts free-form strings (the
    downstream ``_normalize_document_type`` mapping handles aliases like
    ``HCFA-1500`` → ``CMS_1500``); we deliberately do not enum-restrict
    here because the model occasionally returns a synonym we want to
    map rather than refuse.
    """

    document_type: str = Field(
        description="Detected document type (CMS-1500, UB-04, EOB, Superbill, OTHER)."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported classification confidence in [0, 1].",
    )
    reasoning: str = Field(
        default="",
        description="Brief justification of the classification decision.",
    )
    key_features_found: list[str] = Field(
        default_factory=list,
        description="Visual or textual features that drove the decision.",
    )
    alternate_types: list[str] = Field(
        default_factory=list,
        description="Other plausible document types, ordered by likelihood.",
    )


class _AnalyzerEnvelope(BaseModel):
    """Permissive envelope for the analyzer's secondary calls.

    Structure analysis and page-relationship calls have varied response
    shapes that are normalized downstream. The envelope guarantees a
    JSON object back, eliminating malformed-output errors without
    forcing every shape into a strict Pydantic model.
    """

    model_config = ConfigDict(extra="allow")


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

        STATE TRANSITION NOTE:
        This agent sets ``status`` to ``ANALYZING`` for observability, but the
        actual workflow transition to the next node (extractor) is controlled by
        the LangGraph workflow edges defined in ``orchestrator.py``. The status
        field is informational only - the workflow graph determines execution
        order via its edge definitions (e.g., ``add_edge(NODE_ANALYZE, NODE_EXTRACT)``).

        This pattern applies to all agents:
        - AnalyzerAgent sets ANALYZING -> workflow routes to ExtractorAgent
        - ExtractorAgent sets EXTRACTING -> workflow routes to ValidatorAgent
        - ValidatorAgent sets VALIDATING -> workflow routes to RouteNode

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

            # Perform classification with multi-page fallback
            classification_result = self._classify_document(image_data)

            # If first-page confidence is below threshold and we have more pages,
            # try the next pages — page 1 may be a fax cover sheet or header
            if (
                classification_result.get("confidence", 0.0) < self._confidence_threshold
                and len(page_images) > 1
            ):
                self._logger.info(
                    "low_confidence_first_page_trying_additional_pages",
                    first_page_confidence=classification_result.get("confidence", 0.0),
                    threshold=self._confidence_threshold,
                )
                for fallback_page in page_images[1:3]:  # Try up to 2 more pages
                    fallback_image = fallback_page.get("data_uri") or fallback_page.get("base64_encoded", "")
                    if not fallback_image:
                        continue
                    alt_result = self._classify_document(fallback_image)
                    if alt_result.get("confidence", 0.0) > classification_result.get("confidence", 0.0):
                        self._logger.info(
                            "better_classification_from_later_page",
                            page=fallback_page.get("page_number"),
                            new_confidence=alt_result.get("confidence", 0.0),
                            new_type=alt_result.get("document_type"),
                        )
                        classification_result = alt_result
                        image_data = fallback_image  # Use this page for structure analysis too
                    if classification_result.get("confidence", 0.0) >= self._confidence_threshold:
                        break

            # Perform structure analysis on the best-classified page
            structure_result = self._analyze_structure(image_data)

            # Analyze page relationships if multi-page
            page_relationships = {}
            if len(page_images) > 1:
                page_relationships = self._analyze_page_relationships(
                    page_images,
                    classification_result.get("document_type", "OTHER"),
                )

            # V3 Phase 5: profile detection. Runs as a pure-text
            # post-step on the analyzer's already-collected outputs;
            # adds zero VLM calls. The selected profile drives prompt
            # fragment injection, schema overlay, and validator
            # blocking/advisory mode downstream. When detection is
            # disabled in settings, every doc resolves to the
            # configured ``default_profile`` (typically generic).
            settings = get_settings()
            if settings.profile.detection_enabled:
                profile_result = detect_profile(
                    classification_features=classification_result.get(
                        "key_features", []
                    ),
                    page_text=first_page.get("text_content") or "",
                    document_type=classification_result.get("document_type"),
                    profile_override=state.get("profile_override"),
                )
            else:
                # Detection disabled — synthesise a deterministic
                # result naming the configured default. This keeps
                # downstream code simple (it always reads
                # ``state["profile"]`` etc.) without a None branch.
                profile_result = type(
                    "ProfileDetectionResult",
                    (),
                    {
                        "profile_name": settings.profile.default_profile,
                        "confidence": 1.0,
                        "score_by_profile": {settings.profile.default_profile: 1.0},
                        "matched_signals": {settings.profile.default_profile: ["detection_disabled"]},
                        "fallback_to_generic": False,
                    },
                )()

            self._logger.info(
                "profile_detected",
                profile=profile_result.profile_name,
                confidence=profile_result.confidence,
                fallback=profile_result.fallback_to_generic,
                matched=profile_result.matched_signals.get(
                    profile_result.profile_name, []
                ),
            )

            # Select schema (profile-aware: applies overlay if present
            # and ``settings.profile.apply_overlay`` is True).
            schema_result = self._select_schema(
                classification_result,
                state.get("custom_schema"),
                profile_name=profile_result.profile_name,
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

            # WS-3: derive specialized modalities (printed / handwritten /
            # table / form / fax / visual) from the structure detections plus
            # per-page image-quality metrics if preprocessing populated them.
            # User override (state["modality_override"]) wins where present.
            from src.agents.modality import apply_overrides, derive_modalities

            # Carry table_count + layout_type + text_density into the analysis
            # if the structure pass surfaced them, so derive_modalities can
            # consume them without needing the raw structure dict.
            for extra_key in ("table_count", "layout_type", "text_density"):
                if extra_key in structure_result:
                    analysis[extra_key] = structure_result[extra_key]  # type: ignore[literal-required]

            auto_modalities = derive_modalities(
                analysis=dict(analysis),
                quality_metrics=state.get("image_quality") or None,
            )
            user_override = state.get("modality_override") or []
            final_modalities = apply_overrides(auto_modalities, user_override)

            analysis["modalities"] = final_modalities
            analysis["modalities_source"] = (
                "auto"
                if not user_override
                else "user_override"
                if set(user_override) >= set(auto_modalities)
                else "auto_with_override"
            )

            self._logger.info(
                "modalities_derived",
                auto=auto_modalities,
                override=user_override,
                final=final_modalities,
                source=analysis["modalities_source"],
            )

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
                    # WS-3: surface derived modalities at top-level so the
                    # extractor / image enhancer / prompt builder can read
                    # them without walking into analysis.
                    "modalities": final_modalities,
                    # V3 Phase 5: profile context surfaced top-level
                    # for downstream prompt/validator/exporter consumers.
                    "profile": profile_result.profile_name,
                    "profile_confidence": profile_result.confidence,
                    "profile_signals_matched": profile_result.matched_signals.get(
                        profile_result.profile_name, []
                    ),
                    "profile_fallback_to_generic": profile_result.fallback_to_generic,
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
        Classify the document type using VLM with enhanced prompts and retry logic.

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
            include_chain_of_thought=True,  # Enhanced: Add reasoning protocol
        )

        # Use enhanced classification prompt with few-shot examples and step-by-step reasoning
        classification_prompt = build_classification_prompt(
            include_confidence=True,
            include_reasoning=True,
            include_examples=True,  # Enhanced: Add few-shot examples
            include_step_by_step=True,  # Enhanced: Add step-by-step protocol
        )

        # Use retry with exponential backoff for VLM calls
        settings = get_settings()
        retry_config = RetryConfig(
            max_retries=settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=settings.agent.max_retry_delay_ms,
        )

        def make_classification_call() -> dict[str, Any]:
            # V3 Phase 1: schema-bound call. The decoder cannot emit
            # JSON outside ``DocumentClassification`` so the legacy
            # markdown-codeblock + retry-on-malformed dance is gone.
            payload, _trace = self.send_vision_request_with_schema(
                image_data=image_data,
                prompt=classification_prompt,
                schema=DocumentClassification,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            return payload

        try:
            result = retry_with_backoff(
                func=make_classification_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "classification_retry",
                    attempt=attempt + 1,
                    error=str(e),
                ),
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
            # V3 Phase 1: schema-bound (permissive envelope).
            result, _trace = self.send_vision_request_with_schema(
                image_data=image_data,
                prompt=structure_prompt,
                schema=_AnalyzerEnvelope,
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
        Analyze relationships between pages in multi-page documents using VLM.

        Uses vision language model to determine each page's role and relationship
        to other pages in the document. Detects continuation markers, page numbers,
        and unique content on each page.

        Args:
            page_images: List of page image data with 'image' key containing
                        base64-encoded image data.
            document_type: Classified document type for context.

        Returns:
            Mapping of page number to detailed relationship description.
        """
        relationships = {}
        total_pages = len(page_images)

        # For single-page documents, no relationship analysis needed
        if total_pages == 1:
            return {1: "single_page"}

        # Build system prompt for page analysis
        system_prompt = build_grounded_system_prompt(
            additional_context=(
                f"You are analyzing pages of a {document_type} document "
                f"that has {total_pages} total pages. Determine each page's "
                "role and relationship to other pages."
            ),
            include_forbidden=False,
            include_confidence_scale=False,
        )

        # Get the page relationship prompt
        relationship_prompt = build_page_relationship_prompt(total_pages)

        for i, page in enumerate(page_images, start=1):
            image_data = page.get("data_uri") or page.get("base64_encoded", "")

            if not image_data:
                # Fallback to basic role assignment if no image
                if i == 1:
                    relationships[i] = "primary"
                elif i == total_pages:
                    relationships[i] = "final"
                else:
                    relationships[i] = "continuation"
                continue

            try:
                # VLM call counter is auto-incremented by send_vision_request_with_json()

                # Add page context to prompt
                page_prompt = f"This is page {i} of {total_pages}.\n\n" f"{relationship_prompt}"

                # V3 Phase 1: schema-bound page-relationship VLM call.
                result, _trace = self.send_vision_request_with_schema(
                    image_data=image_data,
                    prompt=page_prompt,
                    schema=_AnalyzerEnvelope,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )

                # Extract relationship information from VLM response
                page_role = result.get("page_role", "unknown")
                continues_from = result.get("continues_from_previous", False)
                continues_to = result.get("continues_to_next", False)
                unique_content = result.get("unique_content", [])
                relationship_notes = result.get("relationship_notes", "")

                # Build detailed relationship string
                relationship_parts = [page_role]

                if continues_from:
                    relationship_parts.append("continues_from_previous")
                if continues_to:
                    relationship_parts.append("continues_to_next")

                if unique_content:
                    # Summarize unique content types
                    content_summary = ", ".join(unique_content[:3])  # Limit to 3 items
                    relationship_parts.append(f"contains:{content_summary}")

                if relationship_notes:
                    # Add abbreviated notes
                    notes_summary = relationship_notes[:50]
                    if len(relationship_notes) > 50:
                        notes_summary += "..."
                    relationship_parts.append(f"notes:{notes_summary}")

                relationships[i] = " | ".join(relationship_parts)

                self._logger.debug(
                    "page_relationship_analyzed",
                    page=i,
                    total_pages=total_pages,
                    role=page_role,
                    continues_from=continues_from,
                    continues_to=continues_to,
                )

            except Exception as e:
                # On VLM error, fall back to basic role assignment
                self._logger.warning(
                    "page_relationship_analysis_failed",
                    page=i,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                if i == 1:
                    relationships[i] = "primary (fallback)"
                elif i == total_pages:
                    relationships[i] = "final (fallback)"
                else:
                    relationships[i] = "continuation (fallback)"

        return relationships

    def _select_schema(
        self,
        classification: dict[str, Any],
        custom_schema: dict[str, Any] | None,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Select appropriate extraction schema based on classification.

        Args:
            classification: Document classification result.
            custom_schema: Optional user-provided custom schema.
            profile_name: V3 Phase 5 — selected profile for overlay
                application. When provided, the profile's
                ``schema_overlay_fields`` are merged into the resolved
                schema (de-duplicated by field name) before returning.

        Returns:
            Schema selection result.
        """
        # If custom schema provided, use it
        if custom_schema:
            return {
                "selected_schema": custom_schema.get("name", "custom_schema"),
                "selection_reason": "Custom schema provided by user",
                "schema_compatibility": 1.0,
                "profile_overlay_applied": False,
            }

        # Map document type to schema
        doc_type_str = classification.get("document_type", "OTHER")
        doc_type = DOCUMENT_TYPE_MAP.get(doc_type_str, DocumentType.UNKNOWN)

        # Get schema from registry
        overlay_applied = False
        try:
            schema = self._schema_registry.get_by_type(doc_type)
            # Apply profile overlay if configured.
            if profile_name:
                from src.schemas.profile_overlays import apply_overlay

                profile_descriptor = get_profile(profile_name)
                settings = get_settings()
                if settings.profile.apply_overlay and profile_descriptor.schema_overlay_fields:
                    overlaid = apply_overlay(schema, profile_descriptor)
                    overlay_applied = overlaid is not schema
                    schema = overlaid

            return {
                "selected_schema": schema.name,
                "selection_reason": f"Matched schema for {doc_type.value}",
                "schema_compatibility": classification.get("confidence", 0.8),
                "profile_overlay_applied": overlay_applied,
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
                "profile_overlay_applied": False,
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
