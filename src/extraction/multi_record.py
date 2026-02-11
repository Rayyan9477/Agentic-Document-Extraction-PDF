"""
Multi-record extraction for documents containing multiple entities per page.

Handles documents like:
- Medical superbills (multiple patients per page)
- Patient lists / rosters
- Invoice batches
- Employee records
- Any tabular or list-based multi-entity document

Flow:
  1. Detect document type + entity type (1 VLM call)
  2. Generate adaptive schema per entity (1 VLM call)
  3. Per page: detect record boundaries (1 VLM call)
  4. Per record: extract fields (1 VLM call per record)
  5. [Phase 2] Per record: validate extraction (1 VLM call, optional)
  6. [Phase 2] Per record: correct low-confidence fields (0-1 VLM call, optional)
  7. [Phase 3] Per record: consensus for critical fields (2 + 0-K VLM calls, optional)

Total VLM calls (Phase 1): 2 + pages * (1 + records_per_page)
Total VLM calls (Phase 2): 2 + pages * (1 + records_per_page * [1 + validation + correction])
Total VLM calls (Phase 3): Phase 2 + pages * records_per_page * (2 + disagreements)
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.client.lm_client import LMStudioClient, VisionRequest
from src.config import get_logger
from src.prompts.grounding_rules import build_grounded_system_prompt
from src.utils.date_utils import parse_date
from src.utils.string_utils import clean_currency

logger = get_logger(__name__)


@dataclass
class RecordBoundary:
    """Detected record boundary on a page."""

    record_id: int
    primary_identifier: str
    bounding_box: dict[str, float]
    visual_separator: str
    entity_type: str


@dataclass
class ExtractedRecord:
    """Single extracted record with all fields."""

    record_id: int
    page_number: int
    primary_identifier: str
    entity_type: str
    fields: dict[str, Any]
    confidence: float
    extraction_time_ms: int


@dataclass
class DocumentExtractionResult:
    """Complete multi-page, multi-record extraction result."""

    pdf_path: str
    total_pages: int
    total_records: int
    document_type: str
    entity_type: str
    records: list[ExtractedRecord]
    schema: dict[str, Any]
    total_processing_time_ms: int
    total_vlm_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "total_records": self.total_records,
            "document_type": self.document_type,
            "entity_type": self.entity_type,
            "schema": self.schema,
            "records": [asdict(r) for r in self.records],
            "total_processing_time_ms": self.total_processing_time_ms,
            "total_vlm_calls": self.total_vlm_calls,
        }


class MultiRecordExtractor:
    """
    Extracts multiple distinct records from each page of a document.

    Unlike the single-record pipeline that treats each page as one entity,
    this extractor detects individual record boundaries and extracts each
    record separately, producing one output row per entity (e.g., per patient).

    Uses the existing LMStudioClient for all VLM communication.
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        enable_validation: bool = False,
        enable_self_correction: bool = False,
        confidence_threshold: float = 0.85,
        enable_consensus: bool = False,
        critical_field_keywords: list[str] | None = None,
    ) -> None:
        self._client = client or LMStudioClient()
        self._vlm_calls = 0
        self._enable_validation = enable_validation
        self._enable_self_correction = enable_self_correction
        self._confidence_threshold = confidence_threshold
        self._enable_consensus = enable_consensus
        self._critical_field_keywords = critical_field_keywords or [
            "id", "number", "code", "mrn", "ssn",
            "date", "dob", "amount", "charge", "total", "balance",
        ]

    def _build_grounding_system_prompt(self) -> str:
        """System-level grounding rules for all VLM calls.

        Delegates to the shared grounding_rules module for consistent
        anti-hallucination rules across all extraction pipelines.
        """
        return build_grounded_system_prompt(
            additional_context=(
                "REASONING PROCESS:\n"
                "- First describe what you see (layout, visual elements)\n"
                "- Then identify patterns and repeating elements\n"
                "- Finally classify/extract based on concrete visual evidence"
            ),
            include_forbidden=True,
            include_confidence_scale=False,
            include_chain_of_thought=False,
            include_few_shot_examples=False,
            include_self_verification=False,
            include_constitutional_critique=False,
        )

    @staticmethod
    def _format_bbox(bbox: dict[str, float]) -> str:
        """Format a bounding box dict as a human-readable string for prompts."""
        return (
            f"Top {bbox.get('top', 0):.0%} to Bottom {bbox.get('bottom', 1):.0%}"
        )

    def _get_adaptive_temperature(
        self,
        field_type: str = "text",
        field_name: str = "",
        retry_count: int = 0,
    ) -> float:
        """Determine optimal temperature based on field characteristics."""

        # Exact fields: deterministic
        if any(kw in field_name.lower() for kw in ["id", "number", "code", "ssn", "mrn"]):
            return 0.0

        # Dates: low temperature for structure
        if field_type == "date":
            return 0.05

        # Amounts: low temperature
        if any(kw in field_name.lower() for kw in ["amount", "charge", "total", "balance"]):
            return 0.03

        # Free text: slightly higher for natural variation
        if field_type == "text" and any(
            kw in field_name.lower() for kw in ["description", "note", "comment"]
        ):
            return 0.15

        # Base temperature with retry escalation
        base = 0.1
        if retry_count > 0:
            base = min(0.1 + retry_count * 0.05, 0.3)

        return base

    def _identify_critical_fields(
        self,
        schema: dict[str, Any],
    ) -> list[str]:
        """Identify schema fields that are critical based on keyword matching."""
        critical = []
        for field_def in schema.get("fields", []):
            name_lower = field_def["field_name"].lower()
            if any(kw in name_lower for kw in self._critical_field_keywords):
                critical.append(field_def["field_name"])
        return critical

    # ── Field Type Validation & Coercion ─────────────────────────

    @staticmethod
    def _coerce_number(value: Any) -> Any:
        """Coerce a value to a numeric type using clean_currency.

        Returns the original value if coercion fails (preserves raw extraction).
        """
        if isinstance(value, (int, float)):
            return value
        result = clean_currency(str(value))
        if result is not None:
            return float(result)
        return value

    @staticmethod
    def _normalize_date(value: Any) -> Any:
        """Normalize a date string to ISO format using parse_date.

        Returns the original value if parsing fails (preserves raw extraction).
        """
        if not isinstance(value, str):
            return value
        parsed = parse_date(value)
        if parsed is not None:
            return parsed.isoformat()
        return value

    @staticmethod
    def _coerce_boolean(value: Any) -> Any:
        """Coerce a value to boolean.

        Returns the original value if coercion is ambiguous.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "yes", "1", "checked", "x"):
                return True
            if lower in ("false", "no", "0", "unchecked", ""):
                return False
        return value

    def _validate_field_types(
        self,
        record: ExtractedRecord,
        schema: dict[str, Any],
    ) -> ExtractedRecord:
        """Validate and coerce extracted field values against schema types.

        Runs post-extraction to normalize values (e.g. "$1,234.56" -> 1234.56,
        "01/15/2024" -> "2024-01-15"). Preserves original value if coercion fails.
        """
        for field_def in schema.get("fields", []):
            name = field_def["field_name"]
            ftype = field_def.get("field_type", "text")
            value = record.fields.get(name)
            if value is None:
                continue
            if ftype == "number":
                record.fields[name] = self._coerce_number(value)
            elif ftype == "date":
                record.fields[name] = self._normalize_date(value)
            elif ftype == "boolean":
                record.fields[name] = self._coerce_boolean(value)
        return record

    def _send_vision_json(
        self,
        image_data: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 3000,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Send vision request with intelligent retry strategy."""
        last_error = None
        original_prompt = prompt  # Save original for retry reformulation

        for attempt in range(max_retries):
            try:
                request = VisionRequest(
                    image_data=image_data,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response = self._client.send_vision_request(request)
                self._vlm_calls += 1

                if response.has_json and response.parsed_json:
                    return response.parsed_json

                # Try manual JSON extraction from raw content
                content = response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)

            except json.JSONDecodeError as e:
                # Malformed JSON - emphasize JSON formatting on next retry
                last_error = e
                logger.warning(
                    "vlm_json_parse_error",
                    attempt=attempt + 1,
                    error=str(e),
                    recovery="Will retry with emphasis on JSON format",
                )
                # Reformulate prompt to emphasize JSON output
                if attempt < max_retries - 1:
                    prompt = original_prompt + "\n\nCRITICAL: Return ONLY valid JSON, no additional text."

            except Exception as e:
                last_error = e
                logger.warning(
                    "vlm_call_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

            # Exponential backoff (instead of fixed 2-second delay)
            if attempt < max_retries - 1:
                delay = min(2**attempt, 10)  # 1s, 2s, 4s, 8s, max 10s
                logger.debug(
                    "retry_backoff", attempt=attempt + 1, delay_seconds=delay
                )
                time.sleep(delay)

        raise RuntimeError(
            f"VLM call failed after {max_retries} attempts: {last_error}"
        )

    def detect_document_type(
        self, page_data_uri: str
    ) -> dict[str, Any]:
        """
        Detect document type, entity type, and primary identifier from first page.

        Args:
            page_data_uri: Data URI of the first page image.

        Returns:
            Dict with document_type, entity_type, primary_identifier_field, etc.
        """
        logger.info("detecting_document_type")

        prompt = """Analyze this document. Think step by step:
1. Observe the overall layout, visual elements, and repeating patterns
2. Identify main headers, field labels, and column headers
3. Count distinct records and determine how they are separated
4. Classify the document based on your observations

Return JSON:
{
  "document_type": "medical_superbill",
  "document_description": "Brief description",
  "entity_type": "patient",
  "entity_description": "What each record represents",
  "primary_identifier_field": "patient_name",
  "record_structure": "table",
  "estimated_records_per_page": 5,
  "confidence": 0.95
}

IMPORTANT: Only report what you DIRECTLY SEE. Do not infer or assume."""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=800,
        )

        logger.info(
            "document_type_detected",
            document_type=result.get("document_type"),
            entity_type=result.get("entity_type"),
            primary_id=result.get("primary_identifier_field"),
            confidence=result.get("confidence"),
        )

        return result

    def generate_schema(
        self,
        page_data_uri: str,
        document_type: str,
        entity_type: str,
    ) -> dict[str, Any]:
        """
        Generate adaptive extraction schema from a sample page.

        Args:
            page_data_uri: Data URI of a representative page.
            document_type: Detected document type.
            entity_type: Detected entity type.

        Returns:
            Schema dict with field definitions.
        """
        logger.info("generating_schema", document_type=document_type)

        prompt = f"""Generate an extraction schema for this document. Think step by step:
1. Inspect how {entity_type} records are organized (table, form, list)
2. Identify every visible field label and its data type
3. Determine formatting patterns (dates, currency, codes)
4. Produce the schema

Document Type: {document_type}
Entity Type: {entity_type}

Return JSON:
{{
  "schema_id": "adaptive_{document_type}",
  "entity_type": "{entity_type}",
  "fields": [
    {{
      "field_name": "field_name",
      "display_name": "Field Name",
      "field_type": "text|number|date|boolean|list",
      "description": "What this field contains",
      "required": true
    }}
  ],
  "total_field_count": 10,
  "confidence": 0.92
}}

CRITICAL: Include EVERY field visible in the records. Base schema ONLY on what you directly observe."""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=3000,
        )

        logger.info(
            "schema_generated",
            field_count=len(result.get("fields", [])),
        )
        return result

    def detect_record_boundaries(
        self,
        page_data_uri: str,
        entity_type: str,
        primary_id_field: str,
        page_number: int,
    ) -> list[RecordBoundary]:
        """
        Detect individual record boundaries on a single page.

        Args:
            page_data_uri: Data URI of the page image.
            entity_type: What each record represents.
            primary_id_field: Field that identifies each record.
            page_number: Page number for logging.

        Returns:
            List of RecordBoundary objects found on this page.
        """
        logger.info(
            "detecting_record_boundaries",
            page=page_number,
            entity_type=entity_type,
        )

        prompt = f"""Identify every individual {entity_type.upper()} record on this page. Think step by step:
1. Scan the page for all visible {entity_type} records
2. Identify what visually separates each record (lines, spacing, borders)
3. Locate the {primary_id_field} value for each record
4. Estimate bounding boxes as page percentages (0.0-1.0)

Entity Type: {entity_type}
Primary Identifier: {primary_id_field}

Return JSON:
{{
  "total_records": 5,
  "records": [
    {{
      "record_id": 1,
      "primary_identifier": "extracted {primary_id_field} value",
      "bounding_box": {{
        "top": 0.15,
        "left": 0.0,
        "bottom": 0.30,
        "right": 1.0
      }},
      "visual_separator": "horizontal line below"
    }}
  ]
}}

CRITICAL: Each unique {primary_id_field} = one record. Report ALL records visible on this page."""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=2000,
        )

        boundaries = []
        for rec in result.get("records", []):
            boundaries.append(
                RecordBoundary(
                    record_id=rec.get("record_id", 0),
                    primary_identifier=rec.get("primary_identifier", "unknown"),
                    bounding_box=rec.get(
                        "bounding_box",
                        {"top": 0, "left": 0, "bottom": 1, "right": 1},
                    ),
                    visual_separator=rec.get("visual_separator", ""),
                    entity_type=entity_type,
                )
            )

        logger.info(
            "boundaries_detected",
            page=page_number,
            count=len(boundaries),
            identifiers=[b.primary_identifier for b in boundaries],
        )
        return boundaries

    def extract_single_record(
        self,
        page_data_uri: str,
        boundary: RecordBoundary,
        schema: dict[str, Any],
        page_number: int,
    ) -> ExtractedRecord:
        """
        Extract all fields for a single record identified by its boundary.

        Args:
            page_data_uri: Data URI of the page image.
            boundary: Record boundary with identifier and location.
            schema: Field schema to extract.
            page_number: Page number.

        Returns:
            ExtractedRecord with all extracted fields.
        """
        primary_id = boundary.primary_identifier
        entity_type = boundary.entity_type

        logger.info(
            "extracting_record",
            page=page_number,
            record_id=boundary.record_id,
            primary_id=primary_id,
        )

        # Build field instructions from schema
        field_lines = []
        for f in schema.get("fields", []):
            field_lines.append(
                f"  - {f['field_name']} ({f.get('field_type', 'text')}): "
                f"{f.get('description', f['field_name'])}"
            )
        field_list = "\n".join(field_lines)

        bbox = boundary.bounding_box
        prompt = f"""Extract data for ONE SPECIFIC {entity_type.upper()}. Think step by step:
1. Locate "{primary_id}" in the bounding box and confirm isolation
2. Read each field value exactly as shown in the image
3. Verify each value matches the expected data type
4. Reject any value that might belong to a different record

TARGET RECORD: {primary_id}
BOUNDING BOX: {self._format_bbox(bbox)}

Fields to extract:
{field_list}

Return JSON:
{{
  "record_id": {boundary.record_id},
  "primary_identifier": "{primary_id}",
  "fields": {{
    "field_name": "extracted_value_or_null"
  }},
  "confidence": 0.92
}}

CRITICAL RULES:
- Extract ONLY from "{primary_id}" within the bounding box
- Return null for any field you cannot clearly read
- NEVER guess or infer values not explicitly visible"""

        start_ms = time.time()

        # Use adaptive temperature (base 0.1 for mixed field types)
        temperature = self._get_adaptive_temperature(
            field_type="mixed",
            field_name="multi_field_extraction",
            retry_count=0,
        )

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=2000,
            temperature=temperature,
        )

        elapsed_ms = int((time.time() - start_ms) * 1000)

        record = ExtractedRecord(
            record_id=boundary.record_id,
            page_number=page_number,
            primary_identifier=result.get("primary_identifier", primary_id),
            entity_type=entity_type,
            fields=result.get("fields", {}),
            confidence=float(result.get("confidence", 0.0)),
            extraction_time_ms=elapsed_ms,
        )

        logger.info(
            "record_extracted",
            page=page_number,
            record_id=record.record_id,
            primary_id=record.primary_identifier,
            field_count=len(record.fields),
            confidence=record.confidence,
        )
        return record

    def _validate_extraction(
        self,
        page_data_uri: str,
        record: ExtractedRecord,
        boundary: RecordBoundary,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate an extracted record by asking the VLM to verify each field
        against the original image.

        Args:
            page_data_uri: Data URI of the page image.
            record: The extracted record to validate.
            boundary: Record boundary with location info.
            schema: Field schema for context.

        Returns:
            Validation result dict with per-field verdicts:
            {
                "overall_valid": bool,
                "overall_confidence": float,
                "field_validations": [
                    {
                        "field_name": str,
                        "original_value": Any,
                        "is_correct": bool,
                        "corrected_value": Any | None,
                        "confidence": float,
                        "issue": str | None,
                    }
                ],
                "fields_needing_correction": [str],
            }
        """
        primary_id = record.primary_identifier
        bbox = boundary.bounding_box

        logger.info(
            "validating_extraction",
            record_id=record.record_id,
            primary_id=primary_id,
            field_count=len(record.fields),
        )

        # Build the extracted values summary for the VLM to check
        field_summary_lines = []
        for field_name, value in record.fields.items():
            display_val = str(value)[:200] if value is not None else "null"
            field_summary_lines.append(f"  - {field_name}: {display_val}")
        field_summary = "\n".join(field_summary_lines)

        prompt = f"""VALIDATION TASK: Verify the accuracy of previously extracted data.

TARGET RECORD: {primary_id}
BOUNDING BOX: {self._format_bbox(bbox)}

The following values were extracted from this record. Your job is to VERIFY
each value by looking at the ORIGINAL IMAGE and checking if they match.

EXTRACTED VALUES:
{field_summary}

For EACH field:
1. Locate the field in the image for record "{primary_id}"
2. Read the actual value from the image
3. Compare with the extracted value above
4. Mark as correct (match) or incorrect (mismatch)
5. If incorrect, provide the corrected value

Return JSON:
{{
  "validation_summary": {{
    "record_identifier": "{primary_id}",
    "total_fields_checked": {len(record.fields)},
    "correct_count": 0,
    "incorrect_count": 0,
    "uncertain_count": 0
  }},
  "field_validations": [
    {{
      "field_name": "field_name",
      "original_value": "what was extracted",
      "actual_value_in_image": "what you see in the image",
      "is_correct": true,
      "corrected_value": null,
      "confidence": 0.95,
      "issue": null
    }}
  ],
  "overall_confidence": 0.92
}}

CRITICAL RULES:
- Compare EACH field against what you actually see in the image
- Only mark as correct if the extracted value genuinely matches the image
- If you cannot verify a field (not visible), mark confidence < 0.5
- Provide corrected_value ONLY when the original is wrong"""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=4000,
            temperature=0.0,
        )

        # Identify fields needing correction
        fields_needing_correction = []
        for fv in result.get("field_validations", []):
            if not fv.get("is_correct", True):
                fields_needing_correction.append(fv["field_name"])
            elif fv.get("confidence", 1.0) < self._confidence_threshold:
                fields_needing_correction.append(fv["field_name"])

        result["fields_needing_correction"] = fields_needing_correction
        result["overall_valid"] = len(fields_needing_correction) == 0

        logger.info(
            "validation_complete",
            record_id=record.record_id,
            primary_id=primary_id,
            overall_valid=result["overall_valid"],
            fields_needing_correction=fields_needing_correction,
            overall_confidence=result.get("overall_confidence", 0.0),
        )

        return result

    def _correct_low_confidence_fields(
        self,
        page_data_uri: str,
        record: ExtractedRecord,
        boundary: RecordBoundary,
        fields_to_correct: list[str],
        validation_result: dict[str, Any],
        schema: dict[str, Any],
    ) -> ExtractedRecord:
        """
        Re-extract specific low-confidence or incorrect fields with targeted prompts.

        Uses a focused VLM call that only asks about the problematic fields,
        providing the validation context to help the VLM understand what went wrong.

        Args:
            page_data_uri: Data URI of the page image.
            record: The original extracted record.
            boundary: Record boundary with location info.
            fields_to_correct: List of field names to re-extract.
            validation_result: The validation output with correction hints.
            schema: Field schema for type context.

        Returns:
            Updated ExtractedRecord with corrected field values.
        """
        primary_id = record.primary_identifier
        bbox = boundary.bounding_box

        if not fields_to_correct:
            return record

        logger.info(
            "correcting_fields",
            record_id=record.record_id,
            primary_id=primary_id,
            fields_to_correct=fields_to_correct,
        )

        # Build field-specific context from schema
        schema_fields = {f["field_name"]: f for f in schema.get("fields", [])}

        # Build context about what went wrong per field
        correction_context_lines = []
        validation_map = {
            fv["field_name"]: fv
            for fv in validation_result.get("field_validations", [])
        }

        for field_name in fields_to_correct:
            field_schema = schema_fields.get(field_name, {})
            field_type = field_schema.get("field_type", "text")
            original_value = record.fields.get(field_name, "null")
            validation_info = validation_map.get(field_name, {})
            issue = validation_info.get("issue", "low confidence or incorrect")
            hint = validation_info.get("corrected_value")

            line = f"  - {field_name} (type: {field_type})"
            line += f"\n    Previous extraction: {original_value}"
            line += f"\n    Issue: {issue}"
            if hint:
                line += f"\n    Validator suggested: {hint}"
            correction_context_lines.append(line)

        correction_context = "\n".join(correction_context_lines)

        prompt = f"""TARGETED RE-EXTRACTION: Carefully re-read specific fields from the image.

TARGET RECORD: {primary_id}
BOUNDING BOX: {self._format_bbox(bbox)}

The following fields were flagged as potentially incorrect or uncertain.
Please look VERY CAREFULLY at the original image and re-extract these values.

FIELDS TO RE-EXTRACT:
{correction_context}

INSTRUCTIONS:
1. For each field, locate it precisely in the image within the bounding box
2. Read the value character by character if necessary
3. Pay special attention to commonly confused characters (0/O, 1/l/I, 5/S)
4. For dates, verify the exact format visible in the image
5. For codes/IDs, read each character individually

Return JSON:
{{
  "corrected_fields": {{
    "field_name": {{
      "value": "carefully re-read value",
      "confidence": 0.95,
      "method": "character-by-character reading",
      "differs_from_original": true
    }}
  }},
  "correction_summary": {{
    "total_corrected": 0,
    "total_confirmed": 0,
    "overall_confidence": 0.93
  }}
}}

CRITICAL: Read EXTREMELY carefully. This is a second-chance correction pass."""

        # Use slightly elevated temperature for diversity
        temperature = self._get_adaptive_temperature(
            field_type="mixed",
            field_name="correction",
            retry_count=1,
        )

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=3000,
            temperature=temperature,
        )

        # Apply corrections to the record
        corrected_fields = result.get("corrected_fields", {})
        corrections_applied = 0

        for field_name, correction in corrected_fields.items():
            if field_name in record.fields:
                new_value = correction.get("value")
                new_confidence = correction.get("confidence", 0.0)

                # Only apply correction if confidence is above threshold
                if new_confidence >= self._confidence_threshold:
                    old_value = record.fields[field_name]
                    record.fields[field_name] = new_value
                    corrections_applied += 1

                    logger.debug(
                        "field_corrected",
                        record_id=record.record_id,
                        field=field_name,
                        old_value=str(old_value)[:50],
                        new_value=str(new_value)[:50],
                        confidence=new_confidence,
                    )

        # Update record confidence based on correction results
        correction_summary = result.get("correction_summary", {})
        new_overall_confidence = correction_summary.get(
            "overall_confidence", record.confidence
        )
        if new_overall_confidence > record.confidence:
            record.confidence = new_overall_confidence

        logger.info(
            "correction_complete",
            record_id=record.record_id,
            primary_id=primary_id,
            corrections_applied=corrections_applied,
            total_fields_checked=len(fields_to_correct),
            new_confidence=record.confidence,
        )

        return record

    def _consensus_extract_critical_fields(
        self,
        page_data_uri: str,
        record: ExtractedRecord,
        boundary: RecordBoundary,
        schema: dict[str, Any],
        critical_fields: list[str],
    ) -> ExtractedRecord:
        """
        Re-extract critical fields using dual-pass consensus.

        Runs two independent extraction passes at different temperatures.
        Fields where both passes agree get boosted confidence. Fields that
        disagree trigger a focused tie-breaker VLM call.

        Args:
            page_data_uri: Data URI of the page image.
            record: The extracted (and optionally validated/corrected) record.
            boundary: Record boundary with location info.
            schema: Field schema for context.
            critical_fields: List of field names to verify via consensus.

        Returns:
            Updated ExtractedRecord with consensus-verified critical fields.
        """
        if not critical_fields:
            return record

        primary_id = record.primary_identifier
        bbox = boundary.bounding_box

        logger.info(
            "consensus_starting",
            record_id=record.record_id,
            primary_id=primary_id,
            critical_field_count=len(critical_fields),
        )

        # Build schema context for critical fields
        schema_fields = {f["field_name"]: f for f in schema.get("fields", [])}
        field_list_lines = []
        for fname in critical_fields:
            fschema = schema_fields.get(fname, {})
            ftype = fschema.get("field_type", "text")
            fdesc = fschema.get("description", fname)
            field_list_lines.append(f"  - {fname} (type: {ftype}): {fdesc}")
        field_list = "\n".join(field_list_lines)

        prompt = f"""CONSENSUS EXTRACTION: Re-read specific critical fields for verification.

TARGET RECORD: {primary_id}
BOUNDING BOX: {self._format_bbox(bbox)}

Extract ONLY these critical fields from the image, reading each value
character-by-character with extreme precision:

{field_list}

For each field:
1. Locate it precisely within the bounding box for "{primary_id}"
2. Read the value character-by-character
3. Report your confidence in the reading

Return JSON:
{{
  "critical_fields": {{
    "field_name": {{
      "value": "extracted value",
      "confidence": 0.97
    }}
  }}
}}

CRITICAL: Read EXTREMELY carefully. This is a verification pass for high-value fields."""

        # Pass 1: deterministic (temperature 0.0)
        pass1_result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=2000,
            temperature=0.0,
        )

        # Pass 2: slight variation (temperature 0.1)
        pass2_result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=2000,
            temperature=0.1,
        )

        # Compare results
        agreed = []
        disagreed = []

        pass1_fields = pass1_result.get("critical_fields", {})
        pass2_fields = pass2_result.get("critical_fields", {})

        for fname in critical_fields:
            val1_data = pass1_fields.get(fname, {})
            val2_data = pass2_fields.get(fname, {})
            val1 = val1_data.get("value") if isinstance(val1_data, dict) else val1_data
            val2 = val2_data.get("value") if isinstance(val2_data, dict) else val2_data
            conf1 = (
                val1_data.get("confidence", 0.0)
                if isinstance(val1_data, dict)
                else 0.0
            )
            conf2 = (
                val2_data.get("confidence", 0.0)
                if isinstance(val2_data, dict)
                else 0.0
            )

            if str(val1).strip() == str(val2).strip():
                agreed.append(fname)
                # Boost confidence for agreed fields
                boosted_confidence = min(max(conf1, conf2) * 1.05, 1.0)
                record.fields[fname] = val1
                logger.debug(
                    "consensus_agreed",
                    field=fname,
                    value=str(val1)[:50],
                    confidence=boosted_confidence,
                )
            else:
                disagreed.append((fname, val1, val2))

        # Resolve disagreements via tie-breaker
        for fname, val1, val2 in disagreed:
            resolved = self._resolve_disagreement(
                page_data_uri=page_data_uri,
                boundary=boundary,
                field_name=fname,
                value_1=val1,
                value_2=val2,
                record=record,
            )
            record.fields[fname] = resolved["value"]
            logger.debug(
                "consensus_tiebreaker_resolved",
                field=fname,
                value=str(resolved["value"])[:50],
                confidence=resolved.get("confidence", 0.0),
            )

        logger.info(
            "consensus_complete",
            record_id=record.record_id,
            primary_id=primary_id,
            agreed_count=len(agreed),
            disagreed_count=len(disagreed),
        )

        return record

    def _resolve_disagreement(
        self,
        page_data_uri: str,
        boundary: RecordBoundary,
        field_name: str,
        value_1: Any,
        value_2: Any,
        record: ExtractedRecord,
    ) -> dict[str, Any]:
        """
        Run a focused tie-breaker VLM call when two consensus passes disagree.

        Args:
            page_data_uri: Data URI of the page image.
            boundary: Record boundary with location info.
            field_name: The field that has conflicting values.
            value_1: Value from pass 1.
            value_2: Value from pass 2.
            record: The record being processed.

        Returns:
            Dict with 'value', 'confidence', and 'reasoning'.
        """
        primary_id = record.primary_identifier
        bbox = boundary.bounding_box

        prompt = f"""TIE-BREAKER EXTRACTION for field "{field_name}":

TARGET RECORD: {primary_id}
BOUNDING BOX: {self._format_bbox(bbox)}

Two independent extraction passes DISAGREED on this field:
- Pass 1 extracted: "{value_1}"
- Pass 2 extracted: "{value_2}"

Your task:
1. Locate the EXACT position of "{field_name}" for record "{primary_id}"
2. Read the value character-by-character with extreme precision
3. Consider which value is more plausible given the field type and context
4. If BOTH are wrong, extract the correct value from the image

Return JSON:
{{
  "value": "the_correct_value",
  "confidence": 0.97,
  "reasoning": "why this is the correct reading"
}}

CRITICAL: Read the actual text in the image. Do not guess or infer."""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            system_prompt=self._build_grounding_system_prompt(),
            max_tokens=500,
            temperature=0.0,
        )

        return {
            "value": result.get("value"),
            "confidence": float(result.get("confidence", 0.85)),
            "reasoning": result.get("reasoning", ""),
        }

    def extract_document(
        self,
        page_images: list[dict[str, Any]],
        pdf_path: str = "",
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> DocumentExtractionResult:
        """
        Extract all records from a multi-page document.

        Args:
            page_images: List of page image dicts with 'data_uri' and 'page_number'.
            pdf_path: Path to source PDF (for metadata).
            start_page: Optional first page to process (1-indexed).
            end_page: Optional last page to process (1-indexed).

        Returns:
            DocumentExtractionResult with all extracted records.
        """
        self._vlm_calls = 0
        overall_start = time.time()

        # Filter pages
        if start_page or end_page:
            page_images = [
                p
                for p in page_images
                if (start_page is None or p["page_number"] >= start_page)
                and (end_page is None or p["page_number"] <= end_page)
            ]

        total_pages = len(page_images)
        logger.info(
            "multi_record_extraction_started",
            pdf_path=pdf_path,
            total_pages=total_pages,
        )

        if not page_images:
            raise ValueError("No page images to process")

        # Stage 0: Detect document type from first page
        first_page_uri = page_images[0]["data_uri"]
        doc_metadata = self.detect_document_type(first_page_uri)

        entity_type = doc_metadata.get("entity_type", "record")
        primary_id_field = doc_metadata.get("primary_identifier_field", "name")
        document_type = doc_metadata.get("document_type", "unknown")

        # Stage 1: Generate adaptive schema from first page
        schema = self.generate_schema(first_page_uri, document_type, entity_type)

        # Stage 2: Process all pages
        all_records: list[ExtractedRecord] = []
        global_record_id = 0

        for page_data in page_images:
            page_num = page_data["page_number"]
            page_uri = page_data["data_uri"]

            logger.info("processing_page", page=page_num, total=total_pages)

            # Detect record boundaries on this page
            boundaries = self.detect_record_boundaries(
                page_data_uri=page_uri,
                entity_type=entity_type,
                primary_id_field=primary_id_field,
                page_number=page_num,
            )

            # Extract each record (with optional validation + correction)
            for boundary in boundaries:
                global_record_id += 1
                boundary.record_id = global_record_id

                record = self.extract_single_record(
                    page_data_uri=page_uri,
                    boundary=boundary,
                    schema=schema,
                    page_number=page_num,
                )
                record.record_id = global_record_id

                # Post-extraction: validate and coerce field types
                record = self._validate_field_types(record, schema)

                # Phase 2: Validate + Correct pipeline
                corrected_fields: list[str] = []
                if self._enable_validation:
                    validation_result = self._validate_extraction(
                        page_data_uri=page_uri,
                        record=record,
                        boundary=boundary,
                        schema=schema,
                    )

                    fields_to_fix = validation_result.get(
                        "fields_needing_correction", []
                    )

                    if fields_to_fix and self._enable_self_correction:
                        record = self._correct_low_confidence_fields(
                            page_data_uri=page_uri,
                            record=record,
                            boundary=boundary,
                            fields_to_correct=fields_to_fix,
                            validation_result=validation_result,
                            schema=schema,
                        )
                        corrected_fields = fields_to_fix

                # Phase 3: Consensus for critical fields (final quality gate)
                # Skip fields already corrected by Phase 2 to avoid redundant VLM calls
                if self._enable_consensus:
                    critical_fields = self._identify_critical_fields(schema)
                    if corrected_fields:
                        critical_fields = [
                            f for f in critical_fields
                            if f not in corrected_fields
                        ]
                    if critical_fields:
                        record = self._consensus_extract_critical_fields(
                            page_data_uri=page_uri,
                            record=record,
                            boundary=boundary,
                            schema=schema,
                            critical_fields=critical_fields,
                        )

                all_records.append(record)

            logger.info(
                "page_complete",
                page=page_num,
                records_on_page=len(boundaries),
                total_records_so_far=len(all_records),
            )

        total_time_ms = int((time.time() - overall_start) * 1000)

        result = DocumentExtractionResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_records=len(all_records),
            document_type=document_type,
            entity_type=entity_type,
            records=all_records,
            schema=schema,
            total_processing_time_ms=total_time_ms,
            total_vlm_calls=self._vlm_calls,
        )

        logger.info(
            "multi_record_extraction_complete",
            total_pages=total_pages,
            total_records=len(all_records),
            total_vlm_calls=self._vlm_calls,
            processing_time_s=round(total_time_ms / 1000, 1),
        )

        return result
