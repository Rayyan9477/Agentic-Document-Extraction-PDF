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

Total VLM calls: 2 + pages * (1 + records_per_page)
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.client.lm_client import LMStudioClient, VisionRequest
from src.config import get_logger

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

    def __init__(self, client: LMStudioClient | None = None) -> None:
        self._client = client or LMStudioClient()
        self._vlm_calls = 0

    def _send_vision_json(
        self,
        image_data: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 3000,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Send a vision request and return parsed JSON, with retry."""
        last_error = None
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

            except Exception as e:
                last_error = e
                logger.warning(
                    "vlm_call_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    time.sleep(2)

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

        prompt = """Analyze this document and determine:

1. **Document Type**: What kind of document is this?
   Examples: medical_superbill, medical_patient_list, invoice_list, employee_roster,
             insurance_claims, purchase_orders, etc.

2. **Entity Type**: What does each INDIVIDUAL record represent?
   Examples: patient, invoice, employee, product, transaction, claim, order, etc.

3. **Primary Identifier**: What field uniquely identifies each record?
   Examples: patient_name, invoice_number, employee_id, product_code, etc.

4. **Record Structure**: How are records organized?
   - list: one after another vertically
   - table: rows with columns
   - form: grouped sections
   - mixed: combination

Return JSON:
{
  "document_type": "type_name",
  "document_description": "Brief description",
  "entity_type": "entity_name",
  "entity_description": "What each record represents",
  "primary_identifier_field": "field_name",
  "record_structure": "list|table|form|mixed",
  "estimated_records_per_page": 3,
  "confidence": 0.95
}"""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            max_tokens=1000,
        )

        logger.info(
            "document_type_detected",
            document_type=result.get("document_type"),
            entity_type=result.get("entity_type"),
            primary_id=result.get("primary_identifier_field"),
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

        prompt = f"""Analyze this {document_type} document and generate a data extraction schema.

Document Type: {document_type}
Entity Type: {entity_type}

Identify ALL data fields present in each {entity_type} record.
For each field, determine:
- Field name (snake_case)
- Data type (text, number, date, boolean, list)
- Display name (human-readable)
- Description
- Whether it's required

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
  "total_field_count": 10
}}

Include ALL fields visible in the records, not just common ones."""

        schema = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            max_tokens=3000,
        )

        logger.info(
            "schema_generated",
            field_count=len(schema.get("fields", [])),
        )
        return schema

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

        prompt = f"""Analyze this document page and identify INDIVIDUAL {entity_type.upper()} RECORDS.

Entity Type: {entity_type}
Primary Identifier: {primary_id_field}

Your task:
1. Count the total number of DISTINCT {entity_type} records on this page
2. For each record, identify:
   - The {primary_id_field} value (PRIMARY identifier)
   - The approximate bounding box (top, left, bottom, right as percentages 0.0-1.0)
   - Visual separators between records (lines, whitespace, etc.)

Return JSON:
{{
  "total_records": 3,
  "records": [
    {{
      "record_id": 1,
      "primary_identifier": "extracted {primary_id_field} value",
      "bounding_box": {{
        "top": 0.0,
        "left": 0.0,
        "bottom": 0.33,
        "right": 1.0
      }},
      "visual_separator": "horizontal line"
    }}
  ],
  "layout_notes": "How records are organized on this page"
}}

CRITICAL: Each unique {primary_id_field} marks a SEPARATE record. List ALL records you can see."""

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
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
        prompt = f"""Extract data for THIS SPECIFIC {entity_type.upper()} RECORD ONLY:

TARGET: {primary_id}
LOCATION: Top {bbox.get('top', 0):.0%} to Bottom {bbox.get('bottom', 1):.0%} of page

Extract ONLY the data belonging to "{primary_id}".
Do NOT include data from other {entity_type}s on this page.

Fields to extract:
{field_list}

Return JSON:
{{
  "record_id": {boundary.record_id},
  "primary_identifier": "{primary_id}",
  "fields": {{
    "field_name": "extracted_value"
  }},
  "confidence": 0.90
}}

CRITICAL RULES:
- Extract ONLY data for "{primary_id}" - ignore all other records
- Return null for fields that are not visible or unclear
- Do NOT guess or hallucinate values
- Confidence should reflect actual certainty (0.0-1.0)"""

        start_ms = time.time()

        result = self._send_vision_json(
            image_data=page_data_uri,
            prompt=prompt,
            max_tokens=3000,
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

            # Extract each record
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
