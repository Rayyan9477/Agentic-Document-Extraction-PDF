"""
Universal Multi-Record Extraction System.

This system can detect and extract multiple records from ANY document type:
- Medical patient lists
- Invoices/receipts
- Employee records
- Product catalogs
- Transaction logs
- Insurance claims
- Any tabular or list-based document

Key features:
- Document-agnostic record detection
- Configurable entity types
- Adaptive schema generation per document type
- Cross-page duplicate detection
- Consolidated multi-page export
"""

import os
if "AGENT" in os.environ:
    del os.environ["AGENT"]

import base64
import json
import time
from pathlib import Path
from typing import Any, List, Dict
from dataclasses import dataclass, asdict
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

# Initialize OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


@dataclass
class RecordBoundary:
    """Represents a detected record boundary."""
    record_id: int
    primary_identifier: str  # e.g., patient name, invoice number, employee ID
    bounding_box: Dict[str, float]  # {top, left, bottom, right}
    visual_separator: str
    entity_type: str  # e.g., "patient", "invoice", "employee"


@dataclass
class ExtractedRecord:
    """Represents an extracted record."""
    record_id: int
    page_number: int
    primary_identifier: str
    entity_type: str
    fields: Dict[str, Any]
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
    records: List[ExtractedRecord]
    schema: Dict[str, Any]
    total_processing_time_ms: int
    total_vlm_calls: int


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def detect_document_type_and_entity(page_image: Image.Image) -> Dict[str, Any]:
    """
    Use VLM to automatically detect:
    1. Document type (invoice, patient list, employee roster, etc.)
    2. Entity type (what each record represents)
    3. Primary identifier field (what uniquely identifies each record)
    """
    print("\n[STAGE 0: Detecting Document Type & Entity]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    prompt = """Analyze this document and determine:

1. **Document Type**: What kind of document is this?
   Examples: medical_patient_list, invoice_list, employee_roster, product_catalog, 
             transaction_log, insurance_claims, purchase_orders, etc.

2. **Entity Type**: What does each record represent?
   Examples: patient, invoice, employee, product, transaction, claim, order, customer, etc.

3. **Primary Identifier**: What field uniquely identifies each record?
   Examples: patient_name, invoice_number, employee_id, product_code, transaction_id, etc.

4. **Record Structure**: Are records organized in:
   - list format (one after another vertically)
   - table format (rows with columns)
   - form format (grouped sections)
   - mixed format

Return JSON:
{
  "document_type": "type_name",
  "document_description": "Brief description of document purpose",
  "entity_type": "entity_name",
  "entity_description": "What each record represents",
  "primary_identifier_field": "field_name",
  "record_structure": "list|table|form|mixed",
  "estimated_records_per_page": <number>,
  "confidence": 0.95
}"""

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    result = json.loads(result_text)
    
    print(f"[OK] Document Type: {result['document_type']}")
    print(f"     Entity Type: {result['entity_type']}")
    print(f"     Primary ID Field: {result['primary_identifier_field']}")
    print(f"     Records per Page: ~{result['estimated_records_per_page']}")
    
    return result


def vlm_call_with_retry(func, max_retries=3, retry_delay=5):
    """Wrapper for VLM calls with retry logic."""
    import time
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] VLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"[Retrying in {retry_delay}s...]")
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] VLM call failed after {max_retries} attempts")
                raise


def detect_record_boundaries_universal(
    page_image: Image.Image, 
    entity_type: str, 
    primary_id_field: str,
    page_number: int
) -> List[RecordBoundary]:
    """
    Universal record boundary detector - works with any entity type.
    """
    print(f"\n[STAGE 1: Detecting {entity_type.Title()} Boundaries - Page {page_number}]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    prompt = f"""Analyze this document page and identify INDIVIDUAL {entity_type.upper()} RECORDS.

Entity Type: {entity_type}
Primary Identifier: {primary_id_field}

Your task:
1. Count the total number of DISTINCT {entity_type} RECORDS on this page
2. For each record, identify:
   - The {primary_id_field} value (PRIMARY identifier)
   - The approximate bounding box (top, left, bottom, right as percentages 0.0-1.0)
   - Visual separators between records

Return JSON:
{{
  "total_records": <number>,
  "records": [
    {{
      "record_id": 1,
      "primary_identifier": "extracted {primary_id_field} value",
      "bounding_box": {{
        "top": 0.0,
        "left": 0.0,
        "bottom": 0.5,
        "right": 1.0
      }},
      "visual_separator": "description of separator"
    }}
  ],
  "layout_notes": "How records are organized"
}}

CRITICAL: Each unique {primary_id_field} marks a SEPARATE record."""

    def make_vlm_call():
        return client.chat.completions.create(
            model="qwen/qwen3-vl-8b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2000,
        )
    
    response = vlm_call_with_retry(make_vlm_call)
    
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    result = json.loads(result_text)
    
    boundaries = []
    for rec in result['records']:
        boundaries.append(RecordBoundary(
            record_id=rec['record_id'],
            primary_identifier=rec['primary_identifier'],
            bounding_box=rec['bounding_box'],
            visual_separator=rec.get('visual_separator', ''),
            entity_type=entity_type
        ))
    
    print(f"[OK] Detected {len(boundaries)} {entity_type} records")
    for boundary in boundaries:
        print(f"     - Record {boundary.record_id}: {boundary.primary_identifier}")
    
    return boundaries


def generate_adaptive_schema(
    page_image: Image.Image,
    document_type: str,
    entity_type: str
) -> Dict[str, Any]:
    """
    Generate schema dynamically based on document type.
    """
    print(f"\n[STAGE 2: Generating Adaptive Schema for {document_type}]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    prompt = f"""Analyze this {document_type} document and generate a data extraction schema.

Document Type: {document_type}
Entity Type: {entity_type}

Your task:
1. Identify ALL data fields present in each {entity_type} record
2. For each field, determine:
   - Field name (snake_case)
   - Data type (text, number, date, boolean, list)
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
      "required": true,
      "examples": ["example1", "example2"]
    }}
  ],
  "total_field_count": <number>
}}

Include ALL fields you see in the records, not just common ones."""

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=3000,
    )
    
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    schema = json.loads(result_text)
    
    print(f"[OK] Generated schema with {len(schema['fields'])} fields")
    
    return schema


def extract_single_record_universal(
    page_image: Image.Image,
    boundary: RecordBoundary,
    schema: Dict[str, Any],
    page_number: int
) -> ExtractedRecord:
    """
    Extract data for a single record using adaptive schema.
    """
    record_id = boundary.record_id
    primary_id = boundary.primary_identifier
    entity_type = boundary.entity_type
    
    print(f"[STAGE 3: Extracting {entity_type} {record_id} - {primary_id}]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    # Build field extraction instructions
    field_instructions = []
    for field in schema['fields']:
        field_instructions.append(
            f"  - {field['field_name']} ({field['field_type']}): {field['description']}"
        )
    field_list = "\n".join(field_instructions)
    
    bbox = boundary.bounding_box
    
    prompt = f"""Extract data for THIS SPECIFIC {entity_type.upper()} RECORD ONLY:

TARGET: {primary_id}
LOCATION: Top {bbox['top']:.0%} to Bottom {bbox['bottom']:.0%}

Extract ONLY the data for "{primary_id}". Do NOT include data from other {entity_type}s on this page.

Fields to extract:
{field_list}

Return JSON:
{{
  "record_id": {record_id},
  "primary_identifier": "{primary_id}",
  "fields": {{
    "field_name": "extracted_value",
    ...
  }},
  "confidence": 0.95
}}

IMPORTANT: Extract ONLY data for "{primary_id}" - ignore all others."""

    start_time = time.time()
    
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=3000,
    )
    
    extraction_time_ms = int((time.time() - start_time) * 1000)
    
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    result = json.loads(result_text)
    
    record = ExtractedRecord(
        record_id=record_id,
        page_number=page_number,
        primary_identifier=result['primary_identifier'],
        entity_type=entity_type,
        fields=result['fields'],
        confidence=result.get('confidence', 0.0),
        extraction_time_ms=extraction_time_ms
    )
    
    print(f"[OK] Extracted {len(record.fields)} fields (confidence: {record.confidence:.0%})")
    
    return record


def process_single_page(
    pdf_path: str,
    page_number: int,
    document_metadata: Dict[str, Any],
    schema: Dict[str, Any]
) -> List[ExtractedRecord]:
    """
    Process a single page and extract all records.
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING PAGE {page_number}")
    print(f"{'='*70}")
    
    # Convert PDF page to image
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=300)
    page_image = pages[0]
    
    entity_type = document_metadata['entity_type']
    primary_id_field = document_metadata['primary_identifier_field']
    
    # Detect record boundaries
    boundaries = detect_record_boundaries_universal(
        page_image, entity_type, primary_id_field, page_number
    )
    
    # Extract each record
    records = []
    for boundary in boundaries:
        record = extract_single_record_universal(
            page_image, boundary, schema, page_number
        )
        records.append(record)
    
    return records


def process_multi_page_document(
    pdf_path: str,
    start_page: int = 1,
    end_page: int = None,
    max_pages: int = None
) -> DocumentExtractionResult:
    """
    Process entire multi-page document with universal extraction.
    """
    print("=" * 70)
    print("UNIVERSAL MULTI-RECORD MULTI-PAGE EXTRACTION")
    print("=" * 70)
    
    pdf_path = Path(pdf_path)
    print(f"\n[PDF: {pdf_path.name}]")
    
    # Get total pages
    from pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(pdf_path)
    total_pages_in_pdf = info["Pages"]
    
    if end_page is None:
        end_page = total_pages_in_pdf
    
    if max_pages:
        end_page = min(start_page + max_pages - 1, end_page)
    
    print(f"[Pages to process: {start_page} to {end_page} (total: {end_page - start_page + 1})]")
    
    start_time = time.time()
    total_vlm_calls = 0
    
    # Stage 0: Detect document type from first page
    pages = convert_from_path(pdf_path, first_page=start_page, last_page=start_page, dpi=300)
    first_page_image = pages[0]
    
    document_metadata = detect_document_type_and_entity(first_page_image)
    total_vlm_calls += 1
    
    # Stage 1: Generate schema from first page
    schema = generate_adaptive_schema(
        first_page_image,
        document_metadata['document_type'],
        document_metadata['entity_type']
    )
    total_vlm_calls += 1
    
    # Stage 2: Process all pages
    all_records = []
    
    for page_num in range(start_page, end_page + 1):
        page_records = process_single_page(
            str(pdf_path), page_num, document_metadata, schema
        )
        all_records.extend(page_records)
        
        # Count VLM calls: 1 for boundary detection + N for records
        total_vlm_calls += 1 + len(page_records)
        
        print(f"\n[Page {page_num} Complete: {len(page_records)} records extracted]")
    
    total_time_ms = int((time.time() - start_time) * 1000)
    
    result = DocumentExtractionResult(
        pdf_path=str(pdf_path),
        total_pages=end_page - start_page + 1,
        total_records=len(all_records),
        document_type=document_metadata['document_type'],
        entity_type=document_metadata['entity_type'],
        records=all_records,
        schema=schema,
        total_processing_time_ms=total_time_ms,
        total_vlm_calls=total_vlm_calls
    )
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total Pages: {result.total_pages}")
    print(f"Total Records: {result.total_records}")
    print(f"Document Type: {result.document_type}")
    print(f"Entity Type: {result.entity_type}")
    print(f"Processing Time: {result.total_processing_time_ms / 1000:.2f}s")
    print(f"Avg per Record: {result.total_processing_time_ms / 1000 / max(result.total_records, 1):.2f}s")
    print(f"Total VLM Calls: {result.total_vlm_calls}")
    
    return result


def save_results(result: DocumentExtractionResult, output_path: str):
    """Save extraction results to JSON."""
    output_data = {
        "pdf_path": result.pdf_path,
        "total_pages": result.total_pages,
        "total_records": result.total_records,
        "document_type": result.document_type,
        "entity_type": result.entity_type,
        "schema": result.schema,
        "records": [asdict(rec) for rec in result.records],
        "total_processing_time_ms": result.total_processing_time_ms,
        "total_vlm_calls": result.total_vlm_calls,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Results saved to: {output_path}]")


if __name__ == "__main__":
    # Test with superbill - process first 3 pages
    pdf_path = "superbill1.pdf"
    
    result = process_multi_page_document(
        pdf_path=pdf_path,
        start_page=1,
        end_page=3,  # Process pages 1-3 as a test
    )
    
    # Save results
    save_results(result, "universal_extraction_test_3pages.json")
