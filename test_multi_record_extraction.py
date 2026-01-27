"""
Enhanced VLM extraction for multi-record documents.
This version can detect and extract multiple patient records from a single page.
"""

import os
if "AGENT" in os.environ:
    del os.environ["AGENT"]

import base64
from pathlib import Path
from openai import OpenAI
from pdf2image import convert_from_path

# Initialize OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def encode_image_to_base64(image) -> str:
    """Encode PIL Image to base64."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def detect_record_boundaries(page_image) -> dict:
    """
    Use VLM to detect individual patient record boundaries on the page.
    Returns a list of bounding boxes for each patient record.
    """
    print("\n[STAGE 1: Detecting Record Boundaries]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    prompt = """Analyze this medical document page and identify INDIVIDUAL PATIENT RECORDS.

This document contains multiple patient records on a single page. Each patient record includes:
- Patient demographics (name, sex, ID, DOB, referring physician)
- Sedation details
- Indications
- Findings
- ICD codes
- CPT codes
- Plan

Your task:
1. Count the total number of DISTINCT PATIENT RECORDS on this page
2. For each patient record, identify:
   - The patient name (this is the PRIMARY identifier)
   - The approximate bounding box (top, left, bottom, right as percentages 0.0-1.0)
   - Visual separators (horizontal lines, spacing, etc.)

Return a JSON object with this structure:
{
  "total_records": <number>,
  "records": [
    {
      "record_id": 1,
      "patient_name": "Last, First MI",
      "bounding_box": {
        "top": 0.0,
        "left": 0.0,
        "bottom": 0.5,
        "right": 1.0
      },
      "visual_separator": "horizontal line / whitespace / header"
    }
  ],
  "layout_notes": "Description of how records are organized"
}

CRITICAL: Each patient name marks a SEPARATE record. Count carefully."""

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    
    import json
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    result = json.loads(result_text)
    
    print(f"[OK] Detected {result['total_records']} patient records")
    for rec in result['records']:
        print(f"     - Record {rec['record_id']}: {rec['patient_name']}")
    
    return result


def extract_single_record(page_image, record_info: dict, schema_fields: list) -> dict:
    """
    Extract data for a single patient record using the adaptive schema.
    """
    record_id = record_info['record_id']
    patient_name = record_info['patient_name']
    
    print(f"\n[STAGE 2: Extracting Record {record_id} - {patient_name}]")
    
    image_b64 = encode_image_to_base64(page_image)
    
    # Build field list for extraction
    field_list = "\n".join([f"  - {field['field_name']}: {field['description']}" for field in schema_fields])
    
    bbox = record_info['bounding_box']
    
    prompt = f"""Extract data for THIS SPECIFIC PATIENT RECORD ONLY:

TARGET PATIENT: {patient_name}
RECORD LOCATION: Top {bbox['top']:.0%} to Bottom {bbox['bottom']:.0%}

Extract ONLY the data for patient "{patient_name}". Do NOT include data from other patients on this page.

Fields to extract:
{field_list}

Return a JSON object with this structure:
{{
  "record_id": {record_id},
  "patient_name": "{patient_name}",
  "fields": {{
    "patient_name": "extracted value",
    "patient_sex": "extracted value",
    "patient_id": "extracted value",
    "patient_dob": "extracted value",
    "referring_physician": "extracted value",
    "sedation": "extracted value",
    "indications": "extracted value",
    "findings": "extracted value (list all findings with locations)",
    "icd_codes": "extracted value (list all codes)",
    "cpt_codes": "extracted value (list all codes)",
    "plan": "extracted value (list all plan items)"
  }},
  "confidence": 0.95
}}

IMPORTANT: Extract data ONLY for patient "{patient_name}" - ignore all other patients."""

    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=3000,
    )
    
    import json
    result_text = response.choices[0].message.content.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    result = json.loads(result_text)
    
    print(f"[OK] Extracted {len(result.get('fields', {}))} fields")
    print(f"     Confidence: {result.get('confidence', 0.0):.0%}")
    
    return result


def main():
    print("=" * 70)
    print("Multi-Record VLM Extraction Test")
    print("=" * 70)
    
    pdf_path = "superbill1.pdf"
    page_num = 1
    
    print(f"\n[Loading PDF: {pdf_path}, Page {page_num}]")
    
    # Convert PDF page to image
    pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
    page_image = pages[0]
    
    print(f"[OK] Page loaded: {page_image.size[0]}x{page_image.size[1]} pixels")
    
    # Stage 1: Detect record boundaries
    import time
    start_time = time.time()
    
    boundary_result = detect_record_boundaries(page_image)
    
    boundary_time = time.time() - start_time
    print(f"\n[Boundary detection completed in {boundary_time:.1f}s]")
    
    # Load schema fields from previous results
    import json
    with open("test_full_vlm_results.json", 'r') as f:
        previous_results = json.load(f)
    
    schema_fields = previous_results['adaptive_schema']['fields']
    
    # Stage 2: Extract each record separately
    extracted_records = []
    
    for record_info in boundary_result['records']:
        record_start = time.time()
        
        extraction_result = extract_single_record(page_image, record_info, schema_fields)
        extraction_result['extraction_time_ms'] = int((time.time() - record_start) * 1000)
        
        extracted_records.append(extraction_result)
    
    total_time = time.time() - start_time
    
    # Compile results
    final_result = {
        "pdf_path": pdf_path,
        "page_number": page_num,
        "detection_method": "vlm_multi_record",
        "total_records_detected": boundary_result['total_records'],
        "boundary_detection": boundary_result,
        "extracted_records": extracted_records,
        "total_processing_time_ms": int(total_time * 1000),
        "total_vlm_calls": 1 + len(extracted_records),  # 1 for boundary + N for extractions
    }
    
    # Save results
    output_file = "multi_record_extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total Records Detected: {boundary_result['total_records']}")
    print(f"Total Records Extracted: {len(extracted_records)}")
    print(f"Total VLM Calls: {final_result['total_vlm_calls']}")
    print(f"Total Processing Time: {total_time:.1f}s")
    print(f"\nResults saved to: {output_file}")
    
    # Display extracted data
    print("\n" + "=" * 70)
    print("EXTRACTED RECORDS")
    print("=" * 70)
    
    for record in extracted_records:
        print(f"\n--- Record {record['record_id']}: {record['patient_name']} ---")
        fields = record.get('fields', {})
        for field_name, value in fields.items():
            display_value = str(value)[:100]
            if len(str(value)) > 100:
                display_value += "..."
            print(f"  {field_name}: {display_value}")


if __name__ == "__main__":
    main()
