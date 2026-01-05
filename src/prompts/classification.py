"""
Document classification prompts for the Analyzer agent.

Provides prompts for document type identification, structure detection,
and schema selection.

Enhanced with:
- Few-shot classification examples
- Step-by-step reasoning protocol
- Confidence calibration examples
"""

# Few-shot classification examples for each document type
CLASSIFICATION_EXAMPLES = """
## CLASSIFICATION EXAMPLES - USE THESE AS REFERENCE

### Example 1: CMS-1500 Identification ✓
**Visual Features Observed:**
- Red/pink colored form with numbered boxes (1-33)
- "HEALTH INSURANCE CLAIM FORM" printed at top
- Patient information boxes in upper section (1-13)
- Service line grid with columns 24A-J
- Physician signature area at bottom

**Classification:** CMS-1500
**Confidence:** 0.95
**Reasoning:** Form matches standard CMS-1500 layout with numbered boxes, service line grid, and recognizable header.

---

### Example 2: UB-04 Identification ✓
**Visual Features Observed:**
- Dense form with many small boxes
- "PATIENT NAME" and "ADDRESS" at top
- Revenue code columns in middle section
- Condition codes, occurrence codes visible
- Principal and other diagnosis fields
- Payer information sections

**Classification:** UB-04
**Confidence:** 0.92
**Reasoning:** Institutional claim form layout with revenue codes and form locators typical of UB-04/CMS-1450.

---

### Example 3: EOB Identification ✓
**Visual Features Observed:**
- Insurance company logo/letterhead
- "Explanation of Benefits" title
- Claim number and member ID displayed
- Table with Billed/Allowed/Paid columns
- "Patient Responsibility" section
- "THIS IS NOT A BILL" statement

**Classification:** EOB
**Confidence:** 0.90
**Reasoning:** Letter-style document from insurer with benefit explanation format, payment breakdown, and not-a-bill disclaimer.

---

### Example 4: Superbill Identification ✓
**Visual Features Observed:**
- Practice name and logo at top
- Checkboxes next to procedure descriptions
- CPT codes listed with prices
- Diagnosis code section
- Patient name and date fields
- Provider signature line

**Classification:** SUPERBILL
**Confidence:** 0.88
**Reasoning:** Encounter form with checkbox format for selecting services, typical of medical office superbills.

---

### Example 5: Unknown/Other Classification ✓
**Visual Features Observed:**
- Appears to be a medical record or lab report
- No standard claim form elements
- Custom format without numbered boxes
- May have logo but no EOB characteristics

**Classification:** OTHER
**Confidence:** 0.75
**Reasoning:** Document does not match any standard claim form pattern. May be a custom form, medical record, or correspondence.
"""


DOCUMENT_TYPE_DESCRIPTIONS = {
    "CMS-1500": {
        "full_name": "CMS-1500 Health Insurance Claim Form",
        "description": (
            "The standard paper claim form used by healthcare providers to bill "
            "Medicare, Medicaid, and most private insurers for professional services. "
            "Also known as HCFA-1500 form."
        ),
        "key_identifiers": [
            "Red form with numbered boxes 1-33",
            "Title 'HEALTH INSURANCE CLAIM FORM' at top",
            "OMB approval number in corner",
            "Patient and Insured information sections",
            "21 boxes for diagnosis codes",
            "24A-J rows for service line items",
            "Physician signature box at bottom",
        ],
        "typical_layout": "Portrait, multi-section form with numbered boxes",
    },
    "UB-04": {
        "full_name": "UB-04 Uniform Bill / CMS-1450",
        "description": (
            "The standard claim form used by institutional providers such as "
            "hospitals, skilled nursing facilities, and outpatient centers. "
            "Contains revenue codes and institutional billing information."
        ),
        "key_identifiers": [
            "Large form with many small boxes",
            "Revenue code columns (Form Locator 42)",
            "Condition codes section",
            "Occurrence codes and dates",
            "Value codes section",
            "Principal diagnosis and procedures",
            "Attending physician information",
        ],
        "typical_layout": "Portrait, dense grid layout with form locator numbers",
    },
    "EOB": {
        "full_name": "Explanation of Benefits",
        "description": (
            "A statement from an insurance company explaining what was covered, "
            "what was paid, and what the patient owes for a medical service. "
            "Not a bill but an explanation of claim processing."
        ),
        "key_identifiers": [
            "Insurance company letterhead/logo",
            "'Explanation of Benefits' or 'EOB' title",
            "Claim number and service dates",
            "Columns showing billed, allowed, paid amounts",
            "Patient responsibility section",
            "Deductible and coinsurance information",
            "'THIS IS NOT A BILL' statement often present",
        ],
        "typical_layout": "Portrait, letter-style with tables for claim details",
    },
    "SUPERBILL": {
        "full_name": "Medical Superbill / Encounter Form",
        "description": (
            "An itemized form used by healthcare providers to record services, "
            "procedures, and diagnoses during a patient visit. Used for billing "
            "and often has checkboxes for common procedures."
        ),
        "key_identifiers": [
            "Practice/provider name and logo",
            "Checkboxes next to procedure codes",
            "CPT codes with descriptions",
            "Diagnosis codes section",
            "Patient information area",
            "Date of service field",
            "Provider signature area",
        ],
        "typical_layout": "Portrait, checklist-style with procedure sections",
    },
    "OTHER": {
        "full_name": "Other Medical Document",
        "description": (
            "A medical document that does not match the standard form types. "
            "May include referrals, authorizations, medical records, or "
            "custom practice forms."
        ),
        "key_identifiers": [
            "Does not match CMS-1500, UB-04, EOB, or Superbill patterns",
            "May be custom practice form",
            "Could be correspondence or notes",
        ],
        "typical_layout": "Variable",
    },
}


def build_classification_prompt(
    include_confidence: bool = True,
    include_reasoning: bool = True,
    include_examples: bool = True,
    include_step_by_step: bool = True,
) -> str:
    """
    Build the document classification prompt.

    Args:
        include_confidence: Whether to request confidence score.
        include_reasoning: Whether to request classification reasoning.
        include_examples: Whether to include few-shot classification examples.
        include_step_by_step: Whether to include step-by-step reasoning protocol.

    Returns:
        Classification prompt for the VLM.
    """
    type_descriptions = []
    for doc_type, info in DOCUMENT_TYPE_DESCRIPTIONS.items():
        identifiers = "\n    - ".join(info["key_identifiers"])
        type_descriptions.append(
            f"### {doc_type}: {info['full_name']}\n"
            f"{info['description']}\n\n"
            f"Key identifiers:\n    - {identifiers}\n\n"
            f"Layout: {info['typical_layout']}"
        )

    step_by_step = ""
    if include_step_by_step:
        step_by_step = """
## STEP-BY-STEP CLASSIFICATION PROTOCOL

Follow this reasoning process for accurate classification:

### Step 1: OBSERVE LAYOUT
- Is this a structured form with numbered boxes? → Likely CMS-1500 or UB-04
- Is this a letter-style document with tables? → Likely EOB
- Is this a checklist-style form? → Likely Superbill
- None of the above? → Likely OTHER

### Step 2: LOOK FOR KEY IDENTIFIERS
- "HEALTH INSURANCE CLAIM FORM" → CMS-1500
- Revenue codes, condition codes, form locators → UB-04
- "Explanation of Benefits", "This is not a bill" → EOB
- CPT checkboxes, encounter form → Superbill

### Step 3: VERIFY WITH SECONDARY FEATURES
- Confirm with 2-3 additional identifiers from the descriptions below
- Note any features that don't match

### Step 4: ASSIGN CONFIDENCE
- All key identifiers present → 0.90-1.00
- Most identifiers present with minor variations → 0.80-0.89
- Some identifiers but uncertain → 0.60-0.79
- Best guess → 0.50-0.59

"""

    examples = ""
    if include_examples:
        examples = f"\n{CLASSIFICATION_EXAMPLES}\n"

    prompt = f"""
## DOCUMENT CLASSIFICATION TASK

Analyze this document image and identify its type based on visual characteristics.
{step_by_step}
## KNOWN DOCUMENT TYPES

{chr(10).join(type_descriptions)}
{examples}
## CLASSIFICATION INSTRUCTIONS

1. Examine the overall layout and structure of the document
2. Look for identifying text, form numbers, or standard elements
3. Match visual characteristics to the document type descriptions above
4. If the document matches multiple types, choose the most specific match
5. If the document does not match any known type, classify as "OTHER"

## REQUIRED OUTPUT

Return a JSON object with the following structure:

```json
{{
  "document_type": "CMS-1500 | UB-04 | EOB | SUPERBILL | OTHER",
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this classification was chosen",
  "key_features_found": ["list", "of", "identifying", "features"],
  "alternate_types": ["list of other possible types if uncertain"]
}}
```
"""

    if not include_confidence:
        prompt = prompt.replace('  "confidence": 0.95,\n', "")

    if not include_reasoning:
        prompt = prompt.replace(
            '  "reasoning": "Brief explanation of why this classification was chosen",\n', ""
        )

    return prompt


def build_structure_analysis_prompt() -> str:
    """
    Build prompt for document structure analysis.

    Returns:
        Structure analysis prompt for the VLM.
    """
    return """
## DOCUMENT STRUCTURE ANALYSIS TASK

Analyze the visual structure of this document to identify:
1. Tables and their locations
2. Form fields and input areas
3. Handwritten vs printed content
4. Signatures or stamps
5. Multiple sections or regions
6. Page relationships (if multi-page)

## ANALYSIS INSTRUCTIONS

For each structural element found:
- Describe its location (top, bottom, left, right, center)
- Identify its purpose (data entry, signature, header, etc.)
- Note any quality issues (blur, damage, obscured areas)

## REQUIRED OUTPUT

Return a JSON object with the following structure:

```json
{
  "structures_detected": {
    "tables": [
      {
        "location": "center of page",
        "rows_estimated": 10,
        "columns_estimated": 5,
        "purpose": "service line items",
        "quality": "clear"
      }
    ],
    "form_fields": [
      {
        "location": "top-left",
        "type": "text_input",
        "label": "Patient Name",
        "filled": true,
        "content_type": "printed"
      }
    ],
    "handwriting_regions": [
      {
        "location": "bottom-right",
        "type": "signature",
        "legibility": "poor"
      }
    ],
    "stamps_marks": [
      {
        "location": "top-right",
        "type": "date_stamp",
        "legible": true
      }
    ]
  },
  "overall_quality": {
    "scan_quality": "good | fair | poor",
    "skew_detected": false,
    "noise_level": "low | medium | high",
    "contrast": "good | fair | poor"
  },
  "regions_of_interest": [
    {
      "name": "patient_info_section",
      "location": "top third of page",
      "importance": "high"
    }
  ],
  "extraction_challenges": [
    "handwritten entries may be difficult to read",
    "some boxes are overwritten"
  ]
}
```
"""


def build_page_relationship_prompt(total_pages: int) -> str:
    """
    Build prompt for analyzing relationships between pages.

    Args:
        total_pages: Total number of pages in the document.

    Returns:
        Page relationship analysis prompt.
    """
    return f"""
## MULTI-PAGE DOCUMENT ANALYSIS

This document has {total_pages} pages. Analyze the current page to determine:
1. Its role in the overall document
2. Whether it continues from or leads to other pages
3. What unique information it contains

## ANALYSIS INSTRUCTIONS

- Look for page numbers or continuation indicators
- Identify if this is a cover page, main content, or attachment
- Check for "continued" or "page X of Y" text
- Note any references to other pages

## REQUIRED OUTPUT

Return a JSON object with:

```json
{{
  "page_role": "cover | main_content | continuation | attachment | summary",
  "page_number_shown": null or number shown on page,
  "continues_from_previous": true | false,
  "continues_to_next": true | false,
  "unique_content": ["list of unique data on this page"],
  "relationship_notes": "Description of how this page relates to others"
}}
```
"""


def build_schema_selection_prompt(
    document_type: str,
    available_schemas: list[str],
    custom_schema_provided: bool = False,
) -> str:
    """
    Build prompt for schema selection based on document analysis.

    Args:
        document_type: Detected document type.
        available_schemas: List of available schema names.
        custom_schema_provided: Whether a custom schema was provided.

    Returns:
        Schema selection prompt.
    """
    schema_list = "\n".join(f"- {schema}" for schema in available_schemas)

    custom_note = ""
    if custom_schema_provided:
        custom_note = """
NOTE: A custom schema has been provided for this extraction.
Use the custom schema unless it is clearly incompatible with the document.
"""

    return f"""
## SCHEMA SELECTION TASK

Based on the document classification as "{document_type}", select the most appropriate
extraction schema from the available options.

## AVAILABLE SCHEMAS

{schema_list}
{custom_note}

## SELECTION CRITERIA

1. Match the schema to the document type
2. Consider any detected structure variations
3. Prefer more specific schemas over general ones
4. If custom schema provided, verify compatibility

## REQUIRED OUTPUT

Return a JSON object with:

```json
{{
  "selected_schema": "name of selected schema",
  "selection_reason": "Why this schema was chosen",
  "schema_compatibility": 0.95,
  "field_coverage_estimate": "high | medium | low"
}}
```
"""
