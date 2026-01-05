"""
Extraction prompts for the Extractor agent.

Provides prompts for dual-pass extraction, field-specific extraction,
and table processing.

Enhanced with:
- More differentiated dual-pass prompts
- Chain-of-thought extraction reasoning
- Field-specific extraction examples
- Structured verification protocols
"""

from typing import Any

from src.prompts.grounding_rules import (
    build_null_handling_instruction,
)


# Chain-of-thought reasoning for field extraction
EXTRACTION_REASONING_TEMPLATE = """
## EXTRACTION REASONING PROTOCOL

For EACH field you extract, follow this mental process:

### 1. LOCATE the field
- Where in the document should I look for this field?
- Is there a label or box identifier?
- Can I see the field location clearly?

### 2. READ character-by-character
- What exact characters do I see?
- Is there any blur, smudge, or obstruction?
- Could any character be misread (0 vs O, 1 vs l, 5 vs S)?

### 3. VALIDATE the value
- Does this value make sense for this field type?
- Is it complete or partially visible?
- Does it match the expected format?

### 4. SCORE confidence honestly
- 0.95+: I'm absolutely certain, crystal clear
- 0.85-0.94: Clear with minor uncertainty on 1-2 characters
- 0.70-0.84: Readable but some quality issues
- <0.70: Too uncertain → return null instead
"""


# Negative examples - what NOT to do
EXTRACTION_ANTI_PATTERNS = """
## EXTRACTION ANTI-PATTERNS - NEVER DO THESE

### ❌ DON'T: Calculate or infer values
**Wrong:**
- Field: Total Charges
- Document shows: Three line items ($100, $50, $75) but total box is empty
- BAD output: {"value": "$225.00", "confidence": 0.8}
- Why wrong: Value was calculated, not read

**Correct:** {"value": null, "confidence": 0.0, "location": "Total box is empty"}

---

### ❌ DON'T: Fill in expected patterns
**Wrong:**
- Field: Provider NPI
- Document shows: Partially visible "123456____"
- BAD output: {"value": "1234567890", "confidence": 0.6}
- Why wrong: Last 4 digits were guessed based on typical NPI format

**Correct:** {"value": null, "confidence": 0.0, "location": "NPI partially obscured"}

---

### ❌ DON'T: Complete partial dates
**Wrong:**
- Field: Date of Service
- Document shows: "03/__/2024" (day is smudged)
- BAD output: {"value": "03/15/2024", "confidence": 0.7}
- Why wrong: Day was guessed

**Correct:** {"value": null, "confidence": 0.0, "location": "Day portion unreadable"}

---

### ❌ DON'T: Assume typical names
**Wrong:**
- Field: Patient Name
- Document shows: Handwritten name that's hard to read
- BAD output: {"value": "John Smith", "confidence": 0.6}
- Why wrong: Used common name when actual name was unclear

**Correct:** {"value": null, "confidence": 0.0, "location": "Handwriting illegible"}
"""


def build_extraction_prompt(
    schema_fields: list[dict[str, Any]],
    document_type: str,
    page_number: int,
    total_pages: int,
    is_first_pass: bool = True,
    include_reasoning: bool = True,
    include_anti_patterns: bool = True,
) -> str:
    """
    Build the main extraction prompt for a document page.

    Args:
        schema_fields: List of field definitions to extract.
        document_type: Type of document being processed.
        page_number: Current page number (1-indexed).
        total_pages: Total number of pages.
        is_first_pass: Whether this is the first or second extraction pass.
        include_reasoning: Whether to include chain-of-thought reasoning protocol.
        include_anti_patterns: Whether to include negative examples.

    Returns:
        Complete extraction prompt for the VLM.
    """
    pass_instruction = _get_pass_instruction(is_first_pass)
    field_instructions = _build_field_instructions(schema_fields)

    reasoning_section = ""
    if include_reasoning:
        reasoning_section = EXTRACTION_REASONING_TEMPLATE

    anti_patterns = ""
    if include_anti_patterns and is_first_pass:  # Only on first pass to save tokens
        anti_patterns = EXTRACTION_ANTI_PATTERNS

    prompt = f"""
## DOCUMENT EXTRACTION TASK - {document_type}

{pass_instruction}
{reasoning_section}

### Document Context
- Document Type: {document_type}
- Page: {page_number} of {total_pages}
- Extraction Pass: {"First (Focus: Completeness)" if is_first_pass else "Second (Focus: Accuracy)"}

### Fields to Extract

{field_instructions}

### EXTRACTION RULES

1. Extract ONLY values that are CLEARLY VISIBLE in the image
2. For each field, provide:
   - The extracted value (or null if not visible/unclear)
   - A confidence score from 0.0 to 1.0
   - A brief location description

3. If a field appears multiple times, extract the most prominent instance
4. For multi-value fields (lists), extract all visible values
{anti_patterns}

### REQUIRED OUTPUT FORMAT

Return a JSON object with this structure:

```json
{{
  "page_number": {page_number},
  "extraction_pass": {1 if is_first_pass else 2},
  "fields": {{
    "field_name": {{
      "value": "extracted value or null",
      "confidence": 0.95,
      "location": "where found in document"
    }}
  }},
  "extraction_notes": "any relevant observations about the extraction",
  "quality_issues": ["list of any quality problems encountered"]
}}
```

{build_null_handling_instruction()}

CRITICAL: Return null for any field you cannot read clearly. Do NOT guess.
"""

    return prompt


def build_verification_prompt(
    schema_fields: list[dict[str, Any]],
    document_type: str,
    page_number: int,
    first_pass_results: dict[str, Any],
) -> str:
    """
    Build the second-pass verification extraction prompt.

    This prompt is specifically designed to be different from the first pass
    to catch potential hallucinations through dual-pass comparison.

    Args:
        schema_fields: List of field definitions to extract.
        document_type: Type of document being processed.
        page_number: Current page number.
        first_pass_results: Results from first extraction pass (for context only).

    Returns:
        Verification extraction prompt for the VLM.
    """
    field_instructions = _build_field_instructions(schema_fields)

    # Note: We deliberately do NOT show first_pass_results to the model
    # to ensure independent extraction

    prompt = f"""
## SKEPTICAL VERIFICATION PASS - {document_type}

⚠️ This is an INDEPENDENT VERIFICATION pass. You are acting as a skeptical auditor.
Your job is to catch errors, NOT to confirm what might have been extracted before.

### VERIFICATION MINDSET

Think like a skeptical reviewer:
- "I will not assume any value is correct until I verify it myself"
- "I will be stricter with my confidence scores than I might normally be"
- "When in doubt, I will return null rather than guess"
- "I will read each character individually, not as a word I expect to see"

### CRITICAL VERIFICATION PROTOCOL

For EACH field, perform this verification:

**Step 1: INDEPENDENT LOCATION**
- Find the field location without assuming where it should be
- Verify the field label/identifier is visible

**Step 2: CHARACTER-BY-CHARACTER READING**
- Read each character individually: "I see: S-M-I-T-H"
- Check for easily confused characters: 0/O, 1/l/I, 5/S, 8/B
- Look for overwrites, corrections, or amendments

**Step 3: SKEPTICAL CONFIDENCE SCORING**
Apply stricter thresholds for verification:
| Verification Score | Meaning |
|-------------------|---------|
| 0.90+ | Every character crystal clear, no possible alternate reading |
| 0.75-0.89 | Clear but 1-2 characters could theoretically be different |
| 0.60-0.74 | Readable but would want second opinion |
| <0.60 | Too uncertain → MUST return null |

**Step 4: HALLUCINATION CHECK**
Before reporting each value, ask:
- "Am I reading this, or am I inferring what I expect to see?"
- "Is this suspiciously 'perfect' (round numbers, common names)?"
- "Would I bet money on this exact value?"

### Fields to Verify

{field_instructions}

### REQUIRED OUTPUT FORMAT

```json
{{
  "page_number": {page_number},
  "extraction_pass": 2,
  "verification_mode": true,
  "fields": {{
    "field_name": {{
      "value": "verified value or null",
      "confidence": 0.85,
      "location": "precise location in document",
      "character_verification": "individual characters read: X-X-X-X",
      "alternate_readings": ["other possible readings if any"]
    }}
  }},
  "verification_summary": {{
    "fields_confidently_verified": 0,
    "fields_with_uncertainty": 0,
    "fields_returned_null": 0
  }},
  "skepticism_notes": "any concerns or observations from verification"
}}
```

### VERIFICATION STANDARD

🎯 **Accuracy over completeness**: It is better to return null for a field than to report
   an uncertain value. Your role is to be the last line of defense against errors.

❌ **DO NOT**:
- Assume a value is correct because it "looks right"
- Fill in partially visible values
- Report values you're not confident about
- Use typical patterns to complete unclear fields

✓ **DO**:
- Return null when uncertain
- Note alternate possible readings
- Apply stricter confidence thresholds
- Report exactly what you can verify
"""

    return prompt


def _get_pass_instruction(is_first_pass: bool) -> str:
    """Get specific instructions for first or second extraction pass."""
    if is_first_pass:
        return """
### FIRST PASS EXTRACTION

Your goal is COMPLETENESS. Try to extract every field that is visible in the document.

Approach:
- Systematically scan all areas of the document
- Look for both printed and handwritten content
- Check header, body, and footer sections
- Note any fields that may be on other pages
"""
    return """
### SECOND PASS EXTRACTION (VERIFICATION)

Your goal is ACCURACY. Carefully verify each extraction with heightened scrutiny.

Approach:
- Take extra time on each field
- Double-check numbers and codes character by character
- Verify names are spelled correctly
- Confirm dates have correct month/day/year
- If anything is unclear upon closer inspection, mark as null
"""


def _build_field_instructions(schema_fields: list[dict[str, Any]]) -> str:
    """Build field-by-field extraction instructions."""
    instructions = []

    for field in schema_fields:
        name = field.get("name", "unknown")
        display = field.get("display_name", name)
        field_type = field.get("field_type", "string")
        description = field.get("description", "")
        examples = field.get("examples", [])
        location_hint = field.get("location_hint", "")
        required = field.get("required", False)

        parts = [f"**{display}** (`{name}`)"]

        if description:
            parts.append(f"  - {description}")

        parts.append(f"  - Type: {field_type}")

        if examples:
            example_str = ", ".join(str(e) for e in examples[:3])
            parts.append(f"  - Examples: {example_str}")

        if location_hint:
            parts.append(f"  - Usually found: {location_hint}")

        if required:
            parts.append("  - **Required field**")

        instructions.append("\n".join(parts))

    return "\n\n".join(instructions)


def build_field_prompt(
    field_definition: dict[str, Any],
    document_type: str,
    additional_context: str = "",
) -> str:
    """
    Build a prompt for extracting a single specific field.

    Used for targeted re-extraction of problematic fields.

    Args:
        field_definition: The field definition to extract.
        document_type: Type of document.
        additional_context: Additional extraction context.

    Returns:
        Single-field extraction prompt.
    """
    name = field_definition.get("name", "unknown")
    display = field_definition.get("display_name", name)
    field_type = field_definition.get("field_type", "string")
    description = field_definition.get("description", "")
    examples = field_definition.get("examples", [])
    pattern = field_definition.get("pattern", "")
    location_hint = field_definition.get("location_hint", "")

    example_str = ""
    if examples:
        example_str = f"\nExamples of valid values: {', '.join(str(e) for e in examples[:5])}"

    pattern_str = ""
    if pattern:
        pattern_str = f"\nExpected format pattern: {pattern}"

    location_str = ""
    if location_hint:
        location_str = f"\n\nThis field is typically found: {location_hint}"

    context_str = ""
    if additional_context:
        context_str = f"\n\nAdditional context: {additional_context}"

    return f"""
## SINGLE FIELD EXTRACTION

Extract the following field from this {document_type} document:

### Field: {display}

Technical name: `{name}`
Data type: {field_type}
Description: {description}{example_str}{pattern_str}{location_str}{context_str}

### EXTRACTION INSTRUCTIONS

1. Locate this specific field in the document
2. If found, extract the value exactly as shown
3. Provide a confidence score based on visibility
4. Describe where in the document you found it

### REQUIRED OUTPUT

```json
{{
  "field_name": "{name}",
  "value": "extracted value or null",
  "confidence": 0.95,
  "location": "description of where found",
  "found": true,
  "notes": "any relevant observations"
}}
```

If the field is not visible or cannot be read clearly, return:

```json
{{
  "field_name": "{name}",
  "value": null,
  "confidence": 0.0,
  "location": null,
  "found": false,
  "notes": "reason why field could not be extracted"
}}
```
"""


def build_table_extraction_prompt(
    table_schema: dict[str, Any],
    document_type: str,
    table_location: str = "",
    expected_rows: int | None = None,
) -> str:
    """
    Build prompt for extracting tabular data (e.g., service line items).

    Args:
        table_schema: Schema definition for table columns.
        document_type: Type of document.
        table_location: Description of where table is located.
        expected_rows: Expected number of rows if known.

    Returns:
        Table extraction prompt.
    """
    columns = table_schema.get("columns", [])
    table_name = table_schema.get("name", "table")
    description = table_schema.get("description", "")

    column_instructions = []
    for col in columns:
        col_name = col.get("name", "column")
        col_type = col.get("field_type", "string")
        col_desc = col.get("description", "")
        column_instructions.append(f"- **{col_name}** ({col_type}): {col_desc}")

    columns_str = "\n".join(column_instructions)

    location_str = ""
    if table_location:
        location_str = f"\n\nTable location: {table_location}"

    rows_str = ""
    if expected_rows:
        rows_str = f"\nExpected rows: approximately {expected_rows}"

    return f"""
## TABLE EXTRACTION TASK

Extract the {table_name} table from this {document_type} document.

### Table Description
{description}{location_str}{rows_str}

### Columns to Extract

{columns_str}

### TABLE EXTRACTION RULES

1. Identify all rows in the table
2. Extract each column value for each row
3. Only include rows with actual data (skip blank rows)
4. Maintain row order as shown in document
5. If a cell is empty, use null
6. If a cell is unreadable, use null with low confidence

### REQUIRED OUTPUT FORMAT

```json
{{
  "table_name": "{table_name}",
  "rows": [
    {{
      "row_number": 1,
      "columns": {{
        "column_name": {{
          "value": "cell value or null",
          "confidence": 0.95
        }}
      }}
    }}
  ],
  "total_rows_found": 5,
  "rows_extracted": 5,
  "extraction_quality": "complete | partial | poor",
  "notes": "any observations about the table"
}}
```

### IMPORTANT

- Extract ONLY visible rows with data
- Do NOT create placeholder rows
- If you cannot determine row boundaries, note this in extraction notes
- Each row must have all columns, even if some are null
"""


def build_list_field_extraction_prompt(
    field_name: str,
    item_type: str,
    document_type: str,
    max_items: int = 20,
) -> str:
    """
    Build prompt for extracting list/array fields.

    Args:
        field_name: Name of the list field.
        item_type: Type of items in the list (e.g., 'icd10_code').
        document_type: Type of document.
        max_items: Maximum expected items.

    Returns:
        List extraction prompt.
    """
    return f"""
## LIST FIELD EXTRACTION

Extract all instances of {field_name} from this {document_type} document.

### Field Details
- Field name: {field_name}
- Item type: {item_type}
- Maximum expected items: {max_items}

### EXTRACTION RULES

1. Find all instances of this field type in the document
2. Extract each instance as a separate list item
3. Maintain the order as they appear in the document
4. Include confidence for each item
5. Do NOT include duplicates (same value, same location)

### REQUIRED OUTPUT FORMAT

```json
{{
  "field_name": "{field_name}",
  "items": [
    {{
      "value": "item value",
      "confidence": 0.95,
      "location": "where found",
      "position": 1
    }}
  ],
  "total_found": 5,
  "notes": "any observations"
}}
```

### IMPORTANT

- Only include clearly visible items
- Mark any uncertain items with low confidence
- If no items found, return empty list (not null)
"""
