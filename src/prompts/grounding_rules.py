"""
Grounding rules for anti-hallucination in document extraction.

Provides the foundational prompt engineering layer (Layer 1) of the
3-layer anti-hallucination system. These rules are embedded in all
extraction prompts to ensure VLM outputs are grounded in visual evidence.
"""

from typing import Any


GROUNDING_RULES = """
## CRITICAL GROUNDING RULES - YOU MUST FOLLOW THESE EXACTLY

1. **VISUAL GROUNDING**: Only extract values that are CLEARLY VISIBLE in the document image.
   - If you cannot see the text clearly, return null for that field.
   - If the text is blurry, obscured, or cut off, return null.
   - Never guess or infer values that are not explicitly shown.

2. **NO GUESSING**: If any field is unclear, blurry, or not visible:
   - Return null for that field
   - Do NOT make assumptions based on document type
   - Do NOT use typical/expected values

3. **NO INFERENCE**: Do not calculate or derive values:
   - Do NOT calculate totals from line items
   - Do NOT infer dates from context
   - Do NOT complete partial information

4. **NO DEFAULTS**: Never fill in "typical" or "expected" values:
   - Do NOT use placeholder names like "John Doe"
   - Do NOT use default dates like "01/01/2000"
   - Do NOT use common values like "$0.00" unless clearly shown

5. **CONFIDENCE SCORING**: For EVERY field you extract:
   - Provide a confidence score from 0.0 to 1.0
   - 1.0 = Perfectly clear, no ambiguity
   - 0.8-0.9 = Clear but minor quality issues
   - 0.5-0.7 = Readable but some uncertainty
   - <0.5 = Uncertain, should be null instead

6. **LOCATION DESCRIPTION**: For each extracted value:
   - Describe WHERE in the document you found it
   - Example: "Top-left corner", "Box 21a", "Second row of table"

7. **UNCERTAINTY HANDLING**: When uncertain between multiple readings:
   - Return null rather than guessing
   - Do NOT pick the "most likely" value
   - Mark confidence as 0.0 if you must include a guess
"""

FORBIDDEN_ACTIONS = """
## FORBIDDEN ACTIONS - NEVER DO THESE

❌ Making up patient names, dates, or medical codes
❌ Guessing values based on document type expectations
❌ Filling placeholder values like "N/A", "TBD", "XXX", or "123"
❌ Assuming standard formats if not clearly visible
❌ Completing partial SSN, phone numbers, or account numbers
❌ Calculating totals, balances, or derived values
❌ Using previous extraction results to fill current fields
❌ Inferring provider information from document headers
❌ Assuming dates are in a specific format without visual confirmation
❌ Extrapolating data from similar documents
"""

CONFIDENCE_SCALE = """
## CONFIDENCE SCORE GUIDELINES

| Score | Meaning | When to Use |
|-------|---------|-------------|
| 0.95-1.00 | Certain | Crystal clear text, perfect quality, no ambiguity |
| 0.85-0.94 | High | Clear text with minor quality issues, single valid interpretation |
| 0.70-0.84 | Medium | Readable but some blur/noise, confident in reading |
| 0.50-0.69 | Low | Partially obscured, multiple possible readings |
| 0.01-0.49 | Very Low | Barely visible, significant uncertainty |
| 0.00 | None | Cannot read, should be null |
"""

OUTPUT_FORMAT_INSTRUCTION = """
## REQUIRED OUTPUT FORMAT

Return a JSON object with this exact structure for each field:

```json
{
  "field_name": {
    "value": "extracted value or null",
    "confidence": 0.95,
    "location": "description of where found"
  }
}
```

IMPORTANT:
- Use null (not "null" string, not "", not "N/A") for missing/unreadable fields
- Confidence must be a decimal number between 0.0 and 1.0
- Location must describe where in the document the value was found
"""


def build_grounded_system_prompt(
    additional_context: str = "",
    include_forbidden: bool = True,
    include_confidence_scale: bool = True,
) -> str:
    """
    Build a complete system prompt with grounding rules.

    Args:
        additional_context: Additional context specific to the task.
        include_forbidden: Whether to include forbidden actions list.
        include_confidence_scale: Whether to include confidence guidelines.

    Returns:
        Complete system prompt with grounding rules.
    """
    parts = [
        "You are a document extraction specialist. Your task is to accurately extract "
        "information from document images while strictly adhering to grounding rules "
        "to prevent hallucinations and ensure accuracy.",
        "",
        GROUNDING_RULES,
    ]

    if include_forbidden:
        parts.extend(["", FORBIDDEN_ACTIONS])

    if include_confidence_scale:
        parts.extend(["", CONFIDENCE_SCALE])

    parts.extend(["", OUTPUT_FORMAT_INSTRUCTION])

    if additional_context:
        parts.extend(["", "## ADDITIONAL CONTEXT", "", additional_context])

    return "\n".join(parts)


def build_confidence_instruction(field_name: str, field_type: str) -> str:
    """
    Build confidence scoring instruction for a specific field.

    Args:
        field_name: Name of the field.
        field_type: Type of the field (e.g., 'date', 'currency', 'code').

    Returns:
        Specific confidence instruction for the field.
    """
    type_instructions = {
        "date": (
            f"For '{field_name}' (date field):\n"
            "- High confidence (0.9+): All digits clearly visible, format unambiguous\n"
            "- Medium confidence (0.7-0.9): Most digits visible, format recognizable\n"
            "- Low confidence (<0.7): Some digits unclear, format uncertain\n"
            "- Return null if you cannot read at least the year"
        ),
        "currency": (
            f"For '{field_name}' (currency field):\n"
            "- High confidence (0.9+): Dollar sign and all digits clear\n"
            "- Medium confidence (0.7-0.9): Amount visible but decimal may be unclear\n"
            "- Low confidence (<0.7): Partial amount visible\n"
            "- Return null if you cannot determine the dollar amount"
        ),
        "code": (
            f"For '{field_name}' (medical code field):\n"
            "- High confidence (0.9+): All characters clearly visible\n"
            "- Medium confidence (0.7-0.9): Code visible but one character uncertain\n"
            "- Low confidence (<0.7): Multiple characters unclear\n"
            "- Return null if any character is truly unreadable"
        ),
        "name": (
            f"For '{field_name}' (name field):\n"
            "- High confidence (0.9+): Full name clearly legible\n"
            "- Medium confidence (0.7-0.9): Name readable but some letters unclear\n"
            "- Low confidence (<0.7): Significant portions unclear\n"
            "- Return null if you cannot make out the name"
        ),
        "identifier": (
            f"For '{field_name}' (identifier field):\n"
            "- High confidence (0.9+): All digits/characters clear\n"
            "- Medium confidence (0.7-0.9): Most characters clear\n"
            "- Low confidence (<0.7): Several characters uncertain\n"
            "- Return null if critical characters are unreadable"
        ),
    }

    default_instruction = (
        f"For '{field_name}':\n"
        "- High confidence (0.9+): Value completely clear and unambiguous\n"
        "- Medium confidence (0.7-0.9): Value readable with minor uncertainty\n"
        "- Low confidence (<0.7): Value partially visible or uncertain\n"
        "- Return null if the value cannot be reliably read"
    )

    return type_instructions.get(field_type, default_instruction)


def build_null_handling_instruction() -> str:
    """
    Build instruction for proper null value handling.

    Returns:
        Null handling instruction text.
    """
    return """
## NULL VALUE HANDLING

When to return null for a field:
- The field location is empty or blank
- The text is too blurry to read reliably
- The field is obscured by marks, stamps, or damage
- The value is partially cut off at page edge
- Multiple conflicting values appear in the same location
- Handwriting is illegible
- The expected field does not appear in the document

When NOT to return null:
- The field contains a valid value you can read clearly
- The field shows a zero (0, $0.00, etc.) - this is a value, not null
- The field shows "None", "N/A" as actual document content (extract as written)

IMPORTANT: null means "could not extract" - it does NOT mean "the document shows no value"
"""


def build_hallucination_warning(document_type: str) -> str:
    """
    Build document-type specific hallucination warnings.

    Args:
        document_type: Type of document being extracted.

    Returns:
        Document-specific hallucination warnings.
    """
    warnings = {
        "CMS-1500": """
## CMS-1500 SPECIFIC WARNINGS

Common hallucination patterns to avoid:
- Do NOT assume Box 21 contains ICD-10 codes starting with common letters
- Do NOT fill service line items based on the diagnosis
- Do NOT calculate total charges from line items
- Do NOT assume provider NPI is 10 digits starting with "1"
- Do NOT assume dates follow MM/DD/YYYY format without visual confirmation
- Do NOT fill in Box 33 provider info from letterhead
""",
        "UB-04": """
## UB-04 SPECIFIC WARNINGS

Common hallucination patterns to avoid:
- Do NOT assume admission dates from statement period
- Do NOT calculate total charges from revenue codes
- Do NOT assume HCPCS codes match revenue codes
- Do NOT fill occurrence codes based on admission type
- Do NOT assume patient control number format
""",
        "EOB": """
## EOB SPECIFIC WARNINGS

Common hallucination patterns to avoid:
- Do NOT calculate patient responsibility from allowed amounts
- Do NOT assume payment dates from check numbers
- Do NOT fill adjustment reasons from payment amounts
- Do NOT assume member ID format from plan name
- Do NOT calculate coinsurance percentages
""",
        "SUPERBILL": """
## SUPERBILL SPECIFIC WARNINGS

Common hallucination patterns to avoid:
- Do NOT assume checked services from visible codes
- Do NOT calculate total from individual charges
- Do NOT fill diagnosis codes based on specialty
- Do NOT assume date of service is visit date
- Do NOT extract provider info from pre-printed areas
""",
    }

    return warnings.get(
        document_type,
        """
## GENERAL EXTRACTION WARNINGS

- Verify every extracted value against what is actually visible
- Do not use any prior knowledge about typical document values
- Extract only what you can clearly see in this specific image
"""
    )
