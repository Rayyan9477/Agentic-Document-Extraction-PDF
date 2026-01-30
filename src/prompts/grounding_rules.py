"""
Grounding rules for anti-hallucination in document extraction.

Provides the foundational prompt engineering layer (Layer 1) of the
3-layer anti-hallucination system. These rules are embedded in all
extraction prompts to ensure VLM outputs are grounded in visual evidence.

Enhanced with:
- Chain-of-thought reasoning patterns
- Self-verification checkpoints
- Few-shot examples for proper extraction
- Constitutional AI-style critique prompts
"""

GROUNDING_RULES = """
## CRITICAL GROUNDING RULES - YOU MUST FOLLOW THESE EXACTLY

### Core Principle: Visual Evidence Only
Extract ONLY what you can SEE clearly in the document. Your role is a careful reader,
NOT an intelligent guesser.

### Rule 1: VISUAL GROUNDING
Only extract values that are CLEARLY VISIBLE in the document image.
- If you cannot see the text clearly → return null
- If the text is blurry, obscured, or cut off → return null
- Never guess or infer values that are not explicitly shown

### Rule 2: NO GUESSING
If any field is unclear, blurry, or not visible:
- Return null for that field
- Do NOT make assumptions based on document type
- Do NOT use typical/expected values

### Rule 3: NO INFERENCE
Do not calculate or derive values:
- Do NOT calculate totals from line items
- Do NOT infer dates from context
- Do NOT complete partial information
- Do NOT assume patterns continue

### Rule 4: NO DEFAULTS
Never fill in "typical" or "expected" values:
- Do NOT use placeholder names like "John Doe"
- Do NOT use default dates like "01/01/2000"
- Do NOT use round numbers like "$1000.00" unless exactly shown
- Do NOT use common values like "$0.00" unless clearly shown

### Rule 5: CONFIDENCE SCORING
For EVERY field you extract, provide a confidence score from 0.0 to 1.0:
| Score | Meaning | Action |
|-------|---------|--------|
| 0.95-1.00 | Crystal clear, no ambiguity | Extract with full confidence |
| 0.85-0.94 | Clear with minor quality issues | Extract, note any issues |
| 0.70-0.84 | Readable but some blur/noise | Extract with caution |
| 0.50-0.69 | Partially obscured | Consider returning null instead |
| <0.50 | Significant uncertainty | MUST return null |

### Rule 6: LOCATION DESCRIPTION
For each extracted value, describe WHERE in the document you found it:
- "Top-left corner, Box 1"
- "Service line 3, Column D"
- "Bottom right, signature area"

### Rule 7: UNCERTAINTY HANDLING
When uncertain between multiple readings:
- Return null rather than guessing
- Do NOT pick the "most likely" value
- If forced to include uncertain value, set confidence to 0.0
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


# Chain-of-Thought reasoning pattern for extraction
CHAIN_OF_THOUGHT_TEMPLATE = """
## EXTRACTION REASONING PROTOCOL

For each field, follow this step-by-step reasoning process:

### Step 1: LOCATE
- Where should this field appear based on document type?
- Can I see a label or box for this field?
- Is the area where this field should be visible?

### Step 2: READ
- What characters/text do I see in this location?
- Are all characters clearly visible?
- Is there any blur, overlap, or obstruction?

### Step 3: VERIFY
- Does my reading match the expected field type (date, number, name, etc.)?
- Is this a complete value or is part cut off?
- Could any character be read differently?

### Step 4: CONFIDENCE
- How certain am I about each character?
- Is there any ambiguity in my reading?
- Should I return null instead of a uncertain value?

### Step 5: EXTRACT or SKIP
- If confidence ≥ 0.7: Extract with confidence score
- If confidence < 0.7: Return null with explanation
"""


# Self-verification checkpoint prompts
SELF_VERIFICATION_CHECKPOINT = """
## BEFORE YOU RESPOND - SELF-VERIFICATION CHECKLIST

⏸️ STOP and verify before submitting your extraction:

□ **Visual Verification**: For each value I'm reporting, can I point to the exact
  location in the document where I see this text?

□ **Character Check**: Did I read each character individually, or did I assume
  what the text should say?

□ **Hallucination Check**: Is any value I'm reporting suspiciously:
  - Round numbers ($1000.00, $500.00)?
  - Common placeholder names (John Doe, Jane Smith)?
  - Default dates (01/01/2000, 12/31/9999)?
  - Typical codes that "should" be there?

□ **Null Check**: Did I return null for any field I couldn't read clearly,
  or did I try to "help" by guessing?

□ **Confidence Calibration**: Are my confidence scores honest? A score of 0.95
  means I'm almost certain - is that true for each field?

If ANY answer is "no", revise your extraction before responding.
"""


# Few-shot examples for proper extraction behavior
FEW_SHOT_EXAMPLES = """
## EXTRACTION EXAMPLES - LEARN FROM THESE

### GOOD EXTRACTION EXAMPLE ✓
```
Field: Patient Name
Document shows: "SMITH, JOHN A" printed clearly in Box 2
```
Response:
```json
{
  "patient_name": {
    "value": "SMITH, JOHN A",
    "confidence": 0.95,
    "location": "Box 2, top-left section"
  }
}
```
Why this is correct: Value exactly matches what's visible, high confidence is justified.

### GOOD NULL EXAMPLE ✓
```
Field: Date of Birth
Document shows: Box 3 has handwriting that's partially smudged, looks like "0?/15/19?8"
```
Response:
```json
{
  "date_of_birth": {
    "value": null,
    "confidence": 0.0,
    "location": "Box 3 - text smudged, cannot read reliably"
  }
}
```
Why this is correct: Uncertain characters → return null rather than guess.

### BAD EXTRACTION EXAMPLE ✗
```
Field: Total Charges
Document shows: Line items but total box is empty
```
BAD Response (DO NOT DO THIS):
```json
{
  "total_charges": {
    "value": "$1,234.56",
    "confidence": 0.8,
    "location": "Calculated from line items"
  }
}
```
Why this is WRONG: Value was calculated/inferred, not read from document.

CORRECT Response:
```json
{
  "total_charges": {
    "value": null,
    "confidence": 0.0,
    "location": "Total box is empty"
  }
}
```

### BAD PLACEHOLDER EXAMPLE ✗
```
Field: Provider NPI
Document shows: NPI field is blank
```
BAD Response (DO NOT DO THIS):
```json
{
  "provider_npi": {
    "value": "1234567890",
    "confidence": 0.7,
    "location": "Assumed standard format"
  }
}
```
Why this is WRONG: Value was made up using typical pattern.

CORRECT Response:
```json
{
  "provider_npi": {
    "value": null,
    "confidence": 0.0,
    "location": "NPI field is blank"
  }
}
```
"""


# Constitutional AI-style critique prompt
CONSTITUTIONAL_CRITIQUE = """
## SELF-CRITIQUE PROTOCOL

After generating your extraction, mentally review it as a skeptical auditor:

### Critique Questions:
1. "Is this value actually visible, or am I filling in what I expect to see?"
2. "Would another person reading this document arrive at the same value?"
3. "Am I being overconfident? Should any of my 0.9+ scores actually be lower?"
4. "Did I return null for anything that was unclear, or did I rationalize a guess?"
5. "Are there any suspiciously 'perfect' values that might be hallucinations?"

### If You Catch an Error:
- Revise the extraction before responding
- Lower confidence scores where appropriate
- Change uncertain values to null
- Add notes explaining any ambiguity
"""


def build_grounded_system_prompt(
    additional_context: str = "",
    include_forbidden: bool = True,
    include_confidence_scale: bool = True,
    include_chain_of_thought: bool = False,
    include_few_shot_examples: bool = False,
    include_self_verification: bool = False,
    include_constitutional_critique: bool = False,
) -> str:
    """
    Build a complete system prompt with grounding rules.

    Args:
        additional_context: Additional context specific to the task.
        include_forbidden: Whether to include forbidden actions list.
        include_confidence_scale: Whether to include confidence guidelines.
        include_chain_of_thought: Whether to include reasoning protocol.
        include_few_shot_examples: Whether to include good/bad extraction examples.
        include_self_verification: Whether to include self-verification checklist.
        include_constitutional_critique: Whether to include self-critique protocol.

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

    if include_chain_of_thought:
        parts.extend(["", CHAIN_OF_THOUGHT_TEMPLATE])

    if include_few_shot_examples:
        parts.extend(["", FEW_SHOT_EXAMPLES])

    parts.extend(["", OUTPUT_FORMAT_INSTRUCTION])

    if include_self_verification:
        parts.extend(["", SELF_VERIFICATION_CHECKPOINT])

    if include_constitutional_critique:
        parts.extend(["", CONSTITUTIONAL_CRITIQUE])

    if additional_context:
        parts.extend(["", "## ADDITIONAL CONTEXT", "", additional_context])

    return "\n".join(parts)


def build_enhanced_system_prompt(
    document_type: str,
    is_verification_pass: bool = False,
    structure_context: dict | None = None,
) -> str:
    """
    Build an enhanced system prompt with all anti-hallucination features.

    This is the recommended prompt builder for production use, including:
    - Full grounding rules
    - Chain-of-thought reasoning
    - Few-shot examples
    - Self-verification checkpoints
    - Constitutional critique
    - Document structure awareness (tables, handwriting, layout)

    Args:
        document_type: Type of document being processed.
        is_verification_pass: Whether this is a verification (second) pass.
        structure_context: Optional structure analysis results from the analyzer
            agent, containing detected tables, handwriting, layout type, etc.

    Returns:
        Complete enhanced system prompt.
    """
    # Build structure-aware context if available
    context_parts = [build_hallucination_warning(document_type)]

    if structure_context:
        structure_hints = []
        if structure_context.get("has_tables"):
            table_count = structure_context.get("table_count", 1)
            structure_hints.append(
                f"This document contains {table_count} table(s). "
                "Pay close attention to row/column alignment when extracting table data."
            )
        if structure_context.get("has_handwriting"):
            structure_hints.append(
                "This document contains handwritten text. "
                "Be extra cautious with handwritten fields — return null if illegible."
            )
        if structure_context.get("has_signatures"):
            structure_hints.append(
                "This document contains signature areas. "
                "Do NOT attempt to extract text from signature regions."
            )
        layout_type = structure_context.get("layout_type", "")
        if layout_type:
            structure_hints.append(f"Document layout type: {layout_type}.")
        text_density = structure_context.get("text_density", "")
        if text_density:
            structure_hints.append(f"Text density: {text_density}.")

        if structure_hints:
            context_parts.append(
                "\n\n## DOCUMENT STRUCTURE CONTEXT\n" + "\n".join(f"- {h}" for h in structure_hints)
            )

    additional_context = "\n".join(context_parts)

    # Token-optimized: GROUNDING_RULES already contains a confidence scale (Rule 5)
    # and the user prompt includes EXTRACTION_REASONING_TEMPLATE with reasoning steps.
    # Skipping redundant CONFIDENCE_SCALE and CHAIN_OF_THOUGHT_TEMPLATE saves ~1000 tokens
    # which is critical for the 8B model's effective attention budget.
    return build_grounded_system_prompt(
        additional_context=additional_context,
        include_forbidden=True,
        include_confidence_scale=False,  # Already in GROUNDING_RULES Rule 5
        include_chain_of_thought=False,  # Already in user prompt EXTRACTION_REASONING_TEMPLATE
        include_few_shot_examples=False,  # Zero-shot mode: no examples, rely on grounding rules
        include_self_verification=True,
        include_constitutional_critique=is_verification_pass,  # Add critique on verification
    )


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
""",
    )
