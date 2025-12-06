"""
Validation prompts for the Validator agent.

Provides prompts for hallucination detection, cross-field validation,
and final verification.
"""

from typing import Any


def build_validation_prompt(
    extracted_data: dict[str, Any],
    document_type: str,
    schema_rules: list[dict[str, Any]],
) -> str:
    """
    Build prompt for validating extracted data.

    Args:
        extracted_data: The extracted field values.
        document_type: Type of document.
        schema_rules: Cross-field validation rules from schema.

    Returns:
        Validation prompt for the VLM.
    """
    rules_str = _format_validation_rules(schema_rules)
    data_str = _format_extracted_data(extracted_data)

    return f"""
## EXTRACTION VALIDATION TASK

Review the extracted data for a {document_type} document and validate it against
the document image and validation rules.

### Extracted Data to Validate

{data_str}

### Validation Rules

{rules_str}

### VALIDATION CHECKLIST

For each extracted field, verify:

1. **Visual Confirmation**: Can you see this exact value in the document?
2. **Format Correctness**: Does the value match expected format for its type?
3. **Logical Consistency**: Does the value make sense in context?
4. **Cross-Field Validity**: Does it satisfy all cross-field rules?

### HALLUCINATION DETECTION

Flag as potential hallucination if:
- Value not visually present in document
- Value suspiciously matches "typical" patterns
- Round numbers ($1000.00, $500.00 exactly)
- Placeholder patterns (N/A, TBD, XXX, 123)
- Repetitive values across multiple fields
- Values that seem "too perfect"

### REQUIRED OUTPUT FORMAT

```json
{{
  "validation_passed": true,
  "field_validations": {{
    "field_name": {{
      "valid": true,
      "errors": [],
      "warnings": [],
      "visually_confirmed": true
    }}
  }},
  "cross_field_validations": [
    {{
      "rule": "rule description",
      "passed": true,
      "message": "validation message"
    }}
  ],
  "hallucination_flags": [
    {{
      "field": "field_name",
      "reason": "why flagged as potential hallucination",
      "confidence": 0.8
    }}
  ],
  "overall_assessment": {{
    "quality": "high | medium | low",
    "requires_review": false,
    "review_reason": null
  }}
}}
```
"""


def build_hallucination_check_prompt(
    field_name: str,
    extracted_value: Any,
    field_type: str,
    confidence: float,
) -> str:
    """
    Build prompt for targeted hallucination check on a specific field.

    Args:
        field_name: Name of the field to check.
        extracted_value: The extracted value.
        field_type: Type of the field.
        confidence: Reported confidence score.

    Returns:
        Hallucination check prompt.
    """
    return f"""
## HALLUCINATION VERIFICATION

Carefully verify whether the following extracted value is actually present
in the document image, or if it may be a hallucination.

### Field to Verify

- Field: {field_name}
- Extracted Value: {extracted_value}
- Field Type: {field_type}
- Reported Confidence: {confidence}

### VERIFICATION INSTRUCTIONS

1. Locate where this field should appear in the document
2. Carefully read the actual value shown in the document
3. Compare character-by-character with the extracted value
4. Look for any signs of alteration, correction, or ambiguity

### HALLUCINATION INDICATORS

Check for these warning signs:
- Value not actually visible in document
- Value partially matches but with "filled in" portions
- Suspiciously perfect or round value
- Value matches common placeholder patterns
- Handwritten portion "interpreted" rather than read

### REQUIRED OUTPUT

```json
{{
  "field_name": "{field_name}",
  "verification_result": "confirmed | suspicious | hallucination",
  "actual_value_seen": "what you actually see in the document or null",
  "matches_extraction": true,
  "discrepancies": [
    "list of any differences between extracted and actual"
  ],
  "hallucination_confidence": 0.0,
  "verification_notes": "detailed observations about the verification"
}}
```

### IMPORTANT

Be skeptical. Your job is to catch errors, not confirm extractions.
If there is ANY doubt, mark as suspicious rather than confirmed.
"""


def build_cross_field_validation_prompt(
    field1_name: str,
    field1_value: Any,
    field2_name: str,
    field2_value: Any,
    rule_description: str,
) -> str:
    """
    Build prompt for validating relationship between two fields.

    Args:
        field1_name: Name of first field.
        field1_value: Value of first field.
        field2_name: Name of second field.
        field2_value: Value of second field.
        rule_description: Description of the validation rule.

    Returns:
        Cross-field validation prompt.
    """
    return f"""
## CROSS-FIELD VALIDATION

Verify that the following field relationship is valid:

### Fields

- {field1_name}: {field1_value}
- {field2_name}: {field2_value}

### Rule to Validate

{rule_description}

### VALIDATION INSTRUCTIONS

1. Consider the values of both fields
2. Apply the validation rule
3. Determine if the relationship is valid
4. Check if both values are visually confirmed in the document

### REQUIRED OUTPUT

```json
{{
  "rule": "{rule_description}",
  "field1": "{field1_name}",
  "field2": "{field2_name}",
  "valid": true,
  "message": "Explanation of validation result",
  "both_fields_confirmed": true
}}
```
"""


def build_confidence_recalibration_prompt(
    field_name: str,
    value: Any,
    original_confidence: float,
    pass1_value: Any,
    pass2_value: Any,
    passes_agree: bool,
) -> str:
    """
    Build prompt for recalibrating field confidence based on dual-pass results.

    Args:
        field_name: Name of the field.
        value: Final merged value.
        original_confidence: Originally reported confidence.
        pass1_value: Value from first extraction pass.
        pass2_value: Value from second extraction pass.
        passes_agree: Whether both passes agree.

    Returns:
        Confidence recalibration prompt.
    """
    agreement_status = "agree" if passes_agree else "disagree"

    return f"""
## CONFIDENCE RECALIBRATION

Review the dual-pass extraction results and recalibrate the confidence score.

### Field: {field_name}

- Pass 1 Value: {pass1_value}
- Pass 2 Value: {pass2_value}
- Passes {agreement_status}
- Final Value: {value}
- Original Confidence: {original_confidence}

### RECALIBRATION RULES

When passes AGREE:
- If both high confidence (>0.85): Increase confidence slightly
- If both medium confidence: Maintain average
- If one high, one medium: Use lower value

When passes DISAGREE:
- Significant difference: Set confidence to 0.5 or lower
- Minor formatting difference: Maintain moderate confidence
- One null, one value: Set to 0.6 maximum

### VISUAL VERIFICATION

Look at the document and verify:
- Is the final value clearly visible?
- Are there any ambiguities?
- Could the value be read differently?

### REQUIRED OUTPUT

```json
{{
  "field_name": "{field_name}",
  "original_confidence": {original_confidence},
  "recalibrated_confidence": 0.85,
  "confidence_change": "increased | decreased | maintained",
  "change_reason": "Explanation for confidence adjustment",
  "visual_verification": "confirmed | uncertain | not_visible"
}}
```
"""


def _format_validation_rules(rules: list[dict[str, Any]]) -> str:
    """Format validation rules for prompt."""
    if not rules:
        return "No specific cross-field rules defined."

    formatted = []
    for i, rule in enumerate(rules, 1):
        source = rule.get("source_field", "")
        target = rule.get("target_field", "")
        operator = rule.get("operator", "")
        message = rule.get("error_message", "")

        formatted.append(
            f"{i}. **{source}** {operator} **{target}**\n   {message}"
        )

    return "\n\n".join(formatted)


def _format_extracted_data(data: dict[str, Any]) -> str:
    """Format extracted data for validation prompt."""
    formatted = []

    for field_name, field_data in data.items():
        if isinstance(field_data, dict):
            value = field_data.get("value", field_data)
            confidence = field_data.get("confidence", "N/A")
            formatted.append(
                f"- **{field_name}**: `{value}` (confidence: {confidence})"
            )
        else:
            formatted.append(f"- **{field_name}**: `{field_data}`")

    return "\n".join(formatted)


def build_final_review_prompt(
    extraction_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    document_type: str,
) -> str:
    """
    Build prompt for final extraction review.

    Args:
        extraction_summary: Summary of extracted data.
        validation_summary: Summary of validation results.
        document_type: Type of document.

    Returns:
        Final review prompt.
    """
    return f"""
## FINAL EXTRACTION REVIEW

Perform a final review of the extraction results for this {document_type} document.

### Extraction Summary

Total fields extracted: {extraction_summary.get('total_fields', 0)}
Fields with high confidence: {extraction_summary.get('high_confidence', 0)}
Fields with medium confidence: {extraction_summary.get('medium_confidence', 0)}
Fields with low confidence: {extraction_summary.get('low_confidence', 0)}
Fields not found: {extraction_summary.get('not_found', 0)}

### Validation Summary

Validation passed: {validation_summary.get('passed', False)}
Fields validated: {validation_summary.get('fields_validated', 0)}
Hallucination flags: {validation_summary.get('hallucination_count', 0)}
Cross-field rules passed: {validation_summary.get('rules_passed', 0)}

### FINAL REVIEW CHECKLIST

1. Are all required fields extracted?
2. Do the values make logical sense together?
3. Are there any obvious errors or inconsistencies?
4. Is the overall quality acceptable for downstream use?

### RECOMMENDATION

Based on your review, provide a recommendation:

```json
{{
  "recommendation": "accept | review | reject",
  "confidence_in_recommendation": 0.95,
  "key_concerns": ["list of main concerns if any"],
  "suggested_corrections": [
    {{
      "field": "field_name",
      "issue": "what's wrong",
      "suggestion": "what should be done"
    }}
  ],
  "quality_score": 85,
  "ready_for_export": true
}}
```
"""
