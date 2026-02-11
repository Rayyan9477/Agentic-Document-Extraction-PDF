"""
Synthetic few-shot examples for document-type-specific extraction.

These are NOT real data — purely structural format guides that show the VLM
what good extraction output looks like for specific document types. Embedding
these in the extraction prompt improves field format adherence and reduces
malformed output (dates as ISO, currency as numbers, null for missing).

Each example is ~100-150 tokens to stay within the 8B model's attention budget.
"""

from __future__ import annotations

SYNTHETIC_EXAMPLES: dict[str, dict[str, str]] = {
    "medical_superbill": {
        "patient": (
            "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
            "{\n"
            '  "record_id": 1,\n'
            '  "primary_identifier": "DOE, JANE",\n'
            '  "fields": {\n'
            '    "patient_name": "DOE, JANE",\n'
            '    "date_of_birth": "1985-03-15",\n'
            '    "patient_id": "MRN123456",\n'
            '    "service_date": "2024-01-10",\n'
            '    "total_charge": 150.00,\n'
            '    "diagnosis_code": "Z00.00",\n'
            '    "provider_name": null\n'
            "  },\n"
            '  "confidence": 0.92\n'
            "}\n"
            "Note: Use null for fields not clearly visible. Use numbers (not currency strings) for charges."
        ),
    },
    "invoice": {
        "line_item": (
            "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
            "{\n"
            '  "record_id": 1,\n'
            '  "primary_identifier": "INV-2024-001",\n'
            '  "fields": {\n'
            '    "invoice_number": "INV-2024-001",\n'
            '    "item_description": "Widget Assembly",\n'
            '    "quantity": 10,\n'
            '    "unit_price": 25.00,\n'
            '    "line_total": 250.00,\n'
            '    "tax_amount": null\n'
            "  },\n"
            '  "confidence": 0.95\n'
            "}\n"
            "Note: Use null for fields not clearly visible. Use numbers for monetary values."
        ),
        "invoice": (
            "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
            "{\n"
            '  "record_id": 1,\n'
            '  "primary_identifier": "INV-2024-001",\n'
            '  "fields": {\n'
            '    "invoice_number": "INV-2024-001",\n'
            '    "vendor_name": "Acme Corp",\n'
            '    "invoice_date": "2024-01-15",\n'
            '    "due_date": "2024-02-15",\n'
            '    "total_amount": 1250.00\n'
            "  },\n"
            '  "confidence": 0.93\n'
            "}\n"
            "Note: Use ISO dates (YYYY-MM-DD). Use null for fields not clearly visible."
        ),
    },
    "employee_roster": {
        "employee": (
            "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
            "{\n"
            '  "record_id": 1,\n'
            '  "primary_identifier": "EMP-1001",\n'
            '  "fields": {\n'
            '    "employee_id": "EMP-1001",\n'
            '    "employee_name": "Smith, John",\n'
            '    "department": "Engineering",\n'
            '    "hire_date": "2020-06-15",\n'
            '    "salary": 85000.00\n'
            "  },\n"
            '  "confidence": 0.94\n'
            "}\n"
            "Note: Use null for fields not clearly visible. Use numbers for salary values."
        ),
    },
    "insurance_claim": {
        "claim": (
            "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
            "{\n"
            '  "record_id": 1,\n'
            '  "primary_identifier": "CLM-2024-5678",\n'
            '  "fields": {\n'
            '    "claim_number": "CLM-2024-5678",\n'
            '    "patient_name": "DOE, JOHN",\n'
            '    "service_date": "2024-01-10",\n'
            '    "billed_amount": 500.00,\n'
            '    "allowed_amount": 350.00,\n'
            '    "patient_responsibility": 75.00\n'
            "  },\n"
            '  "confidence": 0.91\n'
            "}\n"
            "Note: Use null for fields not clearly visible. Use numbers for monetary values."
        ),
    },
}

# Lightweight default for unknown document types
_DEFAULT_EXAMPLE = (
    "EXAMPLE OUTPUT (for format reference only — extract real values from the image):\n"
    "{\n"
    '  "record_id": 1,\n'
    '  "primary_identifier": "identifier_value",\n'
    '  "fields": {\n'
    '    "field_name": "extracted_value_or_null"\n'
    "  },\n"
    '  "confidence": 0.90\n'
    "}\n"
    "Note: Use null for fields not clearly visible. Never guess values."
)


def get_synthetic_example(document_type: str, entity_type: str) -> str:
    """Get a synthetic format example for the given document/entity type.

    Falls back to a generic default if no specific example exists.

    Args:
        document_type: e.g. "medical_superbill", "invoice"
        entity_type: e.g. "patient", "line_item"

    Returns:
        Example snippet string to embed in extraction prompt.
    """
    doc_examples = SYNTHETIC_EXAMPLES.get(document_type.lower(), {})
    return doc_examples.get(entity_type.lower(), _DEFAULT_EXAMPLE)
