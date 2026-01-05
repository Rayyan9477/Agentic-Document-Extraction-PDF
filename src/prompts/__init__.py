"""
Prompts module for document extraction.

Provides structured prompts with anti-hallucination grounding rules,
document classification templates, and extraction prompts.
"""

from src.prompts.classification import (
    DOCUMENT_TYPE_DESCRIPTIONS,
    build_classification_prompt,
    build_structure_analysis_prompt,
)
from src.prompts.extraction import (
    build_extraction_prompt,
    build_field_prompt,
    build_table_extraction_prompt,
    build_verification_prompt,
)
from src.prompts.grounding_rules import (
    FORBIDDEN_ACTIONS,
    GROUNDING_RULES,
    build_confidence_instruction,
    build_grounded_system_prompt,
)
from src.prompts.validation import (
    build_hallucination_check_prompt,
    build_validation_prompt,
)


__all__ = [
    # Grounding
    "GROUNDING_RULES",
    "FORBIDDEN_ACTIONS",
    "build_grounded_system_prompt",
    "build_confidence_instruction",
    # Classification
    "build_classification_prompt",
    "build_structure_analysis_prompt",
    "DOCUMENT_TYPE_DESCRIPTIONS",
    # Extraction
    "build_extraction_prompt",
    "build_verification_prompt",
    "build_field_prompt",
    "build_table_extraction_prompt",
    # Validation
    "build_validation_prompt",
    "build_hallucination_check_prompt",
]
