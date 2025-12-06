"""
Prompts module for document extraction.

Provides structured prompts with anti-hallucination grounding rules,
document classification templates, and extraction prompts.
"""

from src.prompts.grounding_rules import (
    GROUNDING_RULES,
    FORBIDDEN_ACTIONS,
    build_grounded_system_prompt,
    build_confidence_instruction,
)
from src.prompts.classification import (
    build_classification_prompt,
    build_structure_analysis_prompt,
    DOCUMENT_TYPE_DESCRIPTIONS,
)
from src.prompts.extraction import (
    build_extraction_prompt,
    build_verification_prompt,
    build_field_prompt,
    build_table_extraction_prompt,
)
from src.prompts.validation import (
    build_validation_prompt,
    build_hallucination_check_prompt,
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
