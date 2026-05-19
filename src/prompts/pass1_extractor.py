"""
V3 Phase 2 — Pass 1 (EXTRACTOR) prompt builder.

Pass 1 runs on the **primary** VLM (Qwen 3.6 27B-VL by default) with
an EXTRACTOR frame: read the page, emit the schema fields fluently.
Bbox is encouraged but not mandated — the auditor frame in Pass 2
(``pass2_auditor.build_pass2_auditor_prompt``) takes care of strict
bbox grounding.

Why a separate file from ``extraction.py``: the legacy single-VLM
prompt mixes EXTRACTOR-style and AUDITOR-style guidance. In a
heterogeneous-dual-VLM world the *prompts* are the second axis of
heterogeneity (alongside model family). Keeping them in distinct
modules makes the divergence easy to evolve without breaking the
legacy single-VLM pipeline.
"""

from __future__ import annotations

from typing import Any

from src.prompts.extraction import build_extraction_prompt
from src.prompts.grounding_rules import build_grounded_system_prompt


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


PASS1_SYSTEM_PROMPT_HEADER = """You are the **EXTRACTOR** in a heterogeneous dual-VLM pipeline.
Your job is to read the document page and produce a fluent, complete
extraction matching the requested schema.

A second model (the AUDITOR) will independently extract the same page
with stricter bbox-grounding and the two outputs will be reconciled
field-by-field. Your priority is **completeness** and **fluency** —
extract every visible field even if confidence is moderate. The
AUDITOR will catch fabricated values that the reconciler can drop.

Bbox coordinates are optional but helpful — when you can locate a
field unambiguously, include it. Do not invent bboxes for values you
cannot localise.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pass1_system_prompt(
    *,
    additional_context: str | None = None,
    include_chain_of_thought: bool = True,
) -> str:
    """Compose the Pass 1 system prompt.

    Layered on top of the existing ``build_grounded_system_prompt`` so
    the anti-hallucination grounding rules stay consistent with the
    legacy pipeline. The EXTRACTOR header is prepended.
    """
    base = build_grounded_system_prompt(
        additional_context=additional_context,
        include_forbidden=True,
        include_confidence_scale=True,
        include_chain_of_thought=include_chain_of_thought,
    )
    return f"{PASS1_SYSTEM_PROMPT_HEADER}\n\n{base}"


def build_pass1_user_prompt(
    *,
    schema_fields: list[dict[str, Any]],
    document_type: str,
    page_number: int,
    page_count: int,
    modalities: list[str] | None = None,
    profile: str | None = None,
) -> str:
    """Compose the Pass 1 user prompt.

    Reuses ``build_extraction_prompt`` (the legacy single-VLM prompt)
    with ``is_first_pass=True`` so EXTRACTOR-style guidance is
    surfaced. Modality fragments (``fax``, ``handwritten``, ...) and
    the V3 Phase 5 profile fragment (``medical-rcm``, ``finance``, …)
    flow through unchanged. The schema-bound decoder enforces the JSON
    structure regardless of prompt phrasing.
    """
    return build_extraction_prompt(
        schema_fields=schema_fields,
        document_type=document_type,
        page_number=page_number,
        total_pages=page_count,
        is_first_pass=True,
        modalities=modalities or [],
        profile=profile,
    )
