"""
V3 Phase 2 — Pass 2 (AUDITOR) prompt builder.

Pass 2 runs on the **secondary** VLM (Gemma 4 31B-VL by default) with
an AUDITOR frame: every emitted value must cite the smallest visual
region (bbox) that justifies it. This forces bbox-grounding to happen
**at decode time** so the reconciler's bbox-overlap tiebreaker
(``HeterogeneousReconciler`` step 2) has reliable input.

Why a separate file from ``pass1_extractor.py``: the heterogeneity in
this pipeline lives in two places — model family AND prompt frame.
Reusing the same prompt across both passes would lose half the
disagreement signal the reconciler depends on.

The schema bound at decode time (``Pass2AuditorEnvelope``) requires a
``bbox`` field on every emitted leaf; the prompt explicitly tells the
model that bbox is mandatory. Together they make bbox-less responses
structurally impossible.
"""

from __future__ import annotations

from typing import Any

from src.prompts.grounding_rules import build_grounded_system_prompt


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


PASS2_SYSTEM_PROMPT_HEADER = """You are the **AUDITOR** in a heterogeneous dual-VLM pipeline.
A first model (the EXTRACTOR) has already extracted this page; you
work independently and your output will be compared field-by-field
to theirs.

Your priorities, in order:

1. **Bbox-ground every value you emit.** For each schema field, return
   the smallest rectangle on the page that visually contains the
   value. Use normalised coordinates ``[x1, y1, x2, y2]`` in [0, 1].
   If you cannot localise a value, return ``null`` for both the value
   AND the bbox — never guess a bbox.

2. **Be skeptical.** If a value is plausible but you cannot read it
   from the image with confidence, mark it ``null``. Lower confidence
   over guessing.

3. **Read characters precisely.** Distinguish 0 vs O, 1 vs l, 5 vs S
   carefully — the EXTRACTOR may have made the opposite trade-off.
   Disagreements between you and the EXTRACTOR are the entire point;
   downstream reconciliation depends on the disagreement signal being
   meaningful.
"""


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


PASS2_USER_PROMPT_TEMPLATE = """## AUDITOR EXTRACTION — Page {page_number} of {total_pages}

Document type: {document_type}
{modality_block}{profile_block}

### YOUR TASK

For each field listed below, locate the value on this page image and
emit a record with these keys:

- ``value`` — the exact value you can read, or ``null`` if you cannot
  localise it on this page.
- ``confidence`` — your confidence in [0.0, 1.0]. **Do not anchor on
  the EXTRACTOR's confidence — they did not see this prompt.**
- ``bbox`` — normalised ``[x1, y1, x2, y2]`` in [0, 1] for the
  smallest rectangle visually containing the value. ``null`` only when
  ``value`` is also ``null``.
- ``location`` — short human-readable hint (e.g. "Box 24a", "right
  column header").

### FIELDS TO LOCATE

{field_block}

### RULES

- Every emitted value MUST carry a bbox. If you cannot bbox it, emit
  ``null`` for both ``value`` and ``bbox`` — do not guess a region.
- Read characters carefully; do not rely on what looks plausible for
  the field type.
- For fields that legitimately span multiple regions on the page,
  return the bbox that contains the **canonical** instance (largest
  printed copy, most prominent location).
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pass2_system_prompt(
    *,
    additional_context: str | None = None,
    include_chain_of_thought: bool = True,
) -> str:
    """Compose the Pass 2 system prompt.

    Layered on top of ``build_grounded_system_prompt`` so the
    anti-hallucination grounding rules stay consistent. The AUDITOR
    header is prepended.
    """
    base = build_grounded_system_prompt(
        additional_context=additional_context,
        include_forbidden=True,
        include_confidence_scale=True,
        include_chain_of_thought=include_chain_of_thought,
    )
    return f"{PASS2_SYSTEM_PROMPT_HEADER}\n\n{base}"


def build_pass2_user_prompt(
    *,
    schema_fields: list[dict[str, Any]],
    document_type: str,
    page_number: int,
    page_count: int,
    modalities: list[str] | None = None,
    profile: str | None = None,
) -> str:
    """Compose the Pass 2 user prompt.

    The user prompt deliberately does NOT reuse the legacy
    ``build_extraction_prompt`` because it would re-introduce
    EXTRACTOR-style guidance. Pass 2 wants distinct framing.

    Args:
        schema_fields: list of ``FieldDefinition``-shaped dicts.
        document_type: e.g. ``"CMS-1500"``.
        page_number, page_count: 1-based page indexing.
        modalities: detected modality labels (``fax``, ``handwritten``).

    Returns:
        The user-message string for the AUDITOR call.
    """
    field_lines: list[str] = []
    for field in schema_fields:
        name = field.get("field_name") or field.get("name") or "<unnamed>"
        description = field.get("description", "")
        examples = field.get("examples") or []
        line = f"- **{name}**"
        if description:
            line += f" — {description}"
        if examples:
            line += f" (examples: {', '.join(str(e) for e in examples[:3])})"
        field_lines.append(line)
    field_block = "\n".join(field_lines) if field_lines else "(no fields requested)"

    modality_block = ""
    if modalities:
        modality_block = f"\nDetected modalities: {', '.join(sorted(modalities))}"

    # V3 Phase 5: profile-specific reminder block. Loaded from the
    # profile descriptor's ``prompt_fragment``; trimmed for the
    # AUDITOR's tighter token budget.
    profile_block = ""
    if profile:
        try:
            from src.profiles import get_profile

            descriptor = get_profile(profile)
            fragment = (descriptor.prompt_fragment or "").strip()
            if fragment:
                profile_block = "\n" + fragment + "\n"
        except Exception:
            profile_block = ""

    return PASS2_USER_PROMPT_TEMPLATE.format(
        page_number=page_number,
        total_pages=page_count,
        document_type=document_type,
        modality_block=modality_block,
        profile_block=profile_block,
        field_block=field_block,
    )
