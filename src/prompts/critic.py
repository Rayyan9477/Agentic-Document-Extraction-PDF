"""
V3 Phase 3 — Critic prompt builder.

The Critic runs as an independent verifier after the legacy validator
(or after the dual-VLM reconciler in dual_vlm mode). It does NOT
re-extract — its task is to *audit*: given the page image and the
already-merged extraction, decide whether each emitted value is
actually visible in the image, plausible-but-not-shown, or
contradicted by what's on the page.

The frame is deliberately different from the EXTRACTOR / AUDITOR
prompts so the Critic occupies a different latent space than the
extraction passes did. A third extractor would tend to agree with
whichever pass it most resembles statistically; a verifier produces
a different output type (CriticReport) and is forced into different
reasoning.

Family rotation (the Critic should run on the model family that's NOT
the consensus of Pass 1 / Pass 2) lives in the orchestrator's role
resolution, not here. The prompt is identical regardless of which
backend serves it.
"""

from __future__ import annotations

import json
from typing import Any

from src.prompts.grounding_rules import build_grounded_system_prompt


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


CRITIC_SYSTEM_PROMPT_HEADER = """You are the **CRITIC** in a heterogeneous extraction pipeline.
A previous extraction has already produced field values for this page;
your job is to **audit** that extraction, not redo it.

For each field, decide whether the value is:

* **supported** — visible in the image at the location described.
* **not_visible** — plausible for the field type but you cannot see it
  on the page. The extractor may have inferred or hallucinated it.
* **contradicts_image** — visible on the page but with a *different*
  value than the extractor recorded. The extractor mis-read it.
* **ambiguous** — partially visible, smudged, or the location is
  unclear; you cannot decide either way.

Your priorities:

1. **Be skeptical.** Default to ``not_visible`` when in doubt. The
   extractor has every incentive to fill fields; you have every
   incentive to flag uncertainty. Wrong extracted values cost
   downstream consumers more than missing extracted values.
2. **Do NOT propose new values.** Even when you flag
   ``contradicts_image``, your job is auditing — name the field, mark
   the issue, never emit a corrected value. The orchestrator decides
   whether to retry or escalate.
3. **Severity calibration.** ``error`` for fields that are clearly
   wrong (contradicts_image, or a billing-impact field that's not
   visible). ``warning`` for not_visible on non-billing fields.
   ``info`` for ambiguous.
"""


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


CRITIC_USER_PROMPT_TEMPLATE = """## CRITIC AUDIT — Page {page_number} of {total_pages}

Document type: {document_type}
{modality_block}

### EXTRACTED VALUES TO AUDIT

The following values were emitted by the extraction pipeline. Verify
each one against the page image.

```json
{extraction_block}
```

### YOUR TASK

For each non-null field above, emit one ``CriticConcern`` describing
what you observe. **Skip fields you can confirm as supported** — only
emit concerns. The trust_score reflects your overall confidence that
the extraction is correct (1.0 = everything verified, 0.0 = nothing
matches the image).

After auditing, choose one ``recommendation``:

* ``accept`` — every emitted value appears visually supported, no
  concerns of severity ``warning`` or higher.
* ``verify_bbox`` — at least one field is ambiguous OR not_visible;
  the orchestrator should crop the bbox and re-read.
* ``retry`` — at least one field is contradicts_image with severity
  ``error``; the extraction should re-run with your concerns embedded
  as negative exemplars.
* ``human_review`` — multiple billing-impact fields fail audit; the
  orchestrator should escalate to a human reviewer.

### RULES

- Do not emit corrected values. Do not propose alternates. Audit only.
- If every field is null in the input, recommend ``human_review`` with
  trust_score 0.0 (the extractor produced nothing actionable).
- ``trust_score`` is a single float in [0, 1] for the *overall*
  extraction, not per-field.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_critic_system_prompt(
    *,
    additional_context: str | None = None,
) -> str:
    """Compose the Critic system prompt.

    Layered on top of ``build_grounded_system_prompt`` so the
    anti-hallucination grounding rules stay consistent. The CRITIC
    header is prepended.

    The Critic does NOT use chain-of-thought because the response
    schema is short (a single ``CriticReport``) and CoT would slow
    the audit without changing the output meaningfully.
    """
    base = build_grounded_system_prompt(
        additional_context=additional_context,
        include_forbidden=False,  # critic doesn't extract; ``forbidden`` text targets extractors
        include_confidence_scale=True,
        include_chain_of_thought=False,
    )
    return f"{CRITIC_SYSTEM_PROMPT_HEADER}\n\n{base}"


def build_critic_user_prompt(
    *,
    extraction: dict[str, Any],
    document_type: str,
    page_number: int,
    page_count: int,
    modalities: list[str] | None = None,
) -> str:
    """Compose the Critic user prompt.

    Args:
        extraction: the merged extraction (post-reconciliation in
            dual_vlm mode, post-extraction in legacy mode). Serialised
            into the prompt as JSON. Pre-truncated by the caller if
            the extraction is unusually large.
        document_type: e.g. ``"CMS-1500"``.
        page_number, page_count: 1-based.
        modalities: detected modality labels.
    """
    modality_block = ""
    if modalities:
        modality_block = f"\nDetected modalities: {', '.join(sorted(modalities))}"

    # Compact JSON keeps token count down. The Critic doesn't need
    # pretty-printing; the schema-bound decoder will produce structured
    # output regardless of input formatting.
    extraction_block = json.dumps(extraction, indent=2, default=str)

    return CRITIC_USER_PROMPT_TEMPLATE.format(
        page_number=page_number,
        total_pages=page_count,
        document_type=document_type,
        modality_block=modality_block,
        extraction_block=extraction_block,
    )
