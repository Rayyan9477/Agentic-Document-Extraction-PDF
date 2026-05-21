"""
Generic-document profile.

The default fallback. When auto-detection cannot confidently match any
specialised profile, we land here. The generic profile:

* Adds no prompt fragment (the schema fields drive the prompt).
* Adds no schema overlay (no medical/legal/financial fields invented
  on top of an unknown document).
* Runs no validator pack in blocking mode — only advisory checks
  (date, currency) that are universal.
* Enables no specialised export emitters.

Critically, the generic profile registers a few low-weight *negative-shaped*
signals so it can win against weak medical signals on a clearly generic
doc. We use this rather than introducing a "generic_score" because the
detection logic is simpler when every profile contributes positively.
"""

from __future__ import annotations

from src.profiles.descriptor import ProfileDescriptor, compile_signal
from src.profiles.registry import ProfileRegistry


# Generic-document signals. We give a low base score to any text that
# looks like a typical office document (memo, letter, invoice header,
# meeting notes, …). These signals are intentionally never strong
# enough to override a medical-RCM hit on the same page; they exist so
# generic can be picked over silence on a clearly mundane doc.
GENERIC_SIGNALS = (
    compile_signal(
        name="memo_header",
        pattern=r"\b(memo|memorandum)\b",
        score=0.4,
        description="'Memo' / 'Memorandum' header keyword.",
    ),
    compile_signal(
        name="letter_salutation",
        pattern=r"\b(dear\s+(mr|mrs|ms|dr)\.?)\b",
        score=0.4,
        description="Letter-style salutation.",
    ),
    compile_signal(
        name="invoice_header",
        pattern=r"\b(invoice|purchase\s+order|p\.?o\.?\s+number)\b",
        score=0.5,
        description="Generic invoice/PO header.",
    ),
    compile_signal(
        name="generic_date_label",
        pattern=r"\bdate:\s*\d",
        score=0.2,
        description="'Date:' followed by a number.",
    ),
)


GENERIC_DOCUMENT = ProfileDescriptor(
    name="generic-document",
    display_name="Generic Document",
    description=(
        "Default fallback profile for documents that don't trigger a "
        "specialised profile. No medical/legal/financial fields are "
        "invented; only universal checks (date, currency) run."
    ),
    signals=GENERIC_SIGNALS,
    prompt_fragment="",  # No fragment — schema fields drive the prompt.
    schema_overlay_fields=(),
    # No blocking validators on a generic doc; we only run advisory
    # date/currency checks. Validator pack names map to keys the
    # validator agent recognises; unknown packs are ignored
    # gracefully, so adding a future blocking pack is safe.
    validator_packs={
        "date_format": "advisory",
        "currency_format": "advisory",
    },
    enabled_emitters=(),
    confidence_floor=0.4,  # Low floor — generic *wants* to win on memos.
)


# Register on import.
ProfileRegistry().register(GENERIC_DOCUMENT)
