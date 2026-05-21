"""
Finance profile.

Activated for tax forms (W-2, 1099 family), bank statements, and
invoices. Mostly mirrors the medical-rcm profile shape but with
finance-specific signals and a much narrower validator pack list (we
don't have the same depth of finance-side standards as we do for
healthcare). Phase 5 ships this as a *thin* profile — enough to win
detection on a clearly financial doc and inject a small reminder
fragment, with the heavy validator/emission build-out deferred to a
future phase.
"""

from __future__ import annotations

from src.profiles.descriptor import ProfileDescriptor, compile_signal
from src.profiles.registry import ProfileRegistry


FINANCE_SIGNALS = (
    compile_signal(
        name="w2_header",
        pattern=r"\bwage\s+and\s+tax\s+statement\b|\bform\s+w[\s\-]?2\b",
        score=0.8,
        description="W-2 Wage and Tax Statement header.",
    ),
    compile_signal(
        name="form_1099_header",
        pattern=r"\bform\s+1099[\s\-]?(misc|nec|int|div|r|b|g|k|s)?\b",
        score=0.8,
        description="1099 family header (MISC/NEC/INT/DIV/R/B/G/K/S).",
    ),
    compile_signal(
        name="bank_statement_header",
        pattern=r"\b(account|bank|monthly)\s+statement\b",
        score=0.7,
        description="Bank/account/monthly statement header.",
    ),
    compile_signal(
        name="invoice_strong",
        pattern=r"\binvoice\s+(number|#|no\.?)\b",
        score=0.6,
        description="'Invoice number/#/no.' label.",
    ),
    compile_signal(
        name="ein_pattern",
        pattern=r"\bEIN\b|employer\s+identification\s+number",
        score=0.3,
        description="EIN label or expansion present.",
    ),
    compile_signal(
        name="routing_account_pair",
        pattern=r"routing\s+number.*account\s+number|account\s+number.*routing\s+number",
        score=0.3,
        description="Routing-number + account-number labels both present.",
    ),
    compile_signal(
        name="tax_year_label",
        pattern=r"\btax\s+year\s+\d{4}\b",
        score=0.2,
        description="'Tax year YYYY' label.",
    ),
)


FINANCE_PROMPT_FRAGMENT = """\
### FINANCE PROFILE NOTES

This is a financial document (tax form / statement / invoice). When
extracting:

1. **Identifiers.**
   - SSN is `NNN-NN-NNNN` (9 digits with hyphens). Mask if the schema
     specifies redaction.
   - EIN is `NN-NNNNNNN` (9 digits with one hyphen).
   - Bank routing numbers are 9 digits; account numbers vary.

2. **Money.**
   - Amounts may carry currency symbols (`$`), parentheses for
     negatives (`(123.45)`), or trailing minus signs (`123.45-`).
     Preserve sign in the extracted value; do not strip parentheses
     and lose the negative.
   - 1099-MISC and 1099-NEC have similar but distinct box layouts —
     do not transpose box numbers.

3. **Dates.**
   - Tax forms commonly use the calendar tax year (`YYYY`) only, no
     month/day. Extract the year as printed.

When in doubt, return `null` rather than guess.
"""


FINANCE = ProfileDescriptor(
    name="finance",
    display_name="Finance",
    description=(
        "Financial documents: W-2 wage statements, 1099 forms, "
        "bank/account statements, invoices. Adds finance-specific "
        "prompt notes and routes through the universal validator "
        "packs (currency/date) plus an SSN format check."
    ),
    signals=FINANCE_SIGNALS,
    prompt_fragment=FINANCE_PROMPT_FRAGMENT,
    schema_overlay_fields=(),  # No overlay — finance schemas are
                                # self-contained.
    validator_packs={
        "currency_format": "blocking",
        "date_format": "blocking",
        "ssn_format": "advisory",
    },
    enabled_emitters=(),
    confidence_floor=0.6,
)


ProfileRegistry().register(FINANCE)
