"""
Medical-RCM profile.

Activated for revenue-cycle-management documents: CMS-1500, UB-04,
EOB, superbill, and the rest of the claim/encounter/remittance corpus.

When this profile is active:

* Prompt gains an RCM-specific reminder block (CPT/ICD/modifier
  conventions, NPI Luhn, place-of-service codes) so the VLM knows the
  *kind* of fields it's reading even when the schema doesn't carry
  every detail.
* The schema overlay re-introduces the healthcare fields that we
  strip from the generic fallback (patient_name, provider_name,
  diagnosis_codes, procedure_codes, service_date) — so a medical doc
  that doesn't match a registered ``DocumentSchema`` (a niche payer
  superbill, say) still extracts the universal medical fields.
* Validator packs run in blocking mode: NPI Luhn, CPT format, ICD-10
  format, POS code lookup, modifier compatibility.
* Export emitters become available: ``ccda``, ``x12_275`` (Phase 7).

Detection signals are tuned so a single strong header match
("HEALTH INSURANCE CLAIM FORM") clears the floor on its own; weaker
signals (NPI/CPT/ICD pattern matches) require corroboration.
"""

from __future__ import annotations

from src.profiles.descriptor import ProfileDescriptor, compile_signal
from src.profiles.registry import ProfileRegistry


# Detection signals.
#
# Score budget calibration:
#   confidence_floor = 0.6
#   - Strong header match alone: 0.8 → fires.
#   - One strong header + any weak signal: clears comfortably.
#   - Three weak signals (NPI + CPT + ICD patterns) without header: 0.6 → fires.
#   - Two weak signals: 0.4 → does NOT fire (would be a false positive
#     on a generic medical-keyword-bearing memo).
MEDICAL_RCM_SIGNALS = (
    compile_signal(
        name="hcfa_header",
        pattern=r"health\s+insurance\s+claim\s+form",
        score=0.8,
        description="CMS-1500 / HCFA-1500 standard form header.",
    ),
    compile_signal(
        name="ub04_header",
        pattern=r"\buniform\s+billing|ub[\s\-]?04|cms[\s\-]?1450\b",
        score=0.8,
        description="UB-04 / CMS-1450 hospital billing form header.",
    ),
    compile_signal(
        name="eob_header",
        pattern=r"explanation\s+of\s+benefits|remittance\s+advice",
        score=0.8,
        description="EOB / remittance-advice header.",
    ),
    compile_signal(
        name="superbill_header",
        pattern=r"\bsuperbill\b|\bencounter\s+form\b",
        score=0.7,
        description="Superbill / encounter-form header.",
    ),
    # Document-type implicit signals. These fire when the analyzer's
    # classification has already settled the question.
    compile_signal(
        name="document_type_cms1500",
        pattern=r"\bCMS[\s\-]?1500\b",
        score=0.6,
        description="Document type already classified as CMS-1500.",
    ),
    compile_signal(
        name="document_type_ub04",
        pattern=r"\bUB[\s\-]?04\b",
        score=0.6,
        description="Document type already classified as UB-04.",
    ),
    compile_signal(
        name="document_type_eob",
        pattern=r"\bEOB\b",
        score=0.6,
        description="Document type already classified as EOB.",
    ),
    # Field-pattern signals (weaker — must corroborate).
    compile_signal(
        name="npi_pattern",
        pattern=r"\bNPI\b|national\s+provider\s+identifier",
        score=0.2,
        description="NPI label or expansion present.",
    ),
    compile_signal(
        name="cpt_pattern",
        pattern=r"\bCPT\b|\bHCPCS\b|procedure\s+code",
        score=0.2,
        description="CPT/HCPCS/procedure-code label present.",
    ),
    compile_signal(
        name="icd10_pattern",
        pattern=r"\bICD[\s\-]?(9|10)\b|diagnosis\s+code",
        score=0.2,
        description="ICD-9/10 or diagnosis-code label present.",
    ),
    compile_signal(
        name="pos_pattern",
        pattern=r"place\s+of\s+service|\bPOS\b",
        score=0.2,
        description="Place-of-service label present.",
    ),
    compile_signal(
        name="modifier_pattern",
        pattern=r"\bmodifier\b\s*\d{2}",
        score=0.2,
        description="CPT modifier reference present.",
    ),
    compile_signal(
        name="patient_provider_pair",
        pattern=r"patient\s+name.*provider|provider.*patient\s+name",
        score=0.2,
        description="Patient/provider field labels both present.",
    ),
)


# Prompt fragment — injected into the extraction prompt when this
# profile is active. Markdown-formatted, self-contained, sized to the
# token budget we leave for the profile section (≤ 800 tokens).
MEDICAL_RCM_PROMPT_FRAGMENT = """\
### MEDICAL / RCM PROFILE NOTES

This is a healthcare revenue-cycle-management document. When extracting:

1. **Codes are exact.**
   - CPT codes are exactly 5 digits (e.g. `99213`).
   - HCPCS Level II codes are 1 letter + 4 digits (e.g. `J3490`).
   - ICD-10-CM codes are 1 letter + 2 digits + optional `.` + up to 4
     more chars (e.g. `E11.65`, `Z79.4`). Never invent decimals.
   - NPI is exactly 10 digits and must satisfy Luhn check digit.
   - Place-of-service (POS) codes are 2 digits (e.g. `11` = office,
     `21` = inpatient).

2. **Modifiers attach to procedures.**
   - CPT modifiers are 2 chars after a CPT code, often hyphenated
     (e.g. `99213-25`). Common: `25`, `50`, `59`, `LT`, `RT`, `TC`,
     `26`. If you see a modifier alone with no procedure, that is a
     formatting error — extract both together.

3. **Money is line-item-then-total.**
   - When a claim shows charges per line and a total, the total
     should equal the sum of line charges. Mismatches are common
     (rounding, discounts) — extract both and let the validator
     reconcile.
   - "Allowed", "paid", "patient responsibility", "adjustment", and
     "total billed" are all distinct values. Do not collapse them.

4. **Diagnosis pointers.**
   - On CMS-1500 line 24E, diagnosis pointers are letters (`A`, `B`,
     `C`, `D` …) referring back to the codes in section 21.
   - On UB-04 the link is positional (FL 67 / 67A / 67B / …).

5. **Dates.**
   - Service dates are typically MM/DD/YYYY in the US. Patient DOB
     can use either MM/DD/YYYY or YYYY-MM-DD; preserve what's
     printed and let the normaliser handle conversion.

When in doubt, return `null` rather than guess a code.
"""


MEDICAL_RCM = ProfileDescriptor(
    name="medical-rcm",
    display_name="Medical / Revenue Cycle",
    description=(
        "Healthcare revenue-cycle-management documents: claim forms "
        "(CMS-1500, UB-04), explanation-of-benefits, superbills, "
        "and the rest of the encounter/billing/remittance corpus. "
        "Activates RCM-specific prompt reminders, healthcare schema "
        "overlay, blocking validator packs (NPI Luhn, CPT/ICD/POS "
        "format), and the C-CDA / X12N 275 export emitters."
    ),
    signals=MEDICAL_RCM_SIGNALS,
    prompt_fragment=MEDICAL_RCM_PROMPT_FRAGMENT,
    # Names of FieldDefinition blocks to overlay onto the resolved
    # schema. The actual definitions live in
    # ``src.schemas.profile_overlays``.
    schema_overlay_fields=("healthcare_core",),
    validator_packs={
        "npi_luhn": "blocking",
        "cpt_format": "blocking",
        "icd10_format": "blocking",
        "pos_code": "blocking",
        "modifier_compat": "advisory",  # Promoted to blocking once the
                                        # modifier table is fully populated.
        "currency_format": "advisory",
        "date_format": "advisory",
    },
    enabled_emitters=("ccda", "x12_275"),
    confidence_floor=0.6,
)


ProfileRegistry().register(MEDICAL_RCM)
