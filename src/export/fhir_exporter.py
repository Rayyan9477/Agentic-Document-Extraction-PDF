"""
WS-8: FHIR R4 export for medical extraction results.

Builds validated FHIR R4 resources from extracted ``ExtractionState``
records, replacing the inline ``_build_fhir_export`` stub in
``json_exporter.py`` for healthcare-grade interoperability.

Supported source schemas → resource bundles:

    * **CMS-1500** → ``Patient`` + ``Coverage`` + ``Claim``
    * **UB-04**    → ``Patient`` + ``Coverage`` + ``Claim``
                     (institutional ``Claim.type`` = "institutional")
    * **EOB**      → ``Patient`` + ``ExplanationOfBenefit``

The exporter uses the ``fhir.resources`` package (Python data classes
for FHIR R4) when available — it's an **optional** dependency declared
under the ``[fhir]`` extra in ``pyproject.toml``. When the package is
not installed, the exporter falls back to **dict-shaped** FHIR
resources that pass JSON-shape validation but skip the
construct-time-validation that ``fhir.resources`` provides. This keeps
the exporter usable in air-gapped or minimal-install scenarios.

Resources are returned as a single FHIR ``Bundle`` of type
``"collection"``. Callers can serialise the bundle directly to JSON
or pass it to a FHIR-compliant downstream system.

Field-name mapping is **lenient** — extraction schemas vary across
projects, so this module tries multiple aliases for each FHIR field
and silently omits resources whose minimum required fields aren't
present.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency probe
# ---------------------------------------------------------------------------


def _has_fhir_resources() -> bool:
    """Return True iff ``fhir.resources`` is importable.

    Cached behind a module-level flag so repeated exports don't pay
    the import cost on every call.
    """
    global _FHIR_AVAILABLE
    if _FHIR_AVAILABLE is not None:
        return _FHIR_AVAILABLE
    try:
        import fhir.resources  # noqa: F401  pylint: disable=unused-import

        _FHIR_AVAILABLE = True
    except ImportError:
        _FHIR_AVAILABLE = False
        logger.info(
            "fhir_resources_not_installed",
            hint="Install with `pip install -e .[fhir]` for validated FHIR output.",
        )
    return _FHIR_AVAILABLE


_FHIR_AVAILABLE: bool | None = None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FHIRBundle:
    """Resulting FHIR bundle plus metadata about how it was built."""

    bundle: dict[str, Any]
    validated: bool  # True iff fhir.resources validated each resource
    document_type: str
    resource_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_fhir(
    record: dict[str, Any],
    *,
    document_type: str = "",
    processing_id: str | None = None,
) -> FHIRBundle:
    """Build a FHIR R4 ``Bundle`` from an extracted record.

    Args:
        record: Flat ``{field_name: value}`` dict, typically taken from
            ``state["merged_extraction"]`` after value-envelope unwrap.
            Nested dicts whose keys are field names are also accepted.
        document_type: Source schema name (``cms1500`` / ``ub04`` /
            ``eob``). Used to choose which resources to build.
        processing_id: Unique identifier for the extraction run, used
            as the bundle ``id``. Auto-generated if omitted.

    Returns:
        ``FHIRBundle`` containing the JSON-serialisable bundle dict.
    """
    document_type = (document_type or "").lower().strip()
    flat = _flatten_value_envelopes(record)
    bundle_id = processing_id or str(uuid.uuid4())

    resources: list[dict[str, Any]] = []
    if document_type in ("cms1500", "ub04"):
        resources.extend(_build_cms_resources(flat, document_type=document_type))
    elif document_type == "eob":
        resources.extend(_build_eob_resources(flat))
    else:
        # Unknown schema — emit a minimal Patient + DocumentReference so
        # downstream callers still get *something* FHIR-shaped.
        patient = _build_patient(flat)
        if patient is not None:
            resources.append(patient)
        resources.append(_build_document_reference(flat, processing_id=bundle_id))

    bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "entry": [
            {
                "fullUrl": f"urn:uuid:{r.get('id') or uuid.uuid4()}",
                "resource": r,
            }
            for r in resources
        ],
    }

    validated = _validate_with_fhir_resources(bundle) if _has_fhir_resources() else False

    return FHIRBundle(
        bundle=bundle,
        validated=validated,
        document_type=document_type or "unknown",
        resource_count=len(resources),
    )


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def _build_cms_resources(
    flat: dict[str, Any],
    *,
    document_type: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    patient = _build_patient(flat)
    if patient is not None:
        out.append(patient)

    coverage = _build_coverage(flat, patient_ref=_ref_for(patient))
    if coverage is not None:
        out.append(coverage)

    claim = _build_claim(
        flat,
        patient_ref=_ref_for(patient),
        coverage_ref=_ref_for(coverage),
        claim_type="institutional" if document_type == "ub04" else "professional",
    )
    if claim is not None:
        out.append(claim)
    return out


def _build_eob_resources(flat: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    patient = _build_patient(flat)
    if patient is not None:
        out.append(patient)

    eob: dict[str, Any] = {
        "resourceType": "ExplanationOfBenefit",
        "id": str(uuid.uuid4()),
        "status": "active",
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                    "code": "professional",
                }
            ]
        },
        "use": "claim",
        "patient": _ref_for(patient) or {"display": "unknown"},
        "created": _coerce_date(_first(flat, ("statement_date", "service_date", "claim_date"))),
        "outcome": "complete",
    }
    paid = _coerce_money(_first(flat, ("amount_paid", "total_paid", "paid_amount")))
    if paid is not None:
        eob["payment"] = {"amount": paid}
    total = _coerce_money(_first(flat, ("total_charges", "billed_amount", "total")))
    if total is not None:
        eob["total"] = [
            {
                "category": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/adjudication",
                            "code": "submitted",
                        }
                    ]
                },
                "amount": total,
            }
        ]

    out.append(eob)
    return out


def _build_patient(flat: dict[str, Any]) -> dict[str, Any] | None:
    given = _first(flat, ("patient_first_name", "patient_given_name", "first_name"))
    family = _first(flat, ("patient_last_name", "patient_family_name", "last_name"))
    full = _first(flat, ("patient_name", "subscriber_name", "member_name"))

    name: dict[str, Any] = {}
    if family:
        name["family"] = family
    if given:
        name["given"] = [given]
    if full and not name:
        # Best-effort split: "Last, First" or "First Last".
        if "," in full:
            family_part, _, given_part = full.partition(",")
            name = {"family": family_part.strip(), "given": [given_part.strip()]}
        else:
            parts = full.split()
            if len(parts) >= 2:
                name = {"family": parts[-1], "given": parts[:-1]}
            else:
                name = {"family": full}

    if not name:
        return None

    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": str(uuid.uuid4()),
        "name": [name],
    }
    dob = _coerce_date(_first(flat, ("patient_dob", "date_of_birth", "dob")))
    if dob:
        patient["birthDate"] = dob
    gender = _first(flat, ("patient_gender", "gender", "sex"))
    if gender:
        patient["gender"] = _coerce_gender(gender)

    return patient


def _build_coverage(
    flat: dict[str, Any],
    *,
    patient_ref: dict[str, Any] | None,
) -> dict[str, Any] | None:
    member_id = _first(flat, ("member_id", "policy_number", "subscriber_id", "insurance_id"))
    if not member_id:
        return None
    coverage: dict[str, Any] = {
        "resourceType": "Coverage",
        "id": str(uuid.uuid4()),
        "status": "active",
        "subscriberId": str(member_id),
        "beneficiary": patient_ref or {"display": "unknown"},
    }
    payor = _first(flat, ("insurance_company", "payer_name", "insurer"))
    if payor:
        coverage["payor"] = [{"display": str(payor)}]
    return coverage


def _build_claim(
    flat: dict[str, Any],
    *,
    patient_ref: dict[str, Any] | None,
    coverage_ref: dict[str, Any] | None,
    claim_type: str,
) -> dict[str, Any] | None:
    claim_number = _first(flat, ("claim_number", "claim_id", "patient_account_number"))
    if not claim_number:
        return None
    claim: dict[str, Any] = {
        "resourceType": "Claim",
        "id": str(uuid.uuid4()),
        "status": "active",
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                    "code": claim_type,
                }
            ]
        },
        "use": "claim",
        "patient": patient_ref or {"display": "unknown"},
        "created": _coerce_date(_first(flat, ("claim_date", "service_date", "statement_date"))),
        "identifier": [{"value": str(claim_number)}],
    }
    if coverage_ref:
        claim["insurance"] = [
            {
                "sequence": 1,
                "focal": True,
                "coverage": coverage_ref,
            }
        ]
    total = _coerce_money(_first(flat, ("total_charges", "billed_amount", "total")))
    if total is not None:
        claim["total"] = total
    return claim


def _build_document_reference(
    flat: dict[str, Any],
    *,
    processing_id: str,
) -> dict[str, Any]:
    """Fallback wrapper resource for unknown schemas."""
    return {
        "resourceType": "DocumentReference",
        "id": str(uuid.uuid4()),
        "status": "current",
        "subject": {"display": "extracted document"},
        "content": [
            {
                "attachment": {
                    "contentType": "application/json",
                    "title": f"Extraction {processing_id}",
                }
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_value_envelopes(record: dict[str, Any]) -> dict[str, Any]:
    """Strip ``{value, confidence, human_corrected, ...}`` envelopes."""
    flat: dict[str, Any] = {}
    for key, value in (record or {}).items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value.get("value")
        else:
            flat[key] = value
    return flat


def _first(flat: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in flat and flat[key] not in (None, ""):
            return flat[key]
    return None


def _ref_for(resource: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a FHIR Reference dict pointing at the given resource."""
    if resource is None:
        return None
    return {"reference": f"{resource['resourceType']}/{resource['id']}"}


def _coerce_date(value: Any) -> str | None:
    """Best-effort ISO-8601 date coercion. Returns None on failure."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    # Already ISO-ish?
    if len(text) >= 8 and text[4:5] == "-":
        return text[:10]
    # MM/DD/YYYY
    if len(text) == 10 and text[2] == "/" and text[5] == "/":
        mm, dd, yyyy = text.split("/")
        try:
            return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        except ValueError:
            return None
    # MM-DD-YYYY
    if len(text) == 10 and text[2] == "-" and text[5] == "-":
        mm, dd, yyyy = text.split("-")
        try:
            return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        except ValueError:
            return None
    return None


def _coerce_money(value: Any) -> dict[str, Any] | None:
    """Coerce a numeric / currency-string value into a FHIR Money dict."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return {"value": float(value), "currency": "USD"}
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return {"value": float(text), "currency": "USD"}
    except ValueError:
        return None


def _coerce_gender(value: Any) -> str:
    """Map a gender string to FHIR's administrative-gender code set."""
    text = str(value).strip().lower()
    if text in ("m", "male"):
        return "male"
    if text in ("f", "female"):
        return "female"
    if text in ("o", "other"):
        return "other"
    return "unknown"


def _validate_with_fhir_resources(bundle: dict[str, Any]) -> bool:
    """Run each entry through ``fhir.resources`` for shape validation.

    Returns True iff every resource validates. Any validation failure
    is logged but does not raise — callers still get the bundle in
    its unvalidated form. This is the cleanest middle ground for an
    optional dependency: when present, validation gives confidence;
    when absent, the dict-form bundle is still returned.
    """
    try:
        from fhir.resources.bundle import Bundle  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - guarded by _has_fhir_resources
        return False
    try:
        Bundle.model_validate(bundle)
        return True
    except Exception as exc:  # pragma: no cover - integration path
        logger.warning("fhir_validation_failed", error=str(exc))
        return False


__all__ = [
    "FHIRBundle",
    "export_fhir",
]
