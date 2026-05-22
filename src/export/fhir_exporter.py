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
    provenance_map: dict[str, dict[str, Any]] | None = None,
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
        provenance_map: V3 Phase 4 — optional ``{field_name: Provenance.to_serialisable()}``
            map. When provided, the bundle gains a ``meta.extension``
            entry stamping the provenance namespace + per-field detail.
            Bundle-level placement keeps the per-resource builders
            unchanged; field-level extension within each resource is a
            follow-up task tracked in ``docs/MVP/PROVENANCE.md``.

    Returns:
        ``FHIRBundle`` containing the JSON-serialisable bundle dict.
    """
    document_type = (document_type or "").lower().strip()
    flat = _flatten_value_envelopes(record)
    bundle_id = processing_id or str(uuid.uuid4())

    # Phase K — normalise the document-type label. The analyzer's adaptive
    # schema reports human-readable names like ``health_insurance_claim_form``
    # or ``hcfa_1500``; the dispatch below expects canonical short keys.
    # This alias table lets the exporter route correctly without forcing
    # the analyzer to use exporter-internal names.
    DOCTYPE_ALIASES = {
        "health_insurance_claim_form": "cms1500",
        "hcfa_1500": "cms1500",
        "hcfa-1500": "cms1500",
        "cms-1500": "cms1500",
        "uniform_billing_04": "ub04",
        "uniform_bill": "ub04",
        "ub-04": "ub04",
        "cms1450": "ub04",
        "cms_1450": "ub04",
        "explanation_of_benefits": "eob",
        "remittance_advice": "eob",
    }
    canonical_type = DOCTYPE_ALIASES.get(document_type, document_type)

    resources: list[dict[str, Any]] = []
    if canonical_type in ("cms1500", "ub04"):
        resources.extend(_build_cms_resources(flat, document_type=canonical_type))
    elif canonical_type == "eob":
        resources.extend(_build_eob_resources(flat))
    else:
        # Unknown schema — emit a minimal Patient + DocumentReference so
        # downstream callers still get *something* FHIR-shaped.
        patient = _build_patient(flat)
        if patient is not None:
            resources.append(patient)
        resources.append(_build_document_reference(flat, processing_id=bundle_id))

    bundle: dict[str, Any] = {
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

    # V3 Phase 4 — attach provenance at bundle level.
    if provenance_map:
        meta_block = _build_provenance_meta(provenance_map)
        if meta_block:
            bundle["meta"] = meta_block

    validated = _validate_with_fhir_resources(bundle) if _has_fhir_resources() else False

    return FHIRBundle(
        bundle=bundle,
        validated=validated,
        document_type=document_type or "unknown",
        resource_count=len(resources),
    )


def _build_provenance_meta(
    provenance_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """V3 Phase 4 — convert per-field provenance into a FHIR ``meta`` block.

    The extension URL is ``urn:veridoc:provenance:1.0`` (configurable
    via ``settings.provenance.fhir_extension_url``). Each field becomes
    one nested ``extension`` entry with sub-extensions for page, bbox,
    extraction_path, agent_signatures, confidence, vlm_model_id.

    Returns ``None`` when ``provenance_map`` is empty so the bundle's
    ``meta`` block stays absent rather than having a no-op extension.
    """
    if not provenance_map:
        return None

    try:
        from src.config import get_settings

        url = get_settings().provenance.fhir_extension_url
    except Exception:
        url = "urn:veridoc:provenance:1.0"

    field_extensions: list[dict[str, Any]] = []
    for field_name, prov in provenance_map.items():
        if not isinstance(prov, dict):
            continue
        sub_exts: list[dict[str, Any]] = [
            {"url": "fieldName", "valueString": field_name},
            {"url": "page", "valueInteger": int(prov.get("page", 0))},
            {
                "url": "extractionPath",
                "valueString": ",".join(prov.get("extraction_path") or []),
            },
            {
                "url": "agentSignatures",
                "valueString": ",".join(prov.get("agent_signatures") or []),
            },
            {
                "url": "confidence",
                "valueDecimal": float(prov.get("confidence", 0.0)),
            },
            {
                "url": "vlmModelId",
                "valueString": prov.get("vlm_model_id", "") or "",
            },
        ]
        bbox = prov.get("bbox")
        if isinstance(bbox, dict):
            x = bbox.get("x")
            y = bbox.get("y")
            w = bbox.get("width", bbox.get("w"))
            h = bbox.get("height", bbox.get("h"))
            if x is not None and y is not None and w is not None and h is not None:
                sub_exts.append(
                    {
                        "url": "bbox",
                        "valueString": f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}",
                    }
                )
        if prov.get("source_block_id"):
            sub_exts.append(
                {"url": "sourceBlockId", "valueString": prov["source_block_id"]}
            )
        mem0 = prov.get("mem0_match")
        if mem0:
            sub_exts.append({"url": "mem0Match", "valueString": mem0})

        field_extensions.append(
            {
                "url": url,
                "extension": sub_exts,
            }
        )

    if not field_extensions:
        return None
    return {"extension": field_extensions}


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
    dob = _coerce_date(
        _first(flat, ("patient_dob", "patient_birth_date", "date_of_birth", "dob"))
    )
    if dob:
        patient["birthDate"] = dob
    gender = _first(flat, ("patient_gender", "gender", "sex", "patient_sex"))
    if gender:
        patient["gender"] = _coerce_gender(gender)
    # Phase K — telecom + address from Gemma's adaptive schema names.
    phone = _first(flat, ("patient_phone", "phone", "telephone"))
    if phone:
        patient["telecom"] = [{"system": "phone", "value": str(phone)}]
    email = _first(flat, ("patient_email", "email"))
    if email:
        patient.setdefault("telecom", []).append({"system": "email", "value": str(email)})
    line = _first(
        flat,
        (
            "patient_address_street",
            "patient_address",
            "address_line_1",
            "address",
            "address_line1",
        ),
    )
    city = _first(flat, ("patient_city", "city"))
    state = _first(flat, ("patient_state", "state"))
    zip_code = _first(flat, ("patient_zip_code", "patient_zip", "zip", "postal_code"))
    if any((line, city, state, zip_code)):
        address: dict[str, Any] = {}
        if line:
            address["line"] = [str(line)]
        if city:
            address["city"] = str(city)
        if state:
            address["state"] = str(state)
        if zip_code:
            address["postalCode"] = str(zip_code)
        patient["address"] = [address]

    return patient


def _build_coverage(
    flat: dict[str, Any],
    *,
    patient_ref: dict[str, Any] | None,
) -> dict[str, Any] | None:
    # Phase K — accept the field names the analyzer's adaptive schema
    # produces ("insured_id_number") alongside the canonical CMS labels.
    member_id = _first(
        flat,
        (
            "member_id",
            "policy_number",
            "subscriber_id",
            "insurance_id",
            "insured_id_number",
            "insured_id",
            "insured_member_id",
        ),
    )
    if not member_id:
        return None
    coverage: dict[str, Any] = {
        "resourceType": "Coverage",
        "id": str(uuid.uuid4()),
        "status": "active",
        "subscriberId": str(member_id),
        "beneficiary": patient_ref or {"display": "unknown"},
    }
    payor = _first(
        flat,
        ("insurance_company", "payer_name", "insurer", "insurance_carrier", "carrier"),
    )
    if payor:
        coverage["payor"] = [{"display": str(payor)}]
    group = _first(flat, ("group_number", "insurance_group", "group_id"))
    if group:
        coverage["class"] = [
            {
                "type": {"text": "group"},
                "value": str(group),
            }
        ]
    return coverage


def _build_claim(
    flat: dict[str, Any],
    *,
    patient_ref: dict[str, Any] | None,
    coverage_ref: dict[str, Any] | None,
    claim_type: str,
) -> dict[str, Any] | None:
    # Phase K — relaxed claim-number requirement. Many CMS-1500 forms
    # leave the optional patient-account-number blank (or the analyzer
    # doesn't capture it); we synthesise an id from the service date +
    # CPT in those cases so the Claim resource still emits and downstream
    # FHIR consumers get the Patient + Coverage + Claim triple.
    claim_number = _first(
        flat,
        (
            "claim_number",
            "claim_id",
            "patient_account_number",
            "claim_control_number",
        ),
    )
    has_billing_signal = any(
        _first(flat, (key,))
        for key in (
            "total_charge",
            "total_charges",
            "billed_amount",
            "total",
            "cpt_hcpcs",
            "cpt_code",
            "procedure_code",
        )
    )
    if not claim_number and not has_billing_signal:
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
        "created": _coerce_date(
            _first(
                flat,
                (
                    "claim_date",
                    "service_date",
                    "service_date_from",
                    "statement_date",
                ),
            )
        ),
        "identifier": [{"value": str(claim_number or flat.get("processing_id") or uuid.uuid4())}],
    }
    if coverage_ref:
        claim["insurance"] = [
            {
                "sequence": 1,
                "focal": True,
                "coverage": coverage_ref,
            }
        ]
    total = _coerce_money(
        _first(flat, ("total_charges", "total_charge", "billed_amount", "total"))
    )
    if total is not None:
        claim["total"] = total
    # Phase K — line-item carry-through. CMS-1500 box 24 is the
    # service-line table; we materialise the primary line so the FHIR
    # Claim has at least one ``item`` entry. Multi-line packets are a
    # follow-up; today's adaptive schema flattens to box-24 line 1.
    primary_cpt = _first(flat, ("cpt_hcpcs", "cpt_code", "procedure_code"))
    if primary_cpt:
        item: dict[str, Any] = {
            "sequence": 1,
            "productOrService": {
                "coding": [
                    {
                        "system": "http://www.ama-assn.org/go/cpt",
                        "code": str(primary_cpt),
                    }
                ]
            },
        }
        modifier = _first(flat, ("modifier", "modifier_line1"))
        if modifier:
            item["modifier"] = [{"text": str(modifier)}]
        units = _first(flat, ("units", "units_line1"))
        if units is not None:
            try:
                item["quantity"] = {"value": float(units)}
            except (TypeError, ValueError):
                pass
        line_charge = _coerce_money(_first(flat, ("charges", "charge_line1", "line1_charge")))
        if line_charge is not None:
            item["net"] = line_charge
        claim["item"] = [item]
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
