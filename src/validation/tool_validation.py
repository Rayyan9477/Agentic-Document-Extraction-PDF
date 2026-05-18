"""
Phase K — post-extraction tool validation.

After the VLM finishes its pass, every extracted record is scrubbed
through the same five medical-code validators that Gemma 4's native
function-calling interface exposes (see
``src/client/backends/gemma_tools.py``):

* ``npi_luhn_check``  — catches dropped check digits on NPIs
* ``cpt_validate``    — catches CPT typos
* ``icd_normalize``   — flags malformed ICD-10 codes
* ``sum_reconcile``   — flags line-items that don't sum to the total
* ``validate_date_ordering`` — catches DOB-after-service-date and
  admission/discharge ordering errors

This module is the **deterministic safety net** that closes the loop
the live faxed-CMS-1500 run showed: when Gemma extracts an NPI like
``1234567890`` (dropped the real check digit ``3``), the Luhn tool
fires and the result lands as a ``failed_validations`` entry on the
record. Downstream consumers (the JSON export, the audit log, the
review UI) can route on it.

Wired into ``extract_pdf_cli`` so every Healthcare-mode extraction
produces a ``validations.json`` artefact alongside the existing
exports. General-mode extractions skip the medical validators
(they'd produce only false positives on invoice fields), but the
``sum_reconcile`` and ``validate_date_ordering`` tools are
profile-agnostic and run when the field shapes match.
"""

from __future__ import annotations

import re
from typing import Any

from src.client.backends.gemma_tools import TOOL_DISPATCH
from src.config import get_logger


logger = get_logger(__name__)


# Field-name → tool routing. The keys are case-insensitive substring
# matches against the extracted record's field-name keys. Order matters
# only for human-readability; each field is routed independently.
_NPI_FIELD_PATTERNS = (
    "npi",  # billing_provider_npi, rendering_npi, etc.
)
_CPT_FIELD_PATTERNS = (
    "cpt",  # cpt_code, cpt_hcpcs, procedure_code w/o cpt — handled below
    "hcpcs",
    "procedure_code",
)
_ICD_FIELD_PATTERNS = (
    "diagnosis_code",
    "icd",
)


def _route_field(field_name: str) -> str | None:
    """Return the tool name that should validate ``field_name``, or None.

    The field-name match is case-insensitive substring. ``npi`` and
    ``hcpcs`` are checked before ``cpt`` because ``hcpcs_cpt`` matches
    both and we want the more specific routing.
    """
    name = field_name.lower()
    if any(p in name for p in _NPI_FIELD_PATTERNS):
        return "npi_luhn_check"
    if any(p in name for p in _ICD_FIELD_PATTERNS):
        return "icd_normalize"
    if any(p in name for p in _CPT_FIELD_PATTERNS):
        return "cpt_validate"
    return None


def _extract_npi_from_value(value: Any) -> str | None:
    """Pull a 10-digit NPI out of a value that may have surrounding text.

    The live run showed Gemma concatenates the NPI into the provider
    name string ('Springfield Family Medicine — NPI 1234567893'). We
    extract the first 10-digit run so the Luhn tool still validates it.
    """
    if value is None:
        return None
    s = str(value)
    match = re.search(r"\b(\d{10})\b", s)
    return match.group(1) if match else None


def _validate_one_field(
    field_name: str,
    value: Any,
    *,
    tool_name: str,
) -> dict[str, Any]:
    """Dispatch one field to its tool. Returns the tool's reply unchanged."""
    if tool_name == "npi_luhn_check":
        npi = _extract_npi_from_value(value)
        if npi is None:
            return {"valid": True, "reason": "no_npi_shape_found", "skipped": True}
        return TOOL_DISPATCH["npi_luhn_check"](npi=npi)
    if tool_name == "cpt_validate":
        return TOOL_DISPATCH["cpt_validate"](cpt_code=str(value))
    if tool_name == "icd_normalize":
        return TOOL_DISPATCH["icd_normalize"](raw_code=str(value))
    return {"valid": True, "reason": "no_tool_route", "skipped": True}


def _validate_sum_reconcile(fields: dict[str, Any]) -> dict[str, Any] | None:
    """If the record carries both line-item charges and a total, reconcile."""
    line_amounts: list[float] = []
    reported_total: float | None = None
    # Common line-charge field names from adaptive + static CMS-1500 schemas.
    for key in (
        "charges",
        "charge_line1",
        "line1_charge",
        "charge_line_1",
        "line_charge",
    ):
        if key in fields and fields[key] is not None:
            try:
                line_amounts.append(float(fields[key]))
                break
            except (TypeError, ValueError):
                pass
    for key in ("total_charge", "total_charges", "billed_amount", "total"):
        if key in fields and fields[key] is not None:
            try:
                reported_total = float(fields[key])
                break
            except (TypeError, ValueError):
                pass
    if not line_amounts or reported_total is None:
        return None
    return TOOL_DISPATCH["sum_reconcile"](
        line_amounts=line_amounts,
        reported_total=reported_total,
        tolerance_cents=2,  # 2-cent rounding tolerance
    )


def _validate_dates(fields: dict[str, Any]) -> dict[str, Any] | None:
    """If DOB + at least one service date are present, run date ordering."""
    dob = None
    for key in ("patient_birth_date", "patient_dob", "date_of_birth", "dob"):
        if key in fields and fields[key]:
            dob = str(fields[key])
            break
    if not dob:
        return None
    service_dates: list[str] = []
    for key in ("service_date_from", "service_date", "date_of_service"):
        if key in fields and fields[key]:
            service_dates.append(str(fields[key]))
            break
    if not service_dates:
        return None
    return TOOL_DISPATCH["validate_date_ordering"](
        date_of_birth=dob,
        service_dates=service_dates,
        admission_date=fields.get("admission_date"),
        discharge_date=fields.get("discharge_date"),
    )


def validate_record(record_fields: dict[str, Any]) -> dict[str, Any]:
    """Validate every recognisable field in ``record_fields``.

    Returns a dict of the form::

        {
          "validations": {
            "<field_name>": {"tool": "...", "valid": bool, ...}
          },
          "summary": {
            "fields_validated": int,
            "failed": int,
            "failed_fields": list[str]
          }
        }

    Fields that don't match any tool's domain are skipped (silently —
    only routed fields appear in the output). Cross-field validations
    (sum_reconcile, validate_date_ordering) appear under the synthetic
    keys ``"_sum_reconcile"`` and ``"_date_ordering"``.
    """
    validations: dict[str, Any] = {}
    failed_fields: list[str] = []

    # Per-field validators.
    for field_name, value in record_fields.items():
        tool = _route_field(field_name)
        if tool is None:
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # don't flag empty fields
        result = _validate_one_field(field_name, value, tool_name=tool)
        validations[field_name] = {
            "tool": tool,
            "value_seen": value,
            **result,
        }
        if not result.get("valid", True) and not result.get("skipped"):
            failed_fields.append(field_name)

    # Cross-field validators.
    sum_result = _validate_sum_reconcile(record_fields)
    if sum_result is not None:
        validations["_sum_reconcile"] = {"tool": "sum_reconcile", **sum_result}
        if not sum_result.get("match", False):
            failed_fields.append("_sum_reconcile")

    date_result = _validate_dates(record_fields)
    if date_result is not None:
        validations["_date_ordering"] = {"tool": "validate_date_ordering", **date_result}
        if not date_result.get("valid", True):
            failed_fields.append("_date_ordering")

    return {
        "validations": validations,
        "summary": {
            "fields_validated": len(validations),
            "failed": len(failed_fields),
            "failed_fields": failed_fields,
        },
    }


def validate_extraction_result(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate every record in a ``DocumentExtractionResult.to_dict()``.

    Returns a summary keyed by record id::

        {
          "records": {
            "<record_id>": {
              "primary_identifier": "...",
              "validations": {...},
              "summary": {...},
            }
          },
          "totals": {
            "records_processed": int,
            "total_failed_validations": int,
          }
        }
    """
    records_out: dict[str, Any] = {}
    total_failed = 0
    for rec in result_dict.get("records", []):
        report = validate_record(rec.get("fields", {}))
        record_key = str(rec.get("record_id", "unknown"))
        records_out[record_key] = {
            "primary_identifier": rec.get("primary_identifier"),
            **report,
        }
        total_failed += report["summary"]["failed"]
        if report["summary"]["failed"]:
            logger.warning(
                "tool_validation_failed_fields",
                record_id=record_key,
                primary_id=rec.get("primary_identifier"),
                failed_fields=report["summary"]["failed_fields"],
            )
    return {
        "records": records_out,
        "totals": {
            "records_processed": len(records_out),
            "total_failed_validations": total_failed,
        },
    }


__all__ = ["validate_record", "validate_extraction_result"]
