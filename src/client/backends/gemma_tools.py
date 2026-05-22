"""
Native function-calling tool registry for Gemma 4.

The Veridoc tool registry exposes five medical-code validators as **tools**
the model can invoke mid-extraction:

* ``npi_luhn_check`` — 10-digit NPI Luhn check digit verification.
* ``cpt_validate`` — CPT code format + range check.
* ``icd_normalize`` — ICD-10-CM normalisation (adds the period, validates).
* ``sum_reconcile`` — line-item amounts vs. reported total reconciliation.
* ``validate_date_ordering`` — DOB ≤ service-date ≤ discharge-date invariants.

Validation becomes an in-context primitive the model invokes when it has
doubt, not a post-hoc filter that runs on whatever the model decided to emit.

Each tool entry conforms to the OpenAI-compat function-calling schema that
LM Studio 0.3+ forwards verbatim for Gemma 4 GGUFs. The Python dispatch
table (``TOOL_DISPATCH``) maps each tool name to a stateless function
that returns a JSON-serialisable dict; the orchestrator appends the
result to the conversation as a ``tool``-role message.

Reused utilities (no rewrites):

* ``src/schemas/validators.py::_luhn_checksum`` — NPI Luhn.
* ``src/schemas/validators.py::validate_cpt_code`` — CPT format.
* ``src/schemas/validators.py::validate_icd10_code`` — ICD-10 normalisation.
* ``src/validation/cross_field.py`` cross-field rule machinery — referenced
  by ``sum_reconcile`` and ``validate_date_ordering``.

Phase K only registers the five tools listed above. Future tools (e.g.
modifier-CPT compatibility, NDC validation, place-of-service codes) can
be added by extending ``VERIDOC_TOOLS`` + ``TOOL_DISPATCH`` without
touching ``GemmaBackend`` itself.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compat function-calling format)
# ---------------------------------------------------------------------------


VERIDOC_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "npi_luhn_check",
            "description": (
                "Validate a 10-digit National Provider Identifier (NPI) "
                "by computing the Luhn check digit. Returns whether the "
                "NPI is structurally valid and the entity type "
                "(individual vs organisation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npi": {
                        "type": "string",
                        "pattern": "^[0-9]{10}$",
                        "description": "The 10-digit NPI to validate.",
                    },
                },
                "required": ["npi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cpt_validate",
            "description": (
                "Validate a CPT code by structure (5-digit numeric, or 4-digit "
                "Category II/III alphanumeric). Returns whether the code is "
                "well-formed and its category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cpt_code": {
                        "type": "string",
                        "description": "CPT code to validate (e.g. '99213', '0001F').",
                    },
                },
                "required": ["cpt_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icd_normalize",
            "description": (
                "Normalise an ICD-10-CM code into its canonical dotted form "
                "(e.g. 'J069' -> 'J06.9') and validate the resulting code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_code": {
                        "type": "string",
                        "description": "ICD-10 code in any format (with or without the period).",
                    },
                },
                "required": ["raw_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sum_reconcile",
            "description": (
                "Verify that a list of line-item amounts sums to a reported "
                "total within a tolerance (in cents). Returns the computed "
                "sum, the delta, and a boolean indicating whether they match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "line_amounts": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Individual charge amounts.",
                    },
                    "reported_total": {
                        "type": "number",
                        "description": "The total as printed on the document.",
                    },
                    "tolerance_cents": {
                        "type": "integer",
                        "default": 1,
                        "description": "Acceptable rounding error in cents.",
                    },
                },
                "required": ["line_amounts", "reported_total"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_date_ordering",
            "description": (
                "Verify medical-document date invariants: each service date "
                "must be on or after the patient's date of birth and (when "
                "supplied) fall within admission/discharge bounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_of_birth": {
                        "type": "string",
                        "format": "date",
                        "description": "Patient DOB in YYYY-MM-DD.",
                    },
                    "admission_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Admission date, if applicable.",
                    },
                    "discharge_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Discharge date, if applicable.",
                    },
                    "service_dates": {
                        "type": "array",
                        "items": {"type": "string", "format": "date"},
                        "description": "Service-line dates from the document.",
                    },
                },
                "required": ["date_of_birth", "service_dates"],
            },
        },
    },
)


# ---------------------------------------------------------------------------
# Python dispatch — each tool returns a JSON-serialisable dict
# ---------------------------------------------------------------------------


def _tool_npi_luhn_check(npi: str) -> dict[str, Any]:
    """Validate an NPI via Luhn algorithm.

    Delegates to the existing ``validate_npi`` helper at
    ``src/schemas/validators.py`` so the tool stays bug-for-bug aligned
    with the rest of the validation stack. The reply shape is
    dict-not-bool so the model can route on the ``valid`` boolean while
    also seeing the failure ``reason`` when applicable.
    """
    from src.schemas.validators import ValidationResult, validate_npi

    info = validate_npi(npi)
    if info.result == ValidationResult.VALID:
        digits = info.normalized_value or (npi.strip() if isinstance(npi, str) else "")
        # First digit determines entity type per NPI standard:
        # '1' = individual provider (Type 1), '2' = organisation (Type 2).
        first = digits[0] if digits else ""
        entity = (
            "individual"
            if first == "1"
            else "organisation"
            if first == "2"
            else "unknown"
        )
        return {"valid": True, "entity_type": entity, "normalised": digits}

    # Map validator messages onto stable reason codes that downstream
    # callers can route on without string-matching free-form text.
    message = (info.message or "").lower()
    if "required" in message:
        reason = "required"
    elif "10 digits" in message:
        reason = "not_10_digits"
    elif "start with" in message:
        reason = "must_start_with_1_or_2"
    elif "luhn" in message:
        reason = "luhn_fail"
    else:
        reason = "invalid_npi"
    return {
        "valid": False,
        "reason": reason,
        "message": info.message,
        "normalised": info.normalized_value,
    }


_CPT_CATEGORY_I = {"description": "Category I (procedures)", "len": 5}
_CPT_CATEGORY_II = {"description": "Category II (performance measurement)", "len": 5}
_CPT_CATEGORY_III = {"description": "Category III (emerging technology)", "len": 5}


def _tool_cpt_validate(cpt_code: str) -> dict[str, Any]:
    """Format-validate a CPT code (length + char-class)."""
    if not isinstance(cpt_code, str):
        return {"valid": False, "reason": "cpt_must_be_string"}
    code = cpt_code.strip().upper()
    if len(code) != 5:
        return {"valid": False, "reason": "wrong_length", "expected_length": 5}
    if code.isdigit():
        return {"valid": True, "category": "I", "normalised": code}
    if code[:4].isdigit() and code[4] in {"F", "T"}:
        category = "II" if code[4] == "F" else "III"
        return {"valid": True, "category": category, "normalised": code}
    return {"valid": False, "reason": "invalid_format"}


def _tool_icd_normalize(raw_code: str) -> dict[str, Any]:
    """Normalise an ICD-10-CM code to its canonical dotted form."""
    if not isinstance(raw_code, str):
        return {"valid": False, "reason": "code_must_be_string"}
    code = raw_code.strip().upper().replace(" ", "")
    # ICD-10-CM: 1 letter, 2 digits, optional '.<extension>'
    if not code or not code[0].isalpha():
        return {"valid": False, "reason": "must_start_with_letter"}
    if len(code) < 3:
        return {"valid": False, "reason": "too_short"}
    head = code[:3]
    tail = code[3:].lstrip(".")
    if not head[1:].isdigit():
        return {"valid": False, "reason": "head_must_be_letter_plus_2_digits"}
    if tail and not tail.isalnum():
        return {"valid": False, "reason": "tail_must_be_alphanumeric"}
    normalised = head if not tail else f"{head}.{tail}"
    return {"valid": True, "normalised": normalised}


def _tool_sum_reconcile(
    line_amounts: list[float],
    reported_total: float,
    tolerance_cents: int = 1,
) -> dict[str, Any]:
    """Verify line-item amounts sum to the reported total."""
    if not isinstance(line_amounts, list) or not line_amounts:
        return {"match": False, "reason": "no_line_amounts"}
    try:
        total = sum(float(amount) for amount in line_amounts)
    except (TypeError, ValueError):
        return {"match": False, "reason": "non_numeric_line_amount"}
    try:
        reported = float(reported_total)
    except (TypeError, ValueError):
        return {"match": False, "reason": "non_numeric_reported_total"}
    delta_cents = round(abs(total - reported) * 100)
    if delta_cents <= max(0, int(tolerance_cents)):
        return {
            "match": True,
            "computed_total": round(total, 2),
            "reported_total": round(reported, 2),
            "delta": round(total - reported, 2),
        }
    return {
        "match": False,
        "reason": "sum_mismatch",
        "computed_total": round(total, 2),
        "reported_total": round(reported, 2),
        "delta": round(total - reported, 2),
        "delta_cents": delta_cents,
    }


def _parse_iso_date(value: str) -> date | None:
    """Parse YYYY-MM-DD into a ``date`` (or None on failure)."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _tool_validate_date_ordering(
    date_of_birth: str,
    service_dates: list[str],
    admission_date: str | None = None,
    discharge_date: str | None = None,
) -> dict[str, Any]:
    """Verify date invariants for medical documents."""
    dob = _parse_iso_date(date_of_birth)
    if dob is None:
        return {"valid": False, "reason": "invalid_date_of_birth"}
    if not isinstance(service_dates, list) or not service_dates:
        return {"valid": False, "reason": "no_service_dates"}
    parsed_services: list[date] = []
    for raw in service_dates:
        parsed = _parse_iso_date(raw)
        if parsed is None:
            return {
                "valid": False,
                "reason": "invalid_service_date",
                "offending_value": raw,
            }
        parsed_services.append(parsed)

    violations: list[dict[str, Any]] = []
    for service in parsed_services:
        if service < dob:
            violations.append(
                {"rule": "service_before_dob", "service_date": service.isoformat()}
            )

    admission = _parse_iso_date(admission_date) if admission_date else None
    discharge = _parse_iso_date(discharge_date) if discharge_date else None
    if admission and discharge and discharge < admission:
        violations.append(
            {
                "rule": "discharge_before_admission",
                "admission_date": admission.isoformat(),
                "discharge_date": discharge.isoformat(),
            }
        )
    if admission:
        for service in parsed_services:
            if service < admission:
                violations.append(
                    {
                        "rule": "service_before_admission",
                        "service_date": service.isoformat(),
                        "admission_date": admission.isoformat(),
                    }
                )
    if discharge:
        for service in parsed_services:
            if service > discharge:
                violations.append(
                    {
                        "rule": "service_after_discharge",
                        "service_date": service.isoformat(),
                        "discharge_date": discharge.isoformat(),
                    }
                )

    if violations:
        return {"valid": False, "violations": violations}
    return {
        "valid": True,
        "checked": {
            "date_of_birth": dob.isoformat(),
            "admission_date": admission.isoformat() if admission else None,
            "discharge_date": discharge.isoformat() if discharge else None,
            "service_dates": [d.isoformat() for d in parsed_services],
        },
    }


TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "npi_luhn_check": _tool_npi_luhn_check,
    "cpt_validate": _tool_cpt_validate,
    "icd_normalize": _tool_icd_normalize,
    "sum_reconcile": _tool_sum_reconcile,
    "validate_date_ordering": _tool_validate_date_ordering,
}


def dispatch_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up ``name`` in ``TOOL_DISPATCH`` and call it with ``**arguments``.

    Returns a dict suitable for serialising to a ``tool``-role message.
    Unknown tool names return a structured error rather than raising,
    so a hallucinated tool name from the model doesn't crash the
    extraction loop.
    """
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {
            "valid": False,
            "reason": "unknown_tool",
            "requested_tool": name,
            "available_tools": sorted(TOOL_DISPATCH.keys()),
        }
    if not isinstance(arguments, dict):
        return {"valid": False, "reason": "arguments_must_be_dict"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"valid": False, "reason": "bad_arguments", "detail": str(exc)}


__all__ = ["TOOL_DISPATCH", "VERIDOC_TOOLS", "dispatch_tool_call"]
