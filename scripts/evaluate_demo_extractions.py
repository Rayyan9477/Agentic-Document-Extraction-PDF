"""
Critical evaluator for the Phase K demo extractions.

Loads the three demo extraction outputs and compares each against the
ground truth at ``data/demo/synthea_patient.json``. Produces a
per-field per-variant table with PASS / NORM / MISS / WRONG verdicts
and a headline fidelity number per variant.

Usage::

    python scripts/evaluate_demo_extractions.py

Assumes extractions live under ``output/{clean,faxed,handwritten}/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parent.parent
_GT = _REPO / "data" / "demo" / "synthea_patient.json"
_INVOICE_GT = _REPO / "data" / "demo" / "invoice_data.json"
_OUTPUT_BASE = _REPO / "output"

# Ground-truth → expected extracted-field mapping. The model emits any
# of several plausible field names per fact; the verifier checks the
# first that exists.
TRUTH_TO_FIELDS: dict[str, list[str]] = {
    "patient_name": [
        "patient_name",
        "patient_full_name",
        "patient",
    ],
    "patient_dob": ["patient_birth_date", "patient_dob", "date_of_birth", "dob"],
    "patient_sex": ["sex", "patient_sex", "gender"],
    "patient_zip": ["patient_zip_code", "patient_zip", "zip"],
    "patient_phone": ["patient_phone", "phone", "telephone"],
    "patient_address": [
        "patient_address_street",
        "patient_address",
        "address_line_1",
        "address",
    ],
    "patient_city": ["patient_city", "city"],
    "patient_state": ["patient_state", "state"],
    "member_id": [
        "insured_id_number",
        "member_id",
        "subscriber_id",
        "insurance_id",
    ],
    "group_number": ["group_number", "group", "insurance_group"],
    "diagnosis_1": ["diagnosis_code_1", "diagnosis_1", "icd_1"],
    "diagnosis_2": ["diagnosis_code_2", "diagnosis_2", "icd_2"],
    "service_date": [
        "service_date_from",
        "service_date",
        "date_of_service",
    ],
    "cpt_line1": ["cpt_hcpcs", "cpt_code", "procedure_code"],
    "modifier_line1": ["modifier", "modifier_line1"],
    "pos_code": ["pos", "place_of_service"],
    "dx_pointer_line1": ["dx_ptr", "dx_pointer", "diagnosis_pointer"],
    "units_line1": ["units", "units_line1"],
    "charge_line1": ["charges", "charge_line1", "line1_charge"],
    "total_charge": ["total_charge", "total"],
    "tax_id": ["tax_id_number", "tax_id", "federal_tax_id"],
    "physician_name": ["physician_signature", "physician_name", "provider_name"],
    "facility_address": [
        "service_facility_location",
        "facility_address",
        "service_facility",
    ],
    "billing_provider_npi": [
        "billing_provider_npi",
        "billing_provider_name",
        "provider_npi",
        "npi",
    ],
}


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    # Drop punctuation that frequently round-trips differently.
    return s.replace("-", "").replace(",", "").replace(".", "").replace(" ", "")


def _truth_values(truth: dict[str, Any]) -> dict[str, Any]:
    pat = truth["patient"]
    ins = truth["insurance"]
    prov = truth["provider"]
    enc = truth["encounter"]
    line1 = enc["service_lines"][0]
    full_name = f"{pat['last_name']}, {pat['first_name']} {pat['middle_name']}"
    return {
        "patient_name": full_name,
        "patient_dob": pat["dob"],
        "patient_sex": pat["sex"],
        "patient_zip": pat["zip"],
        "patient_phone": pat["phone"],
        "patient_address": pat["address_line1"],
        "patient_city": pat["city"],
        "patient_state": pat["state"],
        "member_id": ins["member_id"],
        "group_number": ins["group_number"],
        "diagnosis_1": enc["diagnosis_codes"][0],
        "diagnosis_2": enc["diagnosis_codes"][1],
        "service_date": line1["service_date_from"],
        "cpt_line1": line1["cpt_code"],
        "modifier_line1": line1["modifier"] or "",
        "pos_code": line1["place_of_service"],
        "dx_pointer_line1": line1["diagnosis_pointer"],
        "units_line1": line1["units"],
        "charge_line1": line1["charge"],
        "total_charge": enc["total_charge"],
        "tax_id": prov["tax_id"],
        "physician_name": prov["name"],
        "facility_address": prov["address_line1"],
        "billing_provider_npi": prov["npi"],
    }


def _evaluate_record(record: dict[str, Any], truth: dict[str, Any]) -> dict[str, str]:
    fields = record.get("fields", {})
    expected = _truth_values(truth)
    results: dict[str, str] = {}
    for key, candidates in TRUTH_TO_FIELDS.items():
        gt = expected[key]
        found_field = next((c for c in candidates if c in fields), None)
        if found_field is None:
            results[key] = "MISS"
            continue
        actual = fields[found_field]
        if _normalise(actual) == _normalise(gt):
            results[key] = "PASS"
        elif _normalise(actual) and _normalise(gt) in _normalise(actual):
            results[key] = "NORM"  # substring contains GT — normalisation drift only
        elif actual is None or _normalise(actual) == "":
            results[key] = "MISS"
        else:
            results[key] = f"WRONG ({actual!r} vs {gt!r})"
    return results


def _evaluate_fhir(variant_dir: Path) -> dict[str, Any]:
    """Inspect the emitted FHIR Bundle (if any) for the canonical CMS-1500 triple."""
    fhir_files = list(variant_dir.glob("*.fhir.json"))
    if not fhir_files:
        return {"emitted": False}
    bundle = json.loads(fhir_files[0].read_text(encoding="utf-8"))
    resource_types = [
        e.get("resource", {}).get("resourceType") for e in bundle.get("entry", [])
    ]
    has_patient = "Patient" in resource_types
    has_coverage = "Coverage" in resource_types
    has_claim = "Claim" in resource_types
    has_doc_ref_fallback = "DocumentReference" in resource_types
    # Pull demographics + claim totals for the critique.
    patient = next(
        (e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient"),
        {},
    )
    claim = next(
        (e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Claim"),
        {},
    )
    return {
        "emitted": True,
        "path": str(fhir_files[0].relative_to(variant_dir.parent.parent)),
        "resource_types": resource_types,
        "has_full_triple": has_patient and has_coverage and has_claim,
        "has_fallback_doc_ref": has_doc_ref_fallback,
        "patient_demographics_present": bool(
            patient.get("birthDate") and patient.get("address") and patient.get("telecom")
        ),
        "claim_total": claim.get("total"),
        "claim_item_count": len(claim.get("item", [])),
    }


_INVOICE_TRUTH_TO_FIELDS: dict[str, list[str]] = {
    "vendor_name": ["vendor_name", "vendor", "supplier_name", "seller_name"],
    "vendor_phone": ["vendor_phone", "phone", "supplier_phone"],
    "vendor_tax_id": ["vendor_tax_id", "tax_id", "ein"],
    "customer_name": ["customer_name", "bill_to", "buyer_name", "client_name"],
    "customer_city": ["customer_city", "bill_to_city", "city"],
    "invoice_number": ["invoice_number", "invoice_no", "invoice_id"],
    "invoice_date": ["invoice_date", "date"],
    "due_date": ["due_date", "payment_due_date"],
    "po_number": ["po_number", "purchase_order", "purchase_order_number"],
    "subtotal": ["subtotal", "sub_total"],
    "tax": ["tax", "tax_amount", "sales_tax"],
    "total": ["total", "total_due", "amount_due", "grand_total"],
    "line1_sku": ["line1_sku", "sku", "sku_line1", "item_1_sku"],
    "line1_qty": ["line1_quantity", "quantity", "qty_line1", "item_1_qty"],
    "line1_unit_price": ["line1_unit_price", "unit_price", "price"],
}


def _invoice_truth(truth: dict[str, Any]) -> dict[str, Any]:
    vendor = truth["vendor"]
    customer = truth["customer"]
    inv = truth["invoice"]
    line1 = truth["line_items"][0]
    totals = truth["totals"]
    return {
        "vendor_name": vendor["name"],
        "vendor_phone": vendor["phone"],
        "vendor_tax_id": vendor["tax_id"],
        "customer_name": customer["name"],
        "customer_city": customer["city"],
        "invoice_number": inv["number"],
        "invoice_date": inv["date"],
        "due_date": inv["due_date"],
        "po_number": inv["po_number"],
        "subtotal": totals["subtotal"],
        "tax": totals["tax"],
        "total": totals["total"],
        "line1_sku": line1["sku"],
        "line1_qty": line1["quantity"],
        "line1_unit_price": line1["unit_price"],
    }


def _evaluate_invoice_record(record, truth):
    """Single-record (aggregate) invoice eval. Used when the model
    chose to flatten the whole invoice into one record."""
    fields = record.get("fields", {})
    expected = _invoice_truth(truth)
    results: dict[str, str] = {}
    for key, candidates in _INVOICE_TRUTH_TO_FIELDS.items():
        gt = expected[key]
        found_field = next((c for c in candidates if c in fields), None)
        if found_field is None:
            results[key] = "MISS"
            continue
        actual = fields[found_field]
        if _normalise(actual) == _normalise(gt):
            results[key] = "PASS"
        elif _normalise(actual) and _normalise(gt) in _normalise(actual):
            results[key] = "NORM"
        elif actual is None or _normalise(actual) == "":
            results[key] = "MISS"
        else:
            results[key] = f"WRONG ({actual!r} vs {gt!r})"
    return results


def _records_look_like_line_items(records: list[dict[str, Any]]) -> bool:
    """Heuristic: invoice line-item records carry ``sku`` / ``qty`` /
    ``unit_price`` / ``amount`` shapes. Gemma 4 chose this mode for the
    invoice — extract each line as its own record."""
    if not records:
        return False
    first = records[0].get("fields", {})
    line_signals = {"sku", "qty", "quantity", "unit_price", "amount", "description"}
    return len(line_signals & set(first.keys())) >= 3


def _evaluate_invoice_lines(records, truth) -> dict[str, str]:
    """Per-line-item invoice eval. One record per line in the synthetic
    invoice; we match by SKU when present, else by record index."""
    gt_lines = truth["line_items"]
    results: dict[str, str] = {}
    by_sku = {ln["sku"]: ln for ln in gt_lines}
    matched = 0
    for i, rec in enumerate(records):
        f = rec.get("fields", {})
        sku = f.get("sku") or f.get("SKU")
        gt = by_sku.get(sku) if sku else (gt_lines[i] if i < len(gt_lines) else None)
        if gt is None:
            results[f"line{i+1}_match"] = "WRONG (no GT line)"
            continue
        matched += 1
        # Check each field on the line.
        for fname, gt_val in (
            ("description", gt["description"]),
            ("sku", gt["sku"]),
            ("qty", gt["quantity"]),
            ("unit_price", gt["unit_price"]),
            ("amount", gt["amount"]),
        ):
            actual = f.get(fname)
            if actual is None:
                results[f"line{i+1}_{fname}"] = "MISS"
            elif _normalise(actual) == _normalise(gt_val):
                results[f"line{i+1}_{fname}"] = "PASS"
            elif _normalise(actual) and _normalise(gt_val) in _normalise(actual):
                results[f"line{i+1}_{fname}"] = "NORM"
            else:
                results[f"line{i+1}_{fname}"] = f"WRONG ({actual!r} vs {gt_val!r})"
    results["_lines_matched"] = f"{matched}/{len(gt_lines)}"
    return results


def _evaluate_validations(variant_dir: Path) -> dict[str, Any]:
    vp = variant_dir / "validations.json"
    if not vp.exists():
        return {"emitted": False}
    rep = json.loads(vp.read_text(encoding="utf-8"))
    totals = rep.get("totals", {})
    flagged: list[str] = []
    for rec in rep.get("records", {}).values():
        flagged.extend(rec.get("summary", {}).get("failed_fields", []))
    return {
        "emitted": True,
        "records_processed": totals.get("records_processed", 0),
        "total_failed_validations": totals.get("total_failed_validations", 0),
        "failed_fields": flagged,
    }


def _evaluate_receipt(variant_dir: Path) -> dict[str, Any]:
    """Inspect the signed receipt for hash coverage + signature presence."""
    rp = variant_dir / "receipt.json"
    if not rp.exists():
        return {"emitted": False}
    receipt = json.loads(rp.read_text(encoding="utf-8"))
    return {
        "emitted": True,
        "artefact_count": len(receipt.get("artefact_hashes", {})),
        "signed": receipt.get("signature") is not None,
        "profile": receipt.get("profile"),
    }


def evaluate(
    variant_dir: Path,
    truth: dict[str, Any],
    *,
    is_invoice: bool = False,
) -> dict[str, Any]:
    json_files = list(variant_dir.glob("*_results.json"))
    if not json_files:
        return {"status": "no_extraction_found", "dir": str(variant_dir)}
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    if not data.get("records"):
        return {"status": "no_records", "dir": str(variant_dir)}
    record = data["records"][0]
    if is_invoice and _records_look_like_line_items(data["records"]):
        field_results = _evaluate_invoice_lines(data["records"], truth)
    elif is_invoice:
        field_results = _evaluate_invoice_record(record, truth)
    else:
        field_results = _evaluate_record(record, truth)
    # ``_lines_matched`` is a synthetic informational key, not a verdict —
    # filter it out of the pass/miss aggregates.
    scored = {k: v for k, v in field_results.items() if not k.startswith("_")}
    n = len(scored)
    passes = sum(1 for v in scored.values() if v == "PASS")
    norms = sum(1 for v in scored.values() if v == "NORM")
    misses = sum(1 for v in scored.values() if v == "MISS")
    wrongs = sum(1 for v in scored.values() if v.startswith("WRONG"))
    return {
        "status": "ok",
        "document_type": data.get("document_type"),
        "record_count": data.get("total_records", 0),
        "vlm_calls": data.get("total_vlm_calls", 0),
        "processing_time_s": (data.get("total_processing_time_ms", 0) or 0) / 1000.0,
        "record_confidence": record.get("confidence"),
        "field_count": n,
        "pass": passes,
        "norm": norms,
        "miss": misses,
        "wrong": wrongs,
        "fidelity": round((passes + 0.5 * norms) / n, 3),
        "field_results": field_results,
        "fhir": _evaluate_fhir(variant_dir),
        "receipt": _evaluate_receipt(variant_dir),
        "tool_validation": _evaluate_validations(variant_dir),
    }


def main() -> int:
    truth = json.loads(_GT.read_text(encoding="utf-8"))
    invoice_truth = (
        json.loads(_INVOICE_GT.read_text(encoding="utf-8"))
        if _INVOICE_GT.exists()
        else None
    )
    print("Phase K demo extraction — critical evaluation")
    print("=" * 70)

    variant_dirs: list[tuple[str, Path, dict, bool]] = [
        ("clean", _OUTPUT_BASE / "clean", truth, False),
        ("faxed", _OUTPUT_BASE / "faxed", truth, False),
        ("handwritten", _OUTPUT_BASE / "handwritten", truth, False),
    ]
    if invoice_truth is not None:
        variant_dirs.append(
            ("invoice_clean", _OUTPUT_BASE / "invoice_clean", invoice_truth, True)
        )
        variant_dirs.append(
            ("invoice_faxed", _OUTPUT_BASE / "invoice_faxed", invoice_truth, True)
        )

    summary = {}
    for name, dir_, gt, is_invoice in variant_dirs:
        if not dir_.exists():
            print(f"\n[{name:>14}] skipped: {dir_} missing")
            continue
        report = evaluate(dir_, gt, is_invoice=is_invoice)
        summary[name] = report
        if report["status"] != "ok":
            print(f"\n[{name:>14}] {report['status']} ({report.get('dir')})")
            continue
        print(
            f"\n[{name:>14}] {report['fidelity']:.0%} fidelity "
            f"({report['pass']}/{report['field_count']} pass + {report['norm']} norm + "
            f"{report['miss']} miss + {report['wrong']} wrong) - "
            f"{report['vlm_calls']} VLM calls, "
            f"{report['processing_time_s']:.0f}s, "
            f"confidence {report['record_confidence']:.2f}, "
            f"doc_type={report['document_type']}"
        )
        # Per-field detail
        for key, verdict in report["field_results"].items():
            status_short = verdict if len(verdict) <= 12 else verdict[:60]
            print(f"    {key:<22} {status_short}")
        fhir = report.get("fhir") or {}
        if fhir.get("emitted"):
            tri = "Y" if fhir["has_full_triple"] else "N"
            demo = "Y" if fhir["patient_demographics_present"] else "N"
            print(
                f"    FHIR Bundle: full triple={tri}  patient demographics={demo}  "
                f"resources={fhir['resource_types']}  claim items={fhir['claim_item_count']}"
            )
        else:
            print("    FHIR Bundle: NOT emitted (expected for General mode)" if is_invoice else "    FHIR Bundle: NOT emitted")
        receipt = report.get("receipt") or {}
        if receipt.get("emitted"):
            print(
                f"    Receipt: artefacts={receipt['artefact_count']}, "
                f"signed={receipt['signed']}, profile={receipt['profile']!r}"
            )
        else:
            print("    Receipt: NOT emitted")
        tv = report.get("tool_validation") or {}
        if tv.get("emitted"):
            print(
                f"    Tool validation: {tv['total_failed_validations']} failed check(s)"
                + (f" ({', '.join(tv['failed_fields'])})" if tv['failed_fields'] else "")
            )
        else:
            print("    Tool validation: NOT emitted")

    # Headline comparison
    if summary:
        print()
        print("=" * 70)
        print("Headline:")
        for name, report in summary.items():
            if report.get("status") == "ok":
                tv = report.get("tool_validation", {})
                flagged_str = (
                    f" tool-flagged: {tv.get('total_failed_validations', 0)}"
                    if tv.get("emitted")
                    else ""
                )
                print(
                    f"  {name:<14} {report['fidelity']:.0%} "
                    f"({report['pass']:>2} pass / {report['norm']:>2} norm / "
                    f"{report['miss']:>2} miss / {report['wrong']:>2} wrong) "
                    f"@ {report['processing_time_s']:>5.1f}s{flagged_str}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
