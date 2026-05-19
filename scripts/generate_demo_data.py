"""
Generate the Phase K demo dataset (3 synthetic CMS-1500 PDFs).

This script is the single source of truth for the demo files committed
to ``data/demo/``. All patient identifiers are synthetic (see
``data/demo/synthea_patient.json``).

Outputs:

* ``data/demo/cms1500_clean.pdf`` — clean printed CMS-1500.
* ``data/demo/cms1500_faxed.pdf`` — Otsu-binarised + salt-pepper noise +
  slight rotation; simulates a fax-scanned claim.
* ``data/demo/cms1500_handwritten.pdf`` — patient-name + address +
  signature fields overlaid with Comic Sans (a stand-in for a
  hand-completed form) plus jitter.

Run with::

    python scripts/generate_demo_data.py

Deterministic (seeded RNG); re-runs produce byte-identical output. Safe
to commit to the repo.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Layout constants — CMS-1500 is letter-portrait (8.5x11 inches @ 200 DPI)
# ---------------------------------------------------------------------------


PAGE_W = 1700  # 8.5in × 200dpi
PAGE_H = 2200  # 11in × 200dpi
DPI = 200
MARGIN = 80

# Font paths — Windows defaults. Fall back to PIL's built-in default
# font if these don't resolve, so the script keeps working on Linux/CI.
ARIAL_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
COMIC_REGULAR = Path("C:/Windows/Fonts/comic.ttf")


def _load_font(path: Path, size: int) -> ImageFont.ImageFont:
    try:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    except OSError:
        pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# CMS-1500 form layout
# ---------------------------------------------------------------------------


def _draw_header(draw: ImageDraw.ImageDraw) -> None:
    """The 'HEALTH INSURANCE CLAIM FORM' header is the medical-RCM detection trigger.

    See ``src/profiles/medical_rcm.py::MEDICAL_RCM_SIGNALS`` —
    ``hcfa_header`` matches case-insensitively against this exact phrase.
    """
    title_font = _load_font(ARIAL_BOLD, 38)
    sub_font = _load_font(ARIAL_REGULAR, 20)
    draw.text(
        (PAGE_W // 2, MARGIN + 20),
        "HEALTH INSURANCE CLAIM FORM",
        font=title_font,
        fill="black",
        anchor="mt",
    )
    draw.text(
        (PAGE_W // 2, MARGIN + 70),
        "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE 02/12",
        font=sub_font,
        fill="#666666",
        anchor="mt",
    )


def _draw_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    *,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
) -> None:
    """One labelled CMS-1500 box."""
    draw.rectangle([(x, y), (x + w, y + h)], outline="black", width=2)
    draw.text((x + 8, y + 6), label, font=label_font, fill="#444444")
    # Value drawn just below the label, padded inside the box.
    draw.text((x + 8, y + 26), value, font=value_font, fill="black")


def _draw_form(
    canvas: Image.Image,
    patient: dict[str, Any],
) -> None:
    """Render the CMS-1500 onto ``canvas`` using patient data."""
    draw = ImageDraw.Draw(canvas)
    _draw_header(draw)

    label_font = _load_font(ARIAL_REGULAR, 14)
    value_font = _load_font(ARIAL_REGULAR, 22)
    bold_font = _load_font(ARIAL_BOLD, 16)

    pat = patient["patient"]
    ins = patient["insurance"]
    prov = patient["provider"]
    bill = patient["billing_provider"]
    enc = patient["encounter"]

    row_y = MARGIN + 140
    col_w = (PAGE_W - 2 * MARGIN) // 4

    # Row 1: Insurance type + Insured's ID
    _draw_box(
        draw,
        MARGIN,
        row_y,
        col_w * 2,
        70,
        "1a. INSURED'S I.D. NUMBER (For Program in Item 1)",
        ins["member_id"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 2,
        row_y,
        col_w,
        70,
        "Sex",
        pat["sex"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 3,
        row_y,
        col_w,
        70,
        "Group #",
        ins["group_number"],
        label_font=label_font,
        value_font=value_font,
    )

    row_y += 90
    # Row 2: Patient's name + DOB
    full_name = f"{pat['last_name']}, {pat['first_name']} {pat['middle_name']}"
    _draw_box(
        draw,
        MARGIN,
        row_y,
        col_w * 2,
        70,
        "2. PATIENT'S NAME (Last Name, First Name, Middle Initial)",
        full_name,
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 2,
        row_y,
        col_w,
        70,
        "3. PATIENT'S BIRTH DATE",
        pat["dob"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 3,
        row_y,
        col_w,
        70,
        "Phone",
        pat["phone"],
        label_font=label_font,
        value_font=value_font,
    )

    row_y += 90
    # Row 3: Patient address (3 cells)
    _draw_box(
        draw,
        MARGIN,
        row_y,
        col_w * 2,
        70,
        "5. PATIENT'S ADDRESS (No., Street)",
        pat["address_line1"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 2,
        row_y,
        col_w,
        70,
        "City",
        pat["city"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 3,
        row_y,
        col_w // 2,
        70,
        "State",
        pat["state"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 3 + col_w // 2,
        row_y,
        col_w // 2,
        70,
        "ZIP Code",
        pat["zip"],
        label_font=label_font,
        value_font=value_font,
    )

    row_y += 110
    # Diagnosis codes (ICD-10)
    draw.text(
        (MARGIN, row_y),
        "21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY (ICD-10-CM)",
        font=bold_font,
        fill="black",
    )
    row_y += 30
    for idx, dx in enumerate(enc["diagnosis_codes"], start=1):
        letter = chr(ord("A") + idx - 1)
        _draw_box(
            draw,
            MARGIN + (idx - 1) * (col_w // 2 + 10),
            row_y,
            col_w // 2,
            56,
            f"{letter}.",
            dx,
            label_font=label_font,
            value_font=value_font,
        )

    row_y += 80
    # Service lines (24)
    draw.text(
        (MARGIN, row_y),
        "24. SERVICES (DATE OF SERVICE  |  PLACE  |  CPT/HCPCS  |  MODIFIER  |  DX PTR  |  CHARGES  |  UNITS)",
        font=bold_font,
        fill="black",
    )
    row_y += 30
    column_xs = [MARGIN, MARGIN + 260, MARGIN + 400, MARGIN + 540, MARGIN + 740, MARGIN + 880, MARGIN + 1100, MARGIN + 1280]
    columns = ["From - To", "POS", "CPT/HCPCS", "MOD", "DX PTR", "$ CHARGES", "UNITS"]
    for x, col in zip(column_xs, columns, strict=False):
        draw.text((x + 8, row_y), col, font=label_font, fill="#444444")
    row_y += 30
    for line in enc["service_lines"]:
        cells = [
            f"{line['service_date_from']} - {line['service_date_to']}",
            line["place_of_service"],
            line["cpt_code"],
            line["modifier"] or "—",
            line["diagnosis_pointer"],
            f"${line['charge']:.2f}",
            str(line["units"]),
        ]
        for x, cell in zip(column_xs, cells, strict=False):
            draw.text((x + 8, row_y + 4), cell, font=value_font, fill="black")
        draw.rectangle(
            [(MARGIN, row_y), (column_xs[-1], row_y + 50)],
            outline="black",
            width=1,
        )
        row_y += 56

    row_y += 30
    # Footer: Federal Tax ID + Total + Billing provider info
    _draw_box(
        draw,
        MARGIN,
        row_y,
        col_w,
        70,
        "25. FEDERAL TAX I.D. NUMBER",
        prov["tax_id"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w,
        row_y,
        col_w,
        70,
        "28. TOTAL CHARGE",
        f"${enc['total_charge']:.2f}",
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 2,
        row_y,
        col_w * 2,
        70,
        "33. BILLING PROVIDER INFO & NPI",
        f"{bill['name']} — NPI {bill['npi']}",
        label_font=label_font,
        value_font=value_font,
    )

    row_y += 90
    _draw_box(
        draw,
        MARGIN,
        row_y,
        col_w * 2,
        70,
        "31. SIGNATURE OF PHYSICIAN OR SUPPLIER",
        prov["name"],
        label_font=label_font,
        value_font=value_font,
    )
    _draw_box(
        draw,
        MARGIN + col_w * 2,
        row_y,
        col_w * 2,
        70,
        "32. SERVICE FACILITY LOCATION",
        f"{prov['address_line1']}, {prov['city']} {prov['state']} {prov['zip']}",
        label_font=label_font,
        value_font=value_font,
    )


# ---------------------------------------------------------------------------
# Modality variants
# ---------------------------------------------------------------------------


def _apply_fax_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """Otsu-style binarisation + salt-pepper noise + slight rotation."""
    # Convert to grayscale, threshold at 128, then re-threshold with
    # value-aware noise sprinkled in.
    gray = image.convert("L")
    px = gray.load()
    w, h = gray.size
    # Salt-and-pepper noise (0.3% of pixels flipped).
    n_flips = int(w * h * 0.003)
    for _ in range(n_flips):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        px[x, y] = 0 if rng.random() < 0.5 else 255
    # Binarise — anything < 128 → 0, else 255. Approximates a 1-bit fax.
    bw = gray.point(lambda v: 0 if v < 140 else 255, mode="L")
    # Slight rotation (~1.5 degrees) with white fill.
    rotated = bw.rotate(rng.uniform(-1.5, 1.5), fillcolor=255, resample=Image.BICUBIC)
    # Tiny gaussian blur to mimic fax line bleed.
    return rotated.filter(ImageFilter.GaussianBlur(radius=0.6))


def _apply_handwriting_overlay(
    image: Image.Image,
    patient: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    """Overlay patient-completed fields in a handwriting-style font.

    Keeps the rest of the form printed; only the patient-name + DOB +
    signature regions get the Comic Sans overlay with slight jitter.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    hand_font = _load_font(COMIC_REGULAR, 28)

    pat = patient["patient"]
    full_name = f"{pat['last_name']}, {pat['first_name']} {pat['middle_name']}"

    # Hand-write the patient name on top of the printed value. Slight
    # ink-pen feel via jittered baseline and a darker ink colour.
    name_x = MARGIN + 8
    name_y = MARGIN + 140 + 90 + 30 + rng.randint(-3, 3)
    # White rectangle to hide the original printed text first.
    draw.rectangle(
        [(name_x - 4, name_y - 8), (name_x + 720, name_y + 32)],
        fill="white",
    )
    draw.text((name_x, name_y), full_name, font=hand_font, fill=(40, 40, 90))

    # Hand-write the DOB to the right.
    dob_x = MARGIN + ((PAGE_W - 2 * MARGIN) // 4) * 2 + 8
    dob_y = name_y
    draw.rectangle(
        [(dob_x - 4, dob_y - 8), (dob_x + 320, dob_y + 32)],
        fill="white",
    )
    draw.text((dob_x, dob_y), pat["dob"], font=hand_font, fill=(40, 40, 90))

    # Hand-written signature near the bottom — script-style, slanted.
    sig_y = PAGE_H - 220
    draw.text(
        (MARGIN + 12, sig_y),
        f"{pat['first_name']} {pat['last_name']}",
        font=hand_font,
        fill=(20, 20, 60),
    )
    return overlay


# ---------------------------------------------------------------------------
# PDF writing
# ---------------------------------------------------------------------------


def _png_to_pdf(image: Image.Image, output: Path) -> None:
    """Write a single-page PDF containing ``image``."""
    # PIL's PDF backend handles single-image documents fine and preserves
    # the configured DPI metadata so PyMuPDF reports correct page size.
    image.save(output, "PDF", resolution=DPI)


def _generate_clean(patient: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    _draw_form(canvas, patient)
    return canvas


def generate_demo_pdfs(
    *,
    patient_json: Path,
    output_dir: Path,
    seed: int = 20260517,
) -> list[Path]:
    """Generate the three demo PDFs. Returns the list of created paths."""
    patient = json.loads(patient_json.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # 1. Clean printed CMS-1500.
    clean = _generate_clean(patient)
    clean_path = output_dir / "cms1500_clean.pdf"
    _png_to_pdf(clean, clean_path)
    paths.append(clean_path)

    # 2. Faxed variant — Otsu binarisation + salt-pepper + slight rotation.
    faxed = _apply_fax_noise(clean.copy(), rng)
    faxed_path = output_dir / "cms1500_faxed.pdf"
    _png_to_pdf(faxed.convert("RGB"), faxed_path)
    paths.append(faxed_path)

    # 3. Handwritten variant — Comic Sans overlay on patient fields +
    # slight jitter. Build off a fresh clean canvas so the overlay sits
    # on a crisp background (operators would not fax-noise a handwritten
    # form too — handwriting is the modality, fax is a separate one).
    handwritten = _apply_handwriting_overlay(clean.copy(), patient, rng)
    handwritten_path = output_dir / "cms1500_handwritten.pdf"
    _png_to_pdf(handwritten, handwritten_path)
    paths.append(handwritten_path)

    return paths


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    patient_json = repo_root / "data" / "demo" / "synthea_patient.json"
    output_dir = repo_root / "data" / "demo"

    if not patient_json.exists():
        print(f"ERROR: patient data not found at {patient_json}", file=sys.stderr)
        return 1

    paths = generate_demo_pdfs(
        patient_json=patient_json,
        output_dir=output_dir,
    )
    print("Generated demo PDFs:")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.relative_to(repo_root)} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
