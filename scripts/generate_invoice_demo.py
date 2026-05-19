"""
Phase K — non-medical demo generator (synthetic invoice).

Produces two PDFs for the General-mode demo path:

* ``data/demo/invoice_clean.pdf`` — clean printed invoice.
* ``data/demo/invoice_faxed.pdf`` — same content, fax-style noise.

Lets us validate that the same Veridoc pipeline that does CMS-1500
→ FHIR R4 in Healthcare mode also handles non-medical documents in
General mode without medical-specific hallucinations (no phantom
``patient_name`` / ``diagnosis_code`` fields on an invoice).

All vendor + customer identifiers are synthetic.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PAGE_W = 1700
PAGE_H = 2200
DPI = 200
MARGIN = 100

ARIAL_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def _load_font(path: Path, size: int) -> ImageFont.ImageFont:
    try:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    except OSError:
        pass
    return ImageFont.load_default()


def _draw_invoice(canvas: Image.Image, data: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(ARIAL_BOLD, 50)
    h2_font = _load_font(ARIAL_BOLD, 30)
    body_font = _load_font(ARIAL_REGULAR, 26)
    label_font = _load_font(ARIAL_REGULAR, 18)

    # Vendor block (top-left, bold).
    y = MARGIN
    vendor = data["vendor"]
    draw.text((MARGIN, y), vendor["name"], font=h2_font, fill="black")
    y += 40
    draw.text((MARGIN, y), vendor["address_line1"], font=body_font, fill="black")
    y += 35
    draw.text(
        (MARGIN, y),
        f"{vendor['city']}, {vendor['state']} {vendor['zip']}",
        font=body_font,
        fill="black",
    )
    y += 35
    draw.text((MARGIN, y), f"Phone: {vendor['phone']}", font=body_font, fill="black")
    y += 35
    draw.text((MARGIN, y), f"Tax ID: {vendor['tax_id']}", font=body_font, fill="black")

    # INVOICE title + metadata (top-right).
    invoice_x = PAGE_W - MARGIN - 500
    draw.text((invoice_x, MARGIN), "INVOICE", font=title_font, fill="black")
    y = MARGIN + 70
    inv = data["invoice"]
    draw.text((invoice_x, y), f"Invoice #: {inv['number']}", font=body_font, fill="black")
    y += 35
    draw.text((invoice_x, y), f"Date: {inv['date']}", font=body_font, fill="black")
    y += 35
    draw.text((invoice_x, y), f"Due: {inv['due_date']}", font=body_font, fill="black")
    y += 35
    draw.text((invoice_x, y), f"PO #: {inv['po_number']}", font=body_font, fill="black")

    # Bill-to block.
    y = MARGIN + 260
    draw.rectangle([(MARGIN, y), (MARGIN + 700, y + 200)], outline="black", width=2)
    draw.text((MARGIN + 10, y + 10), "BILL TO", font=label_font, fill="#555")
    y_in = y + 40
    customer = data["customer"]
    draw.text((MARGIN + 10, y_in), customer["name"], font=body_font, fill="black")
    y_in += 35
    draw.text((MARGIN + 10, y_in), customer["address_line1"], font=body_font, fill="black")
    y_in += 35
    draw.text(
        (MARGIN + 10, y_in),
        f"{customer['city']}, {customer['state']} {customer['zip']}",
        font=body_font,
        fill="black",
    )
    y_in += 35
    draw.text((MARGIN + 10, y_in), f"Attn: {customer['contact']}", font=body_font, fill="black")

    # Line items table.
    table_y = y + 230
    draw.rectangle(
        [(MARGIN, table_y), (PAGE_W - MARGIN, table_y + 50)], fill="#222"
    )
    header_xs = [MARGIN + 20, MARGIN + 700, MARGIN + 1000, MARGIN + 1240, MARGIN + 1430]
    for x, h in zip(
        header_xs, ["Description", "SKU", "Qty", "Unit Price", "Amount"], strict=False
    ):
        draw.text((x, table_y + 12), h, font=label_font, fill="white")

    row_y = table_y + 55
    for line in data["line_items"]:
        cells = [
            line["description"],
            line["sku"],
            str(line["quantity"]),
            f"${line['unit_price']:.2f}",
            f"${line['amount']:.2f}",
        ]
        for x, c in zip(header_xs, cells, strict=False):
            draw.text((x, row_y + 12), c, font=body_font, fill="black")
        draw.line(
            [(MARGIN, row_y + 55), (PAGE_W - MARGIN, row_y + 55)],
            fill="#cccccc",
            width=1,
        )
        row_y += 60

    # Totals block (bottom right).
    totals_x = PAGE_W - MARGIN - 600
    totals_y = row_y + 50
    totals = data["totals"]
    rows = [
        ("Subtotal", f"${totals['subtotal']:.2f}"),
        (f"Tax ({totals['tax_rate']:.1%})", f"${totals['tax']:.2f}"),
        ("Total Due", f"${totals['total']:.2f}"),
    ]
    for i, (label, val) in enumerate(rows):
        is_total = label == "Total Due"
        font = h2_font if is_total else body_font
        draw.text((totals_x, totals_y + i * 45), label, font=font, fill="black")
        draw.text(
            (totals_x + 400, totals_y + i * 45),
            val,
            font=font,
            fill="black",
        )
    # Underline the total row.
    line_y = totals_y + len(rows) * 45 + 5
    draw.line(
        [(totals_x, line_y), (totals_x + 580, line_y)], fill="black", width=2
    )

    # Footer payment terms.
    footer_y = PAGE_H - MARGIN - 100
    draw.text(
        (MARGIN, footer_y),
        f"Payment Terms: {data['invoice']['terms']}",
        font=body_font,
        fill="black",
    )
    draw.text(
        (MARGIN, footer_y + 35),
        f"Remit to: {vendor['name']} - {vendor['address_line1']}",
        font=body_font,
        fill="black",
    )


def _apply_fax_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """Otsu binarisation + salt-pepper + slight rotation."""
    gray = image.convert("L")
    px = gray.load()
    w, h = gray.size
    n_flips = int(w * h * 0.003)
    for _ in range(n_flips):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        px[x, y] = 0 if rng.random() < 0.5 else 255
    bw = gray.point(lambda v: 0 if v < 140 else 255, mode="L")
    rotated = bw.rotate(rng.uniform(-1.5, 1.5), fillcolor=255, resample=Image.BICUBIC)
    return rotated.filter(ImageFilter.GaussianBlur(radius=0.6))


def _png_to_pdf(image: Image.Image, output: Path) -> None:
    image.save(output, "PDF", resolution=DPI)


def generate_invoice_pdfs(
    *,
    data_path: Path,
    output_dir: Path,
    seed: int = 20260517,
) -> list[Path]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    _draw_invoice(canvas, data)

    paths: list[Path] = []
    clean = output_dir / "invoice_clean.pdf"
    _png_to_pdf(canvas, clean)
    paths.append(clean)

    faxed_img = _apply_fax_noise(canvas.copy(), rng).convert("RGB")
    faxed = output_dir / "invoice_faxed.pdf"
    _png_to_pdf(faxed_img, faxed)
    paths.append(faxed)

    return paths


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "demo" / "invoice_data.json"
    out = repo_root / "data" / "demo"

    if not data_path.exists():
        print(f"ERROR: invoice data not found at {data_path}", file=sys.stderr)
        return 1

    paths = generate_invoice_pdfs(data_path=data_path, output_dir=out)
    print("Generated invoice PDFs:")
    for p in paths:
        print(f"  - {p.relative_to(repo_root)} ({p.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
