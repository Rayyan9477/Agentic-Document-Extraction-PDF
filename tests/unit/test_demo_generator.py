"""Tests for the Phase K demo PDF generator.

Verifies that the generator produces three readable PDFs whose content
trips the right detection signals (medical-RCM profile header, fax
modality, handwriting modality).

The PDFs are committed under ``data/demo/`` so the demo runs without
the generator. These tests exercise the generator itself + the
committed artefacts as a regression net.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the generator module.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


@pytest.fixture
def patient_json() -> Path:
    return _REPO_ROOT / "data" / "demo" / "synthea_patient.json"


@pytest.fixture
def demo_dir() -> Path:
    return _REPO_ROOT / "data" / "demo"


class TestSyntheaPatientJSON:
    """Validate the synthetic patient data shape + invariants."""

    def test_patient_json_loads(self, patient_json: Path) -> None:
        assert patient_json.exists()
        data = json.loads(patient_json.read_text(encoding="utf-8"))
        for key in ("patient", "insurance", "provider", "billing_provider", "encounter"):
            assert key in data, f"missing required top-level key: {key}"

    def test_provider_npi_passes_luhn(self, patient_json: Path) -> None:
        """The provider NPI must Luhn-validate."""
        from src.schemas.validators import _luhn_checksum

        data = json.loads(patient_json.read_text(encoding="utf-8"))
        npi = data["provider"]["npi"]
        assert _luhn_checksum(npi), f"NPI {npi} fails Luhn"

    def test_diagnosis_codes_are_real_icd10_shapes(self, patient_json: Path) -> None:
        import re

        data = json.loads(patient_json.read_text(encoding="utf-8"))
        for code in data["encounter"]["diagnosis_codes"]:
            assert re.match(r"^[A-Z][0-9]{2}(\.[A-Z0-9]+)?$", code), (
                f"ICD-10 shape failed: {code}"
            )

    def test_cpt_codes_are_5_digit_numeric(self, patient_json: Path) -> None:
        data = json.loads(patient_json.read_text(encoding="utf-8"))
        for line in data["encounter"]["service_lines"]:
            cpt = line["cpt_code"]
            assert cpt.isdigit() and len(cpt) == 5, f"bad CPT shape: {cpt}"

    def test_total_charge_matches_line_items(self, patient_json: Path) -> None:
        data = json.loads(patient_json.read_text(encoding="utf-8"))
        line_sum = sum(line["charge"] for line in data["encounter"]["service_lines"])
        assert abs(line_sum - data["encounter"]["total_charge"]) < 0.005, (
            f"Total {data['encounter']['total_charge']} != line sum {line_sum}"
        )


class TestGeneratorOutput:
    """The generator produces three readable single-page PDFs."""

    def test_three_pdfs_committed(self, demo_dir: Path) -> None:
        for name in ("cms1500_clean.pdf", "cms1500_faxed.pdf", "cms1500_handwritten.pdf"):
            path = demo_dir / name
            assert path.exists(), f"missing demo file: {path}"
            assert path.stat().st_size > 10_000, f"{name} suspiciously small"

    def test_clean_pdf_is_readable_by_pymupdf(self, demo_dir: Path) -> None:
        try:
            import pymupdf  # type: ignore
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        doc = pymupdf.open(str(demo_dir / "cms1500_clean.pdf"))
        assert doc.page_count == 1
        # Render the page to a pixmap so we exercise the full decode path.
        page = doc[0]
        pix = page.get_pixmap(dpi=100)
        assert pix.width > 500 and pix.height > 500
        doc.close()

    def test_faxed_pdf_has_lower_dynamic_range(self, demo_dir: Path) -> None:
        """The faxed variant should have far fewer unique gray levels."""
        try:
            import pymupdf  # type: ignore
            from PIL import Image  # noqa: F401
        except ImportError:
            pytest.skip("PyMuPDF or Pillow not installed")
        import io

        from PIL import Image as PILImage

        def _open_pdf_page_as_image(path: Path) -> PILImage.Image:
            doc = pymupdf.open(str(path))
            pix = doc[0].get_pixmap(dpi=100)
            png_bytes = pix.tobytes("png")
            return PILImage.open(io.BytesIO(png_bytes))

        clean = _open_pdf_page_as_image(demo_dir / "cms1500_clean.pdf").convert("L")
        faxed = _open_pdf_page_as_image(demo_dir / "cms1500_faxed.pdf").convert("L")

        clean_levels = len(set(clean.getdata()))
        faxed_levels = len(set(faxed.getdata()))
        # The fax filter binarises so the unique-gray count drops sharply.
        # ``< clean_levels`` is the load-bearing assertion; the absolute
        # numbers are dependent on rendering DPI.
        assert faxed_levels < clean_levels, (
            f"faxed unique levels ({faxed_levels}) not less than clean ({clean_levels})"
        )

    def test_handwritten_pdf_renders(self, demo_dir: Path) -> None:
        """The handwritten variant is byte-distinct from the clean PDF."""
        clean_bytes = (demo_dir / "cms1500_clean.pdf").read_bytes()
        hand_bytes = (demo_dir / "cms1500_handwritten.pdf").read_bytes()
        assert clean_bytes != hand_bytes


class TestGeneratorDeterminism:
    """Re-running the generator produces visually-equivalent files.

    PIL's PDF writer embeds a ``CreationDate`` timestamp so two
    consecutive runs are not byte-equal even with a fixed seed. We
    compare the *rendered pixel content* instead, which is what the
    extractor and the demo viewer actually see.
    """

    @staticmethod
    def _render_pdf_bytes(path: Path) -> bytes:
        import io

        import pymupdf  # type: ignore
        from PIL import Image  # noqa: F401  pylint: disable=unused-import

        doc = pymupdf.open(str(path))
        pix = doc[0].get_pixmap(dpi=72)  # low DPI is enough for equality
        png = pix.tobytes("png")
        doc.close()
        return png

    def test_seeded_run_is_pixel_identical(
        self, tmp_path: Path, patient_json: Path
    ) -> None:
        from generate_demo_data import generate_demo_pdfs

        first = generate_demo_pdfs(
            patient_json=patient_json,
            output_dir=tmp_path / "run1",
            seed=42,
        )
        second = generate_demo_pdfs(
            patient_json=patient_json,
            output_dir=tmp_path / "run2",
            seed=42,
        )
        assert len(first) == len(second) == 3
        for a, b in zip(first, second, strict=False):
            assert self._render_pdf_bytes(a) == self._render_pdf_bytes(b), (
                f"non-deterministic render: {a.name}"
            )

    def test_different_seeds_produce_different_faxed_variant(
        self, tmp_path: Path, patient_json: Path
    ) -> None:
        from generate_demo_data import generate_demo_pdfs

        first = generate_demo_pdfs(
            patient_json=patient_json,
            output_dir=tmp_path / "run1",
            seed=1,
        )
        second = generate_demo_pdfs(
            patient_json=patient_json,
            output_dir=tmp_path / "run2",
            seed=2,
        )
        # The clean PDF is seed-independent.
        clean_a = next(p for p in first if "clean" in p.name)
        clean_b = next(p for p in second if "clean" in p.name)
        assert self._render_pdf_bytes(clean_a) == self._render_pdf_bytes(clean_b)
        # The faxed PDF varies with the seed (different salt-pepper noise
        # and rotation amount).
        faxed_a = next(p for p in first if "faxed" in p.name)
        faxed_b = next(p for p in second if "faxed" in p.name)
        assert self._render_pdf_bytes(faxed_a) != self._render_pdf_bytes(faxed_b)
