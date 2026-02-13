"""
EDI X12 processor for 837 (claims) and 835 (remittance) transactions.

Unlike image-based processors, EDI files are text-based and produce
structured data directly. The processor also renders a visual
representation as PageImage for the VLM pipeline.
"""

import io
import re
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.config import get_logger
from src.preprocessing.base_processor import (
    BaseFileProcessor,
    FileProcessingError,
    FileValidationError,
)
from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PDFMetadata,
    PageImage,
    ProcessingResult,
)


logger = get_logger(__name__)

# Rendering constants
PAGE_WIDTH_PX = 2550
PAGE_HEIGHT_PX = 3300
MARGIN_PX = 100
LINE_HEIGHT = 32
FONT_SIZE = 22
MAX_FILE_SIZE_MB = 50


class EDIProcessor(BaseFileProcessor):
    """
    Process EDI X12 837/835 transaction files.

    Parses segment-delimited EDI content and renders a human-readable
    visual representation for VLM extraction. Also stores the parsed
    segment data in text_content for hybrid processing.
    """

    def __init__(self, dpi: int = 300) -> None:
        self._dpi = dpi

    def validate(self, file_path: Path) -> None:
        """Validate EDI file."""
        if not file_path.exists():
            raise FileValidationError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in (".edi", ".x12", ".835", ".837"):
            raise FileValidationError(f"Not an EDI file: {suffix}")

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise FileValidationError(
                f"File size {size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB"
            )

        # Check for ISA header
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            header = f.read(3)
            if header != "ISA":
                raise FileValidationError(
                    "EDI file must start with ISA segment"
                )

    def process(self, file_path: Path) -> ProcessingResult:
        """Parse EDI file and render as page images."""
        start_time = time.monotonic()
        self.validate(file_path)

        file_hash = self._compute_file_hash(file_path)
        processing_id = f"edi_{secrets.token_hex(8)}"
        warnings: list[str] = []

        raw_content = file_path.read_text(encoding="utf-8", errors="replace")

        # Detect delimiters from ISA segment
        element_sep, segment_sep = self._detect_delimiters(raw_content)

        # Parse segments
        segments = self._parse_segments(raw_content, element_sep, segment_sep)

        # Detect transaction type
        transaction_type = self._detect_transaction_type(segments)

        # Build human-readable lines
        readable_lines = self._segments_to_readable(segments, element_sep)

        # Render as pages
        pages = self._render_pages(readable_lines)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)

        metadata = PDFMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            file_hash=file_hash,
            page_count=len(pages),
            title=f"EDI {transaction_type}",
            author=None,
            subject=transaction_type,
            keywords="EDI,X12",
            creator="EDIProcessor",
            producer="custom",
            creation_date=datetime.now(UTC),
            modification_date=None,
            pdf_version="N/A",
            is_encrypted=False,
            has_forms=False,
            has_annotations=False,
            processing_id=processing_id,
        )

        logger.info(
            "edi_processing_complete",
            file=file_path.name,
            pages=len(pages),
            segments=len(segments),
            transaction_type=transaction_type,
            time_ms=processing_time_ms,
        )

        return ProcessingResult(
            metadata=metadata,
            pages=pages,
            processing_time_ms=processing_time_ms,
            warnings=warnings,
        )

    def _detect_delimiters(self, content: str) -> tuple[str, str]:
        """Detect element and segment delimiters from ISA header."""
        if len(content) < 106:
            return ("*", "~")

        # ISA segment is fixed-length: element separator is at position 3
        element_sep = content[3]
        # Segment terminator is at position 105
        segment_sep = content[105]

        return (element_sep, segment_sep)

    def _parse_segments(
        self, content: str, element_sep: str, segment_sep: str,
    ) -> list[list[str]]:
        """Parse EDI content into segments (each segment is a list of elements)."""
        # Remove line breaks
        content = content.replace("\n", "").replace("\r", "")

        raw_segments = content.split(segment_sep)
        segments: list[list[str]] = []

        for seg in raw_segments:
            seg = seg.strip()
            if seg:
                elements = seg.split(element_sep)
                segments.append(elements)

        return segments

    def _detect_transaction_type(self, segments: list[list[str]]) -> str:
        """Detect 837/835 transaction type from ST segment."""
        for seg in segments:
            if seg and seg[0] == "ST":
                code = seg[1] if len(seg) > 1 else ""
                if code == "837":
                    return "837 (Health Care Claim)"
                if code == "835":
                    return "835 (Health Care Payment/Remittance)"
                return f"X12 {code}"
        return "Unknown EDI"

    def _segments_to_readable(
        self, segments: list[list[str]], element_sep: str,
    ) -> list[str]:
        """Convert parsed segments to human-readable lines."""
        lines: list[str] = []

        segment_labels: dict[str, str] = {
            "ISA": "Interchange Control Header",
            "GS": "Functional Group Header",
            "ST": "Transaction Set Header",
            "BHT": "Beginning of Hierarchical Transaction",
            "HL": "Hierarchical Level",
            "NM1": "Individual/Organization Name",
            "N3": "Address",
            "N4": "City/State/ZIP",
            "CLM": "Claim Information",
            "DTP": "Date/Time Period",
            "SV1": "Professional Service",
            "SV2": "Institutional Service",
            "REF": "Reference Identification",
            "AMT": "Monetary Amount",
            "CLP": "Claim Payment Information",
            "SVC": "Service Payment Information",
            "PLB": "Provider Level Adjustment",
            "SE": "Transaction Set Trailer",
            "GE": "Functional Group Trailer",
            "IEA": "Interchange Control Trailer",
        }

        for seg in segments:
            if not seg:
                continue
            seg_id = seg[0]
            label = segment_labels.get(seg_id, seg_id)
            elements = element_sep.join(seg)
            lines.append(f"[{label}]")
            lines.append(f"  {elements}")
            lines.append("")

        return lines

    def _render_pages(self, lines: list[str]) -> list[PageImage]:
        """Render readable lines into page images."""
        pages: list[PageImage] = []
        usable_height = PAGE_HEIGHT_PX - 2 * MARGIN_PX
        lines_per_page = usable_height // LINE_HEIGHT

        for chunk_start in range(0, max(len(lines), 1), lines_per_page):
            chunk = lines[chunk_start : chunk_start + lines_per_page]
            page = self._render_page(chunk, len(pages) + 1)
            pages.append(page)

        return pages

    def _render_page(self, lines: list[str], page_number: int) -> PageImage:
        """Render a single page of EDI content."""
        img = Image.new("RGB", (PAGE_WIDTH_PX, PAGE_HEIGHT_PX), "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("consola.ttf", FONT_SIZE)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", FONT_SIZE)
            except OSError:
                font = ImageFont.load_default()

        y = MARGIN_PX
        text_parts: list[str] = []

        for line in lines:
            # Color-code segment labels
            if line.startswith("[") and line.endswith("]"):
                draw.text((MARGIN_PX, y), line, fill="#0066CC", font=font)
            else:
                draw.text((MARGIN_PX, y), line, fill="black", font=font)
            text_parts.append(line)
            y += LINE_HEIGHT

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        orientation = self._detect_orientation(PAGE_WIDTH_PX, PAGE_HEIGHT_PX)

        return PageImage(
            page_number=page_number,
            image_bytes=png_bytes,
            width=PAGE_WIDTH_PX,
            height=PAGE_HEIGHT_PX,
            dpi=self._dpi,
            orientation=orientation,
            original_width_pts=612.0,
            original_height_pts=792.0,
            has_text=True,
            has_images=False,
            rotation=0,
            text_content="\n".join(text_parts),
        )
