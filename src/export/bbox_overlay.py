"""
WS-8: bounding-box overlay PNGs for visual grounding.

For each page that yielded extracted fields with bbox coordinates,
this module renders a copy of the page with translucent rectangles
drawn over each extracted region. Rectangle colour encodes the
field's confidence:

    * **green**  — confidence ≥ 0.85 (high; auto-accept band)
    * **yellow** — 0.50 ≤ confidence < 0.85 (medium; retry band)
    * **red**    — confidence < 0.50 (low; human-review band)

Overlays are written to ``<output_dir>/overlays/page_NN.png``. Pages
without bbox-tagged fields are skipped (no overlay file generated)
so the directory stays sparse.

Inputs:
    * ``page_images``: list of ``PageImage`` instances or their
      ``serialize_page_image`` dict form (image_bytes + width/height).
    * ``field_metadata``: ``dict[field_name, FieldMetadata]`` (or its
      serialised dict form) where each metadata may carry a
      ``BoundingBoxCoords``.

Output:
    * ``OverlayResult`` listing the per-page PNG paths.

The renderer is pure-Python via Pillow; no OpenCV dependency. It
gracefully handles missing pixel coordinates by deriving them from
normalised coordinates + the page's actual width / height.

Visual contract:
    * Rectangle outline is 3 px wide.
    * Fill is 64 / 255 alpha (translucent so the underlying glyph is
      still readable).
    * A small label is drawn at the top-left of the rectangle showing
      ``field_name (cc%)`` where ``cc`` is the confidence percentage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.config import get_logger
from src.pipeline.state import BoundingBoxCoords


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Confidence palette
# ---------------------------------------------------------------------------

# Colours given as RGB; alpha is applied per-call below.
_COLOR_HIGH = (76, 175, 80)     # Material green-500
_COLOR_MEDIUM = (255, 193, 7)   # Material amber-500
_COLOR_LOW = (244, 67, 54)      # Material red-500
_COLOR_UNKNOWN = (158, 158, 158)  # Material grey-500

_OUTLINE_ALPHA = 220
_FILL_ALPHA = 64
_LABEL_BG_ALPHA = 200
_OUTLINE_WIDTH_PX = 3


def _confidence_color(confidence: float | None) -> tuple[int, int, int]:
    """Map confidence → palette colour. ``None`` → grey."""
    if confidence is None:
        return _COLOR_UNKNOWN
    if confidence >= 0.85:
        return _COLOR_HIGH
    if confidence >= 0.50:
        return _COLOR_MEDIUM
    return _COLOR_LOW


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OverlayPage:
    """A single rendered overlay PNG."""

    page_number: int
    output_path: Path
    field_count: int  # number of bboxes drawn on this page


@dataclass(slots=True)
class OverlayResult:
    """Summary of an overlay render pass."""

    overlay_dir: Path
    pages: list[OverlayPage]

    @property
    def total_fields(self) -> int:
        return sum(p.field_count for p in self.pages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_overlays(
    page_images: list[Any],
    field_metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    label_fields: bool = True,
) -> OverlayResult:
    """Render confidence-coloured bbox overlays for every page that has fields.

    Args:
        page_images: List of ``PageImage`` instances or their dict form.
            Each must expose ``page_number``, ``width``, ``height``, and
            either ``image_bytes`` or ``to_pil_image()``.
        field_metadata: Mapping of field name → ``FieldMetadata`` (or
            its dict form). Fields without a bbox are skipped silently.
        output_dir: Directory where overlay PNGs are written. Created if
            missing. Output files are ``page_NN.png``, 1-indexed.
        label_fields: When True, draw the field name + confidence
            percentage in a small label above each bbox. Set False for
            cleaner heat-map style output.

    Returns:
        ``OverlayResult`` with one ``OverlayPage`` per page that
        produced an overlay. Pages without bbox-tagged fields are
        absent from the list rather than emitting blank PNGs.
    """
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Group fields by page so we render each page once.
    fields_by_page: dict[int, list[tuple[str, BoundingBoxCoords, float | None]]] = {}
    for field_name, meta in field_metadata.items():
        bbox = _extract_bbox(meta)
        if bbox is None:
            continue
        confidence = _extract_confidence(meta)
        fields_by_page.setdefault(bbox.page, []).append((field_name, bbox, confidence))

    rendered: list[OverlayPage] = []
    for page in page_images:
        page_no = _page_number(page)
        if page_no not in fields_by_page:
            continue
        try:
            overlay_path = _render_one(
                page=page,
                field_specs=fields_by_page[page_no],
                output_path=overlay_dir / f"page_{page_no:02d}.png",
                label_fields=label_fields,
            )
        except Exception as exc:
            logger.warning(
                "bbox_overlay_render_failed",
                page=page_no,
                error=str(exc),
            )
            continue
        rendered.append(
            OverlayPage(
                page_number=page_no,
                output_path=overlay_path,
                field_count=len(fields_by_page[page_no]),
            )
        )

    logger.info(
        "bbox_overlays_rendered",
        page_count=len(rendered),
        field_count=sum(p.field_count for p in rendered),
        output_dir=str(overlay_dir),
    )
    return OverlayResult(overlay_dir=overlay_dir, pages=rendered)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_one(
    *,
    page: Any,
    field_specs: list[tuple[str, BoundingBoxCoords, float | None]],
    output_path: Path,
    label_fields: bool,
) -> Path:
    """Render a single page with its bbox overlays.

    Composes a translucent overlay layer onto the source image so the
    underlying document is still legible. Each rectangle is painted
    with both an outline (high alpha) and a fill (low alpha).
    """
    base = _to_pil(page).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    page_w, page_h = base.size
    font = _load_font(size=max(12, page_h // 80))

    for field_name, bbox, confidence in field_specs:
        rect = _resolve_pixel_rect(bbox, page_w, page_h)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        rgb = _confidence_color(confidence)

        # Translucent fill
        draw.rectangle((x0, y0, x1, y1), fill=(*rgb, _FILL_ALPHA))
        # Solid-ish outline
        draw.rectangle(
            (x0, y0, x1, y1),
            outline=(*rgb, _OUTLINE_ALPHA),
            width=_OUTLINE_WIDTH_PX,
        )

        if label_fields and font is not None:
            label = _format_label(field_name, confidence)
            _draw_label(draw, font, label, anchor=(x0, y0), color=rgb)

    composed = Image.alpha_composite(base, overlay)
    composed.convert("RGB").save(output_path, format="PNG")
    return output_path


def _format_label(field_name: str, confidence: float | None) -> str:
    if confidence is None:
        return field_name
    return f"{field_name} ({int(round(confidence * 100))}%)"


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    label: str,
    *,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = anchor
    # Measure with textbbox (Pillow ≥ 10) for tight wrapping.
    try:
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w = len(label) * 6
        text_h = 12

    pad_x, pad_y = 4, 2
    bg_rect = (x, max(0, y - text_h - 2 * pad_y), x + text_w + 2 * pad_x, y)
    draw.rectangle(bg_rect, fill=(*color, _LABEL_BG_ALPHA))
    draw.text(
        (bg_rect[0] + pad_x, bg_rect[1] + pad_y),
        label,
        fill=(255, 255, 255, 255),
        font=font,
    )


def _load_font(*, size: int) -> ImageFont.ImageFont | None:
    """Try to load a TrueType font; fall back to PIL default.

    The default bitmap font is fixed-size and ignores the ``size``
    argument, but it's universally available — important for
    headless / containerised deployments without a system font cache.
    """
    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover - PIL guarantees a default
        return None


def _resolve_pixel_rect(
    bbox: BoundingBoxCoords,
    page_w: int,
    page_h: int,
) -> tuple[int, int, int, int] | None:
    """Return ``(x0, y0, x1, y1)`` in pixel space.

    Prefers the bbox's pre-computed pixel coordinates; falls back to
    deriving from the normalised values + the page's actual size.
    Drops degenerate rectangles (zero or negative area).
    """
    if bbox.pixel_width > 0 and bbox.pixel_height > 0:
        x0 = bbox.pixel_x
        y0 = bbox.pixel_y
        x1 = x0 + bbox.pixel_width
        y1 = y0 + bbox.pixel_height
    else:
        x0 = int(bbox.x * page_w)
        y0 = int(bbox.y * page_h)
        x1 = x0 + max(1, int(bbox.width * page_w))
        y1 = y0 + max(1, int(bbox.height * page_h))

    # Clamp to page bounds + reject empty rectangles.
    x0 = max(0, min(x0, page_w - 1))
    y0 = max(0, min(y0, page_h - 1))
    x1 = max(0, min(x1, page_w))
    y1 = max(0, min(y1, page_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _to_pil(page: Any) -> Image.Image:
    """Coerce a PageImage / dict / raw bytes to a PIL Image."""
    if hasattr(page, "to_pil_image"):
        return page.to_pil_image()
    if isinstance(page, dict):
        if "image_bytes" in page:
            return Image.open(io.BytesIO(page["image_bytes"]))
        # Some serialised forms store base64 instead.
        if "image_base64" in page:
            import base64

            return Image.open(io.BytesIO(base64.b64decode(page["image_base64"])))
    raise ValueError("Unsupported page image type for overlay rendering")


def _page_number(page: Any) -> int:
    if hasattr(page, "page_number"):
        return int(page.page_number)
    if isinstance(page, dict):
        return int(page.get("page_number", 1))
    return 1


def _extract_bbox(meta: Any) -> BoundingBoxCoords | None:
    """Pull a ``BoundingBoxCoords`` out of a FieldMetadata or its dict form."""
    if isinstance(meta, BoundingBoxCoords):
        return meta
    bbox_attr = getattr(meta, "bbox", None)
    if isinstance(bbox_attr, BoundingBoxCoords):
        return bbox_attr
    if isinstance(meta, dict):
        bbox_data = meta.get("bbox")
        if isinstance(bbox_data, dict):
            return BoundingBoxCoords.from_dict(bbox_data)
        if isinstance(bbox_data, BoundingBoxCoords):
            return bbox_data
    return None


def _extract_confidence(meta: Any) -> float | None:
    """Pull a confidence score from FieldMetadata / dict / value-envelope."""
    if isinstance(meta, dict):
        c = meta.get("confidence")
        if isinstance(c, (int, float)):
            return float(c)
    c = getattr(meta, "confidence", None)
    if isinstance(c, (int, float)):
        return float(c)
    return None


__all__ = [
    "OverlayPage",
    "OverlayResult",
    "render_overlays",
]
