"""
Image enhancement module using OpenCV.

Provides document image enhancement including deskewing, denoising,
and contrast enhancement (CLAHE) to improve VLM extraction accuracy.
"""

import io
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.config import get_logger, get_settings
from src.preprocessing.pdf_processor import DocumentOrientation, PageImage


logger = get_logger(__name__)


class EnhancementError(Exception):
    """Base exception for image enhancement errors."""


class DeskewError(EnhancementError):
    """Raised when deskewing fails."""


class DenoiseError(EnhancementError):
    """Raised when denoising fails."""


class ContrastError(EnhancementError):
    """Raised when contrast enhancement fails."""


class EnhancementType(str, Enum):
    """Types of image enhancements applied."""

    DESKEW = "deskew"
    DENOISE = "denoise"
    CLAHE = "clahe"
    BINARIZATION = "binarization"
    MORPHOLOGICAL = "morphological"
    DESPECKLE = "despeckle"
    REORIENT = "reorient"


@dataclass(frozen=True, slots=True)
class EnhancementMetrics:
    """
    Metrics from image enhancement operations.

    Attributes:
        original_variance: Laplacian variance of original image (sharpness indicator).
        enhanced_variance: Laplacian variance after enhancement.
        skew_angle: Detected skew angle in degrees.
        skew_corrected: Whether skew was corrected.
        noise_reduction_ratio: Ratio of noise reduction achieved.
        contrast_improvement: Contrast improvement factor.
    """

    original_variance: float
    enhanced_variance: float
    skew_angle: float
    skew_corrected: bool
    noise_reduction_ratio: float
    contrast_improvement: float

    @property
    def sharpness_improvement(self) -> float:
        """Calculate sharpness improvement ratio."""
        if self.original_variance == 0:
            return 0.0
        return self.enhanced_variance / self.original_variance

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "original_variance": self.original_variance,
            "enhanced_variance": self.enhanced_variance,
            "skew_angle": self.skew_angle,
            "skew_corrected": self.skew_corrected,
            "noise_reduction_ratio": self.noise_reduction_ratio,
            "contrast_improvement": self.contrast_improvement,
            "sharpness_improvement": self.sharpness_improvement,
        }


@dataclass(slots=True)
class EnhancementResult:
    """
    Result of image enhancement operation.

    Attributes:
        page_image: Enhanced PageImage.
        original_page: Original PageImage before enhancement.
        enhancements_applied: List of enhancement types applied.
        metrics: Enhancement metrics and statistics.
        processing_time_ms: Time taken for enhancement in milliseconds.
    """

    page_image: PageImage
    original_page: PageImage
    enhancements_applied: list[EnhancementType] = field(default_factory=list)
    metrics: EnhancementMetrics | None = None
    processing_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "page_number": self.page_image.page_number,
            "enhancements_applied": [e.value for e in self.enhancements_applied],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "processing_time_ms": self.processing_time_ms,
            "original_size_kb": self.original_page.size_kb,
            "enhanced_size_kb": self.page_image.size_kb,
        }


class ImageEnhancer:
    """
    Document image enhancement using OpenCV.

    Applies various enhancement techniques to improve document image quality
    for better VLM extraction accuracy:
    - Deskewing: Corrects rotated/skewed documents
    - Denoising: Reduces noise while preserving edges
    - CLAHE: Adaptive contrast enhancement

    Example:
        enhancer = ImageEnhancer()
        result = enhancer.enhance(page_image)
        enhanced_page = result.page_image
    """

    def __init__(
        self,
        enable_deskew: bool | None = None,
        enable_denoise: bool | None = None,
        enable_contrast: bool | None = None,
        clahe_clip_limit: float | None = None,
        clahe_tile_size: int | None = None,
        denoise_strength: int | None = None,
        deskew_max_angle: float | None = None,
    ) -> None:
        """
        Initialize the image enhancer.

        Args:
            enable_deskew: Enable automatic deskewing. Defaults to settings.
            enable_denoise: Enable noise reduction. Defaults to settings.
            enable_contrast: Enable CLAHE contrast enhancement. Defaults to settings.
            clahe_clip_limit: CLAHE clip limit. Defaults to settings.
            clahe_tile_size: CLAHE tile grid size. Defaults to settings.
            denoise_strength: Denoising filter strength. Defaults to settings.
            deskew_max_angle: Maximum angle for deskew detection. Defaults to settings.
        """
        settings = get_settings()

        self._enable_deskew = (
            enable_deskew if enable_deskew is not None else settings.image.enable_deskew
        )
        self._enable_denoise = (
            enable_denoise if enable_denoise is not None else settings.image.enable_denoise
        )
        self._enable_contrast = (
            enable_contrast if enable_contrast is not None else settings.image.enable_contrast
        )
        self._clahe_clip_limit = clahe_clip_limit or settings.image.clahe_clip_limit
        self._clahe_tile_size = clahe_tile_size or settings.image.clahe_tile_size
        self._denoise_strength = denoise_strength or settings.image.denoise_strength
        self._deskew_max_angle = deskew_max_angle or settings.image.deskew_max_angle

        # Pre-create CLAHE object for reuse
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit,
            tileGridSize=(self._clahe_tile_size, self._clahe_tile_size),
        )

        logger.debug(
            "image_enhancer_initialized",
            enable_deskew=self._enable_deskew,
            enable_denoise=self._enable_denoise,
            enable_contrast=self._enable_contrast,
            clahe_clip_limit=self._clahe_clip_limit,
            denoise_strength=self._denoise_strength,
        )

    def enhance(
        self,
        page_image: PageImage,
        *,
        modes: list[str] | None = None,
    ) -> EnhancementResult:
        """
        Apply enhancements to a page image, optionally specialised per modality.

        Args:
            page_image: PageImage to enhance.
            modes: WS-3 modality hints (``"fax"``, ``"handwritten"``,
                ``"printed"``, ``"table"``, ``"form"``, ``"visual"``).
                Multiple modes may be active simultaneously. When omitted
                or empty, the default deskew + denoise + CLAHE pipeline
                runs (legacy behaviour). Mode-specific overrides:

                * ``"fax"`` → deskew + Otsu binarization + morphological
                  opening; CLAHE and Non-Local-Means denoise are skipped
                  (they over-process 1-bit fax artefacts).
                * ``"handwritten"`` → deskew + light denoise; CLAHE is
                  skipped (it distorts pen-stroke gradients).
                * ``"visual"`` → deskew only; preserves natural gradients
                  for radiology / ultrasound / photo content.

                Modes are not mutually exclusive: a fax-of-handwritten
                form will get binarized (the ``fax`` rule wins because
                ``fax`` implies a 1-bit transport channel).

        Returns:
            EnhancementResult with enhanced image and metrics.

        Raises:
            EnhancementError: If enhancement fails.
        """
        import time

        start_time = time.perf_counter()
        enhancements_applied: list[EnhancementType] = []
        active_modes = set(modes or [])

        # Mode flag fan-out. ``fax`` dominates because once a document is
        # 1-bit/CCITT-compressed, the gentler handwritten path is moot.
        is_fax = "fax" in active_modes
        is_handwritten = "handwritten" in active_modes and not is_fax
        is_visual = "visual" in active_modes and not is_fax and not is_handwritten

        try:
            # Convert to OpenCV format
            img = self._page_to_cv2(page_image)

            # Calculate original metrics
            original_variance = self._calculate_variance(img)
            _, _ = cv2.meanStdDev(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            )

            # Track skew metrics
            skew_angle = 0.0
            skew_corrected = False

            # Apply deskewing
            if self._enable_deskew:
                img, detected_angle = self._deskew(img)
                if abs(detected_angle) > 0.1:
                    skew_angle = detected_angle
                    skew_corrected = True
                    enhancements_applied.append(EnhancementType.DESKEW)

            # Apply denoising. Fax mode skips Non-Local-Means entirely
            # (binarization will collapse gradient noise anyway). Handwritten
            # mode runs a *light* denoise because aggressive denoising eats
            # pen strokes; the default path keeps the legacy behaviour.
            noise_reduction_ratio = 1.0
            if self._enable_denoise and not is_fax:
                img_before_denoise = img.copy()
                if is_handwritten:
                    img = self._denoise(img, h=3.0, h_color=3.0)  # gentler
                else:
                    img = self._denoise(img)
                noise_reduction_ratio = self._calculate_noise_reduction(img_before_denoise, img)
                enhancements_applied.append(EnhancementType.DENOISE)

            # CLAHE: skip for fax (1-bit), handwritten (stroke-distorting),
            # and visual (preserves natural gradients).
            contrast_improvement = 1.0
            if self._enable_contrast and not (is_fax or is_handwritten or is_visual):
                img_before_clahe = img.copy()
                img = self._apply_clahe(img)
                contrast_improvement = self._calculate_contrast_improvement(img_before_clahe, img)
                enhancements_applied.append(EnhancementType.CLAHE)

            # Fax-specific path: Otsu threshold + morphological opening
            # + connected-components despeckle to clean speckle noise
            # typical of CCITT-compressed fax scans.
            if is_fax:
                img = self._apply_binarization(img)
                enhancements_applied.append(EnhancementType.BINARIZATION)
                img = self._apply_morphology(img)
                enhancements_applied.append(EnhancementType.MORPHOLOGICAL)
                # V3 Phase 5: drop sub-glyph speckle that morphology
                # leaves behind. Cheap and high-leverage on
                # 1-bit fax scans.
                img = self._apply_despeckle(img)
                enhancements_applied.append(EnhancementType.DESPECKLE)

            # Calculate enhanced metrics
            enhanced_variance = self._calculate_variance(img)

            # Convert back to PageImage
            enhanced_page = self._cv2_to_page(img, page_image)

            # Create metrics
            metrics = EnhancementMetrics(
                original_variance=original_variance,
                enhanced_variance=enhanced_variance,
                skew_angle=skew_angle,
                skew_corrected=skew_corrected,
                noise_reduction_ratio=noise_reduction_ratio,
                contrast_improvement=contrast_improvement,
            )

            # Calculate processing time
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            result = EnhancementResult(
                page_image=enhanced_page,
                original_page=page_image,
                enhancements_applied=enhancements_applied,
                metrics=metrics,
                processing_time_ms=processing_time_ms,
            )

            logger.info(
                "image_enhanced",
                page_number=page_image.page_number,
                enhancements=[e.value for e in enhancements_applied],
                skew_angle=skew_angle,
                sharpness_improvement=metrics.sharpness_improvement,
                processing_time_ms=processing_time_ms,
            )

            return result

        except EnhancementError:
            raise
        except Exception as e:
            raise EnhancementError(f"Image enhancement failed: {e}") from e

    def _page_to_cv2(self, page_image: PageImage) -> NDArray[np.uint8]:
        """Convert PageImage to OpenCV BGR array with proper memory cleanup."""
        pil_image = page_image.to_pil_image()

        try:
            # Convert to RGB if necessary
            if pil_image.mode != "RGB":
                converted = pil_image.convert("RGB")
                pil_image.close()  # Close original after conversion
                pil_image = converted

            # Convert to numpy array
            img_array = np.array(pil_image, dtype=np.uint8)

            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        finally:
            # Ensure PIL image is closed to release memory
            pil_image.close()
            del pil_image

    def _cv2_to_page(
        self,
        img: NDArray[np.uint8],
        original: PageImage,
    ) -> PageImage:
        """Convert OpenCV array back to PageImage."""
        # Convert BGR to RGB
        if len(img.shape) == 3:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_img)

        # Save to bytes, then close PIL image to prevent leak
        try:
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="PNG", optimize=True)
            img_bytes = img_buffer.getvalue()
        finally:
            pil_image.close()

        # Determine orientation
        if img.shape[1] > img.shape[0]:
            orientation = DocumentOrientation.LANDSCAPE
        elif img.shape[0] > img.shape[1]:
            orientation = DocumentOrientation.PORTRAIT
        else:
            orientation = DocumentOrientation.SQUARE

        return PageImage(
            page_number=original.page_number,
            image_bytes=img_bytes,
            width=img.shape[1],
            height=img.shape[0],
            dpi=original.dpi,
            orientation=orientation,
            original_width_pts=original.original_width_pts,
            original_height_pts=original.original_height_pts,
            has_text=original.has_text,
            has_images=original.has_images,
            rotation=original.rotation,
        )

    def _deskew(self, img: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float]:
        """
        Detect and correct document skew.

        Uses Hough Line Transform to detect the dominant angle and rotates
        the image to correct skew.

        Args:
            img: Input image in BGR format.

        Returns:
            Tuple of (deskewed image, detected angle in degrees).

        Raises:
            DeskewError: If deskewing fails.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Dilate edges to connect nearby lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=min(img.shape[0], img.shape[1]) // 8,
                maxLineGap=20,
            )

            if lines is None or len(lines) == 0:
                logger.debug("no_lines_detected_for_deskew")
                return img, 0.0

            # Calculate angles of detected lines
            angles: list[float] = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1

                # Calculate angle in degrees
                if dx != 0:
                    angle = math.degrees(math.atan2(dy, dx))

                    # Normalize to -45 to 45 range (document skew is usually small)
                    while angle > 45:
                        angle -= 90
                    while angle < -45:
                        angle += 90

                    # Only consider angles within max range
                    if abs(angle) <= self._deskew_max_angle:
                        angles.append(angle)

            if not angles:
                logger.debug("no_valid_angles_for_deskew")
                return img, 0.0

            # Use median angle to be robust to outliers
            median_angle = float(np.median(angles))

            # Skip correction for very small angles
            if abs(median_angle) < 0.5:
                return img, median_angle

            # Rotate image to correct skew
            height, width = img.shape[:2]
            center = (width // 2, height // 2)

            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            # Calculate new bounding box size
            cos_angle = abs(math.cos(math.radians(median_angle)))
            sin_angle = abs(math.sin(math.radians(median_angle)))
            new_width = int(height * sin_angle + width * cos_angle)
            new_height = int(height * cos_angle + width * sin_angle)

            # Adjust rotation matrix for new size
            rotation_matrix[0, 2] += (new_width - width) / 2
            rotation_matrix[1, 2] += (new_height - height) / 2

            # Apply rotation with white background
            rotated = cv2.warpAffine(
                img,
                rotation_matrix,
                (new_width, new_height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) if len(img.shape) == 3 else 255,
            )

            logger.debug(
                "image_deskewed",
                detected_angle=median_angle,
                lines_analyzed=len(angles),
            )

            return rotated, median_angle

        except Exception as e:
            raise DeskewError(f"Deskewing failed: {e}") from e

    def _denoise(
        self,
        img: NDArray[np.uint8],
        *,
        h: float | None = None,
        h_color: float | None = None,
    ) -> NDArray[np.uint8]:
        """
        Apply noise reduction using Non-Local Means Denoising.

        Args:
            img: Input image in BGR format.
            h: Optional luminance denoise strength override; defaults to
                ``self._denoise_strength``. Mode-specific callers (WS-3
                handwritten mode) lower this to avoid eating pen strokes.
            h_color: Optional chrominance denoise strength override.

        Returns:
            Denoised image.

        Raises:
            DenoiseError: If denoising fails.
        """
        h_lum = h if h is not None else self._denoise_strength
        h_chr = h_color if h_color is not None else self._denoise_strength
        try:
            if len(img.shape) == 3:
                # OpenCV's Python binding for fastNlMeansDenoisingColored uses
                # positional args: (src, dst, h, hColor, templateWindowSize,
                # searchWindowSize). The keyword names exposed (e.g. ``hColor``,
                # not ``hForColorComponents``) vary across OpenCV builds, so
                # we pass positionally to stay portable.
                denoised = cv2.fastNlMeansDenoisingColored(
                    img,
                    None,
                    float(h_lum),
                    float(h_chr),
                    7,
                    21,
                )
            else:
                # Grayscale image — fastNlMeansDenoising has a stable kw API.
                denoised = cv2.fastNlMeansDenoising(
                    img,
                    None,
                    float(h_lum),
                    7,
                    21,
                )

            return denoised

        except Exception as e:
            raise DenoiseError(f"Denoising failed: {e}") from e

    def _apply_binarization(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Otsu binarization (WS-3 fax mode).

        Faxes are typically 1-bit CCITT-compressed at the source; running
        them through CLAHE / colour preservation pipelines wastes cycles
        and amplifies aliasing. Otsu finds the optimal global threshold
        for the bimodal histogram fax scans produce, then converts the
        image to true binary. Output is BGR-shaped (single channel
        replicated to 3) so the rest of the enhancement chain stays
        type-compatible.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _apply_morphology(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Morphological opening (WS-3 fax mode).

        Pairs with ``_apply_binarization``. Opening = erosion then
        dilation; cleans up isolated speckle pixels that survive Otsu
        thresholding without thinning the actual character strokes. A
        2×2 kernel is small enough to preserve thin glyphs from fax-grade
        ~200 DPI scans.
        """
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def _apply_despeckle(
        self,
        img: NDArray[np.uint8],
        *,
        min_component_area: int = 4,
    ) -> NDArray[np.uint8]:
        """
        V3 Phase 5 — Despeckle for fax / poor-quality scans.

        Uses a connected-components pass to drop tiny black blobs that
        survive Otsu thresholding. Faxes typically include scattered
        speckle pixels (1-3 px each) that morphological opening leaves
        behind because they are larger than the 2x2 erosion kernel but
        smaller than any glyph component.

        We work on a binarised copy: anything that's a connected-black
        component below ``min_component_area`` pixels gets erased to
        white. Glyphs (strokes) at typical fax 200 DPI are ≥ 30 px²,
        so we have a wide safety margin.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # If the input is not already binary, threshold first.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert so connected components run on black-on-white text
        # (cv2.connectedComponents looks for non-zero blobs).
        inverted = cv2.bitwise_not(binary)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        # Build mask: keep components above the area threshold;
        # erase the rest.
        keep_mask = np.zeros_like(inverted)
        for label_idx in range(1, n_labels):  # 0 is background
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                keep_mask[labels == label_idx] = 255
        # Re-invert back to black-on-white.
        cleaned = cv2.bitwise_not(keep_mask)
        # Restore BGR shape so downstream stages don't have to branch.
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    def _classify_orientation(
        self,
        img: NDArray[np.uint8],
    ) -> int:
        """
        V3 Phase 5 — 4-way orientation classifier (0/90/180/270).

        Faxes and phone-camera scans frequently arrive sideways or
        upside-down. PDF rotation metadata catches the easy cases;
        this method catches the rest by analysing pixel-row text
        density profiles.

        Strategy: a correctly-oriented document has roughly equal
        horizontal text-line density top vs. bottom but stronger
        density in the upper half (header/letterhead). We use the
        dispersion of horizontal projection profiles to disambiguate
        portrait-portrait (0°) from upside-down (180°). For 90° vs
        270° we rotate by 90° and re-test. Returns the rotation in
        degrees the *image* should be rotated to be upright.

        This is a coarse heuristic; it does not replace a full OSD
        (orientation + script detection) but it's free and cheap. When
        in doubt it returns 0 so we never mis-rotate a correct page.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        # Threshold to text-like dark pixels.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def _portrait_score(arr: NDArray[np.uint8]) -> float:
            # Sum dark pixels per row; portrait orientation produces
            # strong horizontal text bands.
            row_density = arr.sum(axis=1).astype(np.float64)
            if row_density.size == 0 or row_density.max() == 0:
                return 0.0
            row_density /= row_density.max()
            # Peakiness: variance of normalised row densities. Strong
            # text bands → high variance. Random / noise → low variance.
            return float(np.var(row_density))

        # Test the four rotations and pick the one with the highest
        # portrait score. Note: for "upside-down" portrait (180°), the
        # variance is similar to upright (text bands are still
        # horizontal); we additionally check whether the *upper half*
        # has more density than the *lower half* (typical letterhead
        # heuristic) and only flip to 180° when the lower half clearly
        # dominates.
        scores: dict[int, float] = {}
        for rot in (0, 90, 180, 270):
            if rot == 0:
                rotated = binary
            elif rot == 180:
                rotated = cv2.rotate(binary, cv2.ROTATE_180)
            elif rot == 90:
                rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
            else:  # 270
                rotated = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
            scores[rot] = _portrait_score(rotated)

        # 90 vs 270 vs 0 vs 180: the rotation that maximises the
        # portrait score is the one needed to reach upright.
        best = max(scores, key=lambda k: scores[k])

        # Tie-breaker between 0 and 180: compare upper-half vs
        # lower-half density on the rotated image. If uppers > lower
        # we're upright (rotation 0). Use ``np.sum`` on int64 to avoid
        # uint8 overflow on large pages.
        if best in (0, 180):
            half = h // 2
            upper_density = int(np.sum(binary[:half].astype(np.int64)))
            lower_density = int(np.sum(binary[half:].astype(np.int64)))
            # Letterhead heuristic: if the upper half has noticeably
            # more density than the lower half (>10% gap), we're
            # likely upright; otherwise be conservative and stick
            # with the higher-variance rotation.
            if upper_density > lower_density * 1.1:
                return 0

        return best

    def _apply_clahe(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE improves local contrast while limiting noise amplification,
        making text more readable and improving VLM accuracy.

        Args:
            img: Input image in BGR format.

        Returns:
            Contrast-enhanced image.

        Raises:
            ContrastError: If contrast enhancement fails.
        """
        try:
            if len(img.shape) == 3:
                # Convert to LAB color space for CLAHE on luminance channel
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)

                # Apply CLAHE to luminance channel
                l_enhanced = self._clahe.apply(l_channel)

                # Merge channels back
                lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

                # Convert back to BGR
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Apply directly to grayscale
                enhanced = self._clahe.apply(img)

            return enhanced

        except Exception as e:
            raise ContrastError(f"CLAHE enhancement failed: {e}") from e

    def _calculate_variance(self, img: NDArray[np.uint8]) -> float:
        """
        Calculate Laplacian variance as a measure of image sharpness.

        Higher variance indicates sharper/more detailed image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _calculate_noise_reduction(
        self,
        before: NDArray[np.uint8],
        after: NDArray[np.uint8],
    ) -> float:
        """
        Calculate noise reduction ratio.

        Compares high-frequency content before and after denoising.
        """
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if len(before.shape) == 3 else before
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if len(after.shape) == 3 else after

        # Calculate noise as standard deviation of Laplacian
        noise_before = cv2.Laplacian(gray_before, cv2.CV_64F).std()
        noise_after = cv2.Laplacian(gray_after, cv2.CV_64F).std()

        if noise_before == 0:
            return 1.0

        return noise_after / noise_before

    def _calculate_contrast_improvement(
        self,
        before: NDArray[np.uint8],
        after: NDArray[np.uint8],
    ) -> float:
        """
        Calculate contrast improvement factor.

        Compares standard deviation of pixel values before and after.
        """
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if len(before.shape) == 3 else before
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if len(after.shape) == 3 else after

        std_before = gray_before.std()
        std_after = gray_after.std()

        if std_before == 0:
            return 1.0

        return std_after / std_before

    def enhance_batch(
        self,
        pages: list[PageImage],
    ) -> list[EnhancementResult]:
        """
        Enhance multiple pages.

        Args:
            pages: List of PageImages to enhance.

        Returns:
            List of EnhancementResults.
        """
        results: list[EnhancementResult] = []

        for page in pages:
            try:
                result = self.enhance(page)
                results.append(result)
            except EnhancementError as e:
                logger.warning(
                    "page_enhancement_failed",
                    page_number=page.page_number,
                    error=str(e),
                )
                # Return original page with empty enhancements on failure
                results.append(
                    EnhancementResult(
                        page_image=page,
                        original_page=page,
                        enhancements_applied=[],
                        metrics=None,
                        processing_time_ms=0,
                    )
                )

        return results

    def analyze_quality(self, page_image: PageImage) -> dict[str, Any]:
        """
        Analyze image quality without modifying it.

        Args:
            page_image: PageImage to analyze.

        Returns:
            Dictionary containing quality metrics.
        """
        img = self._page_to_cv2(page_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Calculate various quality metrics
        variance = self._calculate_variance(img)
        mean_brightness = float(gray.mean())
        contrast = float(gray.std())

        # Estimate blur using Laplacian variance
        blur_score = variance

        # Check for low contrast
        low_contrast = contrast < 30

        # Check for dark image
        is_dark = mean_brightness < 100

        # Check for bright/washed out image
        is_washed_out = mean_brightness > 200 and contrast < 40

        # Detect potential issues
        issues: list[str] = []
        if blur_score < 100:
            issues.append("image_may_be_blurry")
        if low_contrast:
            issues.append("low_contrast")
        if is_dark:
            issues.append("image_too_dark")
        if is_washed_out:
            issues.append("image_washed_out")

        return {
            "page_number": page_image.page_number,
            "laplacian_variance": variance,
            "mean_brightness": mean_brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "low_contrast": low_contrast,
            "is_dark": is_dark,
            "is_washed_out": is_washed_out,
            "issues": issues,
            "quality_score": self._calculate_quality_score(variance, contrast, mean_brightness),
        }

    def _calculate_quality_score(
        self,
        variance: float,
        contrast: float,
        brightness: float,
    ) -> float:
        """
        Calculate overall quality score from 0-100.

        Combines sharpness, contrast, and brightness metrics.
        """
        # Normalize metrics
        sharpness_score = min(100, variance / 10)  # Higher variance = sharper
        contrast_score = min(100, contrast * 2)  # Higher std = more contrast
        brightness_score = 100 - abs(brightness - 128) * 0.78  # Optimal around 128

        # Weighted average
        quality_score = sharpness_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3

        return max(0, min(100, quality_score))
