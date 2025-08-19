"""
Image Preprocessing Module
Handles image enhancement, denoising, skew correction, and optimization for OCR.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import math
from skimage import transform, filters
from skimage.transform import hough_line, hough_line_peaks

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for optimal OCR performance."""
    
    def __init__(self):
        self.processing_stats = {}
    
    def preprocess_image(
        self, 
        image: Image.Image, 
        enhance_contrast: bool = True,
        denoise: bool = True,
        correct_skew: bool = True,
        normalize_colors: bool = True
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply comprehensive image preprocessing for OCR optimization.
        
        Args:
            image: Input PIL Image
            enhance_contrast: Apply contrast enhancement
            denoise: Apply noise reduction
            correct_skew: Correct image rotation/skew
            normalize_colors: Normalize color distribution
            
        Returns:
            Tuple of (processed_image, processing_metadata)
        """
        try:
            logger.info(f"Starting image preprocessing: {image.size}")
            
            processing_metadata = {
                "original_size": image.size,
                "processing_steps": [],
                "skew_angle": 0,
                "contrast_factor": 1.0,
                "noise_reduction_applied": False
            }
            
            processed_image = image.copy()
            
            # Step 1: Convert to appropriate format if needed
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')
                processing_metadata["processing_steps"].append("converted_to_rgb")
            
            # Step 2: Denoise
            if denoise:
                processed_image, noise_metadata = self._denoise_image(processed_image)
                processing_metadata["processing_steps"].append("denoised")
                processing_metadata["noise_reduction_applied"] = True
                processing_metadata.update(noise_metadata)
            
            # Step 3: Skew correction
            if correct_skew:
                processed_image, skew_angle = self._correct_skew(processed_image)
                processing_metadata["skew_angle"] = skew_angle
                processing_metadata["processing_steps"].append(f"skew_corrected_{skew_angle:.2f}_deg")
            
            # Step 4: Contrast enhancement
            if enhance_contrast:
                processed_image, contrast_factor = self._enhance_contrast(processed_image)
                processing_metadata["contrast_factor"] = contrast_factor
                processing_metadata["processing_steps"].append(f"contrast_enhanced_{contrast_factor:.2f}")
            
            # Step 5: Color normalization
            if normalize_colors:
                processed_image = self._normalize_colors(processed_image)
                processing_metadata["processing_steps"].append("colors_normalized")
            
            # Step 6: Final optimization for OCR
            processed_image = self._optimize_for_ocr(processed_image)
            processing_metadata["processing_steps"].append("ocr_optimized")
            
            processing_metadata["final_size"] = processed_image.size
            
            logger.info(f"Image preprocessing completed: {len(processing_metadata['processing_steps'])} steps")
            
            return processed_image, processing_metadata
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _denoise_image(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Apply noise reduction techniques."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
            
            # Apply morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(denoised_rgb)
            
            # Additional Gaussian blur for very fine noise
            processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            metadata = {
                "denoise_method": "bilateral_filter_morphology",
                "gaussian_blur_radius": 0.5
            }
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return image, {"denoise_method": "failed"}
    
    def _correct_skew(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """Detect and correct image skew/rotation."""
        try:
            # Convert to grayscale for skew detection
            gray_array = np.array(image.convert('L'))
            
            # Edge detection
            edges = filters.sobel(gray_array)
            
            # Hough line transform to detect dominant lines
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = hough_line(edges, theta=tested_angles)
            
            # Find peaks (dominant lines)
            hough_peaks = hough_line_peaks(h, theta, d, num_peaks=10)
            
            if len(hough_peaks[1]) > 0:
                # Calculate most common angle
                angles_deg = np.rad2deg(hough_peaks[1])
                
                # Filter angles to reasonable skew range (-45 to 45 degrees)
                valid_angles = angles_deg[np.abs(angles_deg) < 45]
                
                if len(valid_angles) > 0:
                    # Use median angle to avoid outliers
                    skew_angle = np.median(valid_angles)
                    
                    # Only correct if angle is significant (> 0.5 degrees)
                    if abs(skew_angle) > 0.5:
                        # Rotate image to correct skew
                        rotated_image = image.rotate(-skew_angle, expand=True, fillcolor='white')
                        logger.info(f"Corrected skew by {skew_angle:.2f} degrees")
                        return rotated_image, skew_angle
            
            # No significant skew detected
            return image, 0.0
            
        except Exception as e:
            logger.error(f"Skew correction failed: {e}")
            return image, 0.0
    
    def _enhance_contrast(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """Enhance image contrast for better OCR performance."""
        try:
            # Calculate current contrast metrics
            gray_array = np.array(image.convert('L'))
            
            # Calculate histogram statistics
            hist, _ = np.histogram(gray_array, bins=256, range=(0, 256))
            
            # Calculate contrast using standard deviation
            current_std = np.std(gray_array)
            
            # Determine optimal contrast enhancement factor
            if current_std < 40:  # Low contrast
                contrast_factor = 1.5
            elif current_std < 60:  # Medium contrast
                contrast_factor = 1.2
            else:  # Already good contrast
                contrast_factor = 1.1
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(contrast_factor)
            
            # Apply adaptive histogram equalization for local contrast
            enhanced_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
            
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced_rgb = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB)
            
            final_image = Image.fromarray(enhanced_rgb)
            
            return final_image, contrast_factor
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image, 1.0
    
    def _normalize_colors(self, image: Image.Image) -> Image.Image:
        """Normalize color distribution for consistent processing."""
        try:
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize each channel independently
            for channel in range(img_array.shape[2]):
                channel_data = img_array[:, :, channel]
                
                # Calculate percentiles for robust normalization
                p2, p98 = np.percentile(channel_data, (2, 98))
                
                # Apply normalization
                if p98 > p2:  # Avoid division by zero
                    channel_data = np.clip(channel_data, p2, p98)
                    channel_data = (channel_data - p2) / (p98 - p2) * 255
                
                img_array[:, :, channel] = channel_data
            
            # Convert back to PIL Image
            normalized_array = np.clip(img_array, 0, 255).astype(np.uint8)
            normalized_image = Image.fromarray(normalized_array)
            
            return normalized_image
            
        except Exception as e:
            logger.error(f"Color normalization failed: {e}")
            return image
    
    def _optimize_for_ocr(self, image: Image.Image) -> Image.Image:
        """Final optimization steps specifically for OCR performance."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply sharpening filter to enhance text edges
            kernel_sharpening = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            sharpened = cv2.filter2D(cv_image, -1, kernel_sharpening)
            
            # Ensure good binarization potential by enhancing text-background contrast
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to preview binarization quality
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # If binarization looks good, keep the sharpened color image
            # Otherwise, fall back to original
            if np.sum(binary == 255) > 0.1 * binary.size:  # Ensure enough white pixels
                final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                optimized_image = Image.fromarray(final_rgb)
            else:
                optimized_image = image
            
            # Final resize if image is too large (optimize processing speed)
            if optimized_image.width > 3000 or optimized_image.height > 3000:
                # Calculate new size maintaining aspect ratio
                ratio = min(3000 / optimized_image.width, 3000 / optimized_image.height)
                new_size = (int(optimized_image.width * ratio), int(optimized_image.height * ratio))
                optimized_image = optimized_image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image for optimization: {new_size}")
            
            return optimized_image
            
        except Exception as e:
            logger.error(f"OCR optimization failed: {e}")
            return image
    
    def create_binary_image(self, image: Image.Image, method: str = "adaptive") -> Image.Image:
        """Create binary (black and white) version for specialized OCR processing."""
        try:
            # Convert to grayscale
            gray_array = np.array(image.convert('L'))
            
            if method == "adaptive":
                # Adaptive thresholding
                binary = cv2.adaptiveThreshold(gray_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            elif method == "otsu":
                # Otsu's thresholding
                _, binary = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Simple thresholding with automatic threshold
                threshold_value = np.mean(gray_array) - 10
                _, binary = cv2.threshold(gray_array, threshold_value, 255, cv2.THRESH_BINARY)
            
            return Image.fromarray(binary)
            
        except Exception as e:
            logger.error(f"Binary image creation failed: {e}")
            return image.convert('L')
    
    def get_image_quality_metrics(self, image: Image.Image) -> Dict[str, float]:
        """Calculate image quality metrics to assess OCR readiness."""
        try:
            # Convert to grayscale for analysis
            gray_array = np.array(image.convert('L'))
            
            metrics = {
                "contrast": np.std(gray_array),
                "brightness": np.mean(gray_array),
                "sharpness": cv2.Laplacian(gray_array, cv2.CV_64F).var(),
                "noise_level": self._estimate_noise_level(gray_array),
                "text_background_ratio": self._estimate_text_ratio(gray_array)
            }
            
            # Overall quality score (0-100)
            quality_score = min(100, (
                min(metrics["contrast"] / 50, 1) * 30 +  # Contrast contribution
                (1 - abs(metrics["brightness"] - 127) / 127) * 20 +  # Brightness contribution
                min(metrics["sharpness"] / 1000, 1) * 30 +  # Sharpness contribution
                (1 - min(metrics["noise_level"], 1)) * 20  # Noise contribution (inverted)
            ))
            
            metrics["quality_score"] = quality_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {"quality_score": 50}  # Default medium quality
    
    def _estimate_noise_level(self, gray_array: np.ndarray) -> float:
        """Estimate noise level in the image."""
        try:
            # Use Laplacian variance as a noise indicator
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (higher values indicate more noise)
            noise_level = min(laplacian_var / 2000, 1.0)
            
            return noise_level
            
        except Exception:
            return 0.5  # Default moderate noise
    
    def _estimate_text_ratio(self, gray_array: np.ndarray) -> float:
        """Estimate the ratio of text to background in the image."""
        try:
            # Simple thresholding to separate text from background
            _, binary = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Assume text is black (0) and background is white (255)
            text_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            
            text_ratio = text_pixels / total_pixels
            
            return text_ratio
            
        except Exception:
            return 0.3  # Default text ratio


# Global processor instance
image_processor = ImageProcessor()