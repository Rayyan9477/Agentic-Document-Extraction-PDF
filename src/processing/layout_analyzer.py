"""
Layout Analysis Module
Handles document structure analysis using Docling and custom algorithms.
Identifies text blocks, tables, forms, and other document elements.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import cv2
from dataclasses import dataclass
import json

# Try to import docling, with fallback for development
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available, using fallback layout analysis")

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class LayoutElement:
    """Represents a document layout element."""
    element_id: str
    element_type: str  # text, table, form, image, line, etc.
    bbox: BoundingBox
    content: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "bbox": {
                "x0": self.bbox.x0,
                "y0": self.bbox.y0,
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "width": self.bbox.width,
                "height": self.bbox.height
            },
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class LayoutAnalyzer:
    """Analyzes document layout and structure for optimal processing."""
    
    def __init__(self):
        self.docling_converter = None
        self._initialize_docling()
    
    def _initialize_docling(self):
        """Initialize Docling converter if available."""
        try:
            if DOCLING_AVAILABLE:
                # Configure pipeline options for medical documents
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True
                
                self.docling_converter = DocumentConverter(
                    format_options={InputFormat.PDF: pipeline_options}
                )
                logger.info("Docling converter initialized successfully")
            else:
                logger.warning("Docling not available, using fallback methods")
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            self.docling_converter = None
    
    def analyze_page_layout(
        self, 
        image: Image.Image, 
        pdf_text_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[LayoutElement]:
        """
        Analyze page layout to identify document structure.
        
        Args:
            image: Page image to analyze
            pdf_text_blocks: Optional text blocks from PDF extraction
            
        Returns:
            List of layout elements with bounding boxes and metadata
        """
        try:
            logger.info(f"Starting layout analysis for image: {image.size}")
            
            if self.docling_converter and DOCLING_AVAILABLE:
                # Use Docling for advanced analysis
                elements = self._analyze_with_docling(image, pdf_text_blocks)
            else:
                # Use fallback OpenCV-based analysis
                elements = self._analyze_with_opencv(image, pdf_text_blocks)
            
            # Post-process and enhance results
            enhanced_elements = self._enhance_layout_elements(elements, image)
            
            logger.info(f"Layout analysis completed: {len(enhanced_elements)} elements found")
            
            return enhanced_elements
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return []
    
    def _analyze_with_docling(
        self, 
        image: Image.Image, 
        pdf_text_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[LayoutElement]:
        """Analyze layout using Docling (when available)."""
        try:
            # Note: Docling typically works with PDF files directly
            # For image-based analysis, we'll simulate the expected structure
            elements = []
            
            # Convert image to OpenCV format for preprocessing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use existing PDF text blocks if available
            if pdf_text_blocks:
                for i, block in enumerate(pdf_text_blocks):
                    bbox = BoundingBox(
                        x0=block["bbox"]["x0"],
                        y0=block["bbox"]["y0"],
                        x1=block["bbox"]["x1"],
                        y1=block["bbox"]["y1"]
                    )
                    
                    element = LayoutElement(
                        element_id=f"text_{i}",
                        element_type="text",
                        bbox=bbox,
                        content=block["text"],
                        confidence=0.9,
                        metadata={
                            "source": "pdf_extraction",
                            "block_id": block.get("block_id", i)
                        }
                    )
                    elements.append(element)
            
            # Detect additional structural elements
            table_elements = self._detect_tables(cv_image)
            elements.extend(table_elements)
            
            form_elements = self._detect_form_fields(cv_image)
            elements.extend(form_elements)
            
            return elements
            
        except Exception as e:
            logger.error(f"Docling-based analysis failed: {e}")
            return self._analyze_with_opencv(image, pdf_text_blocks)
    
    def _analyze_with_opencv(
        self, 
        image: Image.Image, 
        pdf_text_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[LayoutElement]:
        """Fallback layout analysis using OpenCV."""
        try:
            elements = []
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use PDF text blocks if available
            if pdf_text_blocks:
                for i, block in enumerate(pdf_text_blocks):
                    bbox = BoundingBox(
                        x0=block["bbox"]["x0"],
                        y0=block["bbox"]["y0"],
                        x1=block["bbox"]["x1"],
                        y1=block["bbox"]["y1"]
                    )
                    
                    element = LayoutElement(
                        element_id=f"text_{i}",
                        element_type="text",
                        bbox=bbox,
                        content=block["text"],
                        confidence=0.8,
                        metadata={
                            "source": "pdf_extraction",
                            "analysis_method": "opencv_fallback"
                        }
                    )
                    elements.append(element)
            else:
                # Detect text regions using contour analysis
                text_elements = self._detect_text_regions(cv_image)
                elements.extend(text_elements)
            
            # Detect structural elements
            line_elements = self._detect_lines_and_separators(cv_image)
            elements.extend(line_elements)
            
            table_elements = self._detect_tables(cv_image)
            elements.extend(table_elements)
            
            form_elements = self._detect_form_fields(cv_image)
            elements.extend(form_elements)
            
            return elements
            
        except Exception as e:
            logger.error(f"OpenCV-based analysis failed: {e}")
            return []
    
    def _detect_text_regions(self, cv_image: np.ndarray) -> List[LayoutElement]:
        """Detect text regions using contour analysis."""
        try:
            elements = []
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to connect text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small regions
                if w > 50 and h > 10:
                    bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                    
                    element = LayoutElement(
                        element_id=f"text_region_{i}",
                        element_type="text_region",
                        bbox=bbox,
                        content="",  # Will be filled by OCR
                        confidence=0.6,
                        metadata={
                            "detection_method": "contour_analysis",
                            "area": w * h
                        }
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []
    
    def _detect_lines_and_separators(self, cv_image: np.ndarray) -> List[LayoutElement]:
        """Detect horizontal and vertical lines that may separate sections."""
        try:
            elements = []
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find contours for horizontal lines
            h_contours, _ = cv2.findContours(
                horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(h_contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h < 10:  # Long, thin horizontal lines
                    bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                    
                    element = LayoutElement(
                        element_id=f"h_line_{i}",
                        element_type="horizontal_line",
                        bbox=bbox,
                        content="",
                        confidence=0.8,
                        metadata={
                            "line_type": "horizontal",
                            "length": w
                        }
                    )
                    elements.append(element)
            
            # Find contours for vertical lines
            v_contours, _ = cv2.findContours(
                vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(v_contours):
                x, y, w, h = cv2.boundingRect(contour)
                if h > 100 and w < 10:  # Tall, thin vertical lines
                    bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                    
                    element = LayoutElement(
                        element_id=f"v_line_{i}",
                        element_type="vertical_line",
                        bbox=bbox,
                        content="",
                        confidence=0.8,
                        metadata={
                            "line_type": "vertical",
                            "length": h
                        }
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Line detection failed: {e}")
            return []
    
    def _detect_tables(self, cv_image: np.ndarray) -> List[LayoutElement]:
        """Detect table structures in the image."""
        try:
            elements = []
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect both horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine horizontal and vertical lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            
            # Find table regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(
                table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Tables should be reasonably large
                if w > 200 and h > 100:
                    bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                    
                    element = LayoutElement(
                        element_id=f"table_{i}",
                        element_type="table",
                        bbox=bbox,
                        content="",  # Will be filled by table parser
                        confidence=0.7,
                        metadata={
                            "detection_method": "line_intersection",
                            "estimated_rows": h // 20,  # Rough estimate
                            "estimated_cols": w // 100
                        }
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def _detect_form_fields(self, cv_image: np.ndarray) -> List[LayoutElement]:
        """Detect form fields like checkboxes, input fields, etc."""
        try:
            elements = []
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular regions that might be form fields
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Approximate contour to check if it's rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangular shape
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it looks like a checkbox (small square)
                    if 10 <= w <= 30 and 10 <= h <= 30 and abs(w - h) < 5:
                        bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                        
                        element = LayoutElement(
                            element_id=f"checkbox_{i}",
                            element_type="checkbox",
                            bbox=bbox,
                            content="",
                            confidence=0.6,
                            metadata={
                                "form_element_type": "checkbox",
                                "size": (w, h)
                            }
                        )
                        elements.append(element)
                    
                    # Check if it looks like an input field (rectangular, wider than tall)
                    elif w > 100 and 15 <= h <= 40 and w / h > 3:
                        bbox = BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h)
                        
                        element = LayoutElement(
                            element_id=f"input_field_{i}",
                            element_type="input_field",
                            bbox=bbox,
                            content="",
                            confidence=0.5,
                            metadata={
                                "form_element_type": "input_field",
                                "size": (w, h)
                            }
                        )
                        elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Form field detection failed: {e}")
            return []
    
    def _enhance_layout_elements(
        self, 
        elements: List[LayoutElement], 
        image: Image.Image
    ) -> List[LayoutElement]:
        """Post-process and enhance detected layout elements."""
        try:
            # Remove overlapping elements (keep the one with higher confidence)
            non_overlapping = self._remove_overlapping_elements(elements)
            
            # Sort elements by reading order (top to bottom, left to right)
            sorted_elements = self._sort_by_reading_order(non_overlapping)
            
            # Add reading order metadata
            for i, element in enumerate(sorted_elements):
                element.metadata["reading_order"] = i
            
            # Classify medical document regions if possible
            classified_elements = self._classify_medical_regions(sorted_elements)
            
            return classified_elements
            
        except Exception as e:
            logger.error(f"Element enhancement failed: {e}")
            return elements
    
    def _remove_overlapping_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Remove overlapping elements, keeping the one with higher confidence."""
        try:
            non_overlapping = []
            
            for i, element1 in enumerate(elements):
                is_overlapping = False
                
                for j, element2 in enumerate(elements):
                    if i != j and self._calculate_overlap(element1.bbox, element2.bbox) > 0.5:
                        # If overlapping and element1 has lower confidence, skip it
                        if element1.confidence < element2.confidence:
                            is_overlapping = True
                            break
                
                if not is_overlapping:
                    non_overlapping.append(element1)
            
            return non_overlapping
            
        except Exception as e:
            logger.error(f"Overlap removal failed: {e}")
            return elements
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        try:
            # Calculate intersection
            x_overlap = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
            y_overlap = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
            
            intersection_area = x_overlap * y_overlap
            
            # Calculate union
            union_area = bbox1.area + bbox2.area - intersection_area
            
            # Return intersection over union (IoU)
            if union_area > 0:
                return intersection_area / union_area
            return 0.0
            
        except Exception:
            return 0.0
    
    def _sort_by_reading_order(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Sort elements by typical reading order (top-to-bottom, left-to-right)."""
        try:
            return sorted(elements, key=lambda x: (x.bbox.y0, x.bbox.x0))
        except Exception as e:
            logger.error(f"Reading order sorting failed: {e}")
            return elements
    
    def _classify_medical_regions(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Classify elements based on typical medical document patterns."""
        try:
            for element in elements:
                content_lower = element.content.lower()
                
                # Classify based on content keywords
                if any(keyword in content_lower for keyword in ["patient", "name", "dob", "birth"]):
                    element.metadata["medical_region"] = "patient_info"
                elif any(keyword in content_lower for keyword in ["cpt", "procedure", "code"]):
                    element.metadata["medical_region"] = "procedure_codes"
                elif any(keyword in content_lower for keyword in ["diagnosis", "dx", "icd"]):
                    element.metadata["medical_region"] = "diagnosis_codes"
                elif any(keyword in content_lower for keyword in ["provider", "doctor", "physician"]):
                    element.metadata["medical_region"] = "provider_info"
                elif any(keyword in content_lower for keyword in ["insurance", "policy", "coverage"]):
                    element.metadata["medical_region"] = "insurance_info"
                else:
                    element.metadata["medical_region"] = "general"
            
            return elements
            
        except Exception as e:
            logger.error(f"Medical region classification failed: {e}")
            return elements
    
    def visualize_layout(self, image: Image.Image, elements: List[LayoutElement]) -> Image.Image:
        """Create a visualization of the detected layout elements."""
        try:
            # Create a copy of the image for drawing
            viz_image = image.copy()
            draw = ImageDraw.Draw(viz_image)
            
            # Color mapping for different element types
            colors = {
                "text": "red",
                "text_region": "blue",
                "table": "green",
                "horizontal_line": "orange",
                "vertical_line": "orange",
                "checkbox": "purple",
                "input_field": "cyan"
            }
            
            for element in elements:
                # Get color for element type
                color = colors.get(element.element_type, "black")
                
                # Draw bounding box
                bbox = element.bbox
                draw.rectangle(
                    [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    outline=color,
                    width=2
                )
                
                # Add label
                label = f"{element.element_type}_{element.element_id}"
                draw.text((bbox.x0, bbox.y0 - 15), label, fill=color)
            
            return viz_image
            
        except Exception as e:
            logger.error(f"Layout visualization failed: {e}")
            return image


# Global analyzer instance
layout_analyzer = LayoutAnalyzer()