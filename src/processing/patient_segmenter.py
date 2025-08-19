"""
Patient Segmentation Module
Detects and separates multiple patient records within a single page/document.
Uses keyword detection, layout analysis, and LLM assistance for accurate segmentation.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import cv2

from src.processing.layout_analyzer import LayoutElement, BoundingBox
from src.services.llm_service import llm_service

logger = logging.getLogger(__name__)


@dataclass
class PatientRecord:
    """Represents a single patient record within a document."""
    record_id: str
    patient_identifier: Optional[str]
    bbox: BoundingBox
    layout_elements: List[LayoutElement]
    extracted_text: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "patient_identifier": self.patient_identifier,
            "bbox": {
                "x0": self.bbox.x0,
                "y0": self.bbox.y0,
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "width": self.bbox.width,
                "height": self.bbox.height
            },
            "layout_elements_count": len(self.layout_elements),
            "extracted_text": self.extracted_text,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class PatientSegmenter:
    """Handles detection and segmentation of multiple patient records."""
    
    def __init__(self):
        self.patient_indicators = [
            # Common patient identification patterns
            r"patient\s+name\s*:?",
            r"name\s*:?",
            r"patient\s*:?",
            r"pt\s+name\s*:?",
            r"full\s+name\s*:?",
            
            # Date of birth patterns
            r"date\s+of\s+birth\s*:?",
            r"dob\s*:?",
            r"birth\s+date\s*:?",
            r"d\.?o\.?b\.?\s*:?",
            
            # ID patterns
            r"patient\s+id\s*:?",
            r"account\s+number\s*:?",
            r"medical\s+record\s*:?",
            r"mrn\s*:?",
            r"id\s+number\s*:?",
            
            # Form patterns
            r"superbill",
            r"encounter\s+form",
            r"billing\s+statement",
            r"patient\s+information"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.patient_indicators
        ]
    
    def segment_patients(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement],
        extracted_text: str,
        use_llm_assistance: bool = True
    ) -> List[PatientRecord]:
        """
        Segment the page into individual patient records.
        
        Args:
            image: Original page image
            layout_elements: Detected layout elements
            extracted_text: Combined extracted text
            use_llm_assistance: Whether to use LLM for intelligent segmentation
            
        Returns:
            List of segmented patient records
        """
        try:
            logger.info("Starting patient segmentation")
            
            # Step 1: Quick check if multiple patients are likely present
            patient_count_estimate = self._estimate_patient_count(extracted_text, layout_elements)
            logger.info(f"Estimated patient count: {patient_count_estimate}")
            
            if patient_count_estimate <= 1:
                # Single patient - return entire page as one record
                return [self._create_single_patient_record(image, layout_elements, extracted_text)]
            
            # Step 2: Detect patient boundaries using multiple methods
            boundaries = []
            
            # Method 1: Keyword-based detection
            keyword_boundaries = self._detect_keyword_boundaries(layout_elements, extracted_text)
            boundaries.extend(keyword_boundaries)
            
            # Method 2: Layout-based detection (visual separators)
            layout_boundaries = self._detect_layout_boundaries(layout_elements, image)
            boundaries.extend(layout_boundaries)
            
            # Method 3: LLM-assisted segmentation (if enabled)
            if use_llm_assistance and len(boundaries) > 0:
                llm_boundaries = self._llm_assisted_segmentation(extracted_text, boundaries)
                if llm_boundaries:
                    boundaries = llm_boundaries
            
            # Step 3: Merge and validate boundaries
            validated_boundaries = self._validate_and_merge_boundaries(boundaries, image.size)
            
            # Step 4: Create patient records from boundaries
            if len(validated_boundaries) >= 2:
                patient_records = self._create_patient_records(
                    image, layout_elements, extracted_text, validated_boundaries
                )
            else:
                # Fallback to single patient
                patient_records = [self._create_single_patient_record(image, layout_elements, extracted_text)]
            
            logger.info(f"Patient segmentation completed: {len(patient_records)} patients found")
            
            return patient_records
            
        except Exception as e:
            logger.error(f"Patient segmentation failed: {e}")
            # Fallback: return entire page as single patient
            return [self._create_single_patient_record(image, layout_elements, extracted_text)]
    
    def _estimate_patient_count(
        self, 
        text: str, 
        layout_elements: List[LayoutElement]
    ) -> int:
        """Estimate the number of patients in the document."""
        try:
            # Method 1: Count patient name indicators
            name_matches = 0
            for pattern in self.compiled_patterns[:5]:  # Use name patterns only
                matches = pattern.findall(text)
                name_matches += len(matches)
            
            # Method 2: Count recurring patterns
            dob_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
            dob_matches = len(dob_pattern.findall(text))
            
            # Method 3: Count medical record numbers or IDs
            id_pattern = re.compile(r'\b(?:id|mrn|account)[\s:]*\d+\b', re.IGNORECASE)
            id_matches = len(id_pattern.findall(text))
            
            # Method 4: Check layout repetition
            text_elements = [e for e in layout_elements if e.element_type in ["text", "text_region"]]
            repeated_layouts = self._detect_repeated_layouts(text_elements)
            
            # Calculate estimate
            estimates = [name_matches, dob_matches, id_matches, repeated_layouts]
            valid_estimates = [e for e in estimates if e > 0]
            
            if valid_estimates:
                # Use the most common estimate, but cap at reasonable maximum
                estimate = max(set(valid_estimates), key=valid_estimates.count)
                return min(estimate, 10)  # Cap at 10 patients per page
            
            return 1  # Default to single patient
            
        except Exception as e:
            logger.error(f"Patient count estimation failed: {e}")
            return 1
    
    def _detect_keyword_boundaries(
        self, 
        layout_elements: List[LayoutElement], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Detect patient boundaries based on keyword patterns."""
        try:
            boundaries = []
            
            # Find elements that likely indicate new patient records
            for element in layout_elements:
                element_text = element.content.lower() if element.content else ""
                
                # Check if element contains patient indicators
                for i, pattern in enumerate(self.compiled_patterns):
                    if pattern.search(element_text):
                        boundary = {
                            "type": "keyword",
                            "y_position": element.bbox.y0,
                            "x_position": element.bbox.x0,
                            "element_id": element.element_id,
                            "pattern_matched": self.patient_indicators[i],
                            "confidence": 0.8,
                            "bbox": element.bbox
                        }
                        boundaries.append(boundary)
                        break  # Don't match multiple patterns for same element
            
            # Sort boundaries by vertical position
            boundaries.sort(key=lambda x: (x["y_position"], x["x_position"]))
            
            logger.info(f"Found {len(boundaries)} keyword-based boundaries")
            return boundaries
            
        except Exception as e:
            logger.error(f"Keyword boundary detection failed: {e}")
            return []
    
    def _detect_layout_boundaries(
        self, 
        layout_elements: List[LayoutElement], 
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """Detect patient boundaries based on layout structure."""
        try:
            boundaries = []
            
            # Find horizontal lines that might separate patients
            h_lines = [e for e in layout_elements if e.element_type == "horizontal_line"]
            
            for line in h_lines:
                # Lines that span most of the page width are likely separators
                if line.bbox.width > image.width * 0.6:  # Line spans > 60% of page width
                    boundary = {
                        "type": "layout_line",
                        "y_position": line.bbox.y1,  # Use bottom of line
                        "x_position": 0,
                        "element_id": line.element_id,
                        "confidence": 0.7,
                        "bbox": line.bbox
                    }
                    boundaries.append(boundary)
            
            # Find large vertical gaps between text regions
            text_elements = [e for e in layout_elements if e.element_type in ["text", "text_region"]]
            text_elements.sort(key=lambda x: x.bbox.y0)
            
            for i in range(len(text_elements) - 1):
                current = text_elements[i]
                next_elem = text_elements[i + 1]
                
                vertical_gap = next_elem.bbox.y0 - current.bbox.y1
                
                # Large gaps might indicate patient separations
                if vertical_gap > 50:  # Gap > 50 pixels
                    boundary = {
                        "type": "layout_gap",
                        "y_position": current.bbox.y1 + vertical_gap / 2,
                        "x_position": 0,
                        "element_id": f"gap_{i}",
                        "confidence": 0.6,
                        "gap_size": vertical_gap,
                        "bbox": BoundingBox(
                            x0=0, 
                            y0=current.bbox.y1, 
                            x1=image.width, 
                            y1=next_elem.bbox.y0
                        )
                    }
                    boundaries.append(boundary)
            
            logger.info(f"Found {len(boundaries)} layout-based boundaries")
            return boundaries
            
        except Exception as e:
            logger.error(f"Layout boundary detection failed: {e}")
            return []
    
    def _detect_repeated_layouts(self, text_elements: List[LayoutElement]) -> int:
        """Detect repeated layout patterns that might indicate multiple patients."""
        try:
            if len(text_elements) < 4:
                return 1
            
            # Group elements by y-position bands
            y_positions = [e.bbox.y0 for e in text_elements]
            y_bands = self._create_position_bands(y_positions, band_height=100)
            
            # Look for repeating patterns in band structure
            if len(y_bands) >= 2:
                # Check if we have similar patterns repeating
                band_sizes = [len(band) for band in y_bands]
                
                # If we have multiple bands of similar sizes, likely multiple patients
                if len(set(band_sizes)) <= 2 and len(y_bands) >= 2:
                    return len(y_bands)
            
            return 1
            
        except Exception:
            return 1
    
    def _create_position_bands(self, positions: List[float], band_height: float) -> List[List[float]]:
        """Group positions into bands based on proximity."""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        bands = []
        current_band = [sorted_positions[0]]
        
        for pos in sorted_positions[1:]:
            if pos - current_band[-1] <= band_height:
                current_band.append(pos)
            else:
                bands.append(current_band)
                current_band = [pos]
        
        bands.append(current_band)
        return bands
    
    def _llm_assisted_segmentation(
        self, 
        text: str, 
        preliminary_boundaries: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Use LLM to validate and refine patient boundaries."""
        try:
            logger.info("Requesting LLM assistance for patient segmentation")
            
            # Build prompt with context
            segmentation_prompt = f"""Analyze this medical document text and determine patient record boundaries.

PRELIMINARY BOUNDARIES DETECTED:
{len(preliminary_boundaries)} potential boundaries found at y-positions: {[b['y_position'] for b in preliminary_boundaries]}

DOCUMENT TEXT:
{text[:2000]}...

Please analyze the text and provide:
1. Number of distinct patients in this document
2. Key indicators that separate patient records
3. Validation of the preliminary boundaries (which are correct/incorrect)

Respond with a JSON structure containing:
{{
    "patient_count": <number>,
    "validated_boundaries": [list of y-positions that are valid boundaries],
    "reasoning": "<explanation of your analysis>"
}}"""

            response = llm_service.client.chat.completions.create(
                model=llm_service.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a medical document analysis expert specializing in patient record segmentation."},
                    {"role": "user", "content": segmentation_prompt}
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            import json
            llm_analysis = json.loads(response.choices[0].message.content)
            
            if "validated_boundaries" in llm_analysis:
                # Convert LLM validation back to boundary format
                validated_boundaries = []
                for y_pos in llm_analysis["validated_boundaries"]:
                    # Find the closest preliminary boundary
                    closest_boundary = min(
                        preliminary_boundaries, 
                        key=lambda b: abs(b["y_position"] - y_pos)
                    )
                    
                    if abs(closest_boundary["y_position"] - y_pos) < 50:  # Within 50 pixels
                        validated_boundary = closest_boundary.copy()
                        validated_boundary["llm_validated"] = True
                        validated_boundary["confidence"] = min(0.95, validated_boundary["confidence"] + 0.2)
                        validated_boundaries.append(validated_boundary)
                
                logger.info(f"LLM validated {len(validated_boundaries)} boundaries")
                return validated_boundaries
            
            return None
            
        except Exception as e:
            logger.error(f"LLM-assisted segmentation failed: {e}")
            return None
    
    def _validate_and_merge_boundaries(
        self, 
        boundaries: List[Dict[str, Any]], 
        image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Validate and merge overlapping or redundant boundaries."""
        try:
            if not boundaries:
                return []
            
            # Sort by y-position
            sorted_boundaries = sorted(boundaries, key=lambda x: x["y_position"])
            
            # Remove boundaries that are too close together
            merged_boundaries = []
            min_separation = 30  # Minimum 30 pixels between boundaries
            
            for boundary in sorted_boundaries:
                if not merged_boundaries:
                    merged_boundaries.append(boundary)
                else:
                    last_boundary = merged_boundaries[-1]
                    if boundary["y_position"] - last_boundary["y_position"] > min_separation:
                        merged_boundaries.append(boundary)
                    else:
                        # Merge with existing boundary - keep higher confidence
                        if boundary["confidence"] > last_boundary["confidence"]:
                            merged_boundaries[-1] = boundary
            
            # Remove boundaries too close to page edges
            edge_margin = 50
            valid_boundaries = []
            
            for boundary in merged_boundaries:
                y_pos = boundary["y_position"]
                if edge_margin < y_pos < (image_size[1] - edge_margin):
                    valid_boundaries.append(boundary)
            
            logger.info(f"Validated {len(valid_boundaries)} boundaries from {len(boundaries)} initial detections")
            return valid_boundaries
            
        except Exception as e:
            logger.error(f"Boundary validation failed: {e}")
            return boundaries  # Return original if validation fails
    
    def _create_patient_records(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement],
        extracted_text: str,
        boundaries: List[Dict[str, Any]]
    ) -> List[PatientRecord]:
        """Create patient record objects from validated boundaries."""
        try:
            patient_records = []
            
            # Sort boundaries by y-position
            sorted_boundaries = sorted(boundaries, key=lambda x: x["y_position"])
            
            # Create segments between boundaries
            segments = []
            
            # First segment: from top of page to first boundary
            if sorted_boundaries:
                segments.append({
                    "y_start": 0,
                    "y_end": sorted_boundaries[0]["y_position"]
                })
                
                # Middle segments: between consecutive boundaries
                for i in range(len(sorted_boundaries) - 1):
                    segments.append({
                        "y_start": sorted_boundaries[i]["y_position"],
                        "y_end": sorted_boundaries[i + 1]["y_position"]
                    })
                
                # Last segment: from last boundary to bottom of page
                segments.append({
                    "y_start": sorted_boundaries[-1]["y_position"],
                    "y_end": image.height
                })
            
            # Create patient records for each segment
            for i, segment in enumerate(segments):
                # Find layout elements within this segment
                segment_elements = []
                for element in layout_elements:
                    element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
                    if segment["y_start"] <= element_center_y <= segment["y_end"]:
                        segment_elements.append(element)
                
                if segment_elements:  # Only create record if it has content
                    # Calculate bounding box for this patient record
                    x_coords = [e.bbox.x0 for e in segment_elements] + [e.bbox.x1 for e in segment_elements]
                    y_coords = [e.bbox.y0 for e in segment_elements] + [e.bbox.y1 for e in segment_elements]
                    
                    record_bbox = BoundingBox(
                        x0=min(x_coords),
                        y0=max(segment["y_start"], min(y_coords)),
                        x1=max(x_coords),
                        y1=min(segment["y_end"], max(y_coords))
                    )
                    
                    # Extract text from elements in reading order
                    sorted_elements = sorted(segment_elements, key=lambda x: (x.bbox.y0, x.bbox.x0))
                    segment_text = "\n".join([e.content for e in sorted_elements if e.content])
                    
                    # Try to identify patient from text
                    patient_id = self._extract_patient_identifier(segment_text)
                    
                    # Calculate confidence based on content quality
                    confidence = self._calculate_record_confidence(segment_elements, segment_text)
                    
                    patient_record = PatientRecord(
                        record_id=f"patient_{i+1}",
                        patient_identifier=patient_id,
                        bbox=record_bbox,
                        layout_elements=segment_elements,
                        extracted_text=segment_text,
                        confidence=confidence,
                        metadata={
                            "segment_index": i,
                            "element_count": len(segment_elements),
                            "y_range": (segment["y_start"], segment["y_end"]),
                            "text_length": len(segment_text)
                        }
                    )
                    
                    patient_records.append(patient_record)
            
            return patient_records
            
        except Exception as e:
            logger.error(f"Patient record creation failed: {e}")
            return []
    
    def _create_single_patient_record(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement],
        extracted_text: str
    ) -> PatientRecord:
        """Create a single patient record encompassing the entire page."""
        try:
            # Calculate overall bounding box
            if layout_elements:
                x_coords = [e.bbox.x0 for e in layout_elements] + [e.bbox.x1 for e in layout_elements]
                y_coords = [e.bbox.y0 for e in layout_elements] + [e.bbox.y1 for e in layout_elements]
                
                record_bbox = BoundingBox(
                    x0=min(x_coords),
                    y0=min(y_coords),
                    x1=max(x_coords),
                    y1=max(y_coords)
                )
            else:
                record_bbox = BoundingBox(
                    x0=0, y0=0, x1=image.width, y1=image.height
                )
            
            # Extract patient identifier
            patient_id = self._extract_patient_identifier(extracted_text)
            
            # Calculate confidence
            confidence = self._calculate_record_confidence(layout_elements, extracted_text)
            
            return PatientRecord(
                record_id="patient_1",
                patient_identifier=patient_id,
                bbox=record_bbox,
                layout_elements=layout_elements,
                extracted_text=extracted_text,
                confidence=confidence,
                metadata={
                    "segment_type": "single_patient",
                    "element_count": len(layout_elements),
                    "full_page": True,
                    "text_length": len(extracted_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Single patient record creation failed: {e}")
            raise
    
    def _extract_patient_identifier(self, text: str) -> Optional[str]:
        """Extract patient name or identifier from text."""
        try:
            # Look for name patterns
            name_patterns = [
                r"(?:patient\s+name|name)\s*:?\s*([A-Za-z\s,.-]+)(?:\n|$)",
                r"(?:pt\s+name|patient)\s*:?\s*([A-Za-z\s,.-]+)(?:\n|$)",
                r"^([A-Z][a-z]+\s+[A-Z][a-z]+)",  # Simple First Last pattern
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    # Clean up the name
                    name = re.sub(r'[^\w\s.-]', '', name)
                    if len(name) > 3 and len(name.split()) >= 2:
                        return name
            
            # Look for ID patterns as fallback
            id_patterns = [
                r"(?:patient\s+id|id|mrn)\s*:?\s*([A-Z0-9-]+)",
                r"(?:account|acct)\s*:?\s*([A-Z0-9-]+)"
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Patient identifier extraction failed: {e}")
            return None
    
    def _calculate_record_confidence(
        self, 
        elements: List[LayoutElement], 
        text: str
    ) -> float:
        """Calculate confidence score for a patient record."""
        try:
            confidence_factors = []
            
            # Factor 1: Presence of key medical fields
            medical_indicators = [
                r"patient\s+name", r"dob", r"date\s+of\s+birth",
                r"cpt", r"dx", r"diagnosis", r"procedure",
                r"provider", r"doctor", r"insurance"
            ]
            
            found_indicators = 0
            for indicator in medical_indicators:
                if re.search(indicator, text, re.IGNORECASE):
                    found_indicators += 1
            
            confidence_factors.append(min(found_indicators / len(medical_indicators), 1.0))
            
            # Factor 2: Text quality and length
            if len(text) > 100:
                confidence_factors.append(0.8)
            elif len(text) > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Factor 3: Number of layout elements
            if len(elements) > 10:
                confidence_factors.append(0.9)
            elif len(elements) > 5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Factor 4: Presence of structured data (codes, dates, etc.)
            structured_patterns = [
                r'\b\d{5}\b',  # CPT codes
                r'\b[A-Z]\d{2,3}\.\d{1,2}\b',  # ICD codes
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'  # Dates
            ]
            
            structured_count = 0
            for pattern in structured_patterns:
                structured_count += len(re.findall(pattern, text))
            
            if structured_count > 5:
                confidence_factors.append(0.9)
            elif structured_count > 2:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Calculate weighted average
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            return round(overall_confidence, 2)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence


# Global segmenter instance
patient_segmenter = PatientSegmenter()