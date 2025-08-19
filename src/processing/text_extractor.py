"""
Text Extraction Module
Handles text extraction using LLM analysis and OCR engines for validation and post-processing.
Combines Azure GPT-5 vision capabilities with traditional OCR for optimal accuracy.
"""

import logging
import numpy as np
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import cv2

# OCR imports with fallbacks
try:
    import paddleocr
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False

from src.config.azure_config import azure_config
from src.services.llm_service import llm_service
from src.processing.layout_analyzer import LayoutElement, BoundingBox

logger = logging.getLogger(__name__)


class TextExtractor:
    """Handles comprehensive text extraction using LLM + OCR validation."""
    
    def __init__(self):
        self.paddle_ocr = None
        self.easy_ocr = None
        self._initialize_ocr_engines()
    
    def _initialize_ocr_engines(self):
        """Initialize available OCR engines."""
        try:
            if PADDLE_OCR_AVAILABLE:
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False  # Set to True if GPU available
                )
                logger.info("PaddleOCR initialized successfully")
            
            if EASY_OCR_AVAILABLE:
                self.easy_ocr = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized successfully")
            
            if not (PADDLE_OCR_AVAILABLE or EASY_OCR_AVAILABLE):
                logger.warning("No OCR engines available, using LLM-only extraction")
                
        except Exception as e:
            logger.error(f"OCR engine initialization failed: {e}")
    
    def extract_text_comprehensive(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement],
        use_llm: bool = True,
        use_ocr_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive text extraction using LLM + OCR validation.
        
        Args:
            image: Input image to extract text from
            layout_elements: Detected layout elements with bounding boxes
            use_llm: Whether to use LLM for primary extraction
            use_ocr_validation: Whether to use OCR for validation/enhancement
            
        Returns:
            Comprehensive extraction results with confidence scores
        """
        try:
            logger.info("Starting comprehensive text extraction")
            
            extraction_results = {
                "llm_extraction": {},
                "ocr_validation": {},
                "combined_results": {},
                "confidence_scores": {},
                "processing_metadata": {
                    "image_size": image.size,
                    "layout_elements_count": len(layout_elements),
                    "methods_used": []
                }
            }
            
            # Step 1: LLM-based extraction (primary method)
            if use_llm:
                try:
                    llm_results = self._extract_with_llm(image, layout_elements)
                    extraction_results["llm_extraction"] = llm_results
                    extraction_results["processing_metadata"]["methods_used"].append("llm")
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
                    extraction_results["llm_extraction"] = {"error": str(e)}
            
            # Step 2: OCR validation and enhancement
            if use_ocr_validation and (self.paddle_ocr or self.easy_ocr):
                try:
                    ocr_results = self._extract_with_ocr(image, layout_elements)
                    extraction_results["ocr_validation"] = ocr_results
                    extraction_results["processing_metadata"]["methods_used"].append("ocr")
                except Exception as e:
                    logger.error(f"OCR validation failed: {e}")
                    extraction_results["ocr_validation"] = {"error": str(e)}
            
            # Step 3: Combine and validate results
            combined_results = self._combine_extraction_results(
                extraction_results["llm_extraction"],
                extraction_results["ocr_validation"],
                layout_elements
            )
            
            extraction_results["combined_results"] = combined_results["text_data"]
            extraction_results["confidence_scores"] = combined_results["confidence_scores"]
            
            logger.info("Comprehensive text extraction completed successfully")
            
            return extraction_results
            
        except Exception as e:
            logger.error(f"Comprehensive text extraction failed: {e}")
            return {"error": str(e)}
    
    def _extract_with_llm(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement]
    ) -> Dict[str, Any]:
        """Extract text using Azure GPT-5 vision capabilities."""
        try:
            logger.info("Starting LLM-based text extraction")
            
            # Convert image to base64 for GPT-5 vision
            image_base64 = self._image_to_base64(image)
            
            # Build extraction prompt with layout context
            extraction_prompt = self._build_llm_extraction_prompt(layout_elements)
            
            # Call GPT-5 with vision capabilities
            response = llm_service.client.chat.completions.create(
                model=llm_service.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_llm_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": extraction_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse the response
            llm_text = response.choices[0].message.content
            
            # Structure the results
            llm_results = {
                "extracted_text": llm_text,
                "confidence": 0.85,  # Base confidence for LLM
                "method": "azure_gpt5_vision",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Try to parse structured data if possible
            try:
                import json
                if llm_text.strip().startswith('{') and llm_text.strip().endswith('}'):
                    structured_data = json.loads(llm_text)
                    llm_results["structured_data"] = structured_data
            except:
                pass
            
            logger.info("LLM text extraction completed successfully")
            return llm_results
            
        except Exception as e:
            logger.error(f"LLM text extraction failed: {e}")
            return {"error": str(e)}
    
    def _extract_with_ocr(
        self,
        image: Image.Image,
        layout_elements: List[LayoutElement]
    ) -> Dict[str, Any]:
        """Extract text using OCR engines for validation."""
        try:
            logger.info("Starting OCR-based text extraction")
            
            ocr_results = {
                "paddle_ocr": None,
                "easy_ocr": None,
                "combined_text": "",
                "text_blocks": [],
                "confidence": 0.0
            }
            
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract with PaddleOCR
            if self.paddle_ocr:
                try:
                    paddle_results = self.paddle_ocr.ocr(cv_image)
                    if paddle_results and paddle_results[0]:
                        paddle_text_blocks = []
                        total_confidence = 0
                        
                        for line in paddle_results[0]:
                            if len(line) >= 2:
                                bbox_points = line[0]
                                text_info = line[1]
                                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                                confidence = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.8
                                
                                # Convert bbox points to standardized format
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                
                                bbox = BoundingBox(
                                    x0=min(x_coords),
                                    y0=min(y_coords),
                                    x1=max(x_coords),
                                    y1=max(y_coords)
                                )
                                
                                text_block = {
                                    "text": text,
                                    "bbox": bbox,
                                    "confidence": confidence,
                                    "source": "paddleocr"
                                }
                                
                                paddle_text_blocks.append(text_block)
                                total_confidence += confidence
                        
                        ocr_results["paddle_ocr"] = {
                            "text_blocks": paddle_text_blocks,
                            "avg_confidence": total_confidence / len(paddle_text_blocks) if paddle_text_blocks else 0
                        }
                        
                        logger.info(f"PaddleOCR extracted {len(paddle_text_blocks)} text blocks")
                
                except Exception as e:
                    logger.error(f"PaddleOCR extraction failed: {e}")
                    ocr_results["paddle_ocr"] = {"error": str(e)}
            
            # Extract with EasyOCR
            if self.easy_ocr:
                try:
                    easy_results = self.easy_ocr.readtext(cv_image)
                    if easy_results:
                        easy_text_blocks = []
                        total_confidence = 0
                        
                        for result in easy_results:
                            if len(result) >= 3:
                                bbox_points = result[0]
                                text = result[1]
                                confidence = result[2]
                                
                                # Convert bbox points to standardized format
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                
                                bbox = BoundingBox(
                                    x0=min(x_coords),
                                    y0=min(y_coords),
                                    x1=max(x_coords),
                                    y1=max(y_coords)
                                )
                                
                                text_block = {
                                    "text": text,
                                    "bbox": bbox,
                                    "confidence": confidence,
                                    "source": "easyocr"
                                }
                                
                                easy_text_blocks.append(text_block)
                                total_confidence += confidence
                        
                        ocr_results["easy_ocr"] = {
                            "text_blocks": easy_text_blocks,
                            "avg_confidence": total_confidence / len(easy_text_blocks) if easy_text_blocks else 0
                        }
                        
                        logger.info(f"EasyOCR extracted {len(easy_text_blocks)} text blocks")
                
                except Exception as e:
                    logger.error(f"EasyOCR extraction failed: {e}")
                    ocr_results["easy_ocr"] = {"error": str(e)}
            
            # Combine OCR results
            all_text_blocks = []
            if ocr_results["paddle_ocr"] and "text_blocks" in ocr_results["paddle_ocr"]:
                all_text_blocks.extend(ocr_results["paddle_ocr"]["text_blocks"])
            if ocr_results["easy_ocr"] and "text_blocks" in ocr_results["easy_ocr"]:
                all_text_blocks.extend(ocr_results["easy_ocr"]["text_blocks"])
            
            # Sort by reading order and combine text
            if all_text_blocks:
                sorted_blocks = sorted(all_text_blocks, key=lambda x: (x["bbox"].y0, x["bbox"].x0))
                combined_text = "\n".join([block["text"] for block in sorted_blocks])
                avg_confidence = sum([block["confidence"] for block in sorted_blocks]) / len(sorted_blocks)
                
                ocr_results.update({
                    "text_blocks": sorted_blocks,
                    "combined_text": combined_text,
                    "confidence": avg_confidence
                })
            
            logger.info("OCR text extraction completed successfully")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return {"error": str(e)}
    
    def _combine_extraction_results(
        self,
        llm_results: Dict[str, Any],
        ocr_results: Dict[str, Any],
        layout_elements: List[LayoutElement]
    ) -> Dict[str, Any]:
        """Combine LLM and OCR results for optimal accuracy."""
        try:
            combined_results = {
                "text_data": {},
                "confidence_scores": {},
                "validation_notes": []
            }
            
            # Primary source: LLM results
            if "extracted_text" in llm_results and not llm_results.get("error"):
                combined_results["text_data"]["primary_text"] = llm_results["extracted_text"]
                combined_results["confidence_scores"]["primary_confidence"] = llm_results.get("confidence", 0.8)
                
                # Include structured data if available
                if "structured_data" in llm_results:
                    combined_results["text_data"]["structured_data"] = llm_results["structured_data"]
            
            # Secondary source: OCR validation
            if "combined_text" in ocr_results and not ocr_results.get("error"):
                combined_results["text_data"]["ocr_validation_text"] = ocr_results["combined_text"]
                combined_results["confidence_scores"]["ocr_confidence"] = ocr_results.get("confidence", 0.6)
                combined_results["text_data"]["ocr_text_blocks"] = ocr_results.get("text_blocks", [])
            
            # Cross-validation and agreement analysis
            if ("primary_text" in combined_results["text_data"] and 
                "ocr_validation_text" in combined_results["text_data"]):
                
                agreement_score = self._calculate_text_agreement(
                    combined_results["text_data"]["primary_text"],
                    combined_results["text_data"]["ocr_validation_text"]
                )
                
                combined_results["confidence_scores"]["agreement_score"] = agreement_score
                
                # Adjust overall confidence based on agreement
                if agreement_score > 0.8:
                    combined_results["confidence_scores"]["overall_confidence"] = 0.95
                    combined_results["validation_notes"].append("High agreement between LLM and OCR")
                elif agreement_score > 0.6:
                    combined_results["confidence_scores"]["overall_confidence"] = 0.85
                    combined_results["validation_notes"].append("Good agreement between LLM and OCR")
                else:
                    combined_results["confidence_scores"]["overall_confidence"] = 0.70
                    combined_results["validation_notes"].append("Low agreement - manual review recommended")
            
            # Layout-aware text organization
            if layout_elements:
                organized_text = self._organize_text_by_layout(
                    combined_results["text_data"],
                    layout_elements
                )
                combined_results["text_data"]["layout_organized"] = organized_text
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return {
                "text_data": {"error": str(e)},
                "confidence_scores": {"overall_confidence": 0.0},
                "validation_notes": [f"Combination failed: {e}"]
            }
    
    def _calculate_text_agreement(self, text1: str, text2: str) -> float:
        """Calculate agreement score between two text extractions."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.0
            
            jaccard_similarity = len(intersection) / len(union)
            return jaccard_similarity
            
        except Exception:
            return 0.5  # Default moderate agreement
    
    def _organize_text_by_layout(
        self,
        text_data: Dict[str, Any],
        layout_elements: List[LayoutElement]
    ) -> Dict[str, Any]:
        """Organize extracted text according to layout structure."""
        try:
            organized = {
                "by_element_type": {},
                "reading_order": [],
                "medical_regions": {}
            }
            
            # Group by element type
            for element in layout_elements:
                element_type = element.element_type
                if element_type not in organized["by_element_type"]:
                    organized["by_element_type"][element_type] = []
                
                organized["by_element_type"][element_type].append({
                    "element_id": element.element_id,
                    "content": element.content,
                    "bbox": element.bbox.to_dict() if hasattr(element.bbox, 'to_dict') else str(element.bbox),
                    "confidence": element.confidence
                })
            
            # Sort by reading order
            sorted_elements = sorted(layout_elements, key=lambda x: (x.bbox.y0, x.bbox.x0))
            organized["reading_order"] = [
                {
                    "element_id": elem.element_id,
                    "element_type": elem.element_type,
                    "content": elem.content
                }
                for elem in sorted_elements
            ]
            
            # Group by medical regions
            for element in layout_elements:
                medical_region = element.metadata.get("medical_region", "general")
                if medical_region not in organized["medical_regions"]:
                    organized["medical_regions"][medical_region] = []
                
                organized["medical_regions"][medical_region].append({
                    "element_id": element.element_id,
                    "content": element.content,
                    "confidence": element.confidence
                })
            
            return organized
            
        except Exception as e:
            logger.error(f"Text organization failed: {e}")
            return {"error": str(e)}
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API calls."""
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=95)
            buffer.seek(0)
            
            # Convert to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Image to base64 conversion failed: {e}")
            raise
    
    def _build_llm_extraction_prompt(self, layout_elements: List[LayoutElement]) -> str:
        """Build extraction prompt with layout context."""
        prompt = """Analyze this medical superbill image and extract all visible text with high accuracy.

LAYOUT CONTEXT:
"""
        
        if layout_elements:
            prompt += f"The image contains {len(layout_elements)} detected elements:\n"
            for element in layout_elements[:10]:  # Show first 10 elements
                prompt += f"- {element.element_type} at position ({element.bbox.x0:.0f}, {element.bbox.y0:.0f})\n"
        
        prompt += """
EXTRACTION REQUIREMENTS:
1. Extract ALL visible text, including handwritten text
2. Maintain the spatial relationship and reading order
3. Pay special attention to:
   - Patient names and demographic information
   - CPT and diagnosis codes
   - Provider information
   - Insurance details
   - Dates and numerical values
4. For unclear text, include your best interpretation
5. Organize the text by logical sections when possible

Provide the extracted text in a clear, structured format that preserves the original layout and reading flow."""
        
        return prompt
    
    def _get_llm_system_prompt(self) -> str:
        """Get system prompt for LLM text extraction."""
        return """You are a medical document text extraction specialist with expertise in reading medical superbills, forms, and healthcare documents.

Your task is to accurately extract ALL visible text from medical documents, including:
- Typed and handwritten text
- Numbers, codes, and identifiers
- Form fields and their values
- Table contents and structure
- Header and footer information

Maintain high accuracy and attention to detail, especially for:
- Medical codes (CPT, ICD-10, etc.)
- Patient identifiers and demographics
- Provider information
- Financial information

When text is unclear, provide your best interpretation and note any uncertainty."""


# Global extractor instance
text_extractor = TextExtractor()