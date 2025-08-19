"""
PDF Processing Pipeline
Orchestrates the complete preprocessing workflow from PDF to segmented patient data.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import streamlit as st

from src.processing.pdf_converter import pdf_converter, analyze_pdf_structure
from src.processing.image_processor import image_processor
from src.processing.layout_analyzer import layout_analyzer
from src.processing.text_extractor import text_extractor
from src.processing.patient_segmenter import patient_segmenter, PatientRecord
from src.extraction.data_processor import data_processor

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates the complete PDF processing workflow."""
    
    def __init__(self):
        self.current_session_data = {}
        self.processing_stats = {}
    
    def process_pdf_complete(
        self,
        pdf_bytes: bytes,
        filename: str,
        processing_options: Optional[Dict[str, Any]] = None,
        selected_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline from bytes to structured patient data.
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename for reference
            processing_options: Optional processing configuration
            selected_fields: Fields to extract from patient records
            
        Returns:
            Complete processing results with structured patient data
        """
        try:
            start_time = time.time()
            logger.info(f"Starting complete PDF processing for: {filename}")
            
            # Initialize processing options
            options = self._get_default_processing_options()
            if processing_options:
                options.update(processing_options)
            
            # Initialize results structure
            results = {
                "filename": filename,
                "processing_options": options,
                "pdf_analysis": {},
                "pages": [],
                "patient_records": [],
                "structured_data": [],
                "selected_fields": selected_fields or [],
                "processing_metadata": {
                    "start_time": start_time,
                    "stages_completed": [],
                    "errors": [],
                    "warnings": []
                },
                "success": False
            }
            
            # Stage 1: PDF Analysis and Structure Detection
            if st:
                st.info("🔍 Analyzing PDF structure...")
            
            try:
                pdf_analysis = analyze_pdf_structure(pdf_bytes)
                results["pdf_analysis"] = pdf_analysis
                results["processing_metadata"]["stages_completed"].append("pdf_analysis")
                logger.info(f"PDF analysis completed: {pdf_analysis.get('page_count', 0)} pages")
            except Exception as e:
                error_msg = f"PDF analysis failed: {e}"
                results["processing_metadata"]["errors"].append(error_msg)
                logger.error(error_msg)
                return results
            
            # Stage 2: PDF to Image Conversion
            if st:
                st.info("📄 Converting PDF pages to images...")
            
            try:
                page_images = pdf_converter.convert_pdf_to_images(
                    pdf_bytes, 
                    dpi=options["image_dpi"],
                    in_memory=options["use_memory_storage"]
                )
                results["processing_metadata"]["stages_completed"].append("pdf_conversion")
                logger.info(f"PDF conversion completed: {len(page_images)} pages")
            except Exception as e:
                error_msg = f"PDF conversion failed: {e}"
                results["processing_metadata"]["errors"].append(error_msg)
                logger.error(error_msg)
                return results
            
            # Stage 3: Process Each Page
            total_patient_records = []
            total_structured_data = []
            
            for page_idx, page_data in enumerate(page_images):
                try:
                    if st:
                        st.info(f"🔄 Processing page {page_idx + 1}/{len(page_images)}...")
                    
                    page_results = self._process_single_page(
                        page_data, page_idx, pdf_bytes, options, selected_fields
                    )
                    
                    results["pages"].append(page_results)
                    
                    # Collect patient records from this page
                    if page_results.get("patient_records"):
                        total_patient_records.extend(page_results["patient_records"])
                    
                    # Collect structured data from this page
                    if page_results.get("structured_data"):
                        total_structured_data.extend(page_results["structured_data"])
                    
                    logger.info(f"Page {page_idx + 1} processed successfully")
                    
                except Exception as e:
                    error_msg = f"Page {page_idx + 1} processing failed: {e}"
                    results["processing_metadata"]["errors"].append(error_msg)
                    logger.error(error_msg)
                    
                    # Continue with other pages
                    continue
            
            # Stage 4: Post-process and Finalize Results
            if st:
                st.info("🔧 Finalizing results...")
            
            results["patient_records"] = total_patient_records
            results["structured_data"] = total_structured_data
            results["processing_metadata"]["total_patients"] = len(total_patient_records)
            results["processing_metadata"]["structured_records"] = len(total_structured_data)
            results["processing_metadata"]["total_pages"] = len(page_images)
            results["processing_metadata"]["end_time"] = time.time()
            results["processing_metadata"]["total_duration"] = results["processing_metadata"]["end_time"] - start_time
            
            # Mark as successful if we have results
            if total_patient_records or results["pages"]:
                results["success"] = True
                results["processing_metadata"]["stages_completed"].append("pipeline_complete")
            
            # Cleanup temporary resources
            try:
                pdf_converter.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            
            logger.info(f"Complete PDF processing finished: {len(total_patient_records)} patients found")
            
            return results
            
        except Exception as e:
            logger.error(f"Complete PDF processing failed: {e}")
            return {
                "filename": filename,
                "success": False,
                "error": str(e),
                "processing_metadata": {
                    "errors": [str(e)]
                }
            }
    
    def _process_single_page(
        self,
        page_data: Dict[str, Any],
        page_idx: int,
        pdf_bytes: bytes,
        options: Dict[str, Any],
        selected_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a single page through the complete pipeline."""
        try:
            page_results = {
                "page_number": page_idx,
                "page_data": page_data,
                "preprocessing": {},
                "layout_analysis": {},
                "text_extraction": {},
                "patient_segmentation": {},
                "patient_records": [],
                "structured_data": [],
                "field_extraction": {},
                "processing_time": 0
            }
            
            page_start_time = time.time()
            
            # Get the page image
            image = page_data.get("image")
            if not image:
                image = pdf_converter.get_image_from_storage(page_data)
            
            # Step 1: Image Preprocessing
            if options["apply_preprocessing"]:
                try:
                    processed_image, preprocessing_metadata = image_processor.preprocess_image(
                        image,
                        enhance_contrast=options["enhance_contrast"],
                        denoise=options["denoise"],
                        correct_skew=options["correct_skew"],
                        normalize_colors=options["normalize_colors"]
                    )
                    
                    page_results["preprocessing"] = {
                        "applied": True,
                        "metadata": preprocessing_metadata,
                        "quality_metrics": image_processor.get_image_quality_metrics(processed_image)
                    }
                    
                    # Use processed image for further steps
                    working_image = processed_image
                    
                except Exception as e:
                    logger.error(f"Page {page_idx} preprocessing failed: {e}")
                    page_results["preprocessing"] = {"applied": False, "error": str(e)}
                    working_image = image
            else:
                working_image = image
                page_results["preprocessing"] = {"applied": False}
            
            # Step 2: Layout Analysis
            try:
                # Get PDF text blocks if available
                pdf_text_blocks = pdf_converter.get_page_text_bounds(pdf_bytes, page_idx)
                
                # Perform layout analysis
                layout_elements = layout_analyzer.analyze_page_layout(
                    working_image, 
                    pdf_text_blocks
                )
                
                page_results["layout_analysis"] = {
                    "elements_found": len(layout_elements),
                    "elements": [elem.to_dict() for elem in layout_elements],
                    "pdf_text_blocks": len(pdf_text_blocks)
                }
                
            except Exception as e:
                logger.error(f"Page {page_idx} layout analysis failed: {e}")
                page_results["layout_analysis"] = {"error": str(e)}
                layout_elements = []
            
            # Step 3: Text Extraction
            try:
                text_results = text_extractor.extract_text_comprehensive(
                    working_image,
                    layout_elements,
                    use_llm=options["use_llm_extraction"],
                    use_ocr_validation=options["use_ocr_validation"]
                )
                
                page_results["text_extraction"] = text_results
                
                # Get combined text for segmentation
                combined_text = ""
                if "combined_results" in text_results and "primary_text" in text_results["combined_results"]:
                    combined_text = text_results["combined_results"]["primary_text"]
                elif "llm_extraction" in text_results and "extracted_text" in text_results["llm_extraction"]:
                    combined_text = text_results["llm_extraction"]["extracted_text"]
                elif "ocr_validation" in text_results and "combined_text" in text_results["ocr_validation"]:
                    combined_text = text_results["ocr_validation"]["combined_text"]
                
            except Exception as e:
                logger.error(f"Page {page_idx} text extraction failed: {e}")
                page_results["text_extraction"] = {"error": str(e)}
                combined_text = ""
            
            # Step 4: Patient Segmentation
            try:
                patient_records = patient_segmenter.segment_patients(
                    working_image,
                    layout_elements,
                    combined_text,
                    use_llm_assistance=options["use_llm_segmentation"]
                )
                
                # Convert patient records to dictionaries for serialization
                patient_records_dict = [record.to_dict() for record in patient_records]
                
                page_results["patient_segmentation"] = {
                    "patients_found": len(patient_records),
                    "segmentation_method": "multi_method"
                }
                
                page_results["patient_records"] = patient_records_dict
                
                # Step 5: Structured Data Extraction (if fields are selected)
                if selected_fields and patient_records:
                    try:
                        structured_data = []
                        
                        for patient_record in patient_records:
                            # Process patient record through structured extraction
                            processed_data = data_processor.process_patient_record(
                                patient_record.to_dict(),
                                selected_fields,
                                enable_validation=True
                            )
                            
                            structured_data.append(processed_data.to_dict())
                        
                        page_results["structured_data"] = structured_data
                        page_results["field_extraction"] = {
                            "fields_extracted": selected_fields,
                            "records_processed": len(structured_data),
                            "avg_confidence": sum(d["confidence_score"] for d in structured_data) / len(structured_data) if structured_data else 0.0
                        }
                        
                        logger.info(f"Structured extraction completed for {len(structured_data)} patient records")
                        
                    except Exception as e:
                        logger.error(f"Page {page_idx} structured extraction failed: {e}")
                        page_results["field_extraction"] = {"error": str(e)}
                        page_results["structured_data"] = []
                else:
                    page_results["structured_data"] = []
                    page_results["field_extraction"] = {"fields_extracted": [], "records_processed": 0}
                
            except Exception as e:
                logger.error(f"Page {page_idx} patient segmentation failed: {e}")
                page_results["patient_segmentation"] = {"error": str(e)}
                page_results["patient_records"] = []
                page_results["structured_data"] = []
            
            # Calculate processing time
            page_results["processing_time"] = time.time() - page_start_time
            
            return page_results
            
        except Exception as e:
            logger.error(f"Single page processing failed for page {page_idx}: {e}")
            return {
                "page_number": page_idx,
                "error": str(e),
                "processing_time": 0
            }
    
    def _get_default_processing_options(self) -> Dict[str, Any]:
        """Get default processing options."""
        return {
            # Image conversion options
            "image_dpi": 300,
            "use_memory_storage": True,
            
            # Preprocessing options
            "apply_preprocessing": True,
            "enhance_contrast": True,
            "denoise": True,
            "correct_skew": True,
            "normalize_colors": True,
            
            # Text extraction options
            "use_llm_extraction": True,
            "use_ocr_validation": True,
            
            # Segmentation options
            "use_llm_segmentation": True,
            
            # Field extraction options
            "enable_field_extraction": True,
            "enable_validation": True
        }
    
    def get_processing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of processing results."""
        try:
            summary = {
                "filename": results.get("filename", "Unknown"),
                "success": results.get("success", False),
                "total_pages": len(results.get("pages", [])),
                "total_patients": len(results.get("patient_records", [])),
                "structured_records": len(results.get("structured_data", [])),
                "processing_time": results.get("processing_metadata", {}).get("total_duration", 0),
                "stages_completed": results.get("processing_metadata", {}).get("stages_completed", []),
                "errors": results.get("processing_metadata", {}).get("errors", []),
                "warnings": results.get("processing_metadata", {}).get("warnings", [])
            }
            
            # Calculate per-page statistics
            if results.get("pages"):
                page_processing_times = [p.get("processing_time", 0) for p in results["pages"]]
                summary["avg_page_processing_time"] = sum(page_processing_times) / len(page_processing_times)
                summary["pages_with_errors"] = len([p for p in results["pages"] if "error" in p])
            
            # Calculate confidence statistics
            structured_data = results.get("structured_data", [])
            if structured_data:
                confidences = [sd.get("confidence_score", 0) for sd in structured_data]
                summary["avg_confidence"] = sum(confidences) / len(confidences)
                summary["min_confidence"] = min(confidences)
                summary["max_confidence"] = max(confidences)
                
                # Field extraction statistics
                total_fields = sum(len(sd.get("extracted_data", {})) for sd in structured_data)
                missing_fields = sum(len(sd.get("missing_fields", [])) for sd in structured_data)
                summary["field_extraction_rate"] = 1.0 - (missing_fields / max(total_fields, 1))
            else:
                # Fallback to patient records if structured data not available
                patient_records = results.get("patient_records", [])
                if patient_records:
                    confidences = [pr.get("confidence", 0) for pr in patient_records]
                    summary["avg_confidence"] = sum(confidences) / len(confidences)
                    summary["min_confidence"] = min(confidences)
                    summary["max_confidence"] = max(confidences)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}


# Global pipeline instance
processing_pipeline = ProcessingPipeline()