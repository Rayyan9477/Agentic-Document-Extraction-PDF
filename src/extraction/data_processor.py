"""
Data Processing Module
Handles structured data extraction, validation, and post-processing.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

from src.extraction.llm_extractor import llm_extractor, ExtractionResult
from src.extraction.field_manager import field_manager
from src.extraction.schema_generator import schema_generator

logger = logging.getLogger(__name__)

# Import validation and normalization modules
try:
    from src.validation.medical_codes import medical_code_validator
    from src.validation.data_normalizer import data_normalizer
    from src.validation.comprehensive_medical_codes import comprehensive_validator
    VALIDATION_AVAILABLE = True
    COMPREHENSIVE_VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Validation modules not available: {e}")
    medical_code_validator = None
    data_normalizer = None
    comprehensive_validator = None
    VALIDATION_AVAILABLE = False
    COMPREHENSIVE_VALIDATION_AVAILABLE = False


@dataclass
class ProcessedPatientData:
    """Processed patient data with validation and metadata."""
    patient_id: str
    extracted_data: Dict[str, Any]
    validation_results: Dict[str, Any]
    confidence_score: float
    field_confidence: Dict[str, float]
    missing_fields: List[str]
    validation_errors: List[str]
    processing_notes: List[str]
    raw_extraction: ExtractionResult
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "extracted_data": self.extracted_data,
            "validation_results": self.validation_results,
            "confidence_score": self.confidence_score,
            "field_confidence": self.field_confidence,
            "missing_fields": self.missing_fields,
            "validation_errors": self.validation_errors,
            "processing_notes": self.processing_notes,
            "extraction_metadata": self.raw_extraction.to_dict()
        }


class DataProcessor:
    """Processes patient records through structured extraction and validation."""
    
    def __init__(self):
        self.medical_code_validators = {
            'cpt_codes': self._validate_cpt_codes,
            'dx_codes': self._validate_dx_codes
        }
        self.date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',
            r'^\d{1,2}/\d{1,2}/\d{2}$'
        ]
    
    def process_patient_record(
        self,
        patient_record: Dict[str, Any],
        selected_fields: List[str],
        enable_validation: bool = True
    ) -> ProcessedPatientData:
        """
        Process a single patient record through structured extraction.
        
        Args:
            patient_record: Patient record from segmentation
            selected_fields: Fields to extract
            enable_validation: Whether to perform validation
            
        Returns:
            ProcessedPatientData with extraction results
        """
        try:
            patient_id = patient_record.get("record_id", "unknown")
            extracted_text = patient_record.get("extracted_text", "")
            
            logger.info(f"Processing patient record: {patient_id}")
            
            # Step 1: Extract structured data using LLM
            extraction_result = llm_extractor.extract_structured_data(
                text_content=extracted_text,
                selected_fields=selected_fields,
                patient_context={
                    "patient_id": patient_id,
                    "source_bbox": patient_record.get("bbox", {}),
                    "element_count": patient_record.get("metadata", {}).get("element_count", 0)
                }
            )
            
            if not extraction_result.success:
                logger.error(f"Extraction failed for patient {patient_id}")
                return self._create_failed_result(patient_id, extraction_result, selected_fields)
            
            # Step 2: Post-process and normalize extracted data
            processed_data = self._post_process_extracted_data(
                extraction_result.extracted_data, selected_fields
            )
            
            # Step 2.5: Apply advanced normalization if available
            if VALIDATION_AVAILABLE and data_normalizer:
                processed_data = self._apply_advanced_normalization(processed_data, patient_id)
            
            # Step 3: Validate extracted data
            validation_results = {}
            validation_errors = []
            
            if enable_validation:
                validation_results, validation_errors = self._validate_extracted_data(
                    processed_data, selected_fields
                )
            
            # Step 4: Calculate final confidence and quality metrics
            final_confidence = self._calculate_final_confidence(
                extraction_result.confidence_score,
                validation_results,
                len(validation_errors)
            )
            
            # Step 5: Generate processing notes
            processing_notes = self._generate_processing_notes(
                extraction_result, validation_results, validation_errors
            )
            
            return ProcessedPatientData(
                patient_id=patient_id,
                extracted_data=processed_data,
                validation_results=validation_results,
                confidence_score=final_confidence,
                field_confidence=extraction_result.field_confidence,
                missing_fields=extraction_result.missing_fields,
                validation_errors=validation_errors,
                processing_notes=processing_notes,
                raw_extraction=extraction_result
            )
            
        except Exception as e:
            logger.error(f"Patient record processing failed for {patient_id}: {e}")
            
            # Create error result
            empty_extraction = ExtractionResult(
                success=False,
                extracted_data={},
                confidence_score=0.0,
                field_confidence={},
                missing_fields=selected_fields,
                extraction_notes=[f"Processing error: {e}"],
                raw_response="",
                processing_time=0.0
            )
            
            return self._create_failed_result(patient_id, empty_extraction, selected_fields)
    
    def _post_process_extracted_data(
        self, 
        raw_data: Dict[str, Any], 
        selected_fields: List[str]
    ) -> Dict[str, Any]:
        """Post-process extracted data to improve quality."""
        try:
            processed_data = {}
            
            for field_name in selected_fields:
                if field_name not in raw_data:
                    processed_data[field_name] = None
                    continue
                
                field_value = raw_data[field_name]
                field_def = field_manager.get_field_definition(field_name)
                
                if not field_def or field_value is None:
                    processed_data[field_name] = field_value
                    continue
                
                # Apply field-specific post-processing
                if VALIDATION_AVAILABLE and data_normalizer:
                    # Use advanced normalization if available
                    if field_def.field_type.value == "date":
                        processed_data[field_name] = data_normalizer.normalize_date(field_value)
                    elif field_def.field_type.value == "phone":
                        processed_data[field_name] = data_normalizer.normalize_phone_number(field_value)
                    elif field_def.field_type.value == "number":
                        processed_data[field_name] = data_normalizer.normalize_currency(field_value)
                    elif field_def.field_type.value == "array":
                        processed_data[field_name] = self._normalize_array(field_value, field_name)
                    elif field_def.field_type.value == "medical_code":
                        processed_data[field_name] = self._normalize_medical_code(field_value, field_name)
                    else:
                        if field_name in ["patient_name", "provider_name", "referring_provider"]:
                            processed_data[field_name] = data_normalizer.normalize_name(field_value)
                        elif field_name in ["patient_address", "provider_address"]:
                            processed_data[field_name] = data_normalizer.normalize_address(field_value)
                        elif field_name in ["patient_id", "ssn", "mrn", "provider_npi", "policy_number"]:
                            id_type = "auto"
                            if "npi" in field_name:
                                id_type = "npi"
                            elif "ssn" in field_name:
                                id_type = "ssn"
                            elif "policy" in field_name:
                                id_type = "policy"
                            elif "patient_id" in field_name:
                                id_type = "patient_id"
                            elif "mrn" in field_name:
                                id_type = "mrn"
                            processed_data[field_name] = data_normalizer.normalize_id(field_value, id_type)
                        else:
                            processed_data[field_name] = self._normalize_string(field_value)
                else:
                    # Fallback to basic normalization
                    if field_def.field_type.value == "date":
                        processed_data[field_name] = self._normalize_date(field_value)
                    elif field_def.field_type.value == "phone":
                        processed_data[field_name] = self._normalize_phone(field_value)
                    elif field_def.field_type.value == "number":
                        processed_data[field_name] = self._normalize_number(field_value)
                    elif field_def.field_type.value == "array":
                        processed_data[field_name] = self._normalize_array(field_value, field_name)
                    elif field_def.field_type.value == "medical_code":
                        processed_data[field_name] = self._normalize_medical_code(field_value, field_name)
                    else:
                        processed_data[field_name] = self._normalize_string(field_value)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return raw_data
    
    def _apply_advanced_normalization(self, processed_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Apply advanced normalization using the data normalizer."""
        try:
            # Create a mock record for batch normalization
            mock_record = {
                "patient_id": patient_id,
                "extracted_data": processed_data
            }
            
            # Apply batch normalization
            normalized_records = data_normalizer.normalize_batch_data([mock_record])
            
            if normalized_records and len(normalized_records) > 0:
                return normalized_records[0]["extracted_data"]
            else:
                return processed_data
                
        except Exception as e:
            logger.error(f"Advanced normalization failed for {patient_id}: {e}")
            return processed_data
    
    def _normalize_date(self, date_value: Any) -> Optional[str]:
        """Normalize date values to consistent format."""
        if not date_value or not isinstance(date_value, str):
            return None
        
        try:
            # Try to parse and reformat date
            date_str = date_value.strip()
            
            # Check if it matches expected patterns
            for pattern in self.date_patterns:
                if re.match(pattern, date_str):
                    return date_str  # Already in good format
            
            # Try to parse with common formats
            formats = ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y', '%Y-%m-%d']
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%m/%d/%Y')
                except ValueError:
                    continue
            
            # Return original if we can't parse it
            return date_str
            
        except Exception:
            return date_value
    
    def _normalize_phone(self, phone_value: Any) -> Optional[str]:
        """Normalize phone numbers to consistent format."""
        if not phone_value or not isinstance(phone_value, str):
            return None
        
        try:
            # Extract digits only
            digits = re.sub(r'[^\d]', '', phone_value)
            
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits.startswith('1'):
                return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
            else:
                return phone_value  # Return original if format is unclear
                
        except Exception:
            return phone_value
    
    def _normalize_number(self, number_value: Any) -> Optional[float]:
        """Normalize numeric values."""
        if number_value is None:
            return None
        
        try:
            if isinstance(number_value, (int, float)):
                return float(number_value)
            elif isinstance(number_value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[$,]', '', number_value.strip())
                return float(cleaned)
            else:
                return None
                
        except (ValueError, TypeError):
            return None
    
    def _normalize_array(self, array_value: Any, field_name: str) -> List[str]:
        """Normalize array values."""
        if not array_value:
            return []
        
        try:
            if isinstance(array_value, list):
                # Clean and deduplicate
                cleaned_items = []
                for item in array_value:
                    if item and str(item).strip():
                        item_str = str(item).strip()
                        if item_str not in cleaned_items:
                            cleaned_items.append(item_str)
                return cleaned_items
            elif isinstance(array_value, str):
                # Split string into array if it contains separators
                separators = [',', ';', '|', '\n']
                for sep in separators:
                    if sep in array_value:
                        items = [item.strip() for item in array_value.split(sep)]
                        return [item for item in items if item]
                
                # Single item array
                return [array_value.strip()]
            else:
                return [str(array_value)]
                
        except Exception:
            return []
    
    def _normalize_medical_code(self, code_value: Any, field_name: str) -> Optional[str]:
        """Normalize medical codes (CPT, ICD)."""
        if not code_value or not isinstance(code_value, str):
            return None
        
        try:
            code = code_value.strip().upper()
            
            # CPT codes
            if field_name == 'cpt_codes' or 'cpt' in field_name.lower():
                # Remove non-digits for CPT codes
                digits = re.sub(r'[^\d]', '', code)
                if len(digits) == 5:
                    return digits
            
            # ICD codes
            elif field_name == 'dx_codes' or 'dx' in field_name.lower() or 'icd' in field_name.lower():
                # ICD-10 format: Letter + digits + optional decimal + digits
                icd_match = re.match(r'^([A-Z])(\d{2,3})\.?(\d*)$', code)
                if icd_match:
                    letter, primary, secondary = icd_match.groups()
                    if secondary:
                        return f"{letter}{primary}.{secondary}"
                    else:
                        return f"{letter}{primary}"
            
            return code_value  # Return original if no specific formatting
            
        except Exception:
            return code_value
    
    def _normalize_string(self, string_value: Any) -> Optional[str]:
        """Normalize string values."""
        if not string_value:
            return None
        
        try:
            if isinstance(string_value, str):
                # Clean whitespace and normalize
                cleaned = ' '.join(string_value.strip().split())
                return cleaned if cleaned else None
            else:
                return str(string_value).strip()
                
        except Exception:
            return None
    
    def _validate_extracted_data(
        self, 
        extracted_data: Dict[str, Any], 
        selected_fields: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate extracted data against field definitions and medical standards."""
        validation_results = {}
        validation_errors = []
        
        try:
            # Schema validation
            schema_valid, schema_errors = schema_generator.validate_extracted_data(
                extracted_data, selected_fields
            )
            
            validation_results["schema_validation"] = {
                "valid": schema_valid,
                "errors": schema_errors
            }
            
            if not schema_valid:
                validation_errors.extend(schema_errors)
            
            # Medical code validation using comprehensive validator
            if COMPREHENSIVE_VALIDATION_AVAILABLE and comprehensive_validator:
                try:
                    # Collect all medical codes for comprehensive validation
                    all_codes = []
                    code_field_mapping = {}
                    
                    # CPT codes
                    cpt_codes = extracted_data.get('cpt_codes', [])
                    if cpt_codes:
                        if isinstance(cpt_codes, str):
                            cpt_codes = [cpt_codes]
                        for code in cpt_codes:
                            all_codes.append(code)
                            code_field_mapping[code] = 'cpt_codes'
                    
                    # ICD-10/Diagnosis codes
                    dx_codes = extracted_data.get('dx_codes', [])
                    if dx_codes:
                        if isinstance(dx_codes, str):
                            dx_codes = [dx_codes]
                        for code in dx_codes:
                            all_codes.append(code)
                            code_field_mapping[code] = 'dx_codes'
                    
                    # HCPCS codes (if any)
                    hcpcs_codes = extracted_data.get('hcpcs_codes', [])
                    if hcpcs_codes:
                        if isinstance(hcpcs_codes, str):
                            hcpcs_codes = [hcpcs_codes]
                        for code in hcpcs_codes:
                            all_codes.append(code)
                            code_field_mapping[code] = 'hcpcs_codes'
                    
                    # Run comprehensive validation
                    if all_codes:
                        import asyncio
                        validation_results_comprehensive = comprehensive_validator.validate_codes_sync(all_codes)
                        
                        # Process results
                        validated_codes = {
                            'cpt_codes': {'valid': [], 'invalid': [], 'details': {}},
                            'dx_codes': {'valid': [], 'invalid': [], 'details': {}},
                            'hcpcs_codes': {'valid': [], 'invalid': [], 'details': {}}
                        }
                        
                        for result in validation_results_comprehensive:
                            field_type = code_field_mapping.get(result.code, 'unknown')
                            if field_type in validated_codes:
                                if result.is_valid:
                                    validated_codes[field_type]['valid'].append(result.code)
                                else:
                                    validated_codes[field_type]['invalid'].append(result.code)
                                    validation_errors.extend([f"Invalid {result.code_type} code: {result.code}"] + result.errors)
                                
                                validated_codes[field_type]['details'][result.code] = {
                                    'description': result.description,
                                    'category': result.category,
                                    'confidence': result.confidence,
                                    'source': result.source,
                                    'billable': result.billable,
                                    'errors': result.errors,
                                    'warnings': result.warnings
                                }
                        
                        validation_results["comprehensive_validation"] = validated_codes
                        
                        # Generate summary
                        total_codes = len(all_codes)
                        valid_codes = sum(len(v['valid']) for v in validated_codes.values())
                        validation_results["validation_summary"] = {
                            "total_codes": total_codes,
                            "valid_codes": valid_codes,
                            "invalid_codes": total_codes - valid_codes,
                            "validation_rate": valid_codes / max(total_codes, 1)
                        }
                        
                except Exception as e:
                    logger.error(f"Comprehensive validation failed: {e}")
                    validation_errors.append(f"Comprehensive validation error: {e}")
            
            elif VALIDATION_AVAILABLE and medical_code_validator:
                # Fallback to basic validation
                cpt_codes = extracted_data.get('cpt_codes', [])
                dx_codes = extracted_data.get('dx_codes', [])
                
                if cpt_codes:
                    if isinstance(cpt_codes, str):
                        cpt_codes = [cpt_codes]
                    cpt_validation = medical_code_validator.validate_cpt_codes(cpt_codes)
                    validation_results["cpt_validation"] = cpt_validation
                    if cpt_validation.get("invalid_codes"):
                        validation_errors.extend([f"Invalid CPT code: {code}" for code in cpt_validation["invalid_codes"]])
                
                if dx_codes:
                    if isinstance(dx_codes, str):
                        dx_codes = [dx_codes]
                    icd_validation = medical_code_validator.validate_icd10_codes(dx_codes)
                    validation_results["icd10_validation"] = icd_validation
                    if icd_validation.get("invalid_codes"):
                        validation_errors.extend([f"Invalid ICD-10 code: {code}" for code in icd_validation["invalid_codes"]])
            else:
                # Basic fallback validation
                for field_name in selected_fields:
                    if field_name in self.medical_code_validators:
                        field_value = extracted_data.get(field_name)
                        if field_value:
                            validator_func = self.medical_code_validators[field_name]
                            is_valid, field_errors = validator_func(field_value)
                            
                            validation_results[f"{field_name}_validation"] = {
                                "valid": is_valid,
                                "errors": field_errors
                            }
                            
                            if not is_valid:
                                validation_errors.extend(field_errors)
            
            # Cross-field validation
            cross_validation_errors = self._cross_field_validation(extracted_data)
            if cross_validation_errors:
                validation_results["cross_field_validation"] = {
                    "valid": False,
                    "errors": cross_validation_errors
                }
                validation_errors.extend(cross_validation_errors)
            else:
                validation_results["cross_field_validation"] = {
                    "valid": True,
                    "errors": []
                }
            
            return validation_results, validation_errors
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"validation_error": str(e)}, [f"Validation system error: {e}"]
    
    def _validate_cpt_codes(self, cpt_codes: Any) -> Tuple[bool, List[str]]:
        """Validate CPT codes format and basic checks."""
        errors = []
        
        try:
            if isinstance(cpt_codes, list):
                codes_to_check = cpt_codes
            elif isinstance(cpt_codes, str):
                codes_to_check = [cpt_codes]
            else:
                return False, ["CPT codes must be string or array"]
            
            for code in codes_to_check:
                if not isinstance(code, str):
                    errors.append(f"CPT code must be string: {code}")
                    continue
                
                # Basic CPT format validation
                if not re.match(r'^\d{5}$', code):
                    errors.append(f"Invalid CPT code format: {code} (must be 5 digits)")
                
                # Basic range check (CPT codes typically range from 00100-99999)
                try:
                    code_num = int(code)
                    if code_num < 100 or code_num > 99999:
                        errors.append(f"CPT code out of range: {code}")
                except ValueError:
                    errors.append(f"CPT code not numeric: {code}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"CPT validation error: {e}"]
    
    def _validate_dx_codes(self, dx_codes: Any) -> Tuple[bool, List[str]]:
        """Validate ICD-10 diagnosis codes format."""
        errors = []
        
        try:
            if isinstance(dx_codes, list):
                codes_to_check = dx_codes
            elif isinstance(dx_codes, str):
                codes_to_check = [dx_codes]
            else:
                return False, ["Diagnosis codes must be string or array"]
            
            for code in codes_to_check:
                if not isinstance(code, str):
                    errors.append(f"Diagnosis code must be string: {code}")
                    continue
                
                # Basic ICD-10 format validation
                if not re.match(r'^[A-Z]\d{2,3}\.?\d*$', code.upper()):
                    errors.append(f"Invalid ICD-10 code format: {code}")
                
                # Category validation (basic check)
                first_char = code.upper()[0]
                if first_char not in 'ABCDEFGHIJKLMNOPRSTUVWXYZ':
                    errors.append(f"Invalid ICD-10 category: {code}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Diagnosis code validation error: {e}"]
    
    def _cross_field_validation(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Perform cross-field validation checks."""
        errors = []
        
        try:
            # Check date consistency
            dob = extracted_data.get('date_of_birth')
            dos = extracted_data.get('date_of_service')
            
            if dob and dos:
                # Parse dates for comparison
                try:
                    from datetime import datetime
                    dob_date = datetime.strptime(dob, '%m/%d/%Y')
                    dos_date = datetime.strptime(dos, '%m/%d/%Y')
                    
                    if dos_date < dob_date:
                        errors.append("Date of service cannot be before date of birth")
                    
                    # Check if patient would be unreasonably old
                    age_years = (dos_date - dob_date).days / 365.25
                    if age_years > 150:
                        errors.append("Patient age seems unrealistic (>150 years)")
                    elif age_years < 0:
                        errors.append("Patient age is negative")
                        
                except ValueError:
                    # Date parsing failed, but that's handled in field validation
                    pass
            
            # Check code consistency
            cpt_codes = extracted_data.get('cpt_codes', [])
            dx_codes = extracted_data.get('dx_codes', [])
            
            if cpt_codes and not dx_codes:
                errors.append("Warning: CPT codes found but no diagnosis codes")
            elif dx_codes and not cpt_codes:
                errors.append("Warning: Diagnosis codes found but no CPT codes")
            
            # Check required field combinations
            patient_name = extracted_data.get('patient_name')
            patient_id = extracted_data.get('patient_id')
            
            if not patient_name and not patient_id:
                errors.append("Warning: No patient identification found (name or ID)")
            
            return errors
            
        except Exception as e:
            logger.error(f"Cross-field validation failed: {e}")
            return [f"Cross-field validation error: {e}"]
    
    def _calculate_final_confidence(
        self,
        extraction_confidence: float,
        validation_results: Dict[str, Any],
        validation_error_count: int
    ) -> float:
        """Calculate final confidence score considering validation results."""
        try:
            # Start with extraction confidence
            final_confidence = extraction_confidence
            
            # Penalty for validation errors
            if validation_error_count > 0:
                error_penalty = min(0.3, validation_error_count * 0.1)
                final_confidence = max(0.0, final_confidence - error_penalty)
            
            # Bonus for successful validation
            if validation_results.get("schema_validation", {}).get("valid", False):
                final_confidence = min(1.0, final_confidence + 0.05)
            
            return round(final_confidence, 3)
            
        except Exception:
            return extraction_confidence
    
    def _generate_processing_notes(
        self,
        extraction_result: ExtractionResult,
        validation_results: Dict[str, Any],
        validation_errors: List[str]
    ) -> List[str]:
        """Generate human-readable processing notes."""
        notes = []
        
        try:
            # Extraction notes
            notes.extend(extraction_result.extraction_notes)
            
            # Confidence assessment
            if extraction_result.confidence_score >= 0.9:
                notes.append("High confidence extraction - all data appears accurate")
            elif extraction_result.confidence_score >= 0.7:
                notes.append("Good confidence extraction - most data reliable")
            elif extraction_result.confidence_score >= 0.5:
                notes.append("Moderate confidence extraction - some fields may need review")
            else:
                notes.append("Low confidence extraction - manual verification recommended")
            
            # Validation notes
            if validation_errors:
                notes.append(f"Validation issues found: {len(validation_errors)} errors")
            else:
                notes.append("All validation checks passed")
            
            # Performance notes
            if extraction_result.retry_count > 0:
                notes.append(f"Required {extraction_result.retry_count + 1} attempts for successful extraction")
            
            if extraction_result.processing_time > 5.0:
                notes.append("Processing took longer than expected")
            
            return notes
            
        except Exception as e:
            logger.error(f"Note generation failed: {e}")
            return ["Processing completed with errors"]
    
    def _create_failed_result(
        self,
        patient_id: str,
        extraction_result: ExtractionResult,
        selected_fields: List[str]
    ) -> ProcessedPatientData:
        """Create a failed processing result."""
        return ProcessedPatientData(
            patient_id=patient_id,
            extracted_data={},
            validation_results={"failed": True},
            confidence_score=0.0,
            field_confidence={},
            missing_fields=selected_fields,
            validation_errors=[],
            processing_notes=["Processing failed completely"],
            raw_extraction=extraction_result
        )


# Global data processor instance
data_processor = DataProcessor()