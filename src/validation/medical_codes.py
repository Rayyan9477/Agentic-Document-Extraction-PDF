"""
Medical Code Validation Module
Provides comprehensive validation for CPT and ICD-10 codes using databases and APIs.
"""

import logging
import re
import json
from typing import Dict, List, Any, Tuple, Optional
import requests
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class MedicalCodeValidator:
    """Validates medical codes against standard databases and APIs."""
    
    def __init__(self):
        self.cpt_cache = {}
        self.icd10_cache = {}
        self.validation_cache = {}
        
        # Load local code databases if available
        self._load_local_databases()
        
        # API configuration (can be extended for external validation services)
        self.api_config = {
            "timeout": 10,
            "max_retries": 3,
            "enable_external_apis": False  # Set to True when API keys are available
        }
    
    def _load_local_databases(self):
        """Load local medical code databases."""
        try:
            # Load CPT codes (subset for validation)
            self.cpt_database = self._get_cpt_code_database()
            
            # Load ICD-10 codes (subset for validation)
            self.icd10_database = self._get_icd10_code_database()
            
            logger.info(f"Loaded {len(self.cpt_database)} CPT codes and {len(self.icd10_database)} ICD-10 codes")
            
        except Exception as e:
            logger.error(f"Failed to load local databases: {e}")
            self.cpt_database = {}
            self.icd10_database = {}
    
    def _get_cpt_code_database(self) -> Dict[str, Dict[str, Any]]:
        """Get a subset of CPT codes for validation (in real implementation, load from file/API)."""
        return {
            # Office visits
            "99201": {"description": "Office visit, new patient, straightforward", "category": "E/M"},
            "99202": {"description": "Office visit, new patient, low complexity", "category": "E/M"},
            "99203": {"description": "Office visit, new patient, moderate complexity", "category": "E/M"},
            "99204": {"description": "Office visit, new patient, moderate to high complexity", "category": "E/M"},
            "99205": {"description": "Office visit, new patient, high complexity", "category": "E/M"},
            "99211": {"description": "Office visit, established patient, minimal", "category": "E/M"},
            "99212": {"description": "Office visit, established patient, straightforward", "category": "E/M"},
            "99213": {"description": "Office visit, established patient, low complexity", "category": "E/M"},
            "99214": {"description": "Office visit, established patient, moderate complexity", "category": "E/M"},
            "99215": {"description": "Office visit, established patient, high complexity", "category": "E/M"},
            
            # Preventive care
            "99381": {"description": "Preventive care, new patient, infant", "category": "Preventive"},
            "99382": {"description": "Preventive care, new patient, 1-4 years", "category": "Preventive"},
            "99383": {"description": "Preventive care, new patient, 5-11 years", "category": "Preventive"},
            "99384": {"description": "Preventive care, new patient, 12-17 years", "category": "Preventive"},
            "99385": {"description": "Preventive care, new patient, 18-39 years", "category": "Preventive"},
            "99386": {"description": "Preventive care, new patient, 40-64 years", "category": "Preventive"},
            "99387": {"description": "Preventive care, new patient, 65+ years", "category": "Preventive"},
            "99391": {"description": "Preventive care, established patient, infant", "category": "Preventive"},
            "99392": {"description": "Preventive care, established patient, 1-4 years", "category": "Preventive"},
            "99393": {"description": "Preventive care, established patient, 5-11 years", "category": "Preventive"},
            "99394": {"description": "Preventive care, established patient, 12-17 years", "category": "Preventive"},
            "99395": {"description": "Preventive care, established patient, 18-39 years", "category": "Preventive"},
            "99396": {"description": "Preventive care, established patient, 40-64 years", "category": "Preventive"},
            "99397": {"description": "Preventive care, established patient, 65+ years", "category": "Preventive"},
            
            # Common procedures
            "36415": {"description": "Venipuncture, routine", "category": "Laboratory"},
            "80053": {"description": "Comprehensive metabolic panel", "category": "Laboratory"},
            "80061": {"description": "Lipid panel", "category": "Laboratory"},
            "85025": {"description": "Complete blood count with differential", "category": "Laboratory"},
            "93000": {"description": "Electrocardiogram, routine", "category": "Cardiovascular"},
            "90471": {"description": "Immunization administration", "category": "Immunization"},
            "90715": {"description": "Tetanus, diphtheria toxoids and acellular pertussis vaccine", "category": "Immunization"},
            "90686": {"description": "Influenza virus vaccine, quadrivalent", "category": "Immunization"},
        }
    
    def _get_icd10_code_database(self) -> Dict[str, Dict[str, Any]]:
        """Get a subset of ICD-10 codes for validation (in real implementation, load from file/API)."""
        return {
            # Z codes (factors influencing health status)
            "Z00.00": {"description": "Encounter for general adult medical examination without abnormal findings", "category": "Factors influencing health status"},
            "Z00.01": {"description": "Encounter for general adult medical examination with abnormal findings", "category": "Factors influencing health status"},
            "Z01.00": {"description": "Encounter for examination of eyes and vision without abnormal findings", "category": "Factors influencing health status"},
            "Z01.01": {"description": "Encounter for examination of eyes and vision with abnormal findings", "category": "Factors influencing health status"},
            "Z12.11": {"description": "Encounter for screening for malignant neoplasm of colon", "category": "Factors influencing health status"},
            "Z12.31": {"description": "Encounter for screening mammography for malignant neoplasm of breast", "category": "Factors influencing health status"},
            
            # Common conditions
            "I10": {"description": "Essential (primary) hypertension", "category": "Circulatory system"},
            "I25.10": {"description": "Atherosclerotic heart disease of native coronary artery without angina pectoris", "category": "Circulatory system"},
            "E11.9": {"description": "Type 2 diabetes mellitus without complications", "category": "Endocrine, nutritional and metabolic diseases"},
            "E78.5": {"description": "Hyperlipidemia, unspecified", "category": "Endocrine, nutritional and metabolic diseases"},
            "M25.50": {"description": "Pain in unspecified joint", "category": "Musculoskeletal system"},
            "R06.00": {"description": "Dyspnea, unspecified", "category": "Symptoms, signs and abnormal clinical findings"},
            "R50.9": {"description": "Fever, unspecified", "category": "Symptoms, signs and abnormal clinical findings"},
            "J06.9": {"description": "Acute upper respiratory infection, unspecified", "category": "Respiratory system"},
            "K21.9": {"description": "Gastro-esophageal reflux disease without esophagitis", "category": "Digestive system"},
            "N39.0": {"description": "Urinary tract infection, site not specified", "category": "Genitourinary system"},
            
            # Mental health
            "F32.9": {"description": "Major depressive disorder, single episode, unspecified", "category": "Mental and behavioural disorders"},
            "F41.1": {"description": "Generalized anxiety disorder", "category": "Mental and behavioural disorders"},
            
            # Injuries
            "S72.001A": {"description": "Fracture of unspecified part of neck of right femur, initial encounter", "category": "Injury, poisoning and certain other consequences"},
            "T14.8XXA": {"description": "Other injury of unspecified body region, initial encounter", "category": "Injury, poisoning and certain other consequences"},
        }
    
    def validate_cpt_codes(self, codes: List[str]) -> Dict[str, Any]:
        """
        Validate CPT codes against database and format rules.
        
        Args:
            codes: List of CPT codes to validate
            
        Returns:
            Validation results with details for each code
        """
        try:
            results = {
                "valid_codes": [],
                "invalid_codes": [],
                "warnings": [],
                "details": {}
            }
            
            for code in codes:
                if not code or not isinstance(code, str):
                    continue
                
                code = code.strip()
                validation_result = self._validate_single_cpt_code(code)
                
                results["details"][code] = validation_result
                
                if validation_result["is_valid"]:
                    results["valid_codes"].append(code)
                else:
                    results["invalid_codes"].append(code)
                
                if validation_result.get("warnings"):
                    results["warnings"].extend(validation_result["warnings"])
            
            # Overall validation summary
            results["validation_summary"] = {
                "total_codes": len(codes),
                "valid_count": len(results["valid_codes"]),
                "invalid_count": len(results["invalid_codes"]),
                "warning_count": len(results["warnings"]),
                "validation_rate": len(results["valid_codes"]) / max(len(codes), 1)
            }
            
            logger.info(f"CPT validation completed: {len(results['valid_codes'])}/{len(codes)} codes valid")
            
            return results
            
        except Exception as e:
            logger.error(f"CPT code validation failed: {e}")
            return {"error": str(e), "valid_codes": [], "invalid_codes": codes}
    
    def _validate_single_cpt_code(self, code: str) -> Dict[str, Any]:
        """Validate a single CPT code."""
        result = {
            "is_valid": False,
            "format_valid": False,
            "exists_in_database": False,
            "description": None,
            "category": None,
            "warnings": []
        }
        
        try:
            # Format validation: CPT codes are 5 digits
            if re.match(r'^\d{5}$', code):
                result["format_valid"] = True
                
                # Range validation: CPT codes typically range from 00100-99999
                code_num = int(code)
                if 100 <= code_num <= 99999:
                    # Check against local database
                    if code in self.cpt_database:
                        result["exists_in_database"] = True
                        result["description"] = self.cpt_database[code]["description"]
                        result["category"] = self.cpt_database[code]["category"]
                        result["is_valid"] = True
                    else:
                        # Code format is valid but not in our database
                        result["warnings"].append(f"CPT code {code} not found in local database - may still be valid")
                        result["is_valid"] = True  # Assume valid if format is correct
                else:
                    result["warnings"].append(f"CPT code {code} is outside typical range (00100-99999)")
            else:
                result["warnings"].append(f"CPT code {code} has invalid format (must be 5 digits)")
            
            return result
            
        except Exception as e:
            result["warnings"].append(f"Validation error for CPT code {code}: {e}")
            return result
    
    def validate_icd10_codes(self, codes: List[str]) -> Dict[str, Any]:
        """
        Validate ICD-10 codes against database and format rules.
        
        Args:
            codes: List of ICD-10 codes to validate
            
        Returns:
            Validation results with details for each code
        """
        try:
            results = {
                "valid_codes": [],
                "invalid_codes": [],
                "warnings": [],
                "details": {}
            }
            
            for code in codes:
                if not code or not isinstance(code, str):
                    continue
                
                code = code.strip().upper()
                validation_result = self._validate_single_icd10_code(code)
                
                results["details"][code] = validation_result
                
                if validation_result["is_valid"]:
                    results["valid_codes"].append(code)
                else:
                    results["invalid_codes"].append(code)
                
                if validation_result.get("warnings"):
                    results["warnings"].extend(validation_result["warnings"])
            
            # Overall validation summary
            results["validation_summary"] = {
                "total_codes": len(codes),
                "valid_count": len(results["valid_codes"]),
                "invalid_count": len(results["invalid_codes"]),
                "warning_count": len(results["warnings"]),
                "validation_rate": len(results["valid_codes"]) / max(len(codes), 1)
            }
            
            logger.info(f"ICD-10 validation completed: {len(results['valid_codes'])}/{len(codes)} codes valid")
            
            return results
            
        except Exception as e:
            logger.error(f"ICD-10 code validation failed: {e}")
            return {"error": str(e), "valid_codes": [], "invalid_codes": codes}
    
    def _validate_single_icd10_code(self, code: str) -> Dict[str, Any]:
        """Validate a single ICD-10 code."""
        result = {
            "is_valid": False,
            "format_valid": False,
            "exists_in_database": False,
            "description": None,
            "category": None,
            "warnings": []
        }
        
        try:
            # Format validation: ICD-10 codes follow pattern [A-Z][0-9]{2,3}\.?[0-9A-Z]*
            if re.match(r'^[A-Z]\d{2,3}\.?[0-9A-Z]*$', code):
                result["format_valid"] = True
                
                # Category validation
                first_char = code[0]
                if first_char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    # Check against local database
                    if code in self.icd10_database:
                        result["exists_in_database"] = True
                        result["description"] = self.icd10_database[code]["description"]
                        result["category"] = self.icd10_database[code]["category"]
                        result["is_valid"] = True
                    else:
                        # Code format is valid but not in our database
                        result["warnings"].append(f"ICD-10 code {code} not found in local database - may still be valid")
                        result["is_valid"] = True  # Assume valid if format is correct
                else:
                    result["warnings"].append(f"ICD-10 code {code} has invalid first character")
            else:
                result["warnings"].append(f"ICD-10 code {code} has invalid format")
            
            return result
            
        except Exception as e:
            result["warnings"].append(f"Validation error for ICD-10 code {code}: {e}")
            return result
    
    def validate_medical_codes_batch(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate medical codes for a batch of patient records.
        
        Args:
            extracted_data: List of patient records with medical codes
            
        Returns:
            Comprehensive validation results for all records
        """
        try:
            batch_results = {
                "total_records": len(extracted_data),
                "records_with_codes": 0,
                "overall_cpt_validation": {"valid_codes": [], "invalid_codes": [], "warnings": []},
                "overall_icd10_validation": {"valid_codes": [], "invalid_codes": [], "warnings": []},
                "record_validations": []
            }
            
            for i, record in enumerate(extracted_data):
                record_validation = {
                    "record_index": i,
                    "patient_id": record.get("patient_id", f"Patient_{i+1}"),
                    "cpt_validation": None,
                    "icd10_validation": None,
                    "has_codes": False
                }
                
                # Extract CPT codes
                cpt_codes = record.get("extracted_data", {}).get("cpt_codes", [])
                if cpt_codes and isinstance(cpt_codes, list):
                    record_validation["cpt_validation"] = self.validate_cpt_codes(cpt_codes)
                    record_validation["has_codes"] = True
                    
                    # Add to overall results
                    batch_results["overall_cpt_validation"]["valid_codes"].extend(
                        record_validation["cpt_validation"]["valid_codes"]
                    )
                    batch_results["overall_cpt_validation"]["invalid_codes"].extend(
                        record_validation["cpt_validation"]["invalid_codes"]
                    )
                    batch_results["overall_cpt_validation"]["warnings"].extend(
                        record_validation["cpt_validation"]["warnings"]
                    )
                
                # Extract ICD-10 codes
                icd10_codes = record.get("extracted_data", {}).get("dx_codes", [])
                if icd10_codes and isinstance(icd10_codes, list):
                    record_validation["icd10_validation"] = self.validate_icd10_codes(icd10_codes)
                    record_validation["has_codes"] = True
                    
                    # Add to overall results
                    batch_results["overall_icd10_validation"]["valid_codes"].extend(
                        record_validation["icd10_validation"]["valid_codes"]
                    )
                    batch_results["overall_icd10_validation"]["invalid_codes"].extend(
                        record_validation["icd10_validation"]["invalid_codes"]
                    )
                    batch_results["overall_icd10_validation"]["warnings"].extend(
                        record_validation["icd10_validation"]["warnings"]
                    )
                
                if record_validation["has_codes"]:
                    batch_results["records_with_codes"] += 1
                
                batch_results["record_validations"].append(record_validation)
            
            # Calculate overall statistics
            total_cpt_codes = len(batch_results["overall_cpt_validation"]["valid_codes"]) + len(batch_results["overall_cpt_validation"]["invalid_codes"])
            total_icd10_codes = len(batch_results["overall_icd10_validation"]["valid_codes"]) + len(batch_results["overall_icd10_validation"]["invalid_codes"])
            
            batch_results["summary"] = {
                "total_cpt_codes": total_cpt_codes,
                "valid_cpt_rate": len(batch_results["overall_cpt_validation"]["valid_codes"]) / max(total_cpt_codes, 1),
                "total_icd10_codes": total_icd10_codes,
                "valid_icd10_rate": len(batch_results["overall_icd10_validation"]["valid_codes"]) / max(total_icd10_codes, 1),
                "records_with_validation_issues": sum(1 for r in batch_results["record_validations"] 
                                                     if (r.get("cpt_validation", {}).get("invalid_codes") or 
                                                         r.get("icd10_validation", {}).get("invalid_codes")))
            }
            
            logger.info(f"Batch validation completed: {batch_results['records_with_codes']}/{batch_results['total_records']} records had codes")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch medical code validation failed: {e}")
            return {"error": str(e), "total_records": len(extracted_data)}
    
    def get_code_suggestions(self, invalid_code: str, code_type: str = "cpt") -> List[str]:
        """
        Provide suggestions for invalid or ambiguous codes.
        
        Args:
            invalid_code: The invalid code
            code_type: Type of code ("cpt" or "icd10")
            
        Returns:
            List of suggested valid codes
        """
        try:
            suggestions = []
            
            if code_type.lower() == "cpt":
                database = self.cpt_database
            else:
                database = self.icd10_database
            
            # Simple similarity matching (can be enhanced with fuzzy matching)
            for code, info in database.items():
                # Check for similar codes (simple numeric proximity for CPT)
                if code_type.lower() == "cpt" and invalid_code.isdigit() and code.isdigit():
                    if abs(int(code) - int(invalid_code)) <= 2:
                        suggestions.append(f"{code}: {info['description']}")
                
                # Check for partial matches in description
                if invalid_code.lower() in info['description'].lower():
                    suggestions.append(f"{code}: {info['description']}")
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Code suggestion failed: {e}")
            return []


# Global validator instance
medical_code_validator = MedicalCodeValidator()