"""
LLM Data Extraction Module
Handles structured data extraction using Azure GPT-5 with HIPAA-compliant prompts.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from src.services.llm_service import llm_service
from src.extraction.field_manager import field_manager
from src.extraction.schema_generator import schema_generator

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a data extraction operation."""
    success: bool
    extracted_data: Dict[str, Any]
    confidence_score: float
    field_confidence: Dict[str, float]
    missing_fields: List[str]
    extraction_notes: List[str]
    raw_response: str
    processing_time: float
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "extracted_data": self.extracted_data,
            "confidence_score": self.confidence_score,
            "field_confidence": self.field_confidence,
            "missing_fields": self.missing_fields,
            "extraction_notes": self.extraction_notes,
            "processing_time": self.processing_time,
            "retry_count": self.retry_count
        }


class LLMExtractor:
    """Handles structured data extraction using LLM with medical expertise."""
    
    def __init__(self):
        self.max_retries = 3
        self.extraction_cache = {}
        self.hipaa_compliance_enabled = True
    
    def extract_structured_data(
        self,
        text_content: str,
        selected_fields: List[str],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract structured medical data from text using LLM.
        
        Args:
            text_content: Raw text to extract data from
            selected_fields: List of field names to extract
            patient_context: Additional context about the patient/document
            
        Returns:
            ExtractionResult with structured data and metadata
        """
        start_time = time.time()
        retry_count = 0
        
        try:
            logger.info(f"Starting structured extraction for {len(selected_fields)} fields")
            
            # Generate extraction schema and template
            extraction_schema = schema_generator.generate_llm_prompt_schema(selected_fields)
            
            # Build comprehensive extraction prompt
            extraction_prompt = self._build_extraction_prompt(
                text_content, extraction_schema, selected_fields, patient_context
            )
            
            # Attempt extraction with retries
            for attempt in range(self.max_retries + 1):
                try:
                    raw_response = self._call_llm_for_extraction(extraction_prompt)
                    
                    # Parse and validate response
                    extracted_data, parsing_errors = self._parse_extraction_response(
                        raw_response, selected_fields
                    )
                    
                    if extracted_data:
                        # Calculate confidence and quality metrics
                        result = self._analyze_extraction_quality(
                            extracted_data, selected_fields, text_content, parsing_errors
                        )
                        
                        result.raw_response = raw_response
                        result.processing_time = time.time() - start_time
                        result.retry_count = retry_count
                        
                        logger.info(f"Extraction successful on attempt {attempt + 1}")
                        return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                    retry_count += 1
                    
                    if attempt < self.max_retries:
                        # Add parsing guidance for retry
                        extraction_prompt += self._get_json_retry_guidance()
                        continue
                        
                except Exception as e:
                    logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
                    retry_count += 1
                    
                    if attempt < self.max_retries:
                        time.sleep(1)  # Brief pause before retry
                        continue
            
            # All attempts failed
            logger.error("All extraction attempts failed")
            return ExtractionResult(
                success=False,
                extracted_data={},
                confidence_score=0.0,
                field_confidence={},
                missing_fields=selected_fields,
                extraction_notes=["All extraction attempts failed"],
                raw_response="",
                processing_time=time.time() - start_time,
                retry_count=retry_count
            )
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            return ExtractionResult(
                success=False,
                extracted_data={},
                confidence_score=0.0,
                field_confidence={},
                missing_fields=selected_fields,
                extraction_notes=[f"Extraction error: {e}"],
                raw_response="",
                processing_time=time.time() - start_time,
                retry_count=retry_count
            )
    
    def _build_extraction_prompt(
        self,
        text_content: str,
        schema: str,
        selected_fields: List[str],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive extraction prompt with HIPAA compliance."""
        
        # Get field definitions for context
        field_descriptions = []
        for field_name in selected_fields:
            field_def = field_manager.get_field_definition(field_name)
            if field_def:
                hints = ", ".join(field_def.extraction_hints) if field_def.extraction_hints else "N/A"
                examples = ", ".join(field_def.examples[:3]) if field_def.examples else "N/A"
                
                field_descriptions.append(
                    f"• {field_name} ({field_def.display_name}): {field_def.description}\n"
                    f"  Search for: {hints}\n"
                    f"  Examples: {examples}"
                )
        
        prompt = f"""You are a medical data extraction specialist with expertise in processing medical superbills and healthcare documents. Your task is to extract structured data from the provided medical document text.

HIPAA COMPLIANCE NOTICE:
- Handle all patient health information (PHI) with strict confidentiality
- Extract only the requested medical data fields
- Do not make assumptions about missing or unclear information
- Maintain accuracy and precision in medical code extraction
- Never fabricate or guess medical information

EXTRACTION TASK:
Extract the following medical fields from the document text:

{chr(10).join(field_descriptions)}

DOCUMENT TEXT TO ANALYZE:
{text_content}

EXTRACTION REQUIREMENTS:
1. Extract ONLY the fields listed above
2. Use exact values found in the document - do not modify or interpret
3. For missing information, use null values (empty string for text, empty array for lists)
4. For medical codes (CPT, ICD), preserve exact formatting
5. For dates, maintain the original format found in the document
6. For arrays, extract all instances found

OUTPUT FORMAT:
Return your response as a valid JSON object following this exact structure:
{schema}

IMPORTANT INSTRUCTIONS:
- Return ONLY the JSON object, no additional text or explanation
- Ensure all JSON syntax is correct (proper quotes, commas, brackets)
- Use null for missing fields, never leave them undefined
- Double-check medical codes for accuracy
- Maintain patient privacy throughout the extraction process

Begin extraction now:"""

        return prompt
    
    def _call_llm_for_extraction(self, prompt: str) -> str:
        """Call LLM service for data extraction."""
        try:
            response = llm_service.client.chat.completions.create(
                model=llm_service.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2500,
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for medical data extraction."""
        return """You are an expert medical data extraction specialist with extensive knowledge of medical terminology, coding systems (CPT, ICD-10), and healthcare documentation.

Your expertise includes:
- Medical superbill processing and billing documentation
- CPT procedure code identification and validation
- ICD-10 diagnosis code recognition
- Healthcare provider information extraction
- Insurance and billing data processing
- Patient demographic information handling

CORE PRINCIPLES:
1. ACCURACY: Extract information exactly as written in the document
2. COMPLETENESS: Identify all instances of requested data types
3. PRECISION: Maintain exact formatting for medical codes and identifiers
4. CONFIDENTIALITY: Handle all PHI with HIPAA compliance standards
5. CONSISTENCY: Use standardized formats for dates and structured data

MEDICAL CODE EXPERTISE:
- CPT codes: 5-digit numeric codes for procedures and services
- ICD-10 codes: Alphanumeric codes starting with letter (A-Z) followed by numbers
- Modifier codes: 2-character additions to CPT codes
- Place of service codes: 2-digit location identifiers

Always prioritize accuracy over speed and ask for clarification if document content is ambiguous or illegible."""
    
    def _parse_extraction_response(
        self, 
        raw_response: str, 
        selected_fields: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Parse and validate the LLM extraction response."""
        parsing_errors = []
        
        try:
            # Clean the response
            cleaned_response = self._clean_json_response(raw_response)
            
            # Parse JSON
            extracted_data = json.loads(cleaned_response)
            
            # Validate structure
            if not isinstance(extracted_data, dict):
                parsing_errors.append("Response is not a JSON object")
                return {}, parsing_errors
            
            # Clean and validate fields
            cleaned_data = {}
            for field_name in selected_fields:
                if field_name in extracted_data:
                    field_value = extracted_data[field_name]
                    
                    # Clean field value
                    cleaned_value = self._clean_field_value(field_value, field_name)
                    if cleaned_value is not None:
                        cleaned_data[field_name] = cleaned_value
                    else:
                        parsing_errors.append(f"Invalid value for field {field_name}")
                else:
                    # Field not found in response
                    cleaned_data[field_name] = self._get_default_field_value(field_name)
                    parsing_errors.append(f"Field {field_name} not found in response")
            
            return cleaned_data, parsing_errors
            
        except json.JSONDecodeError as e:
            parsing_errors.append(f"JSON decode error: {e}")
            return {}, parsing_errors
            
        except Exception as e:
            parsing_errors.append(f"Parsing error: {e}")
            return {}, parsing_errors
    
    def _clean_json_response(self, raw_response: str) -> str:
        """Clean and prepare JSON response for parsing."""
        # Remove any text before the first '{'
        start_idx = raw_response.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", raw_response, 0)
        
        # Remove any text after the last '}'
        end_idx = raw_response.rfind('}')
        if end_idx == -1:
            raise json.JSONDecodeError("No complete JSON object found", raw_response, 0)
        
        cleaned = raw_response[start_idx:end_idx + 1]
        
        # Fix common JSON issues
        cleaned = cleaned.replace('\\n', '\\n')  # Fix newline escaping
        cleaned = cleaned.replace('\\"', '"')    # Fix quote escaping issues
        
        return cleaned
    
    def _clean_field_value(self, value: Any, field_name: str) -> Any:
        """Clean and validate a field value."""
        if value is None or value == "":
            return None
        
        field_def = field_manager.get_field_definition(field_name)
        if not field_def:
            return value
        
        try:
            # Clean based on field type
            if field_def.field_type.value == "array":
                if isinstance(value, list):
                    # Clean array items
                    cleaned_items = []
                    for item in value:
                        if item and str(item).strip():
                            cleaned_items.append(str(item).strip())
                    return cleaned_items
                elif isinstance(value, str) and value.strip():
                    # Single value that should be an array
                    return [value.strip()]
                else:
                    return []
                    
            elif field_def.field_type.value == "number":
                if isinstance(value, (int, float)):
                    return value
                elif isinstance(value, str):
                    # Try to parse number from string
                    cleaned_str = value.replace('$', '').replace(',', '').strip()
                    try:
                        return float(cleaned_str) if '.' in cleaned_str else int(cleaned_str)
                    except ValueError:
                        return None
                        
            elif field_def.field_type.value == "boolean":
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', 'yes', '1', 'checked', 'selected']
                    
            else:  # String types
                if isinstance(value, str):
                    return value.strip()
                else:
                    return str(value).strip()
                    
        except Exception as e:
            logger.warning(f"Error cleaning field {field_name}: {e}")
            return None
    
    def _get_default_field_value(self, field_name: str) -> Any:
        """Get default value for a missing field."""
        field_def = field_manager.get_field_definition(field_name)
        if not field_def:
            return None
        
        if field_def.field_type.value == "array":
            return []
        elif field_def.field_type.value == "number":
            return None
        elif field_def.field_type.value == "boolean":
            return None
        else:
            return None
    
    def _analyze_extraction_quality(
        self,
        extracted_data: Dict[str, Any],
        selected_fields: List[str],
        original_text: str,
        parsing_errors: List[str]
    ) -> ExtractionResult:
        """Analyze the quality of extraction and calculate confidence scores."""
        try:
            field_confidence = {}
            missing_fields = []
            extraction_notes = []
            
            # Analyze each field
            total_confidence = 0
            valid_field_count = 0
            
            for field_name in selected_fields:
                field_value = extracted_data.get(field_name)
                
                if field_value is None or field_value == "" or field_value == []:
                    missing_fields.append(field_name)
                    field_confidence[field_name] = 0.0
                    extraction_notes.append(f"Field {field_name} not found or empty")
                else:
                    # Calculate field confidence based on various factors
                    confidence = self._calculate_field_confidence(
                        field_value, field_name, original_text
                    )
                    
                    field_confidence[field_name] = confidence
                    total_confidence += confidence
                    valid_field_count += 1
            
            # Calculate overall confidence
            if valid_field_count > 0:
                overall_confidence = total_confidence / len(selected_fields)
            else:
                overall_confidence = 0.0
            
            # Adjust confidence based on parsing errors
            if parsing_errors:
                overall_confidence = max(0.0, overall_confidence - (len(parsing_errors) * 0.1))
                extraction_notes.extend(parsing_errors)
            
            # Add quality notes
            if overall_confidence >= 0.9:
                extraction_notes.append("High quality extraction - all fields found with high confidence")
            elif overall_confidence >= 0.7:
                extraction_notes.append("Good quality extraction - most fields extracted successfully")
            elif overall_confidence >= 0.5:
                extraction_notes.append("Moderate quality extraction - some fields may need validation")
            else:
                extraction_notes.append("Low quality extraction - manual review recommended")
            
            return ExtractionResult(
                success=True,
                extracted_data=extracted_data,
                confidence_score=round(overall_confidence, 3),
                field_confidence=field_confidence,
                missing_fields=missing_fields,
                extraction_notes=extraction_notes,
                raw_response="",  # Will be set by caller
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return ExtractionResult(
                success=False,
                extracted_data=extracted_data,
                confidence_score=0.0,
                field_confidence={},
                missing_fields=selected_fields,
                extraction_notes=[f"Quality analysis error: {e}"],
                raw_response="",
                processing_time=0.0
            )
    
    def _calculate_field_confidence(
        self, 
        field_value: Any, 
        field_name: str, 
        original_text: str
    ) -> float:
        """Calculate confidence score for a specific field."""
        try:
            field_def = field_manager.get_field_definition(field_name)
            if not field_def:
                return 0.5  # Default moderate confidence
            
            confidence_factors = []
            
            # Factor 1: Value format validation
            if field_def.validation_pattern:
                import re
                if isinstance(field_value, str):
                    if re.match(field_def.validation_pattern, field_value):
                        confidence_factors.append(0.9)
                    else:
                        confidence_factors.append(0.6)
                elif isinstance(field_value, list):
                    valid_items = 0
                    for item in field_value:
                        if isinstance(item, str) and re.match(field_def.validation_pattern, item):
                            valid_items += 1
                    if field_value:
                        confidence_factors.append(valid_items / len(field_value))
                    else:
                        confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.8)  # No pattern to validate against
            
            # Factor 2: Value presence and quality
            if isinstance(field_value, str):
                if len(field_value) > 2:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.6)
            elif isinstance(field_value, list):
                if len(field_value) > 0:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.3)
            else:
                confidence_factors.append(0.7)
            
            # Factor 3: Contextual validation (check if value appears in original text)
            if isinstance(field_value, str) and len(field_value) > 2:
                if field_value.lower() in original_text.lower():
                    confidence_factors.append(0.95)
                else:
                    confidence_factors.append(0.7)
            elif isinstance(field_value, list):
                found_items = 0
                for item in field_value:
                    if isinstance(item, str) and item.lower() in original_text.lower():
                        found_items += 1
                if field_value:
                    confidence_factors.append(0.7 + (found_items / len(field_value)) * 0.25)
                else:
                    confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.8)
            
            # Calculate weighted average
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Confidence calculation failed for {field_name}: {e}")
            return 0.5
    
    def _get_json_retry_guidance(self) -> str:
        """Get additional guidance for JSON formatting on retry."""
        return """

IMPORTANT: Your previous response had JSON formatting issues. Please ensure:
1. Use double quotes for all keys and string values
2. No trailing commas after the last item in objects or arrays
3. Proper escaping of special characters in strings
4. Valid JSON structure with matching brackets and braces
5. Return ONLY the JSON object with no additional text"""


# Global extractor instance
llm_extractor = LLMExtractor()