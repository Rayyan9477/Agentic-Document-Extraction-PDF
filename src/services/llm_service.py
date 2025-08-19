"""
LLM Service Module
Handles communication with Azure OpenAI GPT-5 API for structured data extraction.
Includes HIPAA-compliant logging and error handling.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from openai import AzureOpenAI
from src.config.azure_config import azure_config

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Azure OpenAI GPT-5 API."""
    
    def __init__(self):
        self.client = None
        self.deployment_name = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            if not azure_config.validate_configuration():
                raise ValueError("Invalid Azure configuration")
            
            config = azure_config.get_openai_config()
            
            self.client = AzureOpenAI(
                api_key=config['api_key'],
                api_version=config['api_version'],
                azure_endpoint=config['azure_endpoint']
            )
            
            self.deployment_name = config['deployment_name']
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Azure OpenAI API."""
        try:
            test_prompt = "Respond with 'Connection successful' if you can read this message."
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_prompt}
                ],
                max_tokens=50,
                temperature=0
            )
            
            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.deployment_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            logger.info("Azure OpenAI API connection test successful")
            return result
            
        except Exception as e:
            logger.error(f"Azure OpenAI API connection test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.deployment_name
            }
    
    def extract_structured_data(
        self, 
        text_chunk: str, 
        extraction_schema: Dict[str, Any],
        custom_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using GPT-5.
        
        Args:
            text_chunk: Raw text to extract data from
            extraction_schema: JSON schema defining expected output structure
            custom_fields: Additional custom fields to extract
            
        Returns:
            Extracted structured data as JSON
        """
        try:
            # Build the extraction prompt
            prompt = self._build_extraction_prompt(text_chunk, extraction_schema, custom_fields)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Log successful extraction (PHI-masked)
            logger.info(f"Successfully extracted data with {len(extracted_data)} fields")
            
            return {
                "success": True,
                "data": extracted_data,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"success": False, "error": f"JSON parsing error: {e}"}
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for medical data extraction."""
        return """You are a medical data extraction specialist. Your task is to extract structured information from medical superbills and related documents.

IMPORTANT GUIDELINES:
- Extract only the information that is clearly visible and legible in the provided text
- For missing or unclear information, use null values
- Maintain medical code accuracy (CPT, ICD-10, etc.)
- Preserve patient privacy by not making assumptions about missing data
- Return valid JSON format only
- If multiple patients are present, extract data for each patient separately

Focus on accuracy over completeness. It's better to return null for uncertain data than to guess."""
    
    def _build_extraction_prompt(
        self, 
        text_chunk: str, 
        extraction_schema: Dict[str, Any],
        custom_fields: Optional[List[str]] = None
    ) -> str:
        """Build the extraction prompt with schema and custom fields."""
        
        prompt = f"""Extract structured data from the following medical document text:

TEXT TO ANALYZE:
{text_chunk}

EXTRACTION SCHEMA:
{json.dumps(extraction_schema, indent=2)}

"""
        
        if custom_fields:
            prompt += f"""ADDITIONAL CUSTOM FIELDS TO EXTRACT:
{', '.join(custom_fields)}

"""
        
        prompt += """Please extract the data according to the schema above. Return ONLY valid JSON format with no additional text or explanation."""
        
        return prompt
    
    def validate_medical_codes(self, codes: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate CPT and ICD-10 codes using GPT-5's medical knowledge.
        
        Args:
            codes: Dictionary containing 'cpt_codes' and 'dx_codes' lists
            
        Returns:
            Validation results with flagged invalid codes
        """
        try:
            validation_prompt = f"""Review the following medical codes for validity and provide feedback:

CPT CODES: {codes.get('cpt_codes', [])}
ICD-10 CODES: {codes.get('dx_codes', [])}

For each code, indicate if it's:
1. Valid and properly formatted
2. Invalid or improperly formatted
3. Questionable (needs review)

Return JSON format with validation results."""

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a medical coding specialist. Validate medical codes for accuracy and proper formatting."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=1000,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            validation_results = json.loads(response.choices[0].message.content)
            logger.info("Medical code validation completed")
            
            return {
                "success": True,
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Medical code validation failed: {e}")
            return {"success": False, "error": str(e)}


# Global service instance
llm_service = LLMService()