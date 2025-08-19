"""
Schema Generation Module
Creates dynamic JSON schemas and extraction templates based on selected fields.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from src.extraction.field_manager import field_manager, FieldDefinition, FieldType

logger = logging.getLogger(__name__)


class SchemaGenerator:
    """Generates extraction schemas and templates for LLM processing."""
    
    def __init__(self):
        self.schema_cache = {}
    
    def generate_extraction_schema(
        self, 
        selected_fields: List[str],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JSON schema for data extraction.
        
        Args:
            selected_fields: List of field names to include
            include_metadata: Whether to include metadata fields
            
        Returns:
            Complete JSON schema for extraction
        """
        try:
            # Check cache first
            cache_key = f"{','.join(sorted(selected_fields))}_{include_metadata}"
            if cache_key in self.schema_cache:
                return self.schema_cache[cache_key]
            
            # Build base schema
            schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Medical Superbill Data Extraction",
                "description": "Schema for extracting structured data from medical superbill documents",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
            
            # Add selected fields
            for field_name in selected_fields:
                field_def = field_manager.get_field_definition(field_name)
                if field_def:
                    prop = self._create_field_property(field_def)
                    schema["properties"][field_name] = prop
                    
                    if field_def.required:
                        schema["required"].append(field_name)
                else:
                    logger.warning(f"Field definition not found: {field_name}")
            
            # Add metadata fields if requested
            if include_metadata:
                metadata_props = self._create_metadata_properties()
                schema["properties"].update(metadata_props)
            
            # Cache the result
            self.schema_cache[cache_key] = schema
            
            logger.info(f"Generated schema for {len(selected_fields)} fields")
            return schema
            
        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            return {}
    
    def _create_field_property(self, field_def: FieldDefinition) -> Dict[str, Any]:
        """Create JSON schema property for a field definition."""
        prop = {
            "description": field_def.description,
            "title": field_def.display_name
        }
        
        # Set type and constraints based on field type
        if field_def.field_type == FieldType.STRING:
            prop.update({
                "type": "string",
                "minLength": 1 if field_def.required else 0
            })
            
        elif field_def.field_type == FieldType.NUMBER:
            prop.update({
                "type": "number",
                "minimum": 0
            })
            
        elif field_def.field_type == FieldType.DATE:
            prop.update({
                "type": "string",
                "format": "date",
                "pattern": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"
            })
            
        elif field_def.field_type == FieldType.ARRAY:
            prop.update({
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True
            })
            
        elif field_def.field_type == FieldType.BOOLEAN:
            prop["type"] = "boolean"
            
        elif field_def.field_type == FieldType.PHONE:
            prop.update({
                "type": "string",
                "pattern": r"^[\(\)\d\s\-\.+]+$"
            })
            
        elif field_def.field_type == FieldType.EMAIL:
            prop.update({
                "type": "string",
                "format": "email"
            })
            
        elif field_def.field_type == FieldType.ADDRESS:
            prop.update({
                "type": "string",
                "minLength": 10
            })
            
        elif field_def.field_type == FieldType.MEDICAL_CODE:
            prop.update({
                "type": "string",
                "pattern": r"^[A-Za-z0-9\.\-]+$"
            })
        
        # Add validation pattern if specified
        if field_def.validation_pattern and "pattern" not in prop:
            prop["pattern"] = field_def.validation_pattern
        
        # Add examples
        if field_def.examples:
            prop["examples"] = field_def.examples
        
        # Add extraction hints as custom property
        if field_def.extraction_hints:
            prop["extractionHints"] = field_def.extraction_hints
        
        return prop
    
    def _create_metadata_properties(self) -> Dict[str, Dict[str, Any]]:
        """Create metadata properties for extraction tracking."""
        return {
            "extraction_metadata": {
                "type": "object",
                "description": "Metadata about the extraction process",
                "properties": {
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Overall confidence in extracted data"
                    },
                    "extraction_method": {
                        "type": "string",
                        "description": "Method used for extraction"
                    },
                    "field_confidence": {
                        "type": "object",
                        "description": "Per-field confidence scores",
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "missing_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields that could not be extracted"
                    },
                    "extraction_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Notes about the extraction process"
                    }
                }
            }
        }
    
    def generate_extraction_template(self, selected_fields: List[str]) -> Dict[str, Any]:
        """Generate a template with default values for extraction."""
        try:
            template = {}
            
            for field_name in selected_fields:
                field_def = field_manager.get_field_definition(field_name)
                if not field_def:
                    continue
                
                # Set appropriate default value
                if field_def.field_type == FieldType.ARRAY:
                    template[field_name] = []
                elif field_def.field_type == FieldType.NUMBER:
                    template[field_name] = None
                elif field_def.field_type == FieldType.BOOLEAN:
                    template[field_name] = None
                else:
                    template[field_name] = None
            
            # Add metadata template
            template["extraction_metadata"] = {
                "confidence_score": 0.0,
                "extraction_method": "llm_extraction",
                "field_confidence": {},
                "missing_fields": [],
                "extraction_notes": []
            }
            
            return template
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return {}
    
    def generate_llm_prompt_schema(self, selected_fields: List[str]) -> str:
        """Generate a simplified schema description for LLM prompts."""
        try:
            schema_description = "{\n"
            
            for i, field_name in enumerate(selected_fields):
                field_def = field_manager.get_field_definition(field_name)
                if not field_def:
                    continue
                
                # Add field to schema description
                if field_def.field_type == FieldType.ARRAY:
                    value_example = f"[{', '.join([f'\"{ex}\"' for ex in field_def.examples[:2]])}]" if field_def.examples else "[]"
                elif field_def.field_type == FieldType.NUMBER:
                    value_example = field_def.examples[0] if field_def.examples else "0"
                elif field_def.field_type == FieldType.BOOLEAN:
                    value_example = "true"
                else:
                    value_example = f'"{field_def.examples[0]}"' if field_def.examples else '""'
                
                # Add comma if not the last field
                comma = "," if i < len(selected_fields) - 1 else ""
                
                schema_description += f'  "{field_name}": {value_example}{comma}  // {field_def.description}\n'
            
            schema_description += "}"
            
            return schema_description
            
        except Exception as e:
            logger.error(f"LLM schema description generation failed: {e}")
            return "{}"
    
    def validate_extracted_data(
        self, 
        data: Dict[str, Any], 
        selected_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate extracted data against field definitions.
        
        Args:
            data: Extracted data to validate
            selected_fields: Fields that were expected to be extracted
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            errors = []
            
            # Check for required fields
            for field_name in selected_fields:
                field_def = field_manager.get_field_definition(field_name)
                if not field_def:
                    continue
                
                if field_def.required and (field_name not in data or not data[field_name]):
                    errors.append(f"Required field missing or empty: {field_name}")
                
                # Validate field value if present
                if field_name in data and data[field_name]:
                    field_errors = self._validate_field_value(
                        data[field_name], field_def
                    )
                    errors.extend(field_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False, [f"Validation error: {e}"]
    
    def _validate_field_value(
        self, 
        value: Any, 
        field_def: FieldDefinition
    ) -> List[str]:
        """Validate a single field value against its definition."""
        errors = []
        
        try:
            # Type validation
            if field_def.field_type == FieldType.STRING:
                if not isinstance(value, str):
                    errors.append(f"{field_def.name} must be a string")
                    
            elif field_def.field_type == FieldType.NUMBER:
                if not isinstance(value, (int, float)):
                    errors.append(f"{field_def.name} must be a number")
                    
            elif field_def.field_type == FieldType.ARRAY:
                if not isinstance(value, list):
                    errors.append(f"{field_def.name} must be an array")
                    
            elif field_def.field_type == FieldType.BOOLEAN:
                if not isinstance(value, bool):
                    errors.append(f"{field_def.name} must be a boolean")
            
            # Pattern validation
            if (field_def.validation_pattern and 
                isinstance(value, str) and 
                field_def.field_type != FieldType.ARRAY):
                
                import re
                if not re.match(field_def.validation_pattern, value):
                    errors.append(f"{field_def.name} does not match expected format")
            
            # Array pattern validation
            if (field_def.field_type == FieldType.ARRAY and 
                field_def.validation_pattern and 
                isinstance(value, list)):
                
                import re
                for item in value:
                    if isinstance(item, str) and not re.match(field_def.validation_pattern, item):
                        errors.append(f"{field_def.name} contains invalid item: {item}")
                        break
                        
        except Exception as e:
            errors.append(f"Validation error for {field_def.name}: {e}")
        
        return errors
    
    def clear_cache(self):
        """Clear the schema cache."""
        self.schema_cache.clear()
        logger.info("Schema cache cleared")


# Global schema generator instance
schema_generator = SchemaGenerator()