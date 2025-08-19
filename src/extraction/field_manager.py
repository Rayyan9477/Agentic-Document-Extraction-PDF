"""
Field Management Module
Handles medical field definitions, custom field creation, and schema generation.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Enumeration of supported field types."""
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    ARRAY = "array"
    BOOLEAN = "boolean"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    MEDICAL_CODE = "medical_code"


@dataclass
class FieldDefinition:
    """Definition of a medical field for extraction."""
    name: str
    display_name: str
    field_type: FieldType
    description: str
    category: str
    required: bool = False
    validation_pattern: Optional[str] = None
    examples: List[str] = None
    extraction_hints: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.extraction_hints is None:
            self.extraction_hints = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "field_type": self.field_type.value,
            "description": self.description,
            "category": self.category,
            "required": self.required,
            "validation_pattern": self.validation_pattern,
            "examples": self.examples,
            "extraction_hints": self.extraction_hints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldDefinition':
        """Create FieldDefinition from dictionary."""
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            field_type=FieldType(data["field_type"]),
            description=data["description"],
            category=data["category"],
            required=data.get("required", False),
            validation_pattern=data.get("validation_pattern"),
            examples=data.get("examples", []),
            extraction_hints=data.get("extraction_hints", [])
        )


class FieldManager:
    """Manages medical field definitions and schema generation."""
    
    def __init__(self):
        self.predefined_fields = self._initialize_predefined_fields()
        self.custom_fields = {}
        self.field_categories = {
            "Patient Demographics": "Basic patient identification and demographic information",
            "Medical Codes": "CPT procedure codes and ICD diagnosis codes",
            "Provider Information": "Healthcare provider and practice details",
            "Insurance & Billing": "Insurance information and billing details",
            "Clinical Information": "Medical history, symptoms, and clinical notes",
            "Visit Details": "Date of service, location, and visit specifics",
            "Custom Fields": "User-defined fields for specific workflows"
        }
    
    def _initialize_predefined_fields(self) -> Dict[str, FieldDefinition]:
        """Initialize predefined medical fields."""
        fields = {}
        
        # Patient Demographics
        fields["patient_name"] = FieldDefinition(
            name="patient_name",
            display_name="Patient Name",
            field_type=FieldType.STRING,
            description="Full name of the patient",
            category="Patient Demographics",
            required=True,
            validation_pattern=r"^[A-Za-z\s\-\.,']+$",
            examples=["John Smith", "Mary Johnson-Brown", "Dr. Sarah Wilson"],
            extraction_hints=["patient name", "name", "pt name", "patient:", "full name"]
        )
        
        fields["date_of_birth"] = FieldDefinition(
            name="date_of_birth",
            display_name="Date of Birth",
            field_type=FieldType.DATE,
            description="Patient's date of birth",
            category="Patient Demographics",
            required=True,
            validation_pattern=r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
            examples=["01/15/1980", "12-25-1975", "03/08/90"],
            extraction_hints=["dob", "date of birth", "birth date", "d.o.b", "born"]
        )
        
        fields["patient_id"] = FieldDefinition(
            name="patient_id",
            display_name="Patient ID",
            field_type=FieldType.STRING,
            description="Unique patient identifier or medical record number",
            category="Patient Demographics",
            validation_pattern=r"^[A-Za-z0-9\-]+$",
            examples=["MRN123456", "PT-789012", "12345"],
            extraction_hints=["patient id", "mrn", "medical record", "account number", "id"]
        )
        
        fields["patient_address"] = FieldDefinition(
            name="patient_address",
            display_name="Patient Address",
            field_type=FieldType.ADDRESS,
            description="Patient's home address",
            category="Patient Demographics",
            examples=["123 Main St, Anytown, NY 12345", "456 Oak Ave, Springfield, IL 62701"],
            extraction_hints=["address", "street", "home address", "mailing address"]
        )
        
        fields["patient_phone"] = FieldDefinition(
            name="patient_phone",
            display_name="Patient Phone",
            field_type=FieldType.PHONE,
            description="Patient's phone number",
            category="Patient Demographics",
            validation_pattern=r"^[\(\)\d\s\-\.+]+$",
            examples=["(555) 123-4567", "555-987-6543", "+1 555 246 8013"],
            extraction_hints=["phone", "telephone", "cell", "mobile", "contact number"]
        )
        
        # Medical Codes
        fields["cpt_codes"] = FieldDefinition(
            name="cpt_codes",
            display_name="CPT Codes",
            field_type=FieldType.ARRAY,
            description="CPT procedure codes",
            category="Medical Codes",
            required=True,
            validation_pattern=r"^\d{5}$",
            examples=["99213", "99214", "36415", "80053"],
            extraction_hints=["cpt", "procedure code", "service code", "billing code"]
        )
        
        fields["dx_codes"] = FieldDefinition(
            name="dx_codes",
            display_name="Diagnosis Codes",
            field_type=FieldType.ARRAY,
            description="ICD-10 diagnosis codes",
            category="Medical Codes",
            required=True,
            validation_pattern=r"^[A-Z]\d{2,3}\.?\d*$",
            examples=["Z00.00", "M25.50", "I10", "E11.9"],
            extraction_hints=["dx", "diagnosis code", "icd", "icd-10", "condition code"]
        )
        
        fields["procedure_descriptions"] = FieldDefinition(
            name="procedure_descriptions",
            display_name="Procedure Descriptions",
            field_type=FieldType.ARRAY,
            description="Descriptions of procedures performed",
            category="Medical Codes",
            examples=["Office visit, established patient", "Venipuncture", "Basic metabolic panel"],
            extraction_hints=["procedure", "service", "treatment", "performed"]
        )
        
        fields["diagnosis_descriptions"] = FieldDefinition(
            name="diagnosis_descriptions",
            display_name="Diagnosis Descriptions",
            field_type=FieldType.ARRAY,
            description="Descriptions of diagnoses",
            category="Medical Codes",
            examples=["Routine health examination", "Joint pain", "Hypertension", "Type 2 diabetes"],
            extraction_hints=["diagnosis", "condition", "problem", "finding"]
        )
        
        # Provider Information
        fields["provider_name"] = FieldDefinition(
            name="provider_name",
            display_name="Provider Name",
            field_type=FieldType.STRING,
            description="Name of healthcare provider",
            category="Provider Information",
            examples=["Dr. Sarah Wilson", "John Smith, MD", "Mary Johnson, NP"],
            extraction_hints=["provider", "doctor", "physician", "md", "do", "np", "pa"]
        )
        
        fields["provider_npi"] = FieldDefinition(
            name="provider_npi",
            display_name="Provider NPI",
            field_type=FieldType.STRING,
            description="Provider's National Provider Identifier",
            category="Provider Information",
            validation_pattern=r"^\d{10}$",
            examples=["1234567890", "9876543210"],
            extraction_hints=["npi", "national provider", "provider id", "license"]
        )
        
        fields["practice_name"] = FieldDefinition(
            name="practice_name",
            display_name="Practice Name",
            field_type=FieldType.STRING,
            description="Name of medical practice or facility",
            category="Provider Information",
            examples=["ABC Medical Center", "Family Health Clinic", "Springfield Hospital"],
            extraction_hints=["practice", "clinic", "hospital", "medical center", "facility"]
        )
        
        fields["provider_address"] = FieldDefinition(
            name="provider_address",
            display_name="Provider Address",
            field_type=FieldType.ADDRESS,
            description="Address of healthcare provider or practice",
            category="Provider Information",
            examples=["789 Medical Blvd, Health City, CA 90210"],
            extraction_hints=["practice address", "clinic address", "provider location"]
        )
        
        # Insurance & Billing
        fields["insurance_company"] = FieldDefinition(
            name="insurance_company",
            display_name="Insurance Company",
            field_type=FieldType.STRING,
            description="Name of insurance company",
            category="Insurance & Billing",
            examples=["Blue Cross Blue Shield", "Aetna", "Medicare", "Cigna"],
            extraction_hints=["insurance", "carrier", "payer", "plan", "coverage"]
        )
        
        fields["policy_number"] = FieldDefinition(
            name="policy_number",
            display_name="Policy Number",
            field_type=FieldType.STRING,
            description="Insurance policy number",
            category="Insurance & Billing",
            examples=["ABC123456789", "POL-987654321"],
            extraction_hints=["policy", "member id", "subscriber id", "insurance id"]
        )
        
        fields["group_number"] = FieldDefinition(
            name="group_number",
            display_name="Group Number",
            field_type=FieldType.STRING,
            description="Insurance group number",
            category="Insurance & Billing",
            examples=["GRP001234", "12345-AB"],
            extraction_hints=["group", "group number", "employer group"]
        )
        
        fields["copay_amount"] = FieldDefinition(
            name="copay_amount",
            display_name="Copay Amount",
            field_type=FieldType.NUMBER,
            description="Patient copayment amount",
            category="Insurance & Billing",
            validation_pattern=r"^\$?\d+\.?\d*$",
            examples=["$25.00", "35", "50.00"],
            extraction_hints=["copay", "copayment", "co-pay", "patient pay"]
        )
        
        fields["total_charges"] = FieldDefinition(
            name="total_charges",
            display_name="Total Charges",
            field_type=FieldType.NUMBER,
            description="Total charges for services",
            category="Insurance & Billing",
            validation_pattern=r"^\$?\d+\.?\d*$",
            examples=["$150.00", "275", "89.50"],
            extraction_hints=["total", "charges", "amount", "cost", "fee"]
        )
        
        # Visit Details
        fields["date_of_service"] = FieldDefinition(
            name="date_of_service",
            display_name="Date of Service",
            field_type=FieldType.DATE,
            description="Date when services were provided",
            category="Visit Details",
            required=True,
            validation_pattern=r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
            examples=["03/15/2024", "12-10-2023", "01/05/24"],
            extraction_hints=["date of service", "service date", "visit date", "appointment date"]
        )
        
        fields["place_of_service"] = FieldDefinition(
            name="place_of_service",
            display_name="Place of Service",
            field_type=FieldType.STRING,
            description="Location where services were provided",
            category="Visit Details",
            examples=["Office", "Hospital", "Emergency Room", "Patient Home"],
            extraction_hints=["place of service", "location", "facility", "where"]
        )
        
        fields["referring_physician"] = FieldDefinition(
            name="referring_physician",
            display_name="Referring Physician",
            field_type=FieldType.STRING,
            description="Name of referring physician",
            category="Visit Details",
            examples=["Dr. Robert Brown", "Sarah Davis, MD"],
            extraction_hints=["referring", "referred by", "referral from", "ref physician"]
        )
        
        # Clinical Information
        fields["chief_complaint"] = FieldDefinition(
            name="chief_complaint",
            display_name="Chief Complaint",
            field_type=FieldType.STRING,
            description="Primary reason for visit",
            category="Clinical Information",
            examples=["Chest pain", "Annual physical", "Follow-up diabetes"],
            extraction_hints=["chief complaint", "cc", "reason for visit", "presenting complaint"]
        )
        
        fields["symptoms"] = FieldDefinition(
            name="symptoms",
            display_name="Symptoms",
            field_type=FieldType.ARRAY,
            description="Patient-reported symptoms",
            category="Clinical Information",
            examples=["headache", "nausea", "fatigue", "shortness of breath"],
            extraction_hints=["symptoms", "complaints", "patient reports", "experiencing"]
        )
        
        fields["medications"] = FieldDefinition(
            name="medications",
            display_name="Medications",
            field_type=FieldType.ARRAY,
            description="Current medications",
            category="Clinical Information",
            examples=["Metformin 500mg", "Lisinopril 10mg", "Aspirin 81mg"],
            extraction_hints=["medications", "meds", "prescriptions", "taking", "drug"]
        )
        
        return fields
    
    def get_predefined_fields(self) -> Dict[str, FieldDefinition]:
        """Get all predefined field definitions."""
        return self.predefined_fields.copy()
    
    def get_fields_by_category(self) -> Dict[str, List[FieldDefinition]]:
        """Get fields grouped by category."""
        categorized_fields = {}
        
        # Add predefined fields
        for field in self.predefined_fields.values():
            if field.category not in categorized_fields:
                categorized_fields[field.category] = []
            categorized_fields[field.category].append(field)
        
        # Add custom fields
        if self.custom_fields:
            if "Custom Fields" not in categorized_fields:
                categorized_fields["Custom Fields"] = []
            categorized_fields["Custom Fields"].extend(self.custom_fields.values())
        
        return categorized_fields
    
    def add_custom_field(
        self,
        name: str,
        display_name: str,
        field_type: FieldType,
        description: str,
        validation_pattern: Optional[str] = None,
        examples: Optional[List[str]] = None,
        extraction_hints: Optional[List[str]] = None
    ) -> bool:
        """Add a custom field definition."""
        try:
            # Validate field name
            if not self._validate_field_name(name):
                logger.error(f"Invalid field name: {name}")
                return False
            
            # Check for conflicts with existing fields
            if name in self.predefined_fields or name in self.custom_fields:
                logger.error(f"Field name already exists: {name}")
                return False
            
            # Create custom field
            custom_field = FieldDefinition(
                name=name,
                display_name=display_name,
                field_type=field_type,
                description=description,
                category="Custom Fields",
                validation_pattern=validation_pattern,
                examples=examples or [],
                extraction_hints=extraction_hints or []
            )
            
            self.custom_fields[name] = custom_field
            logger.info(f"Added custom field: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom field {name}: {e}")
            return False
    
    def remove_custom_field(self, name: str) -> bool:
        """Remove a custom field."""
        try:
            if name in self.custom_fields:
                del self.custom_fields[name]
                logger.info(f"Removed custom field: {name}")
                return True
            else:
                logger.warning(f"Custom field not found: {name}")
                return False
        except Exception as e:
            logger.error(f"Failed to remove custom field {name}: {e}")
            return False
    
    def get_field_definition(self, name: str) -> Optional[FieldDefinition]:
        """Get field definition by name."""
        if name in self.predefined_fields:
            return self.predefined_fields[name]
        elif name in self.custom_fields:
            return self.custom_fields[name]
        else:
            return None
    
    def generate_extraction_schema(self, selected_fields: List[str]) -> Dict[str, Any]:
        """Generate JSON schema for extraction based on selected fields."""
        try:
            schema = {
                "type": "object",
                "properties": {},
                "required": [],
                "description": "Medical superbill data extraction schema"
            }
            
            for field_name in selected_fields:
                field_def = self.get_field_definition(field_name)
                if not field_def:
                    logger.warning(f"Field definition not found: {field_name}")
                    continue
                
                # Build property definition
                prop_def = {
                    "description": field_def.description,
                    "examples": field_def.examples,
                    "extraction_hints": field_def.extraction_hints
                }
                
                # Set type and format based on field type
                if field_def.field_type == FieldType.STRING:
                    prop_def["type"] = "string"
                elif field_def.field_type == FieldType.NUMBER:
                    prop_def["type"] = "number"
                elif field_def.field_type == FieldType.DATE:
                    prop_def["type"] = "string"
                    prop_def["format"] = "date"
                elif field_def.field_type == FieldType.ARRAY:
                    prop_def["type"] = "array"
                    prop_def["items"] = {"type": "string"}
                elif field_def.field_type == FieldType.BOOLEAN:
                    prop_def["type"] = "boolean"
                else:
                    prop_def["type"] = "string"
                
                # Add validation pattern if available
                if field_def.validation_pattern:
                    if field_def.field_type == FieldType.ARRAY:
                        prop_def["items"]["pattern"] = field_def.validation_pattern
                    else:
                        prop_def["pattern"] = field_def.validation_pattern
                
                schema["properties"][field_name] = prop_def
                
                # Add to required fields if specified
                if field_def.required:
                    schema["required"].append(field_name)
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            return {}
    
    def generate_extraction_template(self, selected_fields: List[str]) -> Dict[str, Any]:
        """Generate a simple template for LLM extraction."""
        try:
            template = {}
            
            for field_name in selected_fields:
                field_def = self.get_field_definition(field_name)
                if not field_def:
                    continue
                
                # Set default value based on field type
                if field_def.field_type == FieldType.ARRAY:
                    template[field_name] = []
                elif field_def.field_type == FieldType.NUMBER:
                    template[field_name] = 0
                elif field_def.field_type == FieldType.BOOLEAN:
                    template[field_name] = False
                else:
                    template[field_name] = ""
            
            return template
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return {}
    
    def _validate_field_name(self, name: str) -> bool:
        """Validate field name format."""
        # Must be valid Python identifier
        if not name.isidentifier():
            return False
        
        # Must not start with underscore
        if name.startswith('_'):
            return False
        
        # Must be reasonable length
        if len(name) < 2 or len(name) > 50:
            return False
        
        return True
    
    def export_field_definitions(self) -> Dict[str, Any]:
        """Export all field definitions for backup/sharing."""
        try:
            export_data = {
                "predefined_fields": {
                    name: field.to_dict() 
                    for name, field in self.predefined_fields.items()
                },
                "custom_fields": {
                    name: field.to_dict() 
                    for name, field in self.custom_fields.items()
                }
            }
            return export_data
        except Exception as e:
            logger.error(f"Field export failed: {e}")
            return {}
    
    def import_field_definitions(self, import_data: Dict[str, Any]) -> bool:
        """Import field definitions from backup."""
        try:
            # Import custom fields only (preserve predefined fields)
            if "custom_fields" in import_data:
                for name, field_data in import_data["custom_fields"].items():
                    field_def = FieldDefinition.from_dict(field_data)
                    self.custom_fields[name] = field_def
            
            logger.info(f"Imported {len(self.custom_fields)} custom fields")
            return True
            
        except Exception as e:
            logger.error(f"Field import failed: {e}")
            return False


# Global field manager instance
field_manager = FieldManager()