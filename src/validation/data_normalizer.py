"""
Data Normalization Module
Provides standardized formatting for dates, phone numbers, IDs, and other data types.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
import phonenumbers
from phonenumbers import NumberParseException

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes and standardizes various data types for consistency."""
    
    def __init__(self):
        # US date formats to try parsing
        self.date_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",  # MM/DD/YYYY variants
            "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",  # MM/DD/YY variants
            "%B %d, %Y", "%b %d, %Y",            # Month DD, YYYY
            "%d/%m/%Y", "%d-%m-%Y",              # DD/MM/YYYY (fallback)
            "%Y-%m-%d", "%Y/%m/%d",              # YYYY-MM-DD (ISO)
        ]
        
        # Phone number patterns for cleaning
        self.phone_patterns = [
            r'[\(\)\-\s\.]',  # Remove common separators
            r'[^\d\+]',       # Keep only digits and plus sign
        ]
        
        # Common ID patterns
        self.id_patterns = {
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'patient_id': r'^[A-Z]{0,3}\d{3,8}$',
            'mrn': r'^[A-Z]{0,2}\d{4,10}$',
            'npi': r'^\d{10}$',
            'policy': r'^[A-Z0-9]{6,20}$'
        }
    
    def normalize_date(self, date_value: Any, target_format: str = "%m/%d/%Y") -> Optional[str]:
        """
        Normalize date to standard US format (MM/DD/YYYY by default).
        
        Args:
            date_value: Date string, datetime object, or other date representation
            target_format: Desired output format (default: MM/DD/YYYY)
            
        Returns:
            Normalized date string or None if parsing fails
        """
        try:
            if not date_value:
                return None
            
            # Handle datetime objects
            if isinstance(date_value, (datetime, date)):
                return date_value.strftime(target_format)
            
            # Convert to string and clean
            date_str = str(date_value).strip()
            if not date_str or date_str.lower() in ['none', 'null', 'n/a', '']:
                return None
            
            # Remove common prefixes/suffixes
            date_str = re.sub(r'^(date|dob|birth|service):\s*', '', date_str, flags=re.IGNORECASE)
            date_str = re.sub(r'\s*(am|pm)$', '', date_str, flags=re.IGNORECASE)
            
            # Try parsing with various formats
            parsed_date = None
            for fmt in self.date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if not parsed_date:
                # Try more flexible parsing for formats like "Jan 15, 2024"
                try:
                    import dateutil.parser
                    parsed_date = dateutil.parser.parse(date_str)
                except:
                    logger.warning(f"Could not parse date: {date_value}")
                    return None
            
            # Validate reasonable date range (1900-2100)
            if parsed_date.year < 1900 or parsed_date.year > 2100:
                logger.warning(f"Date out of reasonable range: {parsed_date}")
                return None
            
            return parsed_date.strftime(target_format)
            
        except Exception as e:
            logger.error(f"Date normalization failed for '{date_value}': {e}")
            return None
    
    def normalize_phone_number(self, phone_value: Any, region: str = "US") -> Optional[str]:
        """
        Normalize phone number to standard format.
        
        Args:
            phone_value: Phone number string or other representation
            region: Country region code (default: US)
            
        Returns:
            Normalized phone number string or None if parsing fails
        """
        try:
            if not phone_value:
                return None
            
            phone_str = str(phone_value).strip()
            if not phone_str or phone_str.lower() in ['none', 'null', 'n/a', '']:
                return None
            
            # Remove common prefixes
            phone_str = re.sub(r'^(phone|tel|cell|mobile):\s*', '', phone_str, flags=re.IGNORECASE)
            
            # Try using phonenumbers library for proper parsing
            try:
                parsed_number = phonenumbers.parse(phone_str, region)
                if phonenumbers.is_valid_number(parsed_number):
                    # Format as (XXX) XXX-XXXX for US numbers
                    if region == "US":
                        return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL)
                    else:
                        return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            except NumberParseException:
                pass
            
            # Fallback: manual cleaning and formatting
            # Extract digits only
            digits = re.sub(r'[^\d]', '', phone_str)
            
            # Handle US phone numbers
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
            elif len(digits) == 7:
                # Local number, assume area code needed
                logger.warning(f"Phone number missing area code: {phone_value}")
                return f"XXX-{digits[:3]}-{digits[3:]}"
            else:
                # International or non-standard format
                if len(digits) >= 7:
                    return f"+{digits}"
                else:
                    logger.warning(f"Phone number too short: {phone_value}")
                    return None
            
        except Exception as e:
            logger.error(f"Phone normalization failed for '{phone_value}': {e}")
            return None
    
    def normalize_id(self, id_value: Any, id_type: str = "auto") -> Optional[str]:
        """
        Normalize ID numbers (SSN, Patient ID, MRN, NPI, etc.).
        
        Args:
            id_value: ID string or other representation
            id_type: Type of ID ("ssn", "patient_id", "mrn", "npi", "policy", "auto")
            
        Returns:
            Normalized ID string or None if parsing fails
        """
        try:
            if not id_value:
                return None
            
            id_str = str(id_value).strip().upper()
            if not id_str or id_str in ['NONE', 'NULL', 'N/A', '']:
                return None
            
            # Remove common prefixes
            id_str = re.sub(r'^(ID|MRN|SSN|NPI|POLICY):\s*', '', id_str, flags=re.IGNORECASE)
            id_str = id_str.strip()
            
            # Auto-detect ID type if not specified
            if id_type == "auto":
                id_type = self._detect_id_type(id_str)
            
            # Normalize based on type
            if id_type == "ssn":
                return self._normalize_ssn(id_str)
            elif id_type == "patient_id":
                return self._normalize_patient_id(id_str)
            elif id_type == "mrn":
                return self._normalize_mrn(id_str)
            elif id_type == "npi":
                return self._normalize_npi(id_str)
            elif id_type == "policy":
                return self._normalize_policy_number(id_str)
            else:
                # Generic ID normalization
                return self._normalize_generic_id(id_str)
            
        except Exception as e:
            logger.error(f"ID normalization failed for '{id_value}': {e}")
            return None
    
    def _detect_id_type(self, id_str: str) -> str:
        """Detect the type of ID based on pattern matching."""
        # Remove all non-alphanumeric characters for pattern matching
        clean_id = re.sub(r'[^\w]', '', id_str)
        
        if re.match(r'^\d{9}$', clean_id):  # 9 digits = SSN without separators
            return "ssn"
        elif re.match(r'^\d{10}$', clean_id):  # 10 digits = NPI
            return "npi"
        elif re.match(r'^[A-Z]{1,3}\d{3,8}$', clean_id):  # Letter prefix + digits = Patient ID
            return "patient_id"
        elif re.match(r'^[A-Z]{0,2}\d{4,10}$', clean_id):  # Optional letters + many digits = MRN
            return "mrn"
        elif re.match(r'^[A-Z0-9]{6,20}$', clean_id):  # Mixed alphanumeric = Policy
            return "policy"
        else:
            return "generic"
    
    def _normalize_ssn(self, ssn_str: str) -> Optional[str]:
        """Normalize SSN to XXX-XX-XXXX format."""
        digits = re.sub(r'[^\d]', '', ssn_str)
        if len(digits) == 9:
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        return None
    
    def _normalize_patient_id(self, pid_str: str) -> str:
        """Normalize patient ID to standard format."""
        # Keep letters and numbers, standardize format
        clean_pid = re.sub(r'[^\w]', '', pid_str)
        return clean_pid
    
    def _normalize_mrn(self, mrn_str: str) -> str:
        """Normalize Medical Record Number."""
        # Keep letters and numbers, standardize format
        clean_mrn = re.sub(r'[^\w]', '', mrn_str)
        return clean_mrn
    
    def _normalize_npi(self, npi_str: str) -> Optional[str]:
        """Normalize National Provider Identifier (10 digits)."""
        digits = re.sub(r'[^\d]', '', npi_str)
        if len(digits) == 10:
            return digits
        return None
    
    def _normalize_policy_number(self, policy_str: str) -> str:
        """Normalize insurance policy number."""
        # Remove common separators, keep alphanumeric
        clean_policy = re.sub(r'[^\w]', '', policy_str)
        return clean_policy
    
    def _normalize_generic_id(self, id_str: str) -> str:
        """Generic ID normalization - remove special characters."""
        return re.sub(r'[^\w]', '', id_str)
    
    def normalize_currency(self, currency_value: Any) -> Optional[float]:
        """
        Normalize currency amounts to float values.
        
        Args:
            currency_value: Currency string or number
            
        Returns:
            Float value or None if parsing fails
        """
        try:
            if not currency_value:
                return None
            
            currency_str = str(currency_value).strip()
            if not currency_str or currency_str.lower() in ['none', 'null', 'n/a', '']:
                return None
            
            # Remove currency symbols and formatting
            cleaned = re.sub(r'[\$\,\s]', '', currency_str)
            cleaned = re.sub(r'^(amount|total|copay|charge):\s*', '', cleaned, flags=re.IGNORECASE)
            
            # Parse as float
            try:
                amount = float(cleaned)
                return round(amount, 2)  # Round to 2 decimal places
            except ValueError:
                logger.warning(f"Could not parse currency: {currency_value}")
                return None
            
        except Exception as e:
            logger.error(f"Currency normalization failed for '{currency_value}': {e}")
            return None
    
    def normalize_name(self, name_value: Any) -> Optional[str]:
        """
        Normalize person names to standard format.
        
        Args:
            name_value: Name string
            
        Returns:
            Normalized name string or None if parsing fails
        """
        try:
            if not name_value:
                return None
            
            name_str = str(name_value).strip()
            if not name_str or name_str.lower() in ['none', 'null', 'n/a', '']:
                return None
            
            # Remove common prefixes
            name_str = re.sub(r'^(patient|name):\s*', '', name_str, flags=re.IGNORECASE)
            
            # Basic name cleaning
            # Remove extra spaces and standardize case
            words = name_str.split()
            normalized_words = []
            
            for word in words:
                # Handle common name prefixes/suffixes
                if word.upper() in ['MD', 'DR', 'MR', 'MRS', 'MS', 'JR', 'SR', 'II', 'III', 'IV']:
                    normalized_words.append(word.upper())
                elif word.lower() in ['van', 'de', 'del', 'la', 'le']:
                    normalized_words.append(word.lower())
                else:
                    # Title case for regular name parts
                    normalized_words.append(word.capitalize())
            
            return ' '.join(normalized_words)
            
        except Exception as e:
            logger.error(f"Name normalization failed for '{name_value}': {e}")
            return None
    
    def normalize_address(self, address_value: Any) -> Optional[str]:
        """
        Normalize address to standard format.
        
        Args:
            address_value: Address string
            
        Returns:
            Normalized address string or None if parsing fails
        """
        try:
            if not address_value:
                return None
            
            address_str = str(address_value).strip()
            if not address_str or address_str.lower() in ['none', 'null', 'n/a', '']:
                return None
            
            # Remove common prefixes
            address_str = re.sub(r'^(address|addr):\s*', '', address_str, flags=re.IGNORECASE)
            
            # Basic address standardization
            address_str = re.sub(r'\s+', ' ', address_str)  # Normalize spaces
            
            # Standardize common abbreviations
            abbreviations = {
                r'\bstreet\b': 'St',
                r'\bavenue\b': 'Ave',
                r'\bboulevard\b': 'Blvd',
                r'\bparkway\b': 'Pkwy',
                r'\bdrive\b': 'Dr',
                r'\blane\b': 'Ln',
                r'\broads?\b': 'Rd',
                r'\bsuite\b': 'Ste',
                r'\bapartment\b': 'Apt',
                r'\bnorth\b': 'N',
                r'\bsouth\b': 'S',
                r'\beast\b': 'E',
                r'\bwest\b': 'W'
            }
            
            for pattern, replacement in abbreviations.items():
                address_str = re.sub(pattern, replacement, address_str, flags=re.IGNORECASE)
            
            return address_str.title()
            
        except Exception as e:
            logger.error(f"Address normalization failed for '{address_value}': {e}")
            return None
    
    def normalize_batch_data(self, extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize data for a batch of patient records.
        
        Args:
            extracted_data: List of patient records with extracted data
            
        Returns:
            List of patient records with normalized data
        """
        try:
            normalized_records = []
            
            for record in extracted_data:
                normalized_record = record.copy()
                extracted = normalized_record.get("extracted_data", {})
                
                # Apply normalization to common fields
                field_normalizers = {
                    # Date fields
                    "date_of_birth": lambda x: self.normalize_date(x),
                    "date_of_service": lambda x: self.normalize_date(x),
                    "service_date": lambda x: self.normalize_date(x),
                    
                    # Phone fields
                    "patient_phone": lambda x: self.normalize_phone_number(x),
                    "provider_phone": lambda x: self.normalize_phone_number(x),
                    "phone_number": lambda x: self.normalize_phone_number(x),
                    
                    # ID fields
                    "patient_id": lambda x: self.normalize_id(x, "patient_id"),
                    "ssn": lambda x: self.normalize_id(x, "ssn"),
                    "mrn": lambda x: self.normalize_id(x, "mrn"),
                    "provider_npi": lambda x: self.normalize_id(x, "npi"),
                    "policy_number": lambda x: self.normalize_id(x, "policy"),
                    
                    # Currency fields
                    "copay_amount": lambda x: self.normalize_currency(x),
                    "total_charges": lambda x: self.normalize_currency(x),
                    "deductible": lambda x: self.normalize_currency(x),
                    "coinsurance": lambda x: self.normalize_currency(x),
                    
                    # Name fields
                    "patient_name": lambda x: self.normalize_name(x),
                    "provider_name": lambda x: self.normalize_name(x),
                    "referring_provider": lambda x: self.normalize_name(x),
                    
                    # Address fields
                    "patient_address": lambda x: self.normalize_address(x),
                    "provider_address": lambda x: self.normalize_address(x),
                }
                
                # Apply normalizers
                for field_name, normalizer in field_normalizers.items():
                    if field_name in extracted:
                        original_value = extracted[field_name]
                        normalized_value = normalizer(original_value)
                        
                        if normalized_value is not None:
                            extracted[field_name] = normalized_value
                            # Track normalization in metadata
                            if "normalization_applied" not in normalized_record:
                                normalized_record["normalization_applied"] = []
                            normalized_record["normalization_applied"].append({
                                "field": field_name,
                                "original": original_value,
                                "normalized": normalized_value
                            })
                
                normalized_records.append(normalized_record)
            
            logger.info(f"Normalized data for {len(normalized_records)} records")
            return normalized_records
            
        except Exception as e:
            logger.error(f"Batch normalization failed: {e}")
            return extracted_data  # Return original data on error


# Global normalizer instance
data_normalizer = DataNormalizer()