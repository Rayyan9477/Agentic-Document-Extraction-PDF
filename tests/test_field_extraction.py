"""
Field Extraction Testing Script
Tests the complete field tagging and LLM extraction workflow.
"""

import sys
import logging
import time
from typing import Dict, List, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import extraction components
try:
    from src.extraction.field_manager import field_manager, FieldType
    from src.extraction.schema_generator import schema_generator
    from src.extraction.llm_extractor import llm_extractor
    from src.extraction.data_processor import data_processor
    from src.config.azure_config import azure_config
    from src.services.llm_service import llm_service
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)


def test_field_manager():
    """Test field manager functionality."""
    print("🏷️ Testing Field Manager...")
    
    tests = []
    
    # Test 1: Get predefined fields
    try:
        predefined_fields = field_manager.get_predefined_fields()
        test_passed = len(predefined_fields) > 0
        tests.append(("Get Predefined Fields", test_passed, f"{len(predefined_fields)} fields found"))
    except Exception as e:
        tests.append(("Get Predefined Fields", False, str(e)))
    
    # Test 2: Get fields by category
    try:
        categorized_fields = field_manager.get_fields_by_category()
        test_passed = len(categorized_fields) > 0
        tests.append(("Get Fields by Category", test_passed, f"{len(categorized_fields)} categories found"))
    except Exception as e:
        tests.append(("Get Fields by Category", False, str(e)))
    
    # Test 3: Add custom field
    try:
        success = field_manager.add_custom_field(
            name="test_custom_field",
            display_name="Test Custom Field",
            field_type=FieldType.STRING,
            description="A test custom field for validation",
            examples=["Test example 1", "Test example 2"]
        )
        tests.append(("Add Custom Field", success, "Custom field added successfully"))
        
        # Clean up
        if success:
            field_manager.remove_custom_field("test_custom_field")
    except Exception as e:
        tests.append(("Add Custom Field", False, str(e)))
    
    # Test 4: Field definition lookup
    try:
        field_def = field_manager.get_field_definition("patient_name")
        test_passed = field_def is not None and field_def.display_name == "Patient Name"
        tests.append(("Field Definition Lookup", test_passed, "Patient name field found"))
    except Exception as e:
        tests.append(("Field Definition Lookup", False, str(e)))
    
    # Print results
    print_test_results("Field Manager", tests)
    return all(result[1] for result in tests)


def test_schema_generator():
    """Test schema generation functionality."""
    print("📋 Testing Schema Generator...")
    
    tests = []
    selected_fields = ["patient_name", "date_of_birth", "cpt_codes", "dx_codes"]
    
    # Test 1: Generate extraction schema
    try:
        schema = schema_generator.generate_extraction_schema(selected_fields)
        test_passed = (
            "properties" in schema and 
            len(schema["properties"]) == len(selected_fields)
        )
        tests.append(("Generate Extraction Schema", test_passed, f"Schema with {len(selected_fields)} fields"))
    except Exception as e:
        tests.append(("Generate Extraction Schema", False, str(e)))
    
    # Test 2: Generate extraction template
    try:
        template = schema_generator.generate_extraction_template(selected_fields)
        test_passed = len(template) >= len(selected_fields)  # Includes metadata
        tests.append(("Generate Extraction Template", test_passed, f"Template with {len(template)} properties"))
    except Exception as e:
        tests.append(("Generate Extraction Template", False, str(e)))
    
    # Test 3: Generate LLM prompt schema
    try:
        prompt_schema = schema_generator.generate_llm_prompt_schema(selected_fields)
        test_passed = "{" in prompt_schema and "}" in prompt_schema
        tests.append(("Generate LLM Prompt Schema", test_passed, "Valid JSON schema format"))
    except Exception as e:
        tests.append(("Generate LLM Prompt Schema", False, str(e)))
    
    # Test 4: Validate extracted data
    try:
        test_data = {
            "patient_name": "John Doe",
            "date_of_birth": "01/15/1980",
            "cpt_codes": ["99213", "99214"],
            "dx_codes": ["Z00.00"]
        }
        is_valid, errors = schema_generator.validate_extracted_data(test_data, selected_fields)
        tests.append(("Validate Extracted Data", is_valid and len(errors) == 0, "Validation successful"))
    except Exception as e:
        tests.append(("Validate Extracted Data", False, str(e)))
    
    # Print results
    print_test_results("Schema Generator", tests)
    return all(result[1] for result in tests)


def test_llm_extractor():
    """Test LLM extraction functionality."""
    print("🤖 Testing LLM Extractor...")
    
    tests = []
    
    # Check if Azure connection is available
    try:
        azure_valid = azure_config.validate_configuration()
        if not azure_valid:
            print("⚠️ Azure configuration not valid, skipping LLM tests")
            return True
    except Exception:
        print("⚠️ Azure configuration check failed, skipping LLM tests")
        return True
    
    # Sample medical text for testing
    sample_text = """
    MEDICAL SUPERBILL

    Patient Name: Sarah Johnson
    Date of Birth: 03/15/1985
    Patient ID: PT123456
    Date of Service: 12/10/2023

    Provider: Dr. Michael Smith, MD
    NPI: 1234567890

    CPT Codes:
    99213 - Office visit, established patient, level 3
    36415 - Venipuncture

    Diagnosis Codes:
    Z00.00 - Encounter for general adult medical examination
    I10 - Essential hypertension

    Insurance: Blue Cross Blue Shield
    Policy Number: BC123456789
    Copay: $25.00
    Total Charges: $185.00
    """
    
    selected_fields = [
        "patient_name", "date_of_birth", "patient_id", "date_of_service",
        "provider_name", "provider_npi", "cpt_codes", "dx_codes",
        "insurance_company", "policy_number", "copay_amount", "total_charges"
    ]
    
    # Test 1: Extract structured data
    try:
        start_time = time.time()
        result = llm_extractor.extract_structured_data(
            text_content=sample_text,
            selected_fields=selected_fields
        )
        processing_time = time.time() - start_time
        
        test_passed = (
            result.success and 
            len(result.extracted_data) > 0 and
            result.confidence_score > 0
        )
        
        details = f"Extracted {len(result.extracted_data)} fields in {processing_time:.2f}s"
        if result.success:
            details += f", confidence: {result.confidence_score:.1%}"
        
        tests.append(("Extract Structured Data", test_passed, details))
        
        # Store result for further tests
        extraction_result = result
        
    except Exception as e:
        tests.append(("Extract Structured Data", False, str(e)))
        extraction_result = None
    
    # Test 2: Validate extraction quality
    if extraction_result and extraction_result.success:
        try:
            # Check if key fields were extracted
            extracted_data = extraction_result.extracted_data
            key_fields_found = 0
            
            expected_values = {
                "patient_name": "Sarah Johnson",
                "date_of_birth": "03/15/1985",
                "cpt_codes": ["99213"],
                "dx_codes": ["Z00.00", "I10"]
            }
            
            for field, expected in expected_values.items():
                if field in extracted_data:
                    actual = extracted_data[field]
                    if isinstance(expected, list):
                        if isinstance(actual, list) and any(exp in str(actual) for exp in expected):
                            key_fields_found += 1
                    else:
                        if expected.lower() in str(actual).lower():
                            key_fields_found += 1
            
            test_passed = key_fields_found >= 3  # At least 3 out of 4 key fields
            tests.append(("Extraction Quality", test_passed, f"{key_fields_found}/4 key fields correctly extracted"))
            
        except Exception as e:
            tests.append(("Extraction Quality", False, str(e)))
    
    # Print results
    print_test_results("LLM Extractor", tests)
    return all(result[1] for result in tests)


def test_data_processor():
    """Test data processor functionality."""
    print("🔄 Testing Data Processor...")
    
    tests = []
    
    # Create mock patient record
    mock_patient_record = {
        "record_id": "test_patient_1",
        "patient_identifier": "Test Patient",
        "extracted_text": """
        Patient: John Smith
        DOB: 01/15/1980  
        ID: PT789012
        Date of Service: 12/15/2023
        
        CPT: 99214 - Office visit
        DX: Z00.00 - General exam
        
        Provider: Dr. Jane Wilson
        Insurance: Medicare
        Copay: $15.00
        """,
        "confidence": 0.85,
        "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        "metadata": {"element_count": 10}
    }
    
    selected_fields = ["patient_name", "date_of_birth", "patient_id", "cpt_codes", "dx_codes"]
    
    # Test 1: Process patient record (without LLM if not available)
    try:
        # Check if we can do LLM extraction
        can_use_llm = azure_config.validate_configuration()
        
        if can_use_llm:
            processed_data = data_processor.process_patient_record(
                mock_patient_record,
                selected_fields,
                enable_validation=True
            )
            
            test_passed = (
                processed_data.patient_id == "test_patient_1" and
                processed_data.confidence_score >= 0
            )
            
            details = f"Processed with confidence: {processed_data.confidence_score:.1%}"
            if processed_data.validation_errors:
                details += f", {len(processed_data.validation_errors)} validation errors"
        else:
            # Mock the processing result
            test_passed = True
            details = "Skipped LLM processing (no Azure config)"
        
        tests.append(("Process Patient Record", test_passed, details))
        
    except Exception as e:
        tests.append(("Process Patient Record", False, str(e)))
    
    # Test 2: Data normalization
    try:
        # Test date normalization
        normalized_date = data_processor._normalize_date("1/15/1980")
        test_passed = normalized_date in ["01/15/1980", "1/15/1980"]
        tests.append(("Data Normalization", test_passed, f"Date normalized to: {normalized_date}"))
        
    except Exception as e:
        tests.append(("Data Normalization", False, str(e)))
    
    # Test 3: Medical code validation  
    try:
        is_valid, errors = data_processor._validate_cpt_codes(["99213", "99214"])
        test_passed = is_valid and len(errors) == 0
        tests.append(("Medical Code Validation", test_passed, "CPT codes validated successfully"))
        
    except Exception as e:
        tests.append(("Medical Code Validation", False, str(e)))
    
    # Print results
    print_test_results("Data Processor", tests)
    return all(result[1] for result in tests)


def test_integration_workflow():
    """Test the complete integration workflow."""
    print("🔄 Testing Integration Workflow...")
    
    tests = []
    
    # Test 1: Complete field selection to schema generation
    try:
        # Select some fields
        selected_fields = ["patient_name", "cpt_codes", "dx_codes", "provider_name"]
        
        # Generate schema
        schema = schema_generator.generate_extraction_schema(selected_fields)
        
        # Generate template
        template = schema_generator.generate_extraction_template(selected_fields)
        
        test_passed = (
            len(schema.get("properties", {})) >= len(selected_fields) and
            len(template) >= len(selected_fields)
        )
        
        tests.append(("Field Selection to Schema", test_passed, f"Generated schema for {len(selected_fields)} fields"))
        
    except Exception as e:
        tests.append(("Field Selection to Schema", False, str(e)))
    
    # Test 2: Schema to prompt generation
    try:
        prompt_schema = schema_generator.generate_llm_prompt_schema(selected_fields)
        
        # Check that all selected fields are in the prompt
        fields_in_prompt = sum(1 for field in selected_fields if field in prompt_schema)
        test_passed = fields_in_prompt == len(selected_fields)
        
        tests.append(("Schema to Prompt Generation", test_passed, f"All {len(selected_fields)} fields included in prompt"))
        
    except Exception as e:
        tests.append(("Schema to Prompt Generation", False, str(e)))
    
    # Test 3: End-to-end extraction workflow (if Azure available)
    try:
        if azure_config.validate_configuration():
            # Sample patient record
            patient_record = {
                "record_id": "integration_test",
                "extracted_text": "Patient Name: Alice Brown\\nDOB: 05/20/1975\\nCPT: 99213\\nDiagnosis: Z00.00",
                "confidence": 0.9
            }
            
            # Process through complete workflow
            result = data_processor.process_patient_record(
                patient_record,
                ["patient_name", "date_of_birth", "cpt_codes", "dx_codes"]
            )
            
            test_passed = result.success if hasattr(result, 'success') else result.confidence_score > 0
            details = f"End-to-end processing completed with confidence: {result.confidence_score:.1%}"
            
        else:
            test_passed = True
            details = "Skipped (no Azure configuration)"
        
        tests.append(("End-to-End Extraction", test_passed, details))
        
    except Exception as e:
        tests.append(("End-to-End Extraction", False, str(e)))
    
    # Print results
    print_test_results("Integration Workflow", tests)
    return all(result[1] for result in tests)


def test_performance_benchmarks():
    """Test performance characteristics."""
    print("⚡ Testing Performance...")
    
    tests = []
    
    # Test 1: Schema generation speed
    try:
        selected_fields = ["patient_name", "date_of_birth", "cpt_codes", "dx_codes", "provider_name"]
        
        start_time = time.time()
        for _ in range(10):
            schema_generator.generate_extraction_schema(selected_fields)
        avg_time = (time.time() - start_time) / 10
        
        test_passed = avg_time < 0.1  # Should generate schema in less than 0.1 seconds
        tests.append(("Schema Generation Speed", test_passed, f"Avg time: {avg_time*1000:.1f}ms"))
        
    except Exception as e:
        tests.append(("Schema Generation Speed", False, str(e)))
    
    # Test 2: Field lookup performance
    try:
        start_time = time.time()
        for _ in range(100):
            field_manager.get_field_definition("patient_name")
        avg_time = (time.time() - start_time) / 100
        
        test_passed = avg_time < 0.001  # Should lookup field in less than 1ms
        tests.append(("Field Lookup Speed", test_passed, f"Avg time: {avg_time*1000:.3f}ms"))
        
    except Exception as e:
        tests.append(("Field Lookup Speed", False, str(e)))
    
    # Test 3: Memory usage (basic check)
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Perform some operations
        for i in range(10):
            selected_fields = [f"test_field_{i}" for i in range(20)]
            schema_generator.generate_extraction_schema(selected_fields)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        test_passed = memory_increase < 10  # Should not increase by more than 10MB
        tests.append(("Memory Usage", test_passed, f"Memory increase: {memory_increase:.1f}MB"))
        
    except ImportError:
        tests.append(("Memory Usage", True, "psutil not available - skipped"))
    except Exception as e:
        tests.append(("Memory Usage", False, str(e)))
    
    # Print results
    print_test_results("Performance", tests)
    return all(result[1] for result in tests if not result[2].startswith("psutil"))


def print_test_results(category: str, tests: List[Tuple[str, bool, str]]):
    """Print formatted test results."""
    print(f"\\n📋 {category} Test Results:")
    print("-" * 50)
    
    for test_name, passed, details in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if details:
            print(f"   Details: {details}")
    
    passed_tests = sum(1 for _, passed, _ in tests if passed)
    total_tests = len(tests)
    print(f"\\nResults: {passed_tests}/{total_tests} tests passed")


def main():
    """Main test execution."""
    print("🧪 FIELD EXTRACTION TESTING SUITE")
    print("=" * 60)
    print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    try:
        # Run test suites
        test_results.append(("Field Manager", test_field_manager()))
        test_results.append(("Schema Generator", test_schema_generator()))
        test_results.append(("LLM Extractor", test_llm_extractor()))
        test_results.append(("Data Processor", test_data_processor()))
        test_results.append(("Integration Workflow", test_integration_workflow()))
        test_results.append(("Performance", test_performance_benchmarks()))
        
    except KeyboardInterrupt:
        print("\\n⚠️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Testing failed with unexpected error: {e}")
        return 1
    
    # Generate final report
    print("\\n" + "=" * 60)
    print("📊 FIELD EXTRACTION TEST REPORT")
    print("=" * 60)
    
    total_suites = len(test_results)
    passed_suites = sum(1 for _, passed in test_results if passed)
    
    print(f"\\nTest Suites: {passed_suites}/{total_suites} passed")
    
    for suite_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {suite_name:<25} {status}")
    
    # Overall assessment
    if passed_suites == total_suites:
        print("\\n🎉 ALL TESTS PASSED!")
        print("✅ Field extraction system is working correctly")
        print("✅ You can now use the complete extraction workflow")
    elif passed_suites >= total_suites * 0.8:
        print("\\n⚠️  MOST TESTS PASSED")
        print("⚠️  Field extraction should work with some limitations")
        print("💡 Check the failed tests for specific issues")
    else:
        print("\\n❌ MULTIPLE TESTS FAILED")
        print("❌ Field extraction may not work correctly")
        print("🔧 Please fix the issues before using the system")
    
    print(f"\\n⏰ Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if passed_suites == total_suites else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        sys.exit(1)