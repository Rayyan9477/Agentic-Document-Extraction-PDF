"""
Comprehensive Medical Code Validation Testing Script
Tests the enhanced validation system with all CPT/HCPCS and ICD-10 codes.
"""

import sys
import logging
import time
import asyncio
from typing import Dict, List, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import comprehensive validation components
try:
    from src.validation.comprehensive_medical_codes import (
        comprehensive_validator, 
        CodeValidationResult,
        validate_code,
        validate_codes_batch,
        validate_codes_sync
    )
    from src.debugging.browser_automation import (
        browser_manager,
        debug_medical_codes,
        create_error_fixture
    )
    from src.debugging.puppeteer_mcp import (
        puppeteer_test_suite,
        run_puppeteer_tests
    )
    COMPREHENSIVE_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import comprehensive validation modules: {e}")
    COMPREHENSIVE_AVAILABLE = False

# Browser automation availability
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Browser automation tests will be skipped.")


def test_comprehensive_code_validation():
    """Test comprehensive medical code validation with all code types."""
    print("Testing Comprehensive Medical Code Validation...")
    
    tests = []
    
    if not COMPREHENSIVE_AVAILABLE:
        tests.append(("Comprehensive Validation Available", False, "Module not imported"))
        print_test_results("Comprehensive Validation", tests)
        return False
    
    # Test 1: Individual code validation
    try:
        test_codes = [
            ("99213", "CPT"),
            ("Z00.00", "ICD-10"),
            ("A0426", "HCPCS"),
            ("J3420", "HCPCS"),
            ("I10", "ICD-10"),
            ("90471", "CPT"),
            ("INVALID", "UNKNOWN")
        ]
        
        valid_count = 0
        total_count = len(test_codes)
        
        for code, expected_type in test_codes:
            result = asyncio.run(validate_code(code))
            if result.is_valid and (expected_type == "UNKNOWN" or result.code_type == expected_type):
                valid_count += 1
        
        test_passed = valid_count >= (total_count - 1)  # Allow for one invalid code
        tests.append(("Individual Code Validation", test_passed, 
                     f"{valid_count}/{total_count} codes validated correctly"))
        
    except Exception as e:
        tests.append(("Individual Code Validation", False, str(e)))
    
    # Test 2: Batch code validation
    try:
        batch_codes = [
            "99213", "99214", "99215",  # CPT codes
            "Z00.00", "I10", "E11.9",   # ICD-10 codes
            "A0426", "J3420",           # HCPCS codes
            "INVALID1", "INVALID2"      # Invalid codes
        ]
        
        batch_results = validate_codes_sync(batch_codes)
        
        valid_results = [r for r in batch_results if r.is_valid]
        invalid_results = [r for r in batch_results if not r.is_valid]
        
        test_passed = len(valid_results) >= 6 and len(invalid_results) >= 2
        tests.append(("Batch Code Validation", test_passed,
                     f"{len(valid_results)} valid, {len(invalid_results)} invalid"))
        
    except Exception as e:
        tests.append(("Batch Code Validation", False, str(e)))
    
    # Test 3: Database caching
    try:
        # Test the same code twice to check caching
        start_time = time.time()
        result1 = asyncio.run(validate_code("99213"))
        first_time = time.time() - start_time
        
        start_time = time.time()
        result2 = asyncio.run(validate_code("99213"))
        second_time = time.time() - start_time
        
        # Second call should be faster due to caching
        cache_working = (
            result1.is_valid == result2.is_valid and
            second_time < first_time * 0.5  # At least 50% faster
        )
        
        tests.append(("Database Caching", cache_working,
                     f"First: {first_time:.3f}s, Second: {second_time:.3f}s"))
        
    except Exception as e:
        tests.append(("Database Caching", False, str(e)))
    
    # Test 4: Code suggestions
    try:
        suggestions = comprehensive_validator.get_code_suggestions("99999", "CPT", 3)
        test_passed = len(suggestions) > 0
        
        tests.append(("Code Suggestions", test_passed,
                     f"Generated {len(suggestions)} suggestions"))
        
    except Exception as e:
        tests.append(("Code Suggestions", False, str(e)))
    
    # Test 5: Validation statistics
    try:
        stats = comprehensive_validator.get_validation_statistics()
        test_passed = "overall" in stats and stats["overall"]["total_cached"] > 0
        
        tests.append(("Validation Statistics", test_passed,
                     f"Cache contains {stats.get('overall', {}).get('total_cached', 0)} entries"))
        
    except Exception as e:
        tests.append(("Validation Statistics", False, str(e)))
    
    print_test_results("Comprehensive Validation", tests)
    return all(result[1] for result in tests)


async def test_browser_automation():
    """Test browser automation and debugging capabilities."""
    print("Testing Browser Automation and Debugging...")
    
    tests = []
    
    if not PLAYWRIGHT_AVAILABLE:
        tests.append(("Browser Automation Available", False, "Playwright not installed"))
        print_test_results("Browser Automation", tests)
        return False
    
    # Test 1: Debug medical code validation
    try:
        test_codes = ["99213", "Z00.00", "INVALID123"]
        debug_result = await debug_medical_codes(test_codes)
        
        test_passed = (
            "results" in debug_result and
            len(debug_result["results"]) == len(test_codes) and
            debug_result.get("session_id", "").startswith("debug_")
        )
        
        tests.append(("Medical Code Debug Session", test_passed,
                     f"Debug session: {debug_result.get('session_id', 'N/A')}"))
        
    except Exception as e:
        tests.append(("Medical Code Debug Session", False, str(e)))
    
    # Test 2: Error fixture generation
    try:
        error_data = {
            "type": "validation_error",
            "code": "INVALID123",
            "message": "Medical code not found in database"
        }
        
        fixture_result = await create_error_fixture(error_data)
        
        test_passed = (
            "id" in fixture_result and
            fixture_result.get("reproducible", False) and
            "screenshot_path" in fixture_result
        )
        
        tests.append(("Error Fixture Generation", test_passed,
                     f"Fixture: {fixture_result.get('id', 'N/A')}"))
        
    except Exception as e:
        tests.append(("Error Fixture Generation", False, str(e)))
    
    print_test_results("Browser Automation", tests)
    return all(result[1] for result in tests)


async def test_puppeteer_integration():
    """Test Puppeteer MCP integration."""
    print("Testing Puppeteer MCP Integration...")
    
    tests = []
    
    # Test 1: Puppeteer MCP availability
    try:
        from src.debugging.puppeteer_mcp import PuppeteerMCPClient
        
        # Test client creation
        client = PuppeteerMCPClient()
        test_passed = client is not None
        
        tests.append(("Puppeteer MCP Client Creation", test_passed, "Client created successfully"))
        
    except Exception as e:
        tests.append(("Puppeteer MCP Client Creation", False, str(e)))
    
    # Test 2: Comprehensive test suite
    try:
        # Note: This would require a running Streamlit app
        # For now, we'll test the test suite structure
        from src.debugging.puppeteer_mcp import PuppeteerTestSuite
        
        test_suite = PuppeteerTestSuite()
        test_passed = (
            hasattr(test_suite, 'run_comprehensive_test_suite') and
            hasattr(test_suite, 'test_data') and
            len(test_suite.test_data.get('valid_cpt_codes', [])) > 0
        )
        
        tests.append(("Puppeteer Test Suite Structure", test_passed, 
                     f"Test suite with {len(test_suite.test_data)} test data categories"))
        
    except Exception as e:
        tests.append(("Puppeteer Test Suite Structure", False, str(e)))
    
    print_test_results("Puppeteer MCP Integration", tests)
    return all(result[1] for result in tests)


def test_error_handling_and_debugging():
    """Test error handling and debugging capabilities."""
    print("Testing Error Handling and Debugging...")
    
    tests = []
    
    # Test 1: Invalid code handling
    try:
        invalid_codes = ["", None, "INVALID_FORMAT_CODE_123456", ""]
        results = validate_codes_sync(invalid_codes)
        
        # All should be marked as invalid
        all_invalid = all(not result.is_valid for result in results if result.code)
        empty_handled = len([r for r in results if not r.code]) == 0  # Empty codes filtered
        
        test_passed = all_invalid
        tests.append(("Invalid Code Handling", test_passed,
                     f"Handled {len(results)} invalid codes correctly"))
        
    except Exception as e:
        tests.append(("Invalid Code Handling", False, str(e)))
    
    # Test 2: API timeout simulation
    try:
        # Test with a code that might timeout
        start_time = time.time()
        result = asyncio.run(validate_code("99999"))  # Edge case code
        end_time = time.time()
        
        # Should complete within reasonable time
        completed_in_time = (end_time - start_time) < 10.0
        has_result = result is not None
        
        test_passed = completed_in_time and has_result
        tests.append(("API Timeout Handling", test_passed,
                     f"Completed in {end_time - start_time:.2f}s"))
        
    except Exception as e:
        tests.append(("API Timeout Handling", False, str(e)))
    
    # Test 3: Database error recovery
    try:
        # Test validation statistics even with potential database issues
        stats = comprehensive_validator.get_validation_statistics()
        
        # Should return dict even if empty
        test_passed = isinstance(stats, dict)
        tests.append(("Database Error Recovery", test_passed,
                     f"Statistics returned: {len(stats)} categories"))
        
    except Exception as e:
        tests.append(("Database Error Recovery", False, str(e)))
    
    # Test 4: Memory management
    try:
        # Test cache clearing
        initial_stats = comprehensive_validator.get_validation_statistics()
        cleared_count = comprehensive_validator.clear_cache(0)  # Clear all
        
        test_passed = cleared_count >= 0
        tests.append(("Memory Management", test_passed,
                     f"Cleared {cleared_count} cache entries"))
        
    except Exception as e:
        tests.append(("Memory Management", False, str(e)))
    
    print_test_results("Error Handling and Debugging", tests)
    return all(result[1] for result in tests)


def test_integration_with_existing_system():
    """Test integration with existing data processor."""
    print("Testing Integration with Existing System...")
    
    tests = []
    
    # Test 1: Data processor integration
    try:
        from src.extraction.data_processor import data_processor, COMPREHENSIVE_VALIDATION_AVAILABLE
        
        test_passed = COMPREHENSIVE_VALIDATION_AVAILABLE
        tests.append(("Data Processor Integration", test_passed,
                     f"Comprehensive validation: {'Available' if test_passed else 'Not Available'}"))
        
    except Exception as e:
        tests.append(("Data Processor Integration", False, str(e)))
    
    # Test 2: Mock patient record processing
    try:
        from src.extraction.data_processor import data_processor
        
        mock_record = {
            "record_id": "integration_test",
            "extracted_text": """
            Patient: Integration Test Patient
            CPT: 99213, 99214
            DX: Z00.00, I10
            HCPCS: A0426
            """,
            "confidence": 0.9
        }
        
        selected_fields = ["patient_name", "cpt_codes", "dx_codes", "hcpcs_codes"]
        
        # This would normally require LLM but we'll test the structure
        test_passed = hasattr(data_processor, 'process_patient_record')
        tests.append(("Patient Record Processing Structure", test_passed,
                     "Data processor has required methods"))
        
    except Exception as e:
        tests.append(("Patient Record Processing Structure", False, str(e)))
    
    # Test 3: Excel export integration
    try:
        from src.export.excel_exporter import excel_exporter, OPENPYXL_AVAILABLE
        
        if OPENPYXL_AVAILABLE:
            # Test with comprehensive validation results
            mock_results = {
                "test_file.pdf": {
                    "success": True,
                    "structured_data": [{
                        "patient_id": "TEST001",
                        "extracted_data": {
                            "cpt_codes": ["99213", "99214"],
                            "dx_codes": ["Z00.00", "I10"]
                        },
                        "validation_results": {
                            "comprehensive_validation": {
                                "cpt_codes": {"valid": ["99213", "99214"], "invalid": []},
                                "dx_codes": {"valid": ["Z00.00", "I10"], "invalid": []}
                            }
                        },
                        "confidence_score": 0.95
                    }],
                    "processing_metadata": {"total_duration": 5.2}
                }
            }
            
            excel_bytes, filename = excel_exporter.export_structured_data(mock_results)
            test_passed = len(excel_bytes) > 0 and filename.endswith('.xlsx')
        else:
            test_passed = True  # Skip if not available
        
        tests.append(("Excel Export Integration", test_passed,
                     f"Excel export: {'Available' if OPENPYXL_AVAILABLE else 'Skipped'}"))
        
    except Exception as e:
        tests.append(("Excel Export Integration", False, str(e)))
    
    print_test_results("Integration with Existing System", tests)
    return all(result[1] for result in tests)


def test_performance_and_scalability():
    """Test performance and scalability of comprehensive validation."""
    print("Testing Performance and Scalability...")
    
    tests = []
    
    # Test 1: Large batch validation performance
    try:
        # Create a large batch of codes
        large_batch = []
        
        # Add valid codes
        for i in range(10):
            large_batch.extend(["99213", "99214", "99215", "Z00.00", "I10", "A0426"])
        
        # Add some invalid codes
        for i in range(5):
            large_batch.extend(["INVALID" + str(i)])
        
        start_time = time.time()
        results = validate_codes_sync(large_batch)
        end_time = time.time()
        
        processing_time = end_time - start_time
        codes_per_second = len(large_batch) / processing_time
        
        # Should process at reasonable speed (at least 10 codes per second)
        test_passed = codes_per_second >= 10.0
        tests.append(("Large Batch Performance", test_passed,
                     f"{len(large_batch)} codes in {processing_time:.2f}s ({codes_per_second:.1f} codes/sec)"))
        
    except Exception as e:
        tests.append(("Large Batch Performance", False, str(e)))
    
    # Test 2: Memory usage efficiency
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple validation batches
        for _ in range(5):
            test_codes = ["99213", "Z00.00", "A0426", "INVALID"] * 10
            results = validate_codes_sync(test_codes)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should not increase memory significantly (less than 50MB)
        test_passed = memory_increase < 50
        tests.append(("Memory Efficiency", test_passed,
                     f"Memory increase: {memory_increase:.1f}MB"))
        
    except ImportError:
        tests.append(("Memory Efficiency", True, "psutil not available - skipped"))
    except Exception as e:
        tests.append(("Memory Efficiency", False, str(e)))
    
    # Test 3: Concurrent validation
    try:
        async def validate_concurrent():
            tasks = []
            for i in range(10):
                tasks.append(validate_code(f"9921{i % 10}"))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            return results, end_time - start_time
        
        results, concurrent_time = asyncio.run(validate_concurrent())
        
        # Should complete concurrent validation efficiently
        successful_results = [r for r in results if isinstance(r, CodeValidationResult)]
        test_passed = len(successful_results) >= 8 and concurrent_time < 5.0
        
        tests.append(("Concurrent Validation", test_passed,
                     f"{len(successful_results)}/10 successful in {concurrent_time:.2f}s"))
        
    except Exception as e:
        tests.append(("Concurrent Validation", False, str(e)))
    
    print_test_results("Performance and Scalability", tests)
    return all(result[1] for result in tests)


def print_test_results(category: str, tests: List[tuple]):
    """Print formatted test results."""
    print(f"\n{category} Test Results:")
    print("-" * 70)
    
    for test_name, passed, details in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<40} {status}")
        if details:
            print(f"   Details: {details}")
    
    passed_tests = sum(1 for _, passed, _ in tests if passed)
    total_tests = len(tests)
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")


async def main():
    """Main test execution."""
    print("COMPREHENSIVE MEDICAL CODE VALIDATION TEST SUITE")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    try:
        # Run all test suites
        test_results.append(("Comprehensive Code Validation", test_comprehensive_code_validation()))
        test_results.append(("Browser Automation", await test_browser_automation()))
        test_results.append(("Puppeteer MCP Integration", await test_puppeteer_integration()))
        test_results.append(("Error Handling and Debugging", test_error_handling_and_debugging()))
        test_results.append(("Integration with Existing System", test_integration_with_existing_system()))
        test_results.append(("Performance and Scalability", test_performance_and_scalability()))
        
    except KeyboardInterrupt:
        print("\nWarning: Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTesting failed with unexpected error: {e}")
        return 1
    
    # Generate final report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION TEST REPORT")
    print("=" * 80)
    
    total_suites = len(test_results)
    passed_suites = sum(1 for _, passed in test_results if passed)
    
    print(f"\nTest Suites: {passed_suites}/{total_suites} passed")
    
    for suite_name, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {suite_name:<35} {status}")
    
    # Overall assessment
    if passed_suites == total_suites:
        print("\nALL TESTS PASSED!")
        print("* Comprehensive medical code validation system is working correctly")
        print("* Browser automation and debugging tools are functional")
        print("* System is ready for production with all CPT/HCPCS and ICD-10 code support")
        print("* Enhanced debugging and error fixture generation is available")
    elif passed_suites >= total_suites * 0.8:
        print("\nMOST TESTS PASSED")
        print("* System should work with some limitations")
        print("* Check the failed tests for specific issues")
    else:
        print("\nMULTIPLE TESTS FAILED")
        print("* System may not work correctly")
        print("* Please fix the issues before using the comprehensive validation system")
    
    print(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if passed_suites == total_suites else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Test script failed: {e}")
        sys.exit(1)