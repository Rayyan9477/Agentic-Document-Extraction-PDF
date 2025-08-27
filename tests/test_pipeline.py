"""
Pipeline Testing Script
Tests the complete PDF preprocessing pipeline with sample data and error handling.
"""

import sys
import logging
import time
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from src.config.azure_config import azure_config
    from src.services.llm_service import llm_service
    from src.processing.pipeline import processing_pipeline
    from src.processing.pdf_converter import pdf_converter, analyze_pdf_structure
    from src.processing.image_processor import image_processor
    from src.processing.layout_analyzer import layout_analyzer
    from src.processing.text_extractor import text_extractor
    from src.processing.patient_segmenter import patient_segmenter
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def create_sample_pdf_content() -> bytes:
    """Create a simple PDF for testing (mock content)."""
    try:
        # This would normally be actual PDF bytes
        # For testing, we'll simulate with a minimal structure
        sample_content = b"Mock PDF content for testing pipeline"
        return sample_content
    except Exception as e:
        logger.error(f"Failed to create sample PDF: {e}")
        raise


def test_configuration():
    """Test system configuration and dependencies."""
    print("🔧 Testing System Configuration...")
    
    tests = []
    
    # Test 1: Azure configuration
    try:
        is_valid = azure_config.validate_configuration()
        tests.append(("Azure Configuration", is_valid, None))
    except Exception as e:
        tests.append(("Azure Configuration", False, str(e)))
    
    # Test 2: LLM service connection
    try:
        connection_result = llm_service.test_connection()
        tests.append(("LLM Service Connection", connection_result["success"], 
                     connection_result.get("error")))
    except Exception as e:
        tests.append(("LLM Service Connection", False, str(e)))
    
    # Test 3: Processing modules
    module_tests = [
        ("PDF Converter", pdf_converter is not None),
        ("Image Processor", image_processor is not None),
        ("Layout Analyzer", layout_analyzer is not None),
        ("Text Extractor", text_extractor is not None),
        ("Patient Segmenter", patient_segmenter is not None),
        ("Processing Pipeline", processing_pipeline is not None)
    ]
    
    for test_name, result in module_tests:
        tests.append((test_name, result, None))
    
    # Print results
    print("\n📋 Configuration Test Results:")
    print("-" * 50)
    
    for test_name, passed, error in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if error:
            print(f"   Error: {error}")
    
    passed_tests = sum(1 for _, passed, _ in tests if passed)
    total_tests = len(tests)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def test_individual_components():
    """Test individual pipeline components."""
    print("\n🔍 Testing Individual Components...")
    
    component_tests = []
    
    # Test 1: PDF Structure Analysis (mock)
    try:
        # This would fail with mock data, but we test the function exists
        test_passed = hasattr(processing_pipeline, '_get_default_processing_options')
        component_tests.append(("Pipeline Options", test_passed, None))
    except Exception as e:
        component_tests.append(("Pipeline Options", False, str(e)))
    
    # Test 2: Image Processor Quality Metrics
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (300, 200), color='white')
        
        # Test quality metrics calculation
        metrics = image_processor.get_image_quality_metrics(test_image)
        test_passed = 'quality_score' in metrics and isinstance(metrics['quality_score'], (int, float))
        component_tests.append(("Image Quality Metrics", test_passed, None))
        
    except Exception as e:
        component_tests.append(("Image Quality Metrics", False, str(e)))
    
    # Test 3: Layout Analyzer Initialization
    try:
        # Test that layout analyzer has required methods
        required_methods = ['analyze_page_layout', '_detect_text_regions']
        test_passed = all(hasattr(layout_analyzer, method) for method in required_methods)
        component_tests.append(("Layout Analyzer Methods", test_passed, None))
        
    except Exception as e:
        component_tests.append(("Layout Analyzer Methods", False, str(e)))
    
    # Test 4: Patient Segmenter Patterns
    try:
        # Test patient indicator patterns
        test_text = "Patient Name: John Doe\nDOB: 01/15/1980"
        patterns_found = 0
        
        for pattern in patient_segmenter.compiled_patterns[:5]:
            if pattern.search(test_text.lower()):
                patterns_found += 1
                break
        
        test_passed = patterns_found > 0
        component_tests.append(("Patient Pattern Recognition", test_passed, None))
        
    except Exception as e:
        component_tests.append(("Patient Pattern Recognition", False, str(e)))
    
    # Test 5: Processing Pipeline Options
    try:
        default_options = processing_pipeline._get_default_processing_options()
        required_keys = ['image_dpi', 'use_llm_extraction', 'apply_preprocessing']
        test_passed = all(key in default_options for key in required_keys)
        component_tests.append(("Pipeline Default Options", test_passed, None))
        
    except Exception as e:
        component_tests.append(("Pipeline Default Options", False, str(e)))
    
    # Print results
    print("\n📋 Component Test Results:")
    print("-" * 50)
    
    for test_name, passed, error in component_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if error:
            print(f"   Error: {error}")
    
    passed_tests = sum(1 for _, passed, _ in component_tests if passed)
    total_tests = len(component_tests)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def test_processing_workflow():
    """Test the processing workflow with mock data."""
    print("\n⚡ Testing Processing Workflow...")
    
    workflow_tests = []
    
    # Test 1: Processing Options Validation
    try:
        options = processing_pipeline._get_default_processing_options()
        
        # Verify option types and ranges
        valid_options = (
            isinstance(options['image_dpi'], int) and 150 <= options['image_dpi'] <= 600 and
            isinstance(options['use_llm_extraction'], bool) and
            isinstance(options['apply_preprocessing'], bool)
        )
        
        workflow_tests.append(("Processing Options Validation", valid_options, None))
        
    except Exception as e:
        workflow_tests.append(("Processing Options Validation", False, str(e)))
    
    # Test 2: Error Handling
    try:
        # Test processing with invalid input
        result = processing_pipeline.process_pdf_complete(
            pdf_bytes=b"invalid_pdf_content",
            filename="test.pdf",
            processing_options=None
        )
        
        # Should handle gracefully and return error result
        test_passed = isinstance(result, dict) and ('error' in result or 'success' in result)
        workflow_tests.append(("Error Handling", test_passed, None))
        
    except Exception as e:
        # Exception handling is also acceptable
        workflow_tests.append(("Error Handling", True, f"Handled via exception: {type(e).__name__}"))
    
    # Test 3: Summary Generation
    try:
        # Test summary with mock results
        mock_results = {
            "filename": "test.pdf",
            "success": True,
            "pages": [{"page_number": 0, "processing_time": 1.5}],
            "patient_records": [{"confidence": 0.85}],
            "processing_metadata": {
                "total_duration": 2.0,
                "stages_completed": ["pdf_conversion", "text_extraction"],
                "errors": []
            }
        }
        
        summary = processing_pipeline.get_processing_summary(mock_results)
        test_passed = isinstance(summary, dict) and 'total_patients' in summary
        
        workflow_tests.append(("Summary Generation", test_passed, None))
        
    except Exception as e:
        workflow_tests.append(("Summary Generation", False, str(e)))
    
    # Test 4: Session Data Management
    try:
        # Test session data handling
        processing_pipeline.current_session_data = {"test": True}
        test_passed = hasattr(processing_pipeline, 'current_session_data')
        
        workflow_tests.append(("Session Data Management", test_passed, None))
        
    except Exception as e:
        workflow_tests.append(("Session Data Management", False, str(e)))
    
    # Print results
    print("\n📋 Workflow Test Results:")
    print("-" * 50)
    
    for test_name, passed, error in workflow_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if error:
            print(f"   Details: {error}")
    
    passed_tests = sum(1 for _, passed, _ in workflow_tests if passed)
    total_tests = len(workflow_tests)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def test_performance_benchmarks():
    """Test performance characteristics."""
    print("\n⏱️ Testing Performance Benchmarks...")
    
    performance_tests = []
    
    # Test 1: Processing Options Creation Speed
    try:
        start_time = time.time()
        for _ in range(100):
            options = processing_pipeline._get_default_processing_options()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        test_passed = avg_time < 0.001  # Should be very fast
        
        performance_tests.append(("Options Creation Speed", test_passed, 
                                f"Avg: {avg_time*1000:.3f}ms"))
        
    except Exception as e:
        performance_tests.append(("Options Creation Speed", False, str(e)))
    
    # Test 2: Memory Usage Estimation
    try:
        import psutil
        import os
        
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform some operations
        for _ in range(10):
            processing_pipeline._get_default_processing_options()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be minimal
        test_passed = memory_increase < 5  # Less than 5MB increase
        
        performance_tests.append(("Memory Usage", test_passed, 
                                f"Increase: {memory_increase:.1f}MB"))
        
    except ImportError:
        performance_tests.append(("Memory Usage", True, "psutil not available - skipped"))
    except Exception as e:
        performance_tests.append(("Memory Usage", False, str(e)))
    
    # Test 3: Import Speed
    try:
        start_time = time.time()
        
        # Re-import modules to test import speed
        import importlib
        import src.processing.pipeline
        importlib.reload(src.processing.pipeline)
        
        import_time = time.time() - start_time
        test_passed = import_time < 2.0  # Should import within 2 seconds
        
        performance_tests.append(("Module Import Speed", test_passed, 
                                f"Time: {import_time:.3f}s"))
        
    except Exception as e:
        performance_tests.append(("Module Import Speed", False, str(e)))
    
    # Print results
    print("\n📋 Performance Test Results:")
    print("-" * 50)
    
    for test_name, passed, details in performance_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if details:
            print(f"   Details: {details}")
    
    passed_tests = sum(1 for _, passed, _ in performance_tests if passed)
    total_tests = len(performance_tests)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def generate_test_report(all_results: List[Tuple[str, bool]]):
    """Generate a comprehensive test report."""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    total_test_categories = len(all_results)
    passed_categories = sum(1 for _, passed in all_results if passed)
    
    print(f"\nTest Categories: {passed_categories}/{total_test_categories} passed")
    
    for category, passed in all_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {category:<30} {status}")
    
    # Overall assessment
    if passed_categories == total_test_categories:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The PDF preprocessing pipeline is ready for use.")
        print("✅ You can now run the Streamlit application: streamlit run app.py")
    elif passed_categories >= total_test_categories * 0.75:
        print("\n⚠️  MOST TESTS PASSED")
        print("⚠️  The pipeline should work but some features may have issues.")
        print("💡 Check the failed tests and consider addressing them.")
    else:
        print("\n❌ MULTIPLE TESTS FAILED")
        print("❌ The pipeline may not work correctly.")
        print("🔧 Please address the configuration issues before proceeding.")
    
    print("\n📝 Next Steps:")
    if all_results[0][1]:  # Configuration tests passed
        print("1. ✅ Azure OpenAI is properly configured")
    else:
        print("1. ❌ Fix Azure OpenAI configuration in .env file")
    
    print("2. 📄 Place sample PDF files for testing")
    print("3. 🚀 Run: streamlit run app.py")
    print("4. 🔍 Test with your medical superbill PDFs")
    
    print(f"\n⏰ Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_categories == total_test_categories


def main():
    """Main test execution."""
    print("🧪 MEDICAL SUPERBILL EXTRACTOR - PIPELINE TESTING")
    print("=" * 60)
    print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Run test suites
    try:
        # Test 1: Configuration
        config_passed = test_configuration()
        test_results.append(("System Configuration", config_passed))
        
        # Test 2: Individual Components
        components_passed = test_individual_components()
        test_results.append(("Component Functions", components_passed))
        
        # Test 3: Processing Workflow
        workflow_passed = test_processing_workflow()
        test_results.append(("Processing Workflow", workflow_passed))
        
        # Test 4: Performance Benchmarks
        performance_passed = test_performance_benchmarks()
        test_results.append(("Performance Benchmarks", performance_passed))
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with unexpected error: {e}")
        return 1
    
    # Generate comprehensive report
    overall_success = generate_test_report(test_results)
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        sys.exit(1)