"""
Azure OpenAI API Connection Test Script
Verifies that the Azure GPT-5 API is properly configured and accessible.
"""

import sys
import logging
from src.config.azure_config import azure_config
from src.services.llm_service import llm_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_azure_configuration():
    """Test Azure configuration validation."""
    print("🔧 Testing Azure Configuration...")
    
    try:
        is_valid = azure_config.validate_configuration()
        if is_valid:
            print("✅ Azure configuration is valid")
            
            # Display configuration (without sensitive data)
            config = azure_config.get_openai_config()
            print(f"   • Endpoint: {config['azure_endpoint']}")
            print(f"   • API Version: {config['api_version']}")
            print(f"   • Deployment: {config['deployment_name']}")
            print(f"   • API Key: {'***' + config['api_key'][-4:] if config['api_key'] else 'Not configured'}")
            
            return True
        else:
            print("❌ Azure configuration is invalid")
            print("   Please check your .env file or environment variables")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_api_connection():
    """Test Azure OpenAI API connection."""
    print("\n🌐 Testing Azure OpenAI API Connection...")
    
    try:
        result = llm_service.test_connection()
        
        if result["success"]:
            print("✅ API connection successful")
            print(f"   • Model: {result.get('model', 'Unknown')}")
            print(f"   • Response: {result.get('response', 'No response')}")
            print(f"   • Token Usage: {result.get('usage', {})}")
            return True
        else:
            print("❌ API connection failed")
            print(f"   • Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False


def test_data_extraction():
    """Test basic data extraction functionality."""
    print("\n📊 Testing Data Extraction...")
    
    # Sample medical text for testing
    sample_text = """
    Patient: John Smith
    DOB: 01/15/1980
    Date of Service: 03/20/2024
    CPT: 99213 - Office visit, established patient
    Diagnosis: Z00.00 - Encounter for general adult medical examination
    Provider: Dr. Jane Wilson, MD
    """
    
    # Simple extraction schema
    schema = {
        "patient_name": {"type": "string", "description": "Patient full name"},
        "date_of_birth": {"type": "string", "description": "Patient date of birth"},
        "cpt_codes": {"type": "array", "description": "CPT procedure codes"},
        "dx_codes": {"type": "array", "description": "Diagnosis codes"}
    }
    
    try:
        result = llm_service.extract_structured_data(sample_text, schema)
        
        if result["success"]:
            print("✅ Data extraction test successful")
            print("   • Extracted data:")
            for key, value in result["data"].items():
                print(f"     - {key}: {value}")
            print(f"   • Token Usage: {result.get('usage', {})}")
            return True
        else:
            print("❌ Data extraction test failed")
            print(f"   • Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Data extraction test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Azure OpenAI API Tests...\n")
    
    # Test sequence
    tests = [
        ("Configuration Validation", test_azure_configuration),
        ("API Connection", test_api_connection),
        ("Data Extraction", test_data_extraction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
        
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if all(results.values()):
        print("🎉 All tests passed! Azure OpenAI is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check configuration and connectivity.")
        print("\n🔧 Troubleshooting Tips:")
        print("1. Verify your .env file contains correct Azure credentials")
        print("2. Check Azure OpenAI service is deployed and accessible")
        print("3. Ensure your IP is not blocked by Azure firewalls")
        print("4. Verify the deployment name matches your Azure resource")
        return 1


if __name__ == "__main__":
    sys.exit(main())