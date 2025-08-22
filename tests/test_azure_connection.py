#!/usr/bin/env python3
"""
Azure OpenAI Connection Test Module
Tests Azure OpenAI service configuration and connectivity.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.azure_config import azure_config
from src.services.llm_service import llm_service


class TestAzureOpenAIConnection:
    """Test suite for Azure OpenAI connection and configuration."""
    
    def test_environment_variables_loaded(self):
        """Test that environment variables are properly loaded."""
        load_dotenv()
        
        # Check if environment variables exist
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        print(f"Environment check:")
        print(f"  ENDPOINT: {'SET' if endpoint and 'your-resource-name' not in endpoint else 'NOT SET'}")
        print(f"  API_KEY: {'SET' if api_key and 'your-api-key' not in api_key else 'NOT SET'}")
        print(f"  DEPLOYMENT: {'SET' if deployment and 'gpt-4o' != deployment else 'PLACEHOLDER'}")
    
    def test_azure_config_validation(self):
        """Test Azure configuration validation."""
        is_valid = azure_config.validate_configuration()
        config = azure_config.get_openai_config()
        
        print(f"Configuration validation: {'PASS' if is_valid else 'FAIL'}")
        print(f"Config keys present: {list(config.keys())}")
        
        if not is_valid:
            pytest.skip("Azure configuration not properly set. Please configure environment variables.")
    
    def test_llm_service_connection(self):
        """Test LLM service connection."""
        if llm_service.development_mode:
            print("⚠️  Running in development mode - Azure credentials not configured")
            pytest.skip("Development mode active. Configure Azure credentials to test connection.")
        
        result = llm_service.test_connection()
        
        print(f"Connection test result: {result}")
        
        if result['success']:
            assert 'response' in result
            assert 'model' in result
            print(f"✅ Connection successful with model: {result['model']}")
        else:
            pytest.fail(f"Connection failed: {result.get('error', 'Unknown error')}")


def test_azure_connection_standalone():
    """Standalone test function for manual execution."""
    print("🔍 Testing Azure OpenAI Configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_DEPLOYMENT_NAME': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    }
    
    print("Environment Variables:")
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value and var_value not in ['your_azure_openai_api_key_here', 'https://your-resource-name.openai.azure.com/', 'gpt-4o']:
            print(f"✅ {var_name}: SET")
        else:
            print(f"❌ {var_name}: NOT SET OR PLACEHOLDER")
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"\n❌ Missing or placeholder values for: {', '.join(missing_vars)}")
        print("\nPlease update your .env file with actual Azure OpenAI credentials.")
        print("\n📖 To get Azure OpenAI credentials:")
        print("1. Go to Azure Portal (https://portal.azure.com)")
        print("2. Navigate to your Azure OpenAI resource")
        print("3. Go to 'Keys and Endpoint' section")
        print("4. Copy the endpoint URL and one of the API keys")
        print("5. Go to 'Model deployments' to find your deployment name")
        return False
    
    print(f"\nAPI Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')}")
    
    # Test the actual connection
    try:
        print("\n🔗 Testing API Connection...")
        
        # Test connection
        result = llm_service.test_connection()
        
        if result['success']:
            print("✅ Azure OpenAI API Connection Successful!")
            print(f"📝 Response: {result['response']}")
            print(f"🤖 Model: {result['model']}")
            if 'usage' in result:
                print(f"💭 Token Usage: {result['usage']['total_tokens']} tokens")
        else:
            print("❌ Azure OpenAI API Connection Failed!")
            print(f"Error: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False
    
    print("\n🎉 All tests passed! Your Azure OpenAI is properly configured.")
    return True


if __name__ == "__main__":
    success = test_azure_connection_standalone()
    sys.exit(0 if success else 1)
