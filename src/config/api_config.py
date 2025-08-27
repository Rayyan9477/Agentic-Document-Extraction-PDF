"""
Unified API Configuration Module
Handles secure connection to both Azure OpenAI and regular OpenAI services.
Ensures HIPAA compliance with proper encryption and secret management.
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ApiProvider(Enum):
    """Enum for API providers."""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


class ApiConfig:
    """Manages API service configurations with security best practices for both Azure and OpenAI."""
    
    def __init__(self):
        self.api_provider = None
        self.azure_openai_api_key = None
        self.azure_openai_endpoint = None
        self.azure_openai_api_version = None
        self.azure_openai_deployment_name = None
        self.openai_api_key = None
        self.openai_model_name = None
        self.encryption_key = None
        self.key_vault_client = None
        
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration from environment variables or Key Vault."""
        try:
            # Initialize encryption
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if encryption_key:
                self.encryption_key = encryption_key.encode()
            else:
                # Generate a new key if none exists (development only)
                self.encryption_key = Fernet.generate_key()
                logger.warning("Generated new encryption key. In production, use Azure Key Vault.")
            
            # Try Key Vault first (production)
            key_vault_url = os.getenv('AZURE_KEY_VAULT_URL')
            if key_vault_url:
                self._initialize_key_vault(key_vault_url)
            else:
                # Load from environment variables (development)
                self._load_from_environment()
            
            # Determine which API provider to use
            self._determine_api_provider()
                
        except Exception as e:
            logger.error(f"Failed to initialize API configuration: {e}")
            raise
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Azure OpenAI configuration
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        self.azure_openai_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model_name = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')
    
    def _initialize_key_vault(self, key_vault_url: str):
        """Initialize Azure Key Vault client for secure secret management."""
        try:
            # Use environment variables for service principal authentication
            tenant_id = os.getenv('AZURE_TENANT_ID')
            client_id = os.getenv('AZURE_CLIENT_ID')
            client_secret = os.getenv('AZURE_CLIENT_SECRET')
            
            if tenant_id and client_id and client_secret:
                # Service Principal authentication (production)
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            else:
                # Default credential chain (development)
                credential = DefaultAzureCredential()
            
            self.key_vault_client = SecretClient(
                vault_url=key_vault_url,
                credential=credential
            )
            
            # Retrieve secrets from Key Vault
            self.azure_openai_api_key = self._get_secret('azure-openai-api-key')
            self.openai_api_key = self._get_secret('openai-api-key')
            
            # Still load other config from environment
            self._load_from_environment()
            
            logger.info("Successfully connected to Azure Key Vault")
            
        except Exception as e:
            logger.error(f"Failed to initialize Key Vault: {e}")
            # Fallback to environment variables
            self._load_from_environment()
    
    def _get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve a secret from Azure Key Vault."""
        try:
            if self.key_vault_client:
                secret = self.key_vault_client.get_secret(secret_name)
                return secret.value
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def _determine_api_provider(self):
        """Determine which API provider to use based on available configuration."""
        # Check Azure OpenAI first (preferred for production)
        if (self.azure_openai_api_key and 
            self.azure_openai_endpoint and 
            self.azure_openai_deployment_name):
            self.api_provider = ApiProvider.AZURE_OPENAI
            logger.info("Using Azure OpenAI API provider")
            
        # Fallback to OpenAI
        elif self.openai_api_key:
            self.api_provider = ApiProvider.OPENAI
            logger.info("Using OpenAI API provider")
            
        else:
            logger.warning("No valid API configuration found - running in development mode")
            self.api_provider = None
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data using Fernet encryption."""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.encrypt(data.encode())
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def validate_configuration(self) -> bool:
        """Validate that required configuration is present for the selected provider."""
        if self.api_provider == ApiProvider.AZURE_OPENAI:
            required_configs = [
                self.azure_openai_api_key,
                self.azure_openai_endpoint,
                self.azure_openai_api_version,
                self.azure_openai_deployment_name
            ]
            
            missing_configs = [config for config in required_configs if not config]
            
            if missing_configs:
                logger.error(f"Missing required Azure OpenAI configuration: {len(missing_configs)} items")
                return False
            
            logger.info("Azure OpenAI configuration validation successful")
            return True
            
        elif self.api_provider == ApiProvider.OPENAI:
            if not self.openai_api_key:
                logger.error("Missing required OpenAI API key")
                return False
            
            logger.info("OpenAI configuration validation successful")
            return True
            
        else:
            logger.error("No valid API provider configured")
            return False
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for the selected provider."""
        if self.api_provider == ApiProvider.AZURE_OPENAI:
            return {
                'provider': 'azure_openai',
                'api_key': self.azure_openai_api_key,
                'azure_endpoint': self.azure_openai_endpoint,
                'api_version': self.azure_openai_api_version,
                'deployment_name': self.azure_openai_deployment_name
            }
        elif self.api_provider == ApiProvider.OPENAI:
            return {
                'provider': 'openai',
                'api_key': self.openai_api_key,
                'model_name': self.openai_model_name
            }
        else:
            return {
                'provider': 'none',
                'error': 'No valid API configuration found'
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current API provider."""
        config = self.get_api_config()
        
        if self.api_provider == ApiProvider.AZURE_OPENAI:
            return {
                'provider': 'Azure OpenAI',
                'model': config['deployment_name'],
                'endpoint': config['azure_endpoint'],
                'api_version': config['api_version']
            }
        elif self.api_provider == ApiProvider.OPENAI:
            return {
                'provider': 'OpenAI',
                'model': config['model_name'],
                'endpoint': 'https://api.openai.com/v1'
            }
        else:
            return {
                'provider': 'None',
                'error': 'No valid configuration'
            }


# Global configuration instance
api_config = ApiConfig()