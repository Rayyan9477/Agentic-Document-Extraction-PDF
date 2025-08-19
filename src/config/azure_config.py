"""
Azure Configuration Module
Handles secure connection to Azure services including OpenAI GPT and Key Vault.
Ensures HIPAA compliance with proper encryption and secret management.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AzureConfig:
    """Manages Azure service configurations with security best practices."""
    
    def __init__(self):
        self.azure_openai_api_key = None
        self.azure_openai_endpoint = None
        self.azure_openai_api_version = None
        self.azure_openai_deployment_name = None
        self.encryption_key = None
        self.key_vault_client = None
        
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration from environment variables or Key Vault."""
        try:
            # Try to load from environment first (development)
            self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            self.azure_openai_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
            
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
                # Fallback to environment variables (development)
                self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure configuration: {e}")
            raise
    
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
            
            logger.info("Successfully connected to Azure Key Vault")
            
        except Exception as e:
            logger.error(f"Failed to initialize Key Vault: {e}")
            # Fallback to environment variables
            self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    
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
        """Validate that all required configuration is present."""
        required_configs = [
            self.azure_openai_api_key,
            self.azure_openai_endpoint,
            self.azure_openai_api_version,
            self.azure_openai_deployment_name
        ]
        
        missing_configs = [config for config in required_configs if not config]
        
        if missing_configs:
            logger.error(f"Missing required Azure configuration: {len(missing_configs)} items")
            return False
        
        logger.info("Azure configuration validation successful")
        return True
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration for API calls."""
        return {
            'api_key': self.azure_openai_api_key,
            'azure_endpoint': self.azure_openai_endpoint,
            'api_version': self.azure_openai_api_version,
            'deployment_name': self.azure_openai_deployment_name
        }


# Global configuration instance
azure_config = AzureConfig()