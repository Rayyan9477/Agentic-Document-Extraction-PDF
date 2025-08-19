# Medical Superbill Structured Extractor

A HIPAA-compliant system for extracting structured data from unstructured medical superbill PDFs using Azure OpenAI GPT-5.

## Phase 1: Project Setup ✅

This phase establishes the foundation with Azure integration and basic Streamlit interface.

### Features Implemented

- ✅ **Project Structure**: Complete Python project with organized modules
- ✅ **Azure OpenAI Integration**: Secure GPT-5 API connection with HIPAA compliance
- ✅ **Security**: AES-256 encryption, Azure Key Vault support, PHI protection
- ✅ **Streamlit UI**: Complete interface with file upload, field selection, and preview
- ✅ **API Testing**: Comprehensive test suite for Azure connectivity

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

3. **Test API Connection**
   ```bash
   python test_api.py
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

### Project Structure

```
├── src/
│   ├── config/
│   │   └── azure_config.py      # Azure services configuration
│   └── services/
│       └── llm_service.py       # GPT-5 integration service
├── app.py                       # Main Streamlit application
├── test_api.py                 # API connectivity tests
├── requirements.txt            # Python dependencies
└── .env.example               # Environment variables template
```

### Configuration

Required environment variables:

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Security
ENCRYPTION_KEY=your_32_byte_key

# Optional: Azure Key Vault
AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net/
```

### Security Features

- 🔒 **HIPAA Compliance**: Encrypted data transmission and storage
- 🔑 **Azure Key Vault**: Secure API key management
- 🛡️ **PHI Protection**: Masked sensitive data in logs
- 🔐 **AES-256 Encryption**: Local data encryption

### UI Components

1. **File Upload**: Multi-file PDF upload with validation
2. **Field Selection**: 
   - Predefined medical fields (Patient Info, CPT/DX Codes, Provider, Insurance)
   - Custom field creation
   - Bulk select/deselect options
3. **Preview Section**: Mock data preview (extraction logic in Phase 2)
4. **API Status**: Real-time Azure OpenAI connectivity testing

### Next Steps: Phase 2

- PDF preprocessing and OCR implementation
- Patient segmentation logic
- Actual data extraction pipeline
- Excel export functionality

### Testing

Run the API test suite:
```bash
python test_api.py
```

Expected output:
```
🚀 Starting Azure OpenAI API Tests...
✅ Azure configuration is valid
✅ API connection successful
✅ Data extraction test successful
🎉 All tests passed! Azure OpenAI is ready for use.
```