# Medical Superbill Structured Extractor

A HIPAA-compliant system for extracting structured data from unstructured medical superbill PDFs using Azure OpenAI GPT-5.

## Phase 1: Project Setup ✅

This phase establishes the foundation with Azure integration and basic Streamlit interface.

## Phase 2: PDF Preprocessing, OCR and Segmentation ✅

Complete PDF processing pipeline with advanced image processing and patient segmentation.

## Phase 3: Field Tagging Interface & LLM Extraction ✅

Advanced field selection system with structured data extraction using Azure GPT-5.

## Phase 4: Validation, Export, and Final UI ✅

Complete validation, normalization, export, and security features with professional UI enhancements.

### Features Implemented

**Phase 1:**
- ✅ **Project Structure**: Complete Python project with organized modules
- ✅ **Azure OpenAI Integration**: Secure GPT-5 API connection with HIPAA compliance
- ✅ **Security**: AES-256 encryption, Azure Key Vault support, PHI protection
- ✅ **Streamlit UI**: Complete interface with file upload, field selection, and preview
- ✅ **API Testing**: Comprehensive test suite for Azure connectivity

**Phase 2:**
- ✅ **PDF to Image Conversion**: PyMuPDF-based secure conversion with encryption
- ✅ **Image Preprocessing**: OpenCV-based denoising, skew correction, contrast enhancement
- ✅ **Layout Analysis**: Docling integration + custom algorithms for structure detection
- ✅ **Text Extraction**: LLM + OCR dual validation with PaddleOCR/EasyOCR
- ✅ **Patient Segmentation**: Multi-method patient boundary detection
- ✅ **Processing Pipeline**: Complete end-to-end workflow orchestration
- ✅ **Real Processing**: Functional PDF processing with actual data extraction

**Phase 3:**
- ✅ **Advanced Field Manager**: 25+ predefined medical fields with custom field creation
- ✅ **Dynamic Schema Generation**: JSON schemas based on selected fields
- ✅ **HIPAA-Compliant LLM Extraction**: Azure GPT-5 with medical expertise prompts
- ✅ **Structured Data Processing**: JSON parsing, validation, and post-processing
- ✅ **Medical Code Validation**: CPT and ICD-10 code format validation
- ✅ **Enhanced UI**: Professional field selection with examples and validation
- ✅ **Results Visualization**: Structured data tables with confidence scoring

**Phase 4:**
- ✅ **Advanced Medical Code Validation**: Comprehensive CPT and ICD-10 database validation with 50+ common codes
- ✅ **Professional Data Normalization**: Smart normalization for dates, phone numbers, IDs, names, and addresses
- ✅ **Excel Export System**: Multi-sheet Excel reports with professional formatting and validation summaries
- ✅ **Batch Processing**: Process and export multiple PDFs with consolidated reporting
- ✅ **Security & Cleanup**: Secure temporary file management with PHI protection and automatic cleanup
- ✅ **Enhanced UI**: Batch export options, validation summaries, and download capabilities
- ✅ **Comprehensive Validation**: Cross-field validation and data quality scoring

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

4. **Test Complete Pipeline**
   ```bash
   python test_pipeline.py
   ```

5. **Test Field Extraction**
   ```bash
   python test_field_extraction.py
   ```

6. **Test Validation & Export (Phase 4)**
   ```bash
   python test_validation_export.py
   ```

7. **Run Application**
   ```bash
   streamlit run app.py
   ```

### Project Structure

```
├── src/
│   ├── config/
│   │   └── azure_config.py         # Azure services configuration
│   ├── services/
│   │   └── llm_service.py          # GPT-5 integration service
│   ├── processing/                 # PDF processing pipeline
│   │   ├── pdf_converter.py        # PDF to image conversion
│   │   ├── image_processor.py      # Image preprocessing & enhancement
│   │   ├── layout_analyzer.py      # Document structure analysis
│   │   ├── text_extractor.py       # LLM + OCR text extraction
│   │   ├── patient_segmenter.py    # Multi-patient detection
│   │   └── pipeline.py             # Complete workflow orchestration
│   ├── extraction/                 # Field extraction system
│   │   ├── field_manager.py        # Medical field definitions & management
│   │   ├── schema_generator.py     # Dynamic JSON schema generation
│   │   ├── llm_extractor.py        # HIPAA-compliant LLM extraction
│   │   └── data_processor.py       # Structured data processing & validation
│   ├── validation/                 # Validation & normalization (Phase 4)
│   │   ├── medical_codes.py        # CPT/ICD-10 code validation with databases
│   │   └── data_normalizer.py      # Advanced data normalization
│   ├── export/                     # Export functionality (Phase 4)
│   │   └── excel_exporter.py       # Professional Excel export with multiple sheets
│   └── security/                   # Security & cleanup (Phase 4)
│       └── data_cleanup.py         # Secure file cleanup & PHI protection
├── app.py                          # Main Streamlit application
├── test_api.py                     # API connectivity tests
├── test_pipeline.py                # Complete pipeline testing
├── test_field_extraction.py        # Field extraction testing
├── test_validation_export.py       # Validation & export testing (Phase 4)
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # Project documentation for Claude Code
└── .env.example                    # Environment variables template
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

### Current Capabilities

**Complete Medical Data Extraction System:**
1. **PDF Analysis**: Structure detection and metadata extraction
2. **Image Processing**: High-quality rendering with OpenCV enhancement
3. **Layout Analysis**: Advanced document structure detection
4. **Text Extraction**: Dual LLM + OCR validation
5. **Patient Segmentation**: Multi-patient boundary detection
6. **Field Selection**: 25+ predefined medical fields + custom field creation
7. **Structured Extraction**: HIPAA-compliant GPT-5 extraction with validation
8. **Results Processing**: JSON parsing, medical code validation, confidence scoring

**Medical Fields Supported:**
- **Patient Demographics**: Name, DOB, ID, Address, Phone
- **Medical Codes**: CPT codes, ICD-10 diagnosis codes with descriptions
- **Provider Information**: Provider details, NPI, practice information
- **Insurance & Billing**: Coverage details, copay, charges
- **Visit Details**: Service dates, locations, referring physicians
- **Clinical Information**: Chief complaint, symptoms, medications
- **Custom Fields**: User-defined fields with validation patterns

**Advanced Features:**
- Real-time structured data extraction
- Comprehensive medical code validation (CPT/ICD-10)
- Advanced data normalization and formatting
- Field confidence scoring and cross-field validation
- JSON schema generation and validation
- Professional results visualization
- Multi-format export capabilities (JSON, Excel)
- Batch processing with consolidated reporting
- Secure PHI handling and automatic cleanup

### Phase 4 Complete Features

- ✅ **Professional Excel Export**: Multi-sheet workbooks with Summary, Patient Data, Validation Report, and Raw Data sheets
- ✅ **Advanced Validation**: 50+ common CPT codes and ICD-10 codes with format and range validation
- ✅ **Smart Data Normalization**: Automatic formatting for dates (MM/DD/YYYY), phone numbers, SSNs, NPIs, and other medical IDs
- ✅ **Batch Processing**: Process multiple PDFs simultaneously with consolidated reporting and export
- ✅ **Security & Compliance**: Secure temporary file handling, PHI masking in logs, automatic cleanup procedures
- ✅ **Enhanced UI**: Batch export options, validation summaries, confidence scoring, and professional download capabilities

### Testing

Run the API test suite:
```bash
python tests\test_api.py
```

Run the complete pipeline tests:
```bash
python tests\test_pipeline.py
```

Run field extraction tests:
```bash
python tests\test_field_extraction.py
```

Run validation and export tests:
```bash
python tests\test_validation_export.py
```

Run Azure connection tests:
```bash
python tests\test_azure_connection.py
```

Expected output for Phase 4 tests:
```
🧪 VALIDATION & EXPORT TESTING SUITE
✅ Medical Code Validation PASSED
✅ Data Normalization PASSED  
✅ Excel Export PASSED
✅ Security & Cleanup PASSED
✅ Integrated Workflow PASSED
🎉 ALL TESTS PASSED! Validation and export system is working correctly
✅ Phase 4 implementation is complete and ready for use
```

## System Capabilities Summary

The Medical Superbill Structured Extractor now provides a complete, production-ready solution for extracting structured data from medical superbills with the following capabilities:

### 🔧 Core Processing
- **PDF Processing**: Secure conversion to images with encryption
- **Image Enhancement**: OpenCV-based preprocessing, denoising, skew correction
- **Layout Analysis**: Advanced document structure detection with Docling integration
- **Text Extraction**: Dual LLM + OCR validation for maximum accuracy
- **Patient Segmentation**: Multi-method boundary detection for multi-patient documents

### 🏷️ Field Management
- **25+ Predefined Fields**: Comprehensive medical field definitions
- **Custom Fields**: User-defined fields with validation patterns
- **Dynamic Schemas**: JSON schema generation based on selected fields
- **Field Categories**: Organized by Patient, Provider, Medical Codes, Insurance, etc.

### 🤖 AI-Powered Extraction
- **Azure GPT-5 Integration**: Secure, HIPAA-compliant LLM processing
- **Medical Expertise**: Specialized prompts for healthcare data extraction
- **Confidence Scoring**: Field-level and overall confidence assessment
- **Retry Logic**: Automatic retry with fallback strategies

### ✅ Validation & Quality
- **Medical Code Validation**: CPT and ICD-10 code verification with databases
- **Data Normalization**: Smart formatting for dates, phones, IDs, names, addresses
- **Cross-Field Validation**: Logical consistency checks between related fields
- **Quality Scoring**: Data quality assessment with error reporting

### 📊 Export & Reporting
- **Professional Excel Export**: Multi-sheet workbooks with formatting
- **Batch Processing**: Multiple PDF processing with consolidated reports
- **JSON Export**: Structured data export with metadata
- **Validation Reports**: Detailed quality assessment and error analysis

### 🔒 Security & Compliance
- **HIPAA Compliance**: End-to-end encryption and secure processing
- **PHI Protection**: Automatic masking of sensitive information in logs
- **Secure Cleanup**: Automatic deletion of temporary files with overwriting
- **Access Controls**: Restrictive file permissions and secure temporary storage

### 💻 User Interface
- **Professional UI**: Clean, medical-focused interface design
- **Real-time Processing**: Live progress tracking and status updates
- **Interactive Results**: Expandable result sections with detailed views
- **Batch Operations**: Multi-file upload and processing capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure OpenAI account with GPT-4/GPT-4o deployment
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd PDF
   pip install -r requirements.txt
   ```

2. **Configure Azure OpenAI:**
   
   **Option A: Using .env file (Recommended)**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```
   
   **Option B: Using PowerShell script**
   ```powershell
   .\scripts\setup_azure_env.ps1
   ```

3. **Test configuration:**
   ```bash
   python tests\test_azure_connection.py
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

### Directory Structure
```
├── app.py                          # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variable template
├── scripts/                      # Setup and utility scripts
│   ├── setup_azure_env.ps1      # PowerShell Azure setup
│   └── README.md                 # Scripts documentation
├── tests/                        # Test modules
│   ├── test_azure_connection.py  # Azure connectivity tests
│   ├── test_*.py                 # Additional test files
│   └── __init__.py
├── src/                          # Core application modules
│   ├── config/                   # Configuration management
│   ├── services/                 # External service integrations
│   ├── processing/               # PDF and image processing
│   ├── extraction/               # Data extraction and schema
│   ├── validation/               # Data validation and medical codes
│   ├── export/                   # Export functionality
│   └── security/                 # Security and compliance
```

This system represents a complete, enterprise-ready solution for medical data extraction that can be deployed in healthcare environments with confidence in its security, accuracy, and compliance standards.