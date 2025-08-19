# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Medical Superbill Structured Extractor** - a HIPAA-compliant system that extracts structured data from unstructured medical superbill PDFs. The system handles patient data, CPT codes, diagnosis codes, insurance details, and provider information.

**Current Status: Phase 4 Complete - Production Ready**
- ✅ Full PDF processing pipeline implemented
- ✅ Real-time document processing with Streamlit UI
- ✅ Multi-patient segmentation capability
- ✅ LLM + OCR dual validation
- ✅ Advanced field selection and structured extraction
- ✅ HIPAA-compliant medical data extraction with validation
- ✅ Advanced medical code validation (CPT/ICD-10) with databases
- ✅ Professional data normalization and formatting
- ✅ Multi-sheet Excel export with professional formatting
- ✅ Batch processing with consolidated reporting
- ✅ Comprehensive security and cleanup procedures

## Architecture

The system follows a comprehensive processing pipeline:

1. **PDF Analysis** - Analyze document structure and metadata
2. **Image Conversion** - Convert PDF pages to high-quality images (PyMuPDF)
3. **Image Preprocessing** - Apply OpenCV enhancements (denoise, skew correction, contrast)
4. **Layout Analysis** - Detect text blocks, tables, forms using Docling + custom algorithms
5. **Text Extraction** - LLM primary extraction + OCR validation (PaddleOCR/EasyOCR)
6. **Patient Segmentation** - Multi-method boundary detection with LLM assistance
7. **Results Processing** - Confidence scoring and quality metrics
8. **Export** - Structured output with full traceability

## Technology Stack

- **Frontend**: Streamlit for user interface
- **Backend**: Python 3.12
- **AI/ML**: Azure GPT-5 API for structured extraction
- **OCR**: Docling, Paddle OCR/EasyOCR for text extraction
- **PDF Processing**: PyMuPDF/pdf2image for page splitting
- **Export**: openpyxl/pandas for Excel output
- **Security**: AES-256 encryption, Azure Key Vault for API keys

## Key Components

### Processing Modules
**Core Processing:**
- **pdf_converter.py** - Secure PDF to image conversion with encryption
- **image_processor.py** - Advanced image preprocessing and quality optimization
- **layout_analyzer.py** - Document structure analysis and element detection
- **text_extractor.py** - Dual LLM+OCR text extraction with validation
- **patient_segmenter.py** - Multi-patient boundary detection
- **pipeline.py** - Complete workflow orchestration

**Field Extraction System:**
- **field_manager.py** - Medical field definitions and custom field management
- **schema_generator.py** - Dynamic JSON schema generation and validation
- **llm_extractor.py** - HIPAA-compliant structured data extraction
- **data_processor.py** - Post-processing, validation, and quality analysis

### Data Flow
- Users upload PDFs via enhanced Streamlit interface
- Real-time processing with progress tracking
- Advanced patient segmentation using multiple detection methods
- LLM + OCR cross-validation for maximum accuracy
- Comprehensive results with confidence scoring
- Full processing audit trail

### Multi-Patient Support
- Keyword-based boundary detection
- Layout structure analysis
- LLM-assisted validation
- Confidence scoring per patient record
- Handles complex layouts and handwritten content

## Security & Compliance

- **HIPAA Compliant**: All PHI handling follows HIPAA requirements
- **Encryption**: AES-256 for temporary data storage with automatic cleanup
- **PHI Masking**: Sensitive data masked in logs and audit trails
- **Azure Confidential Computing**: Secure cloud processing with GPT-5
- **Data Retention**: Encrypted temporary files with automatic cleanup
- **Processing Audit**: Complete traceability of all processing steps
- **Error Handling**: Graceful failure with data protection

## Development Status

**Phase 1: Project Setup** ✅
- Azure OpenAI integration with HIPAA compliance
- Streamlit interface foundation
- Security and encryption framework

**Phase 2: PDF Processing Pipeline** ✅
- Complete PDF to structured data pipeline
- Advanced image preprocessing
- LLM + OCR text extraction
- Multi-patient segmentation
- Real-time processing interface

**Phase 3: Field Extraction System** ✅
- 25+ predefined medical fields with custom field support
- Dynamic JSON schema generation
- HIPAA-compliant Azure GPT-5 extraction
- Medical code validation (CPT/ICD-10)
- Structured data processing with confidence scoring
- Professional field selection interface

**Phase 4: Export & Analytics** (Next)
- Professional Excel export with formatting
- Advanced medical code database integration
- Batch processing optimization
- Comprehensive reporting and analytics

## Current Capabilities

### Processing Pipeline Commands
```bash
# Test complete pipeline
python test_pipeline.py

# Test field extraction system  
python test_field_extraction.py

# Run application
streamlit run app.py
```

### Key Processing Features
- Real PDF processing with structured data extraction
- Multi-patient detection and segmentation
- 25+ medical fields with custom field creation
- HIPAA-compliant LLM extraction with medical expertise
- Medical code validation and confidence scoring
- Advanced field selection UI with examples
- JSON schema generation and validation
- Comprehensive error handling and recovery

### Field Extraction System
- **Predefined Fields**: Patient demographics, medical codes, provider info, insurance details
- **Custom Fields**: User-defined fields with validation patterns and examples
- **Schema Generation**: Dynamic JSON schemas based on selected fields
- **LLM Extraction**: HIPAA-compliant Azure GPT-5 with medical expertise
- **Validation**: Medical code format validation and cross-field checks
- **Results**: Structured data with confidence scores and detailed analysis

### Batch Processing
System supports processing multiple PDFs with structured extraction and individual result management.