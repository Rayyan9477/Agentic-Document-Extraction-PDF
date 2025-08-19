# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Medical Superbill Structured Extractor** - a HIPAA-compliant system that extracts structured data from unstructured medical superbill PDFs. The system handles patient data, CPT codes, diagnosis codes, insurance details, and provider information.

## Architecture

The system follows a pipeline architecture:

1. **PDF Preprocessing** - Split PDFs into images, normalize resolution/orientation
2. **Patient Segmentation** - Detect multiple patient records per page using layout analysis
3. **OCR & Text Extraction** - Use Docling + OCR engines to extract structured text
4. **LLM Processing** - Send chunks to Azure GPT-5 API for structured JSON extraction
5. **Validation** - Cross-check CPT/DX codes against medical databases
6. **Export** - Output to Excel format using openpyxl

## Technology Stack

- **Frontend**: Streamlit for user interface
- **Backend**: Python 3.12
- **AI/ML**: Azure GPT-5 API for structured extraction
- **OCR**: Docling, Paddle OCR/EasyOCR for text extraction
- **PDF Processing**: PyMuPDF/pdf2image for page splitting
- **Export**: openpyxl/pandas for Excel output
- **Security**: AES-256 encryption, Azure Key Vault for API keys

## Key Components

### Data Flow
- Users upload PDFs via Streamlit interface
- System segments pages into patient-specific chunks
- Each chunk is processed by Azure GPT-5 to extract structured JSON
- Results are validated against medical code databases
- Final output exported as formatted Excel sheets

### Field Extraction Schema
The system extracts predefined medical fields:
- Patient Name, DOB
- CPT Codes (procedure codes)
- DX Codes (diagnosis codes)
- Insurance information
- Provider details
- Custom user-defined fields

### Multi-Patient Support
- Handles multiple patient records on single PDF pages
- Uses heuristics and layout analysis for patient boundary detection
- Outputs one patient per Excel row

## Security & Compliance

- **HIPAA Compliant**: All PHI handling follows HIPAA requirements
- **Encryption**: AES-256 for temporary data storage
- **PHI Masking**: Sensitive data masked in logs
- **Azure Confidential Computing**: Secure cloud processing
- **Data Retention**: Temporary files are encrypted and wiped after processing

## Development Phases

The project is structured in 10 phases from infrastructure setup to final compliance/security implementation. Currently in early development stage with project skeleton and Azure API configuration completed.

## Batch Processing
System supports processing multiple PDFs simultaneously with combined export functionality.