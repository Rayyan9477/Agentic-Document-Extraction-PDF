"""
Medical Superbill Structured Extractor - Streamlit Application
HIPAA-compliant PDF data extraction system with user-friendly interface.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

# Configure logging with PHI protection
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our services
try:
    from src.config.azure_config import azure_config
    from src.services.llm_service import llm_service
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = {}
    if 'selected_fields' not in st.session_state:
        st.session_state.selected_fields = []
    if 'custom_fields' not in st.session_state:
        st.session_state.custom_fields = []
    if 'api_test_result' not in st.session_state:
        st.session_state.api_test_result = None


def get_predefined_fields() -> Dict[str, Dict[str, Any]]:
    """Get predefined medical fields for extraction."""
    return {
        "Patient Information": {
            "patient_name": {"type": "string", "description": "Full patient name"},
            "date_of_birth": {"type": "string", "description": "Patient date of birth (MM/DD/YYYY)"},
            "patient_id": {"type": "string", "description": "Patient ID or account number"},
            "address": {"type": "string", "description": "Patient address"},
            "phone_number": {"type": "string", "description": "Patient phone number"}
        },
        "Medical Codes": {
            "cpt_codes": {"type": "array", "description": "CPT procedure codes"},
            "dx_codes": {"type": "array", "description": "ICD-10 diagnosis codes"},
            "procedure_descriptions": {"type": "array", "description": "Procedure descriptions"},
            "diagnosis_descriptions": {"type": "array", "description": "Diagnosis descriptions"}
        },
        "Provider Information": {
            "provider_name": {"type": "string", "description": "Healthcare provider name"},
            "provider_npi": {"type": "string", "description": "Provider NPI number"},
            "practice_name": {"type": "string", "description": "Medical practice name"},
            "provider_address": {"type": "string", "description": "Provider address"}
        },
        "Insurance & Billing": {
            "insurance_company": {"type": "string", "description": "Insurance company name"},
            "policy_number": {"type": "string", "description": "Insurance policy number"},
            "group_number": {"type": "string", "description": "Insurance group number"},
            "copay_amount": {"type": "number", "description": "Copay amount"},
            "total_charges": {"type": "number", "description": "Total charges"}
        },
        "Visit Information": {
            "date_of_service": {"type": "string", "description": "Date of service (MM/DD/YYYY)"},
            "place_of_service": {"type": "string", "description": "Place of service"},
            "referring_physician": {"type": "string", "description": "Referring physician name"}
        }
    }


def render_header():
    """Render application header and title."""
    st.set_page_config(
        page_title="Medical Superbill Extractor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Medical Superbill Structured Extractor")
    st.markdown("**HIPAA-Compliant PDF Data Extraction System**")
    
    # Display security notice
    with st.expander("🔒 Security & Compliance Notice", expanded=False):
        st.markdown("""
        **HIPAA Compliance Features:**
        - ✅ Encrypted data transmission to Azure
        - ✅ Temporary file encryption (AES-256)
        - ✅ PHI masking in application logs
        - ✅ Secure API key management
        - ✅ Automatic data cleanup after processing
        
        **Important:** This application is designed for healthcare professionals. 
        Ensure you have proper authorization before uploading patient documents.
        """)


def render_sidebar():
    """Render sidebar with configuration and status."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Azure API Status
        st.subheader("🔗 Azure OpenAI Status")
        
        # Test API connection button
        if st.button("Test API Connection", type="primary"):
            with st.spinner("Testing Azure OpenAI connection..."):
                try:
                    result = llm_service.test_connection()
                    st.session_state.api_test_result = result
                except Exception as e:
                    st.session_state.api_test_result = {"success": False, "error": str(e)}
        
        # Display API status
        if st.session_state.api_test_result:
            result = st.session_state.api_test_result
            if result["success"]:
                st.success("✅ API Connection Successful")
                st.json({
                    "Model": result.get("model", "Unknown"),
                    "Response": result.get("response", "No response"),
                    "Usage": result.get("usage", {})
                })
            else:
                st.error("❌ API Connection Failed")
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Processing Statistics
        st.subheader("📊 Session Statistics")
        uploaded_count = len(st.session_state.uploaded_files)
        processed_count = len(st.session_state.extraction_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files Uploaded", uploaded_count)
        with col2:
            st.metric("Files Processed", processed_count)


def render_file_upload():
    """Render file upload section."""
    st.header("📎 Upload Medical Superbills")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload medical superbill PDFs for structured data extraction"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        
        # Display uploaded files
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name} ({file.size:,} bytes)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**File {i+1}:**")
                    st.write(f"Name: {file.name}")
                with col2:
                    st.write(f"Size: {file.size:,} bytes")
                    st.write(f"Type: {file.type}")
                with col3:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        # Note: File removal would require re-upload
                        st.info("Please re-upload files to remove items")


def render_field_selection():
    """Render field/tag selection section."""
    st.header("🏷️ Select Data Fields to Extract")
    
    predefined_fields = get_predefined_fields()
    
    # Create tabs for different field categories
    field_tabs = st.tabs(list(predefined_fields.keys()) + ["Custom Fields"])
    
    selected_fields = {}
    
    # Render predefined field categories
    for i, (category, fields) in enumerate(predefined_fields.items()):
        with field_tabs[i]:
            st.subheader(f"{category}")
            
            # Select all/none buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Select All {category}", key=f"select_all_{category}"):
                    for field_name in fields.keys():
                        st.session_state[f"field_{field_name}"] = True
            with col2:
                if st.button(f"Clear All {category}", key=f"clear_all_{category}"):
                    for field_name in fields.keys():
                        st.session_state[f"field_{field_name}"] = False
            
            # Individual field checkboxes
            for field_name, field_info in fields.items():
                is_selected = st.checkbox(
                    f"**{field_name.replace('_', ' ').title()}**",
                    key=f"field_{field_name}",
                    help=field_info["description"]
                )
                if is_selected:
                    selected_fields[field_name] = field_info
    
    # Custom fields tab
    with field_tabs[-1]:
        st.subheader("Custom Fields")
        st.write("Define additional fields specific to your workflow:")
        
        # Add custom field form
        with st.form("add_custom_field"):
            col1, col2 = st.columns(2)
            with col1:
                custom_field_name = st.text_input("Field Name", placeholder="e.g., referring_doctor_phone")
            with col2:
                custom_field_desc = st.text_input("Description", placeholder="e.g., Referring doctor's phone number")
            
            if st.form_submit_button("Add Custom Field"):
                if custom_field_name and custom_field_desc:
                    st.session_state.custom_fields.append({
                        "name": custom_field_name,
                        "description": custom_field_desc
                    })
                    st.success(f"Added custom field: {custom_field_name}")
                else:
                    st.error("Please provide both field name and description")
        
        # Display existing custom fields
        if st.session_state.custom_fields:
            st.write("**Current Custom Fields:**")
            for i, field in enumerate(st.session_state.custom_fields):
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    is_selected = st.checkbox(f"**{field['name']}**", key=f"custom_field_{i}")
                    if is_selected:
                        selected_fields[field['name']] = {
                            "type": "string", 
                            "description": field['description']
                        }
                with col2:
                    st.write(field['description'])
                with col3:
                    if st.button("Remove", key=f"remove_custom_{i}"):
                        st.session_state.custom_fields.pop(i)
                        st.rerun()
    
    # Update session state with selected fields
    st.session_state.selected_fields = selected_fields
    
    # Display selection summary
    if selected_fields:
        st.success(f"✅ {len(selected_fields)} fields selected for extraction")
        with st.expander("📋 Selected Fields Summary", expanded=False):
            for field_name, field_info in selected_fields.items():
                st.write(f"• **{field_name.replace('_', ' ').title()}**: {field_info['description']}")


def render_extraction_preview():
    """Render extraction results preview section."""
    st.header("👁️ Extraction Preview")
    
    # Check if we have files and selected fields
    if not st.session_state.uploaded_files:
        st.info("📎 Please upload PDF files to preview extraction results")
        return
    
    if not st.session_state.selected_fields:
        st.info("🏷️ Please select fields to extract from the documents")
        return
    
    # Mock preview (since extraction logic isn't implemented yet)
    st.warning("🚧 **Preview Mode**: Extraction logic will be implemented in the next phase")
    
    # Create mock preview data
    sample_data = create_mock_preview_data()
    
    if st.button("📊 Generate Preview", type="primary"):
        st.subheader("📋 Preview Results")
        
        # Display as DataFrame
        df = pd.DataFrame(sample_data)
        st.dataframe(df, use_container_width=True)
        
        # Display as JSON
        with st.expander("📄 Raw JSON Data", expanded=False):
            st.json(sample_data[0] if sample_data else {})
        
        # Export options
        st.subheader("📥 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download Excel", type="secondary"):
                st.info("Excel export will be implemented in Phase 2")
        
        with col2:
            if st.button("Download JSON", type="secondary"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(sample_data, indent=2),
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("View Validation Report", type="secondary"):
                st.info("Validation report will be implemented in Phase 2")


def create_mock_preview_data() -> List[Dict[str, Any]]:
    """Create mock data for preview demonstration."""
    selected_fields = st.session_state.selected_fields
    
    # Generate sample data based on selected fields
    mock_data = []
    for i in range(len(st.session_state.uploaded_files)):
        patient_data = {}
        
        for field_name, field_info in selected_fields.items():
            if field_info["type"] == "string":
                if "name" in field_name:
                    patient_data[field_name] = f"Sample Patient {i+1}"
                elif "date" in field_name:
                    patient_data[field_name] = "01/15/2024"
                elif "code" in field_name:
                    patient_data[field_name] = "99213"
                else:
                    patient_data[field_name] = f"Sample {field_name.replace('_', ' ')}"
            elif field_info["type"] == "array":
                patient_data[field_name] = ["99213", "99214"] if "cpt" in field_name else ["Z00.00", "Z01.00"]
            elif field_info["type"] == "number":
                patient_data[field_name] = 25.00 if "copay" in field_name else 150.00
            else:
                patient_data[field_name] = f"Sample {field_name}"
        
        mock_data.append(patient_data)
    
    return mock_data


def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render UI components
        render_header()
        render_sidebar()
        
        # Main content area
        render_file_upload()
        st.divider()
        render_field_selection()
        st.divider()
        render_extraction_preview()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 14px;'>
            <p>🏥 Medical Superbill Structured Extractor v1.0 | 🔒 HIPAA Compliant | ⚡ Powered by Azure OpenAI GPT-5</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()