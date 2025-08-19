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
    from src.processing.pipeline import processing_pipeline
    from src.extraction.field_manager import field_manager, FieldType
    from src.extraction.schema_generator import schema_generator
    
    # Import export functionality
    from src.export.excel_exporter import excel_exporter
    EXCEL_EXPORT_AVAILABLE = excel_exporter is not None
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    EXCEL_EXPORT_AVAILABLE = False
    excel_exporter = None
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
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'processing_options' not in st.session_state:
        st.session_state.processing_options = {}


def get_field_type_display(field_type: FieldType) -> str:
    """Get display string for field type."""
    type_mapping = {
        FieldType.STRING: "📝 Text",
        FieldType.NUMBER: "🔢 Number",
        FieldType.DATE: "📅 Date",
        FieldType.ARRAY: "📋 List",
        FieldType.BOOLEAN: "☑️ Yes/No",
        FieldType.PHONE: "📞 Phone",
        FieldType.EMAIL: "📧 Email",
        FieldType.ADDRESS: "🏠 Address",
        FieldType.MEDICAL_CODE: "🏥 Medical Code"
    }
    return type_mapping.get(field_type, "📝 Text")


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
    """Render enhanced field/tag selection section."""
    st.header("🏷️ Select Data Fields to Extract")
    
    # Get fields by category from field manager
    categorized_fields = field_manager.get_fields_by_category()
    
    # Create tabs for different field categories
    field_tabs = st.tabs(list(categorized_fields.keys()))
    
    selected_fields = []
    
    # Render field categories
    for i, (category, fields) in enumerate(categorized_fields.items()):
        with field_tabs[i]:
            st.subheader(f"{category}")
            
            # Category description
            if category in field_manager.field_categories:
                st.markdown(f"*{field_manager.field_categories[category]}*")
            
            # Select all/none buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Select All {category}", key=f"select_all_{category}"):
                    for field in fields:
                        st.session_state[f"field_{field.name}"] = True
            with col2:
                if st.button(f"Clear All {category}", key=f"clear_all_{category}"):
                    for field in fields:
                        st.session_state[f"field_{field.name}"] = False
            
            # Individual field checkboxes with enhanced display
            for field in fields:
                col_checkbox, col_info = st.columns([3, 1])
                
                with col_checkbox:
                    is_selected = st.checkbox(
                        f"**{field.display_name}**",
                        key=f"field_{field.name}",
                        help=field.description
                    )
                    
                    if is_selected:
                        selected_fields.append(field.name)
                    
                    # Show field details
                    type_display = get_field_type_display(field.field_type)
                    required_indicator = " 🔴" if field.required else ""
                    st.caption(f"{type_display}{required_indicator} - {field.description}")
                    
                    # Show examples if available
                    if field.examples:
                        with st.expander(f"Examples for {field.display_name}", expanded=False):
                            for example in field.examples[:3]:  # Show first 3 examples
                                st.code(example, language=None)
                
                with col_info:
                    if field.validation_pattern:
                        st.caption("📏 Validated")
                    if field.extraction_hints:
                        st.caption(f"💡 {len(field.extraction_hints)} hints")
    
    # Custom fields management (integrated into Custom Fields tab)
    if "Custom Fields" in categorized_fields:
        custom_tab_index = list(categorized_fields.keys()).index("Custom Fields")
        with field_tabs[custom_tab_index]:
            st.divider()
            st.subheader("➕ Add New Custom Field")
            
            with st.form("add_custom_field"):
                col1, col2 = st.columns(2)
                
                with col1:
                    custom_field_name = st.text_input(
                        "Field Name", 
                        placeholder="e.g., referring_doctor_phone",
                        help="Use lowercase with underscores (e.g., my_custom_field)"
                    )
                    
                    custom_field_type = st.selectbox(
                        "Field Type",
                        options=list(FieldType),
                        format_func=get_field_type_display
                    )
                
                with col2:
                    custom_field_display = st.text_input(
                        "Display Name", 
                        placeholder="e.g., Referring Doctor Phone"
                    )
                    
                    custom_field_desc = st.text_area(
                        "Description", 
                        placeholder="e.g., Phone number of the referring physician",
                        max_chars=200
                    )
                
                # Optional advanced settings
                with st.expander("🔧 Advanced Settings", expanded=False):
                    validation_pattern = st.text_input(
                        "Validation Pattern (Regex)",
                        placeholder="e.g., ^[\(\)\d\s\-\.+]+$ for phone numbers",
                        help="Optional regular expression to validate extracted values"
                    )
                    
                    examples_text = st.text_input(
                        "Examples (comma-separated)",
                        placeholder="e.g., (555) 123-4567, 555-987-6543",
                        help="Provide examples to help the AI understand the field"
                    )
                    
                    hints_text = st.text_input(
                        "Extraction Hints (comma-separated)",
                        placeholder="e.g., referring phone, ref doctor phone, referral contact",
                        help="Keywords that might appear near this field in documents"
                    )
                
                if st.form_submit_button("➕ Add Custom Field", type="primary"):
                    if custom_field_name and custom_field_display and custom_field_desc:
                        # Parse examples and hints
                        examples = [ex.strip() for ex in examples_text.split(",") if ex.strip()] if examples_text else None
                        hints = [hint.strip() for hint in hints_text.split(",") if hint.strip()] if hints_text else None
                        
                        success = field_manager.add_custom_field(
                            name=custom_field_name,
                            display_name=custom_field_display,
                            field_type=custom_field_type,
                            description=custom_field_desc,
                            validation_pattern=validation_pattern if validation_pattern else None,
                            examples=examples,
                            extraction_hints=hints
                        )
                        
                        if success:
                            st.success(f"✅ Added custom field: {custom_field_display}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add custom field. Please check the field name.")
                    else:
                        st.error("❌ Please fill in all required fields")
            
            # Manage existing custom fields
            custom_fields = [f for f in categorized_fields.get("Custom Fields", [])]
            if custom_fields:
                st.divider()
                st.subheader("🗂️ Manage Custom Fields")
                
                for field in custom_fields:
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**{field.display_name}** ({field.name})")
                        st.caption(f"{get_field_type_display(field.field_type)} - {field.description}")
                    
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_custom_{field.name}"):
                            if field_manager.remove_custom_field(field.name):
                                st.success(f"Removed {field.display_name}")
                                st.rerun()
                            else:
                                st.error("Failed to remove field")
    
    # Store selected fields in session state
    st.session_state.selected_fields = selected_fields
    
    # Display selection summary
    if selected_fields:
        st.success(f"✅ {len(selected_fields)} fields selected for extraction")
        
        # Generate schema preview
        with st.expander("📋 Selected Fields Summary & Schema Preview", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Selected Fields")
                for field_name in selected_fields:
                    field_def = field_manager.get_field_definition(field_name)
                    if field_def:
                        required_indicator = " 🔴" if field_def.required else ""
                        st.write(f"• **{field_def.display_name}**{required_indicator}")
                        st.caption(f"  {field_def.description}")
            
            with col2:
                st.subheader("JSON Schema Preview")
                schema_preview = schema_generator.generate_llm_prompt_schema(selected_fields)
                st.code(schema_preview, language="json")
    else:
        st.info("👆 Select fields from the tabs above to extract data from your documents")
        


def render_processing_options():
    """Render processing configuration options."""
    st.header("⚙️ Processing Options")
    
    with st.expander("🔧 Advanced Processing Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Processing")
            apply_preprocessing = st.checkbox("Apply image preprocessing", value=True)
            enhance_contrast = st.checkbox("Enhance contrast", value=True)
            denoise = st.checkbox("Apply noise reduction", value=True)
            
            st.subheader("Text Extraction")
            use_llm_extraction = st.checkbox("Use LLM for text extraction", value=True)
            use_ocr_validation = st.checkbox("Use OCR for validation", value=True)
        
        with col2:
            st.subheader("Document Analysis")
            correct_skew = st.checkbox("Correct image skew", value=True)
            normalize_colors = st.checkbox("Normalize colors", value=True)
            
            st.subheader("Patient Segmentation")
            use_llm_segmentation = st.checkbox("Use LLM for segmentation", value=True)
            
            image_dpi = st.slider("Image resolution (DPI)", 150, 600, 300, 50)
        
        # Store options in session state
        st.session_state.processing_options = {
            "apply_preprocessing": apply_preprocessing,
            "enhance_contrast": enhance_contrast,
            "denoise": denoise,
            "correct_skew": correct_skew,
            "normalize_colors": normalize_colors,
            "use_llm_extraction": use_llm_extraction,
            "use_ocr_validation": use_ocr_validation,
            "use_llm_segmentation": use_llm_segmentation,
            "image_dpi": image_dpi,
            "use_memory_storage": True
        }


def render_extraction_preview():
    """Render extraction results preview section."""
    st.header("🚀 PDF Processing & Extraction")
    
    # Check if we have files and selected fields
    if not st.session_state.uploaded_files:
        st.info("📎 Please upload PDF files to begin processing")
        return
    
    if not st.session_state.selected_fields:
        st.info("🏷️ Please select fields to extract from the documents")
        return
    
    # Processing button
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
            process_uploaded_files()
    
    # Display processing results if available
    if st.session_state.processing_results:
        render_processing_results()
        
        # Add batch export option if multiple files processed
        if len(st.session_state.processing_results) > 1:
            render_batch_export_options()


def process_uploaded_files():
    """Process all uploaded PDF files through the complete pipeline."""
    try:
        st.session_state.processing_results = {}
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(st.session_state.uploaded_files)
        
        for i, uploaded_file in enumerate(st.session_state.uploaded_files):
            # Update progress
            progress_bar.progress((i) / total_files)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Read file bytes
                pdf_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Process through pipeline with selected fields
                processing_results = processing_pipeline.process_pdf_complete(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    processing_options=st.session_state.processing_options,
                    selected_fields=st.session_state.selected_fields
                )
                
                # Store results
                st.session_state.processing_results[uploaded_file.name] = processing_results
                
                # Log success
                logger.info(f"Successfully processed {uploaded_file.name}")
                
            except Exception as e:
                error_msg = f"Failed to process {uploaded_file.name}: {e}"
                logger.error(error_msg)
                
                # Store error result
                st.session_state.processing_results[uploaded_file.name] = {
                    "filename": uploaded_file.name,
                    "success": False,
                    "error": str(e)
                }
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Processing completed!")
        
        # Show success message
        successful_files = sum(1 for result in st.session_state.processing_results.values() 
                             if result.get("success", False))
        
        if successful_files > 0:
            st.success(f"✅ Successfully processed {successful_files}/{total_files} files")
        else:
            st.error("❌ No files were processed successfully")
        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        logger.error(f"File processing failed: {e}")


def render_processing_results():
    """Display processing results and extracted data."""
    st.header("📊 Processing Results")
    
    # Summary statistics
    total_files = len(st.session_state.processing_results)
    successful_files = sum(1 for result in st.session_state.processing_results.values() 
                          if result.get("success", False))
    total_patients = sum(len(result.get("patient_records", [])) 
                        for result in st.session_state.processing_results.values())
    total_structured = sum(len(result.get("structured_data", [])) 
                          for result in st.session_state.processing_results.values())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Files Processed", f"{successful_files}/{total_files}")
    with col2:
        st.metric("Total Patients", total_patients)
    with col3:
        st.metric("Structured Records", total_structured)
    with col4:
        avg_confidence = calculate_average_confidence()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col5:
        total_processing_time = sum(result.get("processing_metadata", {}).get("total_duration", 0) 
                                  for result in st.session_state.processing_results.values())
        st.metric("Processing Time", f"{total_processing_time:.1f}s")
    
    # Detailed results per file
    for filename, result in st.session_state.processing_results.items():
        render_file_results(filename, result)


def render_file_results(filename: str, result: Dict[str, Any]):
    """Render results for a single file."""
    with st.expander(f"📄 {filename}", expanded=True):
        if not result.get("success", False):
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return
        
        # File summary
        patient_records = result.get("patient_records", [])
        structured_data = result.get("structured_data", [])
        pages = result.get("pages", [])
        selected_fields = result.get("selected_fields", [])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Pages:** {len(pages)}")
            st.write(f"**Patients Found:** {len(patient_records)}")
        with col2:
            st.write(f"**Structured Records:** {len(structured_data)}")
            st.write(f"**Fields Extracted:** {len(selected_fields)}")
        with col3:
            processing_time = result.get("processing_metadata", {}).get("total_duration", 0)
            st.write(f"**Processing Time:** {processing_time:.1f}s")
            
            stages = result.get("processing_metadata", {}).get("stages_completed", [])
            st.write(f"**Stages Completed:** {len(stages)}")
        
        # Structured data results
        if structured_data:
            render_structured_data_results(filename, structured_data, selected_fields)
        elif patient_records:
            # Fallback to patient records if structured extraction wasn't performed
            st.subheader("👥 Patient Records (Raw)")
            
            # Convert to display format
            display_records = []
            for record in patient_records:
                display_record = {
                    "Patient ID": record.get("patient_identifier", "Unknown"),
                    "Record ID": record.get("record_id", ""),
                    "Confidence": f"{record.get('confidence', 0):.1%}",
                    "Text Length": len(record.get("extracted_text", "")),
                    "Elements": record.get("metadata", {}).get("element_count", 0)
                }
                display_records.append(display_record)
            
            df = pd.DataFrame(display_records)
            st.dataframe(df, use_container_width=True)
            
            # Individual patient details
            st.subheader("📋 Patient Details")
            for i, record in enumerate(patient_records):
                with st.expander(f"Patient {i+1}: {record.get('patient_identifier', 'Unknown')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.text_area(
                            "Extracted Text",
                            record.get("extracted_text", ""),
                            height=200,
                            key=f"text_{filename}_{i}"
                        )
                    
                    with col2:
                        st.json(record.get("metadata", {}))
        
        # Processing details
        with st.expander("🔍 Processing Details", expanded=False):
            st.json(result.get("processing_metadata", {}))


def render_structured_data_results(filename: str, structured_data: List[Dict[str, Any]], selected_fields: List[str]):
    """Render structured data extraction results."""
    st.subheader("🗄 Structured Data Extraction Results")
    
    if not structured_data:
        st.info("No structured data extracted")
        return
    
    # Summary table
    summary_data = []
    for i, record in enumerate(structured_data):
        summary_record = {
            "Patient": record.get("patient_id", f"Patient {i+1}"),
            "Confidence": f"{record.get('confidence_score', 0):.1%}",
            "Fields Extracted": len([k for k, v in record.get("extracted_data", {}).items() if v]),
            "Missing Fields": len(record.get("missing_fields", [])),
            "Validation Errors": len(record.get("validation_errors", []))
        }
        summary_data.append(summary_record)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Detailed extraction results
    st.subheader("📊 Detailed Extraction Results")
    
    for i, record in enumerate(structured_data):
        patient_id = record.get("patient_id", f"Patient {i+1}")
        confidence = record.get("confidence_score", 0)
        extracted_data = record.get("extracted_data", {})
        
        with st.expander(f"👤 {patient_id} (Confidence: {confidence:.1%})", expanded=False):
            tab1, tab2, tab3 = st.tabs(["📄 Extracted Data", "📈 Field Analysis", "🔍 Validation"])
            
            with tab1:
                if extracted_data:
                    # Display extracted fields in a nice format
                    for field_name in selected_fields:
                        field_def = field_manager.get_field_definition(field_name)
                        field_value = extracted_data.get(field_name)
                        
                        if field_value is not None and field_value != "" and field_value != []:
                            display_name = field_def.display_name if field_def else field_name.replace('_', ' ').title()
                            
                            if isinstance(field_value, list):
                                st.write(f"**{display_name}:**")
                                for item in field_value:
                                    st.write(f"  • {item}")
                            else:
                                st.write(f"**{display_name}:** {field_value}")
                        else:
                            display_name = field_def.display_name if field_def else field_name.replace('_', ' ').title()
                            st.write(f"**{display_name}:** *Not found*")
                else:
                    st.warning("No data extracted")
            
            with tab2:
                field_confidence = record.get("field_confidence", {})
                missing_fields = record.get("missing_fields", [])
                
                if field_confidence:
                    st.write("**Field Confidence Scores:**")
                    confidence_data = []
                    for field_name in selected_fields:
                        field_def = field_manager.get_field_definition(field_name)
                        display_name = field_def.display_name if field_def else field_name.replace('_', ' ').title()
                        confidence_score = field_confidence.get(field_name, 0)
                        status = "✅ Found" if field_name not in missing_fields else "❌ Missing"
                        
                        confidence_data.append({
                            "Field": display_name,
                            "Confidence": f"{confidence_score:.1%}",
                            "Status": status
                        })
                    
                    confidence_df = pd.DataFrame(confidence_data)
                    st.dataframe(confidence_df, use_container_width=True)
            
            with tab3:
                validation_errors = record.get("validation_errors", [])
                processing_notes = record.get("processing_notes", [])
                
                if validation_errors:
                    st.write("**Validation Errors:**")
                    for error in validation_errors:
                        st.error(f"⚠️ {error}")
                else:
                    st.success("✅ No validation errors")
                
                if processing_notes:
                    st.write("**Processing Notes:**")
                    for note in processing_notes:
                        st.info(f"📝 {note}")
                
                # Raw extraction metadata
                with st.expander("🔍 Raw Extraction Metadata", expanded=False):
                    st.json(record.get("extraction_metadata", {}))
    
    # Export options
    st.subheader("📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if EXCEL_EXPORT_AVAILABLE:
            if st.button(f"📄 Export to Excel", key=f"excel_{filename}"):
                try:
                    with st.spinner("Generating Excel report..."):
                        # Prepare processing results for this file
                        file_results = {filename: st.session_state.processing_results[filename]}
                        
                        # Generate Excel file
                        excel_bytes, excel_filename = excel_exporter.export_structured_data(
                            file_results, 
                            f"Medical_Superbill_{filename.replace('.pdf', '')}"
                        )
                        
                        # Provide download button
                        st.download_button(
                            label="💾 Download Excel Report",
                            data=excel_bytes,
                            file_name=excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_excel_{filename}"
                        )
                        
                        st.success("✅ Excel report generated successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Excel export failed: {e}")
        else:
            if st.button(f"📄 Export to Excel", key=f"excel_{filename}"):
                st.warning("⚠️ Excel export requires openpyxl. Install with: pip install openpyxl")
    
    with col2:
        # JSON export
        json_data = json.dumps(structured_data, indent=2)
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"structured_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"json_{filename}"
        )
    
    with col3:
        if st.button(f"📈 Validation Report", key=f"report_{filename}"):
            st.info("🚧 Detailed validation report coming in Phase 4")


def render_batch_export_options():
    """Render batch export options for multiple files."""
    st.header("📦 Batch Export Options")
    
    successful_files = {k: v for k, v in st.session_state.processing_results.items() 
                       if v.get("success", False)}
    
    if not successful_files:
        st.info("No successfully processed files available for batch export")
        return
    
    st.info(f"📊 {len(successful_files)} files available for batch export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if EXCEL_EXPORT_AVAILABLE:
            if st.button("📄 Export All to Excel", type="primary"):
                try:
                    with st.spinner("Generating comprehensive Excel report..."):
                        # Generate Excel file with all results
                        excel_bytes, excel_filename = excel_exporter.export_structured_data(
                            successful_files,
                            "Medical_Superbill_Batch_Export"
                        )
                        
                        # Provide download button
                        st.download_button(
                            label="💾 Download Complete Excel Report",
                            data=excel_bytes,
                            file_name=excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_batch_excel"
                        )
                        
                        st.success(f"✅ Batch Excel report generated for {len(successful_files)} files!")
                        
                except Exception as e:
                    st.error(f"❌ Batch Excel export failed: {e}")
        else:
            st.button("📄 Export All to Excel", disabled=True, 
                     help="Excel export requires openpyxl. Install with: pip install openpyxl")
    
    with col2:
        # Batch JSON export
        if st.button("📊 Export All JSON", type="secondary"):
            try:
                # Combine all structured data
                all_structured_data = []
                for filename, result in successful_files.items():
                    for record in result.get("structured_data", []):
                        record_with_source = record.copy()
                        record_with_source["source_file"] = filename
                        all_structured_data.append(record_with_source)
                
                json_data = json.dumps({
                    "export_timestamp": datetime.now().isoformat(),
                    "total_files": len(successful_files),
                    "total_records": len(all_structured_data),
                    "files_processed": list(successful_files.keys()),
                    "structured_data": all_structured_data
                }, indent=2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Medical_Superbill_Batch_{timestamp}.json"
                
                st.download_button(
                    label="💾 Download Batch JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key="download_batch_json"
                )
                
                st.success(f"✅ Batch JSON prepared with {len(all_structured_data)} records!")
                
            except Exception as e:
                st.error(f"❌ Batch JSON export failed: {e}")
    
    with col3:
        # Validation summary
        if st.button("📈 Validation Summary", type="secondary"):
            render_batch_validation_summary(successful_files)


def render_batch_validation_summary(processing_results: Dict[str, Any]):
    """Render batch validation summary."""
    st.subheader("🔍 Batch Validation Summary")
    
    total_records = 0
    total_errors = 0
    field_error_counts = {}
    confidence_scores = []
    
    for filename, result in processing_results.items():
        for record in result.get("structured_data", []):
            total_records += 1
            
            # Collect confidence scores
            confidence = record.get("confidence_score", 0)
            if confidence > 0:
                confidence_scores.append(confidence)
            
            # Count validation errors
            validation_errors = record.get("validation_errors", [])
            total_errors += len(validation_errors)
            
            # Count missing fields
            missing_fields = record.get("missing_fields", [])
            for field in missing_fields:
                field_error_counts[field] = field_error_counts.get(field, 0) + 1
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", total_records)
    with col2:
        st.metric("Validation Errors", total_errors)
    with col3:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        quality_rate = (total_records - len([r for r in processing_results.values() 
                                           for record in r.get("structured_data", []) 
                                           if record.get("validation_errors")])) / max(total_records, 1)
        st.metric("Quality Rate", f"{quality_rate:.1%}")
    
    # Most common missing fields
    if field_error_counts:
        st.subheader("📊 Most Common Missing Fields")
        
        missing_df = pd.DataFrame([
            {
                "Field": field.replace('_', ' ').title(),
                "Missing Count": count,
                "Missing Rate": f"{(count / max(total_records, 1)):.1%}"
            }
            for field, count in sorted(field_error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        st.dataframe(missing_df, use_container_width=True)


def calculate_average_confidence() -> float:
    """Calculate average confidence across all structured records."""
    try:
        all_confidences = []
        for result in st.session_state.processing_results.values():
            if result.get("success"):
                # Use structured data confidence if available
                for record in result.get("structured_data", []):
                    confidence = record.get("confidence_score", 0)
                    if confidence > 0:
                        all_confidences.append(confidence)
                
                # Fallback to patient records if no structured data
                if not result.get("structured_data"):
                    for record in result.get("patient_records", []):
                        confidence = record.get("confidence", 0)
                        if confidence > 0:
                            all_confidences.append(confidence)
        
        if all_confidences:
            return sum(all_confidences) / len(all_confidences)
        return 0.0
    except Exception:
        return 0.0


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
        render_processing_options()
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