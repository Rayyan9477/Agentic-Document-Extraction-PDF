# 📄 Product Requirements Document (PRD)

**Project Name:** Medical Superbill Structured Extractor
**Version:** 1.1
**Owner:** Rayyan Ahmed


---

## 1\. **Objective**

Develop an end-to-end **HIPAA-compliant structured extraction system** for unstructured and semi-structured **medical superbills** in PDF format. The system should:

* Extract **patient data, CPT codes, DX codes, insurance details, provider info**, and **user-defined fields (tags)**.
* Handle **handwriting, tables, and mixed layouts** dynamically.
* Allow **multi-patient extraction from a single PDF page**.
* Export results into a **well-formatted Excel sheet**.
* Provide a **Streamlit-based UI** for easy interaction.

---

## 2\. **Core Features**

1. **PDF Preprocessing**

   * Split PDF into images (page-level or sub-sections).
   * Normalize resolution, orientation, and enhance quality for OCR.

2. **Dynamic Patient Segmentation**

   * Detect how many **patient records** exist on each page.
   * Segment based on layout cues (bounding boxes, whitespace, separators).

3. **Field Tagging \& Configuration**

   * UI for users to **select fields/tags** they want extracted.
   * Predefined medical fields (Name, DOB, CPT, DX, Insurance, Provider, etc.).
   * Custom fields for flexibility.

4. **Chunking \& Parsing**

   * Break document regions into **logical text chunks**.
   * Use **Docling / OCR engines** for table parsing \& structure retention.

5. **LLM Extraction (Azure GPT-5)**

   * Send parsed chunks to **Azure GPT-5 API**.
   * Prompt-engineer for **structured MD/JSON(so we can easily export and validate it) schema output**.
   * Ensure **PHI handling \& HIPAA compliance** (encryption, logging control).

6. **Data Validation**

   * Cross-check extracted CPT/DX codes with **ICD-10/CPT databases**.
   * Flag missing or ambiguous entries.

7. **Export**

   * Output structured results into **Excel sheets** using `openpyxl`.
   * Proper formatting (columns, validation, multiple patients per sheet).

8. **User Interface (Streamlit)**

   * Upload PDF.
   * Select fields/tags to extract.
   * Preview extracted data before export.
   * Download Excel file.

---

## 3\. **Technology Stack**

### **Frontend/UI**

* **Streamlit** → For intuitive user interface and field/tag selection.

### **Backend / Processing**

* **Python 3.12**
* **Azure GPT-5 API** → For structured data extraction (HIPAA-compliant).
* **OCR Engines**:

  * **GPT 5** (Feeding the chunked parts to LLM for text extraction)
  * **Paddle OCR / EasyOCR** (to double check text extraction).
  * **Docling** for document structure parsing.

* **Chunking**: LangChain / custom parser for token-efficient LLM feeding.

### **Storage \& Export**

* **openpyxl / pandas** → Excel export.
* **SQLite (optional)** → Data Retention /Store extraction logs / audit trail.

### **Compliance**

* **Azure Confidential Computing \& HIPAA compliance**.
* **AES-256 encryption** for temporary data storage.
* **PHI masking in logs**.

---

## 4\. **System Architecture**

### **Flow**

1. **User Uploads PDF** → Streamlit
2. **Preprocessing**

   * Split pages into images (PyMuPDF / pdf2image/ or anyother better package).
   * OCR + Layout analysis (Docling + OCR LLM).

3. **Segmentation**

   * Detect multiple patient records (heuristics + GPT-5 assistance).

4. **Chunking \& Field Mapping**

   * Convert into logical chunks (per patient).
   * Tag fields dynamically (user-selected).

5. **LLM Extraction** (Azure GPT-5)

   * JSON structured extraction.
   * Validate against medical code dictionaries.

6. **Post-processing**

   * Normalize fields (date formats, ICD/CPT validation).
   * Error handling (retry ambiguous chunks).

7. **Output**

   * Write results to Excel (one patient per row).
   * Provide download/ export via Streamlit.

8. \*\*Enable Batch PROCESSING \*\* :

   * To Process multiple pdfs at once and export them together.

---

## 5\. **Detailed Implementation Plan**

### **Phase 1 – Setup \& Infrastructure**

* ✅ Configure  project skeleton.
* ✅ Setup Azure GPT-5 API with HIPAA compliance.

### **Phase 2 – PDF Handling**

* Use **PyMuPDF / pdf2image** → Split PDFs into single images(that would be stored locally temporarily then it would be wiped off, even for keeping data temporarily use encryption) .
* Apply preprocessing (OpenCV: denoise, deskew, enhance, colors).

### **Phase 3 – OCR \& Layout Analysis**

* Apply **Docling + Paddle OCR /EasyOCR** → Extract structured text blocks.
* Use **heuristics** (keywords like “Patient Name(first last name)”, “DOB”) to identify record boundaries.

### **Phase 4 – Patient Segmentation**

* Write **segmentation module**:

  * Count records per page.
  * Split into patient-specific text/image chunks.

### **Phase 5 – Tagging \& Field Config**

* Build **Streamlit form** for field/tag selection.
* Maintain predefined schema (`name, dob, cpt, dx, insurance, provider , problems \& diagnosis if mentioned `).
* Allow **custom tags** with user-defined prompts.

### **Phase 6 – LLM Extraction**

* Sample **prompt template**:

```json
  {
    "PatientName": "...",
    "DOB": "...",
    "CPT\_Codes": \["..."],
    "DX\_Codes": \["..."],
    "Provider": "...",
    "Insurance": "...",
    "CustomFields": {...}
  }
  ```

* Feed chunks into **Azure GPT-5**.
* Collect structured JSON output.

### **Phase 7 – Validation \& Post-Processing**

* Cross-check CPT/DX codes against **medical coding DB/API**.
* Normalize dates, phone numbers, insurance IDs.

### **Phase 8 – Export**

* Use **openpyxl** to write results into **Excel**.
* Each row = one patient record.
* Add formatting (headers, auto-fit columns).

### **Phase 9 – UI/UX Polish**

* Show preview of extracted data in Streamlit before final export.
* Add error messages, progress bars, logs.

### **Phase 10 – Security \& Compliance**

* Encrypt intermediate data with **AES**.
* Ensure **PHI masking** in logs.
* Use **Azure Key Vault** for API key management.



---

## 7\. **Timeline**

* Phase 1: Environment setup, Azure API config, Streamlit UI skeleton.
* Phase 2: PDF preprocessing, OCR, patient segmentation.
* Phase 3: Field tagging, LLM integration, JSON schema extraction.
* Phase 4: Validation, Excel export, Streamlit preview.
* Phase 5: Compliance/security, polishing, testing with real superbills.
