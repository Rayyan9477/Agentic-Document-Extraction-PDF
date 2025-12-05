# Local Agentic Document Extraction System

## Implementation Plan - 4-Agent Architecture

---

## Executive Summary

This document outlines a streamlined implementation plan for building a local, HIPAA-compliant agentic document extraction system using **Qwen3-VL 8B** via **LM Studio**.

### Key Specifications

| Specification | Value |
|--------------|-------|
| Model | Qwen3-VL 8B |
| Backend | LM Studio (Local) |
| Agents | 4 Specialized |
| Validation | 3-Layer Anti-Hallucination |
| Compliance | HIPAA Ready |
| Timeline | 12 Weeks |

### Design Improvements (vs. Previous 9-Agent Architecture)

| Metric | Old (9 Agents) | New (4 Agents) |
|--------|----------------|----------------|
| VLM Calls/Page | 6-8 | 3-4 |
| Processing Time | 45-60 sec | 15-25 sec |
| Validation Layers | 6 | 3 |
| State Machine States | 10+ | 5 |
| Coordination Overhead | High | Low |

---

## 4-Agent Architecture

```
PDF Input
    │
    ▼
┌─────────────────┐
│  PREPROCESSOR   │  ← Utility Module (NOT an agent)
│  PDF → Images   │     No VLM calls, pure Python/OpenCV
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ORCHESTRATOR   │  Agent 1: State Machine
│   (LangGraph)   │  VLM Calls: 0
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ANALYZER     │  Agent 2: Document Understanding
│                 │  VLM Calls: 1 per document
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    EXTRACTOR    │  Agent 3: Data Extraction
│                 │  VLM Calls: 2 per page (dual-pass)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    VALIDATOR    │  Agent 4: Quality Assurance
│                 │  VLM Calls: 0-1 per document
└────────┬────────┘
         │
         ▼
    JSON Output
```

### Agent Responsibilities

| Agent | Role | VLM Calls | Key Functions |
|-------|------|-----------|---------------|
| **Orchestrator** | State Machine | 0 | Workflow control, error handling, checkpointing, retry logic |
| **Analyzer** | Document Understanding | 1/doc | Classification, structure detection, page relationships, schema selection |
| **Extractor** | Data Extraction | 2/page | Schema-driven extraction, dual-pass verification, confidence scoring |
| **Validator** | Quality Assurance | 0-1/doc | Schema validation, hallucination detection, cross-page merging, output formatting |

---

## Phase 0: Prerequisites & Setup (Week 1)

### Step 1: Procure Hardware

**Minimum Requirements:**
- GPU: RTX 3090 (24GB VRAM)
- RAM: 32GB
- CPU: 8+ cores
- Storage: 500GB NVMe SSD

**Recommended:**
- GPU: RTX 4090 (24GB) or A6000 (48GB)
- RAM: 64GB
- Storage: 1TB NVMe

### Step 2: Install LM Studio

```bash
# Download from https://lmstudio.ai
# Install and enable vision model support
```

### Step 3: Download Qwen3-VL Model

| Quantization | VRAM Required | Notes |
|-------------|---------------|-------|
| FP16 | ~18GB | Maximum accuracy |
| INT8 | ~10GB | Good balance |
| **INT4 (Q4_K_M)** | **~6GB** | **Recommended** |

```
In LM Studio:
1. Search: "Qwen3-VL-8B"
2. Select Q4_K_M quantization
3. Set GPU layers: All
```

### Step 4: Configure Local Server

```
Server Settings:
- Port: 1234
- Context Length: 8192-32768
- GPU Layers: Maximum
- Enable: Vision/Image inputs
```

### Step 5: Setup Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install langgraph PyMuPDF openai pydantic Pillow tenacity fastapi uvicorn
```


**✓ Milestone:** VLM responds correctly to image + text prompt

---

## Phase 1: Core Infrastructure (Weeks 2-3)

### Week 2: PDF Processing Pipeline

**Step 1: Build PDF Ingestion Module**
```python
# pdf_processor.py
import fitz  # PyMuPDF

class PDFProcessor:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
    
    def validate_pdf(self, path: str) -> bool:
        """Check PDF is valid and readable"""
        pass
    
    def extract_pages(self, path: str) -> list[bytes]:
        """Convert each page to PNG at specified DPI"""
        pass
```

**Step 2: Implement Page Extraction (300 DPI)**
```python
def extract_pages(self, pdf_path: str) -> list[bytes]:
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    return images
```

**Step 3: Add Quality Enhancement Filters**
- Auto-rotation detection
- Contrast normalization
- Noise reduction
- Deskewing

**Step 4: Create Batch Processing Manager**
- Memory-efficient streaming
- Progress tracking
- Error recovery

**Step 5: Write Unit Tests**

### Week 2: LM Studio Client

**Step 1: Create Connection Manager**
```python
# lm_client.py
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
    
    def health_check(self) -> bool:
        """Verify LM Studio is running"""
        pass
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def vision_request(self, image_b64: str, prompt: str) -> str:
        """Send image + prompt, return response"""
        pass
```

**Step 2: Build Vision Request Handler**
- Base64 encoding
- Message formatting
- Token limit management

**Step 3: Implement Response Parser**
- JSON extraction from response
- Error handling for malformed output

**Step 4: Add Retry Logic**
- Exponential backoff
- Max attempts: 3
- Timeout handling

**Step 5: Add Health Check Monitoring**

### Week 3: Schema Definition System

**Step 1: Define Base Schema Structure**
```python
# schemas/base.py
from pydantic import BaseModel
from typing import Optional

class FieldDefinition(BaseModel):
    name: str
    type: str  # string, date, currency, code
    required: bool = True
    pattern: Optional[str] = None
    description: str

class DocumentSchema(BaseModel):
    name: str
    description: str
    fields: list[FieldDefinition]
    cross_field_rules: list[str] = []
```

**Step 2: Create Field Type Validators**
- Date validators (MM/DD/YYYY, YYYY-MM-DD)
- Currency validators ($X,XXX.XX)
- Code validators (ICD-10, CPT, NPI)

**Step 3: Build CMS-1500 Schema**
```python
CMS1500_SCHEMA = DocumentSchema(
    name="CMS-1500",
    description="Professional medical claim form",
    fields=[
        FieldDefinition(name="patient_name", type="string", required=True),
        FieldDefinition(name="date_of_service", type="date", required=True),
        FieldDefinition(name="diagnosis_codes", type="icd10_list", required=True),
        FieldDefinition(name="procedure_codes", type="cpt_list", required=True),
        FieldDefinition(name="total_charges", type="currency", required=True),
        # ... additional fields
    ]
)
```

**Step 4: Build EOB/Remittance Schema**

**Step 5: Add Cross-Field Rule Engine**
- Total charges = sum of line items
- Service date <= current date
- NPI format validation (10 digits, Luhn check)

**✓ Deliverables:** PDF processor, LM client, schema library, unit tests

---

## Phase 2: 4-Agent Framework (Weeks 4-6)

### Week 4: Orchestrator Agent

**Step 1: Setup LangGraph Project**
```python
# orchestrator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ExtractionState(TypedDict):
    pdf_path: str
    images: list[bytes]
    doc_type: str
    schema: dict
    extractions: list[dict]
    validation_result: dict
    status: str
    errors: list[str]
```

**Step 2: Define State Machine States**
```python
STATES = ["INIT", "ANALYZE", "EXTRACT", "VALIDATE", "COMPLETE", "ERROR", "REVIEW"]
```

**Step 3: Build Transition Logic**
```python
def build_graph():
    workflow = StateGraph(ExtractionState)
    
    workflow.add_node("analyze", analyzer_agent)
    workflow.add_node("extract", extractor_agent)
    workflow.add_node("validate", validator_agent)
    
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {"complete": END, "review": "review", "retry": "extract"}
    )
    
    return workflow.compile()
```

**Step 4: Implement Checkpointing**
- Save state after each agent
- Enable resume from failure

**Step 5: Add Error Handling**
- Per-agent error catching
- Graceful degradation
- Error logging

**Step 6: Create Retry Policies**
- Max retries per agent
- Backoff strategy

### Week 4-5: Analyzer Agent

**Step 1: Create Agent Base Class**
```python
# agents/base.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, lm_client: LMStudioClient):
        self.lm_client = lm_client
    
    @abstractmethod
    def process(self, state: ExtractionState) -> ExtractionState:
        pass
```

**Step 2: Build Classification Prompt**
```python
CLASSIFICATION_PROMPT = """
Analyze this document image and determine:
1. Document type (CMS-1500, UB-04, EOB, Lab Report, Other)
2. Number of logical sections
3. Whether this appears to be a multi-page document

Respond in JSON format:
{
    "document_type": "...",
    "confidence": 0.0-1.0,
    "sections": [...],
    "is_multi_page": true/false,
    "page_relationship": "standalone|continuation|header"
}
"""
```

**Step 3: Implement Structure Detector**
- Identify tables, forms, free text
- Detect headers and footers
- Map visual layout

**Step 4: Add Page Relationship Logic**
- Detect continuing tables
- Link related pages
- Track document boundaries

**Step 5: Build Schema Selector**
- Map document type to schema
- Handle unknown types gracefully

**Step 6: Test with Sample Documents**

### Week 5: Extractor Agent

**Step 1: Build Extraction Prompt**
```python
def build_extraction_prompt(schema: DocumentSchema, pass_num: int) -> str:
    prompt = f"""
Extract data from this document according to the schema below.
IMPORTANT: Only extract values you can clearly see. Return null for unclear fields.

Schema: {schema.name}
Fields to extract:
"""
    for field in schema.fields:
        prompt += f"- {field.name} ({field.type}): {field.description}\n"
    
    prompt += """
Respond in JSON format with confidence scores:
{
    "fields": {
        "field_name": {"value": "...", "confidence": 0.0-1.0, "location": "..."}
    }
}
"""
    return prompt
```

**Step 2: Implement Dual-Pass Logic**
```python
def extract(self, state: ExtractionState) -> ExtractionState:
    # Pass 1: Standard extraction
    result1 = self.lm_client.vision_request(image, prompt_v1)
    
    # Pass 2: Verification extraction (different prompt)
    result2 = self.lm_client.vision_request(image, prompt_v2)
    
    # Compare and merge
    merged = self.merge_extractions(result1, result2)
    return merged
```

**Step 3: Add Field Comparison**
- Compare pass 1 vs pass 2
- Flag mismatches
- Calculate agreement score

**Step 4: Create Confidence Scorer**
```python
def calculate_confidence(self, r1: dict, r2: dict) -> float:
    if r1["value"] == r2["value"]:
        return max(r1["confidence"], r2["confidence"])
    else:
        return min(r1["confidence"], r2["confidence"]) * 0.5
```

**Step 5: Add Visual Grounding**
- Ask VLM to describe where value was found
- Validate location makes sense

**Step 6: Handle Table Extraction**
- Row/column structure
- Header detection
- Cell value extraction

### Week 6: Validator Agent

**Step 1: Build Schema Validator**
```python
def validate_schema(self, data: dict, schema: DocumentSchema) -> list[str]:
    errors = []
    for field in schema.fields:
        value = data.get(field.name)
        if field.required and value is None:
            errors.append(f"Missing required field: {field.name}")
        if value and field.pattern:
            if not re.match(field.pattern, str(value)):
                errors.append(f"Invalid format for {field.name}")
    return errors
```

**Step 2: Add Pattern Detection (Hallucination)**
```python
def detect_hallucination_patterns(self, data: dict) -> list[str]:
    warnings = []
    values = list(data.values())
    
    # Check for repetitive values
    if len(values) != len(set(values)):
        warnings.append("Repetitive values detected")
    
    # Check for suspiciously round numbers
    # Check for placeholder patterns
    return warnings
```

**Step 3: Implement Cross-Field Rules**
- Total = sum of line items
- End date >= start date
- Age matches DOB

**Step 4: Build Cross-Page Merger**
- Combine table rows across pages
- Deduplicate headers
- Maintain row order

**Step 5: Add Confidence Calibration**
- Threshold tuning per field type
- Historical accuracy tracking

**Step 6: Create Output Formatter**
- JSON output
- Optional Markdown report
- Audit trail

**✓ Deliverables:** 4 working agents, LangGraph integration, agent tests

---

## Phase 3: 3-Layer Anti-Hallucination (Weeks 7-8)

### Layer 1: Prompt Engineering (All Agents)

**Implementation:**
```python
GROUNDING_RULES = """
CRITICAL RULES:
1. Only extract values you can clearly see in the image
2. If a field is unclear or not visible, return null
3. Do not guess or infer values
4. Do not fill in "typical" or "expected" values
5. Include your confidence level for each field
6. Describe WHERE you found each value
"""
```

**Techniques:**
- Explicit grounding instructions
- "Do not guess" rules
- "Return null if unsure" directive
- Structured output format enforcement

### Layer 2: Dual-Pass Extraction (Extractor Agent)

**Implementation:**
- Two separate extraction passes
- Different prompt phrasings
- Field-by-field comparison
- Mismatch flagging

```python
def dual_pass_extract(self, image: bytes, schema: DocumentSchema) -> dict:
    # Pass 1: Direct extraction
    prompt1 = "Extract the following fields from this document..."
    result1 = self.extract(image, prompt1)
    
    # Pass 2: Verification style
    prompt2 = "Verify and extract these fields, being extra careful..."
    result2 = self.extract(image, prompt2)
    
    # Compare
    final = {}
    for field in schema.fields:
        v1 = result1.get(field.name)
        v2 = result2.get(field.name)
        
        if v1 == v2:
            final[field.name] = {"value": v1, "confidence": "high"}
        else:
            final[field.name] = {"value": v1, "confidence": "low", "mismatch": True}
    
    return final
```

### Layer 3: Pattern + Rule Validation (Validator Agent)

**Hallucination Patterns to Detect:**
- Repetitive values across fields
- Suspiciously round numbers
- Placeholder text ("N/A", "TBD", "XXX")
- Values that don't match field type

**Cross-Field Consistency:**
- Math checks (totals, calculations)
- Date logic (chronological order)
- Code validation (NPI checksum, ICD-10 format)

### Confidence Score Actions

| Score | Action | Description |
|-------|--------|-------------|
| 0.95+ | Auto-Accept | High confidence, proceed |
| 0.85-0.94 | Accept + Note | Flag for audit |
| 0.70-0.84 | Verify | Additional validation |
| 0.50-0.69 | Re-Extract | Try extraction again |
| <0.50 | Human Review | Route to reviewer |

### Human-in-the-Loop Integration

**Components:**
- Auto-flagging system
- Review interface (web UI)
- Correction feedback loop
- Model improvement tracking

**Target:** <10% of documents require human review

**✓ Deliverables:** 3-layer validation system, confidence thresholds, review interface

---

## Phase 4: Integration & Testing (Weeks 9-10)

### Week 9: REST API Development

**Step 1: Setup FastAPI Project**
```python
# main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Document Extraction API")

class ExtractionResult(BaseModel):
    task_id: str
    status: str
    result: dict | None
    errors: list[str]
```

**Step 2: Create Extraction Endpoint**
```python
@app.post("/api/v1/extract")
async def extract_document(
    file: UploadFile,
    schema: str = "auto",
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_document, task_id, file, schema)
    return {"task_id": task_id, "status": "processing"}
```

**Step 3: Add Status Endpoint**
```python
@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = get_task_result(task_id)
    return result
```

**Step 4: Implement Batch Processing**
```python
@app.post("/api/v1/batch")
async def batch_extract(files: list[UploadFile]):
    # Process multiple documents
    pass
```

**Step 5: Add Async Task Queue**
- Celery or similar
- Redis backend
- Worker management

### Week 10: Testing Suite

**Step 1: Unit Tests**
```python
# tests/test_agents.py
def test_analyzer_classification():
    """Test document type classification accuracy"""
    pass

def test_extractor_dual_pass():
    """Test dual-pass extraction agreement"""
    pass

def test_validator_rules():
    """Test cross-field validation rules"""
    pass
```

**Step 2: Integration Tests**
```python
def test_end_to_end_cms1500():
    """Test full pipeline with CMS-1500 document"""
    result = pipeline.process("test_cms1500.pdf")
    assert result["status"] == "complete"
    assert result["accuracy"] > 0.95
```

**Step 3: Accuracy Tests (Golden Dataset)**
- 100+ annotated documents
- Per-field accuracy tracking
- Regression detection

**Step 4: Hallucination Adversarial Tests**
- Deliberately tricky documents
- Edge cases
- Ambiguous layouts

**Step 5: Performance Benchmarks**
- Pages per minute
- Memory usage
- GPU utilization

**✓ Deliverables:** REST API, test suite, benchmarks

---

## Phase 5: Deployment (Weeks 11-12)

### Week 11: HIPAA Compliance

**Step 1: Verify 100% Local Processing**
- Audit all network calls
- Block external endpoints
- Document data flow

**Step 2: Implement AES-256 Encryption**
```python
from cryptography.fernet import Fernet

class SecureStorage:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def encrypt_file(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)
    
    def decrypt_file(self, encrypted: bytes) -> bytes:
        return self.cipher.decrypt(encrypted)
```

**Step 3: Setup Role-Based Access Control**
- Admin: Full access
- Operator: Process documents
- Viewer: Read results only

**Step 4: Enable Audit Logging**
```python
# Every action logged with:
# - Timestamp
# - User ID
# - Action type
# - Document ID (hashed)
# - Result status
```

**Step 5: Configure Secure Temp Cleanup**
- Auto-delete after processing
- Secure wipe (overwrite)
- No persistent PHI storage

### Week 12: Production Launch

**Step 1: Setup Monitoring**
- Prometheus metrics
- Grafana dashboards
- Key metrics: latency, accuracy, throughput

**Step 2: Configure Alerting**
- Error rate threshold
- Latency spikes
- GPU memory warnings

**Step 3: Write Operations Runbook**
- Startup/shutdown procedures
- Common issues and fixes
- Escalation paths

**Step 4: Train Operations Team**

**Step 5: Go-Live**
- Pilot with limited documents
- Gradual rollout
- Monitor and adjust

### Capacity Planning

| Configuration | Throughput |
|--------------|------------|
| Single RTX 4090 | 50-100 pages/hour |
| Multi-GPU (2x) | 200-400 pages/hour |
| Distributed | Scales linearly |

**✓ Deliverables:** HIPAA-compliant deployment, monitoring, documentation

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Field Extraction Accuracy | >95% |
| Hallucination Rate | <2% |
| Processing Speed | 15-25 sec/page |
| VLM Calls per Page | 3-4 |
| System Uptime | >99.5% |
| Human Review Rate | <10% |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hallucinations | High | 3-layer validation system |
| Poor document quality | Medium | Enhancement pipeline |
| HIPAA violations | Critical | 100% local processing |
| Model accuracy drift | Medium | Continuous monitoring |
| Hardware failure | Medium | Checkpointing, recovery |
| Prompt injection | Low | Input sanitization |

---

## Immediate Next Steps

1. **Procure hardware** - RTX 4090 or A6000
2. **Collect sample corpus** - 100+ documents per type
3. **Define custom schemas** - Map your document types
4. **Assign dev team** - 2-3 engineers recommended
5. **Begin Phase 0** - Environment setup

---

**Ready to build enterprise-grade document extraction?**

*12 weeks to production-ready local AI document processing.*
