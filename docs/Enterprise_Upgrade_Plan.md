# Enterprise-Grade Document Intelligence Engine — Implementation Plan

> **"Zero-shot document intelligence that sees, understands, and extracts — turning any document into trusted, structured data without a single line of training."**

---

## Context

The system at `d:\Repo\PDF` is already a strong local-first agentic extraction platform (LangGraph + Qwen3-VL 8B). This plan upgrades it into an **enterprise-grade document intelligence engine** that combines the best innovations from every major competitor — while keeping the local-first, HIPAA-by-architecture advantage none of them can match.

### Competitive DNA We're Absorbing

| Competitor | Their Best Innovation | How We Incorporate It |
|---|---|---|
| **LandingAI ADE** | Visual grounding with bbox coordinates, Schema Wizard, 99.16% DocVQA, Parse→Split→Extract pipeline | Phase 1A (bbox), Phase 2C (Schema Wizard), Phase 5A (DocVQA benchmarking) |
| **Pulse** | Bounding box spatial data, webhook-driven async, VPC deployment | Phase 1A (bbox), Phase 5B (webhooks) — we already have VPC/local |
| **Reducto** | 30+ file formats, layout-aware RAG chunking, Studio UI, SOC2/HIPAA/GDPR | Phase 1B (file formats), already have HIPAA |
| **Sema4.ai** | 3-pass agentic (CV→VLM→self-correction), AI-guided data model via NL, DataFrame querying | Phase 2C (NL schema), Phase 3B (self-correction learning) — we already have 3-pass |
| **Arctic-Extract** | Single-step VLM extraction without OCR, 6.6 GiB compact model, 125 pages/GPU | Phase 3C (multi-model with lightweight options) |
| **Extend.ai** | Schema versioning + regression, Composer agent for schema optimization, 95-99% accuracy | Phase 1C (versioning), Phase 5A (regression), Phase 2C (Composer-like suggestion) |
| **GUIDEX** | Auto-generates domain schemas + synthetic training data for zero-shot IE | Phase 2C (schema generation), Phase 3B (synthetic example learning) |
| **LMDX (Google)** | Coordinate grounding in language model extraction | Phase 1A (coordinate grounding in every extraction call) |
| **DocsRay** | Hierarchical RAG with pseudo-TOC | Phase 2A (document splitting with structural TOC) |

**Our unique moat: All of the above, running 100% locally, HIPAA-by-architecture, zero vendor lock-in.**

---

## Implementation Phases

### PHASE 1: FOUNDATION (3 parallel tracks)

---

#### Phase 1A: Visual Grounding & Bounding Boxes
**Inspired by: LandingAI, Pulse, LMDX (Google)**

Every extracted field gets pixel-level coordinates linking it to the source document.

**Files to modify:**

| File | Change |
|---|---|
| `src/pipeline/state.py` | Add `BoundingBoxCoords` frozen dataclass to `FieldMetadata` |
| `src/agents/extractor.py` | Modify `_build_extraction_prompt()` and `_build_adaptive_extraction_prompt()` to request bbox in JSON output; parse bbox from VLM response in `_merge_pass_results()` |
| `src/extraction/multi_record.py` | Add bbox to multi-record extraction prompts and `ExtractedRecord` |
| `src/prompts/grounding_rules.py` | Add bbox output format to `OUTPUT_FORMAT_INSTRUCTION` |
| `src/export/json_exporter.py` | Include bbox in all export formats |
| `src/export/excel_exporter.py` | Add bbox columns to extraction worksheet |
| `src/validation/pattern_detector.py` | Add spatial validation (field bbox in expected document region) |

**Key data structure:**
```python
# In state.py
@dataclass(frozen=True, slots=True)
class BoundingBoxCoords:
    x: float          # Normalized 0-1 (left edge)
    y: float          # Normalized 0-1 (top edge)
    width: float      # Normalized 0-1
    height: float     # Normalized 0-1
    page: int         # 1-indexed
    pixel_x: int      # Absolute pixel coords (computed from page dimensions)
    pixel_y: int
    pixel_width: int
    pixel_height: int

# Add to FieldMetadata:
bbox: BoundingBoxCoords | None = None
```

**VLM prompt addition** (to extraction prompts):
```
For each field, include bounding box as normalized coordinates (0.0-1.0):
"bbox": {"x": left, "y": top, "w": width, "h": height}
where (0,0) is top-left and (1,1) is bottom-right of the page.
```

**No additional VLM calls** — bbox is requested within existing extraction calls.

---

#### Phase 1B: Broader File Format Support
**Inspired by: Reducto (30+ formats)**

**New files to create:**

| File | Purpose |
|---|---|
| `src/preprocessing/base_processor.py` | `BaseFileProcessor` ABC with `process() -> ProcessingResult` |
| `src/preprocessing/file_factory.py` | `FileProcessorFactory` — routes by extension to processor |
| `src/preprocessing/image_processor.py` | PNG/JPG/TIFF/BMP → PageImage (direct PIL conversion) |
| `src/preprocessing/docx_processor.py` | DOCX → PageImage (python-docx + rendering to images) |
| `src/preprocessing/spreadsheet_processor.py` | XLSX/CSV → PageImage (openpyxl + PIL table render) |
| `src/preprocessing/dicom_processor.py` | DICOM → PageImage (pydicom + PIL, with metadata extraction) |
| `src/preprocessing/edi_processor.py` | EDI X12 837/835 → structured dict (text parsing, no images) |

**Files to modify:**

| File | Change |
|---|---|
| `src/preprocessing/pdf_processor.py` | Extend to implement `BaseFileProcessor` interface |
| `src/pipeline/runner.py` | `_load_and_convert_pdf()` → `_load_and_convert_file()` via factory |
| `src/api/routes/documents.py` | Accept all supported extensions, not just `.pdf` |
| `src/api/models.py` | Add supported file types to validation |

**Supported extensions:** `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.docx`, `.doc`, `.xlsx`, `.csv`, `.dcm`, `.dicom`, `.edi`, `.x12`, `.835`, `.837`

**New dependencies:** `python-docx>=1.1.0`, `pydicom>=2.4.0`

---

#### Phase 1C: Schema Versioning & Regression Testing
**Inspired by: Extend.ai (schema versioning + regression gates)**

**New files to create:**

| File | Purpose |
|---|---|
| `src/schemas/versioning.py` | `SchemaVersion`, `SchemaVersionManager`, diff/migration logic |
| `src/schemas/migrations.py` | Schema migration functions (field renames, type changes, additions) |

**Files to modify:**

| File | Change |
|---|---|
| `src/schemas/base.py` | `SchemaRegistry.register()` now also versions via `SchemaVersionManager` |

**Key classes:**
```python
@dataclass
class SchemaVersion:
    schema_name: str
    version: str          # semver "1.2.0"
    schema_hash: str      # SHA-256 of definition
    fields: list[FieldDefinition]
    created_at: datetime
    migration_from: str | None

class SchemaVersionManager:
    def register_version(schema: DocumentSchema) -> SchemaVersion
    def get_latest(name: str) -> SchemaVersion | None
    def get_history(name: str) -> list[SchemaVersion]
    def diff(v1: str, v2: str) -> SchemaDiff
    def migrate_result(result: dict, from_v: str, to_v: str) -> dict
```

**Storage:** `data/schema_versions/{schema_name}/` with JSON version files + golden datasets.

---

### PHASE 2: INTELLIGENT AGENTS (3 new agents)

---

#### Phase 2A: Document Splitter Agent
**Inspired by: LandingAI (Parse→Split→Extract), DocsRay (hierarchical TOC)**

Auto-splits multi-document PDFs (e.g., 200-page batch of mixed EOBs, claims, intake forms) into sub-documents.

**New files to create:**

| File | Purpose |
|---|---|
| `src/agents/splitter.py` | `SplitterAgent` extending `BaseAgent` |

**Key design:**
```python
class SplitterAgent(BaseAgent):
    """VLM-powered document boundary detection.

    Strategy:
    1. Batch pages in groups of 3-5
    2. Ask VLM: "Classify each page — is it a new document or continuation?"
    3. Group pages into DocumentSegments with type + confidence
    4. Handle edge cases: fax cover sheets, blank pages, appendices
    """
    def process(self, state: ExtractionState) -> ExtractionState: ...
    def _classify_boundaries(self, page_images: list[dict]) -> list[DocumentSegment]: ...
```

**State additions** (`src/pipeline/state.py`):
```python
document_segments: list[dict]    # [{start_page, end_page, doc_type, confidence}]
is_multi_document: bool
active_segment_index: int
```

**Graph integration** (`src/agents/orchestrator.py`):
- New node `NODE_SPLIT` between `NODE_PREPROCESS` and pipeline routing
- Conditional edge: `single_document` → normal pipeline, `multi_document` → segment loop
- Segment loop processes each segment through the full extract→validate pipeline

**VLM calls:** ~1 per 5 pages (batch classification)

---

#### Phase 2B: Table Structure Detection Agent
**Inspired by: Arctic-Extract (single-step table understanding), Reducto (cell-level parsing)**

Dedicated table detection BEFORE extraction — the #1 failure mode in document extraction.

**New files to create:**

| File | Purpose |
|---|---|
| `src/agents/table_detector.py` | `TableDetectorAgent` extending `BaseAgent` |
| `src/pipeline/table_types.py` | `DetectedTable`, `TableCell`, `TableRow`, `TableHeader` TypedDicts |

**Key design:**
```python
class TableDetectorAgent(BaseAgent):
    """Detects and structures tables before field extraction.

    Handles: merged cells, nested tables, multi-page tables (continuation detection),
    header rows, total rows, subtotal rows, empty cells.

    Output: TableStructure with cell-level content + bounding boxes.
    The Extractor then uses this for targeted row-by-row extraction.
    """
    def process(self, state: ExtractionState) -> ExtractionState: ...
```

**State additions:**
```python
detected_tables: list[dict]  # Serialized DetectedTable objects per page
```

**Graph integration:** Optional node between `components` and `extract` in the adaptive pipeline. If tables detected, extractor switches to table-aware extraction prompts.

**VLM calls:** 1 per page with tables

---

#### Phase 2C: Intelligent Schema Suggestion (Schema Wizard)
**Inspired by: LandingAI (Schema Wizard), Sema4.ai (NL-guided data models), Extend.ai (Composer agent), GUIDEX (auto domain schemas)**

User uploads document → system proposes schema → user refines via natural language → schema saved with version.

**New files to create:**

| File | Purpose |
|---|---|
| `src/agents/schema_proposal.py` | `SchemaProposalAgent` extending `BaseAgent` |
| `src/api/routes/schemas.py` | Schema suggestion, refinement, and save endpoints |

**Key design:**
```python
class SchemaProposalAgent(BaseAgent):
    """AI-powered schema suggestion with NL refinement.

    Flow:
    1. Analyze document (reuses LayoutAgent + ComponentDetector)
    2. Generate initial schema proposal (reuses SchemaGeneratorAgent)
    3. Present to user with field descriptions + examples
    4. Accept NL refinement: "add copay field", "make diagnosis required"
    5. Re-generate refined schema via VLM
    6. Save to registry with version
    """
    def propose(self, page_images: list[dict]) -> SchemaProposal: ...
    def refine(self, proposal: SchemaProposal, instruction: str) -> SchemaProposal: ...
    def save_to_registry(self, proposal: SchemaProposal) -> DocumentSchema: ...
```

**API endpoints** (`src/api/routes/schemas.py`):
```
POST /api/v1/schemas/suggest          — Upload doc pages, get proposed schema
POST /api/v1/schemas/refine           — Refine with NL instruction
POST /api/v1/schemas/save             — Save to registry (auto-versioned)
GET  /api/v1/schemas                  — List all registered schemas
GET  /api/v1/schemas/{name}/versions  — List schema version history
```

**VLM calls:** 1-2 per proposal, 1 per refinement

---

### PHASE 3: INTELLIGENCE LAYER

---

#### Phase 3A: Confidence Calibration
**Inspired by: Extend.ai (95-99% accuracy claims with calibrated scores)**

A 0.90 confidence should mean 90% actual accuracy. Calibrate against ground truth.

**New files to create:**

| File | Purpose |
|---|---|
| `src/validation/calibration.py` | `ConfidenceCalibrator` with Platt scaling + isotonic regression |

**Key design:**
```python
class ConfidenceCalibrator:
    """Maps raw VLM confidence → calibrated probability using historical data.

    Methods: 'platt' (logistic regression) or 'isotonic' (non-parametric).
    Fit on golden dataset results, persist model, retrain periodically.
    """
    def fit(self, raw_scores: list[float], true_labels: list[bool]) -> None: ...
    def calibrate(self, raw_score: float) -> float: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

**Integration point:** Called in `ValidatorAgent.process()` (`src/agents/validator.py`) after existing confidence calculation. Raw preserved as `raw_confidence`, calibrated becomes primary.

**State additions:**
```python
raw_confidence: float
calibrated_confidence: float
calibration_method: str | None
```

**New dependency:** `scikit-learn>=1.5.0`

---

#### Phase 3B: Incremental Learning from Corrections
**Inspired by: Sema4.ai (self-correction), GUIDEX (synthetic training data), existing Mem0 + CorrectionTracker**

When users correct extractions, the system learns and improves future prompts automatically.

**New files to create:**

| File | Purpose |
|---|---|
| `src/memory/dynamic_prompt.py` | `DynamicPromptEnhancer` — queries Mem0 for correction patterns, injects into prompts |

**Key design:**
```python
class DynamicPromptEnhancer:
    """Enhances extraction prompts with correction history.

    Flow:
    1. Before extraction, query Mem0 for similar past documents
    2. Retrieve correction patterns for relevant fields
    3. Inject as few-shot correction examples: "Common mistake: extracting '01/01/2000' for DOB.
       This is a hallucination pattern — return null if date looks like a default."
    4. After extraction, if user corrects, store correction → Mem0
    """
    def enhance_prompt(self, base_prompt: str, document_type: str, field_names: list[str]) -> str: ...
```

**Integration points:**
- `src/agents/extractor.py`: Call `enhance_prompt()` before each VLM call in both `_perform_extraction_pass()` and `_extract_page_pass_adaptive()`
- `src/memory/correction_tracker.py`: Enhance `record_correction()` to also store in Mem0 with semantic embedding
- New API endpoint: `POST /api/v1/documents/{processing_id}/corrections`

**No additional VLM calls** — enhances existing prompts only.

---

#### Phase 3C: Multi-Model Strategy
**Inspired by: Arctic-Extract (compact VLM), Sema4.ai (3-pass with different models)**

Route different tasks to different models: lightweight for layout, primary for extraction, fast for verification.

**New files to create:**

| File | Purpose |
|---|---|
| `src/client/model_router.py` | `ModelRouter` — routes by task type to appropriate `LMStudioClient` |

**Key design:**
```python
class ModelTask(str, Enum):
    LAYOUT = "layout"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    VERIFICATION = "verification"
    TABLE_DETECTION = "table"
    SCHEMA_GENERATION = "schema"

class ModelRouter:
    """Routes VLM requests to task-appropriate models.

    Confirmed model strategy:
    - layout/classification/table_detection → Florence-2 (fast, lightweight, ~1.5 GiB)
    - extraction/schema_generation → Qwen3-VL 8B (primary, ~6 GiB VRAM)
    - verification/consensus → Qwen3-VL 8B at temperature 0.0

    Florence-2 runs on a separate LM Studio instance (e.g., localhost:1235).
    Falls back to Qwen3-VL 8B if Florence-2 unavailable.
    """
    def get_client(self, task: ModelTask) -> LMStudioClient: ...
```

**Files to modify:**

| File | Change |
|---|---|
| `src/agents/base.py` | `BaseAgent.__init__()` accepts optional `ModelRouter`; `send_vision_request()` routes via router if available |
| `src/config/settings.py` | Add `ModelRoutingSettings` with per-task model configuration |

---

### PHASE 4: VERTICAL-SPECIFIC (Healthcare + Finance)

---

#### Phase 4A: Enhanced XLSX/Markdown/JSON Export (All Verticals)
**Priority: Universal export quality before sector-specific formats**

**Files to modify:**

| File | Change |
|---|---|
| `src/export/excel_exporter.py` | Add bbox columns, table-aware sheet generation, per-record sheets for multi-record, sector-specific formatting (medical codes highlighted, financial totals validated) |
| `src/export/json_exporter.py` | Add bbox to all formats, add `ExportFormat.STRUCTURED` with nested field groups, add DataFrame-compatible flat format |
| `src/export/consolidated_export.py` | Enhance multi-record export with duplicate highlighting, confidence heatmap styling, cross-record validation summary |

**New file to create:**

| File | Purpose |
|---|---|
| `src/export/markdown_exporter.py` | Dedicated `MarkdownExporter` class (currently inline) with table formatting, confidence badges, bbox references, collapsible sections for detailed metadata |

**Key enhancements:**
- Excel: Conditional formatting for confidence (green/yellow/red), hyperlinked bbox references, auto-width columns, frozen panes, summary dashboard sheet
- JSON: Structured mode with field groups, DataFrame-compatible flat mode, streaming JSON for large documents
- Markdown: Clean tables, confidence indicators, collapsible raw data sections, audit trail

**FHIR R4 and EDI 837/835 exports deferred to a future phase** — the existing `_build_fhir_export()` stub in `json_exporter.py` stays as-is for now.

---

#### Phase 4B: Finance-Specific Schemas
**For billing, tax, and banking**

**New files to create:**

| File | Purpose |
|---|---|
| `src/schemas/invoice.py` | Invoice with line-item arrays, PO reference, payment terms |
| `src/schemas/w2.py` | W-2 wage and tax statement |
| `src/schemas/form_1099.py` | 1099-NEC, 1099-MISC, 1099-INT |
| `src/schemas/bank_statement.py` | Bank statement with transaction tables, running balances |

**New FieldTypes** to add to `src/schemas/field_types.py`:
```python
EIN = "ein"                    # Employer Identification Number
ROUTING_NUMBER = "routing_number"
BANK_ACCOUNT = "bank_account"
```

**Currency/locale handling:** Add `CurrencyValue` dataclass with `amount`, `currency_code`, `locale`, `raw_text` to handle international formats (1,234.56 vs 1.234,56).

---

### PHASE 5: PLATFORM CAPABILITIES

---

#### Phase 5A: Evaluation & Benchmarking Framework
**Inspired by: Extend.ai (regression gates), LandingAI (DocVQA benchmarking)**

**New files to create:**

| File | Purpose |
|---|---|
| `src/evaluation/__init__.py` | Package |
| `src/evaluation/benchmark.py` | `BenchmarkRunner` — orchestrates evaluation runs |
| `src/evaluation/golden_dataset.py` | `GoldenDatasetManager` — manages ground truth samples |
| `src/evaluation/metrics.py` | Per-field precision/recall/F1, exact match, char error rate |
| `src/evaluation/ab_testing.py` | A/B testing for prompts, models, temperatures |
| `src/evaluation/regression.py` | Automated regression detection — blocks deploys if accuracy drops |
| `src/api/routes/evaluation.py` | API endpoints for benchmark management |

**Key classes:**
```python
@dataclass
class FieldAccuracy:
    field_name: str
    precision: float
    recall: float
    f1: float
    exact_match_rate: float
    char_error_rate: float

@dataclass
class BenchmarkResult:
    benchmark_id: str
    dataset_name: str
    overall_f1: float
    per_field: dict[str, FieldAccuracy]
    per_document_type: dict[str, float]
    vlm_calls: int
    duration_ms: int
```

**API endpoints:**
```
POST /api/v1/evaluation/benchmark     — Run benchmark against golden dataset
GET  /api/v1/evaluation/results       — List results
GET  /api/v1/evaluation/compare       — Compare two runs (A/B)
POST /api/v1/evaluation/golden        — Upload golden dataset
GET  /api/v1/evaluation/regression    — Check for regressions vs baseline
```

**Prometheus integration:** Add `benchmark_f1`, `benchmark_regression` gauges to `src/monitoring/metrics.py`.

---

#### Phase 5B: Webhook Delivery System
**Inspired by: Pulse (webhook-driven async)**

Enhance existing webhook infrastructure at `src/queue/webhook.py`.

**New files to create:**

| File | Purpose |
|---|---|
| `src/api/routes/webhooks.py` | Registration, management, test delivery endpoints |
| `src/queue/webhook_store.py` | Persistent webhook subscription storage (SQLite-backed) |

**API endpoints:**
```
POST   /api/v1/webhooks                          — Register webhook URL + events
GET    /api/v1/webhooks                          — List subscriptions
DELETE /api/v1/webhooks/{id}                     — Remove
GET    /api/v1/webhooks/{id}/deliveries          — Delivery history
POST   /api/v1/webhooks/{id}/test                — Send test payload
```

**Events:** `extraction.completed`, `extraction.failed`, `extraction.human_review`, `batch.completed`, `benchmark.completed`, `benchmark.regression`

**Webhook payload:** Signed with HMAC-SHA256 for verification.

---

## New Dependencies (cumulative)

```
# Phase 1B: File formats
python-docx>=1.1.0
pydicom>=2.4.0

# Phase 3A: Confidence calibration
scikit-learn>=1.5.0

# Future (deferred): FHIR
# fhir.resources>=7.1.0
```

---

## ExtractionState Additions (cumulative)

All fields added with `total=False` for backward compatibility:

```python
# Phase 1A: Visual Grounding — bbox added to FieldMetadata dataclass, not state

# Phase 2A: Document Splitter
document_segments: list[dict]
is_multi_document: bool
active_segment_index: int

# Phase 2B: Table Detection
detected_tables: list[dict]

# Phase 3A: Confidence Calibration
raw_confidence: float
calibrated_confidence: float
calibration_method: str | None

# Phase 3C: Multi-Model
model_assignments: dict[str, str]
```

---

## New API Routes (cumulative)

Register in `src/api/app.py`:
```python
from src.api.routes import schemas, webhooks, evaluation
app.include_router(schemas.router, prefix="/api/v1", tags=["schemas"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
```

---

## Test Structure

```
tests/
  unit/
    test_visual_grounding.py        # Phase 1A
    test_file_factory.py            # Phase 1B
    test_schema_versioning.py       # Phase 1C
    test_splitter_agent.py          # Phase 2A
    test_table_detector.py          # Phase 2B
    test_schema_proposal.py         # Phase 2C
    test_confidence_calibration.py  # Phase 3A
    test_dynamic_prompt.py          # Phase 3B
    test_model_router.py            # Phase 3C
    test_enhanced_exports.py        # Phase 4A
    test_finance_schemas.py         # Phase 4B
    test_benchmark.py               # Phase 5A
    test_webhook_store.py           # Phase 5B
  integration/
    test_splitter_pipeline.py
    test_table_extraction_e2e.py
    test_export_e2e.py
    test_benchmark_e2e.py
  regression/
    test_schema_regression.py
    golden_datasets.py
```

---

## Implementation Order & Dependencies

```
Phase 1 (parallel — no dependencies between tracks):
  1A: Visual Grounding         3-4 days
  1B: File Format Support      4-5 days
  1C: Schema Versioning        3-4 days

Phase 2 (after Phase 1):
  2A: Document Splitter        5-7 days  (needs 1A for bbox)
  2B: Table Detection          5-7 days  (needs 1A for bbox)
  2C: Schema Wizard            3-4 days  (needs 1C for versioning)

Phase 3 (after Phase 2):
  3A: Confidence Calibration   2-3 days  (independent)
  3B: Incremental Learning     3-4 days  (independent)
  3C: Multi-Model (Florence-2) 4-5 days  (independent)

Phase 4 (after Phase 1B for file formats):
  4A: Enhanced Exports         3-4 days  (independent)
  4B: Finance Schemas          3-4 days  (independent)

Phase 5 (after Phase 1C for versioning):
  5A: Evaluation Framework     5-7 days  (needs 1C)
  5B: Webhook Enhancement      2-3 days  (independent)

Future (deferred):
  FHIR R4 Export               4-5 days
  EDI 837/835 Support          5-7 days
```

---

## Verification Plan

After each phase:

1. **Unit tests pass:** `pytest tests/unit/test_{feature}.py -v`
2. **Integration tests pass:** `pytest tests/integration/ -v`
3. **Existing tests don't break:** `pytest tests/ -v --tb=short`
4. **Manual smoke test:** Process a sample document through the full pipeline
5. **API verification:** Hit new endpoints via `curl` or Swagger UI at `/docs`
6. **Benchmark (Phase 5A onwards):** Run against golden dataset, verify no regression

---

## Architecture After Implementation

```
Document Input (PDF/DOCX/TIFF/DICOM/XLSX/EDI)
    |
    v
FileProcessorFactory -> PageImage[]
    |
    v
+--------------------------------------------------+
|              AGENT LAYER (LangGraph)             |
|                                                  |
|  Orchestrator (routing, checkpointing, retry)    |
|       |                                          |
|       v                                          |
|  SplitterAgent (multi-doc boundary detection)    |
|       |                                          |
|       +---> [Per segment]                        |
|       |                                          |
|       v                                          |
|  LayoutAgent -> ComponentDetector                |
|       |              |                           |
|       v              v                           |
|  TableDetectorAgent (cell-level structure)       |
|       |                                          |
|       v                                          |
|  SchemaGenerator (zero-shot) OR SchemaRegistry   |
|       |                                          |
|       v                                          |
|  ExtractorAgent (dual-pass + bbox + dynamic      |
|    prompt enhancement from Mem0 corrections)     |
|       |                                          |
|       v                                          |
|  ValidatorAgent (3-layer anti-hallucination      |
|    + confidence calibration + table validation)  |
|       |                                          |
|       v                                          |
|  Route -> Complete / Retry / Human Review        |
+--------------------------------------------------+
    |
    v
+--------------------------------------------------+
|              MODEL LAYER (Multi-Model)           |
|  ModelRouter -> Florence-2 (layout/classify/     |
|                table detection -- fast, 1.5 GiB) |
|             -> Qwen3-VL 8B (extraction/schema    |
|                generation -- primary, 6 GiB)     |
|             -> Qwen3-VL 8B t=0.0 (consensus)    |
+--------------------------------------------------+
    |
    v
+--------------------------------------------------+
|              OUTPUT LAYER                        |
|  JSON / Excel (XLSX) / Markdown                  |
|  + Visual Grounding (bbox overlay on source)     |
+--------------------------------------------------+
    |
    v
+--------------------------------------------------+
|              PLATFORM LAYER                      |
|  Schema Wizard (NL-guided)                       |
|  Evaluation Framework (golden datasets, A/B)     |
|  Webhook Delivery (HMAC-signed)                  |
|  Incremental Learning (Mem0 corrections)         |
|  Schema Versioning (with regression gates)       |
+--------------------------------------------------+
```
