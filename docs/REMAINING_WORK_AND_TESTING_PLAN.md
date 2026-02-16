# Remaining Work & Comprehensive Testing Plan

> Continuation plan for the Enterprise-Grade Document Intelligence Engine at `d:\Repo\PDF`.
> Covers all pending implementation phases + a full-spectrum testing strategy.

---

## Current Status

### Completed (Phase 1 + Phase 2A partial)

| Phase | Deliverable | Tests |
|---|---|---|
| **1A** Visual Grounding | `BoundingBoxCoords`, bbox in FieldMetadata, spatial validation in `pattern_detector.py`, bbox in JSON/Excel exports, extractor `_parse_bbox()` | 55 tests in `test_visual_grounding.py` |
| **1B** File Format Support | `BaseFileProcessor` ABC, `FileProcessorFactory`, `ImageProcessor`, `DocxProcessor`, `SpreadsheetProcessor`, `DicomProcessor`, `EDIProcessor` | 33 tests in `test_file_factory.py` |
| **1C** Schema Versioning | `SchemaVersionManager`, `SchemaVersion`, `SchemaDiff`, diff/migration logic | 18 tests in `test_schema_versioning.py` |
| **2A** Document Splitter | `SplitterAgent`, `DocumentSegment`, state fields (`document_segments`, `is_multi_document`, `active_segment_index`), orchestrator `_run_splitter` node + `NODE_SPLIT` wiring | 38 tests in `test_splitter_agent.py` |

**Total new tests from Phases 1-2A: 144 tests (all passing)**
**Total project tests: ~790 functions across 33 files**

### In Progress

- **Phase 2A.3**: Orchestrator integration for `SplitterAgent` — code written, needs final test run
- **Phase 2A.4**: Full regression test run pending

---

## PART 1: Remaining Implementation Work

### WK-1: Finish Phase 2A (Splitter Wrap-up)

**Status**: Code written, needs test run + verification.

| Task | File | What |
|---|---|---|
| Run regression tests | - | `pytest tests/ -v --tb=short` to confirm orchestrator integration doesn't break anything |
| Verify `NODE_SPLIT` wiring | `src/agents/orchestrator.py` | The `_run_splitter` node and `NODE_SPLIT` edge are already added. Confirm graph compiles. |

---

### WK-2: Phase 2B — Table Structure Detection Agent

**New files:**

| File | Purpose |
|---|---|
| `src/agents/table_detector.py` | `TableDetectorAgent(BaseAgent)` — VLM-based table detection |
| `src/pipeline/table_types.py` | `DetectedTable`, `TableCell`, `TableRow`, `TableHeader` TypedDicts |

**Modify:**

| File | Change |
|---|---|
| `src/pipeline/state.py` | Add `detected_tables: list[dict]` field |
| `src/agents/orchestrator.py` | Add optional `NODE_TABLE_DETECT` between `components` and `extract` |

---

### WK-3: Phase 2C — Schema Wizard

**New files:**

| File | Purpose |
|---|---|
| `src/agents/schema_proposal.py` | `SchemaProposalAgent(BaseAgent)` — propose/refine/save |
| `src/api/routes/schemas.py` | REST endpoints: suggest, refine, save, list, versions |

---

### WK-4: Phase 3A — Confidence Calibration

**New file:** `src/validation/calibration.py` — `ConfidenceCalibrator` (Platt + isotonic).
**Modify:** `src/agents/validator.py` — call calibrator after raw confidence.
**Modify:** `src/pipeline/state.py` — add `raw_confidence`, `calibrated_confidence`, `calibration_method`.
**Dep:** `scikit-learn>=1.5.0`

---

### WK-5: Phase 3B — Incremental Learning

**New file:** `src/memory/dynamic_prompt.py` — `DynamicPromptEnhancer`.
**Modify:** `src/agents/extractor.py` — call `enhance_prompt()` before VLM calls.
**Modify:** `src/memory/correction_tracker.py` — store corrections in Mem0.

---

### WK-6: Phase 3C — Multi-Model Strategy

**New file:** `src/client/model_router.py` — `ModelRouter`, `ModelTask` enum.
**Modify:** `src/agents/base.py` — optional `ModelRouter` in `__init__`, route in `send_vision_request()`.
**Modify:** `src/config/settings.py` — `ModelRoutingSettings`.

---

### WK-7: Phase 4A — Enhanced Exports

**Modify:** `src/export/excel_exporter.py`, `src/export/json_exporter.py`, `src/export/consolidated_export.py`
**New file:** `src/export/markdown_exporter.py` — dedicated `MarkdownExporter` class.

---

### WK-8: Phase 4B — Finance Schemas

**New files:** `src/schemas/invoice.py`, `src/schemas/w2.py`, `src/schemas/form_1099.py`, `src/schemas/bank_statement.py`
**Modify:** `src/schemas/field_types.py` — add `EIN`, `ROUTING_NUMBER`, `BANK_ACCOUNT`.

---

### WK-9: Phase 5A — Evaluation & Benchmarking

**New files:** `src/evaluation/{__init__, benchmark, golden_dataset, metrics, ab_testing, regression}.py`, `src/api/routes/evaluation.py`
**Modify:** `src/monitoring/metrics.py` — add benchmark gauges.

---

### WK-10: Phase 5B — Webhook Delivery

**New files:** `src/api/routes/webhooks.py`, `src/queue/webhook_store.py`
**Modify:** `src/queue/webhook.py`, `src/api/app.py` — register routes.

---

## PART 2: Comprehensive Testing Plan

### Current Test Landscape

| Category | Files | Tests | Notes |
|---|---|---|---|
| Unit | 15 files | ~513 | Core functionality |
| Integration | 3 files | ~76 | Pipeline flows |
| E2E | 1 file | ~9 | Complete auth flow |
| Accuracy | 1 file | ~19 | Golden dataset |
| Security | 1 file | ~24 | Auth security |
| Root-level | 4 files | ~149 | Validation, optimization |
| **Total** | **25 files** | **~790** | |

### Test Coverage Gap Analysis

**UNTESTED modules** (53% of src/ has no direct tests):

| Priority | Module | Risk | Why |
|---|---|---|---|
| P0-Critical | `src/agents/orchestrator.py` | HIGH | Entire workflow routing — 0 unit tests |
| P0-Critical | `src/agents/analyzer.py` | HIGH | Document classification drives all downstream |
| P0-Critical | `src/client/lm_client.py` | HIGH | Every VLM call passes through this |
| P0-Critical | `src/pipeline/runner.py` | HIGH | Pipeline entry point |
| P1-High | `src/agents/layout_agent.py` | MEDIUM | VLM-first layout analysis |
| P1-High | `src/agents/component_detector.py` | MEDIUM | Table/form detection |
| P1-High | `src/agents/schema_generator.py` | MEDIUM | Zero-shot schema creation |
| P1-High | `src/validation/medical_codes.py` | MEDIUM | CPT/ICD-10/NPI validation |
| P1-High | `src/preprocessing/pdf_processor.py` | MEDIUM | Core PDF processing |
| P2-Medium | `src/utils/date_utils.py` | LOW | Date parsing helpers |
| P2-Medium | `src/utils/string_utils.py` | LOW | String normalization |
| P2-Medium | `src/utils/hash_utils.py` | LOW | Hashing helpers |
| P2-Medium | `src/utils/file_utils.py` | LOW | File operations |
| P2-Medium | `src/memory/context_manager.py` | MEDIUM | Extraction context |
| P2-Medium | `src/memory/correction_tracker.py` | MEDIUM | Learning from corrections |
| P2-Medium | `src/memory/mem0_client.py` | MEDIUM | Memory persistence |
| P2-Medium | `src/validation/human_review.py` | LOW | Human review queue |
| P2-Medium | `src/export/consolidated_export.py` | MEDIUM | Multi-format export |
| P3-Low | `src/preprocessing/batch_manager.py` | LOW | Batch coordination |
| P3-Low | `src/api/middleware.py` | LOW | HTTP middleware |
| P3-Low | `src/api/routes/dashboard.py` | LOW | Dashboard API |
| P3-Low | `src/api/routes/documents.py` | LOW | Document upload API |
| P3-Low | `src/prompts/*.py` | LOW | Prompt templates (5 files) |
| P3-Low | `src/schemas/{cms1500,eob,superbill,ub04,generic_fallback}.py` | LOW | Schema definitions |
| P3-Low | `src/client/{connection_manager,health_monitor}.py` | LOW | Client utilities |

---

### TESTING STRATEGY: 7 Test Types

---

#### TYPE 1: Unit Tests (per-module isolation)

Each new phase gets a dedicated test file. Additionally, fill coverage gaps for existing untested modules.

**New feature test files (one per implementation phase):**

| Test File | Module Under Test | Est. Tests |
|---|---|---|
| `tests/unit/test_table_detector.py` | Phase 2B: `table_detector.py`, `table_types.py` | ~30 |
| `tests/unit/test_schema_proposal.py` | Phase 2C: `schema_proposal.py` | ~20 |
| `tests/unit/test_confidence_calibration.py` | Phase 3A: `calibration.py` | ~25 |
| `tests/unit/test_dynamic_prompt.py` | Phase 3B: `dynamic_prompt.py` | ~20 |
| `tests/unit/test_model_router.py` | Phase 3C: `model_router.py` | ~20 |
| `tests/unit/test_enhanced_exports.py` | Phase 4A: enhanced exporters | ~30 |
| `tests/unit/test_finance_schemas.py` | Phase 4B: invoice, W-2, 1099, bank statement | ~25 |
| `tests/unit/test_benchmark.py` | Phase 5A: `benchmark.py`, `metrics.py`, `golden_dataset.py` | ~30 |
| `tests/unit/test_webhook_store.py` | Phase 5B: `webhook_store.py` | ~15 |

**Coverage gap test files (existing untested modules):**

| Test File | Module(s) Under Test | Est. Tests |
|---|---|---|
| `tests/unit/test_orchestrator.py` | `src/agents/orchestrator.py` | ~35 |
| `tests/unit/test_analyzer_agent.py` | `src/agents/analyzer.py` | ~25 |
| `tests/unit/test_lm_client.py` | `src/client/lm_client.py` | ~20 |
| `tests/unit/test_pipeline_runner.py` | `src/pipeline/runner.py` | ~20 |
| `tests/unit/test_layout_agent.py` | `src/agents/layout_agent.py` | ~20 |
| `tests/unit/test_component_detector.py` | `src/agents/component_detector.py` | ~20 |
| `tests/unit/test_schema_generator.py` | `src/agents/schema_generator.py` | ~15 |
| `tests/unit/test_medical_codes.py` | `src/validation/medical_codes.py` | ~25 |
| `tests/unit/test_pdf_processor.py` | `src/preprocessing/pdf_processor.py` | ~15 |
| `tests/unit/test_utils.py` | `src/utils/{date,string,hash,file}_utils.py` | ~40 |
| `tests/unit/test_memory.py` | `src/memory/{context_manager,correction_tracker}.py` | ~20 |
| `tests/unit/test_human_review.py` | `src/validation/human_review.py` | ~10 |

**Unit test approach for each file:**
- Mock all external dependencies (VLM client, file I/O, databases)
- Test every public method with: happy path, edge cases, error paths
- Use `pytest.mark.parametrize` for combinatorial input coverage
- Assert specific return types, state mutations, and exception types

---

#### TYPE 2: Integration Tests (multi-module flows)

Test how modules work together without mocking internal boundaries.

**New integration test files:**

| Test File | What It Tests | Est. Tests |
|---|---|---|
| `tests/integration/test_splitter_pipeline.py` | Splitter -> Orchestrator -> Extract per-segment | ~10 |
| `tests/integration/test_table_extraction_e2e.py` | TableDetector -> Extractor with table-aware prompts | ~10 |
| `tests/integration/test_schema_wizard_flow.py` | Propose -> Refine -> Save -> Extract with new schema | ~8 |
| `tests/integration/test_export_e2e.py` | Extract -> Validate -> Export (JSON/XLSX/MD) with bbox | ~12 |
| `tests/integration/test_multi_format_pipeline.py` | FileFactory -> Processor -> Extract for each format | ~10 |
| `tests/integration/test_confidence_pipeline.py` | Extract -> Validate -> Calibrate -> Route decision | ~8 |
| `tests/integration/test_correction_learning.py` | Extract -> Correct -> Re-extract (prompt improved) | ~6 |
| `tests/integration/test_multi_model_routing.py` | ModelRouter -> different agents use different models | ~8 |

**Integration test approach:**
- Use mock VLM client returning realistic JSON responses
- Test full state flow through 2-4 agent nodes
- Verify state fields are correctly set at each stage
- Test error propagation and fallback paths

---

#### TYPE 3: Regression Tests (golden dataset guard rails)

Prevent accuracy regressions when code changes.

**New regression test files:**

| Test File | Purpose | Est. Tests |
|---|---|---|
| `tests/regression/test_schema_regression.py` | Schema changes don't break existing extractions | ~10 |
| `tests/regression/test_extraction_regression.py` | Known document -> known output field-by-field | ~15 |
| `tests/regression/test_confidence_regression.py` | Confidence scores don't degrade vs. baseline | ~8 |
| `tests/regression/golden_datasets.py` | Golden dataset manager + sample data loading | (utility) |

**Regression test approach:**
- Store expected outputs as JSON fixtures in `tests/regression/fixtures/`
- Each test loads a fixture, runs extraction (with mocked VLM giving deterministic responses), compares field-by-field
- Track per-field F1 score and flag if any drops below baseline
- Run on every PR via `pytest tests/regression/ -m regression`

---

#### TYPE 4: Property-Based Tests (Hypothesis)

Use `hypothesis` (already installed) to discover edge cases automatically.

**New property test files:**

| Test File | Properties Tested | Est. Tests |
|---|---|---|
| `tests/property/test_bbox_properties.py` | BoundingBoxCoords: clamping always in [0,1], from_normalized roundtrip, pixel computation monotonic | ~10 |
| `tests/property/test_state_properties.py` | update_state: idempotent with same input, never loses keys, deep copy isolation | ~8 |
| `tests/property/test_schema_properties.py` | SchemaVersion: hash deterministic, diff symmetric, migration roundtrip | ~8 |
| `tests/property/test_segment_properties.py` | DocumentSegment: page_count = end - start + 1, segments cover all pages, no overlaps | ~6 |
| `tests/property/test_confidence_properties.py` | Calibrated confidence in [0,1], monotonically related to raw | ~5 |
| `tests/property/test_field_types_properties.py` | FieldType validation: valid values accepted, invalid rejected for all types | ~8 |

**Property test approach:**
```python
from hypothesis import given, strategies as st

@given(x=st.floats(0, 1), y=st.floats(0, 1), w=st.floats(0, 1), h=st.floats(0, 1))
def test_bbox_always_clamped(x, y, w, h):
    bbox = BoundingBoxCoords.from_normalized(x, y, w, h)
    assert 0 <= bbox.x <= 1
    assert 0 <= bbox.x + bbox.width <= 1.0001  # float tolerance
```

---

#### TYPE 5: Security Tests (HIPAA + OWASP)

Ensure the system handles PHI safely and resists common attacks.

**Enhance existing + new security test files:**

| Test File | What It Tests | Est. Tests |
|---|---|---|
| `tests/security/test_phi_handling.py` | PHI fields encrypted at rest, never in logs, cleaned on export | ~15 |
| `tests/security/test_input_validation.py` | File upload: path traversal, malicious filenames, oversized files, zip bombs | ~12 |
| `tests/security/test_api_security.py` | Rate limiting, CORS, auth bypass attempts, SQL injection in search | ~15 |
| `tests/security/test_webhook_security.py` | HMAC signature verification, SSRF prevention, payload sanitization | ~10 |

**Security test approach:**
- Test that PHI fields (patient_name, DOB, SSN) are encrypted before storage
- Verify audit logs capture all access to sensitive data
- Test path traversal: `../../etc/passwd` as filename
- Test oversized uploads, corrupted files, binary injection
- Verify webhook URLs can't target internal services (SSRF)

---

#### TYPE 6: Performance / Load Tests

Ensure the system handles real-world document volumes.

**New performance test files:**

| Test File | What It Tests | Est. Tests |
|---|---|---|
| `tests/performance/test_throughput.py` | Pages/second for single-page, 10-page, 100-page documents | ~6 |
| `tests/performance/test_memory.py` | Memory usage stays bounded: 100-page PDF < 2GB RSS | ~4 |
| `tests/performance/test_concurrent.py` | 5 concurrent extractions don't deadlock or corrupt state | ~5 |
| `tests/performance/test_batch_splitting.py` | 200-page multi-doc split completes in < 60s (mocked VLM) | ~3 |

**Performance test approach:**
- Mark with `@pytest.mark.slow` -- excluded from normal CI
- Use `pytest-benchmark` or manual `time.monotonic()` measurements
- Assert upper bounds on time and memory
- Test concurrent access with `concurrent.futures.ThreadPoolExecutor`

---

#### TYPE 7: Contract / Snapshot Tests

Lock down API response shapes and export format structures.

**New contract test files:**

| Test File | What It Tests | Est. Tests |
|---|---|---|
| `tests/contract/test_api_contracts.py` | All API endpoints return expected response shapes (Pydantic models) | ~20 |
| `tests/contract/test_export_contracts.py` | JSON export has required keys, XLSX has expected sheets/columns, MD has expected sections | ~15 |
| `tests/contract/test_state_contracts.py` | ExtractionState always has required keys after each agent node | ~10 |

**Contract test approach:**
- Define expected response schemas as Pydantic models or JSON Schema
- After each API call, validate response against schema
- For exports, verify structural invariants (not content)
- Snapshot test: serialize output, compare to stored snapshot, fail on unexpected changes

---

### Complete Test File Map (after all work)

```
tests/
  conftest.py                            # Shared fixtures (existing)
  __init__.py

  unit/                                  # ~40 files, ~950+ tests
    # === Existing (15 files, ~513 tests) ===
    test_settings.py
    test_export.py
    test_api.py
    test_queue.py
    test_auth_api.py
    test_security.py
    test_monitoring.py
    test_visual_grounding.py             # Phase 1A (55 tests)
    test_file_factory.py                 # Phase 1B (33 tests)
    test_schema_versioning.py            # Phase 1C (18 tests)
    test_splitter_agent.py              # Phase 2A (38 tests)
    test_multi_record_phase{1,2,3,4}.py  # Multi-record (99 tests)

    # === New feature tests (9 files, ~215 tests) ===
    test_table_detector.py               # Phase 2B
    test_schema_proposal.py              # Phase 2C
    test_confidence_calibration.py       # Phase 3A
    test_dynamic_prompt.py               # Phase 3B
    test_model_router.py                 # Phase 3C
    test_enhanced_exports.py             # Phase 4A
    test_finance_schemas.py              # Phase 4B
    test_benchmark.py                    # Phase 5A
    test_webhook_store.py                # Phase 5B

    # === Coverage gap tests (12 files, ~265 tests) ===
    test_orchestrator.py                 # P0: Workflow routing
    test_analyzer_agent.py               # P0: Document classification
    test_lm_client.py                    # P0: VLM client
    test_pipeline_runner.py              # P0: Pipeline entry
    test_layout_agent.py                 # P1: Layout analysis
    test_component_detector.py           # P1: Component detection
    test_schema_generator.py             # P1: Schema generation
    test_medical_codes.py                # P1: Medical code validation
    test_pdf_processor.py                # P1: PDF processing
    test_utils.py                        # P2: Utility functions
    test_memory.py                       # P2: Memory/context
    test_human_review.py                 # P2: Human review queue

  integration/                           # ~11 files, ~150 tests
    # === Existing (3 files) ===
    test_phase5_integration.py
    test_e2e_pipeline.py
    test_auth_integration.py

    # === New (8 files) ===
    test_splitter_pipeline.py
    test_table_extraction_e2e.py
    test_schema_wizard_flow.py
    test_export_e2e.py
    test_multi_format_pipeline.py
    test_confidence_pipeline.py
    test_correction_learning.py
    test_multi_model_routing.py

  regression/                            # 4 files, ~33 tests
    __init__.py
    test_schema_regression.py
    test_extraction_regression.py
    test_confidence_regression.py
    golden_datasets.py                   # Fixture loader
    fixtures/                            # Golden dataset JSON files
      cms1500_sample.json
      eob_sample.json
      invoice_sample.json

  property/                              # 6 files, ~45 tests
    __init__.py
    test_bbox_properties.py
    test_state_properties.py
    test_schema_properties.py
    test_segment_properties.py
    test_confidence_properties.py
    test_field_types_properties.py

  security/                              # 5 files, ~76 tests
    # === Existing (1 file) ===
    test_auth_security.py

    # === New (4 files) ===
    test_phi_handling.py
    test_input_validation.py
    test_api_security.py
    test_webhook_security.py

  performance/                           # 4 files, ~18 tests
    __init__.py
    test_throughput.py
    test_memory.py
    test_concurrent.py
    test_batch_splitting.py

  contract/                              # 3 files, ~45 tests
    __init__.py
    test_api_contracts.py
    test_export_contracts.py
    test_state_contracts.py

  e2e/                                   # Existing (1 file)
    test_complete_auth_flow.py

  accuracy/                              # Existing (1 file)
    test_extraction_accuracy.py
```

---

### Test Execution Commands

```bash
# All tests (fast, no slow/perf)
pytest tests/ -m "not slow" -v --tb=short

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Regression guard (run before every merge)
pytest tests/regression/ -m regression -v

# Property-based (may take longer due to hypothesis)
pytest tests/property/ -v --hypothesis-seed=0

# Security audit
pytest tests/security/ -v

# Performance benchmarks (slow)
pytest tests/performance/ -m slow -v --timeout=120

# Contract/snapshot tests
pytest tests/contract/ -v

# Full coverage report
pytest tests/ --cov=src --cov-report=html --cov-branch

# Specific phase verification
pytest tests/unit/test_table_detector.py tests/integration/test_table_extraction_e2e.py -v
```

---

### Pytest Markers to Register (in `pyproject.toml`)

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "accuracy: marks tests as accuracy/golden dataset tests",
    "regression: marks tests as regression guard tests",
    "security: marks tests as security tests",
    "property: marks tests as property-based (hypothesis) tests",
    "performance: marks tests as performance/load tests",
    "contract: marks tests as API/export contract tests",
]
```

---

### Test Coverage Targets

| Category | Current | Target | Delta |
|---|---|---|---|
| **Line coverage** | ~47% | 80%+ | +33% |
| **Branch coverage** | ~35% | 70%+ | +35% |
| **Module coverage** | 45/95 modules | 85/95 modules | +40 modules |
| **Total test count** | ~790 | ~1,580+ | +790 tests |

---

### Implementation Order (Interleaved: Feature + Tests)

Each work item follows: **implement -> unit test -> integration test -> regression check**.

```
WK-1:  Finish 2A + Run regression suite
WK-2:  Phase 2B (Table Detector) + test_table_detector.py + test_table_extraction_e2e.py
WK-3:  Phase 2C (Schema Wizard) + test_schema_proposal.py + test_schema_wizard_flow.py
WK-4:  Coverage gaps: test_orchestrator.py + test_analyzer_agent.py + test_lm_client.py
WK-5:  Phase 3A (Calibration) + test_confidence_calibration.py + test_confidence_pipeline.py
WK-6:  Phase 3B (Learning) + test_dynamic_prompt.py + test_correction_learning.py
WK-7:  Phase 3C (Multi-Model) + test_model_router.py + test_multi_model_routing.py
WK-8:  Coverage gaps: test_utils.py + test_medical_codes.py + test_pdf_processor.py + test_memory.py
WK-9:  Phase 4A (Exports) + test_enhanced_exports.py + test_export_e2e.py
WK-10: Phase 4B (Finance) + test_finance_schemas.py
WK-11: Property tests: tests/property/*.py (all 6 files)
WK-12: Phase 5A (Eval) + test_benchmark.py + tests/regression/*.py
WK-13: Phase 5B (Webhooks) + test_webhook_store.py
WK-14: Security tests: tests/security/*.py (4 new files)
WK-15: Performance tests + Contract tests + final coverage report
WK-16: Coverage gap agents: test_layout_agent + test_component_detector + test_schema_generator
WK-17: Remaining gaps: test_pipeline_runner + test_human_review + test_multi_format_pipeline
```

---

### Verification Checklist (per work item)

1. `pytest tests/unit/test_{feature}.py -v` -- all new tests pass
2. `pytest tests/ -m "not slow" --tb=short` -- no regressions
3. `pytest tests/ --cov=src --cov-report=term-missing` -- coverage increased
4. New test file registered in correct directory with `__init__.py`
5. New pytest markers registered in `pyproject.toml` if used
