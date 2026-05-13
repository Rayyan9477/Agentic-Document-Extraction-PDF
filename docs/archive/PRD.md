# Product Requirements Document (PRD)

**Project Name:** Local Agentic Medical Document Extraction System
**Version:** 2.0
**Owner:** Rayyan Ahmed
**Last Updated:** November 2025

---

## Table of Contents

1. [Objective](#1-objective)
2. [Key Specifications](#2-key-specifications)
3. [System Architecture](#3-system-architecture)
4. [4-Agent Architecture](#4-4-agent-architecture)
5. [3-Layer Anti-Hallucination System](#5-3-layer-anti-hallucination-system)
6. [Context Management with Mem0](#6-context-management-with-mem0)
7. [Core Features](#7-core-features)
8. [Technology Stack](#8-technology-stack)
9. [Directory Structure](#9-directory-structure)
10. [Implementation Phases](#10-implementation-phases)
11. [API Specification](#11-api-specification)
12. [Success Metrics](#12-success-metrics)
13. [Compliance & Security](#13-compliance--security)
14. [Risk Management](#14-risk-management)

---

## 1. Objective

Build a **production-ready, HIPAA-compliant document extraction system** using **local Vision Language Models (VLM)** with a **4-agent architecture** powered by **LangChain** and **LangGraph** for custom documents and zero shot solutions(zero shot solution so it can work with wider range of projects) and  medical superbills.

### Core Goals

* Extract **patient data, CPT codes, DX codes, insurance details, provider info**, and **user-defined fields**
* Process documents using **100% local AI** (no cloud API dependencies)
* Implement **3-layer anti-hallucination validation** for accuracy
* Handle **handwriting, tables, and mixed layouts** dynamically
* Support **multi-patient extraction from a single PDF page**
* Provide **per-field confidence scores** with extraction metadata
* Enable **zero-shot custom schema definition** without retraining
* Export results into **well-formatted Excel sheets and JSON**
* Provide a **Streamlit-based UI** for easy interaction

---

## 2. Key Specifications

| Attribute | Value |
|-----------|-------|
| **VLM Model** | Qwen3-VL 8B |
| **Model Backend** | LM Studio (Local) |
| **Agent Framework** | LangGraph 1.x + LangChain 1.x |
| **Agents** | 4 Specialized |
| **Validation** | 3-Layer Anti-Hallucination |
| **Compliance** | HIPAA Ready |
| **VLM Calls/Page** | 3-4 |
| **Processing Time** | 15-25 sec/page |
| **Timeline** | 12 Weeks |

### Performance Comparison

| Metric | Old (Cloud API) | New (Local VLM + Agents) |
|--------|-----------------|--------------------------|
| Data Privacy | Cloud-dependent | 100% Local |
| VLM Calls/Page | 6-8 | 3-4 |
| Processing Time | 45-60 sec | 15-25 sec |
| Validation Layers | Partial | 3-Layer Complete |
| Agent Architecture | None | 4-Agent LangGraph |
| HIPAA Compliance | Requires BAA | Built-in |
| Coordination Overhead | High | Low |
| State Management | Manual | LangGraph Checkpointing |

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │   REST API    │    │   Batch Job   │    │   Streamlit   │                │
│  │   (FastAPI)   │    │   (Celery)    │    │      UI       │                │
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘                │
└──────────┼────────────────────┼────────────────────┼────────────────────────┘
           │                    │                    │
           └────────────────────┼────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING LAYER                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      PDF Processor (PyMuPDF)                          │  │
│  │                                                                       │  │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐ │  │
│  │   │ PDF Validate│──▶│ Page Extract│──▶│   Enhance   │──▶│  Output  │ │  │
│  │   │ & Metadata  │   │   300 DPI   │   │   (OpenCV)  │   │  Images  │ │  │
│  │   └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT LAYER (LangGraph State Machine)                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ORCHESTRATOR AGENT                              ││
│  │              LangGraph StateGraph + Checkpointing                       ││
│  │                         (0 VLM Calls)                                   ││
│  └─────────────────────────────────┬───────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          ANALYZER AGENT                                 ││
│  │         Document Classification + Schema Selection                      ││
│  │                        (1 VLM Call/Doc)                                 ││
│  └─────────────────────────────────┬───────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         EXTRACTOR AGENT                                 ││
│  │           Dual-Pass Extraction + Confidence Scoring                     ││
│  │                       (2 VLM Calls/Page)                                ││
│  └─────────────────────────────────┬───────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         VALIDATOR AGENT                                 ││
│  │        Hallucination Detection + Cross-Field Validation                 ││
│  │                      (0-1 VLM Calls/Doc)                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VLM BACKEND                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  LM Studio Server (localhost:1234)                    │  │
│  │                                                                       │  │
│  │   Model: Qwen3-VL 8B (Q4_K_M)    │    Context: 32K Tokens            │  │
│  │   VRAM: ~6GB                      │    API: OpenAI Compatible         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                      │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │     JSON      │    │     Excel     │    │   Database    │                │
│  │  (Pydantic)   │    │   (openpyxl)  │    │   (SQLite)    │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram

```
                              ┌──────────────┐
                              │  PDF Upload  │
                              └──────┬───────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • PDF validation and metadata extraction                           │   │
│  │  • Page-to-image conversion at 300 DPI (PyMuPDF)                    │   │
│  │  • Image enhancement: deskew, denoise, contrast (OpenCV)            │   │
│  │  • Memory-efficient streaming for large documents                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ORCHESTRATOR (LangGraph)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Initialize ExtractionState with all required fields              │   │
│  │  • Create checkpoint for recovery                                    │   │
│  │  • Manage state transitions between agents                           │   │
│  │  • Handle errors and retry logic                                     │   │
│  │  • VLM Calls: 0                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: ANALYZER                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Classify document type (CMS-1500, UB-04, EOB, Superbill)         │   │
│  │  • Detect document structure (tables, forms, handwriting)            │   │
│  │  • Analyze page relationships for multi-page documents               │   │
│  │  • Select appropriate extraction schema                              │   │
│  │  • VLM Calls: 1 per document                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: EXTRACTOR (Dual-Pass)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────────┐                ┌─────────────────┐              │   │
│  │  │     PASS 1      │                │     PASS 2      │              │   │
│  │  │                 │                │                 │              │   │
│  │  │ Standard        │                │ Verification    │              │   │
│  │  │ Extraction      │                │ Extraction      │              │   │
│  │  │                 │                │                 │              │   │
│  │  │ Focus:          │                │ Focus:          │              │   │
│  │  │ Completeness    │                │ Accuracy        │              │   │
│  │  │                 │                │ (Different      │              │   │
│  │  │                 │                │  Prompt Style)  │              │   │
│  │  └────────┬────────┘                └────────┬────────┘              │   │
│  │           │                                  │                        │   │
│  │           └──────────────┬───────────────────┘                        │   │
│  │                          │                                            │   │
│  │                          ▼                                            │   │
│  │           ┌─────────────────────────────┐                            │   │
│  │           │   FIELD-BY-FIELD COMPARE    │                            │   │
│  │           │                             │                            │   │
│  │           │  • Match = High Confidence  │                            │   │
│  │           │  • Mismatch = Flag Review   │                            │   │
│  │           │  • Generate Confidence Score│                            │   │
│  │           └─────────────────────────────┘                            │   │
│  │                                                                       │   │
│  │  • VLM Calls: 2 per page                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: VALIDATOR                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Schema validation against document type rules                     │   │
│  │  • Hallucination pattern detection                                    │   │
│  │  • Medical code validation (CPT, ICD-10, NPI with Luhn check)        │   │
│  │  • Cross-field rule validation (date ordering, math checks)          │   │
│  │  • Cross-page data merging for multi-page documents                   │   │
│  │  • Final confidence score calculation                                 │   │
│  │  • VLM Calls: 0-1 per document                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: CONFIDENCE ROUTING                                                 │
│                                                                             │
│              ┌─────────────────────────────────────┐                       │
│              │       CONFIDENCE SCORE CHECK        │                       │
│              └─────────────────┬───────────────────┘                       │
│                                │                                            │
│         ┌──────────────────────┼──────────────────────┐                    │
│         │                      │                      │                    │
│         ▼                      ▼                      ▼                    │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│  │   ≥ 0.85    │       │  0.50-0.84  │       │   < 0.50    │              │
│  │             │       │             │       │             │              │
│  │ AUTO-ACCEPT │       │   RETRY     │       │   HUMAN     │              │
│  │             │       │  (max 2x)   │       │   REVIEW    │              │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘              │
│         │                      │                      │                    │
│         ▼                      ▼                      ▼                    │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│  │ FORMAT OUT  │       │  RE-EXTRACT │       │ REVIEW QUEUE│              │
│  └──────┬──────┘       └─────────────┘       └──────┬──────┘              │
│         │                                           │                      │
│         └─────────────────────┬─────────────────────┘                      │
│                               ▼                                            │
│                    ┌─────────────────────┐                                 │
│                    │   JSON + Excel Out  │                                 │
│                    │   + Audit Log       │                                 │
│                    └─────────────────────┘                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 4-Agent Architecture

### 4.1 Agent Overview

| Agent | Role | VLM Calls | Key Functions |
|-------|------|-----------|---------------|
| **Orchestrator** | State Machine Controller | 0 | Workflow control, error handling, checkpointing, retry logic, state transitions |
| **Analyzer** | Document Understanding | 1/doc | Classification, structure detection, page relationships, schema selection |
| **Extractor** | Data Extraction | 2/page | Schema-driven extraction, dual-pass verification, confidence scoring, visual grounding |
| **Validator** | Quality Assurance | 0-1/doc | Schema validation, hallucination detection, cross-page merging, output formatting |

### 4.2 LangGraph State Machine Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph Workflow                         │
│                                                                              │
│   ┌─────────────┐                                                           │
│   │    START    │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │ PREPROCESS  │  ← PDF → Images + Enhancement                            │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │   ANALYZE   │  ← Classify document, select schema (1 VLM call)         │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │   EXTRACT   │  ← Dual-pass extraction (2 VLM calls/page)               │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐                                                           │
│   │  VALIDATE   │  ← Check quality, detect hallucinations                  │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          │  ┌────────────────────────────────────────────────────────────┐  │
│          │  │              CONDITIONAL ROUTING                           │  │
│          │  │                                                            │  │
│          │  │  confidence ≥ 0.85  ──────────────────▶  FORMAT_OUTPUT    │  │
│          │  │                                                            │  │
│          │  │  confidence 0.50-0.84 AND retry < 2  ──▶  EXTRACT (retry) │  │
│          │  │                                                            │  │
│          │  │  confidence < 0.50  ──────────────────▶  HUMAN_REVIEW     │  │
│          │  │                                                            │  │
│          │  │  error  ──────────────────────────────▶  ERROR_HANDLER    │  │
│          │  └────────────────────────────────────────────────────────────┘  │
│          │                                                                   │
│          ├────────────▶ FORMAT_OUTPUT ────────▶ END                        │
│          │                                                                   │
│          ├────────────▶ HUMAN_REVIEW ─────────▶ END                        │
│          │                                                                   │
│          └────────────▶ ERROR_HANDLER ────────▶ END                        │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   MemorySaver checkpointing enabled at each state transition                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 ExtractionState Definition

The LangGraph workflow maintains a TypedDict state with the following fields:

| Field Group | Fields | Description |
|-------------|--------|-------------|
| **Input** | pdf_path, images, custom_schema | Source PDF path, converted page images, and optional user-defined schema |
| **Analysis** | doc_type, schema, page_relationships | Document classification and schema selection |
| **Extraction** | extractions, pass1_results, pass2_results, field_metadata | Dual-pass extraction results with per-field confidence |
| **Validation** | validation_result, confidence_scores, hallucination_flags | Quality checks and scoring |
| **Control** | status, errors, retry_count, checkpoint_id | Workflow state management |

### 4.4 Per-Field Confidence & Metadata Tracking

**Field-Level Confidence Scoring**


**Confidence Calculation Logic:**

| Scenario | Confidence Score | Reasoning |
|----------|------------------|------------|
| Dual-pass match + validation passed | 0.90 - 1.00 | Highest confidence |
| Dual-pass match + validation warning | 0.75 - 0.89 | Good extraction, minor issues |
| Dual-pass mismatch + validator chose | 0.50 - 0.74 | Uncertain, needs review |
| Single-pass only + validation passed | 0.60 - 0.75 | Moderate confidence |
| Single-pass only + validation failed | 0.00 - 0.49 | Low confidence, human review |


### 4.5 Zero-Shot Custom Schema Definition

**Flexible Schema System**

Users can define custom extraction schemas without retraining the model:


**Schema Definition Guidelines:**

1. **Field Descriptions**: Detailed descriptions help VLM understand what to extract
2. **Type Hints**: Use proper Python types (str, int, float, List, Optional, Dict)
3. **Default Values**: Specify defaults for optional fields
4. **Validation**: Pydantic automatically validates extracted data against schema
5. **Nested Structures**: Support for complex hierarchical data

**Predefined Schema Library + Custom Schemas:**


**Schema Selection Logic:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEMA SELECTION WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. User provides document + optional custom_schema      │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. Check if custom_schema provided                      │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│         ┌─────────────────┴─────────────────┐                  │
│         │                                   │                  │
│         ▼                                   ▼                  │
│  ┌─────────────┐                   ┌─────────────────┐        │
│  │   YES       │                   │      NO         │        │
│  │             │                   │                 │        │
│  │ Use Custom  │                   │ Analyzer Agent  │        │
│  │   Schema    │                   │ Auto-Detects    │        │
│  └──────┬──────┘                   │ Document Type   │        │
│         │                          └────────┬────────┘        │
│         │                                   │                  │
│         │                          ┌────────▼────────┐         │
│         │                          │ Select Built-in │         │
│         │                          │ Schema (CMS1500,│         │
│         │                          │ Superbill, etc.)│         │
│         │                          └────────┬────────┘         │
│         │                                   │                  │
│         └───────────────┬───────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. Extractor uses schema for guided extraction          │  │
│  │     • Field descriptions → VLM prompts                   │  │
│  │     • Type hints → Output validation                     │  │
│  │     • Defaults → Handle missing fields                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of Zero-Shot Schema Definition:**

✅ **No Retraining Required**: Add new document types without model fine-tuning  
✅ **Domain Flexibility**: Works across medical, legal, financial documents  
✅ **Quick Adaptation**: Deploy new schemas in minutes, not weeks  
✅ **Type Safety**: Pydantic validation ensures data quality  
✅ **Backward Compatible**: Existing built-in schemas still work  
✅ **Version Control**: Schemas are code, can be versioned with Git  

### 4.6 Agent Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR AGENT                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RESPONSIBILITIES:                                                      ││
│  │  • Build and compile LangGraph StateGraph workflow                      ││
│  │  • Initialize ExtractionState for each document                         ││
│  │  • Handle custom schema injection if provided by user                   ││
│  │  • Manage state transitions between agents                              ││
│  │  • Handle checkpointing with MemorySaver                                ││
│  │  • Implement retry logic for failed extractions                         ││
│  │  • Route to human review when confidence is low                         ││
│  │  • Resume interrupted workflows from checkpoints                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  VLM CALLS: 0 (Pure Python orchestration)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                ANALYZER AGENT                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RESPONSIBILITIES:                                                      ││
│  │  • Classify document type (CMS-1500, UB-04, EOB, Superbill, Other)     ││
│  │  • Detect document structure (tables, forms, signatures, handwriting)  ││
│  │  • Analyze page relationships for multi-page documents                  ││
│  │  • Select appropriate extraction schema based on classification         ││
│  │  • Identify key sections and regions of interest                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  VLM CALLS: 1 per document (first page classification)                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               EXTRACTOR AGENT                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RESPONSIBILITIES:                                                      ││
│  │  • Perform dual-pass extraction on each page                            ││
│  │    - Pass 1: Standard extraction with completeness focus                ││
│  │    - Pass 2: Verification extraction with accuracy focus                ││
│  │  • Compare results field-by-field                                       ││
│  │  • Generate confidence scores based on agreement                        ││
│  │  • Include visual grounding (location descriptions)                     ││
│  │  • Apply grounding rules to prevent hallucinations                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  VLM CALLS: 2 per page (dual-pass)                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               VALIDATOR AGENT                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RESPONSIBILITIES:                                                      ││
│  │  • Validate extracted data against schema rules                         ││
│  │  • Detect hallucination patterns:                                       ││
│  │    - Repetitive values across fields                                    ││
│  │    - Suspiciously round numbers                                         ││
│  │    - Placeholder patterns (N/A, TBD, XXX)                              ││
│  │  • Validate medical codes (CPT, ICD-10, NPI with Luhn algorithm)       ││
│  │  • Apply cross-field validation rules                                   ││
│  │  • Merge data from multiple pages                                       ││
│  │  • Calculate final confidence scores                                    ││
│  │  • Format output for JSON/Excel export                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  VLM CALLS: 0-1 per document (optional verification)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 3-Layer Anti-Hallucination System

### 5.1 Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      3-LAYER ANTI-HALLUCINATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 1: PROMPT ENGINEERING                                          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  GROUNDING RULES:                                                     ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  1. VISUAL GROUNDING: Only extract values CLEARLY VISIBLE       │ ║  │
│  ║  │  2. NO GUESSING: If unclear, blurry, or not visible → null      │ ║  │
│  ║  │  3. NO INFERENCE: Do not calculate or infer values              │ ║  │
│  ║  │  4. NO DEFAULTS: Do not fill "typical" or "expected" values     │ ║  │
│  ║  │  5. CONFIDENCE: Include 0.0-1.0 confidence for each field       │ ║  │
│  ║  │  6. LOCATION: Describe WHERE in document value was found        │ ║  │
│  ║  │  7. UNCERTAINTY: When uncertain between values → null           │ ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                       ║  │
│  ║  FORBIDDEN ACTIONS:                                                   ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  • Making up patient names, dates, or medical codes             │ ║  │
│  ║  │  • Guessing values based on document type expectations          │ ║  │
│  ║  │  • Filling placeholder values like "John Doe" or "01/01/2000"   │ ║  │
│  ║  │  • Assuming standard formats if not clearly visible             │ ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘ ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                     │                                        │
│                                     ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 2: DUAL-PASS EXTRACTION                                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  ┌─────────────┐              ┌─────────────┐                        ║  │
│  ║  │   PASS 1    │              │   PASS 2    │                        ║  │
│  ║  │             │              │             │                        ║  │
│  ║  │  Standard   │              │ Verification│                        ║  │
│  ║  │  Extraction │              │  Extraction │                        ║  │
│  ║  │             │              │             │                        ║  │
│  ║  │  "Extract   │              │ "VERIFY:    │                        ║  │
│  ║  │   all       │              │  Carefully  │                        ║  │
│  ║  │   fields"   │              │  re-examine │                        ║  │
│  ║  │             │              │  document"  │                        ║  │
│  ║  └──────┬──────┘              └──────┬──────┘                        ║  │
│  ║         │                            │                                ║  │
│  ║         └──────────────┬─────────────┘                                ║  │
│  ║                        │                                              ║  │
│  ║                        ▼                                              ║  │
│  ║         ┌────────────────────────────────┐                           ║  │
│  ║         │    FIELD-BY-FIELD COMPARISON   │                           ║  │
│  ║         │                                │                           ║  │
│  ║         │  IF value1 == value2:          │                           ║  │
│  ║         │     → confidence = HIGH        │                           ║  │
│  ║         │     → agreement = TRUE         │                           ║  │
│  ║         │                                │                           ║  │
│  ║         │  IF value1 != value2:          │                           ║  │
│  ║         │     → confidence = LOW         │                           ║  │
│  ║         │     → mismatch = TRUE          │                           ║  │
│  ║         │     → flag for review          │                           ║  │
│  ║         └────────────────────────────────┘                           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                     │                                        │
│                                     ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  LAYER 3: PATTERN + RULE VALIDATION                                   ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  HALLUCINATION PATTERN DETECTION:                                     ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  Pattern              │  Detection Method                       │ ║  │
│  ║  │  ────────────────────│──────────────────────────────────────── │ ║  │
│  ║  │  Repetitive values    │  Compare all values for duplicates     │ ║  │
│  ║  │  Round numbers        │  Flag $1000.00, $500.00 exactly        │ ║  │
│  ║  │  Placeholder text     │  Regex: N/A, TBD, XXX, 123, test       │ ║  │
│  ║  │  Type mismatches      │  Validate against field type           │ ║  │
│  ║  │  Cross-field errors   │  Date ordering, math verification      │ ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                       ║  │
│  ║  MEDICAL CODE VALIDATION:                                             ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  Code Type   │  Format                                          │ ║  │
│  ║  │  ──────────  │  ────────────────────────────────────────────── │ ║  │
│  ║  │  CPT         │  5 digits OR 4 digits + modifier (XXXXX-XX)     │ ║  │
│  ║  │  ICD-10      │  Letter + 2 digits + optional decimal           │ ║  │
│  ║  │  NPI         │  10 digits with Luhn algorithm checksum         │ ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                       ║  │
│  ║  CROSS-FIELD RULES:                                                   ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  • date_of_service >= date_of_birth                             │ ║  │
│  ║  │  • total_charges = sum(line_item_charges)                       │ ║  │
│  ║  │  • Required field dependencies                                  │ ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘ ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Confidence Score Actions

| Score Range | Action | Description |
|-------------|--------|-------------|
| ≥ 0.95 | Auto-Accept | High confidence, proceed directly to output |
| 0.85 - 0.94 | Accept + Flag | Accept but flag for audit trail review |
| 0.70 - 0.84 | Verify | Request optional VLM verification pass |
| 0.50 - 0.69 | Re-Extract | Retry extraction with adjusted prompts (max 2x) |
| < 0.50 | Human Review | Route to human review queue |

---

## 6. Context Management with Mem0

### 6.1 Overview

The system integrates **Mem0** as the persistent memory layer to maintain context across extraction sessions, enable learning from corrections, and provide intelligent document processing based on historical patterns.

### 6.2 Why Mem0?

| Feature | Benefit |
|---------|---------|
| **Persistent Memory** | Retains context across sessions for multi-document workflows |
| **Self-Improving** | Learns from corrections to improve future extractions |
| **LangGraph Compatible** | Native integration with LangGraph agents |
| **Local Deployment** | Can run entirely locally for HIPAA compliance |
| **Vector + Graph Memory** | Combines semantic search with relationship tracking |

### 6.3 Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEM0 MEMORY ARCHITECTURE                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        MEMORY LAYER OVERVIEW                            ││
│  │                                                                          ││
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ││
│  │   │   EXTRACTION    │    │    DOCUMENT     │    │   CORRECTION    │    ││
│  │   │    CONTEXT      │    │    PATTERNS     │    │    HISTORY      │    ││
│  │   │                 │    │                 │    │                 │    ││
│  │   │ • Current doc   │    │ • Schema maps   │    │ • User fixes    │    ││
│  │   │ • Field values  │    │ • Field layouts │    │ • Error patterns│    ││
│  │   │ • Page context  │    │ • Provider info │    │ • Improvements  │    ││
│  │   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘    ││
│  │            │                      │                      │              ││
│  │            └──────────────────────┼──────────────────────┘              ││
│  │                                   │                                      ││
│  │                                   ▼                                      ││
│  │            ┌─────────────────────────────────────────┐                  ││
│  │            │            MEM0 MEMORY STORE            │                  ││
│  │            │                                         │                  ││
│  │            │  ┌─────────────┐    ┌─────────────┐    │                  ││
│  │            │  │   VECTOR    │    │    GRAPH    │    │                  ││
│  │            │  │   STORE     │    │    STORE    │    │                  ││
│  │            │  │             │    │             │    │                  ││
│  │            │  │  Semantic   │    │  Relations  │    │                  ││
│  │            │  │  Search     │    │  & Links    │    │                  ││
│  │            │  │  (Qdrant)   │    │             │    │                  ││
│  │            │  └─────────────┘    └─────────────┘    │                  ││
│  │            └─────────────────────────────────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  MEMORY OPERATIONS:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ADD      │  Store new extraction results and document patterns         ││
│  │  SEARCH   │  Retrieve relevant context for current extraction           ││
│  │  UPDATE   │  Modify memories based on user corrections                   ││
│  │  DELETE   │  Remove outdated or incorrect memories                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Memory Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MEM0 INTEGRATION WITH LANGGRAPH                         │
│                                                                              │
│   ┌─────────────┐                                                           │
│   │  NEW DOC    │                                                           │
│   │  UPLOADED   │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 1: CONTEXT RETRIEVAL                                          │   │
│   │  ─────────────────────────────                                       │   │
│   │                                                                       │   │
│   │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │   │
│   │  │   Analyzer   │────────▶│  Mem0 Search │────────▶│   Context    │ │   │
│   │  │    Agent     │         │              │         │   Injected   │ │   │
│   │  │              │         │ "Similar     │         │              │ │   │
│   │  │ "What doc    │         │  documents?" │         │ • Past docs  │ │   │
│   │  │  is this?"   │         │ "Provider    │         │ • Schemas    │ │   │
│   │  │              │         │  patterns?"  │         │ • Patterns   │ │   │
│   │  └──────────────┘         └──────────────┘         └──────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 2: CONTEXT-AWARE EXTRACTION                                    │   │
│   │  ────────────────────────────────                                    │   │
│   │                                                                       │   │
│   │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │   │
│   │  │  Extractor   │         │   Enhanced   │         │   Higher     │ │   │
│   │  │    Agent     │────────▶│   Prompts    │────────▶│   Accuracy   │ │   │
│   │  │              │         │              │         │              │ │   │
│   │  │ + Retrieved  │         │ "Based on    │         │ Fewer errors │ │   │
│   │  │   Context    │         │  similar     │         │ Better       │ │   │
│   │  │              │         │  documents"  │         │ confidence   │ │   │
│   │  └──────────────┘         └──────────────┘         └──────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 3: MEMORY STORAGE                                              │   │
│   │  ───────────────────────                                             │   │
│   │                                                                       │   │
│   │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │   │
│   │  │  Extraction  │         │   Mem0 Add   │         │   Memory     │ │   │
│   │  │   Results    │────────▶│              │────────▶│   Updated    │ │   │
│   │  │              │         │ • Document   │         │              │ │   │
│   │  │ • Fields     │         │   metadata   │         │ Future docs  │ │   │
│   │  │ • Confidence │         │ • Extraction │         │ benefit from │ │   │
│   │  │ • Patterns   │         │   patterns   │         │ this context │ │   │
│   │  └──────────────┘         └──────────────┘         └──────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 4: CORRECTION LEARNING (Optional)                              │   │
│   │  ──────────────────────────────────────                              │   │
│   │                                                                       │   │
│   │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │   │
│   │  │    Human     │         │  Mem0 Update │         │   Improved   │ │   │
│   │  │  Correction  │────────▶│              │────────▶│   Future     │ │   │
│   │  │              │         │ • Error type │         │   Accuracy   │ │   │
│   │  │ "This field  │         │ • Correct    │         │              │ │   │
│   │  │  was wrong"  │         │   value      │         │ Self-        │ │   │
│   │  │              │         │ • Context    │         │ improving    │ │   │
│   │  └──────────────┘         └──────────────┘         └──────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Memory Types

| Memory Type | Purpose | Storage | Retention |
|-------------|---------|---------|-----------|
| **Session Memory** | Current document context | In-memory | Session lifetime |
| **Document Memory** | Historical extraction results | Vector DB | Configurable |
| **Schema Memory** | Document type patterns | Graph DB | Permanent |
| **Correction Memory** | User corrections and fixes | Vector DB | Permanent |

### 6.6 Context Retrieval Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT RETRIEVAL STRATEGY                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RETRIEVAL PRIORITIES                                                   ││
│  │  ────────────────────                                                   ││
│  │                                                                          ││
│  │  1. EXACT MATCH (Highest Priority)                                      ││
│  │     └──▶ Same provider, same document type                              ││
│  │                                                                          ││
│  │  2. SIMILAR DOCUMENTS                                                    ││
│  │     └──▶ Same document type, different provider                         ││
│  │                                                                          ││
│  │  3. RELATED PATTERNS                                                     ││
│  │     └──▶ Similar field layouts, related schemas                         ││
│  │                                                                          ││
│  │  4. CORRECTION HISTORY                                                   ││
│  │     └──▶ Past mistakes and fixes for similar fields                     ││
│  │                                                                          ││
│  │  5. GENERAL CONTEXT                                                      ││
│  │     └──▶ Document type knowledge, medical code patterns                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  RETRIEVAL PARAMETERS:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  • top_k: 5 (number of relevant memories to retrieve)                   ││
│  │  • similarity_threshold: 0.7 (minimum relevance score)                  ││
│  │  • include_metadata: true (provider, date, confidence)                  ││
│  │  • filter_by_user: true (user-specific context)                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.7 Local Deployment Configuration

For HIPAA compliance, Mem0 is configured for 100% local deployment:

| Component | Local Configuration |
|-----------|---------------------|
| **Vector Store** | Faiss (self-hosted on localhost:6333) |
| **Embedding Model** | Local Sentence Transformers |
| **LLM for Memory** | LM Studio (localhost:1234) |
| **Storage** | Encrypted local filesystem |

### 6.8 Memory-Enhanced Agent State

The ExtractionState is extended with memory fields:

| Field | Type | Description |
|-------|------|-------------|
| memory_context | list[dict] | Retrieved relevant memories |
| similar_docs | list[str] | IDs of similar past documents |
| provider_patterns | dict | Known patterns for this provider |
| correction_hints | list[dict] | Past corrections for similar fields |
| session_id | str | Current session identifier |

### 6.9 Benefits of Context Management

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Higher Accuracy** | Learn from past extractions | +5-10% field accuracy |
| **Faster Processing** | Skip re-learning known patterns | -20% processing time |
| **Fewer Human Reviews** | Better confidence from context | -30% review rate |
| **Continuous Improvement** | Self-improving from corrections | Ongoing accuracy gains |
| **Provider-Specific Learning** | Adapt to different form layouts | Better multi-provider support |

---

## 7. Core Features

### 7.1 PDF Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PDF PREPROCESSING PIPELINE                           │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   VALIDATE  │───▶│   EXTRACT   │───▶│   ENHANCE   │───▶│   OUTPUT    │  │
│  │             │    │             │    │             │    │             │  │
│  │  • Check    │    │  • PyMuPDF  │    │  • OpenCV   │    │  • PNG      │  │
│  │    corrupt  │    │  • 300 DPI  │    │  • Deskew   │    │  • Base64   │  │
│  │  • Metadata │    │  • RGB      │    │  • Denoise  │    │  • Batched  │  │
│  │  • Pages    │    │  • Streaming│    │  • Contrast │    │  • Memory   │  │
│  │  • Encrypt  │    │             │    │  • CLAHE    │    │    managed  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  Processing Details:                                                         │
│  • PDF validation: Check for corruption, encryption, page count              │
│  • Page extraction: Convert at 300 DPI using PyMuPDF matrix                 │
│  • Image enhancement: OpenCV pipeline (deskew, denoise, CLAHE contrast)     │
│  • Memory management: Streaming for large documents, batch processing        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Schema Definition System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SCHEMA DEFINITION SYSTEM                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  DOCUMENT SCHEMA                                                        ││
│  │  ─────────────                                                          ││
│  │  • name: Schema name (e.g., "Medical Superbill")                       ││
│  │  • description: Schema purpose                                          ││
│  │  • document_type: Type identifier                                       ││
│  │  • fields: List of FieldDefinition                                      ││
│  │  • cross_field_rules: Validation rules between fields                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  FIELD DEFINITION                                                       ││
│  │  ────────────────                                                       ││
│  │  • name: Field name (e.g., "patient_name")                             ││
│  │  • type: FieldType (STRING, DATE, CURRENCY, CPT_CODE, ICD10_CODE, etc.)││
│  │  • required: Boolean                                                    ││
│  │  • pattern: Regex pattern for validation                                ││
│  │  • description: Human-readable description                              ││
│  │  • examples: Example values for prompting                               ││
│  │  • validation_rules: Additional validation rules                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  SUPPORTED FIELD TYPES:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  STRING    │  DATE       │  CURRENCY  │  INTEGER   │  FLOAT            ││
│  │  CPT_CODE  │  ICD10_CODE │  NPI       │  PHONE     │  SSN    │  LIST   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  PRE-BUILT SCHEMAS:                                                          │
│  • Superbill      - Medical billing superbill                               │
│  • CMS-1500       - Professional medical claim form                         │
│  • UB-04          - Institutional claim form                                │
│  • EOB            - Explanation of Benefits  
|  custom feilds for newer needs and flexible                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 LM Studio Client Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LM STUDIO CLIENT ARCHITECTURE                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  CLIENT CONFIGURATION                                                   ││
│  │  ────────────────────                                                   ││
│  │  • base_url: http://localhost:1234/v1                                   ││
│  │  • model: qwen3-vl                                                      ││
│  │  • max_tokens: 4096                                                     ││
│  │  • temperature: 0.1 (low for accuracy)                                  ││
│  │  • timeout: 120 seconds                                                 ││
│  │  • max_retries: 3                                                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  VISION REQUEST FLOW                                                    ││
│  │  ───────────────────                                                    ││
│  │                                                                          ││
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          ││
│  │  │  Image   │───▶│  Base64  │───▶│  Request │───▶│ Response │          ││
│  │  │  Bytes   │    │  Encode  │    │  + Retry │    │  + Parse │          ││
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘          ││
│  │                                                                          ││
│  │  Retry Logic (Tenacity):                                                 ││
│  │  • stop: after 3 attempts                                                ││
│  │  • wait: exponential backoff (2-30 seconds)                              ││
│  │  • retry on: APIConnectionError, APITimeoutError                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  JSON EXTRACTION                                                        ││
│  │  ────────────────                                                       ││
│  │  • Try direct JSON parse                                                 ││
│  │  • Extract from markdown code blocks                                     ││
│  │  • Regex pattern matching for JSON objects                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Technology Stack

### 8.1 Core Dependencies (November 2025)

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Agent Framework** | langchain | ≥1.0.0 | LangChain core framework |
| | langchain-core | ≥0.3.25 | Core abstractions |
| | langchain-community | ≥0.3.12 | Community integrations |
| | langgraph | ≥1.0.0 | Graph-based agent orchestration |
| | langgraph-checkpoint | ≥2.0.10 | Checkpointing for LangGraph |
| **Memory Layer** | mem0ai | ≥0.1.29 | Persistent memory for AI agents |
| | faiss | ≥1.7.2 | Vector database client |
| | sentence-transformers | ≥3.3.0 | Local embedding models |
| **VLM Client** | openai | ≥1.55.0 | OpenAI-compatible client for LM Studio |
| | tenacity | ≥9.0.0 | Retry logic with exponential backoff |
| **PDF Processing** | PyMuPDF | ≥1.25.0 | PDF to image conversion |
| | Pillow | ≥11.0.0 | Image processing |
| | opencv-python | ≥4.10.0 | Advanced image enhancement |
| **Data Validation** | pydantic | ≥2.10.0 | Data validation and schemas |
| | pydantic-settings | ≥2.6.0 | Settings management |
| **API Framework** | fastapi | ≥0.115.0 | REST API framework |
| | uvicorn | ≥0.32.0 | ASGI server |
| | python-multipart | ≥0.0.17 | File upload support |
| | starlette | ≥0.41.0 | ASGI toolkit |
| **Task Queue** | celery | ≥5.4.0 | Distributed task queue |
| | kombu | ≥5.4.0 | Messaging library |
| **Security** | cryptography | ≥43.0.0 | AES-256 encryption |
| | python-jose | ≥3.3.0 | JWT handling |
| | passlib | ≥1.7.4 | Password hashing |
| | bcrypt | ≥4.2.0 | Bcrypt algorithm |
| **Export** | openpyxl | ≥3.1.5 | Excel export |
| | pandas | ≥2.2.0 | Data manipulation |
| | xlsxwriter | ≥3.2.0 | Excel writing |
| **UI** | streamlit | ≥1.50.0 | Web UI framework |
| | streamlit-extras | ≥0.4.0 | Additional components |
| **Monitoring** | prometheus-client | ≥0.21.0 | Prometheus metrics |
| | structlog | ≥24.4.0 | Structured logging |
| | python-json-logger | ≥2.0.7 | JSON logging |
| **Testing** | pytest | ≥8.3.0 | Testing framework |
| | pytest-asyncio | ≥0.24.0 | Async test support |
| | pytest-cov | ≥6.0.0 | Coverage reporting |
| | httpx | ≥0.27.0 | Async HTTP client for testing |
| | pytest-mock | ≥3.14.0 | Mocking support |
| **Development** | black | ≥24.10.0 | Code formatting |
| | ruff | ≥0.8.0 | Fast linting |
| | mypy | ≥1.13.0 | Type checking |
| | pre-commit | ≥4.0.0 | Git hooks |
| | isort | ≥5.13.0 | Import sorting |
| **Configuration** | python-dotenv | ≥1.0.1 | Environment variables |
| | pyyaml | ≥6.0.0 | YAML configuration |


### 8.4 LM Studio Server Configuration

will setup from the LM Studio app later

### 8.5 Documentation Links for Implementation

The following documentation links are essential for AI coding agents and developers implementing this system:

#### Agent Framework & LLM

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **LangChain** | https://python.langchain.com/docs/ | Core LangChain documentation |
| **LangChain API Reference** | https://python.langchain.com/api_reference/ | Detailed API reference |
| **LangGraph** | https://langchain-ai.github.io/langgraph/ | LangGraph documentation |
| **LangGraph Tutorials** | https://langchain-ai.github.io/langgraph/tutorials/ | Step-by-step tutorials |
| **LangGraph How-To Guides** | https://langchain-ai.github.io/langgraph/how-tos/ | Practical how-to guides |
| **LangGraph Checkpointing** | https://langchain-ai.github.io/langgraph/how-tos/persistence/ | State persistence guide |

#### Memory Layer

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Mem0** | https://docs.mem0.ai/ | Mem0 main documentation |
| **Mem0 Quickstart** | https://docs.mem0.ai/open-source/quickstart | Getting started guide |
| **Mem0 LangChain Integration** | https://docs.mem0.ai/integrations/langchain | LangChain integration |
| **Mem0 LangGraph Integration** | https://docs.mem0.ai/integrations/langgraph | LangGraph integration |
| **Mem0 Components** | https://docs.mem0.ai/components/overview | Memory components overview |
| **Mem0 Vector Stores** | https://docs.mem0.ai/components/vectordbs/overview | Vector database options |
| **Mem0 LLMs** | https://docs.mem0.ai/components/llms/overview | LLM configuration |
| **Mem0 Embedders** | https://docs.mem0.ai/components/embedders/overview | Embedding models |

#### VLM & Model Serving

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **LM Studio** | https://lmstudio.ai/docs | LM Studio documentation |
| **OpenAI Python SDK** | https://platform.openai.com/docs/api-reference | OpenAI API reference |
| **OpenAI Python GitHub** | https://github.com/openai/openai-python | Python SDK repository |

#### PDF Processing

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **PyMuPDF** | https://pymupdf.readthedocs.io/en/latest/ | PyMuPDF documentation |
| **PyMuPDF Tutorial** | https://pymupdf.readthedocs.io/en/latest/tutorial.html | Getting started tutorial |
| **PyMuPDF Recipes** | https://pymupdf.readthedocs.io/en/latest/recipes.html | Common use cases |
| **Pillow** | https://pillow.readthedocs.io/en/stable/ | Pillow documentation |
| **OpenCV Python** | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html | OpenCV Python tutorials |

#### Data Validation

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Pydantic** | https://docs.pydantic.dev/latest/ | Pydantic documentation |
| **Pydantic Models** | https://docs.pydantic.dev/latest/concepts/models/ | Model definition guide |
| **Pydantic Validators** | https://docs.pydantic.dev/latest/concepts/validators/ | Custom validators |
| **Pydantic Settings** | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ | Settings management |

#### API Framework

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **FastAPI** | https://fastapi.tiangolo.com/ | FastAPI documentation |
| **FastAPI Tutorial** | https://fastapi.tiangolo.com/tutorial/ | Step-by-step tutorial |
| **FastAPI Advanced** | https://fastapi.tiangolo.com/advanced/ | Advanced features |
| **Uvicorn** | https://www.uvicorn.org/ | ASGI server documentation |

#### Task Queue

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Celery** | https://docs.celeryq.dev/en/stable/ | Celery documentation |
| **Celery Getting Started** | https://docs.celeryq.dev/en/stable/getting-started/ | Getting started guide |

#### UI Framework

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Streamlit** | https://docs.streamlit.io/ | Streamlit documentation |
| **Streamlit API** | https://docs.streamlit.io/develop/api-reference | API reference |
| **Streamlit Components** | https://docs.streamlit.io/develop/concepts/custom-components | Custom components |

#### Vector Databases (for Mem0)

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Qdrant** | https://qdrant.tech/documentation/ | Qdrant documentation |
| **Qdrant Python Client** | https://python-client.qdrant.tech/ | Python client docs |
| **FAISS** | https://faiss.ai/index.html | FAISS documentation |
| **FAISS Python** | https://github.com/facebookresearch/faiss/wiki | FAISS wiki |

#### Embeddings

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Sentence Transformers** | https://www.sbert.net/ | Sentence Transformers docs |
| **Sentence Transformers Models** | https://www.sbert.net/docs/pretrained_models.html | Pre-trained models |
| **HuggingFace Embeddings** | https://huggingface.co/docs/transformers/main_classes/embeddings | HF embeddings |

#### Testing

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Pytest** | https://docs.pytest.org/en/stable/ | Pytest documentation |
| **Pytest Asyncio** | https://pytest-asyncio.readthedocs.io/en/latest/ | Async testing |
| **HTTPX** | https://www.python-httpx.org/ | HTTPX for API testing |

#### Monitoring

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Prometheus Python** | https://prometheus.github.io/client_python/ | Prometheus client |
| **Structlog** | https://www.structlog.org/en/stable/ | Structured logging |

#### Development Tools

| Component | Documentation URL | Description |
|-----------|-------------------|-------------|
| **Black** | https://black.readthedocs.io/en/stable/ | Code formatter |
| **Ruff** | https://docs.astral.sh/ruff/ | Fast linter |
| **Mypy** | https://mypy.readthedocs.io/en/stable/ | Type checker |

---

## 9. Directory Structure

```
doc-extraction-system/
│
├── README.md                          # Project documentation
├── PRD.md                             # This document
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Pydantic settings
│   │   └── logging_config.py          # Structured logging setup
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py           # PyMuPDF PDF handling
│   │   ├── image_enhancer.py          # OpenCV enhancement
│   │   └── batch_manager.py           # Memory-efficient batching
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   ├── lm_client.py               # LM Studio client
│   │   ├── connection_manager.py      # Connection pooling
│   │   └── health_monitor.py          # Health checks
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── base.py                    # Base schema classes
│   │   ├── field_definition.py         # Field definition for adding new fields 
│   │   ├── validators.py              # Field validators
│   │   ├── cms1500.py                 # CMS-1500 schema
│   │   ├── ub04.py                    # UB-04 schema
│   │   ├── eob.py                     # EOB schema
│   │   └── superbill.py               # Superbill schema
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseAgent class
│   │   ├── orchestrator.py            # LangGraph orchestrator
│   │   ├── analyzer.py                # Document analysis agent
│   │   ├── extractor.py               # Dual-pass extraction agent
│   │   └── validator.py               # Validation agent
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── grounding_rules.py         # Anti-hallucination rules
│   │   ├── classification.py          # Classification prompts
│   │   ├── extraction.py              # Extraction prompts
│   │   └── validation.py              # Validation prompts
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── dual_pass.py               # Dual-pass comparison
│   │   ├── pattern_detector.py        # Hallucination detection
│   │   ├── confidence.py              # Confidence scoring
│   │   ├── medical_codes.py           # CPT/ICD-10/NPI validation
│   │   └── cross_field.py             # Cross-field rules
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── mem0_client.py             # Mem0 client wrapper
│   │   ├── context_manager.py         # Context retrieval and storage
│   │   ├── correction_tracker.py      # Track user corrections
│   │   └── vector_store.py            # Qdrant vector store config
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── state.py                   # ExtractionState TypedDict
│   │   ├── graph.py                   # LangGraph workflow
│   │   └── runner.py                  # Pipeline executor
│   │
│   ├── queue/
│   │   ├── __init__.py
│   │   ├── tasks.py                   # Celery task definitions
│   │   └── worker.py                  # Worker configuration
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   ├── excel_exporter.py          # Multi-sheet Excel export
│   │   └── json_exporter.py           # JSON export with metadata
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── encryption.py              # AES-256 encryption
│   │   ├── audit.py                   # Audit logging
│   │   ├── data_cleanup.py            # Secure file cleanup
│   │   └── rbac.py                    # Role-based access
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Prometheus metrics
│   │   └── alerts.py                  # Alert definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py              # File handling utilities
│       └── json_utils.py              # JSON utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── unit/
│   │   ├── test_pdf_processor.py
│   │   ├── test_lm_client.py
│   │   ├── test_schemas.py
│   │   ├── test_analyzer.py
│   │   ├── test_extractor.py
│   │   └── test_validator.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   └── accuracy/
│       ├── test_superbill_accuracy.py
│       └── golden_dataset/
│
├── app.py                             # Streamlit application
│
├── scripts/
│   ├── setup_environment.sh           # Setup script
│   ├── verify_setup.py                # Verification
│   └── run_benchmarks.py              # Performance tests
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment_guide.md
│

```

---

## 10. Implementation Phases

### Phase 0: Prerequisites & Setup (Week 1)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| Hardware procurement | GPU server ready | RTX 4090 or equivalent |
| LM Studio installation | Server on port 1234 | Health check passes |
| Model download | Qwen3-VL Q4_K_M loaded | Vision requests work |
| Python environment | venv with dependencies | All imports successful |
| Repository setup | Git repo with structure | CI/CD initialized |

### Phase 1: Core Infrastructure (Weeks 2-3)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| PDF Processor | 300 DPI extraction | Unit tests pass |
| Image Enhancer | OpenCV pipeline | Quality improvement verified |
| LM Studio Client | Vision requests work | Retry logic tested |
| Schema System | Pydantic schemas | Validation works |
| Healthcare Schemas | CMS-1500, Superbill, EOB | Cross-field rules work |

### Phase 2: Agent Framework (Weeks 4-6)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| LangGraph Setup | StateGraph compiles | Checkpointing works |
| Orchestrator Agent | Workflow control | State transitions work |
| Analyzer Agent | Document classification | >95% accuracy |
| Extractor Agent | Dual-pass extraction | Comparison works |
| Validator Agent | Pattern detection | Hallucinations flagged |

### Phase 3: Anti-Hallucination System (Weeks 7-8)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| Grounding Rules | Prompt templates | Reduced hallucinations |
| Dual-Pass Logic | Comparison algorithm | Mismatches detected |
| Pattern Detection | Hallucination flags | Adversarial tests pass |
| Confidence Scoring | Thresholds work | <10% human review |
| Human Review Queue | Review interface | Queue operational |

### Phase 4: Integration & Testing (Weeks 9-10)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| Celery Tasks | Async processing | Background tasks work |
| Unit Tests | >80% coverage | All tests pass |
| Integration Tests | E2E pipeline | Sample docs work |
| Accuracy Tests | Golden dataset | >95% accuracy |

### Phase 5: Deployment (Weeks 11-12)

| Task | Deliverable | Exit Criteria |
|------|-------------|---------------|
| HIPAA Compliance | Encryption, audit | Security review passes |
| Monitoring | Prometheus/Grafana | Dashboards live |
| Documentation | API docs, runbooks | Team trained |
| Production Deploy | Go-live | Pilot successful |

---

## 12. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Field Extraction Accuracy | >95% | Golden dataset comparison |
| Hallucination Rate | <2% | Adversarial test suite |
| Processing Speed | 15-25 sec/page | Benchmark suite |
| VLM Calls per Page | 3-4 | Pipeline metrics |
| System Uptime | >99.5% | Prometheus monitoring |
| Human Review Rate | <10% | Production metrics |
| Dual-Pass Agreement | >90% | Extraction metrics |

---

## 12. Compliance & Security

### HIPAA Compliance

| Requirement | Implementation |
|-------------|----------------|
| 100% Local Processing | LM Studio on localhost |
| No Cloud APIs | All AI processing local |
| Encrypted Storage | AES-256 for data at rest |
| Audit Logging | All actions logged with timestamps |
| Secure Cleanup | 3-pass secure file deletion |
| Access Control | RBAC with role-based permissions |

### Security Features

| Feature | Description |
|---------|-------------|
| Network Isolation | LM Studio on localhost only |
| Input Validation | All inputs sanitized |
| PHI Masking | Sensitive data masked in logs |
| Secure Temp Files | Encrypted temporary storage |
| Automatic Cleanup | PHI deleted after processing |
| Audit Trail | Complete action history |

---

## 13. Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hallucinations | Medium | High | 3-layer validation system |
| Poor document quality | High | Medium | OpenCV enhancement pipeline |
| Model accuracy drift | Medium | Medium | Golden dataset regression tests |
| **Custom schema quality** | **Medium** | **Medium** | **Schema validation + field description guidelines** |
| **Per-field confidence miscalibration** | **Low** | **Medium** | **Regular confidence calibration testing** |
| Hardware failure | Low | High | Checkpointing, backups |
| HIPAA violation | Low | Critical | 100% local, audit logging |
| Prompt injection | Low | Medium | Input sanitization |

---

*Document Version: 2.1*  
*Last Updated: December 2025*  
*Framework: LangChain 1.x + LangGraph 1.x*  
*New Features: Per-Field Confidence Scoring, Zero-Shot Custom Schemas*
