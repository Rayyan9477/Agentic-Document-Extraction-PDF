# Local Agentic Document Extraction System
## Complete Project Plan

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Inventory](#3-component-inventory)
4. [Phase 0: Prerequisites & Setup](#4-phase-0-prerequisites--setup-week-1)
5. [Phase 1: Core Infrastructure](#5-phase-1-core-infrastructure-weeks-2-3)
6. [Phase 2: Agent Framework](#6-phase-2-agent-framework-weeks-4-6)
7. [Phase 3: Anti-Hallucination System](#7-phase-3-anti-hallucination-system-weeks-7-8)
8. [Phase 4: Integration & Testing](#8-phase-4-integration--testing-weeks-9-10)
9. [Phase 5: Deployment](#9-phase-5-deployment-weeks-11-12)
10. [Resource Requirements](#10-resource-requirements)
11. [Risk Management](#11-risk-management)
12. [Success Metrics](#12-success-metrics)

---

# 1. Project Overview

## 1.1 Objective

Build a production-ready, HIPAA-compliant document extraction system using local Vision Language Models (VLM) with a 4-agent architecture for healthcare Revenue Cycle Management (RCM) documents.

## 1.2 Key Specifications

| Attribute | Value |
|-----------|-------|
| **Model** | Qwen3-VL 8B |
| **Backend** | LM Studio (Local) |
| **Framework** | LangGraph |
| **Agents** | 4 Specialized |
| **Validation** | 3-Layer Anti-Hallucination |
| **Compliance** | HIPAA Ready |
| **Timeline** | 12 Weeks |
| **Team Size** | 2-3 Engineers |

## 1.3 Project Timeline Summary

```
Week 1      : Phase 0 - Prerequisites & Setup
Weeks 2-3   : Phase 1 - Core Infrastructure
Weeks 4-6   : Phase 2 - Agent Framework
Weeks 7-8   : Phase 3 - Anti-Hallucination System
Weeks 9-10  : Phase 4 - Integration & Testing
Weeks 11-12 : Phase 5 - Deployment
Week 13-14  : Buffer (contingency)
```

---

# 2. System Architecture

## 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  REST API   │  │  Batch Job  │  │  File Drop  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING LAYER                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  PDF Processor                           │    │
│  │  • PDF Validation    • Page Extraction (300 DPI)        │    │
│  │  • Quality Enhancement    • Batch Manager               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER (4 Agents)                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AGENT 1: ORCHESTRATOR                       │    │
│  │  • LangGraph State Machine    • Workflow Control        │    │
│  │  • Error Handling             • Checkpointing           │    │
│  │  • VLM Calls: 0                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AGENT 2: ANALYZER                           │    │
│  │  • Document Classification    • Structure Detection     │    │
│  │  • Page Relationships         • Schema Selection        │    │
│  │  • VLM Calls: 1 per document                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AGENT 3: EXTRACTOR                          │    │
│  │  • Schema-Driven Extraction   • Dual-Pass Verification  │    │
│  │  • Confidence Scoring         • Visual Grounding        │    │
│  │  • VLM Calls: 2 per page                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AGENT 4: VALIDATOR                          │    │
│  │  • Schema Validation          • Hallucination Detection │    │
│  │  • Cross-Page Merging         • Output Formatting       │    │
│  │  • VLM Calls: 0-1 per document                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       VLM BACKEND                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  LM Studio Client                        │    │
│  │  • Connection Manager    • Vision Request Handler       │    │
│  │  • Response Parser       • Retry Logic                  │    │
│  │  • Health Monitoring                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LM Studio Server (localhost:1234)           │    │
│  │  • Qwen3-VL 8B (Q4_K_M)    • 32K Context               │    │
│  │  • ~6GB VRAM              • OpenAI-Compatible API       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    JSON     │  │  Markdown   │  │  Database   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Data Flow

```
PDF Upload
    │
    ▼
[1] PDF Processor validates and converts to images (300 DPI)
    │
    ▼
[2] Orchestrator initializes state and routes to Analyzer
    │
    ▼
[3] Analyzer classifies document and selects schema (1 VLM call)
    │
    ▼
[4] Extractor performs dual-pass extraction (2 VLM calls/page)
    │
    ▼
[5] Validator checks quality and formats output (0-1 VLM call)
    │
    ├── High Confidence (≥0.85) → Auto-accept
    ├── Medium Confidence (0.50-0.84) → Re-extract or verify
    └── Low Confidence (<0.50) → Human review queue
    │
    ▼
JSON Output + Audit Log
```

---

# 3. Component Inventory

## 3.1 Complete Component List

| ID | Component | Type | Phase | Priority |
|----|-----------|------|-------|----------|
| C01 | Hardware Infrastructure | Infrastructure | 0 | Critical |
| C02 | LM Studio Server | Infrastructure | 0 | Critical |
| C03 | Qwen3-VL Model | Model | 0 | Critical |
| C04 | Python Environment | Infrastructure | 0 | Critical |
| C05 | PDF Processor | Utility Module | 1 | Critical |
| C06 | LM Studio Client | Utility Module | 1 | Critical |
| C07 | Schema Definition System | Core Library | 1 | Critical |
| C08 | Healthcare RCM Schemas | Configuration | 1 | High |
| C09 | Orchestrator Agent | Agent | 2 | Critical |
| C10 | Analyzer Agent | Agent | 2 | Critical |
| C11 | Extractor Agent | Agent | 2 | Critical |
| C12 | Validator Agent | Agent | 2 | Critical |
| C13 | Prompt Engineering Layer | Anti-Hallucination | 3 | Critical |
| C14 | Dual-Pass Extraction | Anti-Hallucination | 3 | Critical |
| C15 | Pattern Validation | Anti-Hallucination | 3 | Critical |
| C16 | Confidence Scoring System | Anti-Hallucination | 3 | High |
| C17 | Human-in-the-Loop Interface | Anti-Hallucination | 3 | High |
| C18 | REST API | Integration | 4 | Critical |
| C19 | Task Queue | Integration | 4 | High |
| C20 | Unit Test Suite | Testing | 4 | Critical |
| C21 | Integration Test Suite | Testing | 4 | Critical |
| C22 | Accuracy Test Suite | Testing | 4 | Critical |
| C23 | HIPAA Compliance Module | Security | 5 | Critical |
| C24 | Monitoring System | Operations | 5 | High |
| C25 | Alerting System | Operations | 5 | High |
| C26 | Documentation | Operations | 5 | High |

## 3.2 Directory Structure

```
doc-extraction-system/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # Application settings
│   │   └── logging_config.py        # Logging configuration
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py         # C05: PDF processing
│   │   ├── image_enhancer.py        # Image quality enhancement
│   │   └── batch_manager.py         # Batch processing
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   ├── lm_client.py             # C06: LM Studio client
│   │   ├── connection_manager.py    # Connection pooling
│   │   └── health_monitor.py        # Health checks
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── base.py                  # C07: Base schema classes
│   │   ├── validators.py            # Field validators
│   │   ├── cms1500.py               # C08: CMS-1500 schema
│   │   ├── ub04.py                  # C08: UB-04 schema
│   │   ├── eob.py                   # C08: EOB schema
│   │   └── lab_report.py            # C08: Lab report schema
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base agent class
│   │   ├── orchestrator.py          # C09: Orchestrator agent
│   │   ├── analyzer.py              # C10: Analyzer agent
│   │   ├── extractor.py             # C11: Extractor agent
│   │   └── validator.py             # C12: Validator agent
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── grounding_rules.py       # C13: Grounding instructions
│   │   ├── classification.py        # Classification prompts
│   │   ├── extraction.py            # Extraction prompts
│   │   └── validation.py            # Validation prompts
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── dual_pass.py             # C14: Dual-pass logic
│   │   ├── pattern_detector.py      # C15: Hallucination patterns
│   │   ├── confidence.py            # C16: Confidence scoring
│   │   └── cross_field.py           # Cross-field rules
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── state.py                 # State definitions
│   │   ├── graph.py                 # LangGraph workflow
│   │   └── runner.py                # Pipeline executor
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # C18: FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── extraction.py        # Extraction endpoints
│   │   │   ├── tasks.py             # Task status endpoints
│   │   │   └── health.py            # Health endpoints
│   │   ├── models.py                # API request/response models
│   │   └── dependencies.py          # FastAPI dependencies
│   │
│   ├── queue/
│   │   ├── __init__.py
│   │   ├── tasks.py                 # C19: Celery tasks
│   │   └── worker.py                # Worker configuration
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── encryption.py            # C23: Data encryption
│   │   ├── audit.py                 # Audit logging
│   │   └── rbac.py                  # Access control
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # C24: Prometheus metrics
│   │   └── alerts.py                # C25: Alert definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       └── json_utils.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   │
│   ├── unit/                        # C20: Unit tests
│   │   ├── __init__.py
│   │   ├── test_pdf_processor.py
│   │   ├── test_lm_client.py
│   │   ├── test_schemas.py
│   │   ├── test_orchestrator.py
│   │   ├── test_analyzer.py
│   │   ├── test_extractor.py
│   │   └── test_validator.py
│   │
│   ├── integration/                 # C21: Integration tests
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   │
│   ├── accuracy/                    # C22: Accuracy tests
│   │   ├── __init__.py
│   │   ├── test_cms1500_accuracy.py
│   │   ├── test_eob_accuracy.py
│   │   └── golden_dataset/
│   │       ├── cms1500/
│   │       ├── eob/
│   │       └── annotations/
│   │
│   └── adversarial/                 # Hallucination tests
│       ├── __init__.py
│       └── test_hallucination.py
│
├── scripts/
│   ├── setup_environment.sh
│   ├── download_model.py
│   ├── run_benchmarks.py
│   └── generate_golden_dataset.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
│
├── docs/                            # C26: Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   ├── operations_runbook.md
│   └── schema_guide.md
│
└── config/
    ├── default.yaml
    ├── development.yaml
    ├── production.yaml
    └── schemas/
        ├── cms1500.yaml
        ├── ub04.yaml
        └── eob.yaml
```

---

# 4. Phase 0: Prerequisites & Setup (Week 1)

## 4.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 1 Week (5 days) |
| **Resources** | 1 Engineer + IT Support |
| **Dependencies** | Budget approval, Hardware procurement |
| **Deliverables** | Working development environment |

## 4.2 Tasks Breakdown

### Task 0.1: Hardware Procurement (Days 1-2)

**Owner:** IT/Procurement

**Deliverable:** Hardware ready for setup

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.1.1 | Finalize hardware specifications | 2 hours | Budget approval |
| 0.1.2 | Submit purchase requisition | 1 hour | 0.1.1 |
| 0.1.3 | Receive and inventory hardware | 4 hours | 0.1.2 |
| 0.1.4 | Physical installation | 2 hours | 0.1.3 |
| 0.1.5 | Verify hardware functionality | 1 hour | 0.1.4 |

**Hardware Specifications:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) or A6000 (48GB) |
| RAM | 32GB DDR4 | 64GB DDR5 |
| CPU | 8-core (Intel i7/AMD Ryzen 7) | 16-core (Intel i9/AMD Ryzen 9) |
| Storage | 500GB NVMe SSD | 1TB NVMe SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Task 0.2: LM Studio Installation (Day 2)

**Owner:** Lead Engineer

**Deliverable:** LM Studio running on target machine

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.2.1 | Download LM Studio installer | 15 min | Hardware ready |
| 0.2.2 | Install LM Studio | 30 min | 0.2.1 |
| 0.2.3 | Configure GPU settings | 30 min | 0.2.2 |
| 0.2.4 | Enable vision model support | 15 min | 0.2.3 |
| 0.2.5 | Verify installation | 15 min | 0.2.4 |

**Configuration Settings:**

```yaml
lm_studio:
  server:
    port: 1234
    host: localhost
  gpu:
    layers: all  # All layers on GPU
    memory_fraction: 0.9
  context:
    length: 32768
  features:
    vision: enabled
```

### Task 0.3: Model Download & Configuration (Day 2-3)

**Owner:** Lead Engineer

**Deliverable:** Qwen3-VL model loaded and responding

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.3.1 | Search for Qwen3-VL model | 15 min | LM Studio installed |
| 0.3.2 | Download Q4_K_M quantization | 2-4 hours | 0.3.1 |
| 0.3.3 | Configure model parameters | 30 min | 0.3.2 |
| 0.3.4 | Load model to GPU | 15 min | 0.3.3 |
| 0.3.5 | Test basic text generation | 15 min | 0.3.4 |
| 0.3.6 | Test vision capability | 30 min | 0.3.5 |

**Quantization Options:**

| Quantization | VRAM | Accuracy | Recommended For |
|--------------|------|----------|-----------------|
| FP16 | ~18GB | 100% | Maximum accuracy |
| INT8 | ~10GB | ~99% | Balanced |
| **Q4_K_M** | **~6GB** | **~97%** | **Production (recommended)** |
| Q4_K_S | ~5GB | ~95% | Limited VRAM |

### Task 0.4: Python Environment Setup (Day 3)

**Owner:** Lead Engineer

**Deliverable:** Python environment with all dependencies

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.4.1 | Install Python 3.11 | 30 min | OS ready |
| 0.4.2 | Create virtual environment | 15 min | 0.4.1 |
| 0.4.3 | Install core dependencies | 30 min | 0.4.2 |
| 0.4.4 | Install development tools | 15 min | 0.4.3 |
| 0.4.5 | Configure IDE/editor | 30 min | 0.4.4 |
| 0.4.6 | Verify all imports | 15 min | 0.4.5 |

**Dependencies (requirements.txt):**

```
# Core
langgraph>=0.0.40
langchain>=0.1.0
PyMuPDF>=1.23.0
openai>=1.6.0
pydantic>=2.5.0
Pillow>=10.0.0
tenacity>=8.2.0

# API
fastapi>=0.109.0
uvicorn>=0.25.0
python-multipart>=0.0.6

# Queue
celery>=5.3.0
redis>=5.0.0

# Security
cryptography>=41.0.0
python-jose>=3.3.0

# Monitoring
prometheus-client>=0.19.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
httpx>=0.26.0

# Development
black>=23.12.0
ruff>=0.1.9
mypy>=1.8.0
pre-commit>=3.6.0
```

### Task 0.5: Repository Setup (Day 3-4)

**Owner:** Lead Engineer

**Deliverable:** Git repository with project structure

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.5.1 | Initialize Git repository | 15 min | - |
| 0.5.2 | Create directory structure | 1 hour | 0.5.1 |
| 0.5.3 | Setup pyproject.toml | 30 min | 0.5.2 |
| 0.5.4 | Configure pre-commit hooks | 30 min | 0.5.3 |
| 0.5.5 | Create .env.example | 15 min | 0.5.4 |
| 0.5.6 | Write initial README | 30 min | 0.5.5 |

### Task 0.6: Verification & Documentation (Day 4-5)

**Owner:** Lead Engineer

**Deliverable:** Verified environment, setup documentation

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 0.6.1 | Create verification script | 1 hour | All setup complete |
| 0.6.2 | Test VLM with sample image | 30 min | 0.6.1 |
| 0.6.3 | Test JSON output parsing | 30 min | 0.6.2 |
| 0.6.4 | Benchmark inference speed | 1 hour | 0.6.3 |
| 0.6.5 | Document setup process | 2 hours | 0.6.4 |
| 0.6.6 | Create troubleshooting guide | 1 hour | 0.6.5 |


## 4.3 Phase 0 Deliverables Checklist

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | Hardware installed and functional | GPU detected, VRAM available |
| 2 | LM Studio installed and configured | Server responds on port 1234 |
| 3 | Qwen3-VL model loaded | Model listed in /v1/models |
| 4 | Python environment ready | All imports successful |
| 5 | Repository initialized | Git repo with structure |
| 6 | Vision capability verified | Image + text prompt works |
| 7 | Setup documentation complete | README and guides written |

## 4.4 Phase 0 Exit Criteria

- [ ] Hardware meets minimum specifications
- [ ] LM Studio server running on localhost:1234
- [ ] Qwen3-VL responds to vision prompts
- [ ] Python environment activated with all dependencies
- [ ] Git repository created with project structure
- [ ] Verification script passes all checks
- [ ] Setup documentation complete

---

# 5. Phase 1: Core Infrastructure (Weeks 2-3)

## 5.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2 Weeks (10 days) |
| **Resources** | 2 Engineers |
| **Dependencies** | Phase 0 complete |
| **Deliverables** | PDF processor, LM client, schema system |

## 5.2 Tasks Breakdown

### Task 1.1: PDF Processor Module (Days 1-3)

**Owner:** Engineer 1

**Deliverable:** Complete PDF processing pipeline

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 1.1.1 | Create pdf_processor.py skeleton | 1 hour | Phase 0 |
| 1.1.2 | Implement PDF validation | 2 hours | 1.1.1 |
| 1.1.3 | Implement page extraction (300 DPI) | 3 hours | 1.1.2 |
| 1.1.4 | Add image quality enhancement | 4 hours | 1.1.3 |
| 1.1.5 | Implement batch processing | 3 hours | 1.1.4 |
| 1.1.6 | Add memory management | 2 hours | 1.1.5 |
| 1.1.7 | Write unit tests | 3 hours | 1.1.6 |
| 1.1.8 | Documentation | 1 hour | 1.1.7 |

### Task 1.2: LM Studio Client (Days 3-5)

**Owner:** Engineer 2

**Deliverable:** Robust client for VLM communication

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 1.2.1 | Create lm_client.py skeleton | 1 hour | Phase 0 |
| 1.2.2 | Implement connection manager | 2 hours | 1.2.1 |
| 1.2.3 | Build vision request handler | 3 hours | 1.2.2 |
| 1.2.4 | Implement response parser | 2 hours | 1.2.3 |
| 1.2.5 | Add retry logic with tenacity | 2 hours | 1.2.4 |
| 1.2.6 | Implement health monitoring | 2 hours | 1.2.5 |
| 1.2.7 | Add timeout handling | 1 hour | 1.2.6 |
| 1.2.8 | Write unit tests | 3 hours | 1.2.7 |
| 1.2.9 | Documentation | 1 hour | 1.2.8 |

**Implementation (src/client/lm_client.py):**

```python
"""LM Studio Client for Vision Language Model interactions."""

import base64
import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class LMClientConfig(BaseModel):
    """Configuration for LM Studio client."""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    model: str = "qwen3-vl"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 120
    max_retries: int = 3


class VisionRequest(BaseModel):
    """Request for vision model."""
    image_b64: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class VisionResponse(BaseModel):
    """Response from vision model."""
    content: str
    parsed_json: Optional[dict] = None
    usage: dict
    model: str


class LMStudioClient:
    """Client for interacting with LM Studio vision models."""
    
    def __init__(self, config: Optional[LMClientConfig] = None):
        self.config = config or LMClientConfig()
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client
    
    def health_check(self) -> tuple[bool, str]:
        """
        Check if LM Studio is healthy.
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            models = self.client.models.list()
            if len(models.data) == 0:
                return False, "No models loaded"
            return True, f"Healthy, model: {models.data[0].id}"
        except APIConnectionError:
            return False, "Cannot connect to LM Studio"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    )
    def vision_request(self, request: VisionRequest) -> VisionResponse:
        """
        Send image + prompt to vision model.
        
        Args:
            request: VisionRequest with image and prompt
            
        Returns:
            VisionResponse with model output
        """
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Add user message with image and text
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{request.image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": request.prompt
                }
            ]
        })
        
        logger.debug(f"Sending vision request, prompt length: {len(request.prompt)}")
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=request.temperature or self.config.temperature,
        )
        
        content = response.choices[0].message.content
        
        # Try to parse JSON from response
        parsed_json = self._extract_json(content)
        
        return VisionResponse(
            content=content,
            parsed_json=parsed_json,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
        )
    
    def _extract_json(self, content: str) -> Optional[dict]:
        """Extract JSON from model response."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode('utf-8')
```

### Task 1.3: Schema Definition System (Days 5-7)

**Owner:** Engineer 1

**Deliverable:** Flexible schema system for document types

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 1.3.1 | Design schema architecture | 2 hours | - |
| 1.3.2 | Create base schema classes | 3 hours | 1.3.1 |
| 1.3.3 | Implement field validators | 4 hours | 1.3.2 |
| 1.3.4 | Build cross-field rule engine | 4 hours | 1.3.3 |
| 1.3.5 | Add schema loader (YAML) | 2 hours | 1.3.4 |
| 1.3.6 | Write unit tests | 3 hours | 1.3.5 |


### Task 1.4: Healthcare RCM Schemas (Days 7-10)

**Owner:** Engineer 2

**Deliverable:** Pre-built schemas for healthcare documents

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 1.4.1 | Research CMS-1500 field requirements | 2 hours | Schema system |
| 1.4.2 | Implement CMS-1500 schema | 4 hours | 1.4.1 |
| 1.4.3 | Research UB-04 field requirements | 2 hours | - |
| 1.4.4 | Implement UB-04 schema | 4 hours | 1.4.3 |
| 1.4.5 | Research EOB formats | 2 hours | - |
| 1.4.6 | Implement EOB schema | 4 hours | 1.4.5 |
| 1.4.7 | Implement Lab Report schema | 3 hours | - |
| 1.4.8 | Write integration tests | 3 hours | 1.4.2, 1.4.4, 1.4.6, 1.4.7 |

## 5.3 Phase 1 Deliverables Checklist

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | PDF Processor module | Extracts pages at 300 DPI, passes tests |
| 2 | Image enhancement pipeline | Improves contrast, reduces noise |
| 3 | LM Studio Client | Connects, sends requests, parses responses |
| 4 | Retry logic | Handles transient failures |
| 5 | Health monitoring | Reports LM Studio status |
| 6 | Schema base classes | Defines fields, validators, rules |
| 7 | Field validators | Validates dates, currency, NPI, ICD-10, CPT |
| 8 | CMS-1500 schema | Complete with cross-field rules |
| 9 | UB-04 schema | Complete with cross-field rules |
| 10 | EOB schema | Complete with cross-field rules |
| 11 | Unit tests | >80% coverage |
| 12 | Documentation | API docs for all modules |

## 5.4 Phase 1 Exit Criteria

- [ ] PDF processor extracts pages from test documents
- [ ] LM Studio client connects and receives responses
- [ ] All field validators pass test cases
- [ ] At least 3 healthcare schemas implemented
- [ ] Unit test coverage > 80%
- [ ] Documentation complete for all modules

---

# 6. Phase 2: Agent Framework (Weeks 4-6)

## 6.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 3 Weeks (15 days) |
| **Resources** | 2 Engineers |
| **Dependencies** | Phase 1 complete |
| **Deliverables** | 4 working agents, LangGraph pipeline |

## 6.2 Tasks Breakdown

### Task 2.1: State Machine & Orchestrator (Days 1-4)

**Owner:** Engineer 1

**Deliverable:** LangGraph-based orchestrator

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 2.1.1 | Design state schema | 2 hours | Phase 1 |
| 2.1.2 | Define state transitions | 2 hours | 2.1.1 |
| 2.1.3 | Create LangGraph workflow | 4 hours | 2.1.2 |
| 2.1.4 | Implement checkpointing | 3 hours | 2.1.3 |
| 2.1.5 | Add error handling | 3 hours | 2.1.4 |
| 2.1.6 | Implement retry logic | 2 hours | 2.1.5 |
| 2.1.7 | Add logging/tracing | 2 hours | 2.1.6 |
| 2.1.8 | Write unit tests | 4 hours | 2.1.7 |

### Task 2.2: Analyzer Agent (Days 4-7)

**Owner:** Engineer 2

**Deliverable:** Document classification and analysis agent

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 2.2.1 | Create base agent class | 2 hours | - |
| 2.2.2 | Design classification prompt | 3 hours | 2.2.1 |
| 2.2.3 | Implement structure detection | 4 hours | 2.2.2 |
| 2.2.4 | Add page relationship logic | 3 hours | 2.2.3 |
| 2.2.5 | Build schema selector | 2 hours | 2.2.4 |
| 2.2.6 | Add confidence scoring | 2 hours | 2.2.5 |
| 2.2.7 | Write unit tests | 4 hours | 2.2.6 |

### Task 2.3: Extractor Agent (Days 7-10)

**Owner:** Engineer 1

**Deliverable:** Dual-pass extraction agent

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 2.3.1 | Design extraction prompt | 3 hours | Analyzer complete |
| 2.3.2 | Implement single-pass extraction | 4 hours | 2.3.1 |
| 2.3.3 | Implement dual-pass logic | 4 hours | 2.3.2 |
| 2.3.4 | Add field comparison | 3 hours | 2.3.3 |
| 2.3.5 | Implement confidence scoring | 3 hours | 2.3.4 |
| 2.3.6 | Add visual grounding | 2 hours | 2.3.5 |
| 2.3.7 | Handle tables | 3 hours | 2.3.6 |
| 2.3.8 | Write unit tests | 4 hours | 2.3.7 |

### Task 2.4: Validator Agent (Days 10-15)

**Owner:** Engineer 2

**Deliverable:** Validation and output agent

| Subtask | Description | Duration | Dependency |
|---------|-------------|----------|------------|
| 2.4.1 | Implement schema validation | 4 hours | Extractor complete |
| 2.4.2 | Add hallucination pattern detection | 4 hours | 2.4.1 |
| 2.4.3 | Implement cross-field rules | 4 hours | 2.4.2 |
| 2.4.4 | Build cross-page merger | 3 hours | 2.4.3 |
| 2.4.5 | Add confidence calibration | 3 hours | 2.4.4 |
| 2.4.6 | Create output formatter | 2 hours | 2.4.5 |
| 2.4.7 | Write unit tests | 4 hours | 2.4.6 |

## 6.3 Phase 2 Deliverables Checklist

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | ExtractionState schema | All fields defined |
| 2 | LangGraph workflow | Compiles and runs |
| 3 | Checkpointing | Resume works |
| 4 | Orchestrator Agent | Routes correctly |
| 5 | Analyzer Agent | Classifies documents |
| 6 | Extractor Agent | Dual-pass extraction works |
| 7 | Validator Agent | Validates and formats |
| 8 | Field comparison | Merges dual-pass results |
| 9 | Page merging | Handles multi-page docs |
| 10 | Agent unit tests | >80% coverage |
| 11 | Integration test | End-to-end pipeline works |

## 6.4 Phase 2 Exit Criteria

- [ ] All 4 agents implemented and tested
- [ ] LangGraph workflow executes successfully
- [ ] Checkpointing and resume works
- [ ] Dual-pass extraction produces merged results
- [ ] Multi-page documents handled correctly
- [ ] Unit test coverage > 80%
- [ ] End-to-end test passes with sample documents

---

# 7. Phase 3: Anti-Hallucination System (Weeks 7-8)

## 7.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2 Weeks (10 days) |
| **Resources** | 2 Engineers |
| **Dependencies** | Phase 2 complete |
| **Deliverables** | 3-layer validation, confidence system, HITL interface |

## 7.2 Tasks Breakdown

### Task 3.1: Layer 1 - Prompt Engineering (Days 1-2)

**Owner:** Engineer 1

| Subtask | Description | Duration |
|---------|-------------|----------|
| 3.1.1 | Design grounding rules | 2 hours |
| 3.1.2 | Create prompt templates | 3 hours |
| 3.1.3 | Add structured output enforcement | 2 hours |
| 3.1.4 | Test with edge cases | 4 hours |
| 3.1.5 | Document prompt patterns | 2 hours |

### Task 3.2: Layer 2 - Dual-Pass Enhancement (Days 2-4)

**Owner:** Engineer 2

| Subtask | Description | Duration |
|---------|-------------|----------|
| 3.2.1 | Design variant prompts | 3 hours |
| 3.2.2 | Implement comparison logic | 4 hours |
| 3.2.3 | Add mismatch flagging | 2 hours |
| 3.2.4 | Tune confidence scoring | 3 hours |
| 3.2.5 | Test with misaligned extractions | 4 hours |

### Task 3.3: Layer 3 - Pattern Validation (Days 4-6)

**Owner:** Engineer 1

| Subtask | Description | Duration |
|---------|-------------|----------|
| 3.3.1 | Implement repetition detection | 3 hours |
| 3.3.2 | Add placeholder detection | 2 hours |
| 3.3.3 | Implement cross-field rules | 4 hours |
| 3.3.4 | Add code validators | 3 hours |
| 3.3.5 | Test with adversarial cases | 4 hours |

### Task 3.4: Confidence Scoring System (Days 6-8)

**Owner:** Engineer 2

| Subtask | Description | Duration |
|---------|-------------|----------|
| 3.4.1 | Design scoring algorithm | 3 hours |
| 3.4.2 | Implement threshold actions | 3 hours |
| 3.4.3 | Add per-field confidence | 2 hours |
| 3.4.4 | Implement confidence calibration | 4 hours |
| 3.4.5 | Test and tune thresholds | 4 hours |

### Task 3.5: Human-in-the-Loop Interface (Days 8-10)

**Owner:** Engineer 1

| Subtask | Description | Duration |
|---------|-------------|----------|
| 3.5.1 | Design review queue | 2 hours |
| 3.5.2 | Build review data model | 2 hours |
| 3.5.3 | Create review API endpoints | 4 hours |
| 3.5.4 | Implement correction feedback | 3 hours |
| 3.5.5 | Add review metrics | 2 hours |
| 3.5.6 | Test review workflow | 3 hours |

## 7.3 Phase 3 Exit Criteria

- [ ] Grounding rules reduce hallucinations in testing
- [ ] Dual-pass catches mismatches
- [ ] Pattern detection flags suspicious values
- [ ] Confidence scoring produces actionable thresholds
- [ ] Human review queue operational
- [ ] Target: <10% of documents require human review

---

# 8. Phase 4: Integration & Testing (Weeks 9-10)

## 8.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2 Weeks (10 days) |
| **Resources** | 2 Engineers |
| **Dependencies** | Phase 3 complete |
| **Deliverables** | REST API, task queue, test suites |

## 8.2 Tasks Breakdown

### Task 4.1: REST API Development (Days 1-4)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 4.1.1 | Setup FastAPI project | 2 hours | Eng 1 |
| 4.1.2 | Create extraction endpoint | 4 hours | Eng 1 |
| 4.1.3 | Create task status endpoint | 2 hours | Eng 1 |
| 4.1.4 | Create batch endpoint | 4 hours | Eng 1 |
| 4.1.5 | Add health endpoint | 1 hour | Eng 1 |
| 4.1.6 | Implement file upload | 3 hours | Eng 1 |
| 4.1.7 | Add request validation | 2 hours | Eng 1 |
| 4.1.8 | Write API tests | 4 hours | Eng 1 |

### Task 4.2: Task Queue Setup (Days 3-5)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 4.2.1 | Setup Celery | 2 hours | Eng 2 |
| 4.2.2 | Setup Redis | 1 hour | Eng 2 |
| 4.2.3 | Create task definitions | 3 hours | Eng 2 |
| 4.2.4 | Implement result backend | 2 hours | Eng 2 |
| 4.2.5 | Add task monitoring | 2 hours | Eng 2 |
| 4.2.6 | Test async processing | 3 hours | Eng 2 |

### Task 4.3: Test Suite Development (Days 5-10)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 4.3.1 | Create test fixtures | 3 hours | Eng 1 |
| 4.3.2 | Write unit tests | 8 hours | Eng 1 |
| 4.3.3 | Write integration tests | 8 hours | Eng 2 |
| 4.3.4 | Create golden dataset | 8 hours | Eng 2 |
| 4.3.5 | Write accuracy tests | 6 hours | Eng 1 |
| 4.3.6 | Write adversarial tests | 4 hours | Eng 2 |
| 4.3.7 | Setup CI pipeline | 4 hours | Eng 1 |

## 8.3 Phase 4 Exit Criteria

- [ ] REST API operational
- [ ] Async task processing works
- [ ] Unit test coverage > 80%
- [ ] Integration tests pass
- [ ] Accuracy tests show >95% field accuracy
- [ ] CI pipeline runs on every commit

---

# 9. Phase 5: Deployment (Weeks 11-12)

## 9.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2 Weeks (10 days) |
| **Resources** | 2 Engineers + DevOps |
| **Dependencies** | Phase 4 complete |
| **Deliverables** | Production deployment, monitoring, documentation |

## 9.2 Tasks Breakdown

### Task 5.1: HIPAA Compliance (Days 1-4)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 5.1.1 | Audit network calls | 4 hours | Eng 1 |
| 5.1.2 | Implement encryption | 4 hours | Eng 1 |
| 5.1.3 | Setup RBAC | 4 hours | Eng 2 |
| 5.1.4 | Enable audit logging | 4 hours | Eng 2 |
| 5.1.5 | Configure secure cleanup | 2 hours | Eng 1 |
| 5.1.6 | Security review | 4 hours | Both |

### Task 5.2: Monitoring & Alerting (Days 4-7)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 5.2.1 | Setup Prometheus | 3 hours | DevOps |
| 5.2.2 | Create Grafana dashboards | 4 hours | DevOps |
| 5.2.3 | Define metrics | 2 hours | Eng 1 |
| 5.2.4 | Configure alerts | 3 hours | DevOps |
| 5.2.5 | Test alerting | 2 hours | DevOps |

### Task 5.3: Documentation & Training (Days 7-10)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 5.3.1 | Write API documentation | 4 hours | Eng 1 |
| 5.3.2 | Write deployment guide | 3 hours | DevOps |
| 5.3.3 | Write operations runbook | 4 hours | Eng 2 |
| 5.3.4 | Create training materials | 4 hours | Eng 1 |
| 5.3.5 | Conduct training | 4 hours | Both |

### Task 5.4: Production Launch (Days 8-10)

| Subtask | Description | Duration | Owner |
|---------|-------------|----------|-------|
| 5.4.1 | Production environment setup | 4 hours | DevOps |
| 5.4.2 | Deploy application | 2 hours | DevOps |
| 5.4.3 | Pilot testing | 8 hours | Both |
| 5.4.4 | Fix pilot issues | 4 hours | Both |
| 5.4.5 | Go-live | 2 hours | All |

## 9.3 Phase 5 Exit Criteria

- [ ] All data processed locally (no external calls)
- [ ] Encryption at rest enabled
- [ ] Audit logging operational
- [ ] Monitoring dashboards live
- [ ] Alerting tested
- [ ] Documentation complete
- [ ] Team trained
- [ ] Pilot successful
- [ ] Production go-live complete

---

# 10. Resource Requirements

## 10.1 Team

| Role | Count | Duration | Responsibilities |
|------|-------|----------|------------------|
| Lead Engineer | 1 | 12 weeks | Architecture, core components, code review |
| Engineer | 1-2 | 12 weeks | Implementation, testing |
| DevOps | 0.5 | Weeks 11-12 | Deployment, monitoring |
| Product Owner | 0.25 | 12 weeks | Requirements, prioritization |

## 10.2 Hardware

| Item | Specification | Cost Estimate |
|------|---------------|---------------|
| GPU Server | RTX 4090 24GB, 64GB RAM | $3,000-5,000 |
| Development Machines | Standard dev laptops | Existing |
| Storage | 1TB NVMe SSD | $100-200 |

## 10.3 Software

| Item | Type | Cost |
|------|------|------|
| LM Studio | Free | $0 |
| Python & Libraries | Open Source | $0 |
| Redis | Open Source | $0 |
| Monitoring Stack | Open Source | $0 |

---

# 11. Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hallucinations exceed target | Medium | High | 3-layer validation, continuous tuning |
| Poor document quality | High | Medium | Image enhancement, manual preprocessing |
| Model accuracy drift | Medium | Medium | Monitoring, golden dataset regression |
| Hardware failure | Low | High | Checkpointing, backups |
| Timeline overrun | Medium | Medium | 2-week buffer, scope management |
| HIPAA violation | Low | Critical | 100% local processing, audit logging |

---

# 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Field Extraction Accuracy | >95% | Golden dataset comparison |
| Hallucination Rate | <2% | Adversarial test suite |
| Processing Speed | 15-25 sec/page | Benchmark suite |
| VLM Calls per Page | 3-4 | Pipeline metrics |
| System Uptime | >99.5% | Monitoring |
| Human Review Rate | <10% | Production metrics |
| Time to Production | 12 weeks | Project tracking |

---

# Appendix A: Gantt Chart

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
             |----|----|----|----|----|----|----|----|----|----|----|----|
Phase 0      ████
Phase 1           ████████
Phase 2                     ████████████
Phase 3                                   ████████
Phase 4                                             ████████
Phase 5                                                       ████████
Buffer                                                                  ████
```

---

## API Endpoints

```
POST /api/v1/extract      - Submit document for extraction
GET  /api/v1/tasks/{id}   - Get task status
POST /api/v1/batch        - Submit batch extraction
GET  /api/v1/health       - Health check
```

## Key Files

```
src/pipeline/graph.py     - LangGraph workflow
src/agents/               - 4 agent implementations
src/validation/           - Anti-hallucination system
src/api/main.py           - FastAPI application
config/schemas/           - Document schemas
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
