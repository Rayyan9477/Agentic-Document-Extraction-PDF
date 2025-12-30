"""
Agents module for document extraction.

Provides the 4-agent architecture for document processing:
- Orchestrator: Workflow control and state management
- Analyzer: Document classification and schema selection
- Extractor: Dual-pass data extraction
- Validator: Quality assurance and hallucination detection

LangChain Integration:
- LangSmith tracing for observability
- RAG-enhanced document memory
- Structured validation tools
- Streaming response handlers
"""

from src.agents.base import (
    BaseAgent,
    AgentError,
    AgentResult,
    AnalysisError,
    ExtractionError,
    ValidationError,
    OrchestrationError,
)
from src.agents.analyzer import AnalyzerAgent
from src.agents.extractor import ExtractorAgent
from src.agents.validator import ValidatorAgent
from src.agents.orchestrator import (
    OrchestratorAgent,
    CheckpointerType,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.agents.langchain_integration import (
    # Configuration
    LangSmithConfig,
    DocumentMemoryConfig,
    # Tracing
    LangSmithTracer,
    TraceSpan,
    # Memory
    DocumentMemory,
    # Tools
    create_validation_tools,
    validate_cpt_code,
    validate_icd10_code,
    validate_npi,
    validate_date_range,
    # Streaming
    StreamEvent,
    StreamMessage,
    StreamingHandler,
    # Mixin
    LangChainAgentMixin,
)
from src.agents.adaptive_extraction import (
    # Core classes
    AdaptiveExtractor,
    AdaptiveExtractionMixin,
    # Data classes
    DiscoveredField,
    DocumentAnalysisResult,
    # Constants
    ENHANCED_FALLBACK_FIELDS,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentError",
    "AgentResult",
    # Error types
    "AnalysisError",
    "ExtractionError",
    "ValidationError",
    "OrchestrationError",
    # Agents
    "AnalyzerAgent",
    "ExtractorAgent",
    "ValidatorAgent",
    "OrchestratorAgent",
    # Workflow
    "CheckpointerType",
    "create_extraction_workflow",
    "generate_processing_id",
    "generate_thread_id",
    # LangChain Integration - Configuration
    "LangSmithConfig",
    "DocumentMemoryConfig",
    # LangChain Integration - Tracing
    "LangSmithTracer",
    "TraceSpan",
    # LangChain Integration - Memory
    "DocumentMemory",
    # LangChain Integration - Tools
    "create_validation_tools",
    "validate_cpt_code",
    "validate_icd10_code",
    "validate_npi",
    "validate_date_range",
    # LangChain Integration - Streaming
    "StreamEvent",
    "StreamMessage",
    "StreamingHandler",
    # LangChain Integration - Mixin
    "LangChainAgentMixin",
    # Adaptive Extraction
    "AdaptiveExtractor",
    "AdaptiveExtractionMixin",
    "DiscoveredField",
    "DocumentAnalysisResult",
    "ENHANCED_FALLBACK_FIELDS",
]
