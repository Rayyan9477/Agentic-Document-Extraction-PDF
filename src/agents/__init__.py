"""
Agents module for document extraction.

Provides the 4-agent architecture for document processing:
- Orchestrator: Workflow control and state management
- Analyzer: Document classification and schema selection
- Extractor: Dual-pass data extraction
- Validator: Quality assurance and hallucination detection

Each agent integrates LangChain/LangGraph directly for:
- LangSmith tracing for observability
- LangGraph state management
- Structured output parsing
"""

from src.agents.analyzer import AnalyzerAgent
from src.agents.base import (
    AgentError,
    AgentResult,
    AnalysisError,
    BaseAgent,
    ExtractionError,
    OrchestrationError,
    ValidationError,
)
from src.agents.extractor import ExtractorAgent
from src.agents.orchestrator import (
    CheckpointerType,
    OrchestratorAgent,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.agents.validator import ValidatorAgent


__all__ = [
    "AgentError",
    "AgentResult",
    "AnalysisError",
    "AnalyzerAgent",
    "BaseAgent",
    "CheckpointerType",
    "ExtractionError",
    "ExtractorAgent",
    "OrchestrationError",
    "OrchestratorAgent",
    "ValidationError",
    "ValidatorAgent",
    "create_extraction_workflow",
    "generate_processing_id",
    "generate_thread_id",
]
