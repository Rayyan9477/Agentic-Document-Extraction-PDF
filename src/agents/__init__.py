"""
Agents module for document extraction.

Provides the 4-agent architecture for document processing:
- Orchestrator: Workflow control and state management
- Analyzer: Document classification and schema selection
- Extractor: Dual-pass data extraction
- Validator: Quality assurance and hallucination detection
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
]
