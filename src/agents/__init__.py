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
from src.agents.component_detector import ComponentDetectorAgent
from src.agents.extractor import ExtractorAgent
from src.agents.layout_agent import LayoutAgent
from src.agents.orchestrator import (
    CheckpointerType,
    OrchestratorAgent,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.agents.schema_generator import SchemaGeneratorAgent
from src.agents.schema_proposal import SchemaProposalAgent
from src.agents.splitter import SplitterAgent
from src.agents.table_detector import TableDetectorAgent
from src.agents.validator import ValidatorAgent


__all__ = [
    "AgentError",
    "AgentResult",
    "AnalysisError",
    "AnalyzerAgent",
    "BaseAgent",
    "CheckpointerType",
    "ComponentDetectorAgent",
    "ExtractionError",
    "ExtractorAgent",
    "LayoutAgent",
    "OrchestrationError",
    "OrchestratorAgent",
    "SchemaGeneratorAgent",
    "SchemaProposalAgent",
    "SplitterAgent",
    "TableDetectorAgent",
    "ValidationError",
    "ValidatorAgent",
    "create_extraction_workflow",
    "generate_processing_id",
    "generate_thread_id",
]
