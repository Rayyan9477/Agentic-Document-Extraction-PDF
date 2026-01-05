"""
Configuration module for the document extraction system.

Provides centralized configuration management using Pydantic Settings,
environment variable loading, and validation.
"""

from src.config.logging_config import AuditLogger, configure_logging, get_logger
from src.config.settings import Environment, Settings, get_settings


__all__ = [
    "AuditLogger",
    "Environment",
    "Settings",
    "configure_logging",
    "get_logger",
    "get_settings",
]
