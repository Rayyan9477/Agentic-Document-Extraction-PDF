"""
Configuration module for the document extraction system.

Provides centralized configuration management using Pydantic Settings,
environment variable loading, and validation.
"""

from src.config.logging_config import configure_logging, get_logger, AuditLogger
from src.config.settings import Settings, get_settings, Environment


__all__ = [
    "Settings",
    "get_settings",
    "Environment",
    "configure_logging",
    "get_logger",
    "AuditLogger",
]
