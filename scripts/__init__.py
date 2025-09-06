"""
Scripts package for Greyhound Racing Collector

This package contains shared utilities and scripts for the Greyhound Racing
prediction system. It provides centralized logging, file operations, and
project root resolution functionality.
"""

from .utils import LOG_SCHEMA, forensic_logger, get_project_root, safe_write

__version__ = "1.0.0"
__all__ = ["get_project_root", "forensic_logger", "safe_write", "LOG_SCHEMA"]
