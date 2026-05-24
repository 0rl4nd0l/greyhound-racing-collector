#!/usr/bin/env python3
"""
Logging Configuration for Greyhound Racing Collector
===================================================

This module provides a structured logging configuration that routes 
INFO/DEBUG messages to appropriate sub-logs while maintaining the 
current JSONL format for compatibility with existing systems.

Directory Structure:
- logs/prediction/: Prediction-related logs (ML predictions, model inference)
- logs/test/: Test execution logs (unit tests, integration tests)
- logs/qa/: Quality Assurance logs (data validation, integrity checks)
- gpt_assistant/: Code and assistant-related operations
- config/: Configuration and setup logs

Author: AI Assistant
Date: August 4, 2025
"""

import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class JSONLFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSONL format matching the current system"""

    def __init__(self, component: str = "system"):
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSONL matching existing format"""

        # Extract additional fields from record if they exist
        details = getattr(record, "details", {})
        action = getattr(record, "action", "log_entry")
        cache_status = getattr(record, "cache_status", "unknown")
        validation_errors = getattr(record, "validation_errors", [])
        outcome = getattr(record, "outcome", "logged")

        # Create structured log entry matching main_workflow.jsonl format
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": self.component,
            "file": getattr(record, "log_filename", record.pathname),
            "action": action,
            "cache_status": cache_status,
            "validation_errors": validation_errors,
            "outcome": outcome,
            "message": record.getMessage(),
            "details": details,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class ComponentLogger:
    """Logger that routes messages to appropriate component-specific log files"""

    def __init__(self, name: str, log_dir: str = "./logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.lock = threading.Lock()

        # Ensure log directories exist
        self.log_dir.mkdir(exist_ok=True)
        (self.log_dir / "prediction").mkdir(exist_ok=True)
        (self.log_dir / "test").mkdir(exist_ok=True)
        (self.log_dir / "qa").mkdir(exist_ok=True)
        Path("./gpt_assistant").mkdir(exist_ok=True)
        Path("./config").mkdir(exist_ok=True)

        # Component to directory mapping
        self.component_dirs = {
            "prediction": self.log_dir / "prediction",
            "ml_system": self.log_dir / "prediction",
            "model_training": self.log_dir / "prediction",
            "inference": self.log_dir / "prediction",
            "test": self.log_dir / "test",
            "testing": self.log_dir / "test",
            "qa": self.log_dir / "qa",
            "validation": self.log_dir / "qa",
            "data_integrity": self.log_dir / "qa",
            "gpt_assistant": Path("./gpt_assistant"),
            "assistant": Path("./gpt_assistant"),
            "config": Path("./config"),
            "configuration": Path("./config"),
        }

        # Setup loggers for each component
        self.loggers = {}
        self._setup_loggers()

    def _setup_loggers(self):
        """Setup individual loggers for each component"""

        for component, log_dir in self.component_dirs.items():
            logger = logging.getLogger(f"{self.name}.{component}")
            logger.setLevel(logging.DEBUG)

            # Remove existing handlers to avoid duplicates
            logger.handlers.clear()

            # Create component-specific log file
            log_file = log_dir / f"{component}.jsonl"

            # Add rotating file handler to prevent large files
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )

            # Use JSONL formatter
            formatter = JSONLFormatter(component=component)
            handler.setFormatter(formatter)

            logger.addHandler(handler)

            # Also add to main workflow log for compatibility
            main_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "main_workflow.jsonl",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10,
            )
            main_formatter = JSONLFormatter(component=component)
            main_handler.setFormatter(main_formatter)
            logger.addHandler(main_handler)

            self.loggers[component] = logger

            # Don't propagate to root logger to avoid duplication
            logger.propagate = False

    def _get_component_from_context(self, **kwargs) -> str:
        """Determine component from context clues"""

        # Check explicit component
        if "component" in kwargs:
            return kwargs["component"]

        # Check action/operation for hints
        action = kwargs.get("action", "").lower()
        details = kwargs.get("details", {})

        # Prediction-related keywords
        if any(
            keyword in action
            for keyword in ["predict", "model", "train", "ml", "inference", "calibrat"]
        ):
            return "prediction"

        # Test-related keywords
        if any(
            keyword in action for keyword in ["test", "validate", "assert", "check"]
        ):
            return "test"

        # QA-related keywords
        if any(
            keyword in action for keyword in ["quality", "integrity", "audit", "verify"]
        ):
            return "qa"

        # GPT/Assistant keywords
        if any(keyword in action for keyword in ["gpt", "assistant", "ai", "enhance"]):
            return "gpt_assistant"

        # Config keywords
        if any(keyword in action for keyword in ["config", "setup", "init"]):
            return "config"

        # Check details for more context
        if isinstance(details, dict):
            detail_str = str(details).lower()
            if any(keyword in detail_str for keyword in ["model", "predict", "train"]):
                return "prediction"
            elif any(keyword in detail_str for keyword in ["test", "validate"]):
                return "test"
            elif any(keyword in detail_str for keyword in ["quality", "integrity"]):
                return "qa"

        # Default to config for unknown
        return "config"

    def log(self, level: str, message: str, **kwargs):
        """Log message to appropriate component log"""

        # Determine target component
        component = self._get_component_from_context(**kwargs)

        # Get or create logger for component
        if component not in self.loggers:
            component = "config"  # fallback

        logger = self.loggers[component]

        # Create log record with extra fields
        extra_fields = {
            "details": kwargs.get("details", {}),
            "action": kwargs.get("action", "log_entry"),
            "cache_status": kwargs.get("cache_status", "unknown"),
            "validation_errors": kwargs.get("validation_errors", []),
            "outcome": kwargs.get("outcome", "logged"),
            "log_filename": kwargs.get("filename", component + ".jsonl"),
        }

        # Log at appropriate level
        if level.upper() == "DEBUG":
            logger.debug(message, extra=extra_fields)
        elif level.upper() == "INFO":
            logger.info(message, extra=extra_fields)
        elif level.upper() == "WARNING":
            logger.warning(message, extra=extra_fields)
        elif level.upper() == "ERROR":
            logger.error(message, extra=extra_fields)
        elif level.upper() == "CRITICAL":
            logger.critical(message, extra=extra_fields)
        else:
            logger.info(message, extra=extra_fields)

    def info(self, message: str, **kwargs):
        """Log INFO level message"""
        self.log("INFO", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log DEBUG level message"""
        self.log("DEBUG", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log WARNING level message"""
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log ERROR level message"""
        self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log CRITICAL level message"""
        self.log("CRITICAL", message, **kwargs)


class StructuredLoggingConfig:
    """Main configuration class for structured logging system"""

    def __init__(self, log_dir: str = "./logs", debug_mode: bool = None):
        self.log_dir = Path(log_dir)
        self.debug_mode = (
            debug_mode if debug_mode is not None else self._check_debug_mode()
        )

        # Initialize component logger
        self.component_logger = ComponentLogger("greyhound_racing", str(self.log_dir))

        # Setup root logger configuration
        self._setup_root_logger()

        print(f"📋 Structured Logging initialized - Directory: {self.log_dir}")
        print(f"🔍 Debug mode: {'🐛 ENABLED' if self.debug_mode else 'DISABLED'}")
        print(f"📁 Component directories: prediction, test, qa, gpt_assistant, config")

    def _check_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--debug", action="store_true")
        args, _ = parser.parse_known_args()

        return os.getenv("DEBUG", "0") == "1" or args.debug

    def _setup_root_logger(self):
        """Setup root logger with appropriate level"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

    def get_logger(self, component: str = None) -> ComponentLogger:
        """Get component logger instance"""
        return self.component_logger

    def log_prediction(self, message: str, level: str = "INFO", **kwargs):
        """Log prediction-related message"""
        kwargs["component"] = "prediction"
        self.component_logger.log(level, message, **kwargs)

    def log_test(self, message: str, level: str = "INFO", **kwargs):
        """Log test-related message"""
        kwargs["component"] = "test"
        self.component_logger.log(level, message, **kwargs)

    def log_qa(self, message: str, level: str = "INFO", **kwargs):
        """Log QA-related message"""
        kwargs["component"] = "qa"
        self.component_logger.log(level, message, **kwargs)

    def log_gpt_assistant(self, message: str, level: str = "INFO", **kwargs):
        """Log GPT assistant-related message"""
        kwargs["component"] = "gpt_assistant"
        self.component_logger.log(level, message, **kwargs)

    def log_config(self, message: str, level: str = "INFO", **kwargs):
        """Log configuration-related message"""
        kwargs["component"] = "config"
        self.component_logger.log(level, message, **kwargs)

    def create_workflow_entry(self, **kwargs) -> Dict[str, Any]:
        """Create workflow entry compatible with existing main_workflow.jsonl format"""
        return {
            "timestamp": datetime.now().isoformat(),
            "level": kwargs.get("level", "INFO"),
            "component": kwargs.get("component", "system"),
            "file": kwargs.get("file", "main_workflow.jsonl"),
            "action": kwargs.get("action", "workflow_step"),
            "cache_status": kwargs.get("cache_status", "unknown"),
            "validation_errors": kwargs.get("validation_errors", []),
            "outcome": kwargs.get("outcome", "completed"),
            "details": kwargs.get("details", {}),
            "message": kwargs.get("message", ""),
        }


# Global instance for easy import
structured_logging = StructuredLoggingConfig()


# Convenience functions for backward compatibility
def log_prediction(message: str, level: str = "INFO", **kwargs):
    """Log prediction message"""
    structured_logging.log_prediction(message, level, **kwargs)


def log_test(message: str, level: str = "INFO", **kwargs):
    """Log test message"""
    structured_logging.log_test(message, level, **kwargs)


def log_qa(message: str, level: str = "INFO", **kwargs):
    """Log QA message"""
    structured_logging.log_qa(message, level, **kwargs)


def log_gpt_assistant(message: str, level: str = "INFO", **kwargs):
    """Log GPT assistant message"""
    structured_logging.log_gpt_assistant(message, level, **kwargs)


def log_config(message: str, level: str = "INFO", **kwargs):
    """Log configuration message"""
    structured_logging.log_config(message, level, **kwargs)


def get_component_logger() -> ComponentLogger:
    """Get the component logger instance"""
    return structured_logging.get_logger()


if __name__ == "__main__":
    # Example usage demonstrating the logging system
    print("🧪 Testing structured logging configuration...")

    # Test different component logs
    log_prediction(
        "Model training initiated",
        level="INFO",
        action="train_model",
        outcome="started",
        details={"model_type": "ExtraTreesClassifier", "version": "v4"},
    )

    log_test(
        "Running unit tests",
        level="INFO",
        action="run_tests",
        outcome="in_progress",
        details={"test_suite": "comprehensive", "total_tests": 150},
    )

    log_qa(
        "Data integrity check completed",
        level="INFO",
        action="integrity_check",
        outcome="success",
        details={"records_validated": 10000, "errors_found": 0},
    )

    log_gpt_assistant(
        "Processing GPT enhancement request",
        level="INFO",
        action="gpt_enhance",
        outcome="processing",
        details={"request_type": "prediction_analysis"},
    )

    log_config(
        "System configuration updated",
        level="INFO",
        action="config_update",
        outcome="success",
        details={"config_file": "logging_config.py"},
    )

    print(
        "✅ Structured logging test completed - check log files in respective directories"
    )
