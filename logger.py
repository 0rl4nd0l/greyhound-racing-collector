#!/usr/bin/env python3
"""
Enhanced Logging System for Greyhound Racing Collector
=====================================================

This module provides comprehensive logging capabilities with persistent
error tracking, file rotation, and different log levels.

Features:
- Persistent error logs that don't clear between sessions
- Separate log files for different components
- Structured logging with timestamps
- Log rotation to prevent huge files
- Web-accessible log viewing

Author: AI Assistant
Date: July 11, 2025
"""

import argparse
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class EnhancedLogger:
    """Enhanced logging system with persistent storage and web access"""

    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize log files
        self.process_log_file = self.log_dir / "process.log"
        self.error_log_file = self.log_dir / "errors.log"
        self.system_log_file = self.log_dir / "system.log"
        self.workflow_log_file = self.log_dir / "main_workflow.jsonl"
        self.web_log_file = self.log_dir / "web_access.json"
        self.debug_log_file = self.log_dir / "debug.log"

        # Thread-safe logging
        self.lock = threading.Lock()

        # Debug mode state - can be set via --debug flag or DEBUG env var
        self.debug_mode = self._check_debug_mode()
        # Setup Python logging with JSON formatting
        self.setup_python_logging()

        # Add rotating file handler
        self.add_rotating_file_handler()

        # Initialize web-accessible logs
        self.web_logs = {"process": [], "errors": [], "system": [], "debug": []}

        # Load existing web logs
        self.load_web_logs()

        debug_status = "🐛 ENABLED" if self.debug_mode else "DISABLED"
        print(f"📋 Enhanced Logger initialized - Log directory: {self.log_dir}")
        print(f"🔍 Debug mode: {debug_status}")

    def _check_debug_mode(self) -> bool:
        """Determine if debug mode is enabled via env or argument"""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--debug", action="store_true")
        args, _ = parser.parse_known_args()

        # Check environment variable or command-line argument
        return os.getenv("DEBUG", "0") == "1" or args.debug

    def setup_python_logging(self):
        """Setup Python logging with file handlers"""
        # Set debug level if debug mode is enabled
        log_level = logging.DEBUG if self.debug_mode else logging.INFO

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.system_log_file),
                logging.StreamHandler(),
            ],
        )

        # Create specialized loggers
        self.process_logger = logging.getLogger("process")
        self.error_logger = logging.getLogger("errors")
        self.system_logger = logging.getLogger("system")
        self.debug_logger = logging.getLogger("debug")

        # Set debug level for all loggers if debug mode is enabled
        if self.debug_mode:
            self.process_logger.setLevel(logging.DEBUG)
            self.error_logger.setLevel(logging.DEBUG)
            self.system_logger.setLevel(logging.DEBUG)
            self.debug_logger.setLevel(logging.DEBUG)

        # Add file handlers
        process_handler = logging.FileHandler(self.process_log_file)
        error_handler = logging.FileHandler(self.error_log_file)
        debug_handler = logging.FileHandler(self.debug_log_file)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        process_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        debug_handler.setFormatter(formatter)

        self.process_logger.addHandler(process_handler)
        self.error_logger.addHandler(error_handler)
        self.debug_logger.addHandler(debug_handler)

        # Add debug handler to all loggers in debug mode
        if self.debug_mode:
            self.process_logger.addHandler(debug_handler)
            self.error_logger.addHandler(debug_handler)
            self.system_logger.addHandler(debug_handler)

    def add_rotating_file_handler(self):
        """Add a rotating file handler to prevent log files from growing too large"""
        from logging.handlers import RotatingFileHandler

        # Define a rotating handler with a max size of 5MB and up to 3 backup files
        rotating_file_handler = RotatingFileHandler(
            self.system_log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        rotating_file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add this handler to the root logger
        logging.getLogger().addHandler(rotating_file_handler)

    def load_web_logs(self):
        """Load existing web logs from file"""
        try:
            if self.web_log_file.exists():
                with open(self.web_log_file, "r") as f:
                    self.web_logs = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load web logs: {e}")
            self.web_logs = {"process": [], "errors": [], "system": [], "debug": []}

    def save_web_logs(self):
        """Save web logs to file"""
        try:
            with self.lock:
                with open(self.web_log_file, "w") as f:
                    json.dump(self.web_logs, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save web logs: {e}")

    def log_process(
        self, message: str, level: str = "INFO", details: Optional[Dict] = None
    ):
        """Log process-related messages"""
        timestamp = datetime.now().isoformat()

        # Log to file
        if level == "ERROR":
            self.process_logger.error(message)
        elif level == "WARNING":
            self.process_logger.warning(message)
        else:
            self.process_logger.info(message)

        # Add to web logs
        with self.lock:
            log_entry = {
                "timestamp": timestamp,
                "level": level,
                "component": "process",
                "file": str(self.process_log_file),
                "action": details.get("action", "unknown") if details else "unknown",
                "cache_status": (
                    details.get("cache_status", "unknown") if details else "unknown"
                ),
                "validation_errors": (
                    details.get("validation_errors", []) if details else []
                ),
                "outcome": details.get("outcome", "unknown") if details else "unknown",
                "message": message,
                "details": details or {},
            }
            self.web_logs["process"].append(log_entry)

            # Keep only last 1000 entries
            if len(self.web_logs["process"]) > 1000:
                self.web_logs["process"] = self.web_logs["process"][-1000:]

        self.save_web_logs()

        # Also log to main_workflow.jsonl
        workflow_entry = {
            "timestamp": timestamp,
            "level": level,
            "component": "process",
            "file": str(self.workflow_log_file),
            "action": (details or {}).get("action", "unknown"),
            "cache_status": (details or {}).get("cache_status", "unknown"),
            "validation_errors": (details or {}).get("validation_errors", []),
            "outcome": (details or {}).get("outcome", "unknown"),
            "message": message,
            "details": details or {},
        }
        with open(self.workflow_log_file, "a") as f:
            json.dump(workflow_entry, f)
            f.write("\n")

    def log_race_operation(
        self,
        race_date: str,
        venue: str,
        race_number: str,
        operation: str,
        reason: str,
        http_status: Optional[int] = None,
        verbose_fetch: bool = False,
        level: str = "INFO",
    ):
        """Log per-race operations in the specified format: [SKIP|CACHE|FETCH] 2025-07-25 AP_K R4 – reason

        Args:
            race_date: Date in YYYY-MM-DD format
            venue: Venue code (e.g., AP_K)
            race_number: Race number
            operation: One of SKIP, CACHE, or FETCH
            reason: Descriptive reason for the operation
            http_status: HTTP status code (logged when fetches_attempted)
            verbose_fetch: Whether to log at all (conditional logging)
            level: Log level (INFO, WARNING, ERROR)
        """
        # Always log warnings and errors regardless of verbose_fetch
        if level in ["WARNING", "ERROR"] or verbose_fetch:
            # Format the structured log line
            log_line = f"[{operation}] {race_date} {venue} R{race_number} – {reason}"

            # Add HTTP status if provided
            if http_status is not None:
                log_line += f" (HTTP {http_status})"

            # Log to appropriate level
            if level == "ERROR":
                self.process_logger.error(log_line)
            elif level == "WARNING":
                self.process_logger.warning(log_line)
            else:
                self.process_logger.info(log_line)

            # Add to web logs with structured data
            timestamp = datetime.now().isoformat()
            with self.lock:
                log_entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "component": "race_operation",
                    "operation": operation,
                    "race_date": race_date,
                    "venue": venue,
                    "race_number": race_number,
                    "reason": reason,
                    "http_status": http_status,
                    "message": log_line,
                    "structured_format": True,
                }
                self.web_logs["process"].append(log_entry)

                # Keep only last 1000 entries
                if len(self.web_logs["process"]) > 1000:
                    self.web_logs["process"] = self.web_logs["process"][-1000:]

            self.save_web_logs()

            # Also log to main_workflow.jsonl
            workflow_entry = {
                "timestamp": timestamp,
                "type": "race_operation",
                "level": level,
                "operation": operation,
                "race_date": race_date,
                "venue": venue,
                "race_number": race_number,
                "reason": reason,
                "http_status": http_status,
                "message": log_line,
                "verbose_fetch_enabled": verbose_fetch,
            }
            with open(self.workflow_log_file, "a") as f:
                json.dump(workflow_entry, f)
                f.write("\n")

    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[Dict] = None,
    ):
        """Log error messages with full context and structured JSON format"""
        import traceback

        timestamp = datetime.now().isoformat()

        # Format error message
        error_msg = message
        if error:
            error_msg += f" - {str(error)}"

        # Log to file
        self.error_logger.error(error_msg)

        # Create structured error data
        error_data = {
            "timestamp": timestamp,
            "level": "ERROR",
            "message": message,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "context": context or {},
            "stack_trace": None,
        }

        # Add stack trace for exceptions
        if error:
            try:
                error_data["stack_trace"] = traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            except Exception:
                # Fallback if we can't get the full traceback
                error_data["stack_trace"] = traceback.format_exc()

        # Add to web logs
        with self.lock:
            self.web_logs["errors"].append(error_data)

            # Keep only last 500 error entries
            if len(self.web_logs["errors"]) > 500:
                self.web_logs["errors"] = self.web_logs["errors"][-500:]

        self.save_web_logs()

        # Also log exception details for debugging
        if error:
            self.error_logger.exception(f"Exception details for: {message}")

    def log_system(self, message: str, level: str = "INFO", component: str = "SYSTEM"):
        """Log system-level messages"""
        timestamp = datetime.now().isoformat()

        # Log to file
        if level == "ERROR":
            self.system_logger.error(f"[{component}] {message}")
        elif level == "WARNING":
            self.system_logger.warning(f"[{component}] {message}")
        else:
            self.system_logger.info(f"[{component}] {message}")

        # Add to web logs
        with self.lock:
            log_entry = {
                "timestamp": timestamp,
                "level": level,
                "message": message,
                "component": component,
            }
            self.web_logs["system"].append(log_entry)

            # Keep only last 500 system entries
            if len(self.web_logs["system"]) > 500:
                self.web_logs["system"] = self.web_logs["system"][-500:]

        self.save_web_logs()

    def warning(self, message: str, context: Optional[Dict] = None):
        """Log warning messages"""
        self.log_process(message, level="WARNING", details=context)

    def info(self, message: str, context: Optional[Dict] = None):
        """Log info messages"""
        self.log_process(message, level="INFO", details=context)

    def debug(self, message: str, context: Optional[Dict] = None):
        """Log debug messages"""
        if self.debug_mode:
            # Log to file with debug logger
            self.debug_logger.debug(message)

            # Add to web logs if in debug mode
            timestamp = datetime.now().isoformat()

            with self.lock:
                log_entry = {
                    "timestamp": timestamp,
                    "level": "DEBUG",
                    "component": "debug",
                    "message": message,
                    "details": context or {},
                }
                self.web_logs["debug"].append(log_entry)

                # Keep only last 1000 debug entries
                if len(self.web_logs["debug"]) > 1000:
                    self.web_logs["debug"] = self.web_logs["debug"][-1000:]

            self.save_web_logs()

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error messages (compatibility method)"""
        if exc_info:
            self.exception(message, context=kwargs)
        else:
            self.log_error(message, context=kwargs)

    def get_web_logs(self, log_type: str = "all", limit: int = 100) -> List[Dict]:
        """Get logs for web display"""
        with self.lock:
            if log_type == "all":
                # Combine all logs and sort by timestamp
                all_logs = []
                for log_cat in self.web_logs.values():
                    all_logs.extend(log_cat)
                all_logs.sort(key=lambda x: x["timestamp"], reverse=True)
                return all_logs[:limit]
            elif log_type in self.web_logs:
                return self.web_logs[log_type][-limit:]
            else:
                return []

    def clear_logs(self, log_type: str = "all"):
        """Clear specific log types"""
        with self.lock:
            if log_type == "all":
                self.web_logs = {"process": [], "errors": [], "system": [], "debug": []}
            elif log_type in self.web_logs:
                self.web_logs[log_type] = []

        self.save_web_logs()

    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of log statistics"""
        with self.lock:
            summary = {
                "total_process_logs": len(self.web_logs["process"]),
                "total_error_logs": len(self.web_logs["errors"]),
                "total_system_logs": len(self.web_logs["system"]),
                "recent_errors": len(
                    [
                        log
                        for log in self.web_logs["errors"]
                        if (
                            datetime.now() - datetime.fromisoformat(log["timestamp"])
                        ).days
                        < 1
                    ]
                ),
                "log_files": {
                    "process_log": str(self.process_log_file),
                    "error_log": str(self.error_log_file),
                    "system_log": str(self.system_log_file),
                },
            }
        return summary

    def exception(self, message: str, context: Optional[Dict] = None):
        """Log exception with full stack trace (alias for log_error with current exception)"""
        import sys

        exc_info = sys.exc_info()
        if exc_info[1]:
            self.log_error(message, error=exc_info[1], context=context)
        else:
            self.log_error(message, context=context)

    def create_structured_error(
        self,
        message: str,
        error_code: str = None,
        additional_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a structured JSON error response for API endpoints"""
        structured_error = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "error_code": error_code,
            "additional_data": additional_data or {},
        }

        # Log the structured error
        self.log_error(
            f"Structured error created: {message}",
            context={"error_code": error_code, "additional_data": additional_data},
        )

        return structured_error


class KeyMismatchLogger:
    """Logger adapter for capturing KeyErrors related to dog names."""

    def __init__(self, logger):
        self.logger = logger

    def log_key_error(
        self, error_context=None, dog_record=None, error=None, file_path=None
    ):
        """Logs KeyError with detailed context including dog record and race file path.

        Supports both old and new interfaces:
        - New: log_key_error(error_context={...}, dog_record={...})
        - Old: log_key_error(error, dog_record, file_path)
        """
        # Handle new enhanced interface
        if error_context is not None and isinstance(error_context, dict):
            message = (
                f"KeyError in {error_context.get('operation', 'unknown operation')}"
            )

            # Extract information from error_context
            race_file_path = error_context.get("race_file_path", "unknown")
            missing_key = error_context.get("missing_key", "unknown")
            available_keys = error_context.get("available_keys", [])
            step = error_context.get("step", "unknown")
            operation = error_context.get("operation", "unknown")

            # Log with comprehensive context
            self.logger.log_error(
                message,
                context={
                    "operation": operation,
                    "race_file_path": race_file_path,
                    "missing_key": missing_key,
                    "available_keys": available_keys,
                    "step": step,
                    "dog_record": dog_record or error_context.get("dog_record", {}),
                    "error_type": "KeyError",
                    "enhanced_logging": True,
                },
            )

            # Also print detailed error information for immediate visibility
            print(f"\n🚨 KEYERROR DETECTED:")
            print(f"   Operation: {operation}")
            print(f"   Race File: {race_file_path}")
            print(f"   Missing Key: {missing_key}")
            print(f"   Available Keys: {available_keys}")
            print(f"   Step: {step}")
            if dog_record:
                print(f"   Dog Record: {dog_record}")
            print(f"   Stack trace logged to error files\n")

        # Handle old interface for backward compatibility
        elif error is not None and dog_record is not None:
            self.logger.log_error(
                "KeyError encountered in dog names",
                error=error,
                context={
                    "dog_record": dog_record,
                    "race_file_path": file_path or "unknown",
                    "legacy_interface": True,
                },
            )

        else:
            # Fallback logging
            self.logger.log_error(
                "KeyError logged with insufficient context",
                context={
                    "error_context": error_context,
                    "dog_record": dog_record,
                    "error": str(error) if error else None,
                    "file_path": file_path,
                    "insufficient_context": True,
                },
            )


# Global logger instance
logger = EnhancedLogger()
# Global KeyMismatchLogger instance
key_mismatch_logger = KeyMismatchLogger(logger)
