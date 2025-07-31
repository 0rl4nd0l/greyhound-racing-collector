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
        self.web_log_file = self.log_dir / "web_access.json"

        # Thread-safe logging
        self.lock = threading.Lock()

        # Setup Python logging
        self.setup_python_logging()

        # Initialize web-accessible logs
        self.web_logs = {"process": [], "errors": [], "system": []}

        # Load existing web logs
        self.load_web_logs()

        print(f"📋 Enhanced Logger initialized - Log directory: {self.log_dir}")

    def setup_python_logging(self):
        """Setup Python logging with file handlers"""
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
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

        # Add file handlers
        process_handler = logging.FileHandler(self.process_log_file)
        error_handler = logging.FileHandler(self.error_log_file)

        process_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        error_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        self.process_logger.addHandler(process_handler)
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)

    def load_web_logs(self):
        """Load existing web logs from file"""
        try:
            if self.web_log_file.exists():
                with open(self.web_log_file, "r") as f:
                    self.web_logs = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load web logs: {e}")
            self.web_logs = {"process": [], "errors": [], "system": []}

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
                "message": message,
                "details": details or {},
            }
            self.web_logs["process"].append(log_entry)

            # Keep only last 1000 entries
            if len(self.web_logs["process"]) > 1000:
                self.web_logs["process"] = self.web_logs["process"][-1000:]

        self.save_web_logs()

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
                self.web_logs = {"process": [], "errors": [], "system": []}
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


# Global logger instance
logger = EnhancedLogger()
