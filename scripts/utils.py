"""
Shared utilities for Greyhound Racing Collector scripts

This module provides core functionality used across all scripts:
- Project root resolution
- Forensic logging with structured JSON format
- Safe file writing to prevent overwrites
- Centralized JSON schema for log entries
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Centrally defined JSON schema for log lines
LOG_SCHEMA = {
    "type": "object",
    "required": ["action", "status", "ts", "details", "sha256_of_payload"],
    "properties": {
        "action": {
            "type": "string",
            "description": "The action being performed (e.g., 'model_training', 'data_ingestion', 'prediction')",
        },
        "status": {
            "type": "string",
            "enum": ["started", "completed", "failed", "warning", "info"],
            "description": "Status of the action",
        },
        "ts": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp with timezone",
        },
        "details": {
            "type": "object",
            "description": "Additional metadata about the action",
        },
        "sha256_of_payload": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}$",
            "description": "SHA256 hash of the payload/data associated with this action",
        },
    },
    "additionalProperties": False,
}


def get_project_root() -> Path:
    """
    Resolve the project root directory.

    Searches upward from the current file location for common project indicators:
    - .git directory
    - requirements.txt file
    - setup.py file
    - pyproject.toml file

    Returns:
        Path: The project root directory

    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Start from the directory containing this file
    current_path = Path(__file__).resolve().parent

    # Common project root indicators
    root_indicators = [
        ".git",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "app.py",
    ]

    # Search upward through parent directories
    for parent in [current_path] + list(current_path.parents):
        if any((parent / indicator).exists() for indicator in root_indicators):
            return parent

    # Fallback: if no indicators found, assume parent of scripts directory
    if current_path.name == "scripts":
        return current_path.parent

    raise RuntimeError(
        f"Could not determine project root. Searched from {current_path} "
        f"looking for any of: {root_indicators}"
    )


def forensic_logger(action: str, status: str, meta: Dict[str, Any]) -> None:
    """
    Log structured forensic events to project_snapshot.jsonl

    Creates log entries following the mandated schema with ISO timestamps
    and SHA256 hashes of the payload data.

    Args:
        action: The action being performed (e.g., 'model_training', 'data_ingestion')
        status: Status of the action ('started', 'completed', 'failed', 'warning', 'info')
        meta: Additional metadata dictionary to be logged

    Raises:
        ValueError: If status is not in allowed values
        OSError: If log file cannot be written
    """
    allowed_statuses = {"started", "completed", "failed", "warning", "info"}
    if status not in allowed_statuses:
        raise ValueError(f"Status '{status}' not in allowed values: {allowed_statuses}")

    try:
        project_root = get_project_root()
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / "project_snapshot.jsonl"

        # Create ISO timestamp with timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create payload hash
        payload_str = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        # Create log entry following schema
        log_entry = {
            "action": action,
            "status": status,
            "ts": timestamp,
            "details": meta.copy(),  # Create a copy to avoid mutations
            "sha256_of_payload": payload_hash,
        }

        # Append to JSONL file (one JSON object per line)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        # Re-raise with more context
        raise OSError(f"Failed to write forensic log: {e}") from e


def safe_write(path: Union[str, Path], content: str, encoding: str = "utf-8") -> Path:
    """
    Write content to a file safely, avoiding overwrites by adding .preview suffix if file exists.

    Args:
        path: Target file path
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        Path: The actual path where content was written (may have .preview suffix)

    Raises:
        OSError: If file cannot be written
        TypeError: If content is not a string
    """
    if not isinstance(content, str):
        raise TypeError(f"Content must be a string, got {type(content)}")

    target_path = Path(path)

    # If file exists, add .preview suffix
    if target_path.exists():
        target_path = target_path.with_suffix(target_path.suffix + ".preview")

    try:
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        with open(target_path, "w", encoding=encoding) as f:
            f.write(content)

        # Log the write operation
        forensic_logger(
            action="safe_write",
            status="completed",
            meta={
                "original_path": str(path),
                "actual_path": str(target_path),
                "content_length": len(content),
                "encoding": encoding,
                "preview_mode": str(target_path) != str(path),
            },
        )

        return target_path

    except Exception as e:
        # Log the failure
        forensic_logger(
            action="safe_write",
            status="failed",
            meta={
                "original_path": str(path),
                "error": str(e),
                "content_length": len(content),
                "encoding": encoding,
            },
        )
        raise OSError(f"Failed to write file {target_path}: {e}") from e


def validate_log_entry(log_entry: Dict[str, Any]) -> bool:
    """
    Validate a log entry against the LOG_SCHEMA.

    Args:
        log_entry: Dictionary to validate

    Returns:
        bool: True if valid, False otherwise

    Note:
        This is a basic validation. For production, consider using jsonschema library.
    """
    try:
        required_fields = LOG_SCHEMA["required"]
        properties = LOG_SCHEMA["properties"]

        # Check required fields exist
        for field in required_fields:
            if field not in log_entry:
                return False

        # Check status enum
        if log_entry["status"] not in properties["status"]["enum"]:
            return False

        # Check sha256 pattern (64 hex characters)
        sha256_value = log_entry["sha256_of_payload"]
        if not isinstance(sha256_value, str) or len(sha256_value) != 64:
            return False
        if not all(c in "0123456789abcdef" for c in sha256_value.lower()):
            return False

        # Check details is a dict
        if not isinstance(log_entry["details"], dict):
            return False

        return True

    except (KeyError, TypeError, AttributeError):
        return False


# Module-level convenience functions for common logging patterns
def log_start(action: str, **meta) -> None:
    """Log the start of an action."""
    forensic_logger(action, "started", meta)


def log_complete(action: str, **meta) -> None:
    """Log the completion of an action."""
    forensic_logger(action, "completed", meta)


def log_error(action: str, error: Exception, **meta) -> None:
    """Log an error for an action."""
    meta.update({"error_type": type(error).__name__, "error_message": str(error)})
    forensic_logger(action, "failed", meta)


def log_warning(action: str, message: str, **meta) -> None:
    """Log a warning for an action."""
    meta.update({"warning_message": message})
    forensic_logger(action, "warning", meta)


def log_info(action: str, message: str, **meta) -> None:
    """Log informational message for an action."""
    meta.update({"info_message": message})
    forensic_logger(action, "info", meta)
