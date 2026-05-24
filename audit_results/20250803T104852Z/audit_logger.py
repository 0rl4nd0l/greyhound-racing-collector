#!/usr/bin/env python3
"""
Audit Logger - Structured logging initialization for the bootstrap audit process
Created: 2025-08-03T10:48:52Z
Purpose: Initialize DEBUG level logging for full traceability during audit
"""

import logging
import os
import sys
from datetime import datetime, timezone


def setup_audit_logging(audit_ts=None):
    """Setup structured logging for audit process with DEBUG level"""

    if audit_ts is None:
        audit_ts = os.environ.get(
            "AUDIT_TS", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )

    # Create log file path
    log_dir = f"audit_results/{audit_ts}"
    log_file = f"{log_dir}/audit.log"

    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("audit_logger")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter for structured logging
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler for DEBUG level
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler for INFO level and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Log initialization
    logger.info(f"Audit logging initialized for session: {audit_ts}")
    logger.debug(f"Log file: {log_file}")
    logger.debug(f"Environment bootstrap and baseline capture started")

    return logger


if __name__ == "__main__":
    # Initialize logging when run directly
    logger = setup_audit_logging()
    logger.info("Audit logger standalone initialization complete")
    logger.debug("This is a debug message to test logging functionality")
