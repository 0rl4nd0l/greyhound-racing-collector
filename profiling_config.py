#!/usr/bin/env python3
"""
Global Profiling Configuration Module
=====================================

Centralized profiling configuration to avoid import cycles and provide
zero-overhead profiling control.

This module provides:
- Global PROFILING_ENABLED flag
- is_profiling() helper function
- Thread-safe profiling state management

Author: AI Assistant
Date: January 2025
"""

import threading
from typing import Optional

# Global profiling state - COMPLETELY DISABLED
_profiling_enabled = False  # Always False to prevent conflicts
_profiling_lock = threading.Lock()


def set_profiling_enabled(enabled: bool) -> None:
    """
    Set the global profiling enabled state - DISABLED to prevent conflicts.
    
    Args:
        enabled: Whether profiling should be enabled (ignored - always disabled)
    """
    global _profiling_enabled
    with _profiling_lock:
        _profiling_enabled = False  # Always disabled to prevent conflicts


def is_profiling() -> bool:
    """
    Check if profiling is currently enabled - ALWAYS FALSE to prevent conflicts.
    
    This function is designed to be fast and thread-safe to avoid
    any performance overhead when profiling is disabled.
    
    Returns:
        Always False to prevent profiling conflicts
    """
    return False  # Always False to prevent conflicts


def get_profiling_status() -> dict:
    """
    Get comprehensive profiling status information.
    
    Returns:
        Dictionary with profiling status and configuration
    """
    return {
        "enabled": _profiling_enabled,
        "thread_safe": True,
        "overhead": "zero when disabled",
        "implementation": "global flag with thread safety"
    }


# Backwards compatibility - expose the flag directly
@property
def PROFILING_ENABLED():
    """
    Property accessor for the global profiling flag.
    
    This provides backwards compatibility while maintaining
    thread safety through the is_profiling() function.
    """
    return is_profiling()


# Initialize profiling as PERMANENTLY disabled to prevent conflicts
_profiling_enabled = False  # Permanently disabled

if __name__ == "__main__":
    print("🔧 Profiling Configuration Module")
    print(f"Initial state: {get_profiling_status()}")
    
    # Test the API
    print("\nTesting profiling API:")
    print(f"is_profiling(): {is_profiling()}")
    
    set_profiling_enabled(True)
    print(f"After enabling: {is_profiling()}")
    
    set_profiling_enabled(False)
    print(f"After disabling: {is_profiling()}")
