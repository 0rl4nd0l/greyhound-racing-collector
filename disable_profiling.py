#!/usr/bin/env python3
"""
Disable Profiling Module
========================

This module completely disables all Python profiling functionality to prevent
"Another profiling tool is already active" conflicts.

It monkey-patches the cProfile module and other profiling tools to be no-ops.
"""

import sys
import types
from unittest.mock import MagicMock


def disable_all_profiling():
    """
    Completely disable all Python profiling functionality.
    """

    # Mock cProfile
    if "cProfile" in sys.modules:
        cProfile = sys.modules["cProfile"]

        # Create a dummy Profile class
        class DummyProfile:
            def __init__(self, *args, **kwargs):
                pass

            def enable(self, *args, **kwargs):
                pass

            def disable(self, *args, **kwargs):
                pass

            def dump_stats(self, *args, **kwargs):
                pass

            def print_stats(self, *args, **kwargs):
                pass

        # Replace the Profile class
        cProfile.Profile = DummyProfile

        # Mock the run function
        def dummy_run(statement, filename=None, sort=-1):
            exec(statement)

        cProfile.run = dummy_run

    # Mock profile module if it exists
    if "profile" in sys.modules:
        profile = sys.modules["profile"]
        profile.Profile = DummyProfile if "DummyProfile" in locals() else MagicMock

    # Mock pstats
    if "pstats" in sys.modules:
        pstats = sys.modules["pstats"]

        class DummyStats:
            def __init__(self, *args, **kwargs):
                self.total_calls = 0

            def sort_stats(self, *args, **kwargs):
                return self

            def print_stats(self, *args, **kwargs):
                return ""

        pstats.Stats = DummyStats

    # Mock line_profiler if it exists
    if "line_profiler" in sys.modules:
        line_profiler = sys.modules["line_profiler"]
        line_profiler.LineProfiler = MagicMock

    # Mock memory_profiler if it exists
    if "memory_profiler" in sys.modules:
        mem_prof = sys.modules["memory_profiler"]
        mem_prof.profile = lambda func: func  # Return function unchanged

    print("🚫 All profiling functionality has been disabled to prevent conflicts")


# Auto-disable profiling when this module is imported
disable_all_profiling()


# Add stub functions that Flask app expects
def is_profiling():
    """Stub function that always returns False to indicate profiling is disabled"""
    return False


def set_profiling_enabled(enabled):
    """Stub function for setting profiling state - no-op since profiling is disabled"""
    pass


# Add stub classes for Flask app compatibility
class DummyTracker:
    """Dummy tracker for track_sequence context manager"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def profile_function(func):
    """Stub decorator that returns function unchanged"""
    return func


def track_sequence(step_name, component, step_type="processing"):
    """Stub context manager that returns a dummy tracker"""
    return DummyTracker()
