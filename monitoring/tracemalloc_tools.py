#!/usr/bin/env python3
"""
Tracemalloc Tools - Detailed heap analysis for memory leak detection
================================================================

This module provides utilities for capturing and analyzing memory allocation
patterns using Python's tracemalloc module.
"""

import tracemalloc
import time
import os
import pathlib
import logging
import signal

log = logging.getLogger("tracemalloc_tools")


def init_tracemalloc():
    """Initialize tracemalloc if enabled via environment"""
    if os.environ.get("TRACE_MALLOC", "1") in ("1", "true"):
        tracemalloc.start(25)  # Store up to 25 frames per allocation
        log.info("🔬 tracemalloc initialized (25 frames)")
        return True
    return False


def dump_top_allocs(limit=25, label="snapshot"):
    """
    Take a tracemalloc snapshot and dump top allocations to file.
    
    Args:
        limit: Number of top allocations to dump
        label: Label for the dump file
        
    Returns:
        Path to the dump file
    """
    if not tracemalloc.is_tracing():
        raise RuntimeError("tracemalloc is not enabled")
        
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    # Create output directory
    ts = int(time.time())
    out = pathlib.Path(os.getenv("TRACE_DIR", "./trace"))
    out.mkdir(exist_ok=True)
    
    # Write dump file
    dump_path = out / f"tracemalloc_{label}_{ts}.log"
    with dump_path.open("w") as f:
        f.write(f"Tracemalloc snapshot: {label} at {time.ctime(ts)}\n")
        f.write(f"Total allocations: {len(top_stats)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, stat in enumerate(top_stats[:limit]):
            f.write(f"#{i+1}: {stat}\n")
            # Add frame details for better debugging
            for frame in stat.traceback:
                f.write(f"    {frame}\n")
            f.write("\n")
    
    log.info(f"📊 Tracemalloc dump written: {dump_path} ({limit} entries)")
    return str(dump_path)


def compare_snapshots(snapshot1, snapshot2, limit=10):
    """
    Compare two tracemalloc snapshots and return top differences.
    
    Args:
        snapshot1: First snapshot (baseline)
        snapshot2: Second snapshot (current)
        limit: Number of top differences to return
        
    Returns:
        List of top differences
    """
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    differences = []
    for i, stat in enumerate(top_stats[:limit]):
        differences.append({
            'rank': i + 1,
            'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
            'size_diff_mb': stat.size_diff / 1024 / 1024,
            'count_diff': stat.count_diff,
            'size_mb': stat.size / 1024 / 1024,
            'count': stat.count
        })
    
    return differences


def setup_sigusr1_handler():
    """
    Setup SIGUSR1 signal handler to trigger heap dumps.
    Usage: kill -USR1 <pid>
    """
    def _sigusr1_handler(signum, frame):
        try:
            path = dump_top_allocs(limit=50, label="sigusr1")
            log.warning(f"[MEM] tracemalloc dump triggered via SIGUSR1: {path}")
        except Exception as e:
            log.error(f"[MEM] dump failed: {e}")
    
    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        log.info("📡 SIGUSR1 handler registered for heap dumps")
    except Exception as e:
        log.debug(f"SIGUSR1 handler setup failed: {e}")


def get_current_memory_usage():
    """Get current tracemalloc statistics"""
    if not tracemalloc.is_tracing():
        return {}
        
    current, peak = tracemalloc.get_traced_memory()
    return {
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024,
        'tracing': True
    }


def take_baseline_snapshot():
    """Take and store a baseline snapshot for later comparison"""
    if not tracemalloc.is_tracing():
        return None
        
    snapshot = tracemalloc.take_snapshot()
    
    # Store baseline in module-level variable for comparison
    global _baseline_snapshot
    _baseline_snapshot = snapshot
    
    log.info("📷 Baseline tracemalloc snapshot captured")
    return snapshot


def compare_to_baseline(limit=10):
    """Compare current state to stored baseline snapshot"""
    global _baseline_snapshot
    
    if not tracemalloc.is_tracing():
        return []
        
    if not hasattr(globals(), '_baseline_snapshot') or _baseline_snapshot is None:
        log.warning("No baseline snapshot available")
        return []
        
    current_snapshot = tracemalloc.take_snapshot()
    return compare_snapshots(_baseline_snapshot, current_snapshot, limit)


# Module-level baseline storage
_baseline_snapshot = None
