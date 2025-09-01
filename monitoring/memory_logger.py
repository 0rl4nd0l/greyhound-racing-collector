#!/usr/bin/env python3
"""
Memory Logger - Lightweight background monitoring for memory usage
================================================================

This module provides continuous memory monitoring for the Flask application
to track RSS, VMS, and thread count trends without impacting performance.
"""

import logging
import os
import threading
import time

import psutil

log = logging.getLogger("mem_logger")


def _fmt_mb(bytes_num):
    """Format bytes as MB"""
    return f"{bytes_num/1024/1024:.1f}MB"


def start_memory_logger(interval_sec=10, extra_probes=None):
    """
    Start background memory logger thread.

    Args:
        interval_sec: Logging interval in seconds
        extra_probes: Optional function to call for additional metrics
    """
    if os.environ.get("MEM_LOGGER_DISABLED", "0") in ("1", "true"):
        return

    proc = psutil.Process(os.getpid())

    def _run():
        while True:
            try:
                mi = proc.memory_info()
                threads = proc.num_threads()
                log.info(
                    f"[MEM] RSS={_fmt_mb(mi.rss)} VMS={_fmt_mb(mi.vms)} Threads={threads}"
                )

                if extra_probes:
                    try:
                        extra_probes()
                    except Exception as e:
                        log.warning(f"[MEM] extra_probes error: {e}")

            except Exception as e:
                log.warning(f"[MEM] logger error: {e}")

            time.sleep(interval_sec)

    t = threading.Thread(target=_run, name="MemoryLogger", daemon=True)
    t.start()
    log.info(f"🔍 Memory logger started (interval: {interval_sec}s)")


def log_connection_pool_stats(pool):
    """Log connection pool statistics as extra probe"""
    try:
        if hasattr(pool, "get_stats"):
            stats = pool.get_stats()
            log.info(
                f"[MEM] Pool: active={stats.get('active_connections', 0)} "
                f"created={stats.get('connections_created', 0)} "
                f"reused={stats.get('connections_reused', 0)} "
                f"size={stats.get('pool_size', 0)}"
            )
    except Exception as e:
        log.warning(f"[MEM] pool stats error: {e}")
