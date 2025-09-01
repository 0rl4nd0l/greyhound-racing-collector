#!/usr/bin/env python3
"""
Memory Watchdog - Proactive memory management to prevent OOM kills
================================================================

This module implements a memory watchdog that monitors RSS usage and takes
corrective actions before the process is killed by the system's OOM killer.
"""

import os
import time
import threading
import logging
import psutil
import gc
import signal

log = logging.getLogger("mem_watchdog")


def start_memory_watchdog(rss_soft_mb=900, rss_hard_mb=1200, check_sec=5, on_soft=None, on_hard=None):
    """
    Start memory watchdog to prevent OOM kills.
    
    Args:
        rss_soft_mb: Soft limit in MB (triggers GC and warnings)
        rss_hard_mb: Hard limit in MB (terminates worker gracefully)  
        check_sec: Check interval in seconds
        on_soft: Optional callback for soft limit exceeded
        on_hard: Optional callback for hard limit exceeded
    """
    if os.environ.get("MEM_WATCHDOG_DISABLED", "0") in ("1", "true"):
        return
        
    proc = psutil.Process(os.getpid())
    rss_soft = rss_soft_mb * 1024 * 1024
    rss_hard = rss_hard_mb * 1024 * 1024
    
    def _run():
        while True:
            try:
                rss = proc.memory_info().rss
                
                if rss > rss_soft:
                    log.warning(f"[WATCHDOG] Soft limit exceeded: RSS={rss/1024/1024:.1f}MB")
                    gc.collect()
                    if on_soft:
                        try:
                            on_soft(rss)
                        except Exception as e:
                            log.error(f"[WATCHDOG] soft callback failed: {e}")
                
                if rss > rss_hard:
                    log.error(f"[WATCHDOG] Hard limit exceeded: RSS={rss/1024/1024:.1f}MB – terminating worker")
                    try:
                        if on_hard:
                            on_hard(rss)
                        # Prefer graceful stop; gunicorn will replace the worker
                        os.kill(os.getpid(), signal.SIGTERM)
                    except Exception:
                        # Last resort: force exit with OOM code
                        os._exit(137)
                        
            except Exception as e:
                log.error(f"[WATCHDOG] error: {e}")
                
            time.sleep(check_sec)
    
    threading.Thread(target=_run, name="MemoryWatchdog", daemon=True).start()
    log.info(f"🛡️ Memory watchdog started: soft={rss_soft_mb}MB, hard={rss_hard_mb}MB")


def clear_caches_on_memory_pressure():
    """Default soft limit handler that clears various caches"""
    try:
        # Clear any global caches that might exist
        import sys
        if hasattr(sys.modules.get('db_performance_optimizer'), '_lazy_loader'):
            loader = sys.modules['db_performance_optimizer']._lazy_loader
            if hasattr(loader, '_cache'):
                loader._cache.clear()
                loader._cache_ttl.clear()
                log.info("[WATCHDOG] Cleared lazy loader caches")
    except Exception as e:
        log.debug(f"[WATCHDOG] cache clear error: {e}")


def kill_orphaned_chrome_processes():
    """Default hard limit handler that kills orphaned Chrome processes"""
    try:
        killed = 0
        for p in psutil.process_iter(["name", "ppid"]):
            name = (p.info.get("name") or "").lower()
            if any(k in name for k in ("chromedriver", "chrome", "chromium")):
                if p.info.get("ppid", 0) in (0, 1):  # Orphaned
                    try:
                        p.kill()
                        killed += 1
                    except Exception:
                        pass
        if killed > 0:
            log.warning(f"[WATCHDOG] Killed {killed} orphaned Chrome processes")
    except Exception as e:
        log.debug(f"[WATCHDOG] chrome cleanup error: {e}")


def get_memory_info():
    """Get current memory information dict"""
    try:
        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        return {
            'rss_mb': mi.rss / 1024 / 1024,
            'vms_mb': mi.vms / 1024 / 1024,
            'threads': proc.num_threads(),
            'percent': psutil.virtual_memory().percent
        }
    except Exception:
        return {}
