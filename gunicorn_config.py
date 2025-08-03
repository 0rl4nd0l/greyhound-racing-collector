#!/usr/bin/env python3
"""
Gunicorn Configuration for Performance-Optimized Flask App
=========================================================

This configuration optimizes Gunicorn for concurrent processing
with proper worker management and connection pooling.

Workers: (2 * CPU) + 1 = 17 workers for 8 CPU system
Worker Class: gevent for async I/O handling
"""

import os
import multiprocessing

# Server socket
bind = f"127.0.0.1:{os.environ.get('DEFAULT_PORT', '5000')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1  # 17 workers for 8 CPU
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 120
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "greyhound_predictor"

# Server mechanics
daemon = False
pidfile = "logs/gunicorn.pid"
tmp_upload_dir = None

# Performance tuning
preload_app = True
enable_stdio_inheritance = True

# SSL (disabled for local development)
keyfile = None
certfile = None

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("🚀 Gunicorn server is ready - Workers: %d", workers)

def worker_int(worker):
    """Called just after a worker has been killed."""
    worker.log.info("⚠️ Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("🔧 Pre-fork worker setup")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("✅ Worker %s forked (pid: %s)", worker.id, worker.pid)

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info("❌ Worker %s aborted", worker.id)

# Memory optimization
# Use /tmp on macOS since /dev/shm doesn't exist
worker_tmp_dir = "/tmp" if not os.path.exists("/dev/shm") else "/dev/shm"

print(f"""
🚀 Gunicorn Configuration Loaded
================================
🔧 Workers: {workers}
⚡ Worker Class: {worker_class}
🔗 Connections per Worker: {worker_connections}
🌐 Bind Address: {bind}
📊 Max Requests: {max_requests}
⏱️ Timeout: {timeout}s
📁 Logs: {errorlog}, {accesslog}
💾 Worker Temp Dir: {worker_tmp_dir}
""")
