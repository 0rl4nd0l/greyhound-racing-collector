#!/usr/bin/env python3
"""
TGR System Production Configuration
==================================
"""

from tgr_service_scheduler import SchedulerConfig

# Production-optimized configuration
PRODUCTION_CONFIG = SchedulerConfig(
    monitoring_interval=60,           # 1-minute health checks
    enrichment_batch_size=25,         # Larger batch processing
    max_concurrent_jobs=3,            # Moderate concurrency
    performance_threshold=0.85,       # High performance standards
    data_freshness_hours=12,          # Twice-daily data refresh
    auto_retry_failed_jobs=True,      # Enable intelligent retries
    enable_predictive_scheduling=True # Smart workload management
)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_ROTATION_DAYS = 7
MAX_LOG_SIZE_MB = 100

# Performance settings
WORKER_THREAD_COUNT = 2
QUEUE_TIMEOUT_SECONDS = 300
HEALTH_CHECK_TIMEOUT_SECONDS = 30

# Database settings
DB_CONNECTION_TIMEOUT = 10
DB_QUERY_TIMEOUT = 30
ENABLE_DB_OPTIMIZATION = True

# Monitoring settings
ALERT_EMAIL = None  # Configure for email alerts
ALERT_SLACK_WEBHOOK = None  # Configure for Slack alerts
DASHBOARD_REFRESH_INTERVAL = 30
