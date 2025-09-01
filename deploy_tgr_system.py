#!/usr/bin/env python3
"""
TGR System Production Deployment
================================

Production deployment script for the TGR enrichment system with
proper configuration, logging, and monitoring.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# Configure production logging
def setup_production_logging():
    """Set up production-grade logging."""

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File handler for all logs
    file_handler = logging.FileHandler(
        logs_dir / f"tgr_system_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


class TGRSystemDeployment:
    """Production deployment manager for TGR system."""

    def __init__(self):
        self.logger = setup_production_logging()
        self.scheduler_process = None
        self.monitoring_thread = None
        self.running = False
        self.start_time = datetime.now()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(
            f"Received signal {signum}, initiating graceful shutdown..."
        )
        self.shutdown()

    def validate_environment(self):
        """Validate the deployment environment."""

        self.logger.info("🔍 Validating deployment environment...")

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required for production deployment")

        # Check required files exist
        required_files = [
            "tgr_monitoring_dashboard.py",
            "tgr_enrichment_service.py",
            "tgr_service_scheduler.py",
            "greyhound_racing_data.db",
        ]

        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            raise RuntimeError(f"Missing required files: {missing_files}")

        # Check database accessibility
        try:
            import sqlite3

            conn = sqlite3.connect("greyhound_racing_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dog_race_data LIMIT 1")
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Database validation failed: {e}")

        # Check optional dependencies
        optional_deps = []
        try:
            import schedule

            optional_deps.append("schedule ✅")
        except ImportError:
            optional_deps.append("schedule ❌")

        try:
            import pandas

            optional_deps.append("pandas ✅")
        except ImportError:
            optional_deps.append("pandas ❌")

        try:
            import requests

            optional_deps.append("requests ✅")
        except ImportError:
            optional_deps.append("requests ❌")

        self.logger.info(f"Dependencies: {', '.join(optional_deps)}")
        self.logger.info("✅ Environment validation complete")

    def create_production_config(self):
        """Create optimized production configuration."""

        self.logger.info("⚙️ Creating production configuration...")

        config_content = '''#!/usr/bin/env python3
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
'''

        with open("production_config.py", "w") as f:
            f.write(config_content)

        self.logger.info("✅ Production configuration created")

    def deploy_system(self):
        """Deploy the complete TGR system in production mode."""

        self.logger.info("🚀 Starting TGR system production deployment...")

        try:
            # Validate environment
            self.validate_environment()

            # Create configuration
            self.create_production_config()

            # Start the intelligent scheduler (main service)
            self.logger.info("🎯 Starting TGR intelligent scheduler...")
            self._start_scheduler()

            # Start monitoring thread
            self.logger.info("📊 Starting system monitoring...")
            self._start_monitoring()

            self.running = True
            self.logger.info("✅ TGR system deployment complete!")

            # Print deployment status
            self._print_deployment_status()

            # Keep main thread alive and handle monitoring
            self._main_loop()

        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            self.shutdown()
            raise

    def _start_scheduler(self):
        """Start the scheduler process."""

        scheduler_script = """
import sys
sys.path.insert(0, ".")

from tgr_service_scheduler import TGRServiceScheduler
from production_config import PRODUCTION_CONFIG
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize and start scheduler
scheduler = TGRServiceScheduler(config=PRODUCTION_CONFIG)

try:
    scheduler.start_scheduler()
    
    # Keep scheduler running
    import time
    while True:
        time.sleep(10)
        
        # Check if scheduler is still healthy
        status = scheduler.get_scheduler_status()
        if not status.get('scheduler_running', False):
            logging.error("Scheduler stopped unexpectedly!")
            break
            
except KeyboardInterrupt:
    logging.info("Scheduler shutdown requested")
    scheduler.stop_scheduler()
except Exception as e:
    logging.error(f"Scheduler error: {e}")
    scheduler.stop_scheduler()
    raise
"""

        # Write scheduler script
        with open("_scheduler_runner.py", "w") as f:
            f.write(scheduler_script)

        # Start scheduler as subprocess
        self.scheduler_process = subprocess.Popen(
            [sys.executable, "_scheduler_runner.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a moment for startup
        time.sleep(3)

        # Check if process started successfully
        if self.scheduler_process.poll() is not None:
            stdout, stderr = self.scheduler_process.communicate()
            raise RuntimeError(f"Scheduler failed to start: {stderr.decode()}")

        self.logger.info("✅ Scheduler process started successfully")

    def _start_monitoring(self):
        """Start the monitoring thread."""

        def monitoring_loop():
            """Continuous monitoring loop."""

            from tgr_monitoring_dashboard import TGRMonitoringDashboard

            monitor = TGRMonitoringDashboard()

            while self.running:
                try:
                    # Generate health report
                    report = monitor.generate_comprehensive_report()

                    # Log key metrics
                    health = report["system_health"]
                    quality = report["data_quality"]

                    self.logger.info(
                        f"Health: {health.get('status', 'unknown')}, "
                        f"Quality: {quality.get('overall_score', 0):.1f}/100"
                    )

                    # Check for critical alerts
                    alerts = report.get("alerts", [])
                    critical_alerts = [
                        a for a in alerts if a.get("level") == "critical"
                    ]

                    if critical_alerts:
                        for alert in critical_alerts:
                            self.logger.error(f"CRITICAL ALERT: {alert.get('message')}")

                    # Export periodic health reports
                    if datetime.now().minute % 15 == 0:  # Every 15 minutes
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                        monitor.export_report(f"logs/health_report_{timestamp}.json")

                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

                # Sleep for monitoring interval
                time.sleep(60)  # 1-minute monitoring cycle

        self.monitoring_thread = threading.Thread(
            target=monitoring_loop, name="TGR-Monitor", daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("✅ Monitoring thread started")

    def _print_deployment_status(self):
        """Print current deployment status."""

        uptime = datetime.now() - self.start_time

        print("\n" + "=" * 60)
        print("🎯 TGR SYSTEM PRODUCTION DEPLOYMENT")
        print("=" * 60)
        print(f"🚀 Status: RUNNING")
        print(f"⏱️ Uptime: {uptime}")
        print(f"📊 Scheduler PID: {self.scheduler_process.pid}")
        print(f"🔍 Monitoring: Active")
        print(f"📂 Logs Directory: logs/")
        print(f"⚙️ Config: production_config.py")
        print("=" * 60)
        print("📋 Management Commands:")
        print("   • View logs: tail -f logs/tgr_system_*.log")
        print("   • Stop system: Ctrl+C (graceful shutdown)")
        print("   • Check status: ps aux | grep tgr")
        print("=" * 60)
        print("🔄 System will run continuously...")
        print("   Press Ctrl+C for graceful shutdown\n")

    def _main_loop(self):
        """Main deployment loop."""

        try:
            while self.running:
                # Check scheduler process health
                if self.scheduler_process and self.scheduler_process.poll() is not None:
                    stdout, stderr = self.scheduler_process.communicate()
                    self.logger.error(f"Scheduler process died: {stderr.decode()}")
                    raise RuntimeError("Scheduler process terminated unexpectedly")

                # Status update every 5 minutes
                uptime = datetime.now() - self.start_time
                if uptime.seconds % 300 == 0:
                    self.logger.info(f"System running healthy, uptime: {uptime}")

                time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the system."""

        if not self.running:
            return

        self.logger.info("🛑 Initiating graceful shutdown...")
        self.running = False

        # Stop scheduler process
        if self.scheduler_process:
            self.logger.info("Stopping scheduler process...")
            self.scheduler_process.terminate()

            # Wait for graceful shutdown
            try:
                self.scheduler_process.wait(timeout=30)
                self.logger.info("✅ Scheduler stopped gracefully")
            except subprocess.TimeoutExpired:
                self.logger.warning("Scheduler shutdown timeout, forcing kill")
                self.scheduler_process.kill()

        # Wait for monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.info("Waiting for monitoring thread...")
            self.monitoring_thread.join(timeout=10)

        # Cleanup temporary files
        temp_files = ["_scheduler_runner.py"]
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass

        uptime = datetime.now() - self.start_time
        self.logger.info(f"✅ TGR system shutdown complete. Total uptime: {uptime}")
        print(f"\n✅ TGR system shutdown complete. Total uptime: {uptime}")


def main():
    """Main deployment function."""

    print("🚀 TGR Data Enrichment System - Production Deployment")
    print("=" * 60)

    deployment = TGRSystemDeployment()

    try:
        deployment.deploy_system()
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
