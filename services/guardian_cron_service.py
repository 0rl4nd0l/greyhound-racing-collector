#!/usr/bin/env python3
"""
Guardian Cron Service  
===================

Lightweight external service for file integrity monitoring.
Designed to run via cron with ionice to minimize system impact.

Author: AI Assistant
Date: August 4, 2025
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.file_integrity_guardian import FileIntegrityGuardian


class LightweightGuardianService:
    """Lightweight Guardian service optimized for cron execution"""

    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.guardian = FileIntegrityGuardian()
        self.service_name = "guardian-cron"
        self.pid_file = f"/tmp/{self.service_name}.pid"
        self.log_file = f"./logs/{self.service_name}.log"

        # Ensure log directory exists
        os.makedirs("./logs", exist_ok=True)

        # Set ionice priority if available
        self._set_ionice_priority()

    def _load_config(self, config_file: str = None) -> dict:
        """Load service configuration"""
        default_config = {
            "directories": ["./upcoming_races", "./processed"],
            "extensions": [".csv", ".json"],
            "max_file_age_hours": 24,
            "ionice_class": 3,  # Idle class
            "max_execution_time": 300,  # 5 minutes
            "log_level": "INFO",
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                self._log(f"Warning: Failed to load config {config_file}: {e}")

        return default_config

    def _set_ionice_priority(self):
        """Set ionice priority to minimize system impact"""
        try:
            # Check if ionice is available
            subprocess.run(
                ["which", "ionice"], check=True, capture_output=True, text=True
            )

            # Set ionice class to idle (class 3)
            ionice_class = self.config.get("ionice_class", 3)
            subprocess.run(
                ["ionice", "-c", str(ionice_class), "-p", str(os.getpid())],
                check=True,
                capture_output=True,
            )

            self._log(f"Set ionice priority to class {ionice_class} (idle)")

        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("ionice not available - continuing without priority adjustment")

    def _log(self, message: str, level: str = "INFO"):
        """Simple logging to file and stdout"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"

        # Write to log file
        try:
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")
        except Exception:
            pass  # Silently continue if logging fails

        # Print to stdout
        print(log_message)

    def _write_pid_file(self):
        """Write PID file to prevent concurrent execution"""
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(os.getpid()))
            return True
        except Exception as e:
            self._log(f"Failed to write PID file: {e}", "ERROR")
            return False

    def _remove_pid_file(self):
        """Remove PID file"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
        except Exception as e:
            self._log(f"Failed to remove PID file: {e}", "WARNING")

    def _check_concurrent_execution(self) -> bool:
        """Check if another instance is already running"""
        if not os.path.exists(self.pid_file):
            return False

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process is still running
            os.kill(pid, 0)  # This will raise OSError if process doesn't exist
            return True

        except (OSError, ValueError):
            # Process doesn't exist or PID file is corrupted
            self._remove_pid_file()
            return False

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self._log(f"Received signal {signum}, shutting down gracefully")
            self._remove_pid_file()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def run_scan(self) -> dict:
        """Run a single scan cycle"""
        if self._check_concurrent_execution():
            self._log("Another instance is already running, exiting", "WARNING")
            return {"status": "skipped", "reason": "concurrent_execution"}

        if not self._write_pid_file():
            return {"status": "error", "reason": "pid_file_error"}

        try:
            self._setup_signal_handlers()
            start_time = time.time()

            self._log("Starting Guardian scan cycle")

            # Run the scan with timeout protection
            scan_results = []
            total_files_scanned = 0
            total_issues = 0
            files_quarantined = 0

            for directory in self.config["directories"]:
                if not os.path.exists(directory):
                    self._log(f"Skipping non-existent directory: {directory}")
                    continue

                self._log(f"Scanning directory: {directory}")

                # Scan with age filtering
                results = self.guardian.scan_directory(
                    directory,
                    extensions=self.config["extensions"],
                    max_age_hours=self.config["max_file_age_hours"],
                )

                scan_results.extend(results)
                total_files_scanned += len(results)

                for result in results:
                    if result.issues:
                        total_issues += len(result.issues)
                    if result.should_quarantine:
                        files_quarantined += 1

                # Check execution time limit
                elapsed = time.time() - start_time
                if elapsed > self.config["max_execution_time"]:
                    self._log(
                        f"Execution time limit reached ({elapsed:.1f}s), stopping scan",
                        "WARNING",
                    )
                    break

            # Clean up test files
            test_files_removed = self.guardian.cleanup_test_files(
                self.config["directories"],
                max_age_hours=self.config["max_file_age_hours"],
            )

            # Generate summary report
            execution_time = time.time() - start_time
            summary = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_time, 2),
                "statistics": {
                    "files_scanned": total_files_scanned,
                    "issues_found": total_issues,
                    "files_quarantined": files_quarantined,
                    "test_files_removed": test_files_removed,
                },
                "directories_scanned": len(
                    [d for d in self.config["directories"] if os.path.exists(d)]
                ),
            }

            self._log(
                f"Scan completed in {execution_time:.2f}s: "
                f"{total_files_scanned} files, {total_issues} issues, "
                f"{files_quarantined} quarantined, {test_files_removed} test files removed"
            )

            # Save detailed report if there were issues
            if total_issues > 0 or files_quarantined > 0:
                report = self.guardian.generate_integrity_report(scan_results)
                report.update(summary)

                report_file = f"./logs/guardian_cron_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2)

                self._log(f"Detailed report saved to: {report_file}")

            return summary

        except Exception as e:
            self._log(f"Scan failed with error: {e}", "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        finally:
            self._remove_pid_file()

    def health_check(self) -> dict:
        """Quick health check for monitoring"""
        try:
            # Check basic functionality
            guardian = FileIntegrityGuardian()

            # Check if directories exist
            directories_status = {}
            for directory in self.config["directories"]:
                directories_status[directory] = os.path.exists(directory)

            return {
                "status": "healthy",
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "directories": directories_status,
                    "max_file_age_hours": self.config["max_file_age_hours"],
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def main():
    """Main entry point for cron execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight Guardian Cron Service")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--scan", action="store_true", help="Run file integrity scan")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--install-cron", action="store_true", help="Install cron job")
    parser.add_argument("--uninstall-cron", action="store_true", help="Remove cron job")

    args = parser.parse_args()

    service = LightweightGuardianService(config_file=args.config)

    if args.health_check:
        result = service.health_check()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "healthy" else 1

    elif args.scan:
        result = service.run_scan()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1

    elif args.install_cron:
        return install_cron_job(service)

    elif args.uninstall_cron:
        return uninstall_cron_job()

    else:
        # Default: run scan (for cron execution)
        result = service.run_scan()
        return 0 if result["status"] == "completed" else 1


def install_cron_job(service):
    """Install cron job for the Guardian service"""
    try:
        # Get current script path
        script_path = os.path.abspath(__file__)
        python_path = sys.executable

        # Create cron command with ionice if available
        cron_command = f"cd {os.getcwd()} && {python_path} {script_path} --scan"

        # Try to add ionice if available
        try:
            subprocess.run(["which", "ionice"], check=True, capture_output=True)
            cron_command = f"ionice -c 3 {cron_command}"
        except subprocess.CalledProcessError:
            pass

        # Create cron entry (every 4 hours)
        cron_entry = f"0 */4 * * * {cron_command} >> ./logs/guardian-cron.log 2>&1"

        print("Guardian Cron Service Installation")
        print("=" * 40)
        print(f"Service will run every 4 hours")
        print(f"Command: {cron_command}")
        print()
        print("To install manually, add this line to your crontab:")
        print(f"  {cron_entry}")
        print()
        print("Or run: echo '{cron_entry}' | crontab -")

        return 0

    except Exception as e:
        print(f"Failed to install cron job: {e}")
        return 1


def uninstall_cron_job():
    """Remove Guardian cron job"""
    print("To remove the Guardian cron job, run:")
    print("  crontab -l | grep -v 'guardian_cron_service.py' | crontab -")
    return 0


if __name__ == "__main__":
    sys.exit(main())
