#!/usr/bin/env python3
"""
Comprehensive Workflow Logging and Archiving System
==================================================

This module provides structured JSON line logging for every major command and operation,
plus automated archiving of logs older than 30 days.

Features:
- Structured JSON line logging to main_workflow.jsonl
- Automatic archiving of logs older than 30 days
- Command execution tracking
- Model training/prediction tracking
- System operation monitoring
- Comprehensive error logging with context

Author: AI Assistant
Date: August 2025
"""

import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class WorkflowLogger:
    """Comprehensive workflow logging system with archiving"""

    def __init__(
        self, log_dir: str = "./logs", archive_dir: str = "./backups/archives"
    ):
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir)

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Main workflow log file
        self.workflow_log_file = Path("main_workflow.jsonl")

        # Thread safety
        self.lock = threading.Lock()

        # Archive after 30 days
        self.archive_days = 30

        print(f"🚀 Workflow Logger initialized")
        print(f"   Log file: {self.workflow_log_file}")
        print(f"   Archive directory: {self.archive_dir}")

    def log_command(
        self,
        command: Union[str, List[str]],
        operation: str,
        status: str = "started",
        exit_code: Optional[int] = None,
        duration: Optional[float] = None,
        details: Optional[Dict] = None,
        **kwargs,
    ):
        """Log a command execution with structured data"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "command",
            "operation": operation,
            "command": command if isinstance(command, str) else " ".join(command),
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": duration,
            "details": details or {},
            **kwargs,
        }

        self._write_log_entry(entry)

    def log_model_training(
        self,
        model_version: str,
        model_type: str,
        dataset_info: Dict,
        metrics: Dict,
        hyperparameters: Dict,
        artifacts: Dict,
        status: str = "completed",
        **kwargs,
    ):
        """Log model training events"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "model_training",
            "model_version": model_version,
            "model_type": model_type,
            "status": status,
            "dataset_info": dataset_info,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "artifacts": artifacts,
            **kwargs,
        }

        self._write_log_entry(entry)

    def log_prediction(
        self,
        model_version: str,
        race_info: Dict,
        predictions: List[Dict],
        confidence_metrics: Dict,
        status: str = "completed",
        **kwargs,
    ):
        """Log prediction events"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "prediction",
            "model_version": model_version,
            "status": status,
            "race_info": race_info,
            "predictions": predictions,
            "confidence_metrics": confidence_metrics,
            **kwargs,
        }

        self._write_log_entry(entry)

    def log_data_processing(
        self,
        operation: str,
        input_files: List[str],
        output_files: List[str],
        records_processed: int,
        status: str = "completed",
        errors: Optional[List] = None,
        **kwargs,
    ):
        """Log data processing operations"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "data_processing",
            "operation": operation,
            "status": status,
            "input_files": input_files,
            "output_files": output_files,
            "records_processed": records_processed,
            "errors": errors or [],
            **kwargs,
        }

        self._write_log_entry(entry)

    def log_system_event(
        self,
        event_type: str,
        message: str,
        component: str,
        level: str = "INFO",
        details: Optional[Dict] = None,
        **kwargs,
    ):
        """Log system events and operations"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "component": component,
            "level": level,
            "message": message,
            "details": details or {},
            **kwargs,
        }

        self._write_log_entry(entry)

    def log_error(
        self,
        error_type: str,
        message: str,
        component: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None,
        **kwargs,
    ):
        """Log errors with full context"""
        import traceback

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "component": component,
            "message": message,
            "exception": str(exception) if exception else None,
            "exception_type": type(exception).__name__ if exception else None,
            "context": context or {},
            "stack_trace": traceback.format_exc() if exception else None,
            **kwargs,
        }

        self._write_log_entry(entry)

    def _write_log_entry(self, entry: Dict[str, Any]):
        """Thread-safe writing of log entries"""
        with self.lock:
            try:
                with open(self.workflow_log_file, "a", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, default=str)
                    f.write("\n")
            except Exception as e:
                print(f"❌ Error writing workflow log: {e}")

    def execute_and_log(
        self,
        command: Union[str, List[str]],
        operation: str,
        cwd: Optional[str] = None,
        env: Optional[Dict] = None,
        timeout: Optional[int] = None,
        **log_kwargs,
    ) -> subprocess.CompletedProcess:
        """Execute a command and log its execution"""

        start_time = time.time()

        # Log command start
        self.log_command(
            command=command, operation=operation, status="started", **log_kwargs
        )

        try:
            # Execute command
            if isinstance(command, str):
                cmd_list = command.split()
            else:
                cmd_list = command

            result = subprocess.run(
                cmd_list,
                cwd=cwd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
            )

            duration = time.time() - start_time

            # Log command completion
            self.log_command(
                command=command,
                operation=operation,
                status="completed" if result.returncode == 0 else "failed",
                exit_code=result.returncode,
                duration=duration,
                details={
                    "stdout": (
                        result.stdout[:1000] if result.stdout else ""
                    ),  # Limit output
                    "stderr": result.stderr[:1000] if result.stderr else "",
                    "cwd": str(cwd) if cwd else os.getcwd(),
                },
                **log_kwargs,
            )

            return result

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            self.log_error(
                error_type="CommandTimeout",
                message=f"Command timed out after {timeout}s",
                component="workflow_executor",
                exception=e,
                context={
                    "command": command,
                    "operation": operation,
                    "timeout": timeout,
                    "duration": duration,
                },
            )
            raise

        except Exception as e:
            duration = time.time() - start_time
            self.log_error(
                error_type="CommandError",
                message=f"Command execution failed",
                component="workflow_executor",
                exception=e,
                context={
                    "command": command,
                    "operation": operation,
                    "duration": duration,
                },
            )
            raise

    def archive_old_logs(self, dry_run: bool = False) -> Dict[str, Any]:
        """Archive logs older than 30 days"""

        cutoff_date = datetime.now() - timedelta(days=self.archive_days)
        archived_files = []
        total_size_saved = 0

        print(
            f"🗂️  Archiving logs older than {self.archive_days} days (before {cutoff_date.date()})"
        )

        try:
            # Find old log files
            for log_file in self.log_dir.glob("*.log*"):
                if not log_file.is_file():
                    continue

                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

                if file_mtime < cutoff_date:
                    file_size = log_file.stat().st_size

                    if not dry_run:
                        # Create compressed archive
                        archive_name = (
                            f"{log_file.stem}_{file_mtime.strftime('%Y%m%d')}.gz"
                        )
                        archive_path = self.archive_dir / archive_name

                        with open(log_file, "rb") as f_in:
                            with gzip.open(archive_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        # Remove original
                        log_file.unlink()

                        archived_files.append(
                            {
                                "original": str(log_file),
                                "archive": str(archive_path),
                                "size": file_size,
                                "date": file_mtime.isoformat(),
                            }
                        )
                    else:
                        archived_files.append(
                            {
                                "original": str(log_file),
                                "archive": f"would_create_{log_file.stem}_{file_mtime.strftime('%Y%m%d')}.gz",
                                "size": file_size,
                                "date": file_mtime.isoformat(),
                            }
                        )

                    total_size_saved += file_size

            # Log archiving operation
            archive_summary = {
                "files_archived": len(archived_files),
                "total_size_bytes": total_size_saved,
                "total_size_mb": round(total_size_saved / (1024 * 1024), 2),
                "cutoff_date": cutoff_date.isoformat(),
                "dry_run": dry_run,
                "archived_files": archived_files,
            }

            self.log_system_event(
                event_type="log_archiving",
                message=f"{'Simulated' if dry_run else 'Completed'} log archiving",
                component="workflow_logger",
                level="INFO",
                details=archive_summary,
            )

            if not dry_run:
                print(
                    f"✅ Archived {len(archived_files)} log files ({archive_summary['total_size_mb']} MB)"
                )
            else:
                print(
                    f"🔍 Would archive {len(archived_files)} log files ({archive_summary['total_size_mb']} MB)"
                )

            return archive_summary

        except Exception as e:
            self.log_error(
                error_type="ArchivingError",
                message="Failed to archive old logs",
                component="workflow_logger",
                exception=e,
            )
            raise

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about workflow logs"""

        stats = {
            "workflow_log_exists": self.workflow_log_file.exists(),
            "workflow_log_size_bytes": 0,
            "workflow_log_entries": 0,
            "log_directory_size_bytes": 0,
            "archive_directory_size_bytes": 0,
            "oldest_log_date": None,
            "newest_log_date": None,
        }

        try:
            # Workflow log stats
            if self.workflow_log_file.exists():
                stats["workflow_log_size_bytes"] = self.workflow_log_file.stat().st_size

                # Count entries
                with open(self.workflow_log_file, "r") as f:
                    stats["workflow_log_entries"] = sum(1 for line in f if line.strip())

            # Log directory stats
            for log_file in self.log_dir.rglob("*"):
                if log_file.is_file():
                    stats["log_directory_size_bytes"] += log_file.stat().st_size

            # Archive directory stats
            for archive_file in self.archive_dir.rglob("*"):
                if archive_file.is_file():
                    stats["archive_directory_size_bytes"] += archive_file.stat().st_size

            # Date ranges
            if self.workflow_log_file.exists():
                with open(self.workflow_log_file, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        try:
                            first_entry = json.loads(first_line)
                            stats["oldest_log_date"] = first_entry.get("timestamp")
                        except:
                            pass

                # Get last line
                with open(self.workflow_log_file, "rb") as f:
                    try:
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b"\n":
                            f.seek(-2, os.SEEK_CUR)
                        last_line = f.readline().decode().strip()
                        if last_line:
                            last_entry = json.loads(last_line)
                            stats["newest_log_date"] = last_entry.get("timestamp")
                    except:
                        pass

        except Exception as e:
            self.log_error(
                error_type="StatsError",
                message="Error calculating log statistics",
                component="workflow_logger",
                exception=e,
            )

        return stats


# Global workflow logger instance
workflow_logger = WorkflowLogger()


# Convenience functions for common operations
def log_command_execution(command: str, operation: str, **kwargs):
    """Convenience function to log command execution"""
    return workflow_logger.execute_and_log(command, operation, **kwargs)


def log_ml_training(model_version: str, model_type: str, **kwargs):
    """Convenience function to log ML training"""
    return workflow_logger.log_model_training(model_version, model_type, **kwargs)


def log_data_operation(operation: str, **kwargs):
    """Convenience function to log data operations"""
    return workflow_logger.log_data_processing(operation, **kwargs)


def archive_logs(dry_run: bool = False):
    """Convenience function to archive old logs"""
    return workflow_logger.archive_old_logs(dry_run=dry_run)


if __name__ == "__main__":
    """CLI interface for workflow logging operations"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python workflow_logger.py archive [--dry-run]")
        print("  python workflow_logger.py stats")
        print("  python workflow_logger.py test")
        sys.exit(1)

    command = sys.argv[1]

    if command == "archive":
        dry_run = "--dry-run" in sys.argv
        result = archive_logs(dry_run=dry_run)
        print(f"Archive operation completed: {result}")

    elif command == "stats":
        stats = workflow_logger.get_log_statistics()
        print("Workflow Log Statistics:")
        print(json.dumps(stats, indent=2, default=str))

    elif command == "test":
        # Test logging functionality
        print("Testing workflow logger...")

        workflow_logger.log_system_event(
            event_type="test",
            message="Testing workflow logger functionality",
            component="test_script",
        )

        workflow_logger.log_command(
            command="echo 'test'",
            operation="test_command",
            status="completed",
            exit_code=0,
            duration=0.1,
        )

        print("✅ Test entries logged to main_workflow.jsonl")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
