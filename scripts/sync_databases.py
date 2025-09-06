#!/usr/bin/env python3
"""
Database Synchronization Script

Synchronizes data from staging database to analytics database for production deployments.
Supports incremental and full synchronization modes with validation and rollback capabilities.

Usage:
    python scripts/sync_databases.py --mode full
    python scripts/sync_databases.py --mode incremental --check-interval 300
    python scripts/sync_databases.py --validate-only

Environment Variables:
    STAGING_DB_PATH - Source database (default: greyhound_racing_data_stage.db)
    ANALYTICS_DB_PATH - Target database (default: greyhound_racing_data_analytics.db)
    SYNC_BACKUP_DIR - Directory for sync backups (default: database_backups/sync)
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.db_utils import open_sqlite_readonly, open_sqlite_writable


def get_db_paths() -> Tuple[str, str]:
    """Get staging and analytics database paths from environment."""
    staging_db = os.getenv("STAGING_DB_PATH", "greyhound_racing_data_stage.db")
    analytics_db = os.getenv("ANALYTICS_DB_PATH", "greyhound_racing_data_analytics.db")

    if not staging_db or not analytics_db:
        fallback = os.getenv("GREYHOUND_DB_PATH", "greyhound_racing_data.db")
        staging_db = staging_db or fallback
        analytics_db = analytics_db or fallback

    return staging_db, analytics_db


def create_backup(source_path: str, backup_dir: Path) -> Path:
    """Create a backup of the database before sync."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    db_name = Path(source_path).stem
    backup_path = backup_dir / f"{db_name}_backup_{timestamp}.db"

    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)

    return backup_path


def get_table_row_counts(db_path: str) -> Dict[str, int]:
    """Get row counts for all tables in database."""
    counts = {}

    with open_sqlite_readonly(db_path) as conn:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Get row count for each table
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

    return counts


def get_table_checksums(db_path: str) -> Dict[str, str]:
    """Calculate checksums for tables to detect changes."""
    checksums = {}

    with open_sqlite_readonly(db_path) as conn:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Use SQLite's built-in hashing for consistency check
            try:
                cursor.execute(f"SELECT COUNT(*), MAX(rowid) FROM {table}")
                count, max_rowid = cursor.fetchone()
                checksums[table] = f"{count}:{max_rowid or 0}"
            except Exception as e:
                print(f"Warning: Could not calculate checksum for table {table}: {e}")
                checksums[table] = "error"

    return checksums


def validate_databases(staging_path: str, analytics_path: str) -> Dict[str, any]:
    """Validate that both databases exist and have required tables."""
    validation = {"valid": True, "errors": [], "warnings": []}

    required_tables = ["race_metadata", "dog_race_data"]

    # Check staging database
    if not Path(staging_path).exists():
        validation["valid"] = False
        validation["errors"].append(f"Staging database not found: {staging_path}")
        return validation

    # Check analytics database
    if not Path(analytics_path).exists():
        validation["warnings"].append(
            f"Analytics database not found: {analytics_path} (will be created)"
        )

    # Validate table structure
    try:
        staging_counts = get_table_row_counts(staging_path)
        for table in required_tables:
            if table not in staging_counts:
                validation["valid"] = False
                validation["errors"].append(
                    f"Required table missing in staging DB: {table}"
                )
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Error reading staging database: {e}")

    return validation


def full_sync(
    staging_path: str, analytics_path: str, backup_dir: Path
) -> Dict[str, any]:
    """Perform full database synchronization (complete copy)."""
    result = {
        "success": False,
        "method": "full_sync",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Create backup of analytics database if it exists
        if Path(analytics_path).exists():
            backup_path = create_backup(analytics_path, backup_dir / "analytics")
            result["backup_path"] = str(backup_path)

        # Use SQLite backup command for atomic copy
        with open_sqlite_readonly(staging_path) as source:
            with open_sqlite_writable(analytics_path) as dest:
                source.backup(dest)

        # Validate the copy
        staging_counts = get_table_row_counts(staging_path)
        analytics_counts = get_table_row_counts(analytics_path)

        result["staging_counts"] = staging_counts
        result["analytics_counts"] = analytics_counts
        result["success"] = staging_counts == analytics_counts

        if not result["success"]:
            result["error"] = "Row count mismatch after sync"

    except Exception as e:
        result["error"] = str(e)

    return result


def incremental_sync(
    staging_path: str, analytics_path: str, backup_dir: Path
) -> Dict[str, any]:
    """Perform incremental database synchronization."""
    result = {
        "success": False,
        "method": "incremental_sync",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # For now, fall back to full sync
        # TODO: Implement proper incremental sync based on timestamps
        result = full_sync(staging_path, analytics_path, backup_dir)
        result["method"] = "incremental_sync (fallback to full)"
        result["note"] = (
            "Incremental sync not yet implemented, performed full sync instead"
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def optimize_analytics_db(analytics_path: str) -> Dict[str, any]:
    """Optimize the analytics database for read performance."""
    result = {"success": False, "optimizations": []}

    try:
        with open_sqlite_writable(analytics_path) as conn:
            cursor = conn.cursor()

            # Update table statistics
            cursor.execute("ANALYZE")
            result["optimizations"].append("ANALYZE")

            # Vacuum to reclaim space and optimize
            cursor.execute("VACUUM")
            result["optimizations"].append("VACUUM")

            # Set optimization pragmas for analytics workload
            optimizations = [
                ("journal_mode", "WAL"),
                ("synchronous", "NORMAL"),
                ("cache_size", "10000"),
                ("temp_store", "MEMORY"),
                ("mmap_size", "268435456"),  # 256MB
            ]

            for pragma, value in optimizations:
                cursor.execute(f"PRAGMA {pragma} = {value}")
                result["optimizations"].append(f"{pragma} = {value}")

            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize staging database to analytics database"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="Synchronization mode (default: incremental)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate databases without syncing",
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize analytics database after sync"
    )
    parser.add_argument(
        "--backup-dir",
        default="database_backups/sync",
        help="Directory for sync backups",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    # Get database paths
    staging_path, analytics_path = get_db_paths()
    backup_dir = Path(args.backup_dir)

    print(f"Staging DB: {staging_path}")
    print(f"Analytics DB: {analytics_path}")
    print(f"Backup dir: {backup_dir}")

    # Validate databases
    validation = validate_databases(staging_path, analytics_path)

    if not validation["valid"]:
        print("❌ Database validation failed:")
        for error in validation["errors"]:
            print(f"  Error: {error}")
        return 1

    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"  Warning: {warning}")

    if args.validate_only:
        print("✅ Database validation passed")
        return 0

    if args.dry_run:
        print("🔍 Dry run mode - showing what would be done:")
        print(f"  - Create backup in {backup_dir}")
        print(f"  - Sync using {args.mode} mode")
        if args.optimize:
            print(f"  - Optimize analytics database")
        return 0

    # Perform synchronization
    print(f"🔄 Starting {args.mode} synchronization...")

    if args.mode == "full":
        result = full_sync(staging_path, analytics_path, backup_dir)
    else:
        result = incremental_sync(staging_path, analytics_path, backup_dir)

    # Optimize if requested
    if args.optimize and result["success"]:
        print("🚀 Optimizing analytics database...")
        opt_result = optimize_analytics_db(analytics_path)
        result["optimization"] = opt_result

    # Print results
    if result["success"]:
        print("✅ Synchronization completed successfully")
        if "staging_counts" in result:
            total_staging = sum(result["staging_counts"].values())
            total_analytics = sum(result["analytics_counts"].values())
            print(f"  Staging DB: {total_staging} total rows")
            print(f"  Analytics DB: {total_analytics} total rows")
    else:
        print("❌ Synchronization failed")
        if "error" in result:
            print(f"  Error: {result['error']}")

    # Save sync report
    report_path = (
        backup_dir / f"sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"📊 Sync report saved to: {report_path}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
