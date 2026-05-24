#!/usr/bin/env python3
"""
System Health Monitoring Script

Monitors the health of the greyhound racing prediction system, including:
- Database accessibility and integrity
- Database routing system status
- Key table row counts and data freshness
- Model registry status

Usage:
    python scripts/monitor_system_health_fixed.py
    python scripts/monitor_system_health_fixed.py --json
    python scripts/monitor_system_health_fixed.py --detailed

Output:
    - JSON format: Suitable for monitoring systems integration
    - Detailed format: Human-readable diagnostic information
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.db_utils import open_sqlite_readonly


def get_db_paths() -> Tuple[Optional[str], Optional[str], str]:
    """Get staging, analytics, and fallback database paths."""
    staging_db = os.getenv("STAGING_DB_PATH")
    analytics_db = os.getenv("ANALYTICS_DB_PATH")
    fallback_db = os.getenv("GREYHOUND_DB_PATH", "greyhound_racing_data.db")

    return staging_db, analytics_db, fallback_db


def check_database_health(db_path: str, db_type: str) -> Dict[str, any]:
    """Check the health of a single database."""
    health = {
        "type": db_type,
        "path": db_path,
        "accessible": False,
        "tables": {},
        "issues": [],
    }

    if not Path(db_path).exists():
        health["issues"].append(f"Database file not found: {db_path}")
        return health

    try:
        with open_sqlite_readonly(db_path) as conn:
            cursor = conn.cursor()

            # Check database accessibility
            cursor.execute("SELECT 1")
            health["accessible"] = True

            # Get table information
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            # Get row counts for key tables
            key_tables = [
                "race_metadata",
                "dog_race_data",
                "ml_model_registry",
                "tgr_enhanced_dog_form",
            ]

            for table in key_tables:
                if table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        health["tables"][table] = {"count": count}

                        # Check data freshness for time-sensitive tables
                        if table == "race_metadata":
                            cursor.execute(
                                "SELECT MAX(race_date) FROM race_metadata WHERE race_date IS NOT NULL"
                            )
                            latest_date = cursor.fetchone()[0]
                            health["tables"][table]["latest_date"] = latest_date

                        elif table == "dog_race_data":
                            cursor.execute(
                                "SELECT MAX(extraction_timestamp) FROM dog_race_data WHERE extraction_timestamp IS NOT NULL"
                            )
                            latest_extraction = cursor.fetchone()[0]
                            health["tables"][table][
                                "latest_extraction"
                            ] = latest_extraction

                        elif table == "ml_model_registry":
                            cursor.execute(
                                "SELECT COUNT(*) FROM ml_model_registry WHERE is_active = 1"
                            )
                            active_models = cursor.fetchone()[0]
                            health["tables"][table]["active_models"] = active_models

                    except Exception as e:
                        health["tables"][table] = {"error": str(e)}
                        health["issues"].append(f"Error querying {table}: {e}")
                else:
                    health["tables"][table] = {"missing": True}

            # Database integrity check
            cursor.execute("PRAGMA integrity_check(10)")
            integrity_results = cursor.fetchall()
            if integrity_results and integrity_results[0][0] != "ok":
                health["issues"].extend(
                    [f"Integrity: {row[0]}" for row in integrity_results]
                )

    except Exception as e:
        health["issues"].append(f"Database connection error: {e}")

    return health


def check_routing_system() -> Dict[str, any]:
    """Check the database routing system configuration."""
    routing = {
        "configured": False,
        "staging_path": None,
        "analytics_path": None,
        "fallback_path": None,
        "routing_active": False,
        "issues": [],
    }

    staging_db, analytics_db, fallback_db = get_db_paths()

    routing["staging_path"] = staging_db
    routing["analytics_path"] = analytics_db
    routing["fallback_path"] = fallback_db

    # Check if routing is configured
    if staging_db and analytics_db:
        routing["configured"] = True
        routing["routing_active"] = staging_db != analytics_db

        # Check if paths exist
        if not Path(staging_db).exists():
            routing["issues"].append(f"Staging DB not found: {staging_db}")
        if not Path(analytics_db).exists():
            routing["issues"].append(f"Analytics DB not found: {analytics_db}")

    else:
        routing["issues"].append("Database routing not configured - using fallback")
        if not Path(fallback_db).exists():
            routing["issues"].append(f"Fallback DB not found: {fallback_db}")

    return routing


def generate_health_report(detailed: bool = False) -> Dict[str, any]:
    """Generate a comprehensive system health report."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "unknown",
        "components": {},
        "summary": {"total_issues": 0, "critical_issues": 0, "warnings": 0},
    }

    # Check database routing system
    routing_health = check_routing_system()
    report["components"]["routing"] = routing_health

    # Check individual databases
    staging_db, analytics_db, fallback_db = get_db_paths()

    if staging_db and Path(staging_db).exists():
        staging_health = check_database_health(staging_db, "staging")
        report["components"]["staging_db"] = staging_health

    if analytics_db and Path(analytics_db).exists():
        analytics_health = check_database_health(analytics_db, "analytics")
        report["components"]["analytics_db"] = analytics_health

    if Path(fallback_db).exists():
        fallback_health = check_database_health(fallback_db, "fallback")
        report["components"]["fallback_db"] = fallback_health

    # Calculate summary statistics
    total_issues = 0
    critical_issues = 0
    warnings = 0

    for component_name, component in report["components"].items():
        if "issues" in component:
            component_issues = len(component["issues"])
            total_issues += component_issues

            # Classify severity
            if component_name in ["routing", "fallback_db"] and component_issues > 0:
                critical_issues += component_issues
            else:
                warnings += component_issues

    report["summary"]["total_issues"] = total_issues
    report["summary"]["critical_issues"] = critical_issues
    report["summary"]["warnings"] = warnings

    # Determine overall status
    if critical_issues > 0:
        report["overall_status"] = "critical"
    elif warnings > 3:
        report["overall_status"] = "degraded"
    elif warnings > 0:
        report["overall_status"] = "warning"
    else:
        report["overall_status"] = "healthy"

    return report


def print_detailed_report(report: Dict[str, any]):
    """Print a human-readable detailed health report."""
    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "degraded": "🔶",
        "critical": "❌",
        "unknown": "❓",
    }

    print(f"\n{status_emoji.get(report['overall_status'], '❓')} System Health Report")
    print(f"Generated: {report['timestamp']}")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(
        f"Issues: {report['summary']['total_issues']} total ({report['summary']['critical_issues']} critical, {report['summary']['warnings']} warnings)"
    )

    for component_name, component in report["components"].items():
        print(f"\n📊 {component_name.replace('_', ' ').title()}:")

        if component_name == "routing":
            print(f"  Configured: {component['configured']}")
            print(f"  Active: {component['routing_active']}")
            if component["staging_path"]:
                print(f"  Staging: {component['staging_path']}")
            if component["analytics_path"]:
                print(f"  Analytics: {component['analytics_path']}")
            print(f"  Fallback: {component['fallback_path']}")

        elif "db" in component_name:
            print(f"  Path: {component['path']}")
            print(f"  Accessible: {component['accessible']}")
            if "tables" in component:
                for table, info in component["tables"].items():
                    if "count" in info:
                        print(f"    {table}: {info['count']} rows")
                        if "latest_date" in info and info["latest_date"]:
                            print(f"      Latest date: {info['latest_date']}")
                        if "active_models" in info:
                            print(f"      Active models: {info['active_models']}")
                    elif "missing" in info:
                        print(f"    {table}: MISSING")

        # Print issues
        if "issues" in component and component["issues"]:
            for issue in component["issues"]:
                print(f"    ⚠️  {issue}")


def main():
    parser = argparse.ArgumentParser(description="Monitor system health")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--detailed", action="store_true", help="Show detailed report")

    args = parser.parse_args()

    # Generate health report
    report = generate_health_report(detailed=args.detailed)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_detailed_report(report)

        # Return appropriate exit code
        if report["overall_status"] == "critical":
            return 2
        elif report["overall_status"] in ["degraded", "warning"]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    sys.exit(main())
