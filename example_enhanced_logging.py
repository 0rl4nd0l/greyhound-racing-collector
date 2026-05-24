#!/usr/bin/env python3
"""
Example script demonstrating the enhanced race operation logging functionality

This script shows how the new logging features work:
1. Per-race log lines in format: [SKIP|CACHE|FETCH] 2025-07-25 AP_K R4 – reason
2. HTTP status codes logged on every fetches_attempted
3. Conditional logging based on verbose_fetch flag (always emit warnings/errors)

Usage:
    python example_enhanced_logging.py --verbose-fetch
    python example_enhanced_logging.py  # Only warnings/errors logged
"""

import os
import sys
from datetime import datetime

from logger import logger


def demonstrate_enhanced_logging():
    """Demonstrate the enhanced race operation logging features"""

    print("🔍 Enhanced Race Operation Logging Demo")
    print("=" * 50)

    # Example 1: Cache hit (logged only if verbose_fetch=True)
    print("\n1. Cache Hit Example:")
    logger.log_race_operation(
        race_date="2025-07-25",
        venue="AP_K",
        race_number="4",
        operation="CACHE",
        reason="Race already collected",
        verbose_fetch=True,  # This will be logged
    )

    # Example 2: Cache hit with verbose_fetch=False (not logged)
    print("\n2. Cache Hit with verbose_fetch=False (not logged):")
    logger.log_race_operation(
        race_date="2025-07-25",
        venue="SAN",
        race_number="1",
        operation="CACHE",
        reason="Race already collected",
        verbose_fetch=False,  # This will NOT be logged
    )

    # Example 3: Successful fetch with HTTP status
    print("\n3. Successful Fetch with HTTP Status:")
    logger.log_race_operation(
        race_date="2025-07-25",
        venue="MEA",
        race_number="8",
        operation="FETCH",
        reason="CSV downloaded successfully",
        http_status=200,
        verbose_fetch=True,
    )

    # Example 4: Failed fetch with HTTP error (always logged as WARNING)
    print("\n4. Failed Fetch (Warning - always logged regardless of verbose_fetch):")
    logger.log_race_operation(
        race_date="2025-07-25",
        venue="GEE",
        race_number="3",
        operation="FETCH",
        reason="CSV download failed - no CSV link found",
        http_status=404,
        verbose_fetch=False,  # Still logged because it's a WARNING
        level="WARNING",
    )

    # Example 5: Error during fetch (always logged as ERROR)
    print(
        "\n5. Error During Fetch (Error - always logged regardless of verbose_fetch):"
    )
    logger.log_race_operation(
        race_date="2025-07-25",
        venue="DAPT",
        race_number="6",
        operation="FETCH",
        reason="Exception during download: Connection timeout",
        http_status=500,
        verbose_fetch=False,  # Still logged because it's an ERROR
        level="ERROR",
    )

    # Example 6: Skip due to date parsing error (always logged as ERROR)
    print("\n6. Skip Due to Date Parsing Error (Error - always logged):")
    logger.log_race_operation(
        race_date="invalid-date",
        venue="BEN",
        race_number="2",
        operation="SKIP",
        reason="Date parsing error: time data 'invalid-date' does not match expected format",
        verbose_fetch=False,  # Still logged because it's an ERROR
        level="ERROR",
    )

    # Example 7: Multiple races to show the log pattern
    print("\n7. Multiple Race Operations (simulating a scraping session):")

    races = [
        ("2025-07-25", "AP_K", "1", "CACHE", "Race already collected", None, "INFO"),
        (
            "2025-07-25",
            "AP_K",
            "2",
            "FETCH",
            "CSV downloaded successfully",
            200,
            "INFO",
        ),
        (
            "2025-07-25",
            "AP_K",
            "3",
            "FETCH",
            "CSV download failed - no CSV link found",
            404,
            "WARNING",
        ),
        ("2025-07-25", "AP_K", "4", "CACHE", "Race already collected", None, "INFO"),
        (
            "2025-07-25",
            "AP_K",
            "5",
            "FETCH",
            "CSV downloaded successfully",
            200,
            "INFO",
        ),
        (
            "2025-07-25",
            "AP_K",
            "6",
            "SKIP",
            "Race is upcoming, not historical",
            None,
            "INFO",
        ),
    ]

    for race_date, venue, race_no, operation, reason, http_status, level in races:
        logger.log_race_operation(
            race_date=race_date,
            venue=venue,
            race_number=race_no,
            operation=operation,
            reason=reason,
            http_status=http_status,
            verbose_fetch=True,  # Enable to see all logs
            level=level,
        )

    print("\n" + "=" * 50)
    print("📋 Logging Demo Complete!")
    print("\nCheck the following files for logged output:")
    print(f"   - {logger.process_log_file} (main process log)")
    print(f"   - {logger.workflow_log_file} (structured JSON log)")
    print(f"   - {logger.web_log_file} (web-accessible JSON log)")

    # Show some example log entries from the web logs
    print("\n📄 Recent Race Operation Logs:")
    race_logs = [
        log
        for log in logger.web_logs["process"]
        if log.get("component") == "race_operation"
    ]
    for log in race_logs[-5:]:  # Show last 5 race operation logs
        print(f"   {log['timestamp'][:19]} - {log['message']}")


def demonstrate_fetch_statistics_integration():
    """Show how the enhanced logging integrates with fetch statistics"""

    print("\n🔄 Integration with Fetch Statistics")
    print("=" * 40)

    # Simulate what happens in the scraper during different scenarios
    scenarios = [
        {
            "description": "Cache Hit - No HTTP request made",
            "stats_updates": ["races_requested", "cache_hits"],
            "log_params": {
                "race_date": "2025-07-25",
                "venue": "W_PK",
                "race_number": "7",
                "operation": "CACHE",
                "reason": "Race already collected",
                "verbose_fetch": True,
            },
        },
        {
            "description": "Successful Fetch - HTTP request successful",
            "stats_updates": [
                "races_requested",
                "fetches_attempted",
                "successful_saves",
            ],
            "log_params": {
                "race_date": "2025-07-25",
                "venue": "CANN",
                "race_number": "9",
                "operation": "FETCH",
                "reason": "CSV downloaded successfully",
                "http_status": 200,
                "verbose_fetch": True,
            },
        },
        {
            "description": "Failed Fetch - HTTP request failed",
            "stats_updates": ["races_requested", "fetches_attempted", "fetches_failed"],
            "log_params": {
                "race_date": "2025-07-25",
                "venue": "RICH",
                "race_number": "5",
                "operation": "FETCH",
                "reason": "CSV download failed - server error",
                "http_status": 500,
                "verbose_fetch": True,
                "level": "WARNING",
            },
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['description']}")
        print(f"   Statistics Updated: {', '.join(scenario['stats_updates'])}")
        print(f"   HTTP Status: {scenario['log_params'].get('http_status', 'N/A')}")

        # Log the race operation
        logger.log_race_operation(**scenario["log_params"])

        print(
            f"   Log Entry: [{scenario['log_params']['operation']}] "
            f"{scenario['log_params']['race_date']} {scenario['log_params']['venue']} "
            f"R{scenario['log_params']['race_number']} – {scenario['log_params']['reason']}"
            f"{' (HTTP ' + str(scenario['log_params']['http_status']) + ')' if scenario['log_params'].get('http_status') else ''}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate enhanced race operation logging"
    )
    parser.add_argument(
        "--verbose-fetch",
        action="store_true",
        help="Enable verbose fetch logging (shows all operations)",
    )
    parser.add_argument(
        "--stats-integration",
        action="store_true",
        help="Show integration with fetch statistics",
    )

    args = parser.parse_args()

    print(f"🚀 Starting Enhanced Logging Demo")
    print(
        f"   Verbose Fetch: {'ENABLED' if args.verbose_fetch else 'DISABLED (only warnings/errors)'}"
    )
    print(f"   Log Directory: {logger.log_dir}")

    # Run the basic logging demonstration
    demonstrate_enhanced_logging()

    # Run the statistics integration demo if requested
    if args.stats_integration:
        demonstrate_fetch_statistics_integration()

    print(f"\n✅ Demo completed successfully!")
    print(f"💡 Try running with --verbose-fetch to see all log entries")
    print(
        f"💡 Try running with --stats-integration to see fetch statistics integration"
    )
