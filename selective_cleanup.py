#!/usr/bin/env python3
"""
Selective Cleanup for Single Dog Races
=====================================

This script implements the recommended selective cleanup approach:
1. Keep recent single-dog races (last 30 days)
2. Archive older single-dog races
3. Generate summary report

Usage:
    python3 selective_cleanup.py --preview    # Preview what will be done
    python3 selective_cleanup.py --execute    # Execute the cleanup
"""

import argparse
import os
import shutil
from datetime import datetime

import pandas as pd


def selective_cleanup(execute=False):
    """Perform selective cleanup of single dog races"""

    print("🎯 SELECTIVE CLEANUP FOR SINGLE DOG RACES")
    print("=" * 60)

    # Load the data
    file_path = "reports/data_quality/single_dog_races.csv"
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    df["race_date"] = pd.to_datetime(df["race_date"])

    # Define cutoff (30 days from most recent date)
    recent_cutoff = df["race_date"].max() - pd.Timedelta(days=30)

    # Split data
    recent_races = df[df["race_date"] >= recent_cutoff]
    archive_races = df[df["race_date"] < recent_cutoff]

    print(f"📊 ANALYSIS RESULTS:")
    print(f"   Total single-dog races: {len(df):,}")
    print(f"   Recent races (keep): {len(recent_races):,}")
    print(f"   Archive races: {len(archive_races):,}")
    print(f"   Cutoff date: {recent_cutoff.strftime('%Y-%m-%d')}")

    if not execute:
        print(f"\n🔍 PREVIEW MODE - No changes will be made")
        print(f"   Run with --execute to perform cleanup")
        return

    print(f"\n🚀 EXECUTING CLEANUP...")

    # Create archive directory
    archive_dir = "archive/incomplete_races"
    os.makedirs(archive_dir, exist_ok=True)

    # Save archived races
    archive_file = f"{archive_dir}/single_dog_races_archived_{datetime.now().strftime('%Y%m%d')}.csv"
    archive_races.to_csv(archive_file, index=False)
    print(f"   ✅ Archived {len(archive_races):,} races to: {archive_file}")

    # Save recent races (keep active)
    recent_file = "reports/data_quality/single_dog_races_recent.csv"
    recent_races.to_csv(recent_file, index=False)
    print(f"   ✅ Saved {len(recent_races):,} recent races to: {recent_file}")

    # Backup original file
    backup_file = f"reports/data_quality/single_dog_races_backup_{datetime.now().strftime('%Y%m%d')}.csv"
    shutil.copy2(file_path, backup_file)
    print(f"   ✅ Backed up original to: {backup_file}")

    # Generate summary report
    summary = {
        "cleanup_date": datetime.now().isoformat(),
        "total_original_races": len(df),
        "recent_races_kept": len(recent_races),
        "races_archived": len(archive_races),
        "cutoff_date": recent_cutoff.isoformat(),
        "archive_file": archive_file,
        "recent_file": recent_file,
        "backup_file": backup_file,
        "venue_breakdown": {
            "recent": recent_races["venue"].value_counts().to_dict(),
            "archived": archive_races["venue"].value_counts().to_dict(),
        },
    }

    import json

    summary_file = (
        f"{archive_dir}/cleanup_summary_{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Summary saved to: {summary_file}")

    print(f"\n🎉 CLEANUP COMPLETED!")
    print(f"   • Archived: {len(archive_races):,} races")
    print(f"   • Kept active: {len(recent_races):,} races")
    print(
        f"   • Space saved: {(len(archive_races)/len(df))*100:.1f}% of single-dog races moved to archive"
    )

    # Next steps
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review recent races for data collection issues")
    print(f"   2. Fix race_name column extraction in data pipeline")
    print(f"   3. Monitor for new single-dog race occurrences")
    print(f"   4. Update prediction system to handle 'incomplete race data' flag")


def main():
    parser = argparse.ArgumentParser(
        description="Selective cleanup of single dog races"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Preview cleanup without executing"
    )
    parser.add_argument("--execute", action="store_true", help="Execute the cleanup")

    args = parser.parse_args()

    if not args.preview and not args.execute:
        print("Please specify --preview or --execute")
        return

    selective_cleanup(execute=args.execute)


if __name__ == "__main__":
    main()
