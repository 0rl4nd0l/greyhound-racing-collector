#!/usr/bin/env python3
"""
Standalone Race Status Checker
==============================

Check and analyze race status information without heavy dependencies.

Author: AI Assistant
Date: August 23, 2025
"""

import sqlite3
from datetime import datetime
from typing import Any, Dict, List


def get_pending_race_statistics(
    db_path: str = "greyhound_racing_data.db",
) -> Dict[str, Any]:
    """Get comprehensive statistics about race status"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Total pending races
        cursor.execute(
            """
            SELECT COUNT(*) as total_pending
            FROM race_metadata 
            WHERE results_status = 'pending' 
               OR (results_status IS NULL AND (winner_name IS NULL OR winner_name = ''))
        """
        )
        total_pending = cursor.fetchone()[0]

        # Total complete races
        cursor.execute(
            """
            SELECT COUNT(*) as total_complete
            FROM race_metadata 
            WHERE results_status = 'complete' 
               AND winner_name IS NOT NULL 
               AND winner_name != ''
        """
        )
        total_complete = cursor.fetchone()[0]

        # Total races
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        total_races = cursor.fetchone()[0]

        # Status breakdown
        cursor.execute(
            """
            SELECT 
                COALESCE(results_status, 'NULL') as status,
                COUNT(*) as count
            FROM race_metadata 
            GROUP BY COALESCE(results_status, 'NULL')
            ORDER BY count DESC
        """
        )
        status_breakdown = dict(cursor.fetchall())

        # Winner source breakdown
        cursor.execute(
            """
            SELECT 
                COALESCE(winner_source, 'NULL') as source,
                COUNT(*) as count
            FROM race_metadata 
            WHERE winner_source IS NOT NULL
            GROUP BY COALESCE(winner_source, 'NULL')
            ORDER BY count DESC
        """
        )
        source_breakdown = dict(cursor.fetchall())

        # Pending races by venue
        cursor.execute(
            """
            SELECT venue, COUNT(*) as count
            FROM race_metadata 
            WHERE results_status = 'pending' 
               OR (results_status IS NULL AND (winner_name IS NULL OR winner_name = ''))
            GROUP BY venue
            ORDER BY count DESC
            LIMIT 15
        """
        )
        pending_by_venue = dict(cursor.fetchall())

        # Pending races by scraping attempts
        cursor.execute(
            """
            SELECT 
                COALESCE(scraping_attempts, 0) as attempts,
                COUNT(*) as count
            FROM race_metadata 
            WHERE results_status = 'pending' 
               OR (results_status IS NULL AND (winner_name IS NULL OR winner_name = ''))
            GROUP BY COALESCE(scraping_attempts, 0)
            ORDER BY attempts
        """
        )
        pending_by_attempts = dict(cursor.fetchall())

        # Recent pending races sample
        cursor.execute(
            """
            SELECT race_id, venue, race_number, race_date, 
                   COALESCE(scraping_attempts, 0) as attempts,
                   COALESCE(data_quality_note, '') as note
            FROM race_metadata 
            WHERE results_status = 'pending' 
               OR (results_status IS NULL AND (winner_name IS NULL OR winner_name = ''))
            ORDER BY race_date DESC, race_number ASC
            LIMIT 20
        """
        )
        recent_pending = cursor.fetchall()

        # Recent complete races sample
        cursor.execute(
            """
            SELECT race_id, venue, race_number, race_date, winner_name, winner_source
            FROM race_metadata 
            WHERE results_status = 'complete' 
               AND winner_name IS NOT NULL 
               AND winner_name != ''
            ORDER BY extraction_timestamp DESC
            LIMIT 10
        """
        )
        recent_complete = cursor.fetchall()

        statistics = {
            "total_races": total_races,
            "total_pending": total_pending,
            "total_complete": total_complete,
            "completion_rate": total_complete / total_races if total_races > 0 else 0,
            "status_breakdown": status_breakdown,
            "source_breakdown": source_breakdown,
            "pending_by_venue": pending_by_venue,
            "pending_by_attempts": pending_by_attempts,
            "recent_pending": [
                {
                    "race_id": row[0],
                    "venue": row[1],
                    "race_number": row[2],
                    "race_date": row[3],
                    "attempts": row[4],
                    "note": row[5],
                }
                for row in recent_pending
            ],
            "recent_complete": [
                {
                    "race_id": row[0],
                    "venue": row[1],
                    "race_number": row[2],
                    "race_date": row[3],
                    "winner_name": row[4],
                    "winner_source": row[5],
                }
                for row in recent_complete
            ],
        }

        return statistics

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def print_status_report(stats: Dict[str, Any]):
    """Print a comprehensive status report"""
    if "error" in stats:
        print(f"❌ Error getting statistics: {stats['error']}")
        return

    print("🏁 GREYHOUND RACING DATABASE STATUS REPORT")
    print("=" * 65)

    # Overview
    print(f"\n📊 OVERVIEW:")
    print(f"   Total races in database: {stats['total_races']:,}")
    print(f"   Complete races: {stats['total_complete']:,}")
    print(f"   Pending races: {stats['total_pending']:,}")
    print(f"   Completion rate: {stats['completion_rate']:.1%}")

    # Status breakdown
    print(f"\n🎯 STATUS BREAKDOWN:")
    for status, count in stats["status_breakdown"].items():
        percentage = (
            (count / stats["total_races"]) * 100 if stats["total_races"] > 0 else 0
        )
        print(f"   {status}: {count:,} races ({percentage:.1f}%)")

    # Winner source breakdown
    if stats["source_breakdown"]:
        print(f"\n🎯 WINNER SOURCE BREAKDOWN:")
        for source, count in stats["source_breakdown"].items():
            print(f"   {source}: {count:,} races")

    # Pending by venue
    if stats["pending_by_venue"]:
        print(f"\n📍 TOP VENUES WITH PENDING RACES:")
        for venue, count in list(stats["pending_by_venue"].items())[:10]:
            print(f"   {venue}: {count:,} pending")

    # Pending by attempts
    if stats["pending_by_attempts"]:
        print(f"\n🔄 PENDING RACES BY SCRAPING ATTEMPTS:")
        for attempts, count in stats["pending_by_attempts"].items():
            print(f"   {attempts} attempts: {count:,} races")

    # Recent pending sample
    if stats["recent_pending"]:
        print(f"\n📋 RECENT PENDING RACES (Sample):")
        for race in stats["recent_pending"][:10]:
            note_display = f" - {race['note'][:50]}..." if race["note"] else ""
            print(f"   {race['race_id']}: {race['attempts']} attempts{note_display}")

    # Recent complete sample
    if stats["recent_complete"]:
        print(f"\n✅ RECENT COMPLETE RACES (Sample):")
        for race in stats["recent_complete"]:
            source_display = (
                f" ({race['winner_source']})" if race["winner_source"] else ""
            )
            print(f"   {race['race_id']}: {race['winner_name']}{source_display}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if stats["total_pending"] > 0:
        high_priority = sum(
            count
            for attempts, count in stats["pending_by_attempts"].items()
            if attempts == 0
        )
        medium_priority = sum(
            count
            for attempts, count in stats["pending_by_attempts"].items()
            if 1 <= attempts <= 2
        )
        low_priority = sum(
            count
            for attempts, count in stats["pending_by_attempts"].items()
            if attempts >= 3
        )

        if high_priority > 0:
            print(
                f"   🔴 HIGH PRIORITY: {high_priority:,} races with 0 scraping attempts"
            )
            print(f"      → Run backfill process on these first")

        if medium_priority > 0:
            print(f"   🟡 MEDIUM PRIORITY: {medium_priority:,} races with 1-2 attempts")
            print(f"      → Retry these with updated scraping logic")

        if low_priority > 0:
            print(f"   ⚪ LOW PRIORITY: {low_priority:,} races with 3+ attempts")
            print(f"      → These may have persistent issues, investigate manually")

        completion_rate = stats["completion_rate"]
        if completion_rate < 0.5:
            print(
                f"   ⚠️  Low completion rate ({completion_rate:.1%}) - consider running comprehensive backfill"
            )
        elif completion_rate < 0.8:
            print(
                f"   ✅ Good completion rate ({completion_rate:.1%}) - run targeted backfill on high priority races"
            )
        else:
            print(
                f"   🎉 Excellent completion rate ({completion_rate:.1%}) - minimal backfill needed"
            )
    else:
        print(f"   🎉 All races are complete! No pending races found.")

    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Run: python3 enhanced_comprehensive_processor.py")
    print(f"   2. Or run targeted backfill on pending races")
    print(f"   3. Monitor completion rates and adjust scraping strategy")


def main():
    """Main function"""
    print("🔍 Checking race status...")

    try:
        stats = get_pending_race_statistics()
        print_status_report(stats)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
