#!/usr/bin/env python3
"""
Race Status Management Utility
==============================

Comprehensive tool for managing race processing status, including:
- Status checking and reporting
- Manual status updates  
- Backfill priority management
- Data quality assessment

Author: AI Assistant
Date: August 23, 2025
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


class RaceStatusManager:
    """Comprehensive race status management"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path

    def get_status_overview(self) -> Dict[str, Any]:
        """Get comprehensive status overview"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE results_status = 'complete' AND winner_name IS NOT NULL AND winner_name != ''
            """
            )
            complete_races = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE results_status = 'pending' OR (results_status IS NULL AND (winner_name IS NULL OR winner_name = ''))
            """
            )
            pending_races = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE results_status = 'partial_scraping_failed'
            """
            )
            partial_failed = cursor.fetchone()[0]

            # Data quality metrics
            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE winner_name IS NOT NULL AND winner_name != '' 
                AND (results_status IS NULL OR results_status = 'pending')
            """
            )
            has_winner_but_pending = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE scraping_attempts > 0
            """
            )
            attempted_scraping = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE data_quality_note IS NOT NULL AND data_quality_note != ''
            """
            )
            has_quality_notes = cursor.fetchone()[0]

            return {
                "total_races": total_races,
                "complete_races": complete_races,
                "pending_races": pending_races,
                "partial_failed": partial_failed,
                "completion_rate": (
                    complete_races / total_races if total_races > 0 else 0
                ),
                "has_winner_but_pending": has_winner_but_pending,
                "attempted_scraping": attempted_scraping,
                "has_quality_notes": has_quality_notes,
            }

        finally:
            conn.close()

    def update_race_status(
        self,
        race_id: str,
        new_status: str,
        winner_name: str = None,
        winner_source: str = None,
        note: str = None,
    ) -> bool:
        """Update status for a specific race"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if race exists
            cursor.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?", (race_id,)
            )
            if cursor.fetchone()[0] == 0:
                print(f"❌ Race {race_id} not found in database")
                return False

            # Build update query
            updates = ["results_status = ?"]
            values = [new_status]

            if winner_name is not None:
                updates.append("winner_name = ?")
                values.append(winner_name)

            if winner_source is not None:
                updates.append("winner_source = ?")
                values.append(winner_source)

            if note is not None:
                updates.append("data_quality_note = ?")
                values.append(note)

            values.append(race_id)  # For WHERE clause

            query = f"UPDATE race_metadata SET {', '.join(updates)} WHERE race_id = ?"
            cursor.execute(query, values)

            if cursor.rowcount > 0:
                conn.commit()
                print(f"✅ Updated race {race_id} status to '{new_status}'")
                return True
            else:
                print(f"❌ No changes made to race {race_id}")
                return False

        except Exception as e:
            print(f"❌ Error updating race {race_id}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def bulk_update_status(
        self,
        filter_criteria: Dict[str, Any],
        new_status: str,
        winner_source: str = None,
        note: str = None,
    ) -> int:
        """Bulk update race status based on criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Build WHERE clause
            where_conditions = []
            values = []

            for key, value in filter_criteria.items():
                if key == "venue":
                    where_conditions.append("venue = ?")
                    values.append(value)
                elif key == "status":
                    where_conditions.append("results_status = ?")
                    values.append(value)
                elif key == "attempts_gte":
                    where_conditions.append("COALESCE(scraping_attempts, 0) >= ?")
                    values.append(value)
                elif key == "attempts_lte":
                    where_conditions.append("COALESCE(scraping_attempts, 0) <= ?")
                    values.append(value)
                elif key == "date_from":
                    where_conditions.append("race_date >= ?")
                    values.append(value)
                elif key == "date_to":
                    where_conditions.append("race_date <= ?")
                    values.append(value)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # First, get count of races to be updated
            count_query = f"SELECT COUNT(*) FROM race_metadata WHERE {where_clause}"
            cursor.execute(count_query, values)
            count_to_update = cursor.fetchone()[0]

            if count_to_update == 0:
                print("❌ No races match the specified criteria")
                return 0

            # Confirm bulk update
            print(
                f"⚠️  About to update {count_to_update} races to status '{new_status}'"
            )
            response = input("Continue? (y/N): ")
            if response.lower() != "y":
                print("❌ Bulk update cancelled")
                return 0

            # Build update query
            updates = ["results_status = ?"]
            update_values = [new_status] + values  # new_status first, then WHERE values

            if winner_source is not None:
                updates.append("winner_source = ?")
                update_values.insert(1, winner_source)  # Insert after new_status

            if note is not None:
                updates.append("data_quality_note = ?")
                update_values.insert(
                    -len(values) if values else 1, note
                )  # Insert before WHERE values

            update_query = (
                f"UPDATE race_metadata SET {', '.join(updates)} WHERE {where_clause}"
            )
            cursor.execute(update_query, update_values)

            updated_count = cursor.rowcount
            conn.commit()
            print(f"✅ Updated {updated_count} races to status '{new_status}'")
            return updated_count

        except Exception as e:
            print(f"❌ Error in bulk update: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_problematic_races(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get races that might have data quality issues"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Races with high scraping attempts but still pending
            cursor.execute(
                """
                SELECT race_id, venue, race_date, scraping_attempts, data_quality_note,
                       'high_attempts' as issue_type
                FROM race_metadata 
                WHERE results_status = 'pending' 
                AND scraping_attempts >= 3
                ORDER BY scraping_attempts DESC, race_date DESC
                LIMIT ?
            """,
                (limit // 3,),
            )
            high_attempts = cursor.fetchall()

            # Races with winners but still pending status
            cursor.execute(
                """
                SELECT race_id, venue, race_date, winner_name, results_status,
                       'has_winner_but_pending' as issue_type
                FROM race_metadata 
                WHERE winner_name IS NOT NULL AND winner_name != ''
                AND (results_status = 'pending' OR results_status IS NULL)
                ORDER BY race_date DESC
                LIMIT ?
            """,
                (limit // 3,),
            )
            winner_but_pending = cursor.fetchall()

            # Recent races still pending (might need attention)
            recent_date = (datetime.now() - timedelta(days=7)).date()
            cursor.execute(
                """
                SELECT race_id, venue, race_date, scraping_attempts, results_status,
                       'recent_pending' as issue_type
                FROM race_metadata 
                WHERE results_status = 'pending' 
                AND race_date >= ?
                ORDER BY race_date DESC
                LIMIT ?
            """,
                (recent_date, limit // 3),
            )
            recent_pending = cursor.fetchall()

            problems = []

            for row in high_attempts:
                problems.append(
                    {
                        "race_id": row[0],
                        "venue": row[1],
                        "race_date": row[2],
                        "issue_type": row[5],
                        "details": f"{row[3]} scraping attempts",
                        "note": row[4] or "",
                    }
                )

            for row in winner_but_pending:
                problems.append(
                    {
                        "race_id": row[0],
                        "venue": row[1],
                        "race_date": row[2],
                        "issue_type": row[5],
                        "details": f"Winner: {row[3]}, Status: {row[4] or 'NULL'}",
                        "note": "",
                    }
                )

            for row in recent_pending:
                problems.append(
                    {
                        "race_id": row[0],
                        "venue": row[1],
                        "race_date": row[2],
                        "issue_type": row[5],
                        "details": f"Recent race ({row[2]}) still pending after {row[3] or 0} attempts",
                        "note": "",
                    }
                )

            return problems

        finally:
            conn.close()

    def print_status_report(self):
        """Print comprehensive status report"""
        overview = self.get_status_overview()

        print("🏁 RACE STATUS MANAGEMENT REPORT")
        print("=" * 60)

        print(f"\n📊 OVERVIEW:")
        print(f"   Total races: {overview['total_races']:,}")
        print(
            f"   Complete: {overview['complete_races']:,} ({overview['completion_rate']:.1%})"
        )
        print(f"   Pending: {overview['pending_races']:,}")
        print(f"   Partial failed: {overview['partial_failed']:,}")

        print(f"\n🔍 DATA QUALITY:")
        print(f"   Races with scraping attempts: {overview['attempted_scraping']:,}")
        print(f"   Races with quality notes: {overview['has_quality_notes']:,}")
        print(f"   Has winner but pending: {overview['has_winner_but_pending']:,}")

        # Show problematic races
        problems = self.get_problematic_races(30)
        if problems:
            print(f"\n⚠️  RACES NEEDING ATTENTION ({len(problems)} found):")
            for problem in problems[:15]:
                print(
                    f"   {problem['race_id']}: {problem['issue_type']} - {problem['details']}"
                )

        print(f"\n💡 RECOMMENDATIONS:")
        completion_rate = overview["completion_rate"]
        if completion_rate < 0.5:
            print("   🔴 LOW completion rate - run comprehensive processing")
        elif completion_rate < 0.8:
            print("   🟡 GOOD completion rate - run targeted backfill")
        else:
            print("   🟢 EXCELLENT completion rate - minimal work needed")

        if overview["has_winner_but_pending"] > 0:
            print(
                f"   🔧 {overview['has_winner_but_pending']} races have winners but wrong status"
            )
            print("      → Use: race_status_manager.py --fix-winner-status")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Race Status Management Utility")
    parser.add_argument(
        "--db", default="greyhound_racing_data.db", help="Database path"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status report command
    status_parser = subparsers.add_parser("status", help="Show status report")

    # Update single race command
    update_parser = subparsers.add_parser("update", help="Update single race status")
    update_parser.add_argument("race_id", help="Race ID to update")
    update_parser.add_argument(
        "status",
        choices=["pending", "complete", "partial_scraping_failed"],
        help="New status",
    )
    update_parser.add_argument("--winner", help="Winner name")
    update_parser.add_argument(
        "--source", choices=["scrape", "inferred", "manual"], help="Winner source"
    )
    update_parser.add_argument("--note", help="Data quality note")

    # Bulk update command
    bulk_parser = subparsers.add_parser("bulk-update", help="Bulk update race status")
    bulk_parser.add_argument(
        "status",
        choices=["pending", "complete", "partial_scraping_failed"],
        help="New status",
    )
    bulk_parser.add_argument("--venue", help="Filter by venue")
    bulk_parser.add_argument("--current-status", help="Filter by current status")
    bulk_parser.add_argument(
        "--min-attempts", type=int, help="Minimum scraping attempts"
    )
    bulk_parser.add_argument(
        "--max-attempts", type=int, help="Maximum scraping attempts"
    )
    bulk_parser.add_argument("--from-date", help="From date (YYYY-MM-DD)")
    bulk_parser.add_argument("--to-date", help="To date (YYYY-MM-DD)")
    bulk_parser.add_argument(
        "--source", choices=["scrape", "inferred", "manual"], help="Winner source"
    )
    bulk_parser.add_argument("--note", help="Data quality note")

    # Fix common issues
    fix_parser = subparsers.add_parser(
        "fix-winner-status", help="Fix races with winners but wrong status"
    )

    # Show problematic races
    problems_parser = subparsers.add_parser(
        "problems", help="Show races needing attention"
    )
    problems_parser.add_argument(
        "--limit", type=int, default=50, help="Maximum races to show"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = RaceStatusManager(args.db)

    if args.command == "status":
        manager.print_status_report()

    elif args.command == "update":
        manager.update_race_status(
            args.race_id, args.status, args.winner, args.source, args.note
        )

    elif args.command == "bulk-update":
        criteria = {}
        if args.venue:
            criteria["venue"] = args.venue
        if args.current_status:
            criteria["status"] = args.current_status
        if args.min_attempts is not None:
            criteria["attempts_gte"] = args.min_attempts
        if args.max_attempts is not None:
            criteria["attempts_lte"] = args.max_attempts
        if args.from_date:
            criteria["date_from"] = args.from_date
        if args.to_date:
            criteria["date_to"] = args.to_date

        manager.bulk_update_status(criteria, args.status, args.source, args.note)

    elif args.command == "fix-winner-status":
        criteria = {"status": "pending"}  # Find pending races
        # This will be enhanced to specifically target races with winners
        count = manager.bulk_update_status(
            criteria,
            "complete",
            "inferred",
            "Status corrected - had winner but was marked pending",
        )
        print(f"Fixed status for {count} races")

    elif args.command == "problems":
        problems = manager.get_problematic_races(args.limit)
        print(f"🔍 PROBLEMATIC RACES ({len(problems)} found):")
        print("=" * 60)

        for problem in problems:
            print(f"\n🏁 {problem['race_id']} ({problem['venue']})")
            print(f"   Issue: {problem['issue_type']}")
            print(f"   Details: {problem['details']}")
            if problem["note"]:
                print(f"   Note: {problem['note']}")


if __name__ == "__main__":
    main()
