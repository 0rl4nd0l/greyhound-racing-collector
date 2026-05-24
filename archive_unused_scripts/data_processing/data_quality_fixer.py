#!/usr/bin/env python3
"""
Data Quality Fixer for Greyhound Racing Data
============================================

This script identifies and fixes data quality issues in the greyhound racing database,
specifically addressing:
1. Race 0/Unknown venue issues
2. Missing or incorrect finish positions
3. Dead heat/tie handling
4. Data validation gaps

Usage:
    python data_quality_fixer.py --analyze    # Analyze issues only
    python data_quality_fixer.py --fix        # Fix issues
    python data_quality_fixer.py --validate   # Validate data quality
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class DataQualityFixer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.issues_found = []
        self.fixes_applied = []

    def analyze_data_quality(self):
        """Analyze database for data quality issues"""
        print("🔍 ANALYZING DATA QUALITY ISSUES")
        print("=" * 50)

        conn = sqlite3.connect(self.db_path)

        # 1. Check for unknown races
        self._check_unknown_races(conn)

        # 2. Check position integrity
        self._check_position_integrity(conn)

        # 3. Check for duplicate positions
        self._check_duplicate_positions(conn)

        # 4. Check for missing positions
        self._check_missing_positions(conn)

        # 5. Check race metadata consistency
        self._check_metadata_consistency(conn)

        conn.close()

        return self.issues_found

    def _check_unknown_races(self, conn):
        """Check for unknown/unidentified races"""
        query = """
        SELECT race_id, venue, race_number, race_date, field_size
        FROM race_metadata 
        WHERE venue = 'Unknown' OR venue = 'UNK' OR race_number = 0
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            issue = {
                "type": "unknown_races",
                "severity": "high",
                "count": len(df),
                "description": "Races with unknown venue or race number 0",
                "races": df.to_dict("records"),
            }
            self.issues_found.append(issue)
            print(f"❌ Found {len(df)} unknown/unidentified races")
        else:
            print("✅ No unknown races found")

    def _check_position_integrity(self, conn):
        """Check for position integrity issues"""
        query = """
        SELECT race_id, COUNT(*) as dog_count,
               COUNT(DISTINCT finish_position) as unique_positions,
               MIN(CAST(finish_position AS INTEGER)) as min_pos,
               MAX(CAST(finish_position AS INTEGER)) as max_pos
        FROM dog_race_data 
        WHERE finish_position IS NOT NULL AND finish_position != ''
        GROUP BY race_id
        HAVING unique_positions != dog_count OR min_pos != 1
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            issue = {
                "type": "position_integrity",
                "severity": "high",
                "count": len(df),
                "description": "Races with position integrity issues",
                "races": df.to_dict("records"),
            }
            self.issues_found.append(issue)
            print(f"❌ Found {len(df)} races with position integrity issues")
        else:
            print("✅ No position integrity issues found")

    def _check_duplicate_positions(self, conn):
        """Check for duplicate finish positions in races"""
        query = """
        SELECT race_id, finish_position, COUNT(*) as duplicate_count
        FROM dog_race_data 
        WHERE finish_position IS NOT NULL AND finish_position != ''
        GROUP BY race_id, finish_position
        HAVING COUNT(*) > 1
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            issue = {
                "type": "duplicate_positions",
                "severity": "medium",
                "count": len(df),
                "description": "Duplicate finish positions (dead heats not properly handled)",
                "duplicates": df.to_dict("records"),
            }
            self.issues_found.append(issue)
            print(f"⚠️  Found {len(df)} duplicate position occurrences")
        else:
            print("✅ No duplicate positions found")

    def _check_missing_positions(self, conn):
        """Check for missing consecutive positions"""
        query = """
        WITH race_positions AS (
            SELECT race_id, 
                   CAST(finish_position AS INTEGER) as pos,
                   COUNT(*) as dog_count
            FROM dog_race_data 
            WHERE finish_position IS NOT NULL AND finish_position != ''
            GROUP BY race_id, finish_position
        ),
        race_stats AS (
            SELECT race_id,
                   COUNT(*) as unique_positions,
                   MIN(pos) as min_pos,
                   MAX(pos) as max_pos,
                   SUM(dog_count) as total_dogs
            FROM race_positions
            GROUP BY race_id
        )
        SELECT race_id, unique_positions, min_pos, max_pos, total_dogs
        FROM race_stats
        WHERE max_pos - min_pos + 1 != unique_positions OR min_pos != 1
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            issue = {
                "type": "missing_positions",
                "severity": "high",
                "count": len(df),
                "description": "Races with missing consecutive positions",
                "races": df.to_dict("records"),
            }
            self.issues_found.append(issue)
            print(f"❌ Found {len(df)} races with missing positions")
        else:
            print("✅ No missing positions found")

    def _check_metadata_consistency(self, conn):
        """Check race metadata consistency"""
        # Check field_size vs actual dog count
        query = """
        SELECT rm.race_id, rm.field_size, COUNT(drd.id) as actual_dogs
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        GROUP BY rm.race_id, rm.field_size
        HAVING rm.field_size != COUNT(drd.id)
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            issue = {
                "type": "metadata_inconsistency",
                "severity": "low",
                "count": len(df),
                "description": "Field size mismatch with actual dog count",
                "races": df.to_dict("records"),
            }
            self.issues_found.append(issue)
            print(f"⚠️  Found {len(df)} races with field size inconsistencies")
        else:
            print("✅ No metadata inconsistencies found")

    def fix_data_quality_issues(self):
        """Fix identified data quality issues"""
        print("\n🔧 FIXING DATA QUALITY ISSUES")
        print("=" * 50)

        if not self.issues_found:
            print("ℹ️  No issues found to fix")
            return

        conn = sqlite3.connect(self.db_path)

        for issue in self.issues_found:
            if issue["type"] == "unknown_races":
                self._fix_unknown_races(conn, issue)
            elif issue["type"] == "duplicate_positions":
                self._fix_duplicate_positions(conn, issue)
            elif issue["type"] == "missing_positions":
                self._fix_missing_positions(conn, issue)
            elif issue["type"] == "metadata_inconsistency":
                self._fix_metadata_inconsistency(conn, issue)

        conn.close()

        return self.fixes_applied

    def _fix_unknown_races(self, conn, issue):
        """Fix unknown races by attempting to re-parse source data"""
        print(f"🔧 Fixing {issue['count']} unknown races...")

        for race in issue["races"]:
            race_id = race["race_id"]

            # Try to extract venue and race number from race_id
            if race_id.startswith("UNK_0_UNKNOWN"):
                # This is the problematic race - mark for manual review
                print(f"   ⚠️  {race_id}: Marked for manual review (complex mixed data)")
                self._mark_race_for_review(
                    conn, race_id, "Unknown race with mixed data sources"
                )
            else:
                # Try to parse from other race_ids
                parsed_info = self._parse_race_id(race_id)
                if parsed_info:
                    self._update_race_metadata(conn, race_id, parsed_info)
                    print(
                        f"   ✅ {race_id}: Updated venue to {parsed_info['venue']}, race {parsed_info['race_number']}"
                    )

    def _fix_duplicate_positions(self, conn, issue):
        """Fix duplicate positions by implementing proper dead heat handling"""
        print(f"🔧 Fixing {issue['count']} duplicate position occurrences...")

        for dup in issue["duplicates"]:
            race_id = dup["race_id"]
            position = dup["finish_position"]
            count = dup["duplicate_count"]

            # Get all dogs in this position
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, dog_name, individual_time, margin
                FROM dog_race_data 
                WHERE race_id = ? AND finish_position = ?
                ORDER BY 
                    CASE 
                        WHEN individual_time IS NOT NULL AND individual_time != '' 
                        THEN CAST(individual_time AS REAL) 
                        ELSE 999999 
                    END,
                    CASE 
                        WHEN margin IS NOT NULL AND margin != '' 
                        THEN CAST(margin AS REAL) 
                        ELSE 999999 
                    END
            """,
                (race_id, position),
            )

            dogs = cursor.fetchall()

            if len(dogs) == count:
                # Check if it's a genuine dead heat (same time) or parsing error
                times = [
                    float(dog[2]) if dog[2] and dog[2] != "" else None for dog in dogs
                ]
                unique_times = set(t for t in times if t is not None)

                if len(unique_times) == 1 and len(dogs) <= 3:
                    # Genuine dead heat - mark appropriately
                    for dog in dogs:
                        cursor.execute(
                            """
                            UPDATE dog_race_data 
                            SET finish_position = ?, 
                                data_quality_note = 'Dead heat - tied for position'
                            WHERE id = ?
                        """,
                            (f"{position}=", dog[0]),
                        )
                    print(
                        f"   ✅ {race_id}: Marked {count} dogs as dead heat at position {position}"
                    )
                else:
                    # Parsing error - reassign positions based on time/margin
                    for i, dog in enumerate(dogs):
                        new_position = int(position) + i
                        cursor.execute(
                            """
                            UPDATE dog_race_data 
                            SET finish_position = ?,
                                data_quality_note = 'Position corrected from duplicate'
                            WHERE id = ?
                        """,
                            (str(new_position), dog[0]),
                        )
                    print(
                        f"   ✅ {race_id}: Reassigned {count} dogs from position {position} to {position}-{int(position)+count-1}"
                    )

            conn.commit()

    def _fix_missing_positions(self, conn, issue):
        """Fix missing positions by interpolating or marking gaps"""
        print(f"🔧 Fixing {issue['count']} races with missing positions...")

        for race in issue["races"]:
            race_id = race["race_id"]

            # Get all positions for this race
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT CAST(finish_position AS INTEGER) as pos
                FROM dog_race_data 
                WHERE race_id = ? AND finish_position IS NOT NULL AND finish_position != ''
                ORDER BY pos
            """,
                (race_id,),
            )

            positions = [row[0] for row in cursor.fetchall()]

            if positions:
                min_pos, max_pos = min(positions), max(positions)
                expected_positions = set(range(1, max_pos + 1))
                missing_positions = expected_positions - set(positions)

                if missing_positions:
                    print(
                        f"   ⚠️  {race_id}: Missing positions {sorted(missing_positions)} - marked for manual review"
                    )
                    self._mark_race_for_review(
                        conn, race_id, f"Missing positions: {sorted(missing_positions)}"
                    )

                if 1 not in positions:
                    print(
                        f"   ❌ {race_id}: Missing winner (position 1) - marked for urgent review"
                    )
                    self._mark_race_for_review(
                        conn, race_id, "Missing winner - position 1 not found"
                    )

    def _fix_metadata_inconsistency(self, conn, issue):
        """Fix metadata inconsistencies"""
        print(f"🔧 Fixing {issue['count']} metadata inconsistencies...")

        for race in issue["races"]:
            race_id = race["race_id"]
            actual_dogs = race["actual_dogs"]

            # Update field_size to match actual dog count
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE race_metadata 
                SET field_size = ?,
                    data_quality_note = 'Field size corrected to match actual dog count'
                WHERE race_id = ?
            """,
                (actual_dogs, race_id),
            )

            print(f"   ✅ {race_id}: Updated field_size to {actual_dogs}")

        conn.commit()

    def _parse_race_id(self, race_id):
        """Parse race information from race_id"""
        patterns = [
            r"([A-Z]+)_(\d+)_(.+)",  # VENUE_NUMBER_DATE
            r"([a-z]+)_(\d{4}-\d{2}-\d{2})_(\d+)",  # venue_date_number
        ]

        for pattern in patterns:
            match = re.match(pattern, race_id)
            if match:
                if len(match.groups()) == 3:
                    return {
                        "venue": match.group(1).upper(),
                        "race_number": (
                            int(match.group(2)) if match.group(2).isdigit() else 0
                        ),
                        "date_str": match.group(3),
                    }

        return None

    def _update_race_metadata(self, conn, race_id, parsed_info):
        """Update race metadata with parsed information"""
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE race_metadata 
            SET venue = ?, race_number = ?, data_quality_note = 'Venue and race number parsed from race_id'
            WHERE race_id = ?
        """,
            (parsed_info["venue"], parsed_info["race_number"], race_id),
        )
        conn.commit()

    def _mark_race_for_review(self, conn, race_id, note):
        """Mark a race for manual review"""
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE race_metadata 
            SET data_quality_note = ?, race_status = 'needs_review'
            WHERE race_id = ?
        """,
            (note, race_id),
        )
        conn.commit()

        self.fixes_applied.append(
            {
                "race_id": race_id,
                "action": "marked_for_review",
                "note": note,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def validate_data_quality(self):
        """Validate data quality after fixes"""
        print("\n✅ VALIDATING DATA QUALITY")
        print("=" * 50)

        conn = sqlite3.connect(self.db_path)

        validation_results = {}

        # Check race completeness
        query = """
        SELECT 
            COUNT(*) as total_races,
            COUNT(CASE WHEN venue != 'Unknown' AND venue != 'UNK' THEN 1 END) as identified_races,
            COUNT(CASE WHEN race_number > 0 THEN 1 END) as numbered_races,
            COUNT(CASE WHEN race_status = 'needs_review' THEN 1 END) as needs_review
        FROM race_metadata
        """

        df = pd.read_sql_query(query, conn)
        validation_results["race_metadata"] = df.iloc[0].to_dict()

        # Check position completeness
        query = """
        SELECT 
            race_id,
            COUNT(*) as total_dogs,
            COUNT(CASE WHEN finish_position IS NOT NULL AND finish_position != '' THEN 1 END) as dogs_with_positions,
            MIN(CAST(finish_position AS INTEGER)) as min_position,
            MAX(CAST(finish_position AS INTEGER)) as max_position
        FROM dog_race_data
        GROUP BY race_id
        """

        df = pd.read_sql_query(query, conn)
        validation_results["position_completeness"] = {
            "total_races": len(df),
            "races_with_all_positions": len(
                df[df["total_dogs"] == df["dogs_with_positions"]]
            ),
            "races_with_winner": len(df[df["min_position"] == 1]),
            "races_missing_positions": len(
                df[df["total_dogs"] != df["dogs_with_positions"]]
            ),
        }

        conn.close()

        # Print validation summary
        rm = validation_results["race_metadata"]
        pc = validation_results["position_completeness"]

        print(f"📊 RACE METADATA:")
        print(f"   Total races: {rm['total_races']}")
        print(
            f"   Identified venues: {rm['identified_races']} ({rm['identified_races']/rm['total_races']*100:.1f}%)"
        )
        print(
            f"   Numbered races: {rm['numbered_races']} ({rm['numbered_races']/rm['total_races']*100:.1f}%)"
        )
        print(f"   Needs review: {rm['needs_review']}")

        print(f"\n📊 POSITION DATA:")
        print(f"   Total races: {pc['total_races']}")
        print(
            f"   Complete positions: {pc['races_with_all_positions']} ({pc['races_with_all_positions']/pc['total_races']*100:.1f}%)"
        )
        print(
            f"   Has winner: {pc['races_with_winner']} ({pc['races_with_winner']/pc['total_races']*100:.1f}%)"
        )
        print(f"   Missing positions: {pc['races_missing_positions']}")

        return validation_results

    def generate_report(self, output_file="data_quality_report.json"):
        """Generate comprehensive data quality report"""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "database_path": self.db_path,
            "issues_found": self.issues_found,
            "fixes_applied": self.fixes_applied,
            "summary": {
                "total_issues": len(self.issues_found),
                "high_severity": len(
                    [i for i in self.issues_found if i["severity"] == "high"]
                ),
                "medium_severity": len(
                    [i for i in self.issues_found if i["severity"] == "medium"]
                ),
                "low_severity": len(
                    [i for i in self.issues_found if i["severity"] == "low"]
                ),
                "fixes_applied": len(self.fixes_applied),
            },
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Data quality report saved to: {output_file}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Fix data quality issues in greyhound racing database"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze data quality issues"
    )
    parser.add_argument("--fix", action="store_true", help="Fix data quality issues")
    parser.add_argument("--validate", action="store_true", help="Validate data quality")
    parser.add_argument(
        "--db", default="greyhound_racing_data.db", help="Database path"
    )
    parser.add_argument(
        "--report", default="data_quality_report.json", help="Report output file"
    )

    args = parser.parse_args()

    if not any([args.analyze, args.fix, args.validate]):
        # Run all by default
        args.analyze = args.fix = args.validate = True

    fixer = DataQualityFixer(args.db)

    if args.analyze:
        fixer.analyze_data_quality()

    if args.fix:
        fixer.fix_data_quality_issues()

    if args.validate:
        fixer.validate_data_quality()

    # Always generate report
    fixer.generate_report(args.report)


if __name__ == "__main__":
    main()
