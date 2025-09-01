#!/usr/bin/env python3
"""
Referential Integrity & Relationship Validation Script
Step 3: Comprehensive database integrity analysis for greyhound racing data

This script performs:
1. Foreign key relationship validation using left-join anti-joins to detect orphans
2. Quantifies orphan counts and percentages per table
3. Validates data consistency across related tables
4. Checks for many-to-many relationship integrity
"""

import sqlite3
import sys
from datetime import datetime

import pandas as pd


class ReferentialIntegrityChecker:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}

    def execute_query(self, query, description=""):
        """Execute SQL query and return results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"Error executing query: {description}")
            print(f"Query: {query}")
            print(f"Error: {e}")
            return pd.DataFrame()

    def check_table_counts(self):
        """Get basic table statistics"""
        print("=" * 60)
        print("TABLE COUNTS & BASIC STATISTICS")
        print("=" * 60)

        tables = [
            "races",
            "dog_performances",
            "dogs",
            "venues",
            "predictions",
            "form_guide",
        ]
        counts = {}

        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query, f"Count for {table}")
            count = result["count"].iloc[0] if not result.empty else 0
            counts[table] = count
            print(f"{table:20}: {count:,} records")

        self.results["table_counts"] = counts
        return counts

    def check_fk_orphans_dog_performances_to_races(self):
        """Check for orphaned dog_performances records (no matching race_id)"""
        print("\n" + "=" * 60)
        print("FK ORPHANS: dog_performances → races")
        print("=" * 60)

        # Left anti-join to find orphaned dog_performances
        query = """
        SELECT dp.performance_id, dp.race_id, dp.dog_name, dp.created_at
        FROM dog_performances dp
        LEFT JOIN races r ON dp.race_id = r.race_id
        WHERE r.race_id IS NULL
        ORDER BY dp.performance_id
        """

        orphans = self.execute_query(query, "Dog performances without matching races")

        # Get totals for percentage calculation
        total_performances = self.execute_query(
            "SELECT COUNT(*) as count FROM dog_performances"
        )["count"].iloc[0]
        orphan_count = len(orphans)
        orphan_percentage = (
            (orphan_count / total_performances * 100) if total_performances > 0 else 0
        )

        print(f"Total dog_performances records: {total_performances:,}")
        print(f"Orphaned records (no matching race): {orphan_count:,}")
        print(f"Orphan percentage: {orphan_percentage:.2f}%")

        if orphan_count > 0:
            print(f"\nFirst 10 orphaned records:")
            print(orphans.head(10).to_string(index=False))

        self.results["dog_performances_orphans"] = {
            "count": orphan_count,
            "percentage": orphan_percentage,
            "orphans": orphans,
        }

        return orphans

    def check_fk_orphans_predictions_to_races(self):
        """Check for orphaned predictions records (no matching race_id)"""
        print("\n" + "=" * 60)
        print("FK ORPHANS: predictions → races")
        print("=" * 60)

        query = """
        SELECT p.prediction_id, p.race_id, p.dog_name, p.prediction_date
        FROM predictions p
        LEFT JOIN races r ON p.race_id = r.race_id
        WHERE r.race_id IS NULL
        ORDER BY p.prediction_id
        """

        orphans = self.execute_query(query, "Predictions without matching races")

        total_predictions = self.execute_query(
            "SELECT COUNT(*) as count FROM predictions"
        )["count"].iloc[0]
        orphan_count = len(orphans)
        orphan_percentage = (
            (orphan_count / total_predictions * 100) if total_predictions > 0 else 0
        )

        print(f"Total predictions records: {total_predictions:,}")
        print(f"Orphaned records (no matching race): {orphan_count:,}")
        print(f"Orphan percentage: {orphan_percentage:.2f}%")

        if orphan_count > 0:
            print(f"\nFirst 10 orphaned records:")
            print(orphans.head(10).to_string(index=False))

        self.results["predictions_orphans"] = {
            "count": orphan_count,
            "percentage": orphan_percentage,
            "orphans": orphans,
        }

        return orphans

    def check_dog_name_consistency(self):
        """Check for dog name consistency across tables"""
        print("\n" + "=" * 60)
        print("DOG NAME CONSISTENCY ANALYSIS")
        print("=" * 60)

        # Dogs in dog_performances but not in dogs table
        query1 = """
        SELECT DISTINCT dp.dog_name, COUNT(*) as performance_count
        FROM dog_performances dp
        LEFT JOIN dogs d ON dp.dog_name = d.dog_name
        WHERE d.dog_name IS NULL
        GROUP BY dp.dog_name
        ORDER BY performance_count DESC
        """

        missing_from_dogs = self.execute_query(
            query1, "Dogs in performances but not in dogs table"
        )

        # Dogs in dogs table but not in dog_performances
        query2 = """
        SELECT d.dog_name, d.total_races, d.total_wins
        FROM dogs d
        LEFT JOIN dog_performances dp ON d.dog_name = dp.dog_name
        WHERE dp.dog_name IS NULL
        ORDER BY d.total_races DESC
        """

        missing_from_performances = self.execute_query(
            query2, "Dogs in dogs table but not in performances"
        )

        # Dogs in form_guide but not in dogs table
        query3 = """
        SELECT DISTINCT fg.dog_name, COUNT(*) as form_count
        FROM form_guide fg
        LEFT JOIN dogs d ON fg.dog_name = d.dog_name
        WHERE d.dog_name IS NULL
        GROUP BY fg.dog_name
        ORDER BY form_count DESC
        LIMIT 20
        """

        missing_from_dogs_form = self.execute_query(
            query3, "Dogs in form_guide but not in dogs table"
        )

        print(
            f"Dogs in dog_performances but not in dogs table: {len(missing_from_dogs):,}"
        )
        if len(missing_from_dogs) > 0:
            print("Top 10 missing dogs:")
            print(missing_from_dogs.head(10).to_string(index=False))

        print(
            f"\nDogs in dogs table but not in dog_performances: {len(missing_from_performances):,}"
        )
        if len(missing_from_performances) > 0:
            print("Top 10:")
            print(missing_from_performances.head(10).to_string(index=False))

        print(
            f"\nDogs in form_guide but not in dogs table: {len(missing_from_dogs_form):,}"
        )
        if len(missing_from_dogs_form) > 0:
            print("Top 10:")
            print(missing_from_dogs_form.head(10).to_string(index=False))

        self.results["dog_name_consistency"] = {
            "missing_from_dogs_table": len(missing_from_dogs),
            "missing_from_performances": len(missing_from_performances),
            "missing_from_dogs_form": len(missing_from_dogs_form),
        }

    def check_venue_consistency(self):
        """Check venue consistency and orphaned venue references"""
        print("\n" + "=" * 60)
        print("VENUE CONSISTENCY ANALYSIS")
        print("=" * 60)

        # Venues in races but not in venues table
        query1 = """
        SELECT DISTINCT r.venue, COUNT(*) as race_count
        FROM races r
        LEFT JOIN venues v ON r.venue = v.venue_code OR r.venue = v.venue_name
        WHERE v.venue_code IS NULL AND v.venue_name IS NULL
        GROUP BY r.venue
        ORDER BY race_count DESC
        """

        missing_venues = self.execute_query(
            query1, "Venues in races but not in venues table"
        )

        # Venues in form_guide but not in venues table
        query2 = """
        SELECT DISTINCT fg.venue, COUNT(*) as form_count
        FROM form_guide fg
        LEFT JOIN venues v ON fg.venue = v.venue_code OR fg.venue = v.venue_name
        WHERE v.venue_code IS NULL AND v.venue_name IS NULL
        GROUP BY fg.venue
        ORDER BY form_count DESC
        LIMIT 20
        """

        missing_venues_form = self.execute_query(
            query2, "Venues in form_guide but not in venues table"
        )

        print(f"Venues in races but not in venues table: {len(missing_venues):,}")
        if len(missing_venues) > 0:
            print("Missing venues from races:")
            print(missing_venues.to_string(index=False))

        print(
            f"\nVenues in form_guide but not in venues table: {len(missing_venues_form):,}"
        )
        if len(missing_venues_form) > 0:
            print("Top 10 missing venues from form_guide:")
            print(missing_venues_form.head(10).to_string(index=False))

        self.results["venue_consistency"] = {
            "missing_from_venues_table_races": len(missing_venues),
            "missing_from_venues_table_form": len(missing_venues_form),
        }

    def check_data_uniqueness_constraints(self):
        """Check for potential duplicate data and uniqueness violations"""
        print("\n" + "=" * 60)
        print("DATA UNIQUENESS & DUPLICATE ANALYSIS")
        print("=" * 60)

        # Check for duplicate races
        query1 = """
        SELECT race_name, venue, race_date, COUNT(*) as duplicate_count
        FROM races
        GROUP BY race_name, venue, race_date
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        """

        duplicate_races = self.execute_query(query1, "Duplicate races")

        # Check for duplicate dog performances in same race
        query2 = """
        SELECT race_id, dog_name, box_number, COUNT(*) as duplicate_count
        FROM dog_performances
        GROUP BY race_id, dog_name, box_number
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        """

        duplicate_performances = self.execute_query(
            query2, "Duplicate dog performances"
        )

        # Check for duplicate dogs
        query3 = """
        SELECT dog_name, COUNT(*) as duplicate_count
        FROM dogs
        GROUP BY dog_name
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        """

        duplicate_dogs = self.execute_query(query3, "Duplicate dogs")

        print(f"Duplicate races (same name, venue, date): {len(duplicate_races):,}")
        if len(duplicate_races) > 0:
            print(duplicate_races.to_string(index=False))

        print(
            f"\nDuplicate dog performances (same race, dog, box): {len(duplicate_performances):,}"
        )
        if len(duplicate_performances) > 0:
            print(duplicate_performances.head(10).to_string(index=False))

        print(f"\nDuplicate dogs (same name): {len(duplicate_dogs):,}")
        if len(duplicate_dogs) > 0:
            print(duplicate_dogs.to_string(index=False))

        self.results["duplicates"] = {
            "duplicate_races": len(duplicate_races),
            "duplicate_performances": len(duplicate_performances),
            "duplicate_dogs": len(duplicate_dogs),
        }

    def check_cross_table_data_consistency(self):
        """Check consistency of data across related tables"""
        print("\n" + "=" * 60)
        print("CROSS-TABLE DATA CONSISTENCY")
        print("=" * 60)

        # Check if dog statistics match performance records
        query = """
        SELECT 
            d.dog_name,
            d.total_races as dogs_table_races,
            COUNT(dp.performance_id) as actual_performances,
            d.total_wins as dogs_table_wins,
            SUM(CASE WHEN dp.finish_position = 1 THEN 1 ELSE 0 END) as actual_wins,
            d.total_places as dogs_table_places,
            SUM(CASE WHEN dp.finish_position <= 3 THEN 1 ELSE 0 END) as actual_places
        FROM dogs d
        LEFT JOIN dog_performances dp ON d.dog_name = dp.dog_name
        GROUP BY d.dog_name, d.total_races, d.total_wins, d.total_places
        HAVING 
            d.total_races != COUNT(dp.performance_id) OR
            d.total_wins != SUM(CASE WHEN dp.finish_position = 1 THEN 1 ELSE 0 END) OR
            d.total_places != SUM(CASE WHEN dp.finish_position <= 3 THEN 1 ELSE 0 END)
        ORDER BY ABS(d.total_races - COUNT(dp.performance_id)) DESC
        LIMIT 20
        """

        inconsistent_stats = self.execute_query(
            query, "Dogs with inconsistent statistics"
        )

        print(f"Dogs with inconsistent statistics: {len(inconsistent_stats):,}")
        if len(inconsistent_stats) > 0:
            print("Top 10 dogs with statistical inconsistencies:")
            print(inconsistent_stats.head(10).to_string(index=False))

        self.results["statistical_consistency"] = {
            "inconsistent_dogs": len(inconsistent_stats)
        }

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("REFERENTIAL INTEGRITY SUMMARY REPORT")
        print("=" * 60)

        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.db_path}")

        print("\n📊 TABLE COUNTS:")
        for table, count in self.results["table_counts"].items():
            print(f"  {table:20}: {count:,}")

        print("\n🔗 FOREIGN KEY INTEGRITY:")
        dp_orphans = self.results.get("dog_performances_orphans", {})
        pred_orphans = self.results.get("predictions_orphans", {})
        print(
            f"  dog_performances orphans: {dp_orphans.get('count', 0):,} ({dp_orphans.get('percentage', 0):.2f}%)"
        )
        print(
            f"  predictions orphans     : {pred_orphans.get('count', 0):,} ({pred_orphans.get('percentage', 0):.2f}%)"
        )

        print("\n🐕 DOG NAME CONSISTENCY:")
        dog_consistency = self.results.get("dog_name_consistency", {})
        print(
            f"  Missing from dogs table (performances): {dog_consistency.get('missing_from_dogs_table', 0):,}"
        )
        print(
            f"  Missing from performances: {dog_consistency.get('missing_from_performances', 0):,}"
        )
        print(
            f"  Missing from dogs table (form_guide): {dog_consistency.get('missing_from_dogs_form', 0):,}"
        )

        print("\n🏟️ VENUE CONSISTENCY:")
        venue_consistency = self.results.get("venue_consistency", {})
        print(
            f"  Missing from venues table (races): {venue_consistency.get('missing_from_venues_table_races', 0):,}"
        )
        print(
            f"  Missing from venues table (form): {venue_consistency.get('missing_from_venues_table_form', 0):,}"
        )

        print("\n🔄 DUPLICATE DATA:")
        duplicates = self.results.get("duplicates", {})
        print(f"  Duplicate races: {duplicates.get('duplicate_races', 0):,}")
        print(
            f"  Duplicate performances: {duplicates.get('duplicate_performances', 0):,}"
        )
        print(f"  Duplicate dogs: {duplicates.get('duplicate_dogs', 0):,}")

        print("\n📈 STATISTICAL CONSISTENCY:")
        stats_consistency = self.results.get("statistical_consistency", {})
        print(
            f"  Dogs with inconsistent stats: {stats_consistency.get('inconsistent_dogs', 0):,}"
        )

        # Overall integrity score
        total_issues = (
            dp_orphans.get("count", 0)
            + pred_orphans.get("count", 0)
            + dog_consistency.get("missing_from_dogs_table", 0)
            + venue_consistency.get("missing_from_venues_table_races", 0)
            + duplicates.get("duplicate_races", 0)
            + duplicates.get("duplicate_performances", 0)
            + stats_consistency.get("inconsistent_dogs", 0)
        )

        print(f"\n🎯 OVERALL INTEGRITY:")
        print(f"  Total integrity issues: {total_issues:,}")
        if total_issues == 0:
            print("  ✅ Database appears to have good referential integrity!")
        elif total_issues < 100:
            print("  ⚠️ Minor integrity issues detected")
        else:
            print("  ❌ Significant integrity issues require attention")

    def run_full_analysis(self):
        """Run the complete referential integrity analysis"""
        print("Starting Referential Integrity Analysis...")
        print(f"Database: {self.db_path}")

        self.check_table_counts()
        self.check_fk_orphans_dog_performances_to_races()
        self.check_fk_orphans_predictions_to_races()
        self.check_dog_name_consistency()
        self.check_venue_consistency()
        self.check_data_uniqueness_constraints()
        self.check_cross_table_data_consistency()
        self.generate_summary_report()

        return self.results

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    db_path = "./databases/race_data.db"

    checker = ReferentialIntegrityChecker(db_path)

    try:
        results = checker.run_full_analysis()

        # Save results to file for future reference
        import json

        with open("referential_integrity_results.json", "w") as f:
            # Convert DataFrames to dictionaries for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for subkey, subvalue in value.items():
                        if hasattr(subvalue, "to_dict"):  # DataFrame
                            serializable_results[key][subkey] = subvalue.to_dict(
                                "records"
                            )
                        else:
                            serializable_results[key][subkey] = subvalue
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: referential_integrity_results.json")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()

    finally:
        checker.close()


if __name__ == "__main__":
    main()
