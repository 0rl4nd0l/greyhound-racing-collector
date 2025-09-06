#!/usr/bin/env python3
"""
Deep Dive Investigation of Temporal Anomalies
Investigates specific issues found in temporal coverage analysis and provides actionable recommendations
"""

import sqlite3
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


class AnomalyInvestigator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def investigate_gap_period(self):
        """Investigate the July 19-22 gap period in detail"""
        print("🔍 INVESTIGATING JULY 19-22 GAP PERIOD")
        print("=" * 50)

        # Check races around the gap period
        query = """
        SELECT race_date, COUNT(*) as race_count, 
               GROUP_CONCAT(venue) as venues,
               MIN(created_at) as first_created,
               MAX(created_at) as last_created
        FROM races 
        WHERE race_date BETWEEN '2025-07-15' AND '2025-07-25'
        GROUP BY race_date
        ORDER BY race_date
        """

        gap_data = pd.read_sql_query(query, self.conn)
        print("Race activity around the gap period:")
        print(gap_data.to_string(index=False))

        # Check if any races exist with invalid dates that might belong to gap period
        invalid_date_query = """
        SELECT race_date, venue, race_name, created_at
        FROM races 
        WHERE race_date NOT LIKE '____-__-__'
        OR race_date IS NULL
        """

        invalid_dates = pd.read_sql_query(invalid_date_query, self.conn)
        if not invalid_dates.empty:
            print(f"\n⚠️  Found {len(invalid_dates)} races with invalid dates:")
            print(invalid_dates.to_string(index=False))

        # Analyze venue patterns during gap
        venue_analysis = """
        SELECT venue, 
               SUM(CASE WHEN race_date < '2025-07-19' THEN 1 ELSE 0 END) as before_gap,
               SUM(CASE WHEN race_date >= '2025-07-23' THEN 1 ELSE 0 END) as after_gap
        FROM races 
        WHERE race_date BETWEEN '2025-07-15' AND '2025-07-25'
        GROUP BY venue
        HAVING before_gap > 0 OR after_gap > 0
        ORDER BY (before_gap + after_gap) DESC
        """

        venue_patterns = pd.read_sql_query(venue_analysis, self.conn)
        print(f"\n📍 Venue activity patterns around gap:")
        print(venue_patterns.to_string(index=False))

        return gap_data, invalid_dates, venue_patterns

    def investigate_retroactive_data(self):
        """Investigate the retroactive data issue for Race ID 143"""
        print("\n🔍 INVESTIGATING RETROACTIVE DATA ISSUE (Race ID 143)")
        print("=" * 60)

        # Get details of Race 143 and surrounding races
        query = """
        SELECT race_id, race_name, venue, race_date, created_at,
               LAG(race_date) OVER (ORDER BY created_at) as prev_race_date,
               LAG(created_at) OVER (ORDER BY created_at) as prev_created_at,
               LEAD(race_date) OVER (ORDER BY created_at) as next_race_date,
               LEAD(created_at) OVER (ORDER BY created_at) as next_created_at
        FROM races 
        WHERE race_id BETWEEN 140 AND 146
        ORDER BY created_at
        """

        retroactive_context = pd.read_sql_query(query, self.conn)
        print("Context around Race 143:")
        print(retroactive_context.to_string(index=False))

        # Check for patterns in retroactive edits
        retroactive_pattern_query = """
        WITH race_sequence AS (
            SELECT race_id, race_date, created_at,
                   LAG(race_date) OVER (ORDER BY created_at) as prev_race_date,
                   LAG(created_at) OVER (ORDER BY created_at) as prev_created_at
            FROM races
            ORDER BY created_at
        )
        SELECT COUNT(*) as retroactive_count
        FROM race_sequence
        WHERE race_date < prev_race_date AND created_at > prev_created_at
        """

        retroactive_count = pd.read_sql_query(retroactive_pattern_query, self.conn)
        print(f"\nTotal retroactive data instances: {retroactive_count.iloc[0, 0]}")

        return retroactive_context

    def analyze_data_collection_patterns(self):
        """Analyze data collection patterns and identify potential issues"""
        print("\n🔍 ANALYZING DATA COLLECTION PATTERNS")
        print("=" * 50)

        # Daily collection patterns
        daily_pattern_query = """
        SELECT 
            DATE(created_at) as collection_date,
            COUNT(*) as races_collected,
            COUNT(DISTINCT venue) as unique_venues,
            MIN(race_date) as earliest_race_date,
            MAX(race_date) as latest_race_date,
            (julianday(MAX(race_date)) - julianday(MIN(race_date))) as date_span_days
        FROM races
        GROUP BY DATE(created_at)
        ORDER BY collection_date
        """

        collection_patterns = pd.read_sql_query(daily_pattern_query, self.conn)
        print("Daily collection patterns:")
        print(collection_patterns.to_string(index=False))

        # Check for batch loading patterns
        batch_analysis = """
        SELECT 
            DATETIME(created_at, 'start of hour') as hour_bucket,
            COUNT(*) as races_in_hour,
            COUNT(DISTINCT venue) as venues_in_hour,
            MIN(race_date) as min_race_date,
            MAX(race_date) as max_race_date
        FROM races
        GROUP BY DATETIME(created_at, 'start of hour')
        HAVING races_in_hour > 10
        ORDER BY races_in_hour DESC
        """

        batch_patterns = pd.read_sql_query(batch_analysis, self.conn)
        print(f"\nLarge batch collections (>10 races per hour):")
        print(batch_patterns.to_string(index=False))

        return collection_patterns, batch_patterns

    def identify_schema_changes(self):
        """Identify potential schema changes based on data patterns"""
        print("\n🔍 IDENTIFYING POTENTIAL SCHEMA CHANGES")
        print("=" * 50)

        # Check for changes in venue naming patterns
        venue_evolution_query = """
        SELECT 
            DATE(created_at) as collection_date,
            GROUP_CONCAT(DISTINCT venue) as venues_collected,
            COUNT(DISTINCT venue) as venue_count
        FROM races
        GROUP BY DATE(created_at)
        ORDER BY collection_date
        """

        venue_evolution = pd.read_sql_query(venue_evolution_query, self.conn)

        # Identify unusual venue patterns
        unusual_venues = []
        for idx, row in venue_evolution.iterrows():
            venues = str(row["venues_collected"]).split(",")
            for venue in venues:
                venue = venue.strip()
                # Check for venues that don't follow typical patterns
                if (
                    len(venue) < 3
                    or venue.upper() == venue
                    or not venue.replace(" ", "").isalnum()
                ):
                    if venue not in ["AP", "GRDN"]:  # Known exceptions
                        unusual_venues.append(
                            {
                                "date": row["collection_date"],
                                "venue": venue,
                                "reason": "unusual_pattern",
                            }
                        )

        if unusual_venues:
            print("Unusual venue patterns detected:")
            for venue_issue in unusual_venues:
                print(
                    f"  {venue_issue['date']}: '{venue_issue['venue']}' ({venue_issue['reason']})"
                )
        else:
            print("No unusual venue patterns detected.")

        # Check for data quality degradation over time
        quality_timeline = """
        SELECT 
            DATE(created_at) as collection_date,
            COUNT(*) as total_races,
            SUM(CASE WHEN race_name IS NULL OR race_name = '' THEN 1 ELSE 0 END) as missing_names,
            SUM(CASE WHEN venue IS NULL OR venue = '' THEN 1 ELSE 0 END) as missing_venues,
            SUM(CASE WHEN race_date IS NULL OR race_date = '' THEN 1 ELSE 0 END) as missing_dates,
            ROUND(AVG(LENGTH(race_name)), 1) as avg_name_length
        FROM races
        GROUP BY DATE(created_at)
        ORDER BY collection_date
        """

        quality_data = pd.read_sql_query(quality_timeline, self.conn)
        print(f"\nData quality timeline:")
        print(quality_data.to_string(index=False))

        return venue_evolution, unusual_venues, quality_data

    def generate_recommendations(self):
        """Generate actionable recommendations based on findings"""
        print("\n📋 RECOMMENDATIONS FOR TEMPORAL COVERAGE IMPROVEMENT")
        print("=" * 70)

        recommendations = [
            {
                "priority": "HIGH",
                "category": "Data Gaps",
                "issue": "4-day gap period (July 19-22, 2025)",
                "recommendation": "Implement gap detection alerts in scraping pipeline",
                "action_items": [
                    "Add daily data collection monitoring",
                    "Set up alerts for missing data periods > 2 days",
                    "Create backfill mechanism for detected gaps",
                    "Verify if gap was due to no races or scraping failure",
                ],
            },
            {
                "priority": "MEDIUM",
                "category": "Data Consistency",
                "issue": "Retroactive data insertion detected",
                "recommendation": "Implement timestamp validation and audit logging",
                "action_items": [
                    "Add race_date vs created_at validation rules",
                    "Log all data updates with before/after states",
                    "Flag records where race_date < previous race_date",
                    "Consider implementing data versioning",
                ],
            },
            {
                "priority": "MEDIUM",
                "category": "Data Quality",
                "issue": "Invalid venue codes (AP, GRDN) detected",
                "recommendation": "Standardize venue naming and validation",
                "action_items": [
                    "Create venue master data table",
                    "Implement venue code validation",
                    "Map abbreviations to full venue names",
                    "Add venue data quality checks",
                ],
            },
            {
                "priority": "LOW",
                "category": "Monitoring",
                "issue": "All data collected in single batch",
                "recommendation": "Implement real-time collection monitoring",
                "action_items": [
                    "Add collection timestamp granularity",
                    "Monitor for unusual batch patterns",
                    "Implement distributed collection timing",
                    "Track extraction vs race date correlations",
                ],
            },
        ]

        for rec in recommendations:
            print(f"\n🎯 {rec['priority']} PRIORITY - {rec['category']}")
            print(f"Issue: {rec['issue']}")
            print(f"Recommendation: {rec['recommendation']}")
            print("Action Items:")
            for item in rec["action_items"]:
                print(f"  ✓ {item}")

        return recommendations

    def create_monitoring_queries(self):
        """Create SQL queries for ongoing temporal monitoring"""
        print("\n📊 MONITORING QUERIES FOR ONGOING TEMPORAL ANALYSIS")
        print("=" * 60)

        queries = {
            "daily_gap_detection": """
            -- Daily Gap Detection Query
            WITH date_series AS (
                SELECT date(julianday('now') - days.value) as check_date
                FROM (
                    SELECT 0 as value UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 
                    UNION SELECT 4 UNION SELECT 5 UNION SELECT 6
                ) days
            ),
            daily_counts AS (
                SELECT race_date, COUNT(*) as race_count
                FROM races
                WHERE race_date >= date('now', '-7 days')
                GROUP BY race_date
            )
            SELECT ds.check_date, 
                   COALESCE(dc.race_count, 0) as races,
                   CASE WHEN dc.race_count IS NULL THEN 'GAP DETECTED' ELSE 'OK' END as status
            FROM date_series ds
            LEFT JOIN daily_counts dc ON ds.check_date = dc.race_date
            ORDER BY ds.check_date DESC;
            """,
            "retroactive_data_detection": """
            -- Retroactive Data Detection Query
            WITH race_sequence AS (
                SELECT race_id, race_date, created_at,
                       LAG(race_date) OVER (ORDER BY created_at) as prev_race_date,
                       LAG(created_at) OVER (ORDER BY created_at) as prev_created_at
                FROM races
                WHERE created_at >= datetime('now', '-7 days')
                ORDER BY created_at
            )
            SELECT race_id, race_date, created_at, prev_race_date,
                   'RETROACTIVE DATA' as alert_type
            FROM race_sequence
            WHERE race_date < prev_race_date AND created_at > prev_created_at;
            """,
            "venue_anomaly_detection": """
            -- Venue Anomaly Detection Query
            SELECT venue, COUNT(*) as race_count,
                   MIN(race_date) as first_race, 
                   MAX(race_date) as last_race,
                   CASE 
                       WHEN LENGTH(venue) < 3 THEN 'SHORT_NAME'
                       WHEN venue = UPPER(venue) AND LENGTH(venue) <= 4 THEN 'ABBREVIATION'
                       WHEN venue NOT GLOB '*[A-Za-z]*' THEN 'NON_ALPHABETIC'
                       ELSE 'NORMAL'
                   END as venue_type
            FROM races
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY venue
            HAVING venue_type != 'NORMAL'
            ORDER BY race_count DESC;
            """,
            "collection_pattern_monitoring": """
            -- Collection Pattern Monitoring Query
            SELECT 
                DATE(created_at) as collection_date,
                COUNT(*) as races_collected,
                COUNT(DISTINCT venue) as unique_venues,
                MIN(race_date) as earliest_race_date,
                MAX(race_date) as latest_race_date,
                (julianday(MAX(race_date)) - julianday(MIN(race_date))) as date_span_days,
                CASE 
                    WHEN COUNT(*) > 50 THEN 'HIGH_VOLUME'
                    WHEN COUNT(*) < 5 THEN 'LOW_VOLUME'
                    WHEN date_span_days > 7 THEN 'WIDE_DATE_RANGE'
                    ELSE 'NORMAL'
                END as pattern_type
            FROM races
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY DATE(created_at)
            HAVING pattern_type != 'NORMAL'
            ORDER BY collection_date DESC;
            """,
        }

        for query_name, query_sql in queries.items():
            print(f"\n--- {query_name.replace('_', ' ').title()} ---")
            print(query_sql.strip())

        # Save queries to file for future use
        with open("./temporal_monitoring_queries.sql", "w") as f:
            f.write("-- Temporal Coverage Monitoring Queries\n")
            f.write("-- Generated by Temporal Anomaly Investigation\n\n")
            for query_name, query_sql in queries.items():
                f.write(f"-- {query_name.replace('_', ' ').title()}\n")
                f.write(query_sql.strip())
                f.write("\n\n" + "=" * 50 + "\n\n")

        print(f"\n💾 Monitoring queries saved to 'temporal_monitoring_queries.sql'")

        return queries

    def run_full_investigation(self):
        """Run complete anomaly investigation"""
        print("🚨 TEMPORAL ANOMALY DEEP DIVE INVESTIGATION")
        print("=" * 60)

        # Investigate specific issues
        gap_data, invalid_dates, venue_patterns = self.investigate_gap_period()
        retroactive_context = self.investigate_retroactive_data()
        collection_patterns, batch_patterns = self.analyze_data_collection_patterns()
        venue_evolution, unusual_venues, quality_data = self.identify_schema_changes()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        # Create monitoring queries
        monitoring_queries = self.create_monitoring_queries()

        return {
            "gap_analysis": {
                "gap_data": gap_data,
                "invalid_dates": invalid_dates,
                "venue_patterns": venue_patterns,
            },
            "retroactive_analysis": retroactive_context,
            "collection_analysis": {
                "patterns": collection_patterns,
                "batches": batch_patterns,
            },
            "schema_analysis": {
                "venue_evolution": venue_evolution,
                "unusual_venues": unusual_venues,
                "quality_data": quality_data,
            },
            "recommendations": recommendations,
            "monitoring_queries": monitoring_queries,
        }


def main():
    """Main investigation execution"""
    database_path = "./databases/race_data.db"

    try:
        with AnomalyInvestigator(database_path) as investigator:
            results = investigator.run_full_investigation()
            print(f"\n" + "=" * 60)
            print("🎯 INVESTIGATION COMPLETE")
            print("=" * 60)
            print("Key outputs generated:")
            print("  ✓ Detailed anomaly analysis")
            print("  ✓ Actionable recommendations")
            print("  ✓ Monitoring SQL queries (temporal_monitoring_queries.sql)")
            print("  ✓ Implementation roadmap")

    except Exception as e:
        print(f"Error during investigation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
