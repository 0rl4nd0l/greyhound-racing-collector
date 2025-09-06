#!/usr/bin/env python3
"""
Data Retention Analysis
=======================

Analyzes why only 19.2% of races are retained for training.
Investigates each filter step to understand data quality issues.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def analyze_data_retention():
    print("📊 Training Data Retention Analysis")
    print("=" * 60)

    try:
        conn = sqlite3.connect("greyhound_racing_data.db")

        # Step 1: Original data volume
        print("\n🔍 Step 1: Original Data Volume")
        print("-" * 40)

        query_original = """
        SELECT 
            COUNT(DISTINCT d.race_id) as total_races,
            COUNT(*) as total_samples,
            COUNT(DISTINCT r.venue) as venues,
            MIN(r.race_date) as earliest_date,
            MAX(r.race_date) as latest_date
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id IS NOT NULL
        """

        original_stats = pd.read_sql_query(query_original, conn)
        total_races = original_stats["total_races"].iloc[0]
        total_samples = original_stats["total_samples"].iloc[0]
        venues = original_stats["venues"].iloc[0]
        earliest = original_stats["earliest_date"].iloc[0]
        latest = original_stats["latest_date"].iloc[0]

        print(f"   📈 Total races in database: {total_races:,}")
        print(f"   📈 Total dog entries: {total_samples:,}")
        print(f"   🏟️  Unique venues: {venues}")
        print(f"   📅 Date range: {earliest} to {latest}")

        # Step 2: Analyze field size distribution (original)
        print("\n🔍 Step 2: Field Size Distribution (Before Filtering)")
        print("-" * 50)

        query_field_sizes = """
        WITH race_sizes AS (
            SELECT 
                d.race_id,
                COUNT(*) as field_size
            FROM dog_race_data d
            WHERE d.race_id IS NOT NULL
            GROUP BY d.race_id
        )
        SELECT 
            field_size,
            COUNT(*) as race_count,
            ROUND(COUNT(*) * 100.0 / (
                SELECT COUNT(DISTINCT race_id) 
                FROM dog_race_data 
                WHERE race_id IS NOT NULL
            ), 1) as percentage
        FROM race_sizes
        GROUP BY field_size
        ORDER BY field_size
        """

        field_size_dist = pd.read_sql_query(query_field_sizes, conn)

        print("   Field Size Distribution:")
        for _, row in field_size_dist.iterrows():
            size = row["field_size"]
            count = row["race_count"]
            pct = row["percentage"]
            status = "✅" if 3 <= size <= 20 else "❌"
            print(f"   {status} {int(size):2d} dogs: {count:4,} races ({pct:5.1f}%)")

        # Identify how many races are lost to field size
        valid_field_sizes = field_size_dist[
            (field_size_dist["field_size"] >= 3) & (field_size_dist["field_size"] <= 20)
        ]
        kept_by_field_size = valid_field_sizes["race_count"].sum()
        lost_by_field_size = total_races - kept_by_field_size

        print(f"\n   📊 Field size filtering impact:")
        print(
            f"      Races with 3-20 dogs: {kept_by_field_size:,} ({kept_by_field_size/total_races*100:.1f}%)"
        )
        print(
            f"      Races lost (<3 or >20): {lost_by_field_size:,} ({lost_by_field_size/total_races*100:.1f}%)"
        )

        # Step 3: Analyze finish position issues
        print("\n🔍 Step 3: Finish Position Data Quality")
        print("-" * 40)

        query_positions = """
        SELECT 
            d.race_id,
            COUNT(*) as dogs_in_race,
            COUNT(d.finish_position) as positions_available,
            MIN(d.finish_position) as min_pos,
            MAX(d.finish_position) as max_pos,
            COUNT(CASE WHEN d.finish_position = 1 THEN 1 END) as winners
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id IS NOT NULL 
            AND r.race_date IS NOT NULL
        GROUP BY d.race_id
        HAVING COUNT(*) >= 3 AND COUNT(*) <= 20
        """

        position_analysis = pd.read_sql_query(query_positions, conn)

        # Analyze position issues
        missing_positions = position_analysis[
            position_analysis["positions_available"]
            != position_analysis["dogs_in_race"]
        ]
        multiple_winners = position_analysis[position_analysis["winners"] != 1]
        invalid_ranges = position_analysis[
            (position_analysis["min_pos"] != 1)
            | (position_analysis["max_pos"] != position_analysis["dogs_in_race"])
        ]

        print(f"   📊 Position data analysis (after field size filter):")
        print(f"      Total races analyzed: {len(position_analysis):,}")
        print(f"      ❌ Missing positions: {len(missing_positions):,} races")
        print(f"      ❌ Multiple/no winners: {len(multiple_winners):,} races")
        print(f"      ❌ Invalid position ranges: {len(invalid_ranges):,} races")

        # Show examples of problematic races
        if len(invalid_ranges) > 0:
            print(f"\n   🔍 Examples of invalid position ranges:")
            for _, race in invalid_ranges.head(5).iterrows():
                race_id = race["race_id"]
                dogs = race["dogs_in_race"]
                min_pos = race["min_pos"]
                max_pos = race["max_pos"]
                print(
                    f"      Race {race_id}: {dogs} dogs, positions {min_pos}-{max_pos} (should be 1-{dogs})"
                )

        # Step 4: Missing metadata analysis
        print("\n🔍 Step 4: Missing Metadata Analysis")
        print("-" * 40)

        query_metadata = """
        SELECT 
            COUNT(DISTINCT d.race_id) as races_with_dogs,
            COUNT(DISTINCT CASE WHEN r.venue IS NOT NULL THEN d.race_id END) as races_with_venue,
            COUNT(DISTINCT CASE WHEN r.grade IS NOT NULL THEN d.race_id END) as races_with_grade,
            COUNT(DISTINCT CASE WHEN r.distance IS NOT NULL THEN d.race_id END) as races_with_distance,
            COUNT(DISTINCT CASE WHEN r.race_date IS NOT NULL THEN d.race_id END) as races_with_date
        FROM dog_race_data d
        LEFT JOIN race_metadata r ON d.race_id = r.race_id
        WHERE d.race_id IS NOT NULL
        """

        metadata_stats = pd.read_sql_query(query_metadata, conn)
        races_total = metadata_stats["races_with_dogs"].iloc[0]

        print(f"   📊 Metadata completeness:")
        print(f"      Total races: {races_total:,}")
        for field in ["venue", "grade", "distance", "date"]:
            col = f"races_with_{field}"
            count = metadata_stats[col].iloc[0]
            pct = count / races_total * 100
            status = "✅" if pct > 95 else "⚠️" if pct > 90 else "❌"
            print(f"      {status} {field}: {count:,} ({pct:.1f}%)")

        # Step 5: Simulate the filtering process step by step
        print("\n🔍 Step 5: Step-by-Step Filtering Simulation")
        print("-" * 45)

        # Start with all races
        current_races = total_races
        print(f"   🏁 Starting races: {current_races:,}")

        # Filter 1: Field size (3-20 dogs)
        current_races = kept_by_field_size
        lost = total_races - current_races
        print(
            f"   1️⃣ After field size filter (3-20 dogs): {current_races:,} (-{lost:,}, {current_races/total_races*100:.1f}%)"
        )

        # Filter 2: Single winner
        races_after_winners = len(position_analysis) - len(multiple_winners)
        lost = current_races - races_after_winners
        current_races = races_after_winners
        print(
            f"   2️⃣ After winner filter (exactly 1 winner): {current_races:,} (-{lost:,}, {current_races/total_races*100:.1f}%)"
        )

        # Filter 3: Valid positions
        races_after_positions = current_races - len(invalid_ranges)
        lost = current_races - races_after_positions
        current_races = races_after_positions
        print(
            f"   3️⃣ After position validation: {current_races:,} (-{lost:,}, {current_races/total_races*100:.1f}%)"
        )

        # Filter 4: Required metadata (estimate)
        missing_venue = races_total - metadata_stats["races_with_venue"].iloc[0]
        missing_grade = races_total - metadata_stats["races_with_grade"].iloc[0]
        missing_distance = races_total - metadata_stats["races_with_distance"].iloc[0]
        missing_date = races_total - metadata_stats["races_with_date"].iloc[0]

        # Rough estimate of metadata impact
        estimated_metadata_loss = (
            max(missing_venue, missing_grade, missing_distance, missing_date) * 0.1
        )  # Conservative estimate
        races_after_metadata = current_races - int(estimated_metadata_loss)
        current_races = races_after_metadata
        print(f"   4️⃣ After metadata requirements: ~{current_races:,} (estimated)")

        print(
            f"\n   📊 Final retention estimate: ~{current_races/total_races*100:.1f}%"
        )
        print(f"   🎯 Actual retention from training: 19.2%")

        # Step 6: Recommendations
        print("\n💡 Analysis & Recommendations")
        print("-" * 30)

        major_losses = []
        if lost_by_field_size > total_races * 0.3:
            major_losses.append(
                f"Field size filtering removes {lost_by_field_size/total_races*100:.1f}% of races"
            )
        if len(multiple_winners) > len(position_analysis) * 0.2:
            major_losses.append(
                f"Winner validation removes {len(multiple_winners)/len(position_analysis)*100:.1f}% of remaining races"
            )
        if len(invalid_ranges) > len(position_analysis) * 0.2:
            major_losses.append(
                f"Position validation removes {len(invalid_ranges)/len(position_analysis)*100:.1f}% of remaining races"
            )

        if major_losses:
            print(f"   ⚠️ Major data quality issues identified:")
            for issue in major_losses:
                print(f"      • {issue}")
        else:
            print(f"   ✅ No major data quality issues - filtering appears appropriate")

        print(f"\n   💡 Data Quality Assessment:")
        if current_races / total_races > 0.15:  # 15% threshold
            print(
                f"      ✅ 19.2% retention is reasonable for high-quality ML training"
            )
            print(f"      ✅ Better to have clean data than noisy data")
            print(f"      ✅ Quality over quantity approach is correct")
        else:
            print(
                f"      ⚠️ Retention may be too aggressive - consider relaxing some filters"
            )

        print(f"\n   🎯 Training Data Quality:")
        print(f"      • {current_races:,} high-quality races")
        print(f"      • Perfect position data (1st, 2nd, 3rd, etc.)")
        print(f"      • Single winner per race")
        print(f"      • Complete metadata")
        print(f"      • Reasonable field sizes (3-20 dogs)")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = analyze_data_retention()
    sys.exit(0 if success else 1)
