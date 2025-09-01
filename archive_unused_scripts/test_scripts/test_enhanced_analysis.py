#!/usr/bin/env python3
"""
Test script to debug enhanced analysis issues
"""

import sys
import traceback

from enhanced_race_analyzer import EnhancedRaceAnalyzer

DATABASE_PATH = "greyhound_racing_data.db"


def test_enhanced_analysis():
    try:
        print("[DEBUG] Starting enhanced analysis test")
        analyzer = EnhancedRaceAnalyzer(DATABASE_PATH)

        print("[DEBUG] Step 1: Loading data")
        analyzer.load_data()
        print(f"[DEBUG] Data loaded: {len(analyzer.data)} rows")

        print("[DEBUG] Step 2: Engineering features")
        analyzer.engineer_features()
        print("[DEBUG] Features engineered")

        print("[DEBUG] Step 3: Normalizing performance")
        analyzer.normalize_performance()
        print("[DEBUG] Performance normalized")

        print("[DEBUG] Step 4: Adding race condition features")
        analyzer.add_race_condition_features()
        print("[DEBUG] Race condition features added")

        print("[DEBUG] Step 5: Identifying top performers")
        top_performers = analyzer.identify_top_performers(min_races=2)
        print(f"[DEBUG] Top performers identified: {len(top_performers)} dogs")

        print("[DEBUG] Step 6: Temporal analysis")
        monthly_stats, venue_stats = analyzer.temporal_analysis()
        print(
            f"[DEBUG] Temporal analysis complete: {len(monthly_stats)} months, {len(venue_stats)} venues"
        )

        print("[DEBUG] Step 7: Race condition analysis")
        race_condition_analysis = analyzer.analyze_race_conditions()
        print("[DEBUG] Race condition analysis complete")

        print("[DEBUG] Step 8: Generating insights")
        insights = analyzer.generate_insights()
        print("[DEBUG] Insights generated")

        print("[DEBUG] ✅ All steps completed successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        print("[ERROR] Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_analysis()
    sys.exit(0 if success else 1)
