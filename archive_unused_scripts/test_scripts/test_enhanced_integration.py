#!/usr/bin/env python3
"""
Test Enhanced Race Processor Integration
=======================================

This script tests the integration of the enhanced race processor with the upcoming race predictor
to verify that dead heat handling and enhanced processing are working correctly.

Author: AI Assistant
Date: July 24, 2025
"""

import os
import sqlite3
from pathlib import Path

from upcoming_race_predictor import UpcomingRacePredictor


def test_enhanced_integration():
    """Test the enhanced race processor integration"""
    print("🧪 Testing Enhanced Race Processor Integration")
    print("=" * 50)

    # Initialize predictor with enhanced processing
    predictor = UpcomingRacePredictor()

    # Check availability of enhanced processor
    print(f"Enhanced Processor Available: {predictor.use_enhanced_processor}")
    if predictor.race_processor:
        print("✅ Enhanced race processor initialized successfully")
    else:
        print("❌ Enhanced race processor not available")
        return

    # Test with a sample race file
    test_files = [
        "form_guides/navigator_race_results.csv",
        "upcoming_races/richmond_r3_24_07_2025.csv",
        "form_guides/geelong_r5_22_07_2025.csv",
    ]

    for test_file in test_files:
        file_path = Path(test_file)
        if file_path.exists():
            print(f"\n📂 Testing with: {test_file}")
            try:
                # Test enhanced processing
                result = predictor.race_processor.process_race_results(file_path)

                # Check if processing was successful (look for actual race storage)
                if result.get("races_processed", 0) > 0:
                    print(
                        f"   ✅ Enhanced processing: {result.get('races_processed', 0)} races processed"
                    )

                    # Check for dead heat detection
                    if result.get("dead_heats"):
                        print(f"   🏁 Dead heats detected: {result['dead_heats']}")

                    # Check data quality
                    if result.get("data_quality"):
                        quality = result["data_quality"]
                        print(
                            f"   📊 Data Quality - Warnings: {len(quality.get('warnings', []))}, Errors: {len(quality.get('errors', []))}"
                        )

                        if quality.get("warnings"):
                            for warning in quality["warnings"]:
                                print(f"     ⚠️  {warning}")

                        if quality.get("errors"):
                            for error in quality["errors"]:
                                print(f"     ❌ {error}")
                elif result.get("success", False):
                    print(
                        f"   ✅ Enhanced processing successful: {result.get('summary', 'Success')}"
                    )
                else:
                    print(
                        f"   ❌ Enhanced processing failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"   ❌ Error processing {test_file}: {e}")
        else:
            print(f"   ⚠️  File not found: {test_file}")

    print("\n" + "=" * 50)

    # Test database integration
    print("🗄️  Testing Database Integration")

    # Check recent race entries
    conn = sqlite3.connect("greyhound_racing_data.db")
    cursor = conn.cursor()

    # Get recent entries from enhanced processor
    cursor.execute(
        """
        SELECT race_id, venue, race_number, data_source, race_status, winner_name
        FROM race_metadata 
        WHERE data_source = 'enhanced_race_processor'
        ORDER BY extraction_timestamp DESC
        LIMIT 5
    """
    )

    enhanced_races = cursor.fetchall()
    if enhanced_races:
        print("✅ Recent races processed by enhanced processor:")
        for race in enhanced_races:
            print(
                f"   📍 {race[0]} ({race[1]} R{race[2]}) - Status: {race[4]} - Winner: {race[5] or 'N/A'}"
            )
    else:
        print("⚠️  No races found from enhanced processor")

    # Check for dead heat entries
    cursor.execute(
        """
        SELECT race_id, dog_clean_name, finish_position, data_quality_note
        FROM dog_race_data 
        WHERE finish_position LIKE '%=%' OR data_quality_note LIKE '%dead heat%'
        ORDER BY extraction_timestamp DESC
        LIMIT 10
    """
    )

    dead_heat_entries = cursor.fetchall()
    if dead_heat_entries:
        print("\n🏁 Dead heat entries found:")
        for entry in dead_heat_entries:
            print(
                f"   📍 {entry[0]} - {entry[1]} - Position: {entry[2]} - Note: {entry[3] or 'N/A'}"
            )
    else:
        print("\n⚠️  No dead heat entries found in database")

    conn.close()


def test_legacy_fallback():
    """Test the legacy fallback mechanism"""
    print("\n🔄 Testing Legacy Fallback Mechanism")
    print("=" * 50)

    # Create a predictor without enhanced processor
    predictor = UpcomingRacePredictor()

    # Temporarily disable enhanced processor for testing
    predictor.use_enhanced_processor = False
    predictor.race_processor = None

    print("✅ Enhanced processor disabled for fallback test")

    # Test file (if exists)
    test_file = Path("upcoming_races/richmond_r3_24_07_2025.csv")
    if test_file.exists():
        print(f"📂 Testing legacy fallback with: {test_file}")
        try:
            result = predictor.update_database_with_results(test_file)
            if result:
                print("   ✅ Legacy fallback processing successful")
            else:
                print("   ❌ Legacy fallback processing failed")
        except Exception as e:
            print(f"   ❌ Legacy fallback error: {e}")
    else:
        print("   ⚠️  Test file not found for legacy fallback")


if __name__ == "__main__":
    test_enhanced_integration()
    test_legacy_fallback()

    print("\n🎯 Integration testing complete!")
    print("\nNext steps:")
    print("1. Process race result files with enhanced processor")
    print("2. Verify dead heat handling in database")
    print("3. Check data quality improvements")
