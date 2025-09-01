#!/usr/bin/env python3
"""
Enhanced Form Guide Usage Examples
=================================

This script provides practical examples of how to use the enhanced expert form 
data system to update and enrich your existing greyhound racing data.

Usage Examples:

1. Update recent races (last 30 days)
2. Update specific date range
3. Update specific race URLs
4. Run comprehensive historical update
5. Check database coverage and status

Author: AI Assistant
Date: July 25, 2025
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

from enhanced_data_processor import EnhancedDataProcessor
from enhanced_expert_form_scraper import EnhancedExpertFormScraper

# Import the enhanced components
from historical_race_data_updater import HistoricalRaceDataUpdater
from integrated_enhanced_form_system import IntegratedEnhancedFormSystem


def show_database_status():
    """Show current database status and coverage"""
    print("🔍 CHECKING DATABASE STATUS")
    print("=" * 60)

    updater = HistoricalRaceDataUpdater()
    db_stats = updater.get_current_database_stats()

    if "coverage" in db_stats:
        coverage = db_stats["coverage"]
        print(f"📊 Database Coverage:")
        print(f"   • Total races: {coverage['total_races']:,}")
        print(f"   • Enhanced races: {coverage['enhanced_races']:,}")
        print(f"   • Coverage percentage: {coverage['percentage']:.1f}%")
        print(
            f"   • Races needing update: {coverage['total_races'] - coverage['enhanced_races']:,}"
        )

    for table, info in db_stats.items():
        if isinstance(info, dict) and "description" in info:
            print(f"   • {info['description']}: {info['count']:,}")

    return db_stats


def update_recent_races(days_back: int = 30, max_races: int = 50):
    """Update recent races with enhanced data"""
    print(f"\n🚀 UPDATING RECENT RACES")
    print("=" * 60)
    print(f"📅 Looking back: {days_back} days")
    print(f"🎯 Maximum races: {max_races}")

    updater = HistoricalRaceDataUpdater()
    results = updater.update_recent_races(days_back=days_back, max_races=max_races)

    if results.get("success_rate", 0) > 0:
        print(
            f"✅ Update completed with {results.get('success_rate', 0):.1f}% success rate"
        )
        print(
            f"📊 Results: {results.get('successful_updates', 0)} successful, {results.get('failed_updates', 0)} failed"
        )
        print(f"💾 Records created: {results.get('total_records_created', 0)}")
    else:
        print(f"⚠️ No races found or updated")

    return results


def update_specific_date_range(start_date: str, end_date: str, max_races: int = 100):
    """Update races within a specific date range"""
    print(f"\n🚀 UPDATING RACES BY DATE RANGE")
    print("=" * 60)
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"🎯 Maximum races: {max_races}")

    updater = HistoricalRaceDataUpdater()
    results = updater.update_races_by_date_range(start_date, end_date, max_races)

    if results.get("success"):
        print(f"✅ Date range update completed")
        print(
            f"📊 Results: {results.get('successful_updates', 0)} successful, {results.get('failed_updates', 0)} failed"
        )
        print(f"💾 Records created: {results.get('total_records_created', 0)}")
    else:
        print(f"⚠️ {results.get('message', 'Update failed')}")

    return results


def update_specific_race_urls(race_urls: List[str]):
    """Update specific race URLs with enhanced data"""
    print(f"\n🚀 UPDATING SPECIFIC RACE URLS")
    print("=" * 60)
    print(f"📊 URLs to process: {len(race_urls)}")

    system = IntegratedEnhancedFormSystem()
    results = system.process_race_urls_comprehensively(
        race_urls, use_both_methods=False
    )

    # Process the extracted data
    if results["enhanced_extractions"] > 0:
        data_results = system.process_extracted_data()
        print(f"✅ Data processing completed")
        print(f"💾 Records processed: {data_results.get('processed', 0)}")

    print(f"📊 Results:")
    print(f"   • Total races: {results['total_races']}")
    print(
        f"   • Successful: {results['successful_races']} ({results['success_rate']:.1f}%)"
    )
    print(f"   • Enhanced extractions: {results['enhanced_extractions']}")

    return results


def run_comprehensive_update(max_races: int = 100, days_back: int = 90):
    """Run comprehensive historical update"""
    print(f"\n🚀 COMPREHENSIVE HISTORICAL UPDATE")
    print("=" * 60)
    print(f"🎯 Maximum races: {max_races}")
    print(f"📅 Days back: {days_back}")

    updater = HistoricalRaceDataUpdater()
    results = updater.run_comprehensive_historical_update(
        max_races=max_races, days_back=days_back
    )

    if results.get("success"):
        print(f"✅ Comprehensive update completed successfully")

        # Show results from each phase
        phases = results.get("phases", {})

        if "identification" in phases:
            id_phase = phases["identification"]
            print(f"🔍 Phase 1 - Identification: {id_phase['urls_found']} races found")

        if "updates" in phases:
            update_phase = phases["updates"]
            print(
                f"🔄 Phase 2 - Updates: {update_phase['successful_updates']}/{update_phase['total_races']} successful"
            )
            print(f"💾 Records created: {update_phase['total_records_created']}")

        if "reporting" in phases:
            print(f"📊 Phase 3 - Reporting: Complete")
    else:
        print(
            f"❌ Comprehensive update failed: {results.get('error', 'Unknown error')}"
        )

    return results


def process_new_race_urls(race_urls: List[str]):
    """Process new race URLs and integrate into system"""
    print(f"\n🚀 PROCESSING NEW RACE URLS")
    print("=" * 60)
    print(f"📋 URLs to process: {len(race_urls)}")

    scraper = EnhancedExpertFormScraper()
    processor = EnhancedDataProcessor()

    successful_extractions = 0

    # Process each URL
    for i, url in enumerate(race_urls, 1):
        print(f"\n--- Processing URL {i}/{len(race_urls)} ---")
        print(f"🌐 {url}")

        success = scraper.process_race_url(url)
        if success:
            successful_extractions += 1
            print(f"✅ Extraction successful")
        else:
            print(f"❌ Extraction failed")

    # Process all extracted data
    if successful_extractions > 0:
        print(f"\n🔄 Processing extracted data...")
        processing_results = processor.process_comprehensive_json_files()

        print(f"✅ Processing complete")
        print(f"📊 Final Results:")
        print(f"   • URLs processed: {len(race_urls)}")
        print(f"   • Successful extractions: {successful_extractions}")
        print(
            f"   • Database records created: {processing_results.get('processed', 0)}"
        )

        return {
            "urls_processed": len(race_urls),
            "successful_extractions": successful_extractions,
            "records_created": processing_results.get("processed", 0),
        }
    else:
        print(f"⚠️ No successful extractions to process")
        return {
            "urls_processed": len(race_urls),
            "successful_extractions": 0,
            "records_created": 0,
        }


def main():
    """Main function with usage examples"""
    print("🏁 ENHANCED FORM GUIDE USAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates how to use the enhanced expert form data system")
    print("to update and enrich your existing greyhound racing data.")

    # Show current database status
    db_status = show_database_status()

    # Example usage scenarios
    print(f"\n🎯 EXAMPLE USAGE SCENARIOS")
    print("=" * 60)

    # Scenario 1: Check if we need updates
    if "coverage" in db_status:
        coverage_pct = db_status["coverage"]["percentage"]
        races_needing_update = (
            db_status["coverage"]["total_races"]
            - db_status["coverage"]["enhanced_races"]
        )

        print(f"📊 Current Status:")
        print(f"   • Coverage: {coverage_pct:.1f}%")
        print(f"   • Races needing update: {races_needing_update:,}")

        if coverage_pct < 50:
            print(f"\n💡 Recommendation: Run comprehensive update to improve coverage")

            # Uncomment to run comprehensive update
            # print(f"\n🚀 Running comprehensive update...")
            # comprehensive_results = run_comprehensive_update(max_races=50, days_back=60)

        elif races_needing_update > 0 and races_needing_update <= 100:
            print(f"\n💡 Recommendation: Update recent races to catch up")

            # Uncomment to update recent races
            # print(f"\n🚀 Updating recent races...")
            # recent_results = update_recent_races(days_back=14, max_races=30)

        else:
            print(f"\n✅ Database coverage is good - no immediate updates needed")

    # Example usage patterns
    print(f"\n📚 USAGE PATTERNS")
    print("=" * 40)
    print(f"1. Update recent races (recommended for regular maintenance):")
    print(f"   update_recent_races(days_back=30, max_races=50)")

    print(f"\n2. Update specific date range:")
    print(f"   update_specific_date_range('2025-07-01', '2025-07-20', max_races=100)")

    print(f"\n3. Process specific race URLs:")
    print(f"   race_urls = ['https://www.thedogs.com.au/racing/...']")
    print(f"   update_specific_race_urls(race_urls)")

    print(f"\n4. Comprehensive historical update:")
    print(f"   run_comprehensive_update(max_races=200, days_back=90)")

    # Show available functions
    print(f"\n🔧 AVAILABLE FUNCTIONS")
    print("=" * 40)
    functions = [
        "show_database_status() - Check current database coverage",
        "update_recent_races() - Update recent races with enhanced data",
        "update_specific_date_range() - Update races in date range",
        "update_specific_race_urls() - Process specific URLs",
        "run_comprehensive_update() - Full historical update",
        "process_new_race_urls() - Process new URLs from scratch",
    ]

    for i, func in enumerate(functions, 1):
        print(f"   {i}. {func}")

    print(f"\n💡 TIPS")
    print("=" * 40)
    tips = [
        "Start with recent races to get immediate value",
        "Use smaller batch sizes (10-20 races) for initial testing",
        "Monitor success rates and adjust if errors occur",
        "Run updates during off-peak hours to be respectful to the website",
        "Check database coverage regularly to track progress",
        "Generate reports after major updates to assess data quality",
    ]

    for i, tip in enumerate(tips, 1):
        print(f"   {i}. {tip}")

    return db_status


if __name__ == "__main__":
    # Example: Uncomment one of these to run

    # Show current status
    main()

    # Example 1: Update recent races (uncomment to run)
    # update_recent_races(days_back=7, max_races=10)

    # Example 2: Update specific date range (uncomment to run)
    # update_specific_date_range('2025-07-20', '2025-07-25', max_races=20)

    # Example 3: Process specific URLs (uncomment to run)
    # sample_urls = [
    #     "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f"
    # ]
    # update_specific_race_urls(sample_urls)

    # Example 4: Comprehensive update (uncomment to run)
    # run_comprehensive_update(max_races=30, days_back=30)
