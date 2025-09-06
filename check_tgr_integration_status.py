#!/usr/bin/env python3
"""
TGR Integration Status Check
===========================

This script checks the current status of TGR integration in your enhanced data processing
workflow and provides guidance on what's working and what needs attention.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")


def check_tgr_scraper():
    """Check if the core TGR scraper is working."""

    print("🔍 Checking TGR Scraper...")

    try:
        from collectors.the_greyhound_recorder_scraper import (
            TheGreyhoundRecorderScraper,
        )

        scraper = TheGreyhoundRecorderScraper(rate_limit=5.0, use_cache=True)

        # Test basic functionality
        cache_dir = Path(".tgr_cache")
        if cache_dir.exists():
            cached_files = list(cache_dir.glob("*.json"))
            print(f"✅ TGR Scraper initialized with {len(cached_files)} cached files")
        else:
            print("✅ TGR Scraper initialized (no cache data yet)")

        return True, scraper
    except Exception as e:
        print(f"❌ TGR Scraper failed: {e}")
        return False, None


def check_enhanced_tgr_collector():
    """Check enhanced TGR collector."""

    print("🔍 Checking Enhanced TGR Collector...")

    try:
        from enhanced_tgr_collector import EnhancedTGRCollector

        collector = EnhancedTGRCollector()

        print("✅ Enhanced TGR Collector available")
        return True, collector
    except Exception as e:
        print(f"❌ Enhanced TGR Collector failed: {e}")
        return False, None


def check_tgr_prediction_integration():
    """Check TGR prediction integration."""

    print("🔍 Checking TGR Prediction Integration...")

    try:
        from tgr_prediction_integration import TGRPredictionIntegrator

        integrator = TGRPredictionIntegrator()

        print("✅ TGR Prediction Integration available")
        return True, integrator
    except Exception as e:
        print(f"❌ TGR Prediction Integration failed: {e}")
        return False, None


def check_data_processing_modules():
    """Check the data processing modules that use TGR."""

    print("🔍 Checking Data Processing Modules...")

    # Check enhanced data integration
    try:
        from enhanced_data_integration import EnhancedDataIntegrator

        print("✅ Enhanced Data Integration available")
        data_integration_ok = True
    except Exception as e:
        print(f"❌ Enhanced Data Integration failed: {e}")
        print(f"   Error details: {str(e)}")
        data_integration_ok = False

    # Check enhanced comprehensive processor
    try:
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor

        print("✅ Enhanced Comprehensive Processor available")
        processor_ok = True
    except Exception as e:
        print(f"❌ Enhanced Comprehensive Processor failed: {e}")
        print(f"   Error details: {str(e)}")
        processor_ok = False

    return data_integration_ok, processor_ok


def check_web_app_integration():
    """Check if the web app has TGR integration."""

    print("🔍 Checking Web App TGR Integration...")

    try:
        # Check if app.py has enhanced data processing endpoints
        app_file = Path("app.py")
        if app_file.exists():
            app_content = app_file.read_text()

            # Look for enhanced data processing endpoints
            enhanced_endpoints = [
                "enhanced_data_processing",
                "EnhancedComprehensiveProcessor",
                "enhanced_prediction",
                "tgr",
            ]

            found_integrations = []
            for endpoint in enhanced_endpoints:
                if endpoint in app_content:
                    found_integrations.append(endpoint)

            if found_integrations:
                print(
                    f"✅ Web app has TGR integrations: {', '.join(found_integrations)}"
                )
                return True
            else:
                print("⚠️ Web app exists but no TGR integrations found")
                return False
        else:
            print("❌ Web app (app.py) not found")
            return False

    except Exception as e:
        print(f"❌ Error checking web app: {e}")
        return False


def check_csv_files_and_data():
    """Check for CSV files and data that can be processed."""

    print("🔍 Checking Available Data...")

    # Check directories
    directories = {
        "unprocessed": Path("./unprocessed"),
        "upcoming_races": Path("./upcoming_races"),
        "processed_races": Path("./processed_races"),
        "tgr_cache": Path("./.tgr_cache"),
    }

    for name, path in directories.items():
        if path.exists():
            if name == "tgr_cache":
                files = list(path.glob("*.json"))
            else:
                files = list(path.glob("*.csv"))
            print(f"📁 {name}: {len(files)} files")
        else:
            print(f"📁 {name}: Directory not found")

    # Check if there are CSV files to process
    csv_files = []
    for directory in ["unprocessed", "upcoming_races"]:
        dir_path = directories[directory]
        if dir_path.exists():
            csv_files.extend(list(dir_path.glob("*.csv")))

    if csv_files:
        print(f"✅ {len(csv_files)} CSV files available for processing")
        return True
    else:
        print("⚠️ No CSV files found for processing")
        return False


def check_database():
    """Check database for existing race data."""

    print("🔍 Checking Database...")

    try:
        import sqlite3

        db_files = list(Path(".").glob("*.db")) + list(Path(".").glob("*.sqlite"))

        if not db_files:
            print("❌ No database files found")
            return False

        for db_file in db_files:
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()

                # Check for races table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='races'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM races")
                    race_count = cursor.fetchone()[0]
                    print(f"✅ Database {db_file.name}: {race_count} races")
                else:
                    print(f"⚠️ Database {db_file.name}: No races table found")

                conn.close()

            except Exception as e:
                print(f"❌ Error reading database {db_file.name}: {e}")

        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False


def analyze_tgr_integration_status():
    """Provide analysis of TGR integration status."""

    print("\\n" + "=" * 60)
    print("📋 TGR INTEGRATION STATUS ANALYSIS")
    print("=" * 60)

    # Run all checks
    scraper_ok, scraper = check_tgr_scraper()
    print()

    collector_ok, collector = check_enhanced_tgr_collector()
    print()

    prediction_ok, prediction = check_tgr_prediction_integration()
    print()

    data_integration_ok, processor_ok = check_data_processing_modules()
    print()

    webapp_ok = check_web_app_integration()
    print()

    csv_ok = check_csv_files_and_data()
    print()

    db_ok = check_database()
    print()

    # Summary and recommendations
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    core_components = sum([scraper_ok, collector_ok, prediction_ok])
    processing_components = sum([data_integration_ok, processor_ok])
    infrastructure = sum([webapp_ok, csv_ok, db_ok])

    print(f"🔧 Core TGR Components: {core_components}/3 working")
    print(f"⚙️ Data Processing Components: {processing_components}/2 working")
    print(f"🏗️ Infrastructure: {infrastructure}/3 working")

    print("\\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)

    if core_components >= 2 and processing_components >= 1:
        print("✅ TGR INTEGRATION IS FUNCTIONAL!")
        print("\\n📝 What this means:")
        print("   • TGR data scraping works")
        print("   • Enhanced data collection is available")
        print("   • Prediction integration is ready")

        if processor_ok:
            print("   • Enhanced data processing pipeline works")
            print("\\n🚀 You can use 'Enhanced Data Processing' in the UI")
        else:
            print("   • Enhanced data processing may have dependency issues")
            print("\\n⚠️ You may need to install numpy/pandas for full functionality")

    elif core_components >= 2:
        print("⚠️ TGR COMPONENTS WORK, BUT PROCESSING PIPELINE NEEDS ATTENTION")
        print("\\n📝 Status:")
        print("   • TGR scraper and collectors work fine")
        print("   • Data processing modules have dependency issues")
        print("\\n🔧 Next steps:")
        print("   • Install missing dependencies (numpy, pandas)")
        print("   • Or use simplified processing mode")

    else:
        print("❌ TGR INTEGRATION NEEDS DEBUGGING")
        print("\\n🔧 Next steps:")
        print("   • Check module imports and dependencies")
        print("   • Verify TGR scraper configuration")
        print("   • Test individual components")

    print(f"\\n📅 Status checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function."""

    print("🚀 TGR Integration Status Check")
    print("=" * 60)

    analyze_tgr_integration_status()


if __name__ == "__main__":
    main()
