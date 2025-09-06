#!/usr/bin/env python3
"""
Fix CSV Database Ingestion
==========================

This script fixes the CSV ingestion issue where files were being processed
but not inserted into the database. It uses the proper EnhancedComprehensiveProcessor
to actually insert data into the SQLite database.

Based on the analysis, the previous process was only moving files around
without database insertion. This script rectifies that.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor

    print("✅ Enhanced Comprehensive Processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import processor: {e}")
    sys.exit(1)


def main():
    """Main ingestion process"""
    print("🔧 Starting CSV Database Ingestion Fix")
    print("=" * 50)

    # Initialize the processor with minimal mode for faster processing
    print("🚀 Initializing Enhanced Comprehensive Processor...")
    processor = EnhancedComprehensiveProcessor(
        db_path="greyhound_data.db",
        processing_mode="minimal",  # Skip web scraping for speed
        batch_size=25,
    )

    # Define source directories
    form_guides_dir = Path("form_guides/downloaded")
    processed_excluded_dir = Path("processed/excluded")
    processed_other_dir = Path("processed/other")

    # Collect all CSV files that need proper database ingestion
    csv_files = []

    # Check form_guides/downloaded directory
    if form_guides_dir.exists():
        form_guide_files = list(form_guides_dir.glob("*.csv"))
        csv_files.extend(form_guide_files)
        print(f"📁 Found {len(form_guide_files)} files in form_guides/downloaded")

    # Check processed directories (these were processed but not inserted into DB)
    if processed_excluded_dir.exists():
        excluded_files = list(processed_excluded_dir.glob("*.csv"))
        csv_files.extend(excluded_files)
        print(f"📁 Found {len(excluded_files)} files in processed/excluded")

    if processed_other_dir.exists():
        other_files = list(processed_other_dir.glob("*.csv"))
        csv_files.extend(other_files)
        print(f"📁 Found {len(other_files)} files in processed/other")

    total_files = len(csv_files)
    print(f"\n📊 Total CSV files to process: {total_files}")

    if total_files == 0:
        print("❌ No CSV files found to process")
        return

    # Process files in batches
    successful_count = 0
    failed_count = 0
    batch_size = processor.batch_size

    print(f"\n🔄 Processing files in batches of {batch_size}...")

    for i in range(0, len(csv_files), batch_size):
        batch = csv_files[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(csv_files) + batch_size - 1) // batch_size

        print(f"\n📦 Processing Batch {batch_num}/{total_batches} ({len(batch)} files)")
        print("-" * 40)

        for j, csv_file in enumerate(batch, 1):
            try:
                print(f"[{i+j}/{total_files}] Processing: {csv_file.name}")

                # Process the CSV file using the enhanced processor
                result = processor.process_csv_file(str(csv_file))

                if result and result.get("status") != "error":
                    successful_count += 1
                    print(f"  ✅ Successfully processed and inserted into database")
                else:
                    failed_count += 1
                    error_msg = (
                        result.get("error", "Unknown error")
                        if result
                        else "No result returned"
                    )
                    print(f"  ❌ Failed: {error_msg}")

            except Exception as e:
                failed_count += 1
                print(f"  ❌ Error processing {csv_file.name}: {e}")

            # Brief pause to prevent overwhelming the system
            time.sleep(0.1)

        # Progress update
        progress = ((i + len(batch)) / total_files) * 100
        print(
            f"\n📈 Progress: {progress:.1f}% ({successful_count} successful, {failed_count} failed)"
        )

        # Brief pause between batches
        if batch_num < total_batches:
            print("⏸️  Brief pause before next batch...")
            time.sleep(1)

    # Final summary
    print("\n" + "=" * 50)
    print("🎯 CSV Database Ingestion Complete!")
    print(f"✅ Successfully processed: {successful_count}")
    print(f"❌ Failed to process: {failed_count}")
    print(f"📊 Total files: {total_files}")
    print(f"🔄 Success rate: {(successful_count/total_files)*100:.1f}%")

    # Check database state
    try:
        import sqlite3

        conn = sqlite3.connect("greyhound_data.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        race_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_count = cursor.fetchone()[0]

        conn.close()

        print(f"\n💾 Database Status:")
        print(f"   🏁 Races in database: {race_count:,}")
        print(f"   🐕 Dog records in database: {dog_count:,}")

        if race_count > 0 or dog_count > 0:
            print("✅ Database ingestion was successful!")
        else:
            print("⚠️  Database still appears empty - check for processing errors")

    except Exception as e:
        print(f"⚠️  Could not check database status: {e}")

    # Cleanup
    try:
        processor.cleanup()
        print("🧹 Processor cleanup complete")
    except:
        pass

    print("\n🏁 Script completed!")


if __name__ == "__main__":
    main()
