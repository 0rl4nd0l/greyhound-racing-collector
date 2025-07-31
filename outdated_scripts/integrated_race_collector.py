#!/usr/bin/env python3
"""
Integrated Greyhound Race Data Collector
========================================

This script combines race navigation and comprehensive data collection with:
- Automatic progress tracking and resumption
- Save verification every 5 races
- No artificial limits on race processing
- Integrated race results and enhanced data collection
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from enhanced_odds_collector import EnhancedGreyhoundDataCollector

sys.path.append("..")
from greyhound_results_scraper_navigator import GreyhoundResultsNavigator


class IntegratedRaceCollector:
    def __init__(self):
        self.collector = EnhancedGreyhoundDataCollector()
        self.navigator = GreyhoundResultsNavigator()
        self.progress_file = "./progress_tracker.json"
        self.results_csv = "./form_guides/navigator_race_results.csv"
        self.form_guides_dir = "./form_guides"

    def load_progress(self):
        """Load progress from file to resume where we left off"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "last_processed_index": -1,
            "total_races_processed": 0,
            "last_save_check": 0,
            "last_update": None,
        }

    def save_progress(self, progress):
        """Save current progress to file"""
        progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def get_database_race_count(self):
        """Get current number of races in database"""
        with sqlite3.connect(self.collector.db_path) as conn:
            result = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM race_metadata", conn
            )
            return result.iloc[0]["count"]

    def get_existing_race_ids(self):
        """Get set of existing race IDs in database"""
        with sqlite3.connect(self.collector.db_path) as conn:
            result = pd.read_sql_query("SELECT race_id FROM race_metadata", conn)
            return set(result["race_id"].tolist())

    def race_already_exists(self, race_url):
        """Check if race already exists in database by generating race_id from URL"""
        race_id = self.collector.extract_race_id_from_url(race_url)
        existing_ids = self.get_existing_race_ids()
        return race_id in existing_ids

    def get_all_form_guide_files(self):
        """Get all CSV files from form_guides directory"""
        form_guides_path = Path(self.form_guides_dir)
        csv_files = list(form_guides_path.glob("*.csv"))
        # Filter out result files
        race_files = [f for f in csv_files if "result" not in f.name.lower()]
        return sorted(race_files)

    async def process_form_guide_file(self, file_path, file_index):
        """Process a single form guide CSV file to extract race data"""
        try:
            # Extract race details from filename
            race_number, location, date = self.navigator.extract_race_details(
                file_path.name
            )

            print(f"\n[{file_index + 1}] Processing form guide: {file_path.name}")
            print(f"   Race {race_number} at {location} on {date.strftime('%Y-%m-%d')}")

            # Use navigator to find race results
            results = await self.navigator.fetch_race_results(
                race_number, location, date
            )

            if results:
                race_url = results.get("source_url")
                if race_url:
                    # Check if already exists
                    if self.race_already_exists(race_url):
                        print(f"   ⏭️  Race already exists in database, skipping")
                        return False

                    # Collect comprehensive data
                    race_data = self.collector.collect_comprehensive_race_data(race_url)

                    if race_data and race_data.get("race_metadata"):
                        print(
                            f"   ✅ Successfully processed: {len(race_data['dogs_data'])} dogs"
                        )
                        return True
                    else:
                        print(f"   ❌ Failed to collect comprehensive race data")
                        return False
                else:
                    print(f"   ❌ No source URL found in results")
                    return False
            else:
                print(f"   ❌ No results found for this race")
                return False

        except Exception as e:
            print(f"   ❌ Error processing form guide: {e}")
            return False

    def verify_save_integrity(self, expected_count):
        """Verify that the expected number of races are saved"""
        actual_count = self.get_database_race_count()
        print(f"🔍 Save verification: Expected {expected_count}, Found {actual_count}")

        if actual_count != expected_count:
            print(f"❌ SAVE INTEGRITY FAILURE!")
            print(f"   Expected: {expected_count} races")
            print(f"   Actual: {actual_count} races")
            print(f"   Difference: {expected_count - actual_count}")
            return False

        print(f"✅ Save integrity verified: {actual_count} races saved correctly")
        return True

    async def process_race_from_csv(self, race_row, race_index):
        """Process a single race from CSV data"""
        race_url = race_row.get("source_url")
        race_id = race_row.get("race_id")
        venue = race_row.get("venue")

        print(f"\n[{race_index + 1}] Processing race: {race_id} at {venue}")
        print(f"   URL: {race_url}")

        if not race_url or "thedogs.com.au" not in race_url:
            print(f"   ⚠️  Invalid URL, skipping")
            return False

        # Check if race already exists
        if self.race_already_exists(race_url):
            print(f"   ⏭️  Race already exists in database, skipping")
            return False  # Don't count as success since we didn't add it

        try:
            # Collect comprehensive data
            race_data = self.collector.collect_comprehensive_race_data(race_url)

            if race_data and race_data.get("race_metadata"):
                print(
                    f"   ✅ Successfully processed: {len(race_data['dogs_data'])} dogs"
                )
                return True
            else:
                print(f"   ❌ Failed to collect race data")
                return False

        except Exception as e:
            print(f"   ❌ Error processing race: {e}")
            return False

    async def collect_additional_races(self, start_from_processed_count):
        """Continue collecting races beyond what we have processed"""
        print(
            f"\n🔍 Looking for additional races beyond current {start_from_processed_count}..."
        )

        # Use navigator to find more races from form_guides folder
        try:
            # Get existing results
            existing_df = (
                pd.read_csv(self.results_csv)
                if os.path.exists(self.results_csv)
                else pd.DataFrame()
            )
            existing_count = len(existing_df)

            print(f"Current CSV has {existing_count} races, processing next batch...")

            # Process more races from form_guides
            new_results = await self.navigator.process_form_guides("./form_guides")

            if new_results and len(new_results) > existing_count:
                # Update CSV with new results
                new_df = pd.DataFrame(new_results)

                # Save updated CSV
                new_df.to_csv(self.results_csv, index=False)
                print(
                    f"✅ Updated CSV with {len(new_df)} total races (+{len(new_df) - existing_count} new)"
                )

                return len(new_df)
            else:
                print(f"🔍 No additional races found from form_guides")

        except Exception as e:
            print(f"⚠️  Error collecting additional races: {e}")

        return start_from_processed_count

    async def run_integrated_collection(self):
        """Main collection process with progress tracking and verification"""
        print("🚀 Starting Integrated Race Data Collection")
        print("=" * 60)

        # Load progress
        progress = self.load_progress()
        start_index = progress["last_processed_index"] + 1

        print(f"📋 Resuming from file index: {start_index}")
        print(f"   Last processed: {progress['last_processed_index']}")
        print(f"   Total processed: {progress['total_races_processed']}")

        # Get all form guide files
        all_files = self.get_all_form_guide_files()
        total_files = len(all_files)
        print(f"📊 Found {total_files} form guide files to process")

        # Check current database state
        initial_db_count = self.get_database_race_count()
        current_db_count = initial_db_count
        print(f"🗿️  Current database has {current_db_count} races")

        # Track cumulative new races added
        total_new_races_added = 0

        while start_index < total_files:
            # Process next batch of 5 form guide files
            batch_start = start_index
            batch_end = min(start_index + 5, total_files)

            print(
                f"\n📦 Processing batch: files {batch_start + 1} to {batch_end} of {total_files}"
            )

            success_count = 0
            for i in range(batch_start, batch_end):
                file_path = all_files[i]
                success = await self.process_form_guide_file(file_path, i)

                if success:
                    success_count += 1
                    progress["total_races_processed"] += 1

                progress["last_processed_index"] = i
                self.save_progress(progress)

                # Rate limiting
                time.sleep(2)

            # Update cumulative tracking
            total_new_races_added += success_count

            # Verify save integrity every 5 races
            expected_count = initial_db_count + total_new_races_added

            if success_count > 0 and not self.verify_save_integrity(expected_count):
                print("\n🛑 TERMINATING: Save integrity check failed!")
                print(
                    "   Please investigate the database save issue before continuing."
                )
                print(f"   Progress saved to: {self.progress_file}")
                return False
            elif success_count == 0:
                print(f"🔄 No new races added in this batch (likely duplicates)")

            progress["last_save_check"] = progress["total_races_processed"]
            self.save_progress(progress)

            # Update start index for next batch
            start_index = batch_end

            print(
                f"\n📈 Batch complete! Successfully processed {success_count}/{batch_end - batch_start} files"
            )
            print(f"   Total new races: {progress['total_races_processed']}")
            print(f"   Database count: {self.get_database_race_count()}")
            print(f"   Progress: {batch_end}/{total_files} files processed")

            # Short break between batches
            await asyncio.sleep(5)

        print("\n🎉 Collection completed successfully!")
        print(f"   Final race count: {self.get_database_race_count()}")

        # Final comprehensive data summary
        summary = self.collector.generate_data_summary()
        print(f"\n📊 Final Summary:")
        print(f"   Races: {summary['race_statistics'].get('total_races', 0)}")
        print(f"   Dogs: {summary['dog_statistics'].get('total_dog_entries', 0)}")
        print(f"   Odds: {summary['odds_statistics'].get('total_odds_entries', 0)}")

        return True


async def main():
    """Main execution function"""
    collector = IntegratedRaceCollector()

    try:
        success = await collector.run_integrated_collection()

        if success:
            print("✅ Collection completed successfully!")
        else:
            print("❌ Collection terminated due to issues.")

    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup resources
        if hasattr(collector.collector, "cleanup"):
            collector.collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
