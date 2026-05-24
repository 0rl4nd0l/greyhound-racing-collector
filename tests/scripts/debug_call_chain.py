#!/usr/bin/env python3
"""
Debug Script: Trace Call Chain Analysis
========================================

This script traces the front-end → `/api/upcoming_races_csv` → `load_upcoming_races()` call chain
to identify where duplicates arise and capture reference files.

Author: AI Assistant
Date: January 2025
"""

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f'debug_call_chain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class CallChainDebugger:
    def __init__(self, base_url="http://localhost:5002"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CallChainDebugger/1.0", "Accept": "application/json"}
        )

    def trace_call_chain(self):
        """Main function to trace the complete call chain"""
        logger.info("🔍 Starting Call Chain Debug Analysis")
        logger.info("=" * 60)

        # Step 1: Test the API endpoint directly
        logger.info("📋 STEP 1: Testing /api/upcoming_races_csv endpoint")
        self.test_api_endpoint()

        # Step 2: Analyze the load_upcoming_races() function behavior
        logger.info("\n📋 STEP 2: Analyzing load_upcoming_races() function")
        self.analyze_load_function()

        # Step 3: Check for duplicates and data integrity
        logger.info("\n📋 STEP 3: Checking for duplicates and data integrity")
        self.check_data_integrity()

        # Step 4: Capture reference files
        logger.info("\n📋 STEP 4: Capturing reference files")
        self.capture_reference_files()

        logger.info("\n✅ Call Chain Analysis Complete")

    def test_api_endpoint(self):
        """Test the /api/upcoming_races_csv endpoint with various parameters"""
        endpoints_to_test = [
            "/api/upcoming_races_csv",
            "/api/upcoming_races_csv?refresh=true",
            "/api/upcoming_races_csv?page=1&per_page=5",
            "/api/upcoming_races_csv?search=test",
        ]

        for endpoint in endpoints_to_test:
            logger.info(f"🌐 Testing endpoint: {endpoint}")
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                logger.info(f"   Status Code: {response.status_code}")
                logger.info(f"   Headers: {dict(response.headers)}")

                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"   Success: {data.get('success', 'N/A')}")
                        logger.info(f"   Total Races: {len(data.get('races', []))}")
                        logger.info(f"   Response Keys: {list(data.keys())}")

                        # Log first race for structure analysis
                        races = data.get("races", [])
                        if races:
                            first_race = races[0]
                            logger.info(
                                f"   First Race Keys: {list(first_race.keys())}"
                            )
                            logger.info(
                                f"   First Race Sample: {json.dumps(first_race, indent=2)[:200]}..."
                            )

                            # Check for duplicates by race_id
                            race_ids = [
                                race.get("race_id")
                                for race in races
                                if race.get("race_id")
                            ]
                            unique_race_ids = set(race_ids)
                            if len(race_ids) != len(unique_race_ids):
                                logger.warning(
                                    f"   ⚠️  DUPLICATE DETECTED: {len(race_ids)} total vs {len(unique_race_ids)} unique race_ids"
                                )

                                # Find duplicates
                                from collections import Counter

                                id_counts = Counter(race_ids)
                                duplicates = {
                                    race_id: count
                                    for race_id, count in id_counts.items()
                                    if count > 1
                                }
                                logger.warning(f"   Duplicate race_ids: {duplicates}")

                    except json.JSONDecodeError as e:
                        logger.error(f"   ❌ JSON decode error: {e}")
                        logger.error(
                            f"   Response content (first 500 chars): {response.text[:500]}"
                        )
                else:
                    logger.error(f"   ❌ HTTP Error: {response.status_code}")
                    logger.error(f"   Error content: {response.text[:200]}")

            except requests.RequestException as e:
                logger.error(f"   ❌ Request failed: {e}")

            logger.info("   " + "-" * 40)

    def analyze_load_function(self):
        """Analyze the load_upcoming_races() function behavior by examining the data source"""
        upcoming_dir = "./upcoming_races"
        logger.info(f"📂 Analyzing upcoming races directory: {upcoming_dir}")

        if not os.path.exists(upcoming_dir):
            logger.error(f"   ❌ Directory does not exist: {upcoming_dir}")
            return

        files = os.listdir(upcoming_dir)
        csv_files = [f for f in files if f.endswith(".csv")]
        json_files = [f for f in files if f.endswith(".json")]

        logger.info(
            f"   📊 Found {len(csv_files)} CSV files and {len(json_files)} JSON files"
        )
        logger.info(f"   CSV files: {csv_files}")
        logger.info(f"   JSON files: {json_files}")

        # Analyze CSV files for structure and potential duplicates
        for csv_file in csv_files[:3]:  # Limit to first 3 files for analysis
            self.analyze_csv_file(os.path.join(upcoming_dir, csv_file))

        # Analyze JSON files
        for json_file in json_files[:3]:  # Limit to first 3 files for analysis
            self.analyze_json_file(os.path.join(upcoming_dir, json_file))

    def analyze_csv_file(self, file_path):
        """Analyze a single CSV file"""
        logger.info(f"   📄 Analyzing CSV: {os.path.basename(file_path)}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
            logger.info(f"      Columns: {list(df.columns)}")

            # Check for key columns that might indicate race structure
            key_columns = [
                "Race Name",
                "race_name",
                "Venue",
                "venue",
                "Dog Name",
                "dog_name",
            ]
            found_columns = [col for col in key_columns if col in df.columns]
            logger.info(f"      Key columns found: {found_columns}")

            # Sample a few rows
            if len(df) > 0:
                sample_data = df.head(2).to_dict("records")
                logger.info(
                    f"      Sample rows: {json.dumps(sample_data, indent=2, default=str)[:300]}..."
                )

                # Check if this is a race with multiple dogs (typical form guide structure)
                if "Dog Name" in df.columns or "dog_name" in df.columns:
                    dog_col = "Dog Name" if "Dog Name" in df.columns else "dog_name"
                    unique_dogs = df[dog_col].nunique()
                    logger.info(
                        f"      Unique dogs: {unique_dogs} (suggests this is a race form guide)"
                    )

        except Exception as e:
            logger.error(f"      ❌ Error reading CSV: {e}")

    def analyze_json_file(self, file_path):
        """Analyze a single JSON file"""
        logger.info(f"   📄 Analyzing JSON: {os.path.basename(file_path)}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            logger.info(f"      Data type: {type(data)}")

            if isinstance(data, dict):
                logger.info(f"      Keys: {list(data.keys())}")
                logger.info(
                    f"      Sample: {json.dumps(data, indent=2, default=str)[:300]}..."
                )
            elif isinstance(data, list):
                logger.info(f"      List length: {len(data)}")
                if len(data) > 0:
                    logger.info(f"      First item type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        logger.info(f"      First item keys: {list(data[0].keys())}")
                        logger.info(
                            f"      Sample: {json.dumps(data[0], indent=2, default=str)[:300]}..."
                        )

        except Exception as e:
            logger.error(f"      ❌ Error reading JSON: {e}")

    def check_data_integrity(self):
        """Check for data integrity issues and duplicates"""
        logger.info("🔍 Checking data integrity...")

        try:
            # Get data from the API
            response = self.session.get(f"{self.base_url}/api/upcoming_races_csv")
            if response.status_code != 200:
                logger.error(
                    f"   ❌ Cannot get data for integrity check: {response.status_code}"
                )
                return

            data = response.json()
            races = data.get("races", [])

            if not races:
                logger.warning("   ⚠️  No races found for integrity check")
                return

            logger.info(f"   📊 Analyzing {len(races)} races for integrity issues")

            # Check for duplicate race_ids
            race_ids = [race.get("race_id") for race in races if race.get("race_id")]
            unique_race_ids = set(race_ids)

            if len(race_ids) != len(unique_race_ids):
                logger.warning(
                    f"   ⚠️  DUPLICATE RACE_IDS: {len(race_ids)} total vs {len(unique_race_ids)} unique"
                )

                from collections import Counter

                id_counts = Counter(race_ids)
                duplicates = {
                    race_id: count for race_id, count in id_counts.items() if count > 1
                }
                logger.warning(f"   Duplicate race_ids: {duplicates}")
            else:
                logger.info("   ✅ No duplicate race_ids found")

            # Check for duplicate filenames
            filenames = [race.get("filename") for race in races if race.get("filename")]
            unique_filenames = set(filenames)

            if len(filenames) != len(unique_filenames):
                logger.warning(
                    f"   ⚠️  DUPLICATE FILENAMES: {len(filenames)} total vs {len(unique_filenames)} unique"
                )

                from collections import Counter

                filename_counts = Counter(filenames)
                duplicates = {
                    filename: count
                    for filename, count in filename_counts.items()
                    if count > 1
                }
                logger.warning(f"   Duplicate filenames: {duplicates}")
            else:
                logger.info("   ✅ No duplicate filenames found")

            # Check for missing required fields
            required_fields = ["race_id", "race_name", "venue", "filename"]
            for field in required_fields:
                missing_count = sum(1 for race in races if not race.get(field))
                if missing_count > 0:
                    logger.warning(
                        f"   ⚠️  {missing_count} races missing '{field}' field"
                    )
                else:
                    logger.info(f"   ✅ All races have '{field}' field")

            # Check venue distribution
            venues = [race.get("venue") for race in races if race.get("venue")]
            venue_counts = Counter(venues)
            logger.info(f"   📊 Venue distribution: {dict(venue_counts)}")

        except Exception as e:
            logger.error(f"   ❌ Error in data integrity check: {e}")

    def capture_reference_files(self):
        """Capture reference files for analysis"""
        logger.info("📁 Capturing reference files...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Capture good JSON response
        try:
            response = self.session.get(f"{self.base_url}/api/upcoming_races_csv")
            if response.status_code == 200:
                filename = f"reference_good_response_{timestamp}.json"
                with open(filename, "w") as f:
                    f.write(response.text)
                logger.info(f"   ✅ Saved good JSON response: {filename}")
            else:
                logger.error(
                    f"   ❌ Could not capture good response: {response.status_code}"
                )
        except Exception as e:
            logger.error(f"   ❌ Error capturing good response: {e}")

        # Capture a problematic CSV file if found
        upcoming_dir = "./upcoming_races"
        if os.path.exists(upcoming_dir):
            csv_files = [f for f in os.listdir(upcoming_dir) if f.endswith(".csv")]
            if csv_files:
                # Copy the first CSV file as a reference
                source_file = os.path.join(upcoming_dir, csv_files[0])
                dest_file = f"reference_csv_file_{timestamp}.csv"
                try:
                    import shutil

                    shutil.copy2(source_file, dest_file)
                    logger.info(f"   ✅ Saved reference CSV file: {dest_file}")
                except Exception as e:
                    logger.error(f"   ❌ Error copying CSV file: {e}")

        # Capture request/response with detailed logging
        try:
            logger.info("   🔍 Capturing detailed request/response cycle...")

            # Enable request logging
            import urllib3

            urllib3.disable_warnings()

            # Create detailed request
            response = self.session.get(
                f"{self.base_url}/api/upcoming_races_csv?refresh=true", timeout=30
            )

            # Capture detailed info
            details = {
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "url": f"{self.base_url}/api/upcoming_races_csv?refresh=true",
                    "method": "GET",
                    "headers": dict(self.session.headers),
                },
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content_length": len(response.content),
                    "content_type": response.headers.get("content-type"),
                },
            }

            if response.status_code == 200:
                try:
                    json_data = response.json()
                    details["response"]["json_keys"] = list(json_data.keys())
                    details["response"]["races_count"] = len(json_data.get("races", []))

                    # Check for duplicates in the response
                    races = json_data.get("races", [])
                    if races:
                        race_ids = [r.get("race_id") for r in races if r.get("race_id")]
                        unique_ids = set(race_ids)
                        details["response"]["duplicate_analysis"] = {
                            "total_races": len(races),
                            "total_race_ids": len(race_ids),
                            "unique_race_ids": len(unique_ids),
                            "has_duplicates": len(race_ids) != len(unique_ids),
                        }

                        if len(race_ids) != len(unique_ids):
                            from collections import Counter

                            id_counts = Counter(race_ids)
                            duplicates = {
                                race_id: count
                                for race_id, count in id_counts.items()
                                if count > 1
                            }
                            details["response"]["duplicate_ids"] = duplicates

                except json.JSONDecodeError:
                    details["response"]["json_error"] = "Response is not valid JSON"

            # Save detailed analysis
            detail_file = f"detailed_analysis_{timestamp}.json"
            with open(detail_file, "w") as f:
                json.dump(details, f, indent=2, default=str)
            logger.info(f"   ✅ Saved detailed analysis: {detail_file}")

        except Exception as e:
            logger.error(f"   ❌ Error in detailed capture: {e}")


def main():
    """Main execution function"""
    debugger = CallChainDebugger()

    # Check if the server is running
    try:
        response = requests.get("http://localhost:5002/api/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server is running, starting debug analysis...")
            debugger.trace_call_chain()
        else:
            logger.error(f"❌ Server returned {response.status_code}, cannot proceed")
    except requests.RequestException:
        logger.error(
            "❌ Cannot connect to server. Please ensure Flask app is running on localhost:5002"
        )
        logger.info("💡 Try running: python app.py")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
