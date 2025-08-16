#!/usr/bin/env python3
"""
Comprehensive Form Guide Processor
==================================

This script processes form guide CSV files to extract comprehensive race data by:
1. Reading basic race results from CSV files
2. Web scraping detailed odds and enhanced data
3. Combining both data sources into the database
4. Moving processed files to organized folders
5. Tracking progress to avoid reprocessing
"""

import asyncio
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from enhanced_odds_collector import EnhancedGreyhoundDataCollector


class ComprehensiveFormGuideProcessor:
    def __init__(self):
        self.collector = EnhancedGreyhoundDataCollector()
        self.form_guides_dir = Path("./form_guides")
        self.processed_dir = self.form_guides_dir / "processed"
        self.unprocessed_dir = self.form_guides_dir / "unprocessed"
        self.progress_file = "./form_guide_progress.json"

        # Create directories if they don't exist
        self.processed_dir.mkdir(exist_ok=True)
        self.unprocessed_dir.mkdir(exist_ok=True)

        # Track processing progress
        self.processed_files = self.load_processed_files()

    def load_processed_files(self) -> set:
        """Load list of already processed files"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("processed_files", []))
            except:
                return set()
        return set()

    def save_processed_files(self):
        """Save list of processed files"""
        data = {
            "processed_files": list(self.processed_files),
            "last_update": datetime.now().isoformat(),
            "total_processed": len(self.processed_files),
        }
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def parse_race_filename(self, filename: str) -> Optional[Tuple[int, str, datetime]]:
        """Parse race information from filename"""
        # Format: Race {number} - {venue} - {date}.csv
        pattern = r"Race (\d+) - ([A-Z_]+) - (\d{2} \w+ \d{4})\.csv"
        match = re.match(pattern, filename)

        if match:
            race_number = int(match.group(1))
            venue = match.group(2)
            date_str = match.group(3)

            try:
                race_date = datetime.strptime(date_str, "%d %B %Y")
                return race_number, venue, race_date
            except ValueError:
                print(f"⚠️  Could not parse date from filename: {filename}")
                return None

        print(f"⚠️  Could not parse filename: {filename}")
        return None

    def read_form_guide_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Read and parse form guide CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"❌ Error reading CSV {file_path}: {e}")
            return None

    def extract_race_results_from_csv(self, df: pd.DataFrame) -> Dict:
        """Extract race results from CSV data"""
        try:
            # The CSV contains form guide data with each dog's recent race history
            # Format: Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP
            race_results = {"dogs": [], "form_guide_data": []}

            current_dog = None

            # Process each row in the CSV
            for idx, row in df.iterrows():
                dog_name = row.get("Dog Name", "")

                # Convert to string for processing
                dog_name_str = str(dog_name).strip() if not pd.isna(dog_name) else "nan"

                # Skip truly empty rows (but not NaN rows which might be continuation)
                if dog_name_str == "" or dog_name_str == "None":
                    continue

                # Check if this is a new dog (has a number at the start)
                if (
                    dog_name_str
                    and not dog_name_str.startswith('"')
                    and dog_name_str != "nan"
                ):
                    # Extract dog number and name (e.g., "1. Reiko Barty" -> "Reiko Barty")
                    if ". " in dog_name_str:
                        parts = dog_name_str.split(". ", 1)
                        dog_number = parts[0]
                        actual_name = parts[1]
                    else:
                        dog_number = str(len(race_results["dogs"]) + 1)
                        actual_name = dog_name_str

                    current_dog = {
                        "number": dog_number,
                        "name": actual_name,
                        "sex": row.get("Sex", ""),
                        "recent_form": [],
                        "box": row.get("BOX", None),
                        "weight": row.get("WGT", None),
                    }
                    race_results["dogs"].append(current_dog)

                    # Add the first form line
                    if pd.notna(row.get("PLC")):
                        form_entry = {
                            "placing": row.get("PLC"),
                            "box": row.get("BOX"),
                            "weight": row.get("WGT"),
                            "distance": row.get("DIST"),
                            "date": row.get("DATE"),
                            "track": row.get("TRACK"),
                            "grade": row.get("G"),
                            "time": row.get("TIME"),
                            "win_time": row.get("WIN"),
                            "bonus": row.get("BON"),
                            "first_split": row.get("1 SEC"),
                            "margin": row.get("MGN"),
                            "w2g": row.get("W/2G"),
                            "pir": row.get("PIR"),
                            "sp": row.get("SP"),
                        }
                        current_dog["recent_form"].append(form_entry)

                elif current_dog and (
                    pd.isna(dog_name)
                    or dog_name_str.startswith('"')
                    or dog_name_str == "nan"
                ):
                    # This is a continuation of the previous dog's form
                    if pd.notna(row.get("PLC")):
                        form_entry = {
                            "placing": row.get("PLC"),
                            "box": row.get("BOX"),
                            "weight": row.get("WGT"),
                            "distance": row.get("DIST"),
                            "date": row.get("DATE"),
                            "track": row.get("TRACK"),
                            "grade": row.get("G"),
                            "time": row.get("TIME"),
                            "win_time": row.get("WIN"),
                            "bonus": row.get("BON"),
                            "first_split": row.get("1 SEC"),
                            "margin": row.get("MGN"),
                            "w2g": row.get("W/2G"),
                            "pir": row.get("PIR"),
                            "sp": row.get("SP"),
                        }
                        current_dog["recent_form"].append(form_entry)

            return race_results
        except Exception as e:
            print(f"❌ Error extracting race results: {e}")
            return {}

    def find_race_url(
        self, race_number: int, venue: str, race_date: datetime
    ) -> Optional[str]:
        """Find the race URL on thedogs.com.au"""
        try:
            # Format date for URL
            date_str = race_date.strftime("%Y-%m-%d")

            # Convert venue codes to URL format
            venue_mapping = {
                "AP_K": "angle-park",
                "SAN": "sandown",
                "WAR": "warrnambool",
                "BEN": "bendigo",
                "GEE": "geelong",
                "BAL": "ballarat",
                "HOR": "horsham",
                "TRA": "traralgon",
                "CANN": "cannington",
                "MAND": "mandurah",
                "APWE": "albion-park",
                "IPTU": "ipswich",
                "TEMA": "temora",
                "GOSF": "gosford",
                "DAPT": "dapto",
                "RICH": "richmond",
                "WAGA": "wentworth-park",
                "GAWL": "gawler",
                "ROCK": "rockhampton",
                "TVLE": "townsville",
                "CASE": "casino",
                "LCTN": "launceston",
                "HOBT": "hobart",
                "DUBO": "dubbo",
                "BATT": "bathurst",
                "MAIT": "maitland",
                "GRAF": "grafton",
                "MUSG": "murray-bridge",
                "LISB": "lismore",
                "BULN": "bulli",
                "MEA": "meadows",
                "SHE": "shepparton",
                "IPFR": "ipswich",
                "IPTH": "ipswich",
                "IPMO": "ipswich",
                "IPWE": "ipswich",
                "APTH": "albion-park",
                "APMO": "albion-park",
                "APSU": "albion-park",
                "APFR": "albion-park",
                "APTU": "albion-park",
                "DPRT": "devonport",
                "W_PK": "wentworth-park",
                "MT_G": "mount-gambier",
                "MUSW": "muswellbrook",
                "GRDN": "geraldton",
                "GUNN": "gunnedah",
                "BATT": "bathurst",
                "CASO": "casino",
                "CAPA": "capalaba",
                "HEA": "healesville",
                "SAL": "sale",
                "NOWR": "nowra",
                "WARA": "warragul",
                "DARW": "darwin",
                "RICS": "richmond",
                "CRA": "cranbourne",
                "QOT": "ladbrokes-q-straight",
                "TARE": "taree",
                "GOUL": "goulburn",
            }

            venue_url = venue_mapping.get(venue, venue.lower())

            # Construct the race URL
            race_url = f"https://www.thedogs.com.au/racing/{venue_url}/{date_str}/{race_number}"

            return race_url
        except Exception as e:
            print(f"❌ Error constructing race URL: {e}")
            return None

    def get_database_race_count(self) -> int:
        """Get current number of races in database"""
        try:
            with sqlite3.connect(self.collector.db_path) as conn:
                result = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM race_metadata", conn
                )
                return result.iloc[0]["count"]
        except:
            return 0

    def race_exists_in_database(self, race_id: str) -> bool:
        """Check if race already exists in database"""
        try:
            with sqlite3.connect(self.collector.db_path) as conn:
                result = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM race_metadata WHERE race_id = ?",
                    conn,
                    params=[race_id],
                )
                return result.iloc[0]["count"] > 0
        except:
            return False

    def generate_race_id(
        self, race_number: int, venue: str, race_date: datetime
    ) -> str:
        """Generate consistent race ID"""
        date_str = race_date.strftime("%Y-%m-%d")
        return f"R{race_number:03d}_{date_str}_{venue}"

    async def process_single_form_guide(self, file_path: Path) -> bool:
        """Process a single form guide CSV file"""
        try:
            filename = file_path.name
            print(f"\n📋 Processing: {filename}")

            # Parse filename to get race info
            race_info = self.parse_race_filename(filename)
            if not race_info:
                print(f"❌ Could not parse filename: {filename}")
                return False

            race_number, venue, race_date = race_info
            race_id = self.generate_race_id(race_number, venue, race_date)

            print(
                f"   Race {race_number} at {venue} on {race_date.strftime('%Y-%m-%d')}"
            )
            print(f"   Race ID: {race_id}")

            # Check if already processed
            if self.race_exists_in_database(race_id):
                print(f"   ⏭️  Race already exists in database, skipping")
                return True

            # Read CSV data
            df = self.read_form_guide_csv(file_path)
            if df is None:
                print(f"❌ Could not read CSV file")
                return False

            # Extract basic race results from CSV
            csv_results = self.extract_race_results_from_csv(df)
            print(
                f"   📊 Extracted data for {len(csv_results.get('dogs', []))} dogs from CSV"
            )

            # Find race URL for web scraping
            race_url = self.find_race_url(race_number, venue, race_date)
            if not race_url:
                print(f"❌ Could not construct race URL")
                return False

            print(f"   🌐 Found race URL: {race_url}")

            # Web scrape comprehensive data
            print(f"   🔍 Scraping comprehensive data...")
            web_data = self.collector.collect_comprehensive_race_data(race_url)

            if web_data and web_data.get("race_metadata"):
                # Combine CSV and web data
                combined_data = self.combine_csv_and_web_data(csv_results, web_data)

                # Update database with combined data
                self.update_database_with_combined_data(combined_data, race_id)

                print(
                    f"   ✅ Successfully processed: {len(combined_data.get('dogs_data', []))} dogs"
                )
                print(f"   💾 Saved comprehensive data to database")

                return True
            else:
                print(f"   ⚠️  Web scraping failed, saving CSV data only")
                # Save just the CSV data
                self.save_csv_data_only(
                    csv_results, race_id, race_number, venue, race_date
                )
                return True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False

    def combine_csv_and_web_data(self, csv_data: Dict, web_data: Dict) -> Dict:
        """Combine CSV results with web scraped data"""
        combined_data = web_data.copy()

        # Enhance dogs data with CSV form guide information
        csv_dogs = {dog["name"]: dog for dog in csv_data.get("dogs", [])}

        for dog in combined_data.get("dogs_data", []):
            dog_name = dog.get("name", "")
            if dog_name in csv_dogs:
                csv_dog = csv_dogs[dog_name]
                # Merge CSV data with web data
                dog.update(
                    {
                        "csv_number": csv_dog.get("number"),
                        "csv_sex": csv_dog.get("sex"),
                        "csv_box": csv_dog.get("box"),
                        "csv_weight": csv_dog.get("weight"),
                        "form_guide_data": csv_dog.get("recent_form", []),
                        "data_source": "combined_csv_web",
                    }
                )
            else:
                dog["data_source"] = "web_only"

        # Add any dogs that are in CSV but not in web data
        web_dog_names = {
            dog.get("name", "") for dog in combined_data.get("dogs_data", [])
        }
        for dog_name, csv_dog in csv_dogs.items():
            if dog_name not in web_dog_names:
                # Add CSV-only dog
                csv_only_dog = {
                    "name": dog_name,
                    "csv_number": csv_dog.get("number"),
                    "csv_sex": csv_dog.get("sex"),
                    "csv_box": csv_dog.get("box"),
                    "csv_weight": csv_dog.get("weight"),
                    "form_guide_data": csv_dog.get("recent_form", []),
                    "data_source": "csv_only",
                }
                combined_data["dogs_data"].append(csv_only_dog)

        return combined_data

    def save_csv_data_only(
        self,
        csv_data: Dict,
        race_id: str,
        race_number: int,
        venue: str,
        race_date: datetime,
    ):
        """Save only CSV data when web scraping fails"""
        try:
            with sqlite3.connect(self.collector.db_path) as conn:
                # Save basic race metadata
                race_metadata = {
                    "race_id": race_id,
                    "race_number": race_number,
                    "venue": venue,
                    "race_date": race_date.strftime("%Y-%m-%d"),
                    "data_source": "csv_only",
                }

                metadata_df = pd.DataFrame([race_metadata])
                metadata_df.to_sql(
                    "race_metadata", conn, if_exists="append", index=False
                )

                # Save basic dogs data
                dogs_data = []
                for dog in csv_data.get("dogs", []):
                    dog_data = {
                        "race_id": race_id,
                        "dog_name": dog.get("name"),
                        "box_number": dog.get("box"),
                        "trainer_name": dog.get("trainer"),
                        "finish_position": dog.get("placing"),
                        "individual_time": dog.get("time"),
                        "margin": dog.get("margin"),
                        "odds_decimal": dog.get("starting_price"),
                        "data_source": "csv_only",
                    }
                    dogs_data.append(dog_data)

                if dogs_data:
                    dogs_df = pd.DataFrame(dogs_data)
                    dogs_df.to_sql(
                        "dog_race_data", conn, if_exists="append", index=False
                    )

        except Exception as e:
            print(f"❌ Error saving CSV data: {e}")

    def update_database_with_combined_data(self, combined_data: Dict, race_id: str):
        """Update database with combined CSV and web data"""
        try:
            with sqlite3.connect(self.collector.db_path) as conn:
                # Save race metadata
                race_metadata = combined_data.get("race_metadata", {})
                race_metadata["race_id"] = race_id
                race_metadata["data_source"] = "combined_csv_web"

                metadata_df = pd.DataFrame([race_metadata])
                metadata_df.to_sql(
                    "race_metadata", conn, if_exists="append", index=False
                )

                # Save dogs data with form guide information
                dogs_data = []
                for dog in combined_data.get("dogs_data", []):
                    dog_data = dog.copy()
                    dog_data["race_id"] = race_id

                    # Map field names to match database schema
                    if "name" in dog_data:
                        dog_data["dog_name"] = dog_data.pop("name")

                    # Convert form guide data to JSON string for storage
                    if "form_guide_data" in dog_data:
                        dog_data["form_guide_json"] = json.dumps(
                            dog_data["form_guide_data"]
                        )
                        del dog_data["form_guide_data"]

                    dogs_data.append(dog_data)

                if dogs_data:
                    dogs_df = pd.DataFrame(dogs_data)
                    dogs_df.to_sql(
                        "dog_race_data", conn, if_exists="append", index=False
                    )

                # Save odds data if available
                if "odds_data" in combined_data:
                    odds_data = combined_data["odds_data"]
                    for odds_entry in odds_data:
                        odds_entry["race_id"] = race_id

                    odds_df = pd.DataFrame(odds_data)
                    odds_df.to_sql("odds_data", conn, if_exists="append", index=False)

                # Save market data if available
                if "market_data" in combined_data:
                    market_data = combined_data["market_data"]
                    for market_entry in market_data:
                        market_entry["race_id"] = race_id

                    market_df = pd.DataFrame(market_data)
                    market_df.to_sql(
                        "market_data", conn, if_exists="append", index=False
                    )

        except Exception as e:
            print(f"❌ Error updating database: {e}")

    def move_processed_file(self, file_path: Path):
        """Move processed file to processed folder"""
        try:
            dest_path = self.processed_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            print(f"   📁 Moved to processed folder")
        except Exception as e:
            print(f"❌ Error moving file: {e}")

    def organize_unprocessed_files(self):
        """Move unprocessed files to unprocessed folder"""
        try:
            form_guide_files = list(self.form_guides_dir.glob("Race *.csv"))
            moved_count = 0

            for file_path in form_guide_files:
                if file_path.name not in self.processed_files:
                    dest_path = self.unprocessed_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    moved_count += 1

            if moved_count > 0:
                print(f"📁 Moved {moved_count} unprocessed files to unprocessed folder")
        except Exception as e:
            print(f"❌ Error organizing files: {e}")

    def get_unprocessed_files(self) -> List[Path]:
        """Get list of unprocessed form guide files"""
        unprocessed_files = []

        # Check main directory
        for file_path in self.form_guides_dir.glob("Race *.csv"):
            if file_path.name not in self.processed_files:
                unprocessed_files.append(file_path)

        # Check unprocessed directory
        for file_path in self.unprocessed_dir.glob("Race *.csv"):
            if file_path.name not in self.processed_files:
                unprocessed_files.append(file_path)

        return sorted(unprocessed_files)

    async def process_all_form_guides(self, max_files: Optional[int] = None):
        """Process all unprocessed form guide files"""
        print("🚀 Starting Comprehensive Form Guide Processing")
        print("=" * 60)

        # Get current database stats
        initial_count = self.get_database_race_count()
        print(f"📊 Current database: {initial_count} races")
        print(f"📋 Already processed: {len(self.processed_files)} files")

        # Get unprocessed files
        unprocessed_files = self.get_unprocessed_files()
        print(f"📁 Found {len(unprocessed_files)} unprocessed files")

        if not unprocessed_files:
            print("✅ No unprocessed files found!")
            return

        # Limit processing if specified
        if max_files:
            unprocessed_files = unprocessed_files[:max_files]
            print(f"🔢 Processing first {len(unprocessed_files)} files")

        # Process each file
        successful_count = 0
        failed_count = 0

        for i, file_path in enumerate(unprocessed_files, 1):
            print(f"\n[{i}/{len(unprocessed_files)}] Processing: {file_path.name}")

            success = await self.process_single_form_guide(file_path)

            if success:
                successful_count += 1
                self.processed_files.add(file_path.name)
                self.move_processed_file(file_path)
            else:
                failed_count += 1

            # Save progress every 5 files
            if i % 5 == 0:
                self.save_processed_files()
                print(
                    f"💾 Progress saved: {successful_count} successful, {failed_count} failed"
                )

        # Final save and summary
        self.save_processed_files()
        final_count = self.get_database_race_count()

        print(f"\n" + "=" * 60)
        print(f"🎉 Processing Complete!")
        print(f"✅ Successfully processed: {successful_count} files")
        print(f"❌ Failed: {failed_count} files")
        print(
            f"📊 Database races: {initial_count} → {final_count} (+{final_count - initial_count})"
        )
        print(f"📋 Total processed files: {len(self.processed_files)}")

        # Organize remaining files
        self.organize_unprocessed_files()


async def main():
    """Main function to run the processor"""
    processor = ComprehensiveFormGuideProcessor()

    # Process all form guides (or specify max_files for testing)
    await processor.process_all_form_guides(max_files=None)


if __name__ == "__main__":
    asyncio.run(main())
