#!/usr/bin/env python3
"""
Unified Database Creation Script
Consolidates all CSV race data into a single, efficient SQLite database.
"""

import glob
import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedDatabaseCreator:
    def __init__(self, base_path="/Users/orlandolee/greyhound_racing_collector"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "databases", "race_data.db")
        self.stats = {
            "races_processed": 0,
            "dogs_processed": 0,
            "files_processed": 0,
            "errors": 0,
        }

    def create_unified_database(self):
        """Create unified database with optimized schema"""
        logger.info("🏗️ Creating unified database schema...")

        # Backup existing database if it exists
        if os.path.exists(self.db_path):
            backup_path = (
                f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.rename(self.db_path, backup_path)
            logger.info(f"Backed up existing database to {backup_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create optimized schema
        cursor.executescript(
            """
        CREATE TABLE race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                track_condition TEXT,
                weather TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction TEXT,
                track_record TEXT,
                prize_money_total REAL,
                prize_money_breakdown TEXT,
                race_time TEXT,
                field_size INTEGER,
                url TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                winner_name TEXT,
                winner_odds REAL,
                winner_margin REAL,
                race_status TEXT,
                data_quality_note TEXT, actual_field_size INTEGER, scratched_count INTEGER, scratch_rate REAL, box_analysis TEXT, weather_condition TEXT, precipitation REAL, pressure REAL, visibility REAL, weather_location TEXT, weather_timestamp DATETIME, weather_adjustment_factor REAL, sportsbet_url TEXT, venue_slug TEXT,
                UNIQUE(race_id)
            );
        
        CREATE TABLE dog_race_data(
            id INT,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            dog_id INT,
            box_number INT,
            trainer_name TEXT,
            trainer_id INT,
            weight REAL,
            running_style TEXT,
            odds_decimal REAL,
            odds_fractional TEXT,
            starting_price REAL,
            individual_time TEXT,
            sectional_1st TEXT,
            sectional_2nd TEXT,
            sectional_3rd TEXT,
            margin TEXT,
            beaten_margin REAL,
            was_scratched NUM,
            blackbook_link TEXT,
            extraction_timestamp NUM,
            data_source TEXT,
            form_guide_json TEXT,
            historical_records TEXT,
            performance_rating REAL,
            speed_rating REAL,
            class_rating REAL,
            recent_form TEXT,
            win_probability REAL,
            place_probability REAL,
            scraped_trainer_name TEXT,
            scraped_reaction_time TEXT,
            scraped_nbtt TEXT,
            scraped_race_classification TEXT,
            scraped_raw_result TEXT,
            scraped_finish_position TEXT,
            best_time REAL,
            data_quality_note TEXT, 
            finish_position INTEGER
        );

        CREATE TABLE dogs (
            dog_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dog_name TEXT UNIQUE NOT NULL,
            total_races INTEGER DEFAULT 0,
            total_wins INTEGER DEFAULT 0,
            total_places INTEGER DEFAULT 0,
            best_time REAL,
            average_position REAL,
            last_race_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        )

        conn.commit()
        conn.close()
        logger.info("✅ Database schema created successfully")

    def load_processed_races(self):
        """Load data from processed race files"""
        logger.info("📊 Loading processed race data...")

        conn = sqlite3.connect(self.db_path)

        # Load from completed races
        completed_dir = os.path.join(self.base_path, "processed", "completed")
        csv_files = glob.glob(os.path.join(completed_dir, "*.csv"))

        logger.info(f"Found {len(csv_files)} completed race files")

        for csv_file in csv_files:
            try:
                self._process_race_file(csv_file, conn)
                self.stats["files_processed"] += 1

                if self.stats["files_processed"] % 50 == 0:
                    logger.info(f"Processed {self.stats['files_processed']} files...")

            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                self.stats["errors"] += 1

        conn.close()
        logger.info(
            f"✅ Loaded {self.stats['races_processed']} races from processed files"
        )

    def load_form_guide_data(self):
        """Load form guide data"""
        logger.info("📋 Loading form guide data...")

        conn = sqlite3.connect(self.db_path)

        # Load from form guides
        form_dirs = [
            os.path.join(self.base_path, "form_guides", "downloaded"),
            os.path.join(self.base_path, "unprocessed"),
        ]

        total_files = 0
        for form_dir in form_dirs:
            if os.path.exists(form_dir):
                csv_files = glob.glob(os.path.join(form_dir, "*.csv"))
                total_files += len(csv_files)

                logger.info(
                    f"Loading {len(csv_files)} files from {os.path.basename(form_dir)}"
                )

                for csv_file in csv_files:
                    try:
                        self._process_form_file(csv_file, conn)

                        if total_files % 100 == 0:
                            logger.info(f"Processed {total_files} form files...")

                    except Exception as e:
                        logger.error(f"Error processing form file {csv_file}: {e}")
                        self.stats["errors"] += 1

        conn.close()
        logger.info(f"✅ Loaded form guide data from {total_files} files")

    def _process_race_file(self, csv_file, conn):
        """Process individual race file"""
        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                return

            # Extract race info from filename
            filename = os.path.basename(csv_file)
            race_info = self._parse_race_filename(filename)

            if not race_info:
                return

            # Insert race record
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO races (race_name, venue, race_date, distance, grade)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    race_info["name"],
                    race_info["venue"],
                    race_info["date"],
                    race_info.get("distance"),
                    race_info.get("grade"),
                ),
            )

            # Get race_id
            cursor.execute(
                """
                SELECT race_id FROM races 
                WHERE race_name = ? AND venue = ? AND race_date = ?
            """,
                (race_info["name"], race_info["venue"], race_info["date"]),
            )

            race_result = cursor.fetchone()
            if not race_result:
                return

            race_id = race_result[0]

            # Process dog performances
            for _, row in df.iterrows():
                try:
                    # Handle different column name variations
                    dog_name = self._get_column_value(
                        row, ["Dog", "Dog Name", "dog_name"]
                    )
                    box_number = self._get_column_value(row, ["Box", "BOX", "box"])
                    position = self._get_column_value(
                        row, ["Position", "PLC", "Finish"]
                    )
                    weight = self._get_column_value(row, ["Weight", "WGT", "weight"])
                    trainer = self._get_column_value(row, ["Trainer", "trainer"])

                    if dog_name and not pd.isna(dog_name):
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO dog_performances 
                            (race_id, dog_name, box_number, finish_position, weight, trainer)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """,
                            (
                                race_id,
                                str(dog_name).strip(),
                                box_number,
                                position,
                                weight,
                                trainer,
                            ),
                        )

                        self.stats["dogs_processed"] += 1

                except Exception as e:
                    logger.debug(f"Error processing dog record: {e}")

            conn.commit()
            self.stats["races_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing race file {csv_file}: {e}")

    def _process_form_file(self, csv_file, conn):
        """Process form guide file"""
        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                return

            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    dog_name = self._get_column_value(
                        row, ["Dog Name", "Dog", "dog_name"]
                    )
                    race_date = self._get_column_value(
                        row, ["DATE", "Date", "race_date"]
                    )
                    venue = self._get_column_value(row, ["TRACK", "Track", "venue"])
                    position = self._get_column_value(row, ["PLC", "Position", "place"])
                    weight = self._get_column_value(row, ["WGT", "Weight", "weight"])

                    if dog_name and not pd.isna(dog_name):
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO form_guide 
                            (dog_name, race_date, venue, finish_position, weight)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            (str(dog_name).strip(), race_date, venue, position, weight),
                        )

                except Exception as e:
                    continue  # Skip problematic rows

            conn.commit()

        except Exception as e:
            logger.debug(f"Error processing form file {csv_file}: {e}")

    def _get_column_value(self, row, possible_names):
        """Get value from row using multiple possible column names"""
        for name in possible_names:
            if name in row and not pd.isna(row[name]):
                return row[name]
        return None

    def _parse_race_filename(self, filename):
        """Parse race information from filename"""
        try:
            # Handle different filename formats
            if " - " in filename:
                parts = filename.replace(".csv", "").split(" - ")
                if len(parts) >= 3:
                    return {
                        "name": parts[0].strip(),
                        "venue": parts[1].strip(),
                        "date": parts[2].strip(),
                    }

            # Handle underscore format
            if "_" in filename:
                parts = filename.replace(".csv", "").split("_")
                if len(parts) >= 3:
                    return {
                        "name": parts[0].strip(),
                        "venue": parts[1].strip(),
                        "date": parts[2].strip(),
                    }

            return None

        except Exception:
            return None

    def update_dog_statistics(self):
        """Update dog master table with aggregated statistics"""
        logger.info("📊 Updating dog statistics...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update from dog_performances
        cursor.execute(
            """
            INSERT OR REPLACE INTO dogs (dog_name, total_races, total_wins, total_places, 
                                       average_position, last_race_date)
            SELECT 
                dp.dog_name,
                COUNT(*) as total_races,
                SUM(CASE WHEN dp.finish_position = 1 THEN 1 ELSE 0 END) as total_wins,
                SUM(CASE WHEN dp.finish_position <= 3 THEN 1 ELSE 0 END) as total_places,
                AVG(CAST(dp.finish_position AS REAL)) as average_position,
                MAX(r.race_date) as last_race_date
            FROM dog_performances dp
            JOIN races r ON dp.race_id = r.race_id
            WHERE dp.dog_name IS NOT NULL
            GROUP BY dp.dog_name
        """
        )

        conn.commit()
        conn.close()

        logger.info("✅ Dog statistics updated")

    def create_database_views(self):
        """Create useful database views for common queries"""
        logger.info("🔍 Creating database views...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executescript(
            """
        -- View for recent dog form
        CREATE VIEW IF NOT EXISTS recent_dog_form AS
        SELECT 
            dp.dog_name,
            r.race_date,
            r.venue,
            r.distance,
            dp.finish_position,
            dp.box_number,
            dp.weight,
            ROW_NUMBER() OVER (PARTITION BY dp.dog_name ORDER BY r.race_date DESC) as race_recency
        FROM dog_performances dp
        JOIN races r ON dp.race_id = r.race_id
        WHERE r.race_date IS NOT NULL
        ORDER BY dp.dog_name, r.race_date DESC;
        
        -- View for venue statistics
        CREATE VIEW IF NOT EXISTS venue_stats AS
        SELECT 
            r.venue,
            COUNT(DISTINCT r.race_id) as total_races,
            COUNT(DISTINCT dp.dog_name) as unique_dogs,
            AVG(CAST(dp.finish_position AS REAL)) as avg_finish_position,
            MIN(r.race_date) as first_race_date,
            MAX(r.race_date) as last_race_date
        FROM races r
        JOIN dog_performances dp ON r.race_id = dp.race_id
        GROUP BY r.venue;
        
        -- View for dog performance summary
        CREATE VIEW IF NOT EXISTS dog_summary AS
        SELECT 
            d.dog_name,
            d.total_races,
            d.total_wins,
            d.total_places,
            ROUND(d.average_position, 2) as avg_position,
            ROUND((d.total_wins * 100.0 / d.total_races), 2) as win_rate,
            ROUND((d.total_places * 100.0 / d.total_races), 2) as place_rate,
            d.last_race_date
        FROM dogs d
        WHERE d.total_races > 0
        ORDER BY d.total_races DESC;
        """
        )

        conn.commit()
        conn.close()

        logger.info("✅ Database views created")

    def generate_database_report(self):
        """Generate comprehensive database report"""
        logger.info("📊 Generating database report...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM races")
        total_races = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dog_performances")
        total_performances = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT dog_name) FROM dogs")
        unique_dogs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM form_guide")
        form_records = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT venue) FROM races")
        unique_venues = cursor.fetchone()[0]

        # Database size
        cursor.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        )
        db_size = cursor.fetchone()[0]

        conn.close()

        print("\n" + "=" * 60)
        print("📊 UNIFIED DATABASE REPORT")
        print("=" * 60)
        print(f"Database Location: {self.db_path}")
        print(f"Database Size: {db_size / (1024*1024):.2f} MB")
        print(f"Created: {datetime.now().isoformat()}")
        print()
        print("📈 DATA STATISTICS:")
        print(f"  Total Races: {total_races:,}")
        print(f"  Total Dog Performances: {total_performances:,}")
        print(f"  Unique Dogs: {unique_dogs:,}")
        print(f"  Form Guide Records: {form_records:,}")
        print(f"  Unique Venues: {unique_venues:,}")
        print()
        print("🔧 PROCESSING STATISTICS:")
        print(f"  Files Processed: {self.stats['files_processed']:,}")
        print(f"  Races Processed: {self.stats['races_processed']:,}")
        print(f"  Dogs Processed: {self.stats['dogs_processed']:,}")
        print(f"  Errors: {self.stats['errors']:,}")
        print()
        print("✅ DATABASE BENEFITS:")
        print("  • Single source of truth for all race data")
        print("  • Fast SQL queries across all historical data")
        print("  • Automatic data deduplication")
        print("  • Efficient storage and indexing")
        print("  • Easy backup and replication")
        print("  • Support for complex analytics")
        print("=" * 60)

        return {
            "total_races": total_races,
            "total_performances": total_performances,
            "unique_dogs": unique_dogs,
            "db_size_mb": db_size / (1024 * 1024),
        }

    def run_creation_process(self):
        """Run the complete database creation process"""
        logger.info("🚀 Starting unified database creation process...")

        start_time = datetime.now()

        try:
            # Step 1: Create database schema
            self.create_unified_database()

            # Step 2: Load processed race data
            self.load_processed_races()

            # Step 3: Load form guide data
            self.load_form_guide_data()

            # Step 4: Update dog statistics
            self.update_dog_statistics()

            # Step 5: Create views
            self.create_database_views()

            # Step 6: Generate report
            stats = self.generate_database_report()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"✅ Database creation completed in {duration:.2f} seconds")

            return True

        except Exception as e:
            logger.error(f"❌ Database creation failed: {e}")
            return False


def main():
    """Main function to create unified database"""
    creator = UnifiedDatabaseCreator()
    success = creator.run_creation_process()

    if success:
        print("\n🎉 SUCCESS: Unified database created successfully!")
        print("\n💡 NEXT STEPS:")
        print("1. Update your ML system to use the new database")
        print("2. Modify prediction scripts to query the database")
        print("3. Set up regular database maintenance")
        print("4. Consider archiving old CSV files")
    else:
        print("\n❌ FAILED: Database creation encountered errors")
        print("Check the logs for details and try again")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
