#!/usr/bin/env python3
"""
Database Schema Update Script
=============================

This script updates the database schema to add missing weather columns
that are expected by the ML System V4 temporal feature builder.

Missing columns identified:
- temperature (REAL)
- humidity (REAL) 
- wind_speed (REAL)

These will be added to the race_metadata table with default values.
"""

import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("schema_update.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def backup_database(db_path: str) -> str:
    """Create a backup of the database before making changes"""
    try:
        db_file = Path(db_path)
        if not db_file.exists():
            logger.error(f"❌ Database file not found: {db_path}")
            return ""

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_file.parent / f"{db_file.stem}_backup_{timestamp}.db"

        # Copy database file
        import shutil

        shutil.copy2(db_path, backup_path)

        logger.info(f"💾 Database backed up to: {backup_path}")
        return str(backup_path)

    except Exception as e:
        logger.error(f"❌ Failed to backup database: {e}")
        return ""


def check_existing_columns(db_path: str) -> dict:
    """Check which columns already exist in race_metadata table"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table schema
        cursor.execute("PRAGMA table_info(race_metadata)")
        columns = cursor.fetchall()

        existing_columns = {col[1]: col[2] for col in columns}  # column_name: data_type
        conn.close()

        logger.info(
            f"📊 Found {len(existing_columns)} existing columns in race_metadata"
        )

        return existing_columns

    except Exception as e:
        logger.error(f"❌ Failed to check existing columns: {e}")
        return {}


def add_missing_columns(db_path: str) -> bool:
    """Add missing weather columns to race_metadata table"""
    logger.info("🔧 Adding missing weather columns to race_metadata table")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check existing columns
        existing_columns = check_existing_columns(db_path)

        # Define required columns with defaults
        required_columns = {
            "temperature": ("REAL", 20.0),  # Default 20°C
            "humidity": ("REAL", 50.0),  # Default 50%
            "wind_speed": ("REAL", 0.0),  # Default 0 km/h
        }

        columns_added = 0

        for column_name, (data_type, default_value) in required_columns.items():
            if column_name not in existing_columns:
                try:
                    # Add column with default value
                    alter_sql = f"ALTER TABLE race_metadata ADD COLUMN {column_name} {data_type} DEFAULT {default_value}"
                    cursor.execute(alter_sql)

                    # Update existing NULL values to default
                    update_sql = f"UPDATE race_metadata SET {column_name} = {default_value} WHERE {column_name} IS NULL"
                    cursor.execute(update_sql)

                    columns_added += 1
                    logger.info(
                        f"   ✅ Added column: {column_name} ({data_type}) with default {default_value}"
                    )

                except Exception as e:
                    logger.error(f"   ❌ Failed to add column {column_name}: {e}")
                    continue
            else:
                logger.info(f"   ⏭️ Column {column_name} already exists")

        # Commit changes
        conn.commit()
        conn.close()

        logger.info(f"✅ Schema update complete: {columns_added} columns added")
        return columns_added > 0

    except Exception as e:
        logger.error(f"❌ Schema update failed: {e}")
        return False


def verify_schema_update(db_path: str) -> bool:
    """Verify that schema update was successful"""
    logger.info("🔍 Verifying schema update...")

    try:
        existing_columns = check_existing_columns(db_path)

        required_columns = ["temperature", "humidity", "wind_speed"]
        missing_columns = [
            col for col in required_columns if col not in existing_columns
        ]

        if not missing_columns:
            logger.info("✅ All required weather columns are present")

            # Check that we have some data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM race_metadata WHERE temperature IS NOT NULL"
            )
            temp_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_count = cursor.fetchone()[0]

            conn.close()

            logger.info(
                f"📊 Weather data status: {temp_count}/{total_count} races have temperature data"
            )

            return True
        else:
            logger.error(f"❌ Missing columns: {missing_columns}")
            return False

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


def populate_default_weather_data(db_path: str) -> bool:
    """Populate default weather data for existing races"""
    logger.info("🌤️ Populating default weather data for existing races...")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get races that need weather data
        cursor.execute(
            """
            SELECT race_id, venue, track_condition 
            FROM race_metadata 
            WHERE temperature IS NULL OR humidity IS NULL OR wind_speed IS NULL
        """
        )

        races_to_update = cursor.fetchall()

        if not races_to_update:
            logger.info("✅ All races already have weather data")
            conn.close()
            return True

        logger.info(f"📊 Found {len(races_to_update)} races needing weather data")

        # Define weather defaults based on track conditions and venues
        weather_defaults = {
            "Good": {"temperature": 22.0, "humidity": 45.0, "wind_speed": 8.0},
            "Fast": {"temperature": 25.0, "humidity": 40.0, "wind_speed": 5.0},
            "Slow": {"temperature": 18.0, "humidity": 70.0, "wind_speed": 12.0},
            "Heavy": {"temperature": 15.0, "humidity": 85.0, "wind_speed": 15.0},
            "Good4": {"temperature": 21.0, "humidity": 50.0, "wind_speed": 10.0},
            "Soft5": {"temperature": 19.0, "humidity": 65.0, "wind_speed": 12.0},
        }

        # Default fallback
        default_weather = {"temperature": 20.0, "humidity": 50.0, "wind_speed": 10.0}

        updated_races = 0

        for race_id, venue, track_condition in races_to_update:
            # Get appropriate weather defaults
            weather = weather_defaults.get(track_condition, default_weather)

            # Add some variation based on venue (simple heuristic)
            if venue and "BALLARAT" in str(venue).upper():
                weather = weather.copy()
                weather["temperature"] -= 2.0  # Cooler climate
                weather["humidity"] += 5.0
            elif venue and any(x in str(venue).upper() for x in ["DARWIN", "CAIRNS"]):
                weather = weather.copy()
                weather["temperature"] += 5.0  # Warmer climate
                weather["humidity"] += 10.0

            try:
                cursor.execute(
                    """
                    UPDATE race_metadata 
                    SET temperature = ?, humidity = ?, wind_speed = ?
                    WHERE race_id = ?
                """,
                    (
                        weather["temperature"],
                        weather["humidity"],
                        weather["wind_speed"],
                        race_id,
                    ),
                )

                updated_races += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to update race {race_id}: {e}")
                continue

        conn.commit()
        conn.close()

        logger.info(f"✅ Updated weather data for {updated_races} races")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to populate weather data: {e}")
        return False


def main():
    """Main execution function"""
    db_path = "greyhound_racing_data.db"

    logger.info("🚀 Starting Database Schema Update")
    logger.info("=" * 50)

    try:
        # Check if database exists
        if not Path(db_path).exists():
            logger.error(f"❌ Database not found: {db_path}")
            return False

        # Step 1: Create backup
        logger.info("📋 Step 1: Creating database backup...")
        backup_path = backup_database(db_path)
        if not backup_path:
            logger.error("❌ Failed to create backup, aborting")
            return False

        # Step 2: Add missing columns
        logger.info("📋 Step 2: Adding missing columns...")
        if not add_missing_columns(db_path):
            logger.error("❌ Failed to add columns")
            return False

        # Step 3: Verify schema update
        logger.info("📋 Step 3: Verifying schema update...")
        if not verify_schema_update(db_path):
            logger.error("❌ Schema verification failed")
            return False

        # Step 4: Populate default weather data
        logger.info("📋 Step 4: Populating default weather data...")
        if not populate_default_weather_data(db_path):
            logger.warning("⚠️ Weather data population had issues")

        logger.info("✅ Database schema update completed successfully!")
        logger.info(f"💾 Backup saved at: {backup_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Schema update process failed: {e}")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        logger.info("🎉 Schema update completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Schema update failed!")
        sys.exit(1)
