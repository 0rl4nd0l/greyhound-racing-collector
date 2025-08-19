#!/usr/bin/env python3
"""
Main Database Integrity Check
Verifies integrity, tables existence, and critical data checks.
"""

import sqlite3
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_integrity_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_database_integrity(db_path):
    """Check database integrity and consistency"""
    logger.info("🗃️ Checking database integrity...")
    
    if not os.path.exists(db_path):
        logger.error("MISSING_DATABASE: Database file not found")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verify PRAGMA integrity
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        if result[0] != "ok":
            logger.error(f"INTEGRITY_CHECK_FAILED: {result[0]}")
            return False
        else:
            logger.info("Database passed PRAGMA integrity check")

        # Get all tables in database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found tables: {existing_tables}")
        
        # Check required tables presence (using actual table names)
        required_tables = ['dog_race_data', 'race_metadata', 'track_conditions']
        for table in required_tables:
            if table not in existing_tables:
                logger.error(f"MISSING_TABLE: Required table {table} not found in database")
                return False
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"Table {table}: {count} records")

        # Check critical data in 'race_metadata' table
        if 'race_metadata' in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE race_name IS NULL OR venue IS NULL OR race_date IS NULL")
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                logger.warning(f"NULL_CRITICAL_DATA: {null_count} races have NULL critical fields")

            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]
            logger.info(f"Total races recorded in the database: {total_races}")

            # Check for duplicates
            cursor.execute("SELECT race_name, venue, race_date, COUNT(*) FROM race_metadata GROUP BY race_name, venue, race_date HAVING COUNT(*) > 1")
            duplicates = cursor.fetchall()
            if duplicates:
                logger.warning("DUPLICATE_RACES_FOUND:")
                for row in duplicates:
                    logger.warning(f"Duplicate race: {row}")
        
        # Check dog_race_data integrity
        if 'dog_race_data' in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            dog_records = cursor.fetchone()[0]
            logger.info(f"Total dog race records: {dog_records}")
            
            # Check for NULL values in critical fields
            cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE dog_name IS NULL OR finish_position IS NULL")
            null_dog_data = cursor.fetchone()[0]
            if null_dog_data > 0:
                logger.warning(f"NULL_DOG_DATA: {null_dog_data} dog records have NULL critical fields")

        conn.close()
        logger.info("✅ Database integrity check completed successfully")

    except Exception as e:
        logger.error(f"DATABASE_ERROR: Database error {str(e)}")
        return False
    
    return True

def main():
    """Run integrity check on the database"""
    db_path = "/Users/orlandolee/greyhound_racing_collector/databases/comprehensive_greyhound_data.db"
    
    print("🔍 Starting Main Database Integrity Check...")
    result = check_database_integrity(db_path)
    
    if result:
        print("✅ All database checks passed!")
    else:
        print("❌ Some issues were detected in the database!")

if __name__ == "__main__":
    main()

