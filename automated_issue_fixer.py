#!/usr/bin/env python3
"""
Automated Issue Fixer

This script automatically fixes common data quality issues identified by the monitoring system.
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedIssueFixer:
    def __init__(self, db_path='greyhound_racing_data.db'):
        self.db_path = db_path
        
    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def fix_zero_runner_races(self):
        """Fix races with zero runners by removing orphaned metadata"""
        logger.info("=== FIXING ZERO RUNNER RACES ===")
        
        with self.connect_db() as conn:
            # Find races with zero runners
            zero_runner_query = """
            SELECT race_id, venue, race_date
            FROM race_metadata 
            WHERE field_size = 0
            """
            zero_runners = pd.read_sql_query(zero_runner_query, conn)
            
            if len(zero_runners) == 0:
                logger.info("No zero runner races found")
                return
            
            logger.info(f"Found {len(zero_runners)} races with zero runners")
            
            # Option 1: Delete orphaned race metadata
            delete_query = """
            DELETE FROM race_metadata 
            WHERE field_size = 0 AND race_id NOT IN (
                SELECT DISTINCT race_id FROM dog_race_data
            )
            """
            
            cursor = conn.cursor()
            cursor.execute(delete_query)
            deleted_rows = cursor.rowcount
            logger.info(f"Deleted {deleted_rows} orphaned race metadata records")
            conn.commit()
    
    def fix_missing_race_metadata(self):
        """Fix dog race data without corresponding race metadata"""
        logger.info("=== FIXING MISSING RACE METADATA ===")
        
        with self.connect_db() as conn:
            # Find dog race data without race metadata
            missing_metadata_query = """
            SELECT DISTINCT drd.race_id
            FROM dog_race_data drd
            LEFT JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE rm.race_id IS NULL
            LIMIT 20
            """
            
            missing_races = pd.read_sql_query(missing_metadata_query, conn)
            
            if len(missing_races) == 0:
                logger.info("No missing race metadata found")
                return
            
            logger.info(f"Found {len(missing_races)} races missing metadata")
            
            # Create basic race metadata from dog_race_data
            for _, row in missing_races.iterrows():
                race_id = row['race_id']
                
                # Get race info from dog_race_data
                race_info_query = """
                SELECT 
                    race_id,
                    COUNT(*) as field_size
                FROM dog_race_data 
                WHERE race_id = ?
                GROUP BY race_id
                """
                
                race_info = pd.read_sql_query(race_info_query, conn, params=[race_id])
                
                if not race_info.empty:
                    field_size = race_info.iloc[0]['field_size']
                    
                    # Extract venue and other info from race_id if possible
                    venue = 'UNKNOWN'
                    race_date = '2025-01-01'  # Default date
                    
                    # Try to parse race_id for venue info
                    if '_' in race_id:
                        parts = race_id.split('_')
                        if len(parts) >= 2:
                            venue = parts[0].upper()
                            # Try to extract date
                            for part in parts:
                                if len(part) == 10 and part.count('-') == 2:
                                    race_date = part
                                    break
                    
                    # Insert basic race metadata
                    insert_query = """
                    INSERT INTO race_metadata (
                        race_id, venue, race_date, field_size, actual_field_size,
                        distance, grade, extraction_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    cursor = conn.cursor()
                    cursor.execute(insert_query, [
                        race_id, venue, race_date, field_size, field_size,
                        '400m', 'Unknown', datetime.now()
                    ])
                    
                    logger.info(f"Created metadata for race {race_id}: {venue}, {field_size} runners")
            
            conn.commit()
    
    def fix_missing_essential_data(self):
        """Fix records with missing essential data"""
        logger.info("=== FIXING MISSING ESSENTIAL DATA ===")
        
        with self.connect_db() as conn:
            # Update null venues
            update_venue_query = """
            UPDATE race_metadata 
            SET venue = 'UNKNOWN'
            WHERE venue IS NULL
            """
            
            # Update null distances
            update_distance_query = """
            UPDATE race_metadata 
            SET distance = '400m'
            WHERE distance IS NULL
            """
            
            # Update null race dates
            update_date_query = """
            UPDATE race_metadata 
            SET race_date = '2025-01-01'
            WHERE race_date IS NULL
            """
            
            cursor = conn.cursor()
            
            cursor.execute(update_venue_query)
            venues_updated = cursor.rowcount
            
            cursor.execute(update_distance_query)
            distances_updated = cursor.rowcount
            
            cursor.execute(update_date_query)
            dates_updated = cursor.rowcount
            
            logger.info(f"Updated {venues_updated} null venues")
            logger.info(f"Updated {distances_updated} null distances") 
            logger.info(f"Updated {dates_updated} null dates")
            
            conn.commit()
    
    def optimize_single_runner_races(self):
        """Analyze and optimize single runner races"""
        logger.info("=== ANALYZING SINGLE RUNNER RACES ===")
        
        with self.connect_db() as conn:
            # Get statistics on single runner races
            single_runner_stats = """
            SELECT 
                venue,
                grade,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM race_metadata) as percentage
            FROM race_metadata 
            WHERE field_size = 1
            GROUP BY venue, grade
            ORDER BY count DESC
            LIMIT 10
            """
            
            stats_df = pd.read_sql_query(single_runner_stats, conn)
            
            logger.info("Top venues with single runner races:")
            for _, row in stats_df.iterrows():
                logger.info(f"  {row['venue']} {row['grade']}: {row['count']} races ({row['percentage']:.2f}%)")
            
            # Mark suspicious single runner races
            mark_suspicious_query = """
            UPDATE race_metadata 
            SET data_quality_note = COALESCE(data_quality_note || '; ', '') || 'Single runner race - verify legitimacy'
            WHERE field_size = 1 
            AND grade NOT LIKE '%walkover%' 
            AND grade NOT LIKE '%scratch%'
            AND data_quality_note NOT LIKE '%Single runner race%'
            """
            
            cursor = conn.cursor()
            cursor.execute(mark_suspicious_query)
            marked_races = cursor.rowcount
            logger.info(f"Marked {marked_races} suspicious single runner races for review")
            
            conn.commit()
    
    def create_data_quality_indexes(self):
        """Create database indexes for better data quality monitoring"""
        logger.info("=== CREATING DATA QUALITY INDEXES ===")
        
        with self.connect_db() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_race_metadata_field_size ON race_metadata(field_size)",
                "CREATE INDEX IF NOT EXISTS idx_race_metadata_venue_date ON race_metadata(venue, race_date)",
                "CREATE INDEX IF NOT EXISTS idx_dog_race_data_race_id ON dog_race_data(race_id)",
                "CREATE INDEX IF NOT EXISTS idx_race_metadata_data_quality ON race_metadata(data_quality_note)"
            ]
            
            cursor = conn.cursor()
            for idx_query in indexes:
                try:
                    cursor.execute(idx_query)
                    logger.info(f"Created index: {idx_query.split()[-1]}")
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
            
            conn.commit()

def main():
    """Main fixing function"""
    logger.info("Starting Automated Issue Fixer...")
    
    fixer = AutomatedIssueFixer()
    
    # Fix issues in order of severity
    fixer.fix_zero_runner_races()
    fixer.fix_missing_race_metadata()
    fixer.fix_missing_essential_data()
    fixer.optimize_single_runner_races()
    fixer.create_data_quality_indexes()
    
    logger.info("\n=== AUTOMATED FIXES COMPLETE ===")
    logger.info("Run the monitoring system again to verify fixes")

if __name__ == "__main__":
    main()
