#!/usr/bin/env python3
"""
Script to fix field size and implement automated validation checks.
"""

import sqlite3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFixer:
    def __init__(self, db_path='greyhound_racing_data.db'):
        self.db_path = db_path

    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def update_field_sizes(self):
        """Fix field size calculations in database"""
        logger.info("=== UPDATING FIELD SIZES ===")
        
        with self.connect_db() as conn:
            # Update field_size based on actual runner count
            update_query = """
            UPDATE race_metadata 
            SET field_size = (
                SELECT COUNT(*) 
                FROM dog_race_data drd 
                WHERE drd.race_id = race_metadata.race_id
            ),
            actual_field_size = (
                SELECT COUNT(*) 
                FROM dog_race_data drd 
                WHERE drd.race_id = race_metadata.race_id
            );
            """
            
            cursor = conn.cursor()
            cursor.execute(update_query)
            updated_rows = cursor.rowcount
            logger.info(f"Updated field sizes for {updated_rows} races")
            conn.commit()

    def implement_validation_checks(self):
        """Implement simple validation checks"""
        logger.info("=== IMPLEMENTING VALIDATION CHECKS ===")

        validation_queries = [
            {
                'description': 'Detect races with zero runners',
                'query': """SELECT race_id FROM race_metadata WHERE field_size == 0;"""
            },
            {
                'description': 'Detect races with more than 12 runners',
                'query': """SELECT race_id FROM race_metadata WHERE field_size > 12;"""
            }
        ]

        with self.connect_db() as conn:
            for check in validation_queries:
                cursor = conn.cursor()
                cursor.execute(check['query'])
                rows = cursor.fetchall()
                logger.info(f"{check['description']} - Found {len(rows)} issues")

            conn.commit()
            logger.info("Validation checks completed.")


if __name__ == "__main__":
    logger.info("Starting data fix and validation...")
    fixer = DataFixer()
    fixer.update_field_sizes()
    fixer.implement_validation_checks()
    logger.info("Completed data fix and validation.")
