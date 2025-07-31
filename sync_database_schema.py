#!/usr/bin/env python3
"""
Database Schema Synchronization for Enhancer Methods
====================================================

This script ensures that the database schema is aligned with the expectations
of the GPT prediction enhancer and related analysis methods.

Key objectives:
1. Verify existence of required tables (gpt_analysis, race_metadata, dog_race_data)
2. Add missing columns non-destructively
3. Create indexes for better performance
4. Validate schema integrity
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSchemaSynchronizer:
    """Synchronizes database schema with enhancer expectations"""
    
    def __init__(self, database_path: str = "greyhound_racing_data.db"):
        self.database_path = database_path
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Connect to the database"""
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to database: {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the database"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return bool(self.cursor.fetchone())
    
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in self.cursor.fetchall()]
        return column_name in columns
    
    def create_gpt_analysis_table(self):
        """Create the gpt_analysis table if it doesn't exist"""
        if self.table_exists('gpt_analysis'):
            logger.info("✓ gpt_analysis table already exists")
            return
        
        logger.info("Creating gpt_analysis table...")
        self.cursor.execute('''
            CREATE TABLE gpt_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                confidence_score REAL,
                tokens_used INTEGER,
                timestamp TEXT NOT NULL,
                model_used TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        ''')
        
        # Create indexes for better performance
        self.cursor.execute('''
            CREATE INDEX idx_gpt_analysis_race_id ON gpt_analysis(race_id)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX idx_gpt_analysis_timestamp ON gpt_analysis(timestamp)
        ''')
        
        logger.info("✓ Created gpt_analysis table with indexes")
    
    def validate_race_metadata_table(self):
        """Validate and enhance race_metadata table"""
        if not self.table_exists('race_metadata'):
            logger.error("✗ race_metadata table does not exist!")
            raise Exception("race_metadata table is required but missing")
        
        logger.info("✓ race_metadata table exists")
        
        # Check for required columns
        required_columns = {
            'race_id': 'TEXT',
            'venue': 'TEXT',
            'race_number': 'INTEGER',
            'race_date': 'DATE',
            'grade': 'TEXT',
            'distance': 'TEXT',
            'track_condition': 'TEXT',
            'weather': 'TEXT',
            'field_size': 'INTEGER'
        }
        
        missing_columns = []
        for column, column_type in required_columns.items():
            if not self.column_exists('race_metadata', column):
                missing_columns.append((column, column_type))
        
        if missing_columns:
            logger.warning(f"Missing columns in race_metadata: {[col[0] for col in missing_columns]}")
            for column, column_type in missing_columns:
                try:
                    self.cursor.execute(f'ALTER TABLE race_metadata ADD COLUMN {column} {column_type}')
                    logger.info(f"✓ Added column {column} to race_metadata")
                except Exception as e:
                    logger.warning(f"Could not add column {column}: {e}")
        else:
            logger.info("✓ All required columns exist in race_metadata")
    
    def validate_dog_race_data_table(self):
        """Validate and enhance dog_race_data table"""
        if not self.table_exists('dog_race_data'):
            logger.error("✗ dog_race_data table does not exist!")
            raise Exception("dog_race_data table is required but missing")
        
        logger.info("✓ dog_race_data table exists")
        
        # Check for required columns
        required_columns = {
            'race_id': 'TEXT',
            'dog_name': 'TEXT',
            'dog_clean_name': 'TEXT',
            'box_number': 'INTEGER',
            'finish_position': 'INTEGER',
            'trainer_name': 'TEXT',
            'weight': 'REAL',
            'starting_price': 'REAL',
            'individual_time': 'TEXT',
            'win_probability': 'REAL',
            'place_probability': 'REAL'
        }
        
        missing_columns = []
        for column, column_type in required_columns.items():
            if not self.column_exists('dog_race_data', column):
                missing_columns.append((column, column_type))
        
        if missing_columns:
            logger.warning(f"Missing columns in dog_race_data: {[col[0] for col in missing_columns]}")
            for column, column_type in missing_columns:
                try:
                    self.cursor.execute(f'ALTER TABLE dog_race_data ADD COLUMN {column} {column_type}')
                    logger.info(f"✓ Added column {column} to dog_race_data")
                except Exception as e:
                    logger.warning(f"Could not add column {column}: {e}")
        else:
            logger.info("✓ All required columns exist in dog_race_data")
    
    def create_performance_indexes(self):
        """Create performance indexes for better query performance"""
        indexes = [
            ('idx_race_metadata_venue_date', 'race_metadata', ['venue', 'race_date']),
            ('idx_race_metadata_race_id', 'race_metadata', ['race_id']),
            ('idx_dog_race_data_race_id', 'dog_race_data', ['race_id']),
            ('idx_dog_race_data_dog_name', 'dog_race_data', ['dog_clean_name']),
            ('idx_dog_race_data_finish_pos', 'dog_race_data', ['finish_position'])
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                columns_str = ', '.join(columns)
                self.cursor.execute(f'''
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table_name}({columns_str})
                ''')
                logger.info(f"✓ Created/verified index {index_name}")
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")
    
    def validate_foreign_keys(self):
        """Validate foreign key relationships"""
        # Enable foreign key constraints
        self.cursor.execute("PRAGMA foreign_keys = ON")
        
        # Check if gpt_analysis foreign key constraint exists
        if self.table_exists('gpt_analysis'):
            self.cursor.execute("PRAGMA foreign_key_check(gpt_analysis)")
            fk_violations = self.cursor.fetchall()
            if fk_violations:
                logger.warning(f"Foreign key violations in gpt_analysis: {len(fk_violations)}")
            else:
                logger.info("✓ Foreign key constraints validated for gpt_analysis")
    
    def run_schema_validation_tests(self):
        """Run comprehensive schema validation tests"""
        logger.info("Running schema validation tests...")
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Required tables exist
        tests_total += 1
        required_tables = ['race_metadata', 'dog_race_data']
        if self.table_exists('gpt_analysis'):
            required_tables.append('gpt_analysis')
        
        all_tables_exist = all(self.table_exists(table) for table in required_tables)
        if all_tables_exist:
            tests_passed += 1
            logger.info("✓ Test 1 passed: All required tables exist")
        else:
            logger.error("✗ Test 1 failed: Missing required tables")
        
        # Test 2: Critical columns exist in race_metadata
        tests_total += 1
        critical_race_columns = ['race_id', 'venue', 'race_date']
        race_columns_exist = all(self.column_exists('race_metadata', col) for col in critical_race_columns)
        if race_columns_exist:
            tests_passed += 1
            logger.info("✓ Test 2 passed: Critical race_metadata columns exist")
        else:
            logger.error("✗ Test 2 failed: Missing critical race_metadata columns")
        
        # Test 3: Critical columns exist in dog_race_data
        tests_total += 1
        critical_dog_columns = ['race_id', 'dog_name', 'finish_position']
        dog_columns_exist = all(self.column_exists('dog_race_data', col) for col in critical_dog_columns)
        if dog_columns_exist:
            tests_passed += 1
            logger.info("✓ Test 3 passed: Critical dog_race_data columns exist")
        else:
            logger.error("✗ Test 3 failed: Missing critical dog_race_data columns")
        
        # Test 4: Data integrity check
        tests_total += 1
        try:
            self.cursor.execute("SELECT COUNT(*) FROM race_metadata")
            race_count = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            dog_count = self.cursor.fetchone()[0]
            
            if race_count > 0 and dog_count > 0:
                tests_passed += 1
                logger.info(f"✓ Test 4 passed: Data exists (races: {race_count}, dogs: {dog_count})")
            else:
                logger.warning(f"⚠ Test 4 warning: Limited data (races: {race_count}, dogs: {dog_count})")
        except Exception as e:
            logger.error(f"✗ Test 4 failed: Data integrity check error: {e}")
        
        logger.info(f"Schema validation: {tests_passed}/{tests_total} tests passed")
        return tests_passed == tests_total
    
    def synchronize_schema(self):
        """Run complete schema synchronization"""
        try:
            self.connect()
            
            logger.info("🔄 Starting database schema synchronization...")
            
            # Step 1: Create/validate core tables
            self.create_gpt_analysis_table()
            self.validate_race_metadata_table()
            self.validate_dog_race_data_table()
            
            # Step 2: Create performance indexes
            self.create_performance_indexes()
            
            # Step 3: Validate foreign keys
            self.validate_foreign_keys()
            
            # Step 4: Commit changes
            self.connection.commit()
            
            # Step 5: Run validation tests
            validation_passed = self.run_schema_validation_tests()
            
            if validation_passed:
                logger.info("✅ Schema synchronization completed successfully!")
            else:
                logger.warning("⚠️ Schema synchronization completed with warnings")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"❌ Schema synchronization failed: {e}")
            if self.connection:
                self.connection.rollback()
            raise
        finally:
            self.disconnect()

def main():
    """Main function to run schema synchronization"""
    print("🗄️ DATABASE SCHEMA SYNCHRONIZATION")
    print("=" * 50)
    
    synchronizer = DatabaseSchemaSynchronizer()
    
    try:
        success = synchronizer.synchronize_schema()
        
        if success:
            print("\n🎉 SYNCHRONIZATION SUCCESSFUL!")
            print("✅ Database schema is now aligned with enhancer expectations")
        else:
            print("\n⚠️ SYNCHRONIZATION COMPLETED WITH WARNINGS")
            print("⚠️ Please review the log messages above")
            
    except Exception as e:
        print(f"\n❌ SYNCHRONIZATION FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
