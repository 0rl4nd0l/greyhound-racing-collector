#!/usr/bin/env python3
"""
Enhanced Automated Deduplication Script
========================================

This script automatically fixes data integrity issues found by the integrity system:
1. Removes dog-day rule violations (dogs racing multiple times per day)
2. Fixes invalid box numbers 
3. Removes duplicate records
4. Creates detailed reports of all changes

Author: AI Assistant
Date: 2025-01-27
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set
from pathlib import Path
import pandas as pd

class AutomatedDeduplicator:
    """Automated system for fixing data integrity issues"""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.setup_logging()
        self.connection = None
        self.stats = {
            'dog_day_violations_fixed': 0,
            'invalid_box_numbers_fixed': 0,
            'duplicate_records_removed': 0,
            'records_backed_up': 0
        }
        
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/automated_deduplication.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to database"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
        return self.connection
        
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def create_backup_table(self, table_name: str, suffix: str = None) -> str:
        """Create backup table before making changes"""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        backup_table_name = f"{table_name}_backup_{suffix}"
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Create backup table with same structure
            cursor.execute(f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}")
            
            # Count backed up records
            cursor.execute(f"SELECT COUNT(*) FROM {backup_table_name}")
            backup_count = cursor.fetchone()[0]
            
            conn.commit()
            self.logger.info(f"Created backup table {backup_table_name} with {backup_count} records")
            self.stats['records_backed_up'] += backup_count
            
            return backup_table_name
            
        except sqlite3.Error as e:
            self.logger.error(f"Error creating backup table: {e}")
            conn.rollback()
            raise
    
    def fix_dog_day_violations(self) -> int:
        """Fix dog-day rule violations by keeping the most complete record per dog per day"""
        self.logger.info("Starting to fix dog-day rule violations")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        fixed_count = 0
        
        try:
            # Create backup first
            backup_table = self.create_backup_table("enhanced_expert_data", "dog_day_fix")
            
            # Find all dog-day violations
            cursor.execute("""
                SELECT dog_clean_name, race_date, COUNT(*) as violation_count,
                       GROUP_CONCAT(id) as record_ids,
                       GROUP_CONCAT(race_id) as race_ids
                FROM enhanced_expert_data
                GROUP BY dog_clean_name, race_date
                HAVING COUNT(*) > 1
                ORDER BY dog_clean_name, race_date
            """)
            
            violations = cursor.fetchall()
            
            for violation in violations:
                dog_name, race_date, count, record_ids, race_ids = violation
                record_id_list = record_ids.split(',')
                
                self.logger.info(f"Fixing violation: {dog_name} on {race_date} ({count} records)")
                
                # Get detailed info for each duplicate record
                cursor.execute("""
                    SELECT id, race_id, position, weight, distance, grade, race_time, 
                           win_time, bonus_time, first_sectional, margin, pir_rating, starting_price
                    FROM enhanced_expert_data
                    WHERE id IN ({})
                    ORDER BY 
                        CASE WHEN position IS NOT NULL AND position != '' THEN 0 ELSE 1 END,
                        CASE WHEN weight IS NOT NULL AND weight > 0 THEN 0 ELSE 1 END,
                        CASE WHEN race_time IS NOT NULL AND race_time > 0 THEN 0 ELSE 1 END,
                        id DESC
                """.format(','.join(['?'] * len(record_id_list))), record_id_list)
                
                duplicate_records = cursor.fetchall()
                
                if duplicate_records:
                    # Keep the first record (most complete based on ORDER BY)
                    keep_record_id = duplicate_records[0][0]
                    remove_record_ids = [str(record[0]) for record in duplicate_records[1:]]
                    
                    if remove_record_ids:
                        # Remove duplicate records
                        cursor.execute(f"""
                            DELETE FROM enhanced_expert_data 
                            WHERE id IN ({','.join(['?'] * len(remove_record_ids))})
                        """, remove_record_ids)
                        
                        removed_count = cursor.rowcount
                        fixed_count += removed_count
                        
                        self.logger.info(f"Kept record {keep_record_id}, removed {removed_count} duplicates")
            
            conn.commit()
            self.stats['dog_day_violations_fixed'] = fixed_count
            self.logger.info(f"Fixed {fixed_count} dog-day rule violations")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error fixing dog-day violations: {e}")
            conn.rollback()
            raise
            
        return fixed_count
    
    def fix_invalid_box_numbers(self) -> int:
        """Fix invalid box numbers by setting them to NULL or correcting obvious errors"""
        self.logger.info("Starting to fix invalid box numbers")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        fixed_count = 0
        
        try:
            # Create backup first
            backup_table = self.create_backup_table("dog_race_data", "box_number_fix")
            
            # Find records with invalid box numbers
            cursor.execute("""
                SELECT id, race_id, dog_clean_name, box_number
                FROM dog_race_data
                WHERE box_number < 1 OR box_number > 8
                ORDER BY race_id, dog_clean_name
            """)
            
            invalid_records = cursor.fetchall()
            
            for record in invalid_records:
                record_id, race_id, dog_name, box_number = record
                
                self.logger.info(f"Fixing invalid box number: {dog_name} in race {race_id} (box {box_number})")
                
                # Try to infer correct box number from other data or set to NULL
                new_box_number = None
                
                # Strategy 1: Check if there's a pattern (like box_number = 0 should be NULL)
                if box_number == 0:
                    new_box_number = None
                elif box_number < 0:
                    new_box_number = None
                elif box_number > 8:
                    # If it's something like 10, 11, 12, might be 1, 2, 3
                    if 10 <= box_number <= 18:
                        new_box_number = box_number - 10 if box_number - 10 >= 1 else None
                    else:
                        new_box_number = None
                
                # Update the record
                cursor.execute("""
                    UPDATE dog_race_data 
                    SET box_number = ?
                    WHERE id = ?
                """, (new_box_number, record_id))
                
                if cursor.rowcount > 0:
                    fixed_count += 1
                    self.logger.info(f"Updated box number from {box_number} to {new_box_number}")
            
            conn.commit()
            self.stats['invalid_box_numbers_fixed'] = fixed_count
            self.logger.info(f"Fixed {fixed_count} invalid box numbers")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error fixing invalid box numbers: {e}")
            conn.rollback()
            raise
            
        return fixed_count
    
    def remove_duplicate_records(self) -> int:
        """Remove any remaining duplicate records across all tables"""
        self.logger.info("Removing duplicate records from all tables")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        total_removed = 0
        
        tables_to_deduplicate = [
            {
                'name': 'race_metadata',
                'unique_fields': ['race_id'],
                'keep_criteria': 'id DESC'  # Keep latest
            },
            {
                'name': 'dog_race_data', 
                'unique_fields': ['race_id', 'dog_clean_name', 'box_number'],
                'keep_criteria': '''
                    CASE WHEN finish_position IS NOT NULL AND finish_position != '' THEN 0 ELSE 1 END,
                    CASE WHEN weight IS NOT NULL AND weight > 0 THEN 0 ELSE 1 END,
                    id DESC
                '''
            }
        ]
        
        try:
            for table_info in tables_to_deduplicate:
                table_name = table_info['name']
                unique_fields = table_info['unique_fields']
                keep_criteria = table_info['keep_criteria']
                
                # Create backup
                backup_table = self.create_backup_table(table_name, f"dedup_{table_name}")
                
                # Find duplicates
                field_list = ', '.join(unique_fields)
                cursor.execute(f"""
                    SELECT {field_list}, COUNT(*) as dup_count
                    FROM {table_name}
                    GROUP BY {field_list}
                    HAVING COUNT(*) > 1
                """)
                
                duplicates = cursor.fetchall()
                
                for duplicate_group in duplicates:
                    # Get the values for the unique fields
                    field_values = duplicate_group[:-1]  # Exclude count
                    
                    # Build WHERE clause
                    where_conditions = []
                    params = []
                    for i, field in enumerate(unique_fields):
                        where_conditions.append(f"{field} = ?")
                        params.append(field_values[i])
                    
                    where_clause = ' AND '.join(where_conditions)
                    
                    # Get all duplicate records, ordered by keep criteria
                    cursor.execute(f"""
                        SELECT id FROM {table_name}
                        WHERE {where_clause}
                        ORDER BY {keep_criteria}
                    """, params)
                    
                    duplicate_ids = [row[0] for row in cursor.fetchall()]
                    
                    if len(duplicate_ids) > 1:
                        # Keep the first one, remove the rest
                        keep_id = duplicate_ids[0]
                        remove_ids = duplicate_ids[1:]
                        
                        # Remove duplicates
                        cursor.execute(f"""
                            DELETE FROM {table_name}
                            WHERE id IN ({','.join(['?'] * len(remove_ids))})
                        """, remove_ids)
                        
                        removed_count = cursor.rowcount
                        total_removed += removed_count
                        
                        self.logger.info(f"Removed {removed_count} duplicates from {table_name}, kept id {keep_id}")
            
            conn.commit()
            self.stats['duplicate_records_removed'] = total_removed
            self.logger.info(f"Removed {total_removed} duplicate records total")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error removing duplicates: {e}")
            conn.rollback()
            raise
            
        return total_removed
    
    def generate_deduplication_report(self) -> str:
        """Generate detailed report of deduplication activities"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/deduplication_report_{timestamp}.json"
        
        os.makedirs("reports", exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_path': self.db_path,
            'operations_performed': [
                'dog_day_violations_fix',
                'invalid_box_numbers_fix', 
                'duplicate_records_removal'
            ],
            'statistics': self.stats,
            'total_records_modified': (
                self.stats['dog_day_violations_fixed'] + 
                self.stats['invalid_box_numbers_fixed'] + 
                self.stats['duplicate_records_removed']
            ),
            'backup_tables_created': [],
            'recommendations': [
                'Review data ingestion processes to prevent future duplicates',
                'Implement validation at data entry points',
                'Run integrity checks regularly'
            ]
        }
        
        # Get list of backup tables created
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%_backup_%'
                ORDER BY name DESC
                LIMIT 10
            """)
            
            backup_tables = [row[0] for row in cursor.fetchall()]
            report['backup_tables_created'] = backup_tables
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting backup tables: {e}")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Deduplication report generated: {report_path}")
        return report_path
    
    def run_full_deduplication(self) -> Dict:
        """Run complete deduplication process"""
        self.logger.info("Starting full automated deduplication process")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Fix dog-day violations
            print("1. Fixing dog-day rule violations...")
            dog_day_fixed = self.fix_dog_day_violations()
            print(f"✓ Fixed {dog_day_fixed} dog-day violations")
            
            # Step 2: Fix invalid box numbers
            print("2. Fixing invalid box numbers...")
            box_numbers_fixed = self.fix_invalid_box_numbers()
            print(f"✓ Fixed {box_numbers_fixed} invalid box numbers")
            
            # Step 3: Remove remaining duplicates
            print("3. Removing duplicate records...")
            duplicates_removed = self.remove_duplicate_records()
            print(f"✓ Removed {duplicates_removed} duplicate records")
            
            # Step 4: Generate report
            print("4. Generating deduplication report...")
            report_path = self.generate_deduplication_report()
            print(f"✓ Report generated: {report_path}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'success': True,
                'duration_seconds': duration,
                'total_fixes': (
                    dog_day_fixed + box_numbers_fixed + duplicates_removed
                ),
                'statistics': self.stats,
                'report_path': report_path
            }
            
            print(f"\n=== DEDUPLICATION COMPLETE ===")
            print(f"Total fixes: {summary['total_fixes']}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Report: {report_path}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during deduplication: {e}")
            return {
                'success': False,
                'error': str(e),
                'statistics': self.stats
            }

def main():
    """Main function for automated deduplication"""
    print("=== Automated Deduplication System ===\n")
    
    deduplicator = AutomatedDeduplicator()
    
    try:
        with deduplicator:
            result = deduplicator.run_full_deduplication()
            
            if result['success']:
                print("\n✅ Deduplication completed successfully!")
                return 0
            else:
                print(f"\n❌ Deduplication failed: {result['error']}")
                return 1
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
