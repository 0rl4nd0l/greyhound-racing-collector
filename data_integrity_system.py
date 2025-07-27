#!/usr/bin/env python3
"""
Comprehensive Data Integrity System for Greyhound Racing Data
=============================================================

This system implements multiple layers of protection to prevent duplicate entries:
1. Database schema constraints
2. Pre-ingestion validation
3. Data validation on ingestion
4. Automated monitoring and alerts
5. Regular integrity checks

Author: AI Assistant
Date: 2025-01-27
"""

import sqlite3
import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import pandas as pd
import time

class DataIntegrityManager:
    """Main class for managing data integrity and preventing duplicates"""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.setup_logging()
        self.connection = None
        self.integrity_log_path = "logs/data_integrity.log"
        self.validation_rules = self.load_validation_rules()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs("logs", exist_ok=True)
        
        # Configure main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_integrity.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Configure integrity-specific logger
        self.integrity_logger = logging.getLogger('integrity')
        integrity_handler = logging.FileHandler('logs/integrity_alerts.log')
        integrity_handler.setFormatter(
            logging.Formatter('%(asctime)s - INTEGRITY - %(levelname)s - %(message)s')
        )
        self.integrity_logger.addHandler(integrity_handler)
        
    def load_validation_rules(self) -> Dict:
        """Load validation rules configuration"""
        return {
            'race_metadata': {
                'required_fields': ['race_id', 'venue', 'race_date', 'race_number'],
                'unique_constraints': ['race_id'],
                'data_types': {
                    'race_number': int,
                    'race_date': str,
                    'temperature': (int, float, type(None)),
                    'humidity': (int, float, type(None))
                }
            },
            'dog_race_data': {
                'required_fields': ['race_id', 'dog_clean_name', 'box_number'],
                'unique_constraints': ['race_id', 'dog_clean_name', 'box_number'],
                'business_rules': {
                    'box_number_range': (1, 8),
                    'finish_position_range': (1, 8)
                }
            },
            'enhanced_expert_data': {
                'required_fields': ['race_id', 'dog_clean_name'],
                'unique_constraints': ['race_id', 'dog_clean_name'],
                'business_rules': {
                    'one_race_per_dog_per_day': True
                }
            }
        }
    
    def connect(self):
        """Establish database connection with integrity constraints enabled"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA integrity_check")
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

    def add_database_constraints(self):
        """Add comprehensive database constraints to prevent duplicates"""
        self.logger.info("Adding database constraints to prevent duplicates")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Create constraints for race_metadata
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_race_metadata_unique 
                ON race_metadata(race_id)
            """)
            
            # Create constraints for dog_race_data - prevent duplicate dog+race combinations
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_dog_race_unique 
                ON dog_race_data(race_id, dog_clean_name, box_number)
            """)
            
            # Create constraints for enhanced_expert_data
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_enhanced_expert_unique 
                ON enhanced_expert_data(race_id, dog_clean_name)
            """)
            
            # Create index to enforce one race per dog per day rule
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dog_date_check 
                ON enhanced_expert_data(dog_clean_name, race_date)
            """)
            
            # Additional performance indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_race_date 
                ON race_metadata(race_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dog_name 
                ON dog_race_data(dog_clean_name)
            """)
            
            conn.commit()
            self.logger.info("Database constraints added successfully")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error adding database constraints: {e}")
            conn.rollback()
            raise
    
    def validate_record_before_insert(self, table_name: str, record: Dict) -> Tuple[bool, List[str]]:
        """Validate a record before insertion to prevent duplicates and ensure data quality"""
        errors = []
        
        if table_name not in self.validation_rules:
            return True, []  # No rules defined, allow insertion
            
        rules = self.validation_rules[table_name]
        
        # Check required fields
        for field in rules.get('required_fields', []):
            if field not in record or record[field] is None or record[field] == '':
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        for field, expected_type in rules.get('data_types', {}).items():
            if field in record and record[field] is not None:
                if not isinstance(record[field], expected_type):
                    errors.append(f"Invalid data type for {field}: expected {expected_type}, got {type(record[field])}")
        
        # Check business rules
        business_rules = rules.get('business_rules', {})
        
        if 'box_number_range' in business_rules:
            min_box, max_box = business_rules['box_number_range']
            if 'box_number' in record and record['box_number'] is not None:
                if not (min_box <= record['box_number'] <= max_box):
                    errors.append(f"Box number {record['box_number']} outside valid range {min_box}-{max_box}")
        
        # Check for existing duplicates
        duplicate_errors = self.check_for_duplicates_before_insert(table_name, record)
        errors.extend(duplicate_errors)
        
        # Special rule: one race per dog per day
        if business_rules.get('one_race_per_dog_per_day') and 'dog_clean_name' in record and 'race_date' in record:
            dog_day_errors = self.check_one_race_per_dog_per_day(record['dog_clean_name'], record['race_date'], record.get('race_id'))
            errors.extend(dog_day_errors)
        
        return len(errors) == 0, errors
    
    def check_for_duplicates_before_insert(self, table_name: str, record: Dict) -> List[str]:
        """Check for duplicate records before insertion"""
        errors = []
        
        if table_name not in self.validation_rules:
            return errors
            
        unique_constraints = self.validation_rules[table_name].get('unique_constraints', [])
        
        if not unique_constraints:
            return errors
            
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Build WHERE clause for unique constraints
            where_conditions = []
            params = []
            
            for field in unique_constraints:
                if field in record and record[field] is not None:
                    where_conditions.append(f"{field} = ?")
                    params.append(record[field])
            
            if where_conditions:
                query = f"SELECT COUNT(*) FROM {table_name} WHERE {' AND '.join(where_conditions)}"
                cursor.execute(query, params)
                count = cursor.fetchone()[0]
                
                if count > 0:
                    constraint_desc = " + ".join([f"{field}={record.get(field)}" for field in unique_constraints])
                    errors.append(f"Duplicate record found in {table_name}: {constraint_desc}")
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error checking for duplicates in {table_name}: {e}")
            errors.append(f"Database error during duplicate check: {e}")
            
        return errors
    
    def check_one_race_per_dog_per_day(self, dog_name: str, race_date: str, exclude_race_id: str = None) -> List[str]:
        """Ensure a dog can only race once per day"""
        errors = []
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Check enhanced_expert_data table
            query = """
                SELECT race_id, COUNT(*) as race_count 
                FROM enhanced_expert_data 
                WHERE dog_clean_name = ? AND race_date = ?
            """
            params = [dog_name, race_date]
            
            if exclude_race_id:
                query += " AND race_id != ?"
                params.append(exclude_race_id)
                
            query += " GROUP BY dog_clean_name, race_date"
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result and result[1] > 0:
                errors.append(f"Dog {dog_name} already has a race on {race_date} (race_id: {result[0]})")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error checking one-race-per-day rule: {e}")
            errors.append(f"Database error during one-race-per-day check: {e}")
            
        return errors
    
    def safe_insert_record(self, table_name: str, record: Dict) -> Tuple[bool, List[str]]:
        """Safely insert a record with comprehensive validation"""
        # Pre-insertion validation
        is_valid, validation_errors = self.validate_record_before_insert(table_name, record)
        
        if not is_valid:
            self.integrity_logger.warning(f"Record validation failed for {table_name}: {validation_errors}")
            return False, validation_errors
        
        # Attempt insertion
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Build INSERT statement
            fields = list(record.keys())
            placeholders = ['?'] * len(fields)
            values = [record[field] for field in fields]
            
            query = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
            
            cursor.execute(query, values)
            conn.commit()
            
            self.logger.info(f"Successfully inserted record into {table_name}")
            return True, []
            
        except sqlite3.IntegrityError as e:
            error_msg = f"Integrity constraint violation in {table_name}: {e}"
            self.integrity_logger.error(error_msg)
            conn.rollback()
            return False, [error_msg]
            
        except sqlite3.Error as e:
            error_msg = f"Database error during insertion in {table_name}: {e}"
            self.logger.error(error_msg)
            conn.rollback()
            return False, [error_msg]
    
    def run_comprehensive_integrity_check(self) -> Dict:
        """Run comprehensive integrity checks and return detailed report"""
        self.logger.info("Starting comprehensive integrity check")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_path': self.db_path,
            'checks_performed': [],
            'issues_found': [],
            'statistics': {},
            'recommendations': []
        }
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Check 1: Database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            report['checks_performed'].append('sqlite_integrity_check')
            
            if integrity_result != 'ok':
                report['issues_found'].append(f"SQLite integrity check failed: {integrity_result}")
            
            # Check 2: Duplicate records in each table
            tables_to_check = ['race_metadata', 'dog_race_data', 'enhanced_expert_data']
            
            for table in tables_to_check:
                if table in self.validation_rules:
                    unique_fields = self.validation_rules[table].get('unique_constraints', [])
                    if unique_fields:
                        duplicates = self.find_duplicate_records(table, unique_fields)
                        if duplicates:
                            report['issues_found'].append(f"Found {len(duplicates)} duplicate groups in {table}")
                            report['statistics'][f'{table}_duplicates'] = len(duplicates)
            
            # Check 3: One race per dog per day rule
            dog_day_violations = self.find_dog_day_violations()
            if dog_day_violations:
                report['issues_found'].append(f"Found {len(dog_day_violations)} dog-day rule violations")
                report['statistics']['dog_day_violations'] = len(dog_day_violations)
            
            # Check 4: Orphaned records
            orphaned_records = self.find_orphaned_records()
            if orphaned_records:
                report['issues_found'].append(f"Found orphaned records: {orphaned_records}")
            
            # Check 5: Data quality issues
            quality_issues = self.check_data_quality()
            if quality_issues:
                report['issues_found'].extend(quality_issues)
            
            # Generate statistics
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    report['statistics'][f'{table}_record_count'] = count
                except sqlite3.Error:
                    continue
            
            # Generate recommendations
            if report['issues_found']:
                report['recommendations'].append("Run deduplication script immediately")
                report['recommendations'].append("Review data ingestion processes")
                report['recommendations'].append("Consider implementing stricter validation")
            else:
                report['recommendations'].append("Data integrity is good - maintain current practices")
            
            report['checks_performed'].extend([
                'duplicate_detection',
                'dog_day_rule_check',
                'orphaned_records_check',
                'data_quality_check'
            ])
            
        except sqlite3.Error as e:
            report['issues_found'].append(f"Error during integrity check: {e}")
            
        self.logger.info(f"Integrity check completed. Found {len(report['issues_found'])} issues")
        return report
    
    def find_duplicate_records(self, table_name: str, unique_fields: List[str]) -> List[Dict]:
        """Find duplicate records based on unique constraint fields"""
        conn = self.connect()
        cursor = conn.cursor()
        
        duplicates = []
        
        try:
            # Build query to find duplicates
            field_list = ', '.join(unique_fields)
            query = f"""
                SELECT {field_list}, COUNT(*) as dup_count
                FROM {table_name}
                GROUP BY {field_list}
                HAVING COUNT(*) > 1
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            for row in results:
                duplicate_info = {}
                for i, field in enumerate(unique_fields):
                    duplicate_info[field] = row[i]
                duplicate_info['count'] = row[-1]
                duplicates.append(duplicate_info)
                
        except sqlite3.Error as e:
            self.logger.error(f"Error finding duplicates in {table_name}: {e}")
            
        return duplicates
    
    def find_dog_day_violations(self) -> List[Dict]:
        """Find violations of the one-race-per-dog-per-day rule"""
        conn = self.connect()
        cursor = conn.cursor()
        
        violations = []
        
        try:
            query = """
                SELECT dog_clean_name, race_date, COUNT(*) as race_count, 
                       GROUP_CONCAT(race_id) as race_ids
                FROM enhanced_expert_data
                GROUP BY dog_clean_name, race_date
                HAVING COUNT(*) > 1
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            for row in results:
                violations.append({
                    'dog_name': row[0],
                    'race_date': row[1],
                    'race_count': row[2],
                    'race_ids': row[3].split(',') if row[3] else []
                })
                
        except sqlite3.Error as e:
            self.logger.error(f"Error finding dog-day violations: {e}")
            
        return violations
    
    def find_orphaned_records(self) -> Dict:
        """Find orphaned records (records without valid foreign key references)"""
        conn = self.connect()
        cursor = conn.cursor()
        
        orphaned = {}
        
        try:
            # Check for dog_race_data records without corresponding race_metadata
            cursor.execute("""
                SELECT COUNT(*)
                FROM dog_race_data d
                LEFT JOIN race_metadata r ON d.race_id = r.race_id
                WHERE r.race_id IS NULL
            """)
            orphaned_dog_data = cursor.fetchone()[0]
            
            if orphaned_dog_data > 0:
                orphaned['dog_race_data_without_race'] = orphaned_dog_data
                
        except sqlite3.Error as e:
            self.logger.error(f"Error finding orphaned records: {e}")
            
        return orphaned
    
    def check_data_quality(self) -> List[str]:
        """Check for data quality issues"""
        issues = []
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Check for records with missing critical data
            cursor.execute("""
                SELECT COUNT(*) FROM race_metadata 
                WHERE race_id IS NULL OR race_id = '' OR venue IS NULL OR venue = ''
            """)
            missing_critical = cursor.fetchone()[0]
            
            if missing_critical > 0:
                issues.append(f"Found {missing_critical} race records with missing critical data")
            
            # Check for invalid box numbers
            cursor.execute("""
                SELECT COUNT(*) FROM dog_race_data 
                WHERE box_number < 1 OR box_number > 8
            """)
            invalid_boxes = cursor.fetchone()[0]
            
            if invalid_boxes > 0:
                issues.append(f"Found {invalid_boxes} records with invalid box numbers")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error checking data quality: {e}")
            
        return issues
    
    def create_backup(self, backup_suffix: str = None) -> str:
        """Create a backup of the database before any operations"""
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"greyhound_racing_data_backup_{backup_suffix}.db"
        
        try:
            # Use SQLite backup API for consistent backup
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(str(backup_path))
            
            source_conn.backup(backup_conn)
            
            source_conn.close()
            backup_conn.close()
            
            self.logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def generate_integrity_report(self, output_path: str = None) -> str:
        """Generate a comprehensive integrity report"""
        if output_path is None:
            output_path = f"reports/integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        os.makedirs("reports", exist_ok=True)
        
        report = self.run_comprehensive_integrity_check()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Integrity report generated: {output_path}")
        return output_path

def main():
    """Main function for running integrity checks"""
    print("=== Greyhound Racing Data Integrity System ===\n")
    
    integrity_manager = DataIntegrityManager()
    
    try:
        with integrity_manager:
            # Add database constraints
            print("1. Adding database constraints...")
            integrity_manager.add_database_constraints()
            print("✓ Database constraints added\n")
            
            # Create backup
            print("2. Creating database backup...")
            backup_path = integrity_manager.create_backup()
            print(f"✓ Backup created: {backup_path}\n")
            
            # Run comprehensive integrity check
            print("3. Running comprehensive integrity check...")
            report_path = integrity_manager.generate_integrity_report()
            print(f"✓ Integrity report generated: {report_path}\n")
            
            # Run integrity check and display summary
            report = integrity_manager.run_comprehensive_integrity_check()
            
            print("=== INTEGRITY CHECK SUMMARY ===")
            print(f"Timestamp: {report['timestamp']}")
            print(f"Checks performed: {len(report['checks_performed'])}")
            print(f"Issues found: {len(report['issues_found'])}")
            
            if report['issues_found']:
                print("\n⚠️  ISSUES DETECTED:")
                for issue in report['issues_found']:
                    print(f"  - {issue}")
                    
                print("\n💡 RECOMMENDATIONS:")
                for rec in report['recommendations']:
                    print(f"  - {rec}")
            else:
                print("\n✅ No integrity issues detected!")
                
            print(f"\n📊 STATISTICS:")
            for key, value in report['statistics'].items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error during integrity check: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
