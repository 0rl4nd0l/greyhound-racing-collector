#!/usr/bin/env python3
"""
Comprehensive Data Integrity and Duplication Check System
Performs thorough verification of the greyhound racing data system.
"""

import os
import sys
import json
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
import logging
from pathlib import Path
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_integrity_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIntegrityChecker:
    def __init__(self, base_path="/Users/orlandolee/greyhound_racing_collector"):
        self.base_path = base_path
        self.issues = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def log_issue(self, issue_type, message, severity="ERROR"):
        """Log an issue found during checking"""
        entry = {
            'type': issue_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        if severity == "ERROR":
            self.issues.append(entry)
            logger.error(f"{issue_type}: {message}")
        else:
            self.warnings.append(entry)
            logger.warning(f"{issue_type}: {message}")

    def check_file_integrity(self):
        """Check file integrity across all data directories"""
        logger.info("🔍 Checking file integrity...")
        
        # Define key directories to check
        key_dirs = [
            'processed/completed',
            'processed/excluded', 
            'processed/other',
            'unprocessed',
            'form_guides',
            'form_guides/downloaded',
            'consolidated_data',
            'databases',
            'predictions'
        ]
        
        for dir_name in key_dirs:
            dir_path = os.path.join(self.base_path, dir_name)
            if not os.path.exists(dir_path):
                self.log_issue("MISSING_DIRECTORY", f"Directory {dir_name} does not exist")
                continue
                
            # Count files by type
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            db_files = glob.glob(os.path.join(dir_path, "*.db"))
            
            self.stats[f"{dir_name}_csv_count"] = len(csv_files)
            self.stats[f"{dir_name}_json_count"] = len(json_files)
            self.stats[f"{dir_name}_db_count"] = len(db_files)
            
            # Check for empty or corrupted files
            for file_path in csv_files + json_files:
                try:
                    if os.path.getsize(file_path) == 0:
                        self.log_issue("EMPTY_FILE", f"File is empty: {file_path}")
                        continue
                        
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        if df.empty:
                            self.log_issue("EMPTY_DATA", f"CSV has no data: {file_path}")
                        elif df.shape[0] < 2:  # Header + at least 1 row
                            self.log_issue("INSUFFICIENT_DATA", f"CSV has insufficient data: {file_path}", "WARNING")
                            
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if not data:
                                self.log_issue("EMPTY_JSON", f"JSON is empty: {file_path}")
                                
                except Exception as e:
                    self.log_issue("CORRUPTED_FILE", f"Cannot read {file_path}: {str(e)}")

    def check_database_integrity(self):
        """Check database integrity and consistency"""
        logger.info("🗃️ Checking database integrity...")
        
        db_path = os.path.join(self.base_path, "greyhound_racing_data.db")
        if not os.path.exists(db_path):
            self.log_issue("MISSING_DATABASE", "Main database file not found")
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['races', 'dogs', 'predictions']
            for table in expected_tables:
                if table not in tables:
                    self.log_issue("MISSING_TABLE", f"Expected table '{table}' not found in database")
                else:
                    # Check row counts
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.stats[f"db_{table}_count"] = count
                    logger.info(f"Database table '{table}': {count} records")
                    
                    if count == 0:
                        self.log_issue("EMPTY_TABLE", f"Table '{table}' is empty", "WARNING")
            
            # Check for NULL values in critical fields
            if 'races' in tables:
                cursor.execute("SELECT COUNT(*) FROM races WHERE race_name IS NULL OR venue IS NULL OR race_date IS NULL")
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    self.log_issue("NULL_CRITICAL_DATA", f"Found {null_count} races with NULL critical fields")
            
            conn.close()
            
        except Exception as e:
            self.log_issue("DATABASE_ERROR", f"Database error: {str(e)}")

    def check_duplication_logic(self):
        """Test and verify deduplication logic"""
        logger.info("🔄 Checking deduplication logic...")
        
        # Check for duplicate files
        file_hashes = defaultdict(list)
        
        # Scan all CSV files in processed directories
        for root, dirs, files in os.walk(os.path.join(self.base_path, "processed")):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            file_hashes[file_hash].append(file_path)
                    except Exception as e:
                        self.log_issue("HASH_ERROR", f"Cannot hash file {file_path}: {str(e)}")
        
        # Report duplicates
        for hash_val, files in file_hashes.items():
            if len(files) > 1:
                self.log_issue("DUPLICATE_FILES", f"Identical files found: {files}")
        
        # Check for duplicate race records within CSV files
        self.check_csv_duplicates()

    def check_csv_duplicates(self):
        """Check for duplicate records within CSV files"""
        logger.info("📊 Checking CSV file duplicates...")
        
        processed_dirs = [
            'processed/completed',
            'processed/excluded',
            'processed/other'
        ]
        
        for dir_name in processed_dirs:
            dir_path = os.path.join(self.base_path, dir_name)
            if not os.path.exists(dir_path):
                continue
                
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Check for duplicate rows
                    duplicates = df.duplicated()
                    dup_count = duplicates.sum()
                    
                    if dup_count > 0:
                        self.log_issue("CSV_DUPLICATES", f"Found {dup_count} duplicate rows in {csv_file}")
                    
                    # Check for duplicate dogs in same race
                    if 'Dog' in df.columns:
                        dog_duplicates = df['Dog'].duplicated()
                        dog_dup_count = dog_duplicates.sum()
                        
                        if dog_dup_count > 0:
                            self.log_issue("DUPLICATE_DOGS", f"Found {dog_dup_count} duplicate dogs in {csv_file}")
                    
                    self.stats[f"records_in_{os.path.basename(csv_file)}"] = len(df)
                    
                except Exception as e:
                    self.log_issue("CSV_READ_ERROR", f"Cannot read CSV {csv_file}: {str(e)}")

    def check_data_consistency(self):
        """Check data consistency across different sources"""
        logger.info("⚖️ Checking data consistency...")
        
        # Check venue name consistency
        venue_variations = defaultdict(set)
        
        processed_dirs = ['processed/completed', 'processed/excluded', 'processed/other']
        
        for dir_name in processed_dirs:
            dir_path = os.path.join(self.base_path, dir_name)
            if not os.path.exists(dir_path):
                continue
                
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            
            for csv_file in csv_files:
                try:
                    # Extract venue from filename
                    filename = os.path.basename(csv_file)
                    if ' - ' in filename:
                        parts = filename.split(' - ')
                        if len(parts) >= 2:
                            venue_from_filename = parts[1]
                            venue_variations[venue_from_filename].add(csv_file)
                    
                    # Check data format consistency
                    df = pd.read_csv(csv_file)
                    
                    # Check for required columns
                    expected_columns = ['Dog', 'Box', 'Trainer', 'Weight']
                    missing_columns = [col for col in expected_columns if col not in df.columns]
                    
                    if missing_columns:
                        self.log_issue("MISSING_COLUMNS", 
                                     f"Missing columns {missing_columns} in {csv_file}", "WARNING")
                    
                    # Check for data type consistency
                    if 'Box' in df.columns:
                        non_numeric_boxes = df[pd.to_numeric(df['Box'], errors='coerce').isna()]
                        if not non_numeric_boxes.empty:
                            self.log_issue("INVALID_BOX_NUMBERS", 
                                         f"Non-numeric box numbers in {csv_file}")
                    
                except Exception as e:
                    self.log_issue("CONSISTENCY_CHECK_ERROR", f"Error checking {csv_file}: {str(e)}")

    def check_backup_integrity(self):
        """Check backup system integrity"""
        logger.info("💾 Checking backup integrity...")
        
        backup_dirs = [
            'cached_backup',
            'backup_before_cleanup',
            'predictions'
        ]
        
        for backup_dir in backup_dirs:
            backup_path = os.path.join(self.base_path, backup_dir)
            if os.path.exists(backup_path):
                file_count = len(glob.glob(os.path.join(backup_path, "**", "*.*"), recursive=True))
                self.stats[f"backup_{backup_dir}_files"] = file_count
                logger.info(f"Backup directory '{backup_dir}': {file_count} files")
            else:
                self.log_issue("MISSING_BACKUP_DIR", f"Backup directory '{backup_dir}' not found", "WARNING")

    def check_prediction_integrity(self):
        """Check prediction system integrity"""
        logger.info("🎯 Checking prediction integrity...")
        
        predictions_dir = os.path.join(self.base_path, "predictions")
        if not os.path.exists(predictions_dir):
            self.log_issue("MISSING_PREDICTIONS_DIR", "Predictions directory not found")
            return
        
        json_files = glob.glob(os.path.join(predictions_dir, "*.json"))
        backup_files = glob.glob(os.path.join(predictions_dir, "*.backup*"))
        
        self.stats["prediction_files"] = len(json_files)
        self.stats["prediction_backups"] = len(backup_files)
        
        # Check prediction file integrity
        for json_file in json_files[:10]:  # Check first 10 to avoid overload
            try:
                with open(json_file, 'r') as f:
                    prediction_data = json.load(f)
                    
                    # Check required fields
                    required_fields = ['race_info', 'predictions', 'model_info']
                    missing_fields = [field for field in required_fields if field not in prediction_data]
                    
                    if missing_fields:
                        self.log_issue("INCOMPLETE_PREDICTION", 
                                     f"Missing fields {missing_fields} in {json_file}", "WARNING")
                    
                    # Check prediction probabilities sum to ~1.0
                    if 'predictions' in prediction_data:
                        for dog_pred in prediction_data['predictions']:
                            if 'probability' in dog_pred:
                                prob = dog_pred['probability']
                                if not (0 <= prob <= 1):
                                    self.log_issue("INVALID_PROBABILITY", 
                                                 f"Invalid probability {prob} in {json_file}")
                    
            except Exception as e:
                self.log_issue("PREDICTION_READ_ERROR", f"Cannot read prediction {json_file}: {str(e)}")

    def run_unit_tests(self):
        """Run available unit tests"""
        logger.info("🧪 Running unit tests...")
        
        test_files = [
            'test_comprehensive_csv_loading.py',
            'test_csv_loading.py', 
            'test_duplicate_detection.py'
        ]
        
        for test_file in test_files:
            test_path = os.path.join(self.base_path, test_file)
            if os.path.exists(test_path):
                try:
                    # Import and run test
                    import subprocess
                    result = subprocess.run([sys.executable, test_path], 
                                          capture_output=True, text=True, 
                                          cwd=self.base_path)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Test {test_file} passed")
                        self.stats[f"test_{test_file}_status"] = "PASSED"
                    else:
                        self.log_issue("TEST_FAILURE", f"Test {test_file} failed: {result.stderr}")
                        self.stats[f"test_{test_file}_status"] = "FAILED"
                        
                except Exception as e:
                    self.log_issue("TEST_ERROR", f"Error running test {test_file}: {str(e)}")
            else:
                self.log_issue("MISSING_TEST", f"Test file {test_file} not found", "WARNING")

    def manual_sampling_check(self):
        """Perform manual sampling verification"""
        logger.info("🔍 Performing manual sampling check...")
        
        # Sample some files for manual verification
        processed_files = glob.glob(os.path.join(self.base_path, "processed", "completed", "*.csv"))
        
        if len(processed_files) >= 5:
            sample_files = processed_files[:5]  # Take first 5 files
            
            for file_path in sample_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Basic checks
                    filename = os.path.basename(file_path)
                    logger.info(f"📋 Sampling {filename}:")
                    logger.info(f"  - Rows: {len(df)}")
                    logger.info(f"  - Columns: {list(df.columns)}")
                    
                    if 'Dog' in df.columns:
                        unique_dogs = df['Dog'].nunique()
                        total_dogs = len(df)
                        logger.info(f"  - Dogs: {unique_dogs} unique out of {total_dogs} total")
                        
                        if unique_dogs != total_dogs:
                            self.log_issue("SAMPLE_DUPLICATE_DOGS", 
                                         f"Duplicate dogs found in sample {filename}", "WARNING")
                    
                    # Check for NaN values
                    nan_counts = df.isnull().sum()
                    if nan_counts.sum() > 0:
                        logger.info(f"  - NaN values: {nan_counts.to_dict()}")
                        
                except Exception as e:
                    self.log_issue("SAMPLE_ERROR", f"Error sampling {file_path}: {str(e)}")

    def generate_report(self):
        """Generate comprehensive integrity report"""
        logger.info("📊 Generating integrity report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'check_status': 'PASSED' if len(self.issues) == 0 else 'FAILED'
            },
            'statistics': dict(self.stats),
            'issues': self.issues,
            'warnings': self.warnings
        }
        
        # Save detailed report
        report_path = os.path.join(self.base_path, f"data_integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 DATA INTEGRITY CHECK SUMMARY")
        print("="*80)
        print(f"Total Issues Found: {len(self.issues)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print(f"Overall Status: {report['summary']['check_status']}")
        print(f"Report saved to: {report_path}")
        
        # Print key statistics
        print("\n📈 KEY STATISTICS:")
        for key, value in self.stats.items():
            if 'count' in key or 'files' in key:
                print(f"  {key}: {value}")
        
        # Print most critical issues
        if self.issues:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in self.issues[:10]:  # Show top 10
                print(f"  - {issue['type']}: {issue['message']}")
                
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings[:5]:  # Show top 5
                print(f"  - {warning['type']}: {warning['message']}")
        
        print("\n" + "="*80)
        
        return report

    def run_all_checks(self):
        """Run all integrity checks"""
        logger.info("🚀 Starting comprehensive data integrity check...")
        
        try:
            self.check_file_integrity()
            self.check_database_integrity()
            self.check_duplication_logic()
            self.check_data_consistency()
            self.check_backup_integrity()
            self.check_prediction_integrity()
            self.run_unit_tests()
            self.manual_sampling_check()
            
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"Critical error during integrity check: {str(e)}")
            self.log_issue("CRITICAL_ERROR", f"Integrity check failed: {str(e)}")
            return self.generate_report()

def main():
    """Main function to run the integrity check"""
    print("🔍 Starting Comprehensive Data Integrity Check...")
    
    checker = DataIntegrityChecker()
    report = checker.run_all_checks()
    
    # Return appropriate exit code
    if report['summary']['check_status'] == 'PASSED':
        print("✅ All checks passed!")
        return 0
    else:
        print("❌ Issues found - see report for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
