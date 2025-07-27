#!/usr/bin/env python3
"""
Data Integrity and Efficiency Check Script
=========================================
This script performs a comprehensive analysis of the greyhound racing data:
1. Identifies duplicate files and entries
2. Validates file integrity (CSV/JSON format)
3. Checks for corrupted data
4. Analyzes data usage efficiency
5. Provides recommendations for cleanup
"""

import os
import json
import csv
import hashlib
import sqlite3
import pandas as pd
from collections import defaultdict, Counter
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegrityChecker:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.duplicate_files = []
        self.corrupted_files = []
        self.empty_files = []
        self.file_hashes = {}
        self.db_duplicates = []
        self.stats = {
            'total_csv_files': 0,
            'total_json_files': 0,
            'duplicate_files': 0,
            'corrupted_files': 0,
            'empty_files': 0,
            'database_duplicates': 0,
            'storage_waste': 0
        }
    
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def find_duplicate_files(self):
        """Find files with identical content"""
        logger.info("Scanning for duplicate files...")
        hash_to_files = defaultdict(list)
        
        # Process CSV files
        csv_files = list(self.base_path.rglob("*.csv"))
        self.stats['total_csv_files'] = len(csv_files)
        
        for csv_file in csv_files:
            if csv_file.stat().st_size == 0:
                self.empty_files.append(str(csv_file))
                continue
                
            file_hash = self.calculate_file_hash(csv_file)
            if file_hash:
                hash_to_files[file_hash].append(str(csv_file))
        
        # Process JSON files
        json_files = list(self.base_path.rglob("*.json"))
        self.stats['total_json_files'] = len(json_files)
        
        for json_file in json_files:
            if json_file.stat().st_size == 0:
                self.empty_files.append(str(json_file))
                continue
                
            file_hash = self.calculate_file_hash(json_file)
            if file_hash:
                hash_to_files[file_hash].append(str(json_file))
        
        # Identify duplicates
        for file_hash, files in hash_to_files.items():
            if len(files) > 1:
                self.duplicate_files.extend(files[1:])  # Keep first, mark others as duplicates
                total_size = sum(os.path.getsize(f) for f in files[1:])
                self.stats['storage_waste'] += total_size
        
        self.stats['duplicate_files'] = len(self.duplicate_files)
        self.stats['empty_files'] = len(self.empty_files)
    
    def check_csv_integrity(self, file_path):
        """Check if CSV file is well-formed"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read with pandas first (more robust)
                df = pd.read_csv(f)
                if df.empty:
                    return False, "Empty DataFrame"
                return True, None
        except pd.errors.EmptyDataError:
            return False, "Empty CSV file"
        except pd.errors.ParserError as e:
            return False, f"Parser error: {e}"
        except UnicodeDecodeError as e:
            # Try different encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    df = pd.read_csv(f)
                return True, "Used latin-1 encoding"
            except Exception:
                return False, f"Unicode error: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"
    
    def check_json_integrity(self, file_path):
        """Check if JSON file is well-formed"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    return False, "Empty JSON"
                return True, None
        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {e}"
        except UnicodeDecodeError as e:
            return False, f"Unicode error: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"
    
    def validate_file_integrity(self):
        """Validate integrity of all CSV and JSON files"""
        logger.info("Validating file integrity...")
        
        # Check CSV files
        for csv_file in self.base_path.rglob("*.csv"):
            is_valid, error = self.check_csv_integrity(csv_file)
            if not is_valid:
                self.corrupted_files.append({
                    'file': str(csv_file),
                    'type': 'csv',
                    'error': error
                })
        
        # Check JSON files
        for json_file in self.base_path.rglob("*.json"):
            is_valid, error = self.check_json_integrity(json_file)
            if not is_valid:
                self.corrupted_files.append({
                    'file': str(json_file),
                    'type': 'json',
                    'error': error
                })
        
        self.stats['corrupted_files'] = len(self.corrupted_files)
    
    def check_database_duplicates(self):
        """Check for duplicate entries in the database"""
        logger.info("Checking database for duplicates...")
        db_path = self.base_path / "race_data.db"
        
        if not db_path.exists():
            logger.warning("Database file not found")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for duplicate races
            cursor.execute("""
                SELECT race_name, race_date, venue, COUNT(*) as count
                FROM races 
                GROUP BY race_name, race_date, venue 
                HAVING COUNT(*) > 1
            """)
            
            race_duplicates = cursor.fetchall()
            
            # Check for duplicate dogs
            cursor.execute("""
                SELECT dog_name, race_id, COUNT(*) as count
                FROM race_entries 
                GROUP BY dog_name, race_id 
                HAVING COUNT(*) > 1
            """)
            
            dog_duplicates = cursor.fetchall()
            
            self.db_duplicates = {
                'race_duplicates': race_duplicates,
                'dog_duplicates': dog_duplicates
            }
            
            self.stats['database_duplicates'] = len(race_duplicates) + len(dog_duplicates)
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking database: {e}")
    
    def analyze_filename_patterns(self):
        """Analyze filename patterns to identify potential systematic duplicates"""
        logger.info("Analyzing filename patterns...")
        
        # Look for files with numbered suffixes (_1, _2, etc.)
        numbered_files = defaultdict(list)
        
        for csv_file in self.base_path.rglob("*.csv"):
            filename = csv_file.name
            # Check if filename ends with _number.csv
            if '_' in filename:
                base_name = filename.rsplit('_', 1)[0]
                suffix = filename.rsplit('_', 1)[1]
                if suffix.replace('.csv', '').isdigit():
                    numbered_files[base_name].append(str(csv_file))
        
        # Files with the same base name but different numbers
        systematic_duplicates = {k: v for k, v in numbered_files.items() if len(v) > 1}
        
        return systematic_duplicates
    
    def generate_report(self):
        """Generate comprehensive integrity report"""
        logger.info("Generating integrity report...")
        
        report = {
            'summary': self.stats,
            'duplicate_files': self.duplicate_files,
            'corrupted_files': self.corrupted_files,
            'empty_files': self.empty_files,
            'database_duplicates': self.db_duplicates,
            'systematic_duplicates': self.analyze_filename_patterns()
        }
        
        # Calculate storage efficiency
        total_size = sum(
            os.path.getsize(f) 
            for f in self.base_path.rglob("*") 
            if f.is_file()
        )
        
        efficiency_ratio = (total_size - self.stats['storage_waste']) / total_size * 100
        
        report['storage_analysis'] = {
            'total_storage_bytes': total_size,
            'wasted_storage_bytes': self.stats['storage_waste'],
            'efficiency_percentage': round(efficiency_ratio, 2)
        }
        
        return report
    
    def run_full_check(self):
        """Run all integrity checks"""
        logger.info("Starting comprehensive data integrity check...")
        
        self.find_duplicate_files()
        self.validate_file_integrity()
        self.check_database_duplicates()
        
        report = self.generate_report()
        
        # Save report
        report_path = self.base_path / "data_integrity_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Integrity check complete. Report saved to {report_path}")
        return report

def print_report_summary(report):
    """Print a summary of the integrity report"""
    print("\n" + "="*60)
    print("DATA INTEGRITY REPORT SUMMARY")
    print("="*60)
    
    stats = report['summary']
    print(f"Total CSV files: {stats['total_csv_files']:,}")
    print(f"Total JSON files: {stats['total_json_files']:,}")
    print(f"Duplicate files found: {stats['duplicate_files']:,}")
    print(f"Corrupted files found: {stats['corrupted_files']:,}")
    print(f"Empty files found: {stats['empty_files']:,}")
    print(f"Database duplicates: {stats['database_duplicates']:,}")
    
    storage = report['storage_analysis']
    print(f"\nStorage Analysis:")
    print(f"Total storage: {storage['total_storage_bytes'] / (1024*1024):.2f} MB")
    print(f"Wasted storage: {storage['wasted_storage_bytes'] / (1024*1024):.2f} MB")
    print(f"Storage efficiency: {storage['efficiency_percentage']:.2f}%")
    
    if report['corrupted_files']:
        print(f"\nCorrupted Files (first 5):")
        for corrupt in report['corrupted_files'][:5]:
            print(f"  - {corrupt['file']}: {corrupt['error']}")
    
    if report['systematic_duplicates']:
        print(f"\nSystematic Duplicates (files with numbered suffixes):")
        count = 0
        for base_name, files in report['systematic_duplicates'].items():
            if count < 5:  # Show first 5
                print(f"  - {base_name}: {len(files)} variants")
                count += 1
        if len(report['systematic_duplicates']) > 5:
            print(f"  ... and {len(report['systematic_duplicates']) - 5} more")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run the integrity check
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    checker = DataIntegrityChecker(base_path)
    
    try:
        report = checker.run_full_check()
        print_report_summary(report)
        
        print(f"\nFull detailed report saved to: {base_path}/data_integrity_report.json")
        
    except KeyboardInterrupt:
        print("\nIntegrity check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during integrity check: {e}")
        sys.exit(1)
