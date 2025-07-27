#!/usr/bin/env python3
"""
Comprehensive Data Quality Analyzer for Greyhound Racing Data

This script analyzes both processed CSV files and the database to identify:
1. Duplicate entries and box conflicts
2. Missing or incomplete data
3. Data consistency issues
4. Database vs CSV file discrepancies
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime
import json
import numpy as np
from collections import defaultdict

class DataQualityAnalyzer:
    def __init__(self, processed_dir, database_path):
        self.processed_dir = processed_dir
        self.database_path = database_path
        self.issues_found = []
        
    def log_issue(self, category, description, severity="WARNING"):
        """Log an issue found during analysis"""
        self.issues_found.append({
            'category': category,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[{severity}] {category}: {description}")
    
    def analyze_csv_files(self):
        """Analyze processed CSV files for quality issues"""
        print("=== ANALYZING PROCESSED CSV FILES ===")
        
        csv_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files to analyze")
        
        file_issues = []
        total_races = 0
        total_dogs = 0
        
        for filename in csv_files:
            file_path = os.path.join(self.processed_dir, filename)
            
            try:
                # Read CSV with pipe separator
                df = pd.read_csv(file_path, sep='|', header=0)
                
                # Basic file info
                total_races += 1
                dogs_in_race = len(df)
                total_dogs += dogs_in_race
                
                # Check for duplicate box numbers
                if 'BOX' in df.columns:
                    box_counts = df['BOX'].value_counts()
                    duplicates = box_counts[box_counts > 1]
                    if len(duplicates) > 0:
                        self.log_issue('DUPLICATE_BOXES', 
                                     f"File {filename}: Duplicate box numbers {duplicates.index.tolist()}")
                
                # Check for missing dog names
                if 'Dog Name' in df.columns:
                    missing_names = df['Dog Name'].isna().sum()
                    if missing_names > 0:
                        self.log_issue('MISSING_NAMES',
                                     f"File {filename}: {missing_names} missing dog names")
                
                # Check for missing essential data
                essential_cols = ['Dog Name', 'BOX', 'PLC']
                for col in essential_cols:
                    if col in df.columns:
                        missing_count = df[col].isna().sum()
                        if missing_count > 0:
                            self.log_issue('MISSING_DATA',
                                         f"File {filename}: {missing_count} missing {col} values")
                
                # Check if race has results (PLC column should have finishing positions)
                if 'PLC' in df.columns:
                    valid_positions = df['PLC'].dropna().astype(str).str.match(r'^\d+$').sum()
                    if valid_positions == 0:
                        self.log_issue('NO_RESULTS',
                                     f"File {filename}: No valid finishing positions found")
                
                # Check for unusual number of dogs
                if dogs_in_race < 4 or dogs_in_race > 12:
                    self.log_issue('UNUSUAL_FIELD_SIZE',
                                 f"File {filename}: Unusual field size of {dogs_in_race} dogs")
                
            except Exception as e:
                self.log_issue('FILE_ERROR', f"Error reading {filename}: {str(e)}", "ERROR")
        
        print(f"\nCSV Analysis Summary:")
        print(f"Total races analyzed: {total_races}")
        print(f"Total dog entries: {total_dogs}")
        print(f"Average dogs per race: {total_dogs/total_races:.1f}")
        
        return total_races, total_dogs
    
    def analyze_database(self):
        """Analyze database for quality issues"""
        print("\n=== ANALYZING DATABASE ===")
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Database tables: {tables}")
            
            if 'race_metadata' not in tables or 'dog_race_data' not in tables:
                self.log_issue('DATABASE_STRUCTURE', 
                             "Missing required tables (race_metadata or dog_race_data)", "ERROR")
                return
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            total_races = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            total_dogs = cursor.fetchone()[0]
            
            print(f"Database contains {total_races} races and {total_dogs} dog entries")
            
            # Check for duplicate box assignments
            cursor.execute("""
                SELECT race_id, box_number, COUNT(*) as count
                FROM dog_race_data 
                GROUP BY race_id, box_number 
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 10
            """)
            duplicate_boxes = cursor.fetchall()
            
            if duplicate_boxes:
                total_duplicates = len(duplicate_boxes)
                self.log_issue('DATABASE_DUPLICATES',
                             f"Found {total_duplicates} races with duplicate box assignments")
                
                # Show worst cases
                for race_id, box_num, count in duplicate_boxes[:5]:
                    self.log_issue('DUPLICATE_DETAIL',
                                 f"Race {race_id}, Box {box_num}: {count} dogs assigned")
            
            # Check for missing essential data in race_metadata
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN venue IS NULL OR venue = '' THEN 1 ELSE 0 END) as missing_venue,
                    SUM(CASE WHEN race_date IS NULL OR race_date = '' THEN 1 ELSE 0 END) as missing_date,
                    SUM(CASE WHEN grade IS NULL OR grade = '' THEN 1 ELSE 0 END) as missing_grade,
                    SUM(CASE WHEN winner_name IS NULL OR winner_name = '' THEN 1 ELSE 0 END) as missing_winner
                FROM race_metadata
            """)
            
            race_quality = cursor.fetchone()
            if race_quality:
                total, missing_venue, missing_date, missing_grade, missing_winner = race_quality
                
                if missing_venue > 0:
                    self.log_issue('MISSING_VENUE', f"{missing_venue}/{total} races missing venue")
                if missing_date > 0:
                    self.log_issue('MISSING_DATE', f"{missing_date}/{total} races missing date")
                if missing_grade > 0:
                    self.log_issue('MISSING_GRADE', f"{missing_grade}/{total} races missing grade")
                if missing_winner > 0:
                    self.log_issue('MISSING_WINNER', f"{missing_winner}/{total} races missing winner")
            
            # Check for missing essential data in dog_race_data
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN dog_name IS NULL OR dog_name = '' OR dog_name = 'nan' THEN 1 ELSE 0 END) as missing_name,
                    SUM(CASE WHEN box_number IS NULL THEN 1 ELSE 0 END) as missing_box,
                    SUM(CASE WHEN finish_position IS NULL OR finish_position = '' THEN 1 ELSE 0 END) as missing_position
                FROM dog_race_data
            """)
            
            dog_quality = cursor.fetchone()
            if dog_quality:
                total, missing_name, missing_box, missing_position = dog_quality
                
                if missing_name > 0:
                    self.log_issue('MISSING_DOG_NAME', f"{missing_name}/{total} dogs missing names")
                if missing_box > 0:
                    self.log_issue('MISSING_BOX', f"{missing_box}/{total} dogs missing box numbers")
                if missing_position > 0:
                    self.log_issue('MISSING_POSITION', f"{missing_position}/{total} dogs missing finishing positions")
            
            # Check for races with unusual field sizes
            cursor.execute("""
                SELECT race_id, COUNT(*) as dog_count
                FROM dog_race_data
                GROUP BY race_id
                HAVING dog_count > 20 OR dog_count < 4
                ORDER BY dog_count DESC
                LIMIT 10
            """)
            
            unusual_races = cursor.fetchall()
            for race_id, count in unusual_races:
                self.log_issue('UNUSUAL_DB_FIELD_SIZE',
                             f"Race {race_id}: {count} dogs in database")
            
            # Check for data consistency issues
            cursor.execute("""
                SELECT race_id, COUNT(DISTINCT finish_position) as unique_positions, COUNT(*) as total_dogs
                FROM dog_race_data
                WHERE finish_position IS NOT NULL AND finish_position != ''
                GROUP BY race_id
                HAVING unique_positions != total_dogs AND total_dogs > 0
                LIMIT 10
            """)
            
            position_conflicts = cursor.fetchall()
            for race_id, unique_pos, total_dogs in position_conflicts:
                self.log_issue('POSITION_CONFLICT',
                             f"Race {race_id}: {unique_pos} unique positions for {total_dogs} dogs")
            
            conn.close()
            
        except Exception as e:
            self.log_issue('DATABASE_ERROR', f"Error analyzing database: {str(e)}", "ERROR")
    
    def cross_reference_data(self):
        """Cross-reference CSV files with database data"""
        print("\n=== CROSS-REFERENCING CSV FILES AND DATABASE ===")
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get all race IDs from database
            cursor.execute("SELECT DISTINCT race_id FROM race_metadata")
            db_races = set(row[0] for row in cursor.fetchall())
            
            # Get all CSV files and extract race info
            csv_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
            csv_race_info = []
            
            for filename in csv_files:
                # Try to extract race info from filename
                # Expected format: "Race X - VENUE - DATE.csv"
                parts = filename.replace('.csv', '').split(' - ')
                if len(parts) >= 3:
                    race_num = parts[0].replace('Race ', '')
                    venue = parts[1]
                    date_str = parts[2]
                    
                    # Try to parse date and create race_id
                    try:
                        date_obj = datetime.strptime(date_str, '%d %B %Y')
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                        race_id = f"{venue.lower()}_{formatted_date}_{race_num}"
                        csv_race_info.append((filename, race_id, venue, formatted_date))
                    except ValueError:
                        self.log_issue('FILENAME_PARSING', f"Could not parse date from {filename}")
            
            csv_race_ids = set(info[1] for info in csv_race_info)
            
            # Find missing races
            csv_only = csv_race_ids - db_races
            db_only = db_races - csv_race_ids
            
            if csv_only:
                self.log_issue('MISSING_FROM_DB', 
                             f"{len(csv_only)} races in CSV but not in database")
                for race_id in list(csv_only)[:5]:  # Show first 5
                    self.log_issue('MISSING_RACE_DETAIL', f"CSV race not in DB: {race_id}")
            
            if db_only:
                self.log_issue('MISSING_FROM_CSV', 
                             f"{len(db_only)} races in database but no corresponding CSV")
                for race_id in list(db_only)[:5]:  # Show first 5
                    self.log_issue('EXTRA_RACE_DETAIL', f"DB race not in CSV: {race_id}")
            
            conn.close()
            
        except Exception as e:
            self.log_issue('CROSS_REF_ERROR', f"Error in cross-reference: {str(e)}", "ERROR")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== SUMMARY REPORT ===")
        
        # Group issues by category
        issue_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for issue in self.issues_found:
            issue_counts[issue['category']] += 1
            severity_counts[issue['severity']] += 1
        
        print(f"Total issues found: {len(self.issues_found)}")
        print(f"Issues by severity: {dict(severity_counts)}")
        print(f"Issues by category: {dict(issue_counts)}")
        
        # Critical issues that need immediate attention
        critical_categories = ['DATABASE_DUPLICATES', 'DUPLICATE_BOXES', 'MISSING_DATA', 'NO_RESULTS']
        critical_issues = [issue for issue in self.issues_found 
                          if issue['category'] in critical_categories]
        
        if critical_issues:
            print(f"\n{len(critical_issues)} CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for issue in critical_issues:
                print(f"  - {issue['description']}")
        
        # Save detailed report
        report_path = '/Users/orlandolee/greyhound_racing_collector/data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_issues': len(self.issues_found),
                    'severity_counts': dict(severity_counts),
                    'category_counts': dict(issue_counts)
                },
                'issues': self.issues_found
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        return len(critical_issues) == 0  # Return True if no critical issues

def main():
    """Main execution function"""
    processed_dir = '/Users/orlandolee/greyhound_racing_collector/form_guides/processed'
    database_path = '/Users/orlandolee/greyhound_racing_collector/databases/comprehensive_greyhound_data.db'
    
    analyzer = DataQualityAnalyzer(processed_dir, database_path)
    
    print("Starting comprehensive data quality analysis...")
    print(f"Processed files directory: {processed_dir}")
    print(f"Database path: {database_path}")
    
    # Run all analyses
    analyzer.analyze_csv_files()
    analyzer.analyze_database()
    analyzer.cross_reference_data()
    
    # Generate summary
    data_is_clean = analyzer.generate_summary_report()
    
    if data_is_clean:
        print("\n✅ DATA QUALITY CHECK PASSED - No critical issues found!")
    else:
        print("\n❌ DATA QUALITY CHECK FAILED - Critical issues found that need fixing!")
        print("Recommendation: Run data cleanup and rebuilding scripts before proceeding.")
    
    return data_is_clean

if __name__ == "__main__":
    main()
