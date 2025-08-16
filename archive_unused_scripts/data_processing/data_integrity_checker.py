#!/usr/bin/env python3
"""
Data Integrity Checker
======================

Regular checks to catch data corruption early and maintain database quality.
Runs automated validation checks on the database to ensure:
- Field sizes are reasonable (≤12 dogs per race)
- No duplicate dog names per race
- No corrupted race IDs
- Finish positions are valid

Author: AI Assistant
Date: July 24, 2025
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

DATABASE_PATH = 'greyhound_racing_data.db'

class DataIntegrityChecker:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.issues = []
        
    def check_field_sizes(self):
        """Check for races with unreasonable field sizes"""
        print("🔍 Checking field sizes...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Find races with field sizes > 12
        query = """
        SELECT rm.race_id, rm.venue, rm.race_date, rm.field_size, 
               COUNT(DISTINCT drd.dog_name) as actual_dogs
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE rm.field_size > 12 OR rm.field_size IS NULL
        GROUP BY rm.race_id, rm.venue, rm.race_date, rm.field_size
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            self.issues.extend([
                {
                    'type': 'field_size_violation',
                    'race_id': row['race_id'],
                    'venue': row['venue'],
                    'field_size': row['field_size'],
                    'actual_dogs': row['actual_dogs'],
                    'severity': 'high'
                }
                for _, row in df.iterrows()
            ])
            print(f"   ❌ Found {len(df)} races with field size issues")
        else:
            print(f"   ✅ All field sizes are valid")
        
        return len(df) == 0
    
    def check_duplicate_dogs(self):
        """Check for duplicate dog names within the same race"""
        print("🔍 Checking for duplicate dogs per race...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT race_id, dog_name, COUNT(*) as count
        FROM dog_race_data
        WHERE dog_name IS NOT NULL AND dog_name != 'nan'
        GROUP BY race_id, dog_name
        HAVING COUNT(*) > 1
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            self.issues.extend([
                {
                    'type': 'duplicate_dogs',
                    'race_id': row['race_id'],
                    'dog_name': row['dog_name'],
                    'count': row['count'],
                    'severity': 'high'
                }
                for _, row in df.iterrows()
            ])
            print(f"   ❌ Found {len(df)} duplicate dog entries")
        else:
            print(f"   ✅ No duplicate dogs found")
        
        return len(df) == 0
    
    def check_corrupted_race_ids(self):
        """Check for corrupted or invalid race IDs"""
        print("🔍 Checking race ID validity...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT race_id, venue, race_date
        FROM race_metadata
        WHERE race_id IN ('UNK_0_UNKNOWN', 'UNKNOWN', '') 
           OR race_id IS NULL
           OR venue IN ('Unknown', 'UNKNOWN', 'UNK', '')
           OR venue IS NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            self.issues.extend([
                {
                    'type': 'invalid_race_id',
                    'race_id': row['race_id'],
                    'venue': row['venue'],
                    'race_date': row['race_date'],
                    'severity': 'critical'
                }
                for _, row in df.iterrows()
            ])
            print(f"   ❌ Found {len(df)} invalid race IDs")
        else:
            print(f"   ✅ All race IDs are valid")
        
        return len(df) == 0
    
    def check_excessive_dogs_per_race(self):
        """Check for races with too many unique dogs (corruption indicator)"""
        print("🔍 Checking for excessive dogs per race...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT rm.race_id, rm.venue, rm.race_date, COUNT(DISTINCT drd.dog_name) as unique_dogs
        FROM race_metadata rm
        JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE drd.dog_name IS NOT NULL AND drd.dog_name != 'nan'
        GROUP BY rm.race_id, rm.venue, rm.race_date
        HAVING COUNT(DISTINCT drd.dog_name) > 12
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            self.issues.extend([
                {
                    'type': 'excessive_dogs',
                    'race_id': row['race_id'],
                    'venue': row['venue'],
                    'unique_dogs': row['unique_dogs'],
                    'severity': 'critical'
                }
                for _, row in df.iterrows()
            ])
            print(f"   ❌ Found {len(df)} races with excessive dogs")
        else:
            print(f"   ✅ All races have reasonable dog counts")
        
        return len(df) == 0
    
    def check_invalid_finish_positions(self):
        """Check for invalid finish positions"""
        print("🔍 Checking finish positions...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT race_id, dog_name, finish_position
        FROM dog_race_data
        WHERE finish_position IS NOT NULL 
          AND finish_position NOT IN ('N/A', '', 'nan')
          AND (
              CAST(REPLACE(finish_position, '=', '') AS INTEGER) <= 0 
              OR CAST(REPLACE(finish_position, '=', '') AS INTEGER) > 12
          )
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            
            if len(df) > 0:
                self.issues.extend([
                    {
                        'type': 'invalid_finish_position',
                        'race_id': row['race_id'],
                        'dog_name': row['dog_name'],
                        'finish_position': row['finish_position'],
                        'severity': 'medium'
                    }
                    for _, row in df.iterrows()
                ])
                print(f"   ❌ Found {len(df)} invalid finish positions")
            else:
                print(f"   ✅ All finish positions are valid")
                
        except Exception as e:
            print(f"   ⚠️ Error checking finish positions: {e}")
            return False
        finally:
            conn.close()
        
        return len(df) == 0
    
    def check_data_consistency(self):
        """Check general data consistency"""
        print("🔍 Checking data consistency...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check for races without any dogs
        query1 = """
        SELECT rm.race_id, rm.venue, rm.race_date
        FROM race_metadata rm
        LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
        WHERE drd.race_id IS NULL
        """
        
        df1 = pd.read_sql_query(query1, conn)
        
        # Check for dogs without race metadata
        query2 = """
        SELECT drd.race_id, COUNT(*) as dog_count
        FROM dog_race_data drd
        LEFT JOIN race_metadata rm ON drd.race_id = rm.race_id
        WHERE rm.race_id IS NULL
        GROUP BY drd.race_id
        """
        
        df2 = pd.read_sql_query(query2, conn)
        conn.close()
        
        total_issues = len(df1) + len(df2)
        
        if len(df1) > 0:
            self.issues.extend([
                {
                    'type': 'race_without_dogs',
                    'race_id': row['race_id'],
                    'venue': row['venue'],
                    'severity': 'medium'
                }
                for _, row in df1.iterrows()
            ])
        
        if len(df2) > 0:
            self.issues.extend([
                {
                    'type': 'dogs_without_race',
                    'race_id': row['race_id'],
                    'dog_count': row['dog_count'],
                    'severity': 'medium'
                }
                for _, row in df2.iterrows()
            ])
        
        if total_issues > 0:
            print(f"   ❌ Found {total_issues} consistency issues")
        else:
            print(f"   ✅ Data consistency is good")
        
        return total_issues == 0
    
    def generate_summary_report(self):
        """Generate a summary report of all issues found"""
        print("\n📊 Data Integrity Summary")
        print("=" * 50)
        
        if not self.issues:
            print("✅ No data integrity issues found!")
            return
        
        # Group issues by severity
        critical_issues = [i for i in self.issues if i['severity'] == 'critical']
        high_issues = [i for i in self.issues if i['severity'] == 'high']
        medium_issues = [i for i in self.issues if i['severity'] == 'medium']
        
        print(f"🚨 Critical Issues: {len(critical_issues)}")
        print(f"⚠️  High Priority Issues: {len(high_issues)}")
        print(f"📋 Medium Priority Issues: {len(medium_issues)}")
        print(f"📈 Total Issues: {len(self.issues)}")
        
        # Show details for critical and high issues
        if critical_issues:
            print("\n🚨 Critical Issues (require immediate attention):")
            for issue in critical_issues:
                print(f"   - {issue['type']}: {issue['race_id']} ({issue.get('venue', 'unknown venue')})")
        
        if high_issues:
            print("\n⚠️ High Priority Issues:")
            for issue in high_issues[:5]:  # Show first 5
                print(f"   - {issue['type']}: {issue['race_id']} ({issue.get('venue', 'unknown venue')})")
            if len(high_issues) > 5:
                print(f"   ... and {len(high_issues) - 5} more")
        
        # Save detailed report
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"data_integrity_report_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.issues),
            'critical_count': len([i for i in self.issues if i['severity'] == 'critical']),
            'high_count': len([i for i in self.issues if i['severity'] == 'high']),
            'medium_count': len([i for i in self.issues if i['severity'] == 'medium']),
            'issues': self.issues
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Error saving report: {e}")
    
    def run_all_checks(self):
        """Run all integrity checks"""
        print("🔍 GREYHOUND RACING DATA INTEGRITY CHECKER")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        checks = [
            self.check_field_sizes,
            self.check_duplicate_dogs,
            self.check_corrupted_race_ids,
            self.check_excessive_dogs_per_race,
            self.check_invalid_finish_positions,
            self.check_data_consistency
        ]
        
        passed_checks = 0
        for check in checks:
            try:
                if check():
                    passed_checks += 1
            except Exception as e:
                print(f"   ❌ Check failed with error: {e}")
        
        print(f"\n📊 Check Summary: {passed_checks}/{len(checks)} checks passed")
        
        self.generate_summary_report()
        
        return len(self.issues) == 0

def main():
    """Main function"""
    checker = DataIntegrityChecker()
    
    # Run all checks
    all_good = checker.run_all_checks()
    
    if all_good:
        print("\n🎉 Database integrity is excellent!")
    else:
        print(f"\n⚠️ Found {len(checker.issues)} integrity issues that need attention.")
        print("Run the appropriate cleanup scripts to fix these issues.")

if __name__ == "__main__":
    main()
