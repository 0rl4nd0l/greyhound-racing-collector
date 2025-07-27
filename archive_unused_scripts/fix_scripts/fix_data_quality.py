#!/usr/bin/env python3
"""
Data Quality Fixer
==================

This script identifies and fixes all data quality issues in the greyhound racing database
and analysis pipeline to ensure accurate predictions and reporting.

Author: AI Assistant
Date: July 24, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from enhanced_race_analyzer import EnhancedRaceAnalyzer

class DataQualityFixer:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.issues_found = []
        self.fixes_applied = []
        
    def diagnose_all_issues(self):
        """Comprehensive diagnosis of all data quality issues"""
        print("🔍 COMPREHENSIVE DATA QUALITY DIAGNOSIS")
        print("=" * 50)
        
        # 1. Database integrity check
        self.check_database_integrity()
        
        # 2. Enhanced analyzer validation
        self.check_enhanced_analyzer()
        
        # 3. Performance calculation validation
        self.check_performance_calculations()
        
        # 4. Data completeness check
        self.check_data_completeness()
        
        # 5. Duplicate detection
        self.check_for_duplicates()
        
        print(f"\n📊 DIAGNOSIS SUMMARY:")
        print(f"   Issues found: {len(self.issues_found)}")
        for issue in self.issues_found:
            print(f"   ❌ {issue}")
            
        return len(self.issues_found) == 0
    
    def check_database_integrity(self):
        """Check basic database integrity"""
        print("\n1. 🗄️ Database Integrity Check")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['race_metadata', 'dog_race_data']
            for table in required_tables:
                if table not in tables:
                    self.issues_found.append(f"Missing table: {table}")
                else:
                    print(f"   ✅ Table {table} exists")
            
            # Check record counts
            cursor.execute("SELECT COUNT(*) FROM race_metadata")
            races_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            dogs_count = cursor.fetchone()[0]
            
            print(f"   📊 Race records: {races_count}")
            print(f"   📊 Dog records: {dogs_count}")
            
            if races_count == 0:
                self.issues_found.append("No race records in database")
            if dogs_count == 0:
                self.issues_found.append("No dog records in database")
                
            # Check for null/invalid data
            cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE dog_name IS NULL OR dog_name = '' OR dog_name = 'nan'")
            invalid_dogs = cursor.fetchone()[0]
            
            if invalid_dogs > 0:
                self.issues_found.append(f"{invalid_dogs} records with invalid dog names")
                print(f"   ⚠️ {invalid_dogs} records with invalid dog names")
            else:
                print("   ✅ Dog names are valid")
                
            conn.close()
            
        except Exception as e:
            self.issues_found.append(f"Database connection error: {str(e)}")
            print(f"   ❌ Database error: {e}")
    
    def check_enhanced_analyzer(self):
        """Check enhanced analyzer functionality"""
        print("\n2. 🧠 Enhanced Analyzer Check")
        
        try:
            analyzer = EnhancedRaceAnalyzer(self.db_path)
            analyzer.load_data()
            
            if len(analyzer.data) == 0:
                self.issues_found.append("Enhanced analyzer loaded no data")
                print("   ❌ No data loaded")
                return
            
            print(f"   ✅ Loaded {len(analyzer.data)} records")
            
            # Test feature engineering
            analyzer.engineer_features()
            print("   ✅ Feature engineering completed")
            
            # Test performance normalization
            analyzer.normalize_performance()
            
            # Check performance score validity
            perf_stats = analyzer.data['performance_score'].describe()
            print(f"   📊 Performance scores: min={perf_stats['min']:.3f}, max={perf_stats['max']:.3f}, mean={perf_stats['mean']:.3f}")
            
            if perf_stats['min'] < 0 or perf_stats['max'] > 1:
                self.issues_found.append("Performance scores out of valid range (0-1)")
            
            # Test venue analysis
            venue_stats = analyzer.data.groupby('venue')['performance_score'].mean()
            zero_venues = (venue_stats == 0).sum()
            
            if zero_venues > 0:
                self.issues_found.append(f"{zero_venues} venues with zero performance scores")
                print(f"   ⚠️ {zero_venues} venues with zero performance")
            else:
                print("   ✅ Venue performance calculations valid")
                
        except Exception as e:
            self.issues_found.append(f"Enhanced analyzer error: {str(e)}")
            print(f"   ❌ Enhanced analyzer error: {e}")
    
    def check_performance_calculations(self):
        """Validate performance calculation logic"""
        print("\n3. 📈 Performance Calculation Check")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get sample data for validation
            query = """
            SELECT d.dog_name, d.finish_position, r.field_size, d.race_id
            FROM dog_race_data d
            JOIN race_metadata r ON d.race_id = r.race_id
            WHERE d.finish_position IS NOT NULL 
            AND d.dog_name IS NOT NULL 
            AND d.dog_name != '' 
            AND d.dog_name != 'nan'
            LIMIT 100
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) == 0:
                self.issues_found.append("No valid race data for performance calculation")
                return
            
            # Test performance calculation
            df['calc_performance'] = (df['field_size'] - df['finish_position'] + 1) / df['field_size']
            
            # Validate ranges
            invalid_scores = ((df['calc_performance'] < 0) | (df['calc_performance'] > 1)).sum()
            
            if invalid_scores > 0:
                self.issues_found.append(f"{invalid_scores} records produce invalid performance scores")
                print(f"   ⚠️ {invalid_scores} invalid performance calculations")
            else:
                print("   ✅ Performance calculation logic is correct")
                
            print(f"   📊 Sample performance range: {df['calc_performance'].min():.3f} to {df['calc_performance'].max():.3f}")
            
        except Exception as e:
            self.issues_found.append(f"Performance calculation error: {str(e)}")
            print(f"   ❌ Performance calculation error: {e}")
    
    def check_data_completeness(self):
        """Check data completeness and quality"""
        print("\n4. 📋 Data Completeness Check")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check essential fields
            cursor = conn.cursor()
            
            # Race metadata completeness
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE venue IS NULL OR venue = ''")
            missing_venues = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE race_date IS NULL OR race_date = ''")
            missing_dates = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE field_size IS NULL OR field_size = 0")
            missing_field_sizes = cursor.fetchone()[0]
            
            if missing_venues > 0:
                self.issues_found.append(f"{missing_venues} races missing venue")
                print(f"   ⚠️ {missing_venues} races missing venue")
                
            if missing_dates > 0:
                self.issues_found.append(f"{missing_dates} races missing date")
                print(f"   ⚠️ {missing_dates} races missing date")
                
            if missing_field_sizes > 0:
                self.issues_found.append(f"{missing_field_sizes} races missing field size")
                print(f"   ⚠️ {missing_field_sizes} races missing field size")
            
            # Dog data completeness
            cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NULL")
            missing_positions = cursor.fetchone()[0]
            
            if missing_positions > 0:
                self.issues_found.append(f"{missing_positions} dogs missing finish position")
                print(f"   ⚠️ {missing_positions} dogs missing finish position")
            
            if all([missing_venues == 0, missing_dates == 0, missing_field_sizes == 0, missing_positions == 0]):
                print("   ✅ All essential data fields are complete")
            
            conn.close()
            
        except Exception as e:
            self.issues_found.append(f"Data completeness check error: {str(e)}")
            print(f"   ❌ Data completeness error: {e}")
    
    def check_for_duplicates(self):
        """Check for duplicate records"""
        print("\n5. 🔍 Duplicate Detection")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for duplicate dog entries in same race
            cursor.execute("""
                SELECT race_id, dog_name, COUNT(*) as count
                FROM dog_race_data 
                WHERE dog_name IS NOT NULL AND dog_name != '' AND dog_name != 'nan'
                GROUP BY race_id, dog_name 
                HAVING COUNT(*) > 1
            """)
            
            duplicates = cursor.fetchall()
            
            if duplicates:
                self.issues_found.append(f"{len(duplicates)} duplicate dog entries found")
                print(f"   ⚠️ {len(duplicates)} duplicate dog entries")
                for race_id, dog_name, count in duplicates[:5]:  # Show first 5
                    print(f"      - {dog_name} appears {count} times in race {race_id}")
            else:
                print("   ✅ No duplicate dog entries found")
            
            conn.close()
            
        except Exception as e:
            self.issues_found.append(f"Duplicate check error: {str(e)}")
            print(f"   ❌ Duplicate check error: {e}")
    
    def fix_all_issues(self):
        """Apply fixes for all identified issues"""
        print("\n🔧 APPLYING FIXES")
        print("=" * 30)
        
        if not self.issues_found:
            print("✅ No issues found to fix!")
            return True
        
        success_count = 0
        
        # Fix duplicates
        if any("duplicate" in issue.lower() for issue in self.issues_found):
            if self.fix_duplicates():
                success_count += 1
        
        # Fix invalid data
        if any("invalid" in issue.lower() for issue in self.issues_found):
            if self.fix_invalid_data():
                success_count += 1
        
        # Fix missing data
        if any("missing" in issue.lower() for issue in self.issues_found):
            if self.fix_missing_data():
                success_count += 1
        
        print(f"\n📊 FIXES APPLIED: {success_count}")
        for fix in self.fixes_applied:
            print(f"   ✅ {fix}")
            
        return success_count > 0
    
    def fix_duplicates(self):
        """Remove duplicate dog entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create a cleaned table without duplicates
            cursor.execute("""
                CREATE TEMPORARY TABLE dog_race_data_clean AS
                SELECT * FROM dog_race_data
                WHERE rowid IN (
                    SELECT MIN(rowid)
                    FROM dog_race_data
                    WHERE dog_name IS NOT NULL AND dog_name != '' AND dog_name != 'nan'
                    GROUP BY race_id, dog_name
                )
            """)
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM dog_race_data")
            original_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dog_race_data_clean")
            clean_count = cursor.fetchone()[0]
            
            if clean_count < original_count:
                # Replace original table
                cursor.execute("DELETE FROM dog_race_data")
                cursor.execute("INSERT INTO dog_race_data SELECT * FROM dog_race_data_clean")
                
                removed = original_count - clean_count
                self.fixes_applied.append(f"Removed {removed} duplicate dog entries")
                print(f"   ✅ Removed {removed} duplicate entries")
                
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to fix duplicates: {e}")
            return False
    
    def fix_invalid_data(self):
        """Fix invalid data entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove records with invalid dog names
            cursor.execute("DELETE FROM dog_race_data WHERE dog_name IS NULL OR dog_name = '' OR dog_name = 'nan'")
            deleted = cursor.rowcount
            
            if deleted > 0:
                self.fixes_applied.append(f"Removed {deleted} records with invalid dog names")
                print(f"   ✅ Removed {deleted} invalid dog name records")
            
            # Fix non-numeric finish positions (convert to NULL if not numeric)
            cursor.execute("""
                UPDATE dog_race_data 
                SET finish_position = NULL 
                WHERE finish_position = 'N/A' OR finish_position = '' OR 
                      finish_position NOT GLOB '[0-9]*' OR 
                      CAST(finish_position AS INTEGER) <= 0 OR 
                      CAST(finish_position AS INTEGER) > 12
            """)
            
            updated = cursor.rowcount
            if updated > 0:
                self.fixes_applied.append(f"Fixed {updated} invalid finish positions")
                print(f"   ✅ Fixed {updated} invalid finish positions")
            
            # Convert valid finish positions to integers
            cursor.execute("""
                UPDATE dog_race_data 
                SET finish_position = CAST(finish_position AS INTEGER)
                WHERE finish_position IS NOT NULL AND finish_position GLOB '[0-9]*'
            """)
            
            converted = cursor.rowcount
            if converted > 0:
                self.fixes_applied.append(f"Converted {converted} finish positions to integers")
                print(f"   ✅ Converted {converted} finish positions to integers")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to fix invalid data: {e}")
            return False
    
    def fix_missing_data(self):
        """Fix missing essential data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fix missing field sizes (use maximum finish position as proxy)
            cursor.execute("""
                UPDATE race_metadata 
                SET field_size = (
                    SELECT MAX(finish_position) 
                    FROM dog_race_data 
                    WHERE dog_race_data.race_id = race_metadata.race_id
                    AND finish_position IS NOT NULL
                )
                WHERE field_size IS NULL OR field_size = 0
            """)
            
            updated = cursor.rowcount
            if updated > 0:
                self.fixes_applied.append(f"Fixed {updated} missing field sizes")
                print(f"   ✅ Fixed {updated} missing field sizes")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to fix missing data: {e}")
            return False
    
    def validate_fixes(self):
        """Validate that fixes worked"""
        print("\n🔍 VALIDATING FIXES")
        print("=" * 25)
        
        # Re-run diagnosis
        original_issues = len(self.issues_found)
        self.issues_found = []  # Reset
        
        self.diagnose_all_issues()
        
        remaining_issues = len(self.issues_found)
        fixed_issues = original_issues - remaining_issues
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Original issues: {original_issues}")
        print(f"   Fixed issues: {fixed_issues}")
        print(f"   Remaining issues: {remaining_issues}")
        
        if remaining_issues == 0:
            print("   🎉 ALL ISSUES FIXED!")
            return True
        else:
            print("   ⚠️ Some issues remain:")
            for issue in self.issues_found:
                print(f"      - {issue}")
            return False

def main():
    """Main execution"""
    print("🎯 GREYHOUND RACING DATA QUALITY FIXER")
    print("=" * 50)
    
    fixer = DataQualityFixer()
    
    # Step 1: Diagnose all issues
    is_clean = fixer.diagnose_all_issues()
    
    if is_clean:
        print("\n🎉 DATA IS ALREADY CLEAN!")
        return
    
    # Step 2: Apply fixes
    fixes_applied = fixer.fix_all_issues()
    
    if not fixes_applied:
        print("\n❌ NO FIXES COULD BE APPLIED")
        return
    
    # Step 3: Validate fixes
    validation_success = fixer.validate_fixes()
    
    if validation_success:
        print("\n✅ DATA QUALITY FIXED SUCCESSFULLY!")
        print("   Predictions and reports should now be accurate.")
    else:
        print("\n⚠️ PARTIAL SUCCESS")
        print("   Some issues were fixed, but others remain.")
    
    print(f"\n📋 SUMMARY:")
    print(f"   Issues diagnosed: {len(fixer.issues_found) if hasattr(fixer, 'issues_found') else 0}")
    print(f"   Fixes applied: {len(fixer.fixes_applied)}")
    print(f"   Next step: Re-run predictions to verify improvements")

if __name__ == "__main__":
    main()
