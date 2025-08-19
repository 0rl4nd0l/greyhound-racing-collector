#!/usr/bin/env python3
"""
Database Validation and Data Usage Efficiency Checker
=====================================================
This script validates the database integrity and checks if data is being used efficiently.
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseValidator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "race_data.db"
        self.validation_results = {
            'database_exists': False,
            'tables_found': [],
            'total_records': {},
            'data_consistency': {},
            'usage_efficiency': {},
            'recommendations': []
        }
    
    def check_database_existence(self):
        """Check if database exists and is accessible"""
        if not self.db_path.exists():
            logger.warning("Database file does not exist")
            self.validation_results['database_exists'] = False
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.validation_results['database_exists'] = True
            self.validation_results['tables_found'] = tables
            
            logger.info(f"Database found with tables: {tables}")
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.validation_results['database_exists'] = False
            return False
    
    def analyze_table_contents(self):
        """Analyze the contents of each table"""
        if not self.validation_results['database_exists']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in self.validation_results['tables_found']:
                try:
                    # Get record count
                    df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
                    count = df.iloc[0]['count']
                    self.validation_results['total_records'][table] = count
                    
                    # Get sample data and column info
                    sample_df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
                    
                    logger.info(f"Table {table}: {count} records, {len(sample_df.columns)} columns")
                    
                    # Check for null values
                    full_df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    null_counts = full_df.isnull().sum()
                    
                    self.validation_results['data_consistency'][table] = {
                        'total_records': count,
                        'columns': list(full_df.columns),
                        'null_values': dict(null_counts[null_counts > 0]),
                        'data_types': dict(full_df.dtypes.astype(str))
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing table {table}: {e}")
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
    
    def check_data_relationships(self):
        """Check relationships between data in different tables"""
        if not self.validation_results['database_exists']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check for common relationships
            relationships = {}
            
            # If we have races and entries tables, check referential integrity
            if 'races' in self.validation_results['tables_found'] and 'race_entries' in self.validation_results['tables_found']:
                cursor = conn.cursor()
                
                # Check for orphaned entries
                cursor.execute("""
                    SELECT COUNT(*) FROM race_entries 
                    WHERE race_id NOT IN (SELECT id FROM races)
                """)
                orphaned_entries = cursor.fetchone()[0]
                relationships['orphaned_entries'] = orphaned_entries
                
                # Check for races without entries
                cursor.execute("""
                    SELECT COUNT(*) FROM races 
                    WHERE id NOT IN (SELECT DISTINCT race_id FROM race_entries)
                """)
                empty_races = cursor.fetchone()[0]
                relationships['empty_races'] = empty_races
            
            self.validation_results['data_consistency']['relationships'] = relationships
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Error checking relationships: {e}")
    
    def analyze_data_usage_efficiency(self):
        """Analyze how efficiently the data is being used"""
        csv_files = list(self.base_path.rglob("*.csv"))
        json_files = list(self.base_path.rglob("*.json"))
        
        # Count files by category
        file_categories = {
            'race_data': 0,
            'form_guides': 0,
            'enhanced_analysis': 0,
            'upcoming_races': 0,
            'processed': 0,
            'other': 0
        }
        
        for csv_file in csv_files:
            path_str = str(csv_file).lower()
            if 'race_data' in path_str or 'organized_csvs' in path_str:
                file_categories['race_data'] += 1
            elif 'form_guide' in path_str:
                file_categories['form_guides'] += 1
            elif 'enhanced' in path_str:
                file_categories['enhanced_analysis'] += 1
            elif 'upcoming' in path_str:
                file_categories['upcoming_races'] += 1
            elif 'processed' in path_str:
                file_categories['processed'] += 1
            else:
                file_categories['other'] += 1
        
        # Calculate storage by category
        storage_by_category = {}
        for category in file_categories:
            storage_by_category[category] = 0
        
        for csv_file in csv_files:
            path_str = str(csv_file).lower()
            size = csv_file.stat().st_size
            
            if 'race_data' in path_str or 'organized_csvs' in path_str:
                storage_by_category['race_data'] += size
            elif 'form_guide' in path_str:
                storage_by_category['form_guides'] += size
            elif 'enhanced' in path_str:
                storage_by_category['enhanced_analysis'] += size
            elif 'upcoming' in path_str:
                storage_by_category['upcoming_races'] += size
            elif 'processed' in path_str:
                storage_by_category['processed'] += size
            else:
                storage_by_category['other'] += size
        
        self.validation_results['usage_efficiency'] = {
            'file_counts': file_categories,
            'storage_bytes': storage_by_category,
            'total_csv_files': len(csv_files),
            'total_json_files': len(json_files)
        }
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Database recommendations
        if not self.validation_results['database_exists']:
            recommendations.append("Create database schema and populate with existing CSV data")
        elif not self.validation_results['tables_found']:
            recommendations.append("Database exists but contains no tables - initialize schema")
        else:
            # Check data quality
            for table, info in self.validation_results['data_consistency'].items():
                if isinstance(info, dict) and 'null_values' in info:
                    if info['null_values']:
                        recommendations.append(f"Address null values in table {table}: {info['null_values']}")
            
            # Check relationships
            relationships = self.validation_results['data_consistency'].get('relationships', {})
            if relationships.get('orphaned_entries', 0) > 0:
                recommendations.append(f"Fix {relationships['orphaned_entries']} orphaned race entries")
            if relationships.get('empty_races', 0) > 0:
                recommendations.append(f"Review {relationships['empty_races']} races without entries")
        
        # File organization recommendations
        usage = self.validation_results['usage_efficiency']
        if usage['file_counts']['other'] > usage['file_counts']['race_data'] * 0.1:
            recommendations.append("Categorize and organize miscellaneous files")
        
        if usage['file_counts']['upcoming_races'] > 100:
            recommendations.append("Archive or remove old upcoming race files")
        
        # Data consolidation recommendations
        if usage['file_counts']['processed'] > usage['file_counts']['race_data']:
            recommendations.append("Consider consolidating processed files to reduce redundancy")
        
        self.validation_results['recommendations'] = recommendations
    
    def run_validation(self):
        """Run complete validation"""
        logger.info("Starting database and data usage validation...")
        
        self.check_database_existence()
        self.analyze_table_contents()
        self.check_data_relationships()
        self.analyze_data_usage_efficiency()
        self.generate_recommendations()
        
        # Save validation report
        report_path = self.base_path / "database_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation complete. Report saved to {report_path}")
        return self.validation_results

def print_validation_summary(results):
    """Print validation summary"""
    print("\n" + "="*60)
    print("DATABASE VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Database exists: {'Yes' if results['database_exists'] else 'No'}")
    
    if results['database_exists']:
        print(f"Tables found: {len(results['tables_found'])}")
        for table in results['tables_found']:
            count = results['total_records'].get(table, 0)
            print(f"  - {table}: {count:,} records")
    
    print(f"\nData Usage Efficiency:")
    usage = results['usage_efficiency']
    print(f"Total CSV files: {usage['total_csv_files']:,}")
    print(f"Total JSON files: {usage['total_json_files']:,}")
    
    print(f"\nFile distribution:")
    for category, count in usage['file_counts'].items():
        storage_mb = usage['storage_bytes'][category] / (1024*1024)
        print(f"  - {category}: {count:,} files ({storage_mb:.2f} MB)")
    
    if results['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    validator = DatabaseValidator(base_path)
    
    try:
        results = validator.run_validation()
        print_validation_summary(results)
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
