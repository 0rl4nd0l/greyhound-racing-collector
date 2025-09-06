#!/usr/bin/env python3
"""
Database Analysis Tool
======================

Analyzes all databases in the greyhound racing collector project to understand
their purpose, size, and current usage.
"""

import os
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

def analyze_database(db_path):
    """Analyze a single database and return summary info."""
    try:
        # Get file size
        file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        
        with sqlite3.connect(db_path) as conn:
            # Get all table names
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Count records in each table
            table_counts = {}
            key_tables = []
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_counts[table] = count
                    
                    # Identify key data tables (not metadata/system tables)
                    if count > 0 and table not in ['alembic_version', 'db_meta', 'sqlite_sequence']:
                        key_tables.append((table, count))
                        
                except Exception as e:
                    table_counts[table] = f"Error: {e}"
            
            # Sort key tables by record count
            key_tables.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'path': db_path,
                'size_mb': file_size,
                'total_tables': len(tables),
                'table_counts': table_counts,
                'key_tables': key_tables[:10],  # Top 10 tables
                'status': 'OK'
            }
            
    except Exception as e:
        return {
            'path': db_path,
            'size_mb': 0,
            'total_tables': 0,
            'table_counts': {},
            'key_tables': [],
            'status': f'Error: {e}'
        }

def analyze_all_databases():
    """Analyze all databases in the project."""
    print("🔍 Analyzing all databases in greyhound racing collector...")
    print("=" * 70)
    
    # Find all database files
    db_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    
    # Sort by category
    production_dbs = []
    staging_dbs = []
    backup_dbs = []
    test_dbs = []
    archive_dbs = []
    
    for db_path in sorted(db_files):
        if 'test' in db_path.lower():
            test_dbs.append(db_path)
        elif 'backup' in db_path.lower() or 'system_backup' in db_path:
            backup_dbs.append(db_path)
        elif 'staging' in db_path.lower():
            staging_dbs.append(db_path)
        elif 'archive' in db_path.lower() or 'old' in db_path.lower():
            archive_dbs.append(db_path)
        else:
            production_dbs.append(db_path)
    
    print(f"📊 Found {len(db_files)} database files:")
    print(f"   • Production: {len(production_dbs)}")
    print(f"   • Staging: {len(staging_dbs)}")
    print(f"   • Backup: {len(backup_dbs)}")  
    print(f"   • Test: {len(test_dbs)}")
    print(f"   • Archive: {len(archive_dbs)}")
    print()
    
    # Analyze each category
    all_results = {}
    
    def analyze_category(category_name, db_list, emoji):
        if not db_list:
            return
            
        print(f"{emoji} {category_name.upper()} DATABASES")
        print("-" * 50)
        
        for db_path in db_list:
            result = analyze_database(db_path)
            all_results[db_path] = result
            
            print(f"📂 {db_path}")
            print(f"   Size: {result['size_mb']:.2f} MB")
            print(f"   Tables: {result['total_tables']}")
            print(f"   Status: {result['status']}")
            
            if result['key_tables']:
                print("   Key Tables (top 5):")
                for table, count in result['key_tables'][:5]:
                    print(f"     • {table}: {count:,} records")
            print()
    
    # Analyze each category
    analyze_category("Production", production_dbs, "🏭")
    analyze_category("Staging", staging_dbs, "🚧") 
    analyze_category("Backup", backup_dbs, "💾")
    analyze_category("Test", test_dbs, "🧪")
    analyze_category("Archive", archive_dbs, "🗄️")
    
    # Summary analysis
    print("📋 DATABASE USAGE SUMMARY")
    print("=" * 50)
    
    # Find the main active database
    main_candidates = [
        './greyhound_racing_data.db',
        './greyhound_racing_data_staging.db', 
        './databases/greyhound_racing_data.db'
    ]
    
    print("🎯 PRIMARY DATABASES:")
    for candidate in main_candidates:
        if candidate in all_results:
            result = all_results[candidate]
            total_records = sum(count for count in result['table_counts'].values() if isinstance(count, int))
            print(f"   • {candidate}")
            print(f"     Size: {result['size_mb']:.2f} MB")
            print(f"     Total Records: {total_records:,}")
            print(f"     Key Tables: {len(result['key_tables'])}")
            print()
    
    # Find largest databases
    print("📈 LARGEST DATABASES:")
    sorted_by_size = sorted(all_results.items(), key=lambda x: x[1]['size_mb'], reverse=True)
    for db_path, result in sorted_by_size[:5]:
        if result['size_mb'] > 1:  # Only show DBs > 1MB
            total_records = sum(count for count in result['table_counts'].values() if isinstance(count, int))
            print(f"   • {db_path}")
            print(f"     Size: {result['size_mb']:.2f} MB")
            print(f"     Records: {total_records:,}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    
    # Check for main production DB
    main_db = './greyhound_racing_data.db'
    staging_db = './greyhound_racing_data_staging.db'
    
    if main_db in all_results and staging_db in all_results:
        main_size = all_results[main_db]['size_mb']
        staging_size = all_results[staging_db]['size_mb']
        
        main_records = sum(count for count in all_results[main_db]['table_counts'].values() if isinstance(count, int))
        staging_records = sum(count for count in all_results[staging_db]['table_counts'].values() if isinstance(count, int))
        
        print(f"   • Main DB: {main_records:,} records ({main_size:.1f} MB)")
        print(f"   • Staging DB: {staging_records:,} records ({staging_size:.1f} MB)")
        
        if staging_records > main_records * 2:
            print("   ⚠️  Staging DB has significantly more data than main DB")
            print("      Consider syncing or switching primary database")
        
        if main_records < 1000:
            print("   ⚠️  Main DB has very little data - may need data import")
    
    # Check for cleanup opportunities
    total_backup_size = sum(result['size_mb'] for path, result in all_results.items() if 'backup' in path.lower())
    if total_backup_size > 100:
        print(f"   • Backup files total {total_backup_size:.1f} MB - consider archiving old backups")
    
    total_test_size = sum(result['size_mb'] for path, result in all_results.items() if 'test' in path.lower())
    if total_test_size > 50:
        print(f"   • Test files total {total_test_size:.1f} MB - consider cleanup")

if __name__ == "__main__":
    analyze_all_databases()
