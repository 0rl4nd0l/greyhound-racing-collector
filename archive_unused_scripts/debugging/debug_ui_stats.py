#!/usr/bin/env python3
"""
UI Statistics Debug Tool
=======================

This tool helps identify where the UI is getting its statistics and
why there might be a discrepancy between database records and UI display.
"""

import sqlite3
import os
import json
from datetime import datetime

def investigate_ui_statistics():
    """Investigate all possible sources of UI statistics"""
    
    print("🔍 UI Statistics Investigation")
    print("=" * 50)
    
    # 1. Check main database tables
    print("\n📊 MAIN DATABASE STATISTICS:")
    try:
        conn = sqlite3.connect('greyhound_racing_data.db')
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count:,} records")
            
            # Show sample data for key tables
            if table_name in ['race_metadata', 'enhanced_expert_data']:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample = cursor.fetchall()
                if sample:
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    print(f"    Sample columns: {', '.join(columns[:5])}")
        
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Error accessing database: {e}")
    
    # 2. Check race filtering (winners vs non-winners)
    print("\n🏁 RACE FILTERING ANALYSIS:")
    try:
        conn = sqlite3.connect('greyhound_racing_data.db')
        cursor = conn.cursor()
        
        # Total races
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        total_races = cursor.fetchone()[0]
        
        # Races with winners
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != '' AND winner_name != 'nan'")
        races_with_winners = cursor.fetchone()[0]
        
        # Races without winners
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NULL OR winner_name = '' OR winner_name = 'nan'")
        races_without_winners = cursor.fetchone()[0]
        
        print(f"  Total races in database: {total_races:,}")
        print(f"  Races with winners (displayed): {races_with_winners:,}")
        print(f"  Races without winners (form guides): {races_without_winners:,}")
        
        # Check processing status column
        cursor.execute("SELECT processing_status, COUNT(*) FROM race_metadata GROUP BY processing_status")
        status_counts = cursor.fetchall()
        print("  Processing status breakdown:")
        for status, count in status_counts:
            print(f"    {status or 'NULL'}: {count:,}")
        
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Error analyzing race filtering: {e}")
    
    # 3. Check file directories
    print("\n📁 FILE DIRECTORY ANALYSIS:")
    directories = {
        'unprocessed': './unprocessed',
        'processed': './processed', 
        'historical_races': './historical_races',
        'upcoming_races': './upcoming_races',
        'enhanced_expert_data/csv': './enhanced_expert_data/csv',
        'enhanced_expert_data/json': './enhanced_expert_data/json'
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            try:
                if name == 'processed':
                    # Count recursively for processed
                    count = 0
                    for root, dirs, files in os.walk(path):
                        count += len([f for f in files if f.endswith('.csv')])
                else:
                    files = [f for f in os.listdir(path) if f.endswith(('.csv', '.json'))]
                    count = len(files)
                print(f"  {name}: {count:,} files")
                
                # Show sample filenames for key directories
                if name in ['enhanced_expert_data/csv', 'enhanced_expert_data/json'] and count > 0:
                    if name == 'processed':
                        sample_files = []
                        for root, dirs, files in os.walk(path):
                            sample_files.extend([f for f in files if f.endswith('.csv')][:3])
                    else:
                        sample_files = files[:3]
                    print(f"    Sample files: {', '.join(sample_files)}")
                    
            except Exception as e:
                print(f"  {name}: ❌ Error - {e}")
        else:
            print(f"  {name}: Directory not found")
    
    # 4. Check for any special counting logic
    print("\n🔄 SPECIAL PROCESSING STATUS CHECK:")
    try:
        # Check if there are any special filtering conditions
        conn = sqlite3.connect('greyhound_racing_data.db')
        cursor = conn.cursor()
        
        # Check for recent additions
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE filename IS NOT NULL AND filename != ''")
        races_with_filenames = cursor.fetchone()[0]
        print(f"  Races with filename data: {races_with_filenames:,}")
        
        # Check for enhanced status
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE processing_status = 'enhanced'")
        enhanced_races = cursor.fetchone()[0]
        print(f"  Races marked as 'enhanced': {enhanced_races:,}")
        
        # Check recent dates
        cursor.execute("SELECT MAX(extraction_timestamp) FROM race_metadata")
        latest_timestamp = cursor.fetchone()[0]
        print(f"  Latest extraction timestamp: {latest_timestamp}")
        
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Error checking special status: {e}")
    
    # 5. Check for cached data or reports
    print("\n📋 CACHED DATA & REPORTS:")
    report_files = [f for f in os.listdir('.') if 'report' in f.lower() and f.endswith('.json')]
    for report_file in report_files[-3:]:  # Show last 3 reports
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                print(f"  {report_file}:")
                if 'cleanup_summary' in data:
                    summary = data['cleanup_summary']
                    print(f"    Files cleaned: {summary.get('files_cleaned', 'N/A')}")
                    print(f"    Files integrated: {summary.get('files_integrated', 'N/A')}")
                    print(f"    DB records added: {summary.get('database_records_added', 'N/A')}")
                elif 'summary' in data:
                    summary = data['summary']
                    print(f"    Total CSV files: {summary.get('total_csv_files', 'N/A')}")
                    print(f"    Enhanced files: {summary.get('enhanced_csv_files', 'N/A')}")
        except Exception as e:
            print(f"    ❌ Error reading {report_file}: {e}")
    
    # 6. Generate current status summary
    print("\n📝 CURRENT STATUS SUMMARY:")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    try:
        conn = sqlite3.connect('greyhound_racing_data.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        total_db_races = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
        total_enhanced_records = cursor.fetchone()[0]
        
        # Count enhanced CSV files
        enhanced_csv_count = 0
        if os.path.exists('./enhanced_expert_data/csv'):
            enhanced_csv_count = len([f for f in os.listdir('./enhanced_expert_data/csv') if f.endswith('.csv')])
        
        print(f"  🗄️ Database races: {total_db_races:,}")
        print(f"  🗄️ Enhanced expert records: {total_enhanced_records:,}")
        print(f"  📁 Enhanced CSV files: {enhanced_csv_count:,}")
        
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Error generating summary: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 INVESTIGATION COMPLETE")
    print("\nIf you're seeing '32 total and 11 processed' in your UI,")
    print("please let me know which specific page/section you're looking at.")
    print("This will help identify the exact source of those numbers.")

if __name__ == "__main__":
    investigate_ui_statistics()
