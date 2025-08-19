#!/usr/bin/env python3
"""
Migration: Add processed_race_files table
=========================================

This migration adds the processed_race_files table for robust caching
and de-duplication functionality.

Author: AI Assistant
Date: August 3, 2025
Version: 1.0.0 - Initial implementation
"""

import sqlite3
import os
import sys
from datetime import datetime

def get_database_paths():
    """Get all database paths that might need migration"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    potential_dbs = [
        os.path.join(base_dir, "databases", "greyhound_racing.db"),
        os.path.join(base_dir, "databases", "race_data.db"),
        os.path.join(base_dir, "databases", "comprehensive_greyhound_data.db"),
        os.path.join(base_dir, "databases", "unified_racing.db"),
        os.path.join(base_dir, "databases", "unified_data.db"),
    ]
    
    # Only return paths that exist
    return [db for db in potential_dbs if os.path.exists(db)]

def table_exists(cursor, table_name):
    """Check if a table exists in the database"""
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None

def migrate_database(db_path):
    """Migrate a single database"""
    print(f"🔄 Migrating database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table already exists
        if table_exists(cursor, 'processed_race_files'):
            print(f"   ✅ Table 'processed_race_files' already exists")
            conn.close()
            return True
        
        # Create the processed_race_files table
        cursor.execute('''
            CREATE TABLE processed_race_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                race_date DATE NOT NULL,
                venue TEXT NOT NULL, 
                race_no INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processed',
                error_message TEXT
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX idx_processed_files_hash ON processed_race_files(file_hash)')
        cursor.execute('CREATE INDEX idx_processed_files_race_key ON processed_race_files(race_date, venue, race_no)')
        cursor.execute('CREATE INDEX idx_processed_files_processed_at ON processed_race_files(processed_at)')
        
        # Commit changes
        conn.commit()
        
        print(f"   ✅ Successfully created table and indexes")
        
        # Verify table creation
        cursor.execute("SELECT COUNT(*) FROM processed_race_files")
        count = cursor.fetchone()[0]
        print(f"   📊 Table initialized with {count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def run_migration():
    """Run the migration on all databases"""
    print("🚀 Starting migration: Add processed_race_files table")
    print(f"📅 Migration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    db_paths = get_database_paths()
    
    if not db_paths:
        print("⚠️ No databases found to migrate")
        return True
    
    print(f"🎯 Found {len(db_paths)} databases to migrate:")
    for db_path in db_paths:
        print(f"   - {db_path}")
    
    print()
    
    success_count = 0
    
    for db_path in db_paths:
        if migrate_database(db_path):
            success_count += 1
        print()  # Add spacing between databases
    
    print(f"📊 Migration Summary:")
    print(f"   Total databases: {len(db_paths)}")
    print(f"   Successfully migrated: {success_count}")
    print(f"   Failed: {len(db_paths) - success_count}")
    
    if success_count == len(db_paths):
        print(f"\n🎉 All migrations completed successfully!")
        return True
    else:
        print(f"\n💥 Some migrations failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
