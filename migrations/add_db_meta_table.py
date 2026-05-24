#!/usr/bin/env python3
"""
Database Migration: Add db_meta Table
=====================================

This migration adds the db_meta table to support the mtime heuristic optimization
for file scanning. The table stores key-value pairs for system metadata.

Usage:
    python migrations/add_db_meta_table.py [database_path]

Author: AI Assistant
Date: 2025-01-04
"""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def migrate_database(db_path: str):
    """
    Add the db_meta table to the database.

    Args:
        db_path: Path to the SQLite database file
    """
    print(f"🔄 Migrating database: {db_path}")

    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if db_meta table already exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='db_meta'
        """
        )

        if cursor.fetchone():
            print("ℹ️ db_meta table already exists, skipping migration")
            conn.close()
            return True

        # Create db_meta table
        print("📊 Creating db_meta table...")
        cursor.execute(
            """
            CREATE TABLE db_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meta_key VARCHAR(255) UNIQUE NOT NULL,
                meta_value VARCHAR(500),
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index for performance
        print("🚀 Creating index on meta_key...")
        cursor.execute(
            """
            CREATE INDEX idx_db_meta_key ON db_meta(meta_key)
        """
        )

        # Add a migration record
        cursor.execute(
            """
            INSERT INTO db_meta (meta_key, meta_value, last_updated)
            VALUES ('migration_db_meta_table', ?, CURRENT_TIMESTAMP)
        """,
            (datetime.now().isoformat(),),
        )

        conn.commit()
        print("✅ db_meta table created successfully")

        # Verify table was created
        cursor.execute("SELECT COUNT(*) FROM db_meta")
        count = cursor.fetchone()[0]
        print(f"📋 db_meta table contains {count} records")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def main():
    """Main migration function"""
    print("🔧 Database Migration: Add db_meta Table")
    print("=" * 50)

    # Get database path from command line or use default
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Try to find database in common locations
        common_paths = [
            "database.sqlite",
            "greyhound_racing_data.db",
            "./database.sqlite",
            "./greyhound_racing_data.db",
        ]

        db_path = None
        for path in common_paths:
            if os.path.exists(path):
                db_path = path
                break

        if not db_path:
            print("❌ No database found. Please specify database path:")
            print("Usage: python migrations/add_db_meta_table.py [database_path]")
            sys.exit(1)

    print(f"📂 Target database: {db_path}")

    # Run migration
    success = migrate_database(db_path)

    if success:
        print("\n🎉 Migration completed successfully!")
        print("The mtime heuristic is now available for optimized file scanning.")
        print("\nUsage examples:")
        print("  python run.py analyze                    # Use mtime optimization")
        print("  python run.py analyze --strict-scan      # Full re-scan")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
