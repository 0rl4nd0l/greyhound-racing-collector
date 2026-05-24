#!/usr/bin/env python3
"""
Database Access Verification Script
===================================

Verifies read-only access to the unified greyhound database
and provides basic schema information for Step 1 completion.

Usage: python verify_database_access.py
"""

import os
import sqlite3
import sys
from pathlib import Path

DATABASE_PATH = "greyhound_racing_data.db"


def verify_database_access():
    """Verify database connectivity and schema integrity."""

    print("=" * 60)
    print("🔍 Greyhound Database Access Verification")
    print("=" * 60)

    # Check if database file exists
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ ERROR: Database file not found: {DATABASE_PATH}")
        return False

    # Get database file info
    db_size = os.path.getsize(DATABASE_PATH) / (1024 * 1024)  # MB
    print(f"📁 Database File: {DATABASE_PATH}")
    print(f"📊 Database Size: {db_size:.1f} MB")

    try:
        # Connect to database (read-only)
        conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
        print("✅ Database connection successful (read-only mode)")

        # Get table list
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = cursor.fetchall()

        print(f"📋 Total Tables: {len(tables)}")
        print("\n🗂️  Table Structure:")
        print("-" * 40)

        # Show first 20 tables with row counts
        for i, (table_name,) in enumerate(tables[:20]):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                print(f"  {i+1:2d}. {table_name:<30} ({row_count:,} rows)")
            except Exception as e:
                print(f"  {i+1:2d}. {table_name:<30} (error: {str(e)[:20]})")

        if len(tables) > 20:
            print(f"  ... and {len(tables) - 20} more tables")

        # Check critical tables
        print("\n🔍 Critical Table Verification:")
        print("-" * 40)
        critical_tables = ["race_metadata", "dog_race_data", "dogs", "predictions"]

        for table in critical_tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            exists = cursor.fetchone()
            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✅ {table:<20} ({count:,} records)")
            else:
                print(f"  ❌ {table:<20} (missing)")

        # Check for indexes
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
        index_count = cursor.fetchone()[0]
        print(f"\n📈 Database Indexes: {index_count}")

        # Check database integrity
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        if integrity == "ok":
            print("✅ Database integrity: OK")
        else:
            print(f"⚠️  Database integrity: {integrity}")

        conn.close()

        print("\n" + "=" * 60)
        print("✅ Step 1 Database Verification: COMPLETE")
        print("🚀 Ready for Step 2: Codebase Reconnaissance")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False


def generate_schema_summary():
    """Generate a quick schema summary for reference."""

    try:
        conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get table schemas
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        schemas = cursor.fetchall()

        print(f"\n📝 Schema Summary ({len(schemas)} tables):")
        print("-" * 50)

        for schema in schemas[:5]:  # Show first 5 table definitions
            if schema[0]:
                lines = schema[0].split("\n")
                table_name = (
                    lines[0].split()[-2] if len(lines[0].split()) > 2 else "unknown"
                )
                print(f"  📋 {table_name}")

        if len(schemas) > 5:
            print(f"  ... and {len(schemas) - 5} more table definitions")

        conn.close()

    except Exception as e:
        print(f"⚠️  Schema summary failed: {str(e)}")


if __name__ == "__main__":
    print("Starting database verification...")

    success = verify_database_access()

    if success:
        generate_schema_summary()
        sys.exit(0)
    else:
        print("\n❌ Database verification failed!")
        sys.exit(1)
