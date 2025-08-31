#!/usr/bin/env python3
"""
Database Duplicate Cleanup Script
=================================

Fix duplicate (race_id, box_number) combinations and other data quality issues.
"""

import sqlite3
import os
import json
from datetime import datetime
import pandas as pd

def fix_database_duplicates(db_path="greyhound_racing_data.db"):
    """Fix duplicate records and data quality issues."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"🔧 Fixing database duplicates in: {db_path}")
    
    # Backup database first
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.system(f"cp '{db_path}' '{backup_path}'")
    print(f"✅ Database backed up to: {backup_path}")
    
    conn = sqlite3.connect(db_path)
    
    fixes_applied = {
        "duplicate_race_box_removed": 0,
        "invalid_positions_fixed": 0,
        "null_winners_identified": 0,
        "orphaned_records_removed": 0
    }
    
    try:
        # 1. Find and remove duplicate (race_id, box_number) combinations
        print("🔍 Finding duplicate (race_id, box_number) combinations...")
        
        duplicates_query = """
        SELECT race_id, box_number, COUNT(*) as cnt, 
               GROUP_CONCAT(ROWID) as rowids
        FROM dog_race_data 
        WHERE race_id IS NOT NULL AND box_number IS NOT NULL
        GROUP BY race_id, box_number 
        HAVING cnt > 1
        """
        
        duplicates = conn.execute(duplicates_query).fetchall()
        print(f"Found {len(duplicates)} duplicate (race_id, box_number) combinations")
        
        for race_id, box_number, count, rowids in duplicates:
            rowid_list = rowids.split(',')
            # Keep the first record, delete the rest
            for rowid in rowid_list[1:]:
                conn.execute("DELETE FROM dog_race_data WHERE ROWID = ?", (rowid,))
                fixes_applied["duplicate_race_box_removed"] += 1
        
        # 2. Fix invalid finish positions
        print("🔍 Fixing invalid finish positions...")
        
        invalid_positions = conn.execute("""
            UPDATE dog_race_data 
            SET finish_position = NULL 
            WHERE finish_position IS NOT NULL 
            AND (finish_position < 1 OR finish_position > 20)
        """).rowcount
        
        fixes_applied["invalid_positions_fixed"] = invalid_positions
        print(f"Fixed {invalid_positions} invalid finish positions")
        
        # 3. Identify races without winners (for manual review)
        print("🔍 Identifying races without winner names...")
        
        null_winners = conn.execute("""
            SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NULL
        """).fetchone()[0]
        
        fixes_applied["null_winners_identified"] = null_winners
        print(f"Found {null_winners} races without winner names (requires manual review)")
        
        # 4. Remove orphaned dog records without race metadata
        print("🔍 Removing orphaned dog records...")
        
        orphaned_removed = conn.execute("""
            DELETE FROM dog_race_data 
            WHERE race_id NOT IN (SELECT race_id FROM race_metadata WHERE race_id IS NOT NULL)
        """).rowcount
        
        fixes_applied["orphaned_records_removed"] = orphaned_removed
        print(f"Removed {orphaned_removed} orphaned dog records")
        
        # 5. Update database statistics
        print("📊 Updating database statistics...")
        conn.execute("ANALYZE")
        
        conn.commit()
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "backup_path": backup_path,
            "fixes_applied": fixes_applied,
            "status": "completed"
        }
        
        # Save report
        os.makedirs("reports", exist_ok=True)
        with open("reports/database_fixes_applied.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Database fixes completed successfully!")
        print(f"📊 Summary:")
        for fix, count in fixes_applied.items():
            print(f"   {fix}: {count}")
        print(f"📄 Report saved: reports/database_fixes_applied.json")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
        
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    db_path = os.getenv("GREYHOUND_DB_PATH", "greyhound_racing_data.db")
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    fix_database_duplicates(db_path)
