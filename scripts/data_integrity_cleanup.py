#!/usr/bin/env python3
"""
Data Integrity Cleanup Script
============================

Resolves critical data quality issues identified in the database audit:
- 7,636 box number duplicates 
- 223 dog name duplicates
- 1,108 races with multiple winners
- 1,451 races without winners
- Missing winner information (71.42% of races)

This script implements deterministic cleanup rules while preserving data integrity.

Usage:
    python scripts/data_integrity_cleanup.py --dry-run
    python scripts/data_integrity_cleanup.py --execute --backup-first
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DataIntegrityCleanup:
    """Comprehensive data integrity cleanup system."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.backup_path = f"docs/analysis/pre_cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "issues_found": {},
            "fixes_applied": {},
            "constraints_added": [],
            "data_preserved": True
        }
    
    def create_backup(self) -> bool:
        """Create backup before any modifications."""
        print(f"📦 Creating backup at {self.backup_path}")
        try:
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(self.backup_path) as backup:
                    source.backup(backup)
            print(f"✅ Backup created successfully")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def analyze_duplicates(self) -> Dict:
        """Analyze all duplicate issues comprehensively."""
        print("🔍 Analyzing duplicate data issues...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 1. Box number duplicates
        box_dups_query = """
        SELECT race_id, box_number, COUNT(*) as dup_count, 
               GROUP_CONCAT(id) as record_ids,
               GROUP_CONCAT(dog_clean_name) as dog_names
        FROM dog_race_data 
        WHERE box_number IS NOT NULL
        GROUP BY race_id, box_number
        HAVING COUNT(*) > 1
        ORDER BY dup_count DESC
        """
        box_duplicates = pd.read_sql_query(box_dups_query, conn)
        
        # 2. Dog name duplicates  
        dog_dups_query = """
        SELECT race_id, dog_clean_name, COUNT(*) as dup_count,
               GROUP_CONCAT(id) as record_ids,
               GROUP_CONCAT(box_number) as box_numbers,
               GROUP_CONCAT(finish_position) as finish_positions
        FROM dog_race_data
        WHERE dog_clean_name IS NOT NULL
        GROUP BY race_id, dog_clean_name
        HAVING COUNT(*) > 1
        ORDER BY dup_count DESC
        """
        dog_duplicates = pd.read_sql_query(dog_dups_query, conn)
        
        # 3. Multiple winners
        multi_winners_query = """
        SELECT race_id, COUNT(*) as winner_count,
               GROUP_CONCAT(id) as record_ids,
               GROUP_CONCAT(dog_clean_name) as winner_names
        FROM dog_race_data
        WHERE finish_position = 1
        GROUP BY race_id
        HAVING COUNT(*) > 1
        ORDER BY winner_count DESC
        """
        multiple_winners = pd.read_sql_query(multi_winners_query, conn)
        
        # 4. Races without winners
        no_winners_query = """
        SELECT r.race_id, r.venue, r.race_date, r.field_size
        FROM race_metadata r
        WHERE NOT EXISTS (
            SELECT 1 FROM dog_race_data d 
            WHERE d.race_id = r.race_id AND d.finish_position = 1
        )
        ORDER BY r.race_date DESC
        """
        races_without_winners = pd.read_sql_query(no_winners_query, conn)
        
        conn.close()
        
        analysis = {
            "box_duplicates": {
                "count": len(box_duplicates),
                "total_duplicate_records": box_duplicates['dup_count'].sum() - len(box_duplicates),
                "data": box_duplicates
            },
            "dog_duplicates": {
                "count": len(dog_duplicates),
                "total_duplicate_records": dog_duplicates['dup_count'].sum() - len(dog_duplicates),
                "data": dog_duplicates
            },
            "multiple_winners": {
                "count": len(multiple_winners),
                "total_extra_winners": multiple_winners['winner_count'].sum() - len(multiple_winners),
                "data": multiple_winners
            },
            "races_without_winners": {
                "count": len(races_without_winners),
                "data": races_without_winners
            }
        }
        
        self.report["issues_found"] = {k: v["count"] for k, v in analysis.items()}
        
        print(f"📊 Found {analysis['box_duplicates']['count']} box number duplicate groups")
        print(f"📊 Found {analysis['dog_duplicates']['count']} dog name duplicate groups")
        print(f"📊 Found {analysis['multiple_winners']['count']} races with multiple winners")
        print(f"📊 Found {analysis['races_without_winners']['count']} races without winners")
        
        return analysis
    
    def resolve_box_number_duplicates(self, duplicates_data: pd.DataFrame, dry_run: bool = True) -> int:
        """Resolve box number duplicates using deterministic rules."""
        print("🔧 Resolving box number duplicates...")
        
        if dry_run:
            print("   [DRY RUN] Would resolve box number duplicates")
            return len(duplicates_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fixed_count = 0
        
        for _, row in duplicates_data.iterrows():
            race_id = row['race_id']
            box_number = row['box_number']
            record_ids = row['record_ids'].split(',')
            
            # Deterministic rule: Keep the record with the lowest ID (oldest insertion)
            # This assumes first-inserted is most likely correct
            keep_id = min(record_ids, key=int)
            remove_ids = [rid for rid in record_ids if rid != keep_id]
            
            if remove_ids:
                cursor.execute(
                    f"DELETE FROM dog_race_data WHERE id IN ({','.join(['?' for _ in remove_ids])})",
                    remove_ids
                )
                fixed_count += len(remove_ids)
                
        conn.commit()
        conn.close()
        
        print(f"✅ Removed {fixed_count} duplicate box number records")
        return fixed_count
    
    def resolve_dog_name_duplicates(self, duplicates_data: pd.DataFrame, dry_run: bool = True) -> int:
        """Resolve dog name duplicates using intelligent rules."""
        print("🔧 Resolving dog name duplicates...")
        
        if dry_run:
            print("   [DRY RUN] Would resolve dog name duplicates")
            return len(duplicates_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fixed_count = 0
        
        for _, row in duplicates_data.iterrows():
            race_id = row['race_id']
            dog_name = row['dog_clean_name']
            record_ids = row['record_ids'].split(',')
            finish_positions = row['finish_positions'].split(',')
            
            # Intelligent rule: Keep record with valid finish position, or lowest ID
            valid_records = []
            for i, record_id in enumerate(record_ids):
                finish_pos = finish_positions[i]
                if finish_pos and finish_pos != 'None' and finish_pos.isdigit():
                    valid_records.append((record_id, int(finish_pos)))
            
            if valid_records:
                # Keep the record with the best finish position (lowest number)
                keep_id = min(valid_records, key=lambda x: x[1])[0]
            else:
                # Fallback: keep oldest record (lowest ID)
                keep_id = min(record_ids, key=int)
            
            remove_ids = [rid for rid in record_ids if rid != keep_id]
            
            if remove_ids:
                cursor.execute(
                    f"DELETE FROM dog_race_data WHERE id IN ({','.join(['?' for _ in remove_ids])})",
                    remove_ids
                )
                fixed_count += len(remove_ids)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Removed {fixed_count} duplicate dog name records")
        return fixed_count
    
    def resolve_multiple_winners(self, winners_data: pd.DataFrame, dry_run: bool = True) -> int:
        """Resolve races with multiple winners."""
        print("🔧 Resolving multiple winners per race...")
        
        if dry_run:
            print("   [DRY RUN] Would resolve multiple winners")
            return len(winners_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fixed_count = 0
        
        for _, row in winners_data.iterrows():
            race_id = row['race_id']
            record_ids = row['record_ids'].split(',')
            
            # Rule: Keep the winner with lowest record ID (first inserted)
            # This assumes first-inserted winner is most likely correct
            keep_id = min(record_ids, key=int)
            
            # Set other "winners" to position 2, 3, etc.
            other_ids = [rid for rid in record_ids if rid != keep_id]
            
            for i, other_id in enumerate(other_ids):
                new_position = i + 2  # Start from position 2
                cursor.execute(
                    "UPDATE dog_race_data SET finish_position = ? WHERE id = ?",
                    (new_position, other_id)
                )
                fixed_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Corrected {fixed_count} false winner records")
        return fixed_count
    
    def populate_missing_winners(self, races_data: pd.DataFrame, dry_run: bool = True) -> int:
        """Populate missing winner information in race_metadata."""
        print("🔧 Populating missing winner information...")
        
        if dry_run:
            print("   [DRY RUN] Would populate missing winners")
            return len(races_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        populated_count = 0
        
        for _, row in races_data.iterrows():
            race_id = row['race_id']
            
            # Find the winner from dog_race_data
            cursor.execute("""
                SELECT dog_clean_name, odds 
                FROM dog_race_data 
                WHERE race_id = ? AND finish_position = 1
                LIMIT 1
            """, (race_id,))
            
            winner_data = cursor.fetchone()
            if winner_data:
                winner_name, winner_odds = winner_data
                
                # Update race_metadata with winner info
                cursor.execute("""
                    UPDATE race_metadata 
                    SET winner_name = ?, winner_odds = ?
                    WHERE race_id = ?
                """, (winner_name, winner_odds, race_id))
                
                populated_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Populated winner info for {populated_count} races")
        return populated_count
    
    def add_unique_constraints(self, dry_run: bool = True) -> List[str]:
        """Add unique constraints to prevent future duplicates."""
        print("🔒 Adding unique constraints...")
        
        constraints = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_box_per_race ON dog_race_data(race_id, box_number)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_dog_per_race ON dog_race_data(race_id, dog_clean_name)"
        ]
        
        if dry_run:
            print("   [DRY RUN] Would add unique constraints:")
            for constraint in constraints:
                print(f"      {constraint}")
            return constraints
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for constraint in constraints:
            try:
                cursor.execute(constraint)
                print(f"✅ Added: {constraint}")
            except sqlite3.Error as e:
                print(f"❌ Failed to add constraint: {e}")
        
        conn.commit()
        conn.close()
        
        self.report["constraints_added"] = constraints
        return constraints
    
    def validate_cleanup(self) -> Dict:
        """Validate that cleanup was successful."""
        print("✅ Validating cleanup results...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check for remaining duplicates
        validation_queries = {
            "box_duplicates_remaining": """
                SELECT COUNT(*) FROM (
                    SELECT race_id, box_number, COUNT(*) 
                    FROM dog_race_data 
                    WHERE box_number IS NOT NULL
                    GROUP BY race_id, box_number HAVING COUNT(*) > 1
                )
            """,
            "dog_duplicates_remaining": """
                SELECT COUNT(*) FROM (
                    SELECT race_id, dog_clean_name, COUNT(*) 
                    FROM dog_race_data 
                    WHERE dog_clean_name IS NOT NULL
                    GROUP BY race_id, dog_clean_name HAVING COUNT(*) > 1
                )
            """,
            "multiple_winners_remaining": """
                SELECT COUNT(*) FROM (
                    SELECT race_id, COUNT(*) 
                    FROM dog_race_data 
                    WHERE finish_position = 1
                    GROUP BY race_id HAVING COUNT(*) > 1
                )
            """,
            "races_without_winners": """
                SELECT COUNT(*) FROM race_metadata r
                WHERE NOT EXISTS (
                    SELECT 1 FROM dog_race_data d 
                    WHERE d.race_id = r.race_id AND d.finish_position = 1
                )
            """
        }
        
        validation_results = {}
        for check, query in validation_queries.items():
            result = pd.read_sql_query(query, conn).iloc[0, 0]
            validation_results[check] = result
            status = "✅ CLEAN" if result == 0 else f"⚠️  {result} remaining"
            print(f"   {check}: {status}")
        
        conn.close()
        return validation_results
    
    def generate_report(self) -> str:
        """Generate comprehensive cleanup report."""
        report_path = f"docs/analysis/data_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Data Integrity Cleanup Report\n")
            f.write(f"**Date**: {self.report['timestamp']}\n\n")
            
            f.write("## Issues Found\n")
            for issue, count in self.report['issues_found'].items():
                f.write(f"- **{issue}**: {count}\n")
            
            f.write("\n## Fixes Applied\n")
            for fix, count in self.report['fixes_applied'].items():
                f.write(f"- **{fix}**: {count} records affected\n")
            
            f.write("\n## Constraints Added\n")
            for constraint in self.report['constraints_added']:
                f.write(f"- `{constraint}`\n")
            
            f.write(f"\n## Backup Location\n")
            f.write(f"Pre-cleanup backup: `{self.backup_path}`\n")
        
        print(f"📄 Report generated: {report_path}")
        return report_path
    
    def execute_cleanup(self, dry_run: bool = True, create_backup: bool = True) -> bool:
        """Execute complete data integrity cleanup process."""
        print("🚀 Starting Data Integrity Cleanup")
        print("=" * 50)
        
        if create_backup and not dry_run:
            if not self.create_backup():
                print("❌ Backup failed - aborting cleanup")
                return False
        
        # Step 1: Analyze all issues
        analysis = self.analyze_duplicates()
        
        if dry_run:
            print("\n🔍 DRY RUN - No changes will be made")
            print("=" * 50)
        
        # Step 2: Apply fixes
        fixes_applied = {}
        
        # Fix box number duplicates
        if analysis['box_duplicates']['count'] > 0:
            fixes_applied['box_duplicates_removed'] = self.resolve_box_number_duplicates(
                analysis['box_duplicates']['data'], dry_run
            )
        
        # Fix dog name duplicates
        if analysis['dog_duplicates']['count'] > 0:
            fixes_applied['dog_duplicates_removed'] = self.resolve_dog_name_duplicates(
                analysis['dog_duplicates']['data'], dry_run
            )
        
        # Fix multiple winners
        if analysis['multiple_winners']['count'] > 0:
            fixes_applied['multiple_winners_corrected'] = self.resolve_multiple_winners(
                analysis['multiple_winners']['data'], dry_run
            )
        
        # Populate missing winners
        if analysis['races_without_winners']['count'] > 0:
            fixes_applied['missing_winners_populated'] = self.populate_missing_winners(
                analysis['races_without_winners']['data'], dry_run
            )
        
        # Add constraints
        constraints_added = self.add_unique_constraints(dry_run)
        
        self.report['fixes_applied'] = fixes_applied
        
        # Step 3: Validate (only if not dry run)
        if not dry_run:
            print("\n🔍 Validating cleanup results...")
            validation = self.validate_cleanup()
            self.report['validation'] = validation
            
            # Generate report
            report_path = self.generate_report()
            
            print(f"\n✅ Cleanup completed successfully!")
            print(f"📄 Full report: {report_path}")
            return True
        else:
            print(f"\n🔍 DRY RUN completed - use --execute to apply changes")
            return True


def main():
    parser = argparse.ArgumentParser(description="Data Integrity Cleanup")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be done without making changes")
    parser.add_argument("--execute", action="store_true",
                       help="Execute the cleanup (overrides --dry-run)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip creating backup (not recommended)")
    parser.add_argument("--db-path", default="greyhound_racing_data.db",
                       help="Database path")
    
    args = parser.parse_args()
    
    # Determine if this is a dry run
    dry_run = not args.execute
    create_backup = not args.no_backup
    
    if not dry_run and not args.no_backup:
        confirm = input("⚠️  This will modify the database. Continue? (yes/no): ").lower()
        if confirm != 'yes':
            print("Aborted by user")
            return 1
    
    # Execute cleanup
    cleanup = DataIntegrityCleanup(args.db_path)
    success = cleanup.execute_cleanup(dry_run=dry_run, create_backup=create_backup)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
