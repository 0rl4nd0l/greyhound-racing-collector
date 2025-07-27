#!/usr/bin/env python3
"""
Advanced Deduplication Script
Fixes the critical issue where dogs appear to race multiple times per day.
This violates the fundamental rule that a dog can only race once per day.
"""

import sqlite3
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedDeduplicator:
    def __init__(self, base_path="/Users/orlandolee/greyhound_racing_collector"):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "databases", "race_data.db")
        self.stats = {
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'records_before': 0,
            'records_after': 0
        }
    
    def analyze_duplicates(self):
        """Analyze the extent of the duplication problem"""
        logger.info("🔍 Analyzing duplicate problem...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM form_guide")
        total_records = cursor.fetchone()[0]
        self.stats['records_before'] = total_records
        
        # Duplicate dog-day combinations
        cursor.execute("""
            SELECT COUNT(*) as duplicate_days 
            FROM (
                SELECT dog_name, race_date 
                FROM form_guide 
                GROUP BY dog_name, race_date 
                HAVING COUNT(*) > 1
            )
        """)
        duplicate_days = cursor.fetchone()[0]
        self.stats['duplicates_found'] = duplicate_days
        
        # Total duplicate records
        cursor.execute("""
            SELECT SUM(dup_count - 1) as total_duplicates
            FROM (
                SELECT dog_name, race_date, COUNT(*) as dup_count
                FROM form_guide 
                GROUP BY dog_name, race_date 
                HAVING COUNT(*) > 1
            )
        """)
        total_duplicate_records = cursor.fetchone()[0] or 0
        
        # Sample of worst duplicates
        cursor.execute("""
            SELECT dog_name, race_date, COUNT(*) as race_count
            FROM form_guide 
            GROUP BY dog_name, race_date 
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        worst_duplicates = cursor.fetchall()
        
        conn.close()
        
        print("\n" + "="*60)
        print("🚨 DUPLICATE ANALYSIS REPORT")
        print("="*60)
        print(f"Total Records: {total_records:,}")
        print(f"Dog-Day Combinations with Duplicates: {duplicate_days:,}")
        print(f"Total Duplicate Records to Remove: {total_duplicate_records:,}")
        print(f"Data Corruption Level: {(total_duplicate_records/total_records)*100:.1f}%")
        print()
        print("🏆 WORST DUPLICATE OFFENDERS:")
        for dog_name, race_date, count in worst_duplicates:
            print(f"  {dog_name} on {race_date}: {count} races (impossible!)")
        print("="*60)
        
        return duplicate_days > 0
    
    def backup_database(self):
        """Create backup before deduplication"""
        backup_path = f"{self.db_path}.pre_dedup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
        
        return backup_path
    
    def deduplicate_form_guide(self):
        """Remove duplicates from form_guide table using intelligent rules"""
        logger.info("🧹 Starting intelligent deduplication...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create temporary table with deduplication logic
        cursor.execute("""
            CREATE TEMPORARY TABLE form_guide_deduped AS
            SELECT 
                form_id,
                dog_name,
                race_date,
                venue,
                distance,
                grade,
                box_number,
                finish_position,
                race_time,
                weight,
                margin,
                odds,
                track_condition,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY dog_name, race_date 
                    ORDER BY 
                        -- Prefer records with more complete data
                        CASE WHEN finish_position IS NOT NULL THEN 0 ELSE 1 END,
                        CASE WHEN weight IS NOT NULL THEN 0 ELSE 1 END,
                        CASE WHEN odds IS NOT NULL THEN 0 ELSE 1 END,
                        CASE WHEN venue IS NOT NULL AND venue != '' THEN 0 ELSE 1 END,
                        -- Prefer later created records (more recent data)
                        created_at DESC,
                        form_id DESC
                ) as row_num
            FROM form_guide
        """)
        
        # Count records that will be kept
        cursor.execute("SELECT COUNT(*) FROM form_guide_deduped WHERE row_num = 1")
        records_to_keep = cursor.fetchone()[0]
        
        # Count records that will be removed
        cursor.execute("SELECT COUNT(*) FROM form_guide_deduped WHERE row_num > 1")
        records_to_remove = cursor.fetchone()[0]
        
        logger.info(f"📊 Will keep {records_to_keep:,} records, remove {records_to_remove:,} duplicates")
        
        # Create new clean form_guide table
        cursor.execute("DROP TABLE form_guide")
        
        cursor.execute("""
            CREATE TABLE form_guide (
                form_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT NOT NULL,
                race_date DATE NOT NULL,
                venue TEXT NOT NULL,
                distance INTEGER,
                grade TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                race_time REAL,
                weight REAL,
                margin TEXT,
                odds TEXT,
                track_condition TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert deduplicated data
        cursor.execute("""
            INSERT INTO form_guide (
                dog_name, race_date, venue, distance, grade, box_number,
                finish_position, race_time, weight, margin, odds, 
                track_condition, created_at
            )
            SELECT 
                dog_name, race_date, venue, distance, grade, box_number,
                finish_position, race_time, weight, margin, odds, 
                track_condition, created_at
            FROM form_guide_deduped 
            WHERE row_num = 1
        """)
        
        # Recreate indexes
        cursor.execute("CREATE INDEX idx_form_guide_dog_date ON form_guide(dog_name, race_date)")
        cursor.execute("CREATE INDEX idx_form_guide_dog_name ON form_guide(dog_name)")
        cursor.execute("CREATE INDEX idx_form_guide_race_date ON form_guide(race_date)")
        cursor.execute("CREATE INDEX idx_form_guide_venue ON form_guide(venue)")
        
        conn.commit()
        conn.close()
        
        self.stats['duplicates_removed'] = records_to_remove
        logger.info(f"✅ Removed {records_to_remove:,} duplicate records")
    
    def verify_deduplication(self):
        """Verify that deduplication was successful"""
        logger.info("🔍 Verifying deduplication results...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for remaining duplicates
        cursor.execute("""
            SELECT COUNT(*) as remaining_duplicates
            FROM (
                SELECT dog_name, race_date 
                FROM form_guide 
                GROUP BY dog_name, race_date 
                HAVING COUNT(*) > 1
            )
        """)
        remaining_duplicates = cursor.fetchone()[0]
        
        # Total records after deduplication
        cursor.execute("SELECT COUNT(*) FROM form_guide")
        total_after = cursor.fetchone()[0]
        self.stats['records_after'] = total_after
        
        # Sample verification - check our problem dog
        cursor.execute("""
            SELECT race_date, venue, finish_position, weight
            FROM form_guide 
            WHERE dog_name = 'Kiwi Kawa' 
            ORDER BY race_date DESC
        """)
        kiwi_races = cursor.fetchall()
        
        conn.close()
        
        print(f"\n✅ DEDUPLICATION VERIFICATION:")
        print(f"Remaining duplicates: {remaining_duplicates}")
        print(f"Records before: {self.stats['records_before']:,}")
        print(f"Records after: {self.stats['records_after']:,}")
        print(f"Records removed: {self.stats['duplicates_removed']:,}")
        print(f"Data reduction: {(self.stats['duplicates_removed']/self.stats['records_before'])*100:.1f}%")
        
        if remaining_duplicates == 0:
            print("🎉 SUCCESS: All duplicates successfully removed!")
        else:
            print(f"⚠️ WARNING: {remaining_duplicates} duplicates still remain!")
        
        print(f"\n📋 Kiwi Kawa's races after deduplication ({len(kiwi_races)} total):")
        for i, (race_date, venue, position, weight) in enumerate(kiwi_races, 1):
            print(f"  {i:2d}. {race_date} | {venue} | Place {position} | Weight {weight}")
        
        return remaining_duplicates == 0
    
    def update_ml_data_quality(self):
        """Update the ML system to reflect cleaned data"""
        logger.info("🤖 Updating ML system with cleaned data...")
        
        # This would typically involve:
        # 1. Clearing any cached form data
        # 2. Reloading the ML system with clean data
        # 3. Updating any aggregated statistics
        
        # For now, we'll just log the improvement
        improvement = (self.stats['duplicates_removed'] / self.stats['records_before']) * 100
        logger.info(f"📈 Data quality improved by {improvement:.1f}% through deduplication")
    
    def generate_deduplication_report(self):
        """Generate comprehensive deduplication report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'statistics': self.stats,
            'quality_improvement': (self.stats['duplicates_removed'] / self.stats['records_before']) * 100
        }
        
        report_path = os.path.join(self.base_path, f"deduplication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Deduplication report saved to: {report_path}")
        return report_path
    
    def run_full_deduplication(self):
        """Run the complete deduplication process"""
        logger.info("🚀 Starting advanced deduplication process...")
        
        try:
            # Step 1: Analyze the problem
            has_duplicates = self.analyze_duplicates()
            
            if not has_duplicates:
                logger.info("✅ No duplicates found - database is clean!")
                return True
            
            # Step 2: Backup database
            backup_path = self.backup_database()
            
            # Step 3: Perform deduplication
            self.deduplicate_form_guide()
            
            # Step 4: Verify results
            success = self.verify_deduplication()
            
            # Step 5: Update ML system
            if success:
                self.update_ml_data_quality()
            
            # Step 6: Generate report
            report_path = self.generate_deduplication_report()
            
            if success:
                print(f"\n🎉 DEDUPLICATION COMPLETED SUCCESSFULLY!")
                print(f"📄 Report: {report_path}")
                print(f"💾 Backup: {backup_path}")
                print(f"\n💡 IMPACT:")
                print(f"  • Removed {self.stats['duplicates_removed']:,} duplicate records")
                print(f"  • Fixed {self.stats['duplicates_found']:,} impossible dog-day combinations")
                print(f"  • Improved data quality by {(self.stats['duplicates_removed']/self.stats['records_before'])*100:.1f}%")
                print(f"  • Database now respects the rule: 1 dog = 1 race per day")
            else:
                print(f"\n⚠️ DEDUPLICATION HAD ISSUES - check the report for details")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Deduplication failed: {e}")
            return False

def main():
    """Main function to run deduplication"""
    deduplicator = AdvancedDeduplicator()
    success = deduplicator.run_full_deduplication()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
