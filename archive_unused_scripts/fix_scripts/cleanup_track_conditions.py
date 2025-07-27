#!/usr/bin/env python3
"""
Track Condition Database Cleanup Script
=======================================

This script cleans up false positive track condition data that was extracted
from race names, sponsorship text, or other non-track-condition sources.

False positives include:
- "Fast" extracted from "ladbrokes-fast-withdrawals" or "sportsbet-fast-form" 
- "nan" values
- Track conditions that appear in the race URL (indicating sponsorship text)
"""

import sqlite3
import re
from datetime import datetime
from typing import List, Dict, Tuple

class TrackConditionCleaner:
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        
    def analyze_current_data(self) -> Dict:
        """Analyze current track condition data to identify cleanup candidates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all races with track conditions
        cursor.execute("""
            SELECT race_id, track_condition, url, race_name
            FROM race_metadata 
            WHERE track_condition IS NOT NULL AND track_condition != ''
            ORDER BY track_condition, race_date DESC
        """)
        
        races = cursor.fetchall()
        conn.close()
        
        analysis = {
            'total_with_conditions': len(races),
            'false_positives': [],
            'suspicious': [],
            'likely_valid': [],
            'condition_counts': {}
        }
        
        for race_id, condition, url, race_name in races:
            # Count occurrences
            analysis['condition_counts'][condition] = analysis['condition_counts'].get(condition, 0) + 1
            
            # Identify false positives
            is_false_positive = False
            reason = ""
            
            # Rule 1: "nan" values are always false positives
            if condition.lower() == 'nan':
                is_false_positive = True
                reason = "Invalid 'nan' value"
            
            # Rule 2: Condition appears in URL (sponsorship text)
            elif url and condition.lower() in url.lower():
                is_false_positive = True
                reason = f"Condition '{condition}' found in URL sponsorship text"
            
            # Rule 3: Known sponsorship patterns
            elif url and any(sponsor in url.lower() for sponsor in ['fast-withdrawals', 'fast-form', 'sportsbet-fast']):
                if condition.lower() == 'fast':
                    is_false_positive = True
                    reason = "Fast extracted from known sponsorship text"
            
            if is_false_positive:
                analysis['false_positives'].append({
                    'race_id': race_id,
                    'condition': condition,
                    'url': url,
                    'reason': reason
                })
            else:
                # Check if suspicious (needs manual review)
                is_suspicious = False
                
                # Suspicious if condition is common word that might be in race names
                if condition.lower() in ['good', 'heavy', 'fast', 'slow'] and url:
                    # Check if the condition word appears in the race name part of URL
                    url_parts = url.split('/')
                    if len(url_parts) > 0:
                        race_name_part = url_parts[-1].replace('?trial=false', '')
                        if condition.lower() in race_name_part.lower():
                            is_suspicious = True
                
                if is_suspicious:
                    analysis['suspicious'].append({
                        'race_id': race_id,
                        'condition': condition,
                        'url': url,
                        'reason': 'Condition word appears in race name URL'
                    })
                else:
                    analysis['likely_valid'].append({
                        'race_id': race_id,
                        'condition': condition,
                        'url': url
                    })
        
        return analysis
    
    def clean_false_positives(self, dry_run: bool = True) -> Dict:
        """Clean up false positive track conditions"""
        analysis = self.analyze_current_data()
        
        print(f"📊 Track Condition Cleanup Analysis")
        print(f"   Total races with conditions: {analysis['total_with_conditions']}")
        print(f"   False positives identified: {len(analysis['false_positives'])}")
        print(f"   Suspicious entries: {len(analysis['suspicious'])}")
        print(f"   Likely valid entries: {len(analysis['likely_valid'])}")
        
        print(f"\n📋 Condition counts:")
        for condition, count in sorted(analysis['condition_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {condition}: {count}")
        
        print(f"\n❌ False positives to be removed:")
        for fp in analysis['false_positives']:
            print(f"   {fp['race_id']}: '{fp['condition']}' - {fp['reason']}")
        
        print(f"\n⚠️  Suspicious entries (manual review recommended):")
        for sus in analysis['suspicious']:
            print(f"   {sus['race_id']}: '{sus['condition']}' - {sus['reason']}")
            print(f"      URL: {sus['url']}")
        
        if dry_run:
            print(f"\n🧪 DRY RUN - No changes made to database")
            return analysis
        
        # Actually perform the cleanup
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        race_ids_to_clean = [fp['race_id'] for fp in analysis['false_positives']]
        
        if race_ids_to_clean:
            # Create backup of current state
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_query = f"""
                CREATE TABLE IF NOT EXISTS track_condition_backup_{timestamp} AS 
                SELECT race_id, track_condition, weather, url, extraction_timestamp
                FROM race_metadata 
                WHERE race_id IN ({','.join(['?' for _ in race_ids_to_clean])})
            """
            cursor.execute(backup_query, race_ids_to_clean)
            
            # Clear false positive track conditions
            update_query = """
                UPDATE race_metadata 
                SET track_condition = NULL, 
                    data_quality_note = CASE 
                        WHEN data_quality_note IS NULL OR data_quality_note = '' 
                        THEN 'Track condition cleared (false positive)'
                        ELSE data_quality_note || '; Track condition cleared (false positive)'
                    END
                WHERE race_id IN ({})
            """.format(','.join(['?' for _ in race_ids_to_clean]))
            
            cursor.execute(update_query, race_ids_to_clean)
            
            conn.commit()
            print(f"\n✅ Cleaned {len(race_ids_to_clean)} false positive track conditions")
            print(f"   Backup created: track_condition_backup_{timestamp}")
        
        conn.close()
        return analysis
    
    def restore_from_backup(self, backup_timestamp: str):
        """Restore track conditions from a backup table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        backup_table = f"track_condition_backup_{backup_timestamp}"
        
        # Check if backup exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (backup_table,))
        
        if not cursor.fetchone():
            print(f"❌ Backup table {backup_table} not found")
            conn.close()
            return
        
        # Restore from backup
        restore_query = f"""
            UPDATE race_metadata 
            SET track_condition = b.track_condition,
                weather = b.weather
            FROM {backup_table} b
            WHERE race_metadata.race_id = b.race_id
        """
        
        cursor.execute(restore_query)
        conn.commit()
        
        rows_affected = cursor.rowcount
        print(f"✅ Restored {rows_affected} track conditions from backup {backup_timestamp}")
        
        conn.close()

def main():
    cleaner = TrackConditionCleaner()
    
    print("🔍 Analyzing track condition data for cleanup...")
    
    # First, run analysis in dry-run mode
    analysis = cleaner.clean_false_positives(dry_run=True)
    
    if analysis['false_positives']:
        print(f"\n❓ Found {len(analysis['false_positives'])} false positives.")
        response = input("Do you want to proceed with cleanup? (y/N): ").strip().lower()
        
        if response == 'y':
            print("\n🧹 Performing cleanup...")
            cleaner.clean_false_positives(dry_run=False)
        else:
            print("❌ Cleanup cancelled")
    else:
        print("\n✅ No false positives found - database is clean!")

if __name__ == "__main__":
    main()
