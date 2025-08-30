#!/usr/bin/env python3
"""
Winner Name Data Quality Fix
Repairs missing winner names by cross-referencing with dog race data
"""

import sqlite3
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_winner_names():
    """Fix missing winner names in race_metadata table"""
    
    conn = sqlite3.connect('greyhound_racing_data.db')
    cursor = conn.cursor()
    
    logger.info("🔧 Starting winner name repair process...")
    
    # Step 1: Try to fix using box_analysis JSON data
    logger.info("Step 1: Extracting winners from box_analysis data...")
    
    cursor.execute('''
        SELECT id, race_id, box_analysis
        FROM race_metadata 
        WHERE box_analysis IS NOT NULL 
        AND box_analysis != '' 
        AND box_analysis != 'null'
        AND (winner_name IS NULL OR winner_name = '')
    ''')
    
    box_analysis_races = cursor.fetchall()
    fixed_from_box_analysis = 0
    
    for race_id_pk, race_id, box_analysis in box_analysis_races:
        try:
            box_data = json.loads(box_analysis)
            winner_name = None
            
            # Find the winner in the box analysis
            for box, details in box_data.items():
                if details.get('was_winner'):
                    winner_name = details.get('dog_name')
                    break
            
            if winner_name:
                cursor.execute('''
                    UPDATE race_metadata 
                    SET winner_name = ?, 
                        data_quality_note = COALESCE(data_quality_note, '') || '; Winner extracted from box_analysis'
                    WHERE id = ?
                ''', (winner_name, race_id_pk))
                fixed_from_box_analysis += 1
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in box_analysis for race {race_id}")
    
    logger.info(f"✅ Fixed {fixed_from_box_analysis} races using box_analysis data")
    
    # Step 2: Try to fix using dog_race_data where finish_position = '1' or '1='
    logger.info("Step 2: Cross-referencing with dog race data for position 1 finishers...")
    
    cursor.execute('''
        SELECT DISTINCT rm.id, rm.race_id
        FROM race_metadata rm
        WHERE (rm.winner_name IS NULL OR rm.winner_name = '')
        AND rm.data_source = 'csv_ingestion'
    ''')
    
    races_to_fix = cursor.fetchall()
    fixed_from_dog_data = 0
    
    for race_id_pk, race_id in races_to_fix:
        # Look for dogs with position 1 in various formats
        cursor.execute('''
            SELECT DISTINCT dog_clean_name, finish_position
            FROM dog_race_data
            WHERE race_id = ? 
            AND (finish_position = '1' OR finish_position = '1=' OR finish_position = 1)
            LIMIT 1
        ''', (race_id,))
        
        winner_result = cursor.fetchone()
        
        if winner_result:
            winner_name, position = winner_result
            cursor.execute('''
                UPDATE race_metadata 
                SET winner_name = ?,
                    data_quality_note = COALESCE(data_quality_note, '') || '; Winner extracted from dog_race_data position ' || ?
                WHERE id = ?
            ''', (winner_name, position, race_id_pk))
            fixed_from_dog_data += 1
    
    logger.info(f"✅ Fixed {fixed_from_dog_data} races using dog race data")
    
    # Step 3: Statistical report
    cursor.execute('''
        SELECT 
            COUNT(*) as total_races,
            COUNT(CASE WHEN winner_name IS NOT NULL AND winner_name != '' THEN 1 END) as races_with_winners,
            COUNT(CASE WHEN data_source = 'csv_ingestion' THEN 1 END) as csv_races,
            COUNT(CASE WHEN data_source = 'csv_ingestion' AND winner_name IS NOT NULL AND winner_name != '' THEN 1 END) as csv_with_winners
        FROM race_metadata
    ''')
    
    stats = cursor.fetchone()
    total_races, races_with_winners, csv_races, csv_with_winners = stats
    
    winner_coverage = (races_with_winners / total_races) * 100 if total_races > 0 else 0
    csv_coverage = (csv_with_winners / csv_races) * 100 if csv_races > 0 else 0
    
    conn.commit()
    conn.close()
    
    # Final report
    logger.info("📊 Winner Name Repair Complete!")
    logger.info(f"Total fixes applied: {fixed_from_box_analysis + fixed_from_dog_data}")
    logger.info(f"From box_analysis: {fixed_from_box_analysis}")
    logger.info(f"From dog_race_data: {fixed_from_dog_data}")
    logger.info(f"Overall winner coverage: {winner_coverage:.1f}% ({races_with_winners}/{total_races})")
    logger.info(f"CSV ingestion winner coverage: {csv_coverage:.1f}% ({csv_with_winners}/{csv_races})")
    
    return fixed_from_box_analysis + fixed_from_dog_data

if __name__ == "__main__":
    total_fixed = fix_winner_names()
    print(f"\n🎯 Successfully fixed {total_fixed} winner names!")
