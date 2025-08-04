#!/usr/bin/env python3
"""
Bulk CSV Ingestion Script
========================

This script processes all CSV files in the form_guides/downloaded directory
and saves them to the database properly.
"""

import os
import time
import sqlite3
from pathlib import Path
from csv_ingestion import FormGuideCsvIngestor, save_to_database, FormGuideCsvIngestionError

def bulk_ingest_with_database_save():
    """
    Bulk ingest all CSV files and save to database.
    """
    print('🚀 Starting comprehensive CSV ingestion with database saves...')
    
    # Find all CSV files
    csv_files = []
    for root, _, files in os.walk('form_guides/downloaded'):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f'📊 Found {len(csv_files):,} CSV files to process')
    
    import filelock

    # Initialize ingestor
    ingestor = FormGuideCsvIngestor(db_path='greyhound_racing_data.db')

    # Prepare file lock
    lock_file = './csv_ingestion.lock'
    file_lock = filelock.FileLock(lock_file, timeout=10)
    
    # Track statistics
    stats = {
        'processed_files': 0,
        'failed_files': 0,
        'total_records': 0,
        'total_races_added': 0,
        'total_dogs_added': 0,
        'errors': []
    }
    
    # Get initial database counts
    conn = sqlite3.connect('greyhound_racing_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM race_metadata')
    initial_races = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM dog_race_data')
    initial_dogs = cursor.fetchone()[0]
    conn.close()
    
    print(f'📊 Initial database state: {initial_races:,} races, {initial_dogs:,} dog records')
    
    # Process files in batches
    batch_size = 100
    start_time = time.time()
    
    with file_lock:
        print('🚀 Acquired file lock for processing.')

        for i in range(0, len(csv_files), batch_size):
            batch = csv_files[i:i+batch_size]

            print(f'🔄 Processing batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size}...')

            for csv_file in batch:
                try:
                    # Step 1: Process CSV file
                    processed_data, validation_result = ingestor.ingest_csv(csv_file)
                    
                    if processed_data:
                        # Step 2: Save to database
                        save_to_database(processed_data, 'greyhound_racing_data.db')
                        
                        stats['processed_files'] += 1
                        stats['total_records'] += len(processed_data)
                        
                        # Show progress every 50 files
                        if stats['processed_files'] % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = stats['processed_files'] / (elapsed / 60) if elapsed > 0 else 0
                            eta = (len(csv_files) - stats['processed_files']) / rate if rate > 0 else 0
                            print(f'   Progress: {stats["processed_files"]:,}/{len(csv_files):,} '
                                  f'({stats["processed_files"]/len(csv_files)*100:.1f}%) - '
                                  f'Rate: {rate:.1f}/min - ETA: {eta:.0f}min')
                    else:
                        print(f'   ⚠️ No data extracted from {csv_file}')
                        
                except FormGuideCsvIngestionError as e:
                    stats['failed_files'] += 1
                    stats['errors'].append(f'{csv_file}: {str(e)[:100]}...')
                    continue
                except Exception as e:
                    stats['failed_files'] += 1
                    stats['errors'].append(f'{csv_file}: Unexpected error - {str(e)[:100]}...')
                    continue
            
            # Small pause between batches
            time.sleep(0.05)
    
    # Get final database counts
    conn = sqlite3.connect('greyhound_racing_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM race_metadata')
    final_races = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM dog_race_data')
    final_dogs = cursor.fetchone()[0]
    conn.close()
    
    stats['total_races_added'] = final_races - initial_races
    stats['total_dogs_added'] = final_dogs - initial_dogs
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f'\n' + '='*60)
    print(f'✅ BULK CSV INGESTION COMPLETE!')
    print(f'='*60)
    print(f'⏱️  Total time: {elapsed_time/60:.1f} minutes')
    print(f'📁 Files processed: {stats["processed_files"]:,}/{len(csv_files):,}')
    print(f'❌ Files failed: {stats["failed_files"]:,}')
    print(f'📊 Total records processed: {stats["total_records"]:,}')
    print(f'🏁 New races added: {stats["total_races_added"]:,}')
    print(f'🐕 New dog records added: {stats["total_dogs_added"]:,}')
    print(f'📈 Processing rate: {stats["processed_files"]/(elapsed_time/60):.1f} files/min')
    
    if stats['errors']:
        print(f'\n⚠️  Sample errors ({min(5, len(stats["errors"]))} of {len(stats["errors"])}):')
        for error in stats['errors'][:5]:
            print(f'   - {error}')
    
    return stats

if __name__ == '__main__':
    bulk_ingest_with_database_save()
