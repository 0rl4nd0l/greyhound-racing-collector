#!/usr/bin/env python3
"""
Bulk CSV Ingestion Script
========================

This script processes all CSV files in the form_guides/downloaded directory
and saves them to the database using batch processing for improved performance.

Key features:
- Batch processing with configurable BATCH_SIZE
- Pre-computed metadata for efficient filtering
- Database insert optimization with executemany()
- Hash-based duplicate detection
- File content validation (HTML detection, size checks)
"""

import os
import time
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterator
from csv_ingestion import FormGuideCsvIngestor, FormGuideCsvIngestionError

# Import early-exit optimization utilities
from utils.early_exit_optimizer import (
    create_early_exit_optimizer, 
    check_directory_for_early_exit,
    EarlyExitConfig
)
from utils.caching_utils import get_processed_filenames
from utils.race_file_utils import RaceFileManager, compute_file_hash
from utils.file_content_validator import FileContentValidator

# Configuration
BATCH_SIZE = 100  # Configurable batch size for processing

def chunked(iterable: List, size: int) -> Iterator[List]:
    """
    Yield successive chunks of specified size from iterable.
    
    Args:
        iterable: List to chunk
        size: Size of each chunk
        
    Yields:
        Lists of size 'size' (last chunk may be smaller)
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def compute_needed_info(batch: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compute metadata needed for batch processing (hash and file stats) using shared utilities.
    
    Args:
        batch: List of file paths to analyze
        
    Returns:
        Dictionary mapping file paths to their metadata
    """
    metadata = {}
    
    for file_path in batch:
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                # Use shared compute_file_hash utility for consistency
                file_hash = compute_file_hash(file_path)
                
                metadata[file_path] = {
                    'hash': file_hash,
                    'size': path_obj.stat().st_size,
                    'mtime': path_obj.stat().st_mtime,
                    'exists': True
                }
            else:
                metadata[file_path] = {
                    'hash': None,
                    'size': 0,
                    'mtime': 0,
                    'exists': False
                }
        except Exception as e:
            metadata[file_path] = {
                'hash': None,
                'size': 0,
                'mtime': 0,
                'exists': False,
                'error': str(e)
            }
    
    return metadata

def process_batch(batch: List[str], ingestor: FormGuideCsvIngestor,
                 metadata: Dict[str, Dict[str, Any]], validator: FileContentValidator = None) -> Dict[str, Any]:
    """
    Process a batch of CSV files using existing logic with file content validation.
    
    Args:
        batch: List of file paths to process
        ingestor: CSV ingestor instance
        metadata: Pre-computed file metadata
        validator: File content validator instance (optional)
        
    Returns:
        Dictionary with batch processing results
    """
    batch_results = {
        'processed_files': [],
        'failed_files': [],
        'skipped_files': [],  # Track files skipped due to validation
        'total_records': 0,
        'batch_data': []
    }
    
    # Initialize validator if not provided
    if validator is None:
        validator = FileContentValidator(min_file_size=100, log_skipped_files=True)
    
    for csv_file in batch:
        file_metadata = metadata.get(csv_file, {})
        
        # Skip non-existent files
        if not file_metadata.get('exists', False):
            batch_results['failed_files'].append({
                'file': csv_file,
                'error': 'File does not exist'
            })
            continue
        
        # Skip files with no hash (couldn't read)
        if not file_metadata.get('hash'):
            batch_results['failed_files'].append({
                'file': csv_file,
                'error': file_metadata.get('error', 'Could not compute file hash')
            })
            continue
            
        # Validate file content before processing
        is_valid, validation_message, file_info = validator.validate_file(csv_file)
        if not is_valid:
            batch_results['skipped_files'].append({
                'file': csv_file,
                'reason': validation_message,
                'file_info': file_info
            })
            continue
        
        try:
            # Process CSV file using existing logic
            processed_data, validation_result = ingestor.ingest_csv(csv_file)
            
            if processed_data:
                batch_results['processed_files'].append({
                    'file': csv_file,
                    'records': len(processed_data),
                    'hash': file_metadata['hash'],
                    'warnings': validation_result.warnings if hasattr(validation_result, 'warnings') else []
                })
                batch_results['total_records'] += len(processed_data)
                
                # Store processed data for batch database insert
                batch_results['batch_data'].extend(processed_data)
            else:
                batch_results['failed_files'].append({
                    'file': csv_file,
                    'error': 'No data extracted from file'
                })
                
        except FormGuideCsvIngestionError as e:
            batch_results['failed_files'].append({
                'file': csv_file,
                'error': f'Ingestion error: {str(e)[:100]}...'
            })
        except Exception as e:
            batch_results['failed_files'].append({
                'file': csv_file,
                'error': f'Unexpected error: {str(e)[:100]}...'
            })
    
    return batch_results

def batch_save_to_database(processed_data_list: List[Dict[str, Any]],
                          db_path: str = 'greyhound_racing_data.db') -> Tuple[int, int]:
    """
    Save processed data to database using executemany() for efficiency.
    
    Args:
        processed_data_list: List of processed race data dictionaries
        db_path: Path to database file
        
    Returns:
        Tuple of (races_saved, dogs_saved)
    """
    if not processed_data_list:
        return 0, 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Group records by race for batch processing
        races = {}
        for record in processed_data_list:
            # Generate race ID from available data
            track = record.get('track', 'unknown').lower()
            date = record.get('date', 'unknown')
            distance = record.get('distance', 'unknown')
            grade = record.get('grade', 'unknown')
            
            race_key = f"{track}_{date}_{distance}_{grade}"
            race_id = race_key.replace(' ', '_').replace('/', '_')
            
            if race_key not in races:
                races[race_key] = {
                    'race_id': race_id,
                    'venue': track.title(),
                    'race_date': date,
                    'distance': distance,
                    'grade': grade,
                    'dogs': []
                }
            
            # Add dog data to race
            races[race_key]['dogs'].append({
                'race_id': race_id,
                'dog_name': record.get('dog_name', ''),
                'dog_clean_name': record.get('dog_name', '').strip() if record.get('dog_name') else '',
                'box_number': record.get('box'),
                'finish_position': record.get('place'),
                'weight': record.get('weight'),
                'starting_price': record.get('starting_price'),
                'individual_time': record.get('time'),
                'sectional_1st': record.get('first_sectional'),
                'margin': record.get('margin'),
                'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': 'csv_batch_ingestion'
            })
        
        # Prepare batch data for executemany()
        race_batch_data = []
        dog_batch_data = []
        
        for race_key, race_data in races.items():
            # Prepare race metadata for batch insert
            race_batch_data.append((
                race_data['race_id'],
                race_data['venue'],
                race_data['race_date'],
                race_data['distance'],
                race_data['grade'],
                len(race_data['dogs']),
                time.strftime('%Y-%m-%d %H:%M:%S'),
                'csv_batch_ingestion'
            ))
            
            # Prepare dog data for batch insert
            for dog in race_data['dogs']:
                dog_batch_data.append((
                    dog['race_id'],
                    dog['dog_name'],
                    dog['dog_clean_name'],
                    dog['box_number'],
                    dog['finish_position'],
                    dog['weight'],
                    dog['starting_price'],
                    dog['individual_time'],
                    dog['sectional_1st'],
                    dog['margin'],
                    dog['extraction_timestamp'],
                    dog['data_source']
                ))
        
        # Execute batch inserts using executemany()
        races_saved = 0
        dogs_saved = 0
        
        if race_batch_data:
            cursor.executemany("""
                INSERT OR IGNORE INTO race_metadata 
                (race_id, venue, race_date, distance, grade, field_size, extraction_timestamp, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, race_batch_data)
            races_saved = cursor.rowcount
        
        if dog_batch_data:
            cursor.executemany("""
                INSERT OR IGNORE INTO dog_race_data 
                (race_id, dog_name, dog_clean_name, box_number, finish_position, weight, 
                 starting_price, individual_time, sectional_1st, margin, extraction_timestamp, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, dog_batch_data)
            dogs_saved = cursor.rowcount
        
        # Commit all changes
        conn.commit()
        
        return races_saved, dogs_saved
        
    except Exception as e:
        print(f"❌ Error in batch database save: {e}")
        conn.rollback()
        return 0, 0
    finally:
        conn.close()
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
    
    # Process files using batch processing approach
    start_time = time.time()
    
    with file_lock:
        print('🚀 Acquired file lock for processing.')
        print(f'📦 Using batch processing with BATCH_SIZE = {BATCH_SIZE}')

        # Replace per-file loop with batch processing
        batch_count = 0
        total_batches = (len(csv_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch in chunked(csv_files, BATCH_SIZE):
            batch_count += 1
            print(f'🔄 Processing batch {batch_count}/{total_batches} ({len(batch)} files)...')
            
            # Step 1: Compute needed info (hash or small stat calls)
            metadata = compute_needed_info(batch)
            
            # Step 2: Process batch (existing logic reused)
            results = process_batch(batch, ingestor, metadata)
            
            # Step 3: Batch database insert using executemany()
            if results['batch_data']:
                races_saved, dogs_saved = batch_save_to_database(results['batch_data'], 'greyhound_racing_data.db')
                stats['total_races_added'] += races_saved
                stats['total_dogs_added'] += dogs_saved
            
            # Update statistics
            stats['processed_files'] += len(results['processed_files'])
            stats['failed_files'] += len(results['failed_files'])
            stats['total_records'] += results['total_records']
            
            # Track skipped files if not already in stats
            if 'skipped_files' not in stats:
                stats['skipped_files'] = 0
            stats['skipped_files'] += len(results.get('skipped_files', []))
            
            # Add errors from this batch
            for failed_file in results['failed_files']:
                stats['errors'].append(f"{failed_file['file']}: {failed_file['error']}")
            
            # Log skipped files details if any
            if results.get('skipped_files'):
                for skipped_file in results['skipped_files']:
                    print(f"   ⚠️  Skipped: {os.path.basename(skipped_file['file'])} - {skipped_file['reason']}")
            
            # Show progress
            if batch_count % 5 == 0 or batch_count == total_batches:  # Every 5 batches or final batch
                elapsed = time.time() - start_time
                rate = stats['processed_files'] / (elapsed / 60) if elapsed > 0 else 0
                eta = (len(csv_files) - stats['processed_files']) / rate if rate > 0 else 0
                print(f'   Progress: {stats["processed_files"]:,}/{len(csv_files):,} '
                      f'({stats["processed_files"]/len(csv_files)*100:.1f}%) - '
                      f'Rate: {rate:.1f}/min - ETA: {eta:.0f}min')
                print(
                    f"   Batch results: {len(results['processed_files'])} processed, "
                    f"{len(results['failed_files'])} failed, "
                    f"{len(results.get('skipped_files', []))} skipped, "
                    f"{results['total_records']} records"
                )
            
            # Small pause between batches
            time.sleep(0.05)
    
    # Get final database counts for verification
    conn = sqlite3.connect('greyhound_racing_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM race_metadata')
    final_races = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM dog_race_data')
    final_dogs = cursor.fetchone()[0]
    conn.close()
    
    # Update final stats (batch processing already tracked additions)
    final_races_added = final_races - initial_races
    final_dogs_added = final_dogs - initial_dogs
    
    # Verify our batch tracking was accurate
    if abs(stats['total_races_added'] - final_races_added) > 10:  # Allow small discrepancy for concurrency
        print(f"⚠️ Race count discrepancy: tracked {stats['total_races_added']}, actual {final_races_added}")
        stats['total_races_added'] = final_races_added
    
    if abs(stats['total_dogs_added'] - final_dogs_added) > 10:
        print(f"⚠️ Dog count discrepancy: tracked {stats['total_dogs_added']}, actual {final_dogs_added}")
        stats['total_dogs_added'] = final_dogs_added
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f'\n' + '='*60)
    print(f'✅ BULK CSV INGESTION COMPLETE!')
    print(f'='*60)
    print(f'⏱️  Total time: {elapsed_time/60:.1f} minutes')
    print(f'📁 Files processed: {stats["processed_files"]:,}/{len(csv_files):,}')
    print(f'❌ Files failed: {stats["failed_files"]:,}')
    if stats.get('skipped_files', 0) > 0:
        print(f'⚠️  Files skipped: {stats["skipped_files"]:,} (HTML/empty/too small)')
    print(f'📊 Total records processed: {stats["total_records"]:,}')
    print(f'🏁 New races added: {stats["total_races_added"]:,}')
    print(f'🐕 New dog records added: {stats["total_dogs_added"]:,}')
    print(f'📈 Processing rate: {stats["processed_files"]/(elapsed_time/60):.1f} files/min')
    
    if stats['errors']:
        print(f'\n⚠️  Sample errors ({min(5, len(stats["errors"]))} of {len(stats["errors"])}):')
        for error in stats['errors'][:5]:
            print(f'   - {error}')
    
    return stats


def bulk_ingest_with_early_exit_optimization(
    cache_ratio_threshold: float = 0.95,
    unprocessed_threshold: int = 5,
    enable_early_exit: bool = True
):
    """
    Bulk ingest CSV files with early-exit optimization for mostly cached directories.
    
    This function implements Step 6 of the optimization plan: Early-exit strategy
    for "mostly cached" directories. If pre-filtering removes ≥95% of files AND
    unprocessed count < threshold, it prints summary and returns immediately.
    
    Args:
        cache_ratio_threshold: Minimum cache ratio for early exit (default: 0.95)
        unprocessed_threshold: Maximum unprocessed files for early exit (default: 5)
        enable_early_exit: Master switch for early exit functionality (default: True)
    """
    print('🚀 Starting optimized CSV ingestion with early-exit strategy...')
    
    # Configuration
    directory = 'form_guides/downloaded'
    db_path = 'greyhound_racing_data.db'
    
    # Get processed files set for O(1) lookups
    print('📋 Loading processed files cache for fast lookups...')
    processed_files_set = get_processed_filenames(directory, db_path)
    print(f'✅ Loaded {len(processed_files_set):,} processed files into cache')
    
    # Initialize early-exit optimizer
    optimizer = create_early_exit_optimizer(
        cache_ratio_threshold=cache_ratio_threshold,
        unprocessed_threshold=unprocessed_threshold,
        enable_early_exit=enable_early_exit,
        verbose_summary=True
    )
    
    # Check if directory qualifies for early exit
    should_early_exit, unprocessed_files = check_directory_for_early_exit(
        directory, 
        processed_files_set, 
        cache_ratio_threshold, 
        unprocessed_threshold
    )
    
    # Early exit path - skip detailed progress printing
    if should_early_exit:
        print(f"\n💨 Early-exit optimization active - processing {len(unprocessed_files)} files quickly")
        
        if not unprocessed_files:
            print("✅ No unprocessed files found - directory is fully cached!")
            return {
                'processed_files': 0,
                'failed_files': 0,
                'total_records': 0,
                'early_exit_triggered': True,
                'cache_hit_ratio': 1.0
            }
        
        # Process only the few unprocessed files without detailed progress
        return _process_files_fast(unprocessed_files, db_path, early_exit=True)
    
    # Normal processing path for directories with many unprocessed files
    print(f"\n⏳ Directory has too many unprocessed files for early exit - using normal processing")
    
    # Find all CSV files for normal processing
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f'📊 Found {len(csv_files):,} total CSV files')
    
    # Filter out already processed files
    unprocessed_csv_files = [
        file_path for file_path in csv_files 
        if os.path.basename(file_path) not in processed_files_set
    ]
    
    print(f'🔄 Files requiring processing: {len(unprocessed_csv_files):,}')
    
    if not unprocessed_csv_files:
        print("✅ All files have been processed previously!")
        return {
            'processed_files': 0,
            'failed_files': 0,
            'total_records': 0,
            'early_exit_triggered': False,
            'cache_hit_ratio': 1.0
        }
    
    # Process unprocessed files using normal batch processing
    return _process_files_fast(unprocessed_csv_files, db_path, early_exit=False)


def _process_files_fast(
    files_to_process: List[str], 
    db_path: str, 
    early_exit: bool = False
) -> Dict[str, Any]:
    """
    Internal function to process files efficiently.
    
    Args:
        files_to_process: List of file paths to process
        db_path: Database path
        early_exit: Whether this is an early-exit scenario (affects logging)
    
    Returns:
        Processing statistics dictionary
    """
    import filelock
    
    # Initialize ingestor
    ingestor = FormGuideCsvIngestor(db_path=db_path)
    
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
        'errors': [],
        'early_exit_triggered': early_exit
    }
    
    # Get initial database counts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM race_metadata')
    initial_races = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM dog_race_data')
    initial_dogs = cursor.fetchone()[0]
    conn.close()
    
    start_time = time.time()
    
    with file_lock:
        if early_exit:
            print(f'⚡ Processing {len(files_to_process)} files with early-exit optimization')
        else:
            print(f'🔄 Processing {len(files_to_process):,} unprocessed files with batch optimization')
            print(f'📦 Using batch processing with BATCH_SIZE = {BATCH_SIZE}')
        
        batch_count = 0
        total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch in chunked(files_to_process, BATCH_SIZE):
            batch_count += 1
            
            # For early exit, suppress detailed batch progress
            if not early_exit:
                print(f'🔄 Processing batch {batch_count}/{total_batches} ({len(batch)} files)...')
            
            # Step 1: Compute needed info
            metadata = compute_needed_info(batch)
            
            # Step 2: Process batch
            results = process_batch(batch, ingestor, metadata)
            
            # Step 3: Batch database insert
            if results['batch_data']:
                races_saved, dogs_saved = batch_save_to_database(results['batch_data'], db_path)
                stats['total_races_added'] += races_saved
                stats['total_dogs_added'] += dogs_saved
            
            # Update statistics
            stats['processed_files'] += len(results['processed_files'])
            stats['failed_files'] += len(results['failed_files'])
            stats['total_records'] += results['total_records']
            
            # Add errors from this batch
            for failed_file in results['failed_files']:
                stats['errors'].append(f"{failed_file['file']}: {failed_file['error']}")
            
            # Show progress (suppressed in early-exit mode)
            if not early_exit and (batch_count % 5 == 0 or batch_count == total_batches):
                elapsed = time.time() - start_time
                rate = stats['processed_files'] / (elapsed / 60) if elapsed > 0 else 0
                eta = (len(files_to_process) - stats['processed_files']) / rate if rate > 0 else 0
                print(f'   Progress: {stats["processed_files"]:,}/{len(files_to_process):,} '
                      f'({stats["processed_files"]/len(files_to_process)*100:.1f}%) - '
                      f'Rate: {rate:.1f}/min - ETA: {eta:.0f}min')
                print(
                    f"   Batch results: {len(results['processed_files'])} processed, "
                    f"{len(results['failed_files'])} failed, "
                    f"{results['total_records']} records"
                )
            
            # Minimal pause between batches
            time.sleep(0.01 if early_exit else 0.05)
    
    # Get final database counts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM race_metadata')
    final_races = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM dog_race_data')
    final_dogs = cursor.fetchone()[0]
    conn.close()
    
    # Calculate final additions
    stats['total_races_added'] = final_races - initial_races
    stats['total_dogs_added'] = final_dogs - initial_dogs
    
    # Print results (concise for early exit)
    elapsed_time = time.time() - start_time
    
    if early_exit:
        print(f'\n⚡ EARLY-EXIT PROCESSING COMPLETE!')
        print(f'⏱️  Time: {elapsed_time:.2f} seconds')
        print(f'📁 Files: {stats["processed_files"]:,} processed, {stats["failed_files"]:,} failed')
        print(f'🏁 Added: {stats["total_races_added"]:,} races, {stats["total_dogs_added"]:,} dog records')
    else:
        print(f'\n' + '='*60)
        print(f'✅ OPTIMIZED CSV INGESTION COMPLETE!')
        print(f'='*60)
        print(f'⏱️  Total time: {elapsed_time/60:.1f} minutes')
        print(f'📁 Files processed: {stats["processed_files"]:,}/{len(files_to_process):,}')
        print(f'❌ Files failed: {stats["failed_files"]:,}')
        print(f'📊 Total records processed: {stats["total_records"]:,}')
        print(f'🏁 New races added: {stats["total_races_added"]:,}')
        print(f'🐕 New dog records added: {stats["total_dogs_added"]:,}')
        print(f'📈 Processing rate: {stats["processed_files"]/(elapsed_time/60):.1f} files/min')
    
    if stats['errors'] and not early_exit:
        print(f'\n⚠️  Sample errors ({min(5, len(stats["errors"]))} of {len(stats["errors"])}):')
        for error in stats['errors'][:5]:
            print(f'   - {error}')
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk CSV ingestion with optimization")
    parser.add_argument('--cache-ratio', type=float, default=0.95,
                       help='Cache ratio threshold for early exit (default: 0.95)')
    parser.add_argument('--unprocessed-threshold', type=int, default=5,
                       help='Max unprocessed files for early exit (default: 5)')
    parser.add_argument('--disable-early-exit', action='store_true',
                       help='Disable early exit optimization')
    parser.add_argument('--legacy-mode', action='store_true',
                       help='Use legacy bulk ingestion without optimization')
    
    args = parser.parse_args()
    
    if args.legacy_mode:
        print("🔄 Running in legacy mode without optimization")
        bulk_ingest_with_database_save()
    else:
        bulk_ingest_with_early_exit_optimization(
            cache_ratio_threshold=args.cache_ratio,
            unprocessed_threshold=args.unprocessed_threshold,
            enable_early_exit=not args.disable_early_exit
        )
