#!/usr/bin/env python3
"""
Cleanup Problematic Files
=========================
This script identifies and removes files with problematic naming patterns
that were generated during the automated renaming process.
"""

import os
import re
from pathlib import Path
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_problematic_file(filename):
    """Identify files with problematic naming patterns"""
    # Files with long chains of numbers/suffixes
    if re.search(r'_\d{2}_\d{2}_\d{2}', filename):
        return True
    
    # Files with excessive suffix chains
    if re.search(r'(_\d{2}){5,}', filename):
        return True
    
    # Non-racing data files that got renamed
    problem_indicators = [
        'umath-validation',
        'philox-testset',
        'sfc64-testset',
        'wine_data',
        'breast_cancer',
        'iris',
        'msft',
        'data_x_x2_x3'
    ]
    
    for indicator in problem_indicators:
        if indicator.lower() in filename.lower():
            return True
    
    return False

def cleanup_problematic_files(base_dir="."):
    """Clean up problematic files"""
    base_path = Path(base_dir)
    cleanup_archive = base_path / "cleanup_archive"
    
    # Create cleanup archive if it doesn't exist
    cleanup_archive.mkdir(exist_ok=True)
    
    moved_files = []
    removed_files = []
    
    # Find all problematic files
    for file_path in base_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.csv', '.json']:
            # Skip files already in backup or cleanup directories
            if any(skip_dir in str(file_path) for skip_dir in ['backup_before_cleanup', 'cleanup_archive', 'quarantine']):
                continue
            
            if is_problematic_file(file_path.name):
                try:
                    # If it's clearly not racing data, move to cleanup_archive
                    if any(indicator in file_path.name.lower() for indicator in ['umath', 'philox', 'sfc64', 'wine', 'breast', 'iris', 'msft', 'data_x']):
                        target_path = cleanup_archive / file_path.name
                        # If target exists, add a counter
                        counter = 1
                        while target_path.exists():
                            stem = file_path.stem
                            suffix = file_path.suffix
                            target_path = cleanup_archive / f"{stem}_{counter:03d}{suffix}"
                            counter += 1
                        
                        shutil.move(str(file_path), str(target_path))
                        moved_files.append((str(file_path), str(target_path)))
                        logger.info(f"Moved non-racing file: {file_path.name} -> {target_path.name}")
                    
                    # If it's a racing file with problematic suffixes, try to clean the name
                    elif file_path.name.startswith('Race_') and '_UNKNOWN_' in file_path.name:
                        # Extract the basic race info
                        match = re.match(r'Race_(\d{2})_UNKNOWN_(\d{4}-\d{2}-\d{2})', file_path.name)
                        if match:
                            race_num, date = match.groups()
                            clean_name = f"Race_{race_num}_UNKNOWN_{date}{file_path.suffix}"
                            clean_path = file_path.parent / clean_name
                            
                            # If clean name already exists, remove the problematic duplicate
                            if clean_path.exists() and clean_path != file_path:
                                file_path.unlink()
                                removed_files.append(str(file_path))
                                logger.info(f"Removed duplicate problematic file: {file_path.name}")
                            elif clean_path != file_path:
                                file_path.rename(clean_path)
                                logger.info(f"Cleaned filename: {file_path.name} -> {clean_name}")
                                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
    
    return moved_files, removed_files

def verify_data_integrity():
    """Verify that important racing data is still intact"""
    logger.info("Verifying data integrity...")
    
    # Check key directories
    key_dirs = ['race_data', 'upcoming_races', 'form_guides', 'processed_races']
    
    for dir_name in key_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            csv_count = len(list(dir_path.glob("*.csv")))
            logger.info(f"{dir_name}: {csv_count} CSV files")
        else:
            logger.warning(f"Directory {dir_name} not found")

if __name__ == "__main__":
    logger.info("Starting cleanup of problematic files...")
    
    moved, removed = cleanup_problematic_files()
    
    logger.info(f"Cleanup completed:")
    logger.info(f"  - Moved {len(moved)} non-racing files to cleanup_archive")
    logger.info(f"  - Removed {len(removed)} duplicate/problematic files")
    
    verify_data_integrity()
    
    logger.info("Cleanup process finished.")
