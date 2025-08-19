#!/usr/bin/env python3
"""
Cleanup Duplicate Predictions Script
====================================

This script identifies and removes duplicate prediction files, keeping only the most recent
version of each unique race prediction.

Author: AI Assistant
Date: July 26, 2025
"""

import os
import json
import shutil
from datetime import datetime
from collections import defaultdict

def cleanup_duplicate_predictions():
    """Clean up duplicate prediction files"""
    predictions_dir = './predictions'
    
    if not os.path.exists(predictions_dir):
        print("No predictions directory found")
        return
    
    print("🧹 Starting prediction cleanup...")
    
    # Group files by race identifier
    race_groups = defaultdict(list)
    
    # Get all prediction files
    for filename in os.listdir(predictions_dir):
        if not filename.endswith('.json') or 'summary' in filename:
            continue
            
        # Skip backup files
        if filename.endswith('.backup'):
            continue
            
        # Skip files that don't match prediction patterns
        if not (filename.startswith('prediction_') or 
                filename.startswith('unified_prediction_') or 
                filename.startswith('comprehensive_prediction_')):
            continue
        
        try:
            file_path = os.path.join(predictions_dir, filename)
            
            # Read the prediction file to get race info
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract race identifier
            race_info = data.get('race_info', {})
            race_context = data.get('race_context', {})
            
            # Get race filename or create identifier
            race_filename = race_info.get('filename', '')
            if not race_filename:
                # Try to construct from race_context
                venue = race_context.get('venue', 'Unknown')
                race_date = race_context.get('race_date', 'Unknown')
                race_number = race_context.get('race_number', 'Unknown')
                race_filename = f"Race {race_number} - {venue} - {race_date}.csv"
            
            # Remove .csv extension for grouping
            race_id = race_filename.replace('.csv', '')
            
            # Get file modification time
            file_mtime = os.path.getmtime(file_path)
            
            race_groups[race_id].append({
                'filename': filename,
                'filepath': file_path,
                'mtime': file_mtime,
                'data': data
            })
            
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"⚠️  Error reading {filename}: {e}")
            continue
    
    # Process each race group
    total_files = 0
    kept_files = 0
    removed_files = 0
    backup_dir = os.path.join(predictions_dir, 'cleanup_backup')
    
    for race_id, files in race_groups.items():
        total_files += len(files)
        
        if len(files) <= 1:
            kept_files += len(files)
            continue
        
        print(f"📊 Race: {race_id} - Found {len(files)} duplicates")
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Keep the most recent file
        latest_file = files[0]
        duplicates = files[1:]
        
        print(f"  ✅ Keeping: {latest_file['filename']}")
        kept_files += 1
        
        # Create backup directory if needed
        if duplicates and not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Move duplicates to backup
        for duplicate in duplicates:
            backup_path = os.path.join(backup_dir, duplicate['filename'])
            try:
                shutil.move(duplicate['filepath'], backup_path)
                print(f"  🗑️  Moved to backup: {duplicate['filename']}")
                removed_files += 1
            except Exception as e:
                print(f"  ❌ Error moving {duplicate['filename']}: {e}")
    
    print(f"\n📈 Cleanup Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files kept: {kept_files}")
    print(f"  Files moved to backup: {removed_files}")
    
    if removed_files > 0:
        print(f"  Backup location: {backup_dir}")
        print(f"  💡 You can safely delete the backup folder after verifying the cleanup")

def consolidate_naming_conventions():
    """Consolidate different naming conventions to a standard format"""
    predictions_dir = './predictions'
    
    if not os.path.exists(predictions_dir):
        return
    
    print("\n🔄 Consolidating naming conventions...")
    
    renamed_count = 0
    
    for filename in os.listdir(predictions_dir):
        if not filename.endswith('.json') or 'summary' in filename:
            continue
        
        old_path = os.path.join(predictions_dir, filename)
        
        # Skip if already in standard format
        if filename.startswith('prediction_') and not filename.startswith('prediction_Race'):
            continue
        
        try:
            # Read file to get race info
            with open(old_path, 'r') as f:
                data = json.load(f)
            
            race_info = data.get('race_info', {})
            race_filename = race_info.get('filename', '')
            
            if race_filename and race_filename.endswith('.csv'):
                # Create standardized name
                race_id = race_filename.replace('.csv', '')
                new_filename = f"prediction_{race_id}.json"
                new_path = os.path.join(predictions_dir, new_filename)
                
                # Only rename if the new name doesn't exist
                if not os.path.exists(new_path) and filename != new_filename:
                    shutil.move(old_path, new_path)
                    print(f"  📝 Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                    
        except Exception as e:
            print(f"  ⚠️  Error processing {filename}: {e}")
            continue
    
    print(f"  Renamed {renamed_count} files to standard format")

if __name__ == "__main__":
    cleanup_duplicate_predictions()
    consolidate_naming_conventions()
    print("\n✅ Cleanup completed!")
