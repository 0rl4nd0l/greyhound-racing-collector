#!/usr/bin/env python3
"""
Backup System Verification and Cleanup
Ensures data backup integrity and removes duplicate files.
"""

import os
import shutil
import hashlib
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def find_duplicate_files(base_path):
    """Find duplicate files across the system"""
    print("🔍 Scanning for duplicate files...")
    
    file_hashes = defaultdict(list)
    
    # Scan all processed directories
    for root, dirs, files in os.walk(os.path.join(base_path, "processed")):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    file_hashes[file_hash].append(file_path)
    
    # Find duplicates
    duplicates = []
    for hash_val, files in file_hashes.items():
        if len(files) > 1:
            duplicates.append(files)
    
    return duplicates

def clean_duplicate_files(duplicates, base_path):
    """Clean up duplicate files, keeping the best copy"""
    print("🧹 Cleaning duplicate files...")
    
    cleaned_files = []
    backup_dir = os.path.join(base_path, "duplicate_cleanup_backup")
    
    if duplicates and not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    for duplicate_group in duplicates:
        print(f"\n📁 Processing duplicate group:")
        for file_path in duplicate_group:
            print(f"   - {file_path}")
        
        # Determine which file to keep (prefer 'completed' over others)
        keep_file = None
        remove_files = []
        
        for file_path in duplicate_group:
            if '/completed/' in file_path:
                if keep_file is None:
                    keep_file = file_path
                else:
                    remove_files.append(file_path)
            else:
                remove_files.append(file_path)
        
        # If no completed file, keep the first one
        if keep_file is None:
            keep_file = duplicate_group[0]
            remove_files = duplicate_group[1:]
        
        print(f"   ✅ Keeping: {keep_file}")
        
        # Move duplicates to backup before removing
        for remove_file in remove_files:
            try:
                # Create backup copy
                backup_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(remove_file)}"
                backup_path = os.path.join(backup_dir, backup_filename)
                shutil.copy2(remove_file, backup_path)
                
                # Remove original
                os.remove(remove_file)
                cleaned_files.append(remove_file)
                print(f"   🗑️  Removed: {remove_file} (backed up to {backup_filename})")
                
            except Exception as e:
                print(f"   ❌ Error removing {remove_file}: {e}")
    
    return cleaned_files

def verify_backup_integrity(base_path):
    """Verify backup directory integrity"""
    print("\n💾 Verifying backup integrity...")
    
    backup_dirs = ['cached_backup', 'backup_before_cleanup', 'predictions']
    backup_stats = {}
    
    for backup_dir in backup_dirs:
        backup_path = os.path.join(base_path, backup_dir)
        if os.path.exists(backup_path):
            files = []
            total_size = 0
            
            for root, dirs, filenames in os.walk(backup_path):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    try:
                        file_size = os.path.getsize(filepath)
                        files.append({
                            'name': filename,
                            'path': filepath,
                            'size': file_size,
                            'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                        })
                        total_size += file_size
                    except Exception as e:
                        print(f"Error accessing {filepath}: {e}")
            
            backup_stats[backup_dir] = {
                'files': len(files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_file': min(files, key=lambda x: x['modified'])['modified'] if files else None,
                'newest_file': max(files, key=lambda x: x['modified'])['modified'] if files else None
            }
            
            print(f"📂 {backup_dir}:")
            print(f"   - Files: {backup_stats[backup_dir]['files']}")
            print(f"   - Total size: {backup_stats[backup_dir]['total_size_mb']} MB")
            print(f"   - Date range: {backup_stats[backup_dir]['oldest_file']} to {backup_stats[backup_dir]['newest_file']}")
        else:
            print(f"⚠️  Backup directory {backup_dir} not found")
            backup_stats[backup_dir] = None
    
    return backup_stats

def create_system_backup(base_path):
    """Create a comprehensive system backup"""
    print("\n📦 Creating comprehensive system backup...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"system_backup_{timestamp}"
    backup_path = os.path.join(base_path, backup_name)
    
    try:
        os.makedirs(backup_path)
        
        # Backup critical directories
        critical_dirs = [
            'processed',
            'predictions', 
            'databases',
            'form_guides'
        ]
        
        backup_manifest = {
            'timestamp': timestamp,
            'backup_name': backup_name,
            'directories': {}
        }
        
        for dir_name in critical_dirs:
            source_path = os.path.join(base_path, dir_name)
            if os.path.exists(source_path):
                dest_path = os.path.join(backup_path, dir_name)
                print(f"   📁 Backing up {dir_name}...")
                
                # Copy directory
                shutil.copytree(source_path, dest_path)
                
                # Count files
                file_count = sum([len(files) for r, d, files in os.walk(dest_path)])
                backup_manifest['directories'][dir_name] = {
                    'files': file_count,
                    'status': 'completed'
                }
                print(f"   ✅ {dir_name}: {file_count} files backed up")
            else:
                backup_manifest['directories'][dir_name] = {
                    'files': 0,
                    'status': 'not_found'
                }
                print(f"   ⚠️  {dir_name}: directory not found")
        
        # Save manifest
        manifest_path = os.path.join(backup_path, 'backup_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(backup_manifest, f, indent=2, default=str)
        
        print(f"✅ System backup completed: {backup_name}")
        return backup_path
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None

def validate_data_consistency(base_path):
    """Validate data consistency across directories"""
    print("\n⚖️ Validating data consistency...")
    
    issues = []
    
    # Check processed directories for consistency
    processed_base = os.path.join(base_path, "processed")
    subdirs = ['completed', 'excluded', 'other']
    
    all_files = {}
    
    for subdir in subdirs:
        subdir_path = os.path.join(processed_base, subdir)
        if os.path.exists(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.csv'):
                    if filename in all_files:
                        issues.append(f"File {filename} exists in both {all_files[filename]} and {subdir}")
                    else:
                        all_files[filename] = subdir
    
    # Check for files with consistent data format
    sample_files = list(all_files.keys())[:10]  # Sample first 10 files
    
    for filename in sample_files:
        subdir = all_files[filename]
        file_path = os.path.join(processed_base, subdir, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            # Check basic requirements
            if df.empty:
                issues.append(f"File {filename} is empty")
            
            if len(df.columns) < 5:  # Expect reasonable number of columns
                issues.append(f"File {filename} has only {len(df.columns)} columns")
                
        except Exception as e:
            issues.append(f"Cannot read file {filename}: {str(e)}")
    
    if issues:
        print("⚠️  Data consistency issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Data consistency validation passed")
    
    return issues

def main():
    """Main function to run comprehensive data integrity verification"""
    base_path = "/Users/orlandolee/greyhound_racing_collector"
    
    print("🚀 Starting Backup System Verification & Cleanup")
    print("=" * 60)
    
    # 1. Find and clean duplicate files
    duplicates = find_duplicate_files(base_path)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate file groups")
        cleaned_files = clean_duplicate_files(duplicates, base_path)
        print(f"✅ Cleaned {len(cleaned_files)} duplicate files")
    else:
        print("✅ No duplicate files found")
    
    # 2. Verify backup integrity
    backup_stats = verify_backup_integrity(base_path)
    
    # 3. Validate data consistency
    consistency_issues = validate_data_consistency(base_path)
    
    # 4. Create fresh system backup
    backup_path = create_system_backup(base_path)
    
    # 5. Generate summary report
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'duplicates_found': len(duplicates) if duplicates else 0,
        'duplicates_cleaned': len(cleaned_files) if 'cleaned_files' in locals() else 0,
        'backup_stats': backup_stats,
        'consistency_issues': len(consistency_issues),
        'system_backup': backup_path,
        'status': 'PASSED' if len(consistency_issues) == 0 else 'ISSUES_FOUND'
    }
    
    # Save report
    report_path = os.path.join(base_path, f"backup_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Report saved to: {report_path}")
    print(f"Overall Status: {report['status']}")
    
    if report['status'] == 'PASSED':
        print("✅ All verification checks passed!")
        return 0
    else:
        print("⚠️  Some issues were found - check the report for details")
        return 1

if __name__ == "__main__":
    exit(main())
