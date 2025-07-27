#!/usr/bin/env python3
"""
Utility script to manually move files from download to unprocessed directory
"""

import os
import shutil

def move_files_to_unprocessed():
    """Move CSV files from download directory to unprocessed directory"""
    
    download_dir = "./form_guides/downloaded"
    unprocessed_dir = "./unprocessed"
    
    # Create unprocessed directory if it doesn't exist
    os.makedirs(unprocessed_dir, exist_ok=True)
    
    if not os.path.exists(download_dir):
        print(f"❌ Download directory not found: {download_dir}")
        return
    
    # Get CSV files from download directory
    csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("⭕ No CSV files found in download directory")
        return
    
    print(f"📋 Found {len(csv_files)} CSV files in download directory")
    
    moved_count = 0
    skipped_count = 0
    
    for filename in csv_files:
        download_path = os.path.join(download_dir, filename)
        unprocessed_path = os.path.join(unprocessed_dir, filename)
        
        # Check if file already exists in unprocessed
        if os.path.exists(unprocessed_path):
            print(f"   ⚪ Skipping (already exists): {filename}")
            skipped_count += 1
            continue
        
        try:
            # Copy file to unprocessed directory
            shutil.copy2(download_path, unprocessed_path)
            print(f"   ✅ Moved: {filename}")
            moved_count += 1
            
        except Exception as e:
            print(f"   ❌ Error moving {filename}: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"   📊 Files moved: {moved_count}")
    print(f"   📊 Files skipped: {skipped_count}")
    print(f"   📊 Total files in unprocessed: {len([f for f in os.listdir(unprocessed_dir) if f.endswith('.csv')])}")
    
    if moved_count > 0:
        print(f"\n💡 Files are now ready for analysis in: {unprocessed_dir}")

if __name__ == "__main__":
    print("🔄 Moving Files to Unprocessed Directory")
    print("=" * 50)
    move_files_to_unprocessed()
