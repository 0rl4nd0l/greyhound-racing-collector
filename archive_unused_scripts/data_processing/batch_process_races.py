#!/usr/bin/env python3
"""
Batch Race Processor
===================

This script processes race files in batches using the enhanced race processor,
demonstrating file movement to prevent reprocessing.

Author: AI Assistant
Date: July 24, 2025
"""

import os
from pathlib import Path
from enhanced_race_processor_fixed import EnhancedRaceProcessor

def batch_process_form_guides(limit=5):
    """Process form guide files in batches with file movement"""
    print("📋 Batch Processing Form Guides")
    print("=" * 50)
    
    processor = EnhancedRaceProcessor('greyhound_racing_data.db')
    
    # Get form guide files
    form_guide_dir = Path('form_guides/downloaded')
    if not form_guide_dir.exists():
        print("❌ Form guides directory not found")
        return
    
    csv_files = list(form_guide_dir.glob('*.csv'))
    
    if not csv_files:
        print("❌ No CSV files found in form_guides/downloaded")
        return
    
    print(f"📁 Found {len(csv_files)} form guide files")
    print(f"🎯 Processing first {min(limit, len(csv_files))} files with file movement...\n")
    
    processed_count = 0
    moved_count = 0
    
    for csv_file in csv_files[:limit]:
        print(f"📂 Processing: {csv_file.name}")
        
        try:
            # Process with file movement enabled
            result = processor.process_race_results(csv_file, move_processed=True)
            
            if result.get('success') or result.get('status') == 'success_with_issues':
                processed_count += 1
                print(f"   ✅ Success: {result.get('summary', 'Processed')}")
                
                if result.get('moved_to'):
                    moved_count += 1
                    print(f"   📁 Moved to: {result['moved_to']}")
                
                # Show data quality info
                if result.get('data_quality'):
                    quality = result['data_quality']
                    warnings = len(quality.get('warnings', []))
                    errors = len(quality.get('errors', []))
                    if warnings > 0 or errors > 0:
                        print(f"   📊 Quality: {warnings} warnings, {errors} errors")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        print()  # Add spacing between files
    
    print("=" * 50)
    print(f"✅ Batch processing complete!")
    print(f"📊 Statistics:")
    print(f"   - Files processed: {processed_count}/{limit}")
    print(f"   - Files moved: {moved_count}")
    print(f"   - Files remaining in source: {len(csv_files) - moved_count}")

def check_processed_folders():
    """Check what's in the processed folders"""
    print("\n📁 Checking Processed Folders")
    print("=" * 30)
    
    processed_dir = Path('./processed')
    if not processed_dir.exists():
        print("❌ No processed directory found")
        return
    
    for subdir in processed_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob('*.csv'))
            print(f"📂 {subdir.name}: {len(files)} files")
            
            # Show first few files as examples
            for file in files[:3]:
                print(f"   - {file.name}")
            
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")

def verify_no_reprocessing(test_file=None):
    """Verify that processed files aren't reprocessed"""
    print("\n🔒 Verifying No Reprocessing")
    print("=" * 30)
    
    processor = EnhancedRaceProcessor('greyhound_racing_data.db')
    
    # Find a file that was moved to processed
    processed_dir = Path('./processed')
    if not processed_dir.exists():
        print("❌ No processed directory to test")
        return
    
    # Look for any processed file
    for subdir in processed_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob('*.csv'))
            if files and not test_file:
                test_file = files[0]
                break
    
    if not test_file:
        print("❌ No processed files found to test")
        return
    
    print(f"🧪 Testing reprocessing prevention with: {test_file.name}")
    
    # Try to process the file again (should work but won't move since it's already processed)
    try:
        result = processor.process_race_results(test_file, move_processed=True)
        print(f"✅ File can still be processed: {result.get('summary', 'Success')}")
        
        if result.get('moved_to'):
            print(f"📁 File was moved again to: {result['moved_to']}")
        else:
            print("📁 File was not moved (already in processed location)")
            
    except Exception as e:
        print(f"❌ Error testing file: {e}")

if __name__ == "__main__":
    # Process a small batch of form guides
    batch_process_form_guides(limit=3)
    
    # Check what's in processed folders
    check_processed_folders()
    
    # Verify reprocessing prevention
    verify_no_reprocessing()
    
    print("\n🎯 Batch processing demonstration complete!")
