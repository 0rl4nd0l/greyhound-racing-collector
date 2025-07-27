#!/usr/bin/env python3
"""
Test script to verify the improved file movement workflow
"""

import os
from form_guide_csv_scraper import FormGuideCsvScraper

def test_file_movement():
    """Test the file movement functionality"""
    
    print("🔍 Testing File Movement Workflow")
    print("=" * 60)
    
    # Create scraper instance
    scraper = FormGuideCsvScraper()
    
    # Check initial state
    print("📊 Initial state:")
    unprocessed_files = [f for f in os.listdir(scraper.unprocessed_dir) if f.endswith('.csv')] if os.path.exists(scraper.unprocessed_dir) else []
    download_files = [f for f in os.listdir(scraper.download_dir) if f.endswith('.csv')] if os.path.exists(scraper.download_dir) else []
    
    print(f"   • Unprocessed directory: {len(unprocessed_files)} files")
    print(f"   • Download directory: {len(download_files)} files")
    
    # Test the move function
    print(f"\n📞 Testing move function...")
    moved_count = scraper.move_downloaded_to_unprocessed()
    
    # Check final state
    print(f"\n📊 Final state:")
    unprocessed_files_after = [f for f in os.listdir(scraper.unprocessed_dir) if f.endswith('.csv')] if os.path.exists(scraper.unprocessed_dir) else []
    download_files_after = [f for f in os.listdir(scraper.download_dir) if f.endswith('.csv')] if os.path.exists(scraper.download_dir) else []
    
    print(f"   • Unprocessed directory: {len(unprocessed_files_after)} files")
    print(f"   • Download directory: {len(download_files_after)} files")
    
    # Show some sample files in unprocessed
    print(f"\n📋 Sample files in unprocessed directory:")
    for i, filename in enumerate(unprocessed_files_after[:5]):
        print(f"   {i+1}. {filename}")
    
    if len(unprocessed_files_after) > 5:
        print(f"   ... and {len(unprocessed_files_after) - 5} more files")
    
    print(f"\n✅ File movement test complete!")
    print(f"📊 Files moved: {moved_count}")
    print(f"📊 Total files ready for analysis: {len(unprocessed_files_after)}")

if __name__ == "__main__":
    test_file_movement()
