#!/usr/bin/env python3
"""
Test script to verify duplicate prevention works correctly
"""

from form_guide_csv_scraper import FormGuideCsvScraper
import os

def test_duplicate_prevention():
    """Test that duplicate prevention works correctly"""
    
    print("🔍 Testing Duplicate Prevention")
    print("=" * 60)
    
    # Create scraper instance
    scraper = FormGuideCsvScraper()
    
    # Test race info
    test_race_info = {
        'race_number': '1',
        'venue': 'TEST',
        'date': '11 July 2025'
    }
    
    filename = f"Race {test_race_info['race_number']} - {test_race_info['venue']} - {test_race_info['date']}.csv"
    
    print(f"📋 Testing with filename: {filename}")
    print(f"📊 Current existing files count: {len(scraper.existing_files)}")
    
    # Test 1: Check if file doesn't exist initially
    if scraper.file_already_exists(filename, test_race_info):
        print("❌ File incorrectly reported as existing when it shouldn't")
    else:
        print("✅ File correctly reported as not existing")
    
    # Test 2: Check against known existing files
    print(f"\n📋 Sample existing files:")
    for i, existing_file in enumerate(list(scraper.existing_files)[:5]):
        print(f"  {i+1}. {existing_file}")
    
    # Test 3: Check if refresh works
    print(f"\n🔄 Testing refresh...")
    initial_count = len(scraper.existing_files)
    scraper.refresh_existing_files()
    refreshed_count = len(scraper.existing_files)
    
    print(f"📊 Initial count: {initial_count}")
    print(f"📊 Refreshed count: {refreshed_count}")
    
    if initial_count == refreshed_count:
        print("✅ Refresh working correctly (counts match)")
    else:
        print("⚠️ Refresh count changed - this might be normal if files were added/removed")
    
    # Test 4: Check a known existing file
    if scraper.existing_files:
        known_file = list(scraper.existing_files)[0]
        print(f"\n📋 Testing with known existing file: {known_file}")
        
        # Extract race info from known file
        if " - " in known_file:
            parts = known_file.replace('.csv', '').split(' - ')
            if len(parts) >= 3:
                race_num = parts[0].replace('Race ', '')
                venue = parts[1]
                date = parts[2]
                
                known_race_info = {
                    'race_number': race_num,
                    'venue': venue,
                    'date': date
                }
                
                if scraper.file_already_exists(known_file, known_race_info):
                    print("✅ Known file correctly reported as existing")
                else:
                    print("❌ Known file incorrectly reported as not existing")
    
    print("\n" + "=" * 60)
    print("🏁 Duplicate Prevention Test Complete")

if __name__ == "__main__":
    test_duplicate_prevention()
