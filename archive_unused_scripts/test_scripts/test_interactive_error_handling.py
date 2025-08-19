#!/usr/bin/env python3
"""
Test script to demonstrate interactive error handling for duplicate files
"""

import os
import sys

from form_guide_csv_scraper import FormGuideCsvScraper


def create_test_duplicate():
    """Create a test scenario where a file already exists"""
    scraper = FormGuideCsvScraper()
    
    # Create a test race info that would likely have an existing file
    test_race_info = {
        'race_number': '1',
        'venue': 'MAND',
        'date': '08 July 2025'
    }
    
    # Generate the filename that would be created
    filename = f"Race {test_race_info['race_number']} - {test_race_info['venue']} - {test_race_info['date']}.csv"
    
    # Check if it already exists in our directories
    if scraper.file_already_exists(filename, test_race_info):
        print(f"🎯 Perfect! Found existing file that matches: {filename}")
        print("This will trigger the interactive error handling.")
        
        # Test the interactive handling
        choice = scraper.handle_file_exists_interaction(filename, test_race_info)
        print(f"\n✅ User chose: {choice}")
        
        if choice == 'skip':
            print("   📋 Process would skip this race and continue")
        elif choice == 'overwrite':
            print("   🔄 Process would overwrite the existing file")
        elif choice == 'quit':
            print("   🛑 Process would exit completely")
        elif choice == 'auto_skip':
            print("   ⏭️  Process would skip all remaining duplicates automatically")
            
    else:
        print(f"❌ No existing file found for: {filename}")
        print("Let's check what files we do have:")
        
        # Show some existing files for reference
        unprocessed_dir = "./unprocessed"
        if os.path.exists(unprocessed_dir):
            files = [f for f in os.listdir(unprocessed_dir) if f.endswith('.csv')][:5]
            print(f"\n📋 Sample files in unprocessed directory:")
            for file in files:
                print(f"   • {file}")

if __name__ == "__main__":
    print("🧪 TESTING INTERACTIVE ERROR HANDLING")
    print("=" * 50)
    create_test_duplicate()
