#!/usr/bin/env python3
"""
Example: Fast Processed File Lookup API
======================================

This example demonstrates how to use the new get_processed_filenames API
for fast O(1) membership tests when checking if files have been processed.

Author: AI Assistant  
Date: January 2025
"""

import os
import sys
import time

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from form_guide_csv_scraper import FormGuideCsvScraper
from utils.caching_utils import get_processed_filenames

def main():
    print("🚀 Fast Processed File Lookup API Demo")
    print("=" * 50)
    
    # Method 1: Using the FormGuideCsvScraper class
    print("\n📋 Method 1: Using FormGuideCsvScraper.get_processed_filenames()")
    scraper = FormGuideCsvScraper()
    
    # Get processed files for specific directory
    unprocessed_files = scraper.get_processed_filenames("./unprocessed")
    print(f"   Files processed from ./unprocessed: {len(unprocessed_files)}")
    
    processed_files = scraper.get_processed_filenames("./processed")  
    print(f"   Files processed from ./processed: {len(processed_files)}")
    
    # Get all processed files
    all_files = scraper.get_processed_filenames("")
    print(f"   Total processed files: {len(all_files)}")
    
    # Method 2: Using the standalone utility function
    print("\n📋 Method 2: Using utils.caching_utils.get_processed_filenames()")
    
    # Fast lookup for multiple directories
    directories = ["./unprocessed", "./processed", "./form_guides/downloaded"]
    
    for directory in directories:
        start_time = time.time()
        files = get_processed_filenames(directory)
        elapsed = time.time() - start_time
        print(f"   {directory}: {len(files)} files ({elapsed:.4f}s)")
    
    # Demo: O(1) membership testing
    print("\n🔍 Demo: O(1) Membership Testing")
    
    # Get all processed filenames once
    start_time = time.time()
    all_processed = get_processed_filenames("")
    lookup_prep_time = time.time() - start_time
    print(f"   Loaded {len(all_processed)} processed filenames in {lookup_prep_time:.4f}s")
    
    # Test files to check
    test_files = [
        "Race 1 - SAN - 2025-01-15.csv",
        "Race 2 - MEA - 2025-01-15.csv", 
        "Race 3 - DAPT - 2025-01-15.csv",
        "nonexistent_file.csv"
    ]
    
    print("   Testing membership for sample files:")
    for test_file in test_files:
        start_time = time.time()
        is_processed = test_file in all_processed  # O(1) lookup!
        elapsed = time.time() - start_time
        status = "✅ PROCESSED" if is_processed else "❌ NOT PROCESSED"
        print(f"     {test_file}: {status} ({elapsed:.6f}s)")
    
    # Performance comparison demo
    print("\n⚡ Performance Demo: Batch File Processing Check")
    
    # Simulate checking 1000 files
    simulated_files = [f"Race_{i}_SAN_2025-01-{i%30+1:02d}.csv" for i in range(1000)]
    
    # Method A: Query database for each file (slow)
    print("   Method A: Individual database queries (NOT recommended)")
    start_time = time.time()
    processed_count_slow = 0
    # We'll skip this for demo but show what it would look like
    print("     [Skipped - would be very slow with individual DB queries]")
    
    # Method B: Single query + set lookups (fast) 
    print("   Method B: Single query + set lookups (RECOMMENDED)")
    start_time = time.time()
    
    # Single query to get all processed files
    processed_set = get_processed_filenames("")
    
    # Fast O(1) lookups for all files
    processed_count_fast = 0
    for file in simulated_files:
        if file in processed_set:  # O(1) lookup
            processed_count_fast += 1
    
    elapsed_fast = time.time() - start_time
    print(f"     Checked {len(simulated_files)} files in {elapsed_fast:.4f}s")
    print(f"     Found {processed_count_fast} already processed")
    print(f"     Average per file: {(elapsed_fast/len(simulated_files))*1000:.3f}ms")
    
    print("\n🎯 Key Benefits:")
    print("   • Single SQL query loads all processed filenames")
    print("   • O(1) membership tests using Python sets")
    print("   • Indexed file_path column for fast database queries") 
    print("   • Available in both FormGuideCsvScraper class and standalone utility")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
