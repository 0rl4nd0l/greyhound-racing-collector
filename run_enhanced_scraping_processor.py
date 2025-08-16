#!/usr/bin/env python3
"""
Enhanced CSV Processing with Web Scraping for Race Results
=========================================================

This script runs the enhanced comprehensive processor in full mode with web scraping
enabled to collect actual race results and winners, combining them with form guide data.

Key Features:
- Web scraping for actual race results and winners
- Track condition and weather data collection
- Expert form data extraction with speed ratings
- Comprehensive database population with complete race data
- Batch processing with progress reporting

Author: AI Assistant
Date: August 2, 2025
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check for ChromeDriver
    import shutil
    chromedriver_path = shutil.which("chromedriver")
    if not chromedriver_path:
        print("❌ ChromeDriver not found in PATH")
        print("   Please install ChromeDriver for web scraping:")
        print("   brew install chromedriver  # On macOS")
        print("   Or download from: https://chromedriver.chromium.org/")
        return False
    else:
        print(f"✅ ChromeDriver found: {chromedriver_path}")
    
    # Check for required Python packages
    required_packages = [
        ('selenium', 'Selenium WebDriver'),
        ('bs4', 'BeautifulSoup HTML parsing'),  # Fixed import name
        ('requests', 'HTTP requests'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computations')
    ]
    
    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {description} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {description} not available")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_csv_directories():
    """Setup CSV directories for processing"""
    print("\n📁 Setting up CSV directories...")
    
    # Find CSV files from previous processing
    csv_dirs = [
        "form_guides/downloaded",
        "processed/excluded", 
        "processed/other",
        "processed/completed"
    ]
    
    all_csv_files = []
    for csv_dir in csv_dirs:
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            print(f"   📂 {csv_dir}: {len(csv_files)} CSV files")
            all_csv_files.extend([os.path.join(csv_dir, f) for f in csv_files])
    
    # Create unprocessed directory for the enhanced processor
    unprocessed_dir = "./unprocessed"
    os.makedirs(unprocessed_dir, exist_ok=True)
    
    # Copy CSV files to unprocessed directory for re-processing with web scraping
    import shutil
    copied_count = 0
    
    # Focus on excluded files first (these need web scraping to get actual results)
    priority_dirs = ["processed/excluded", "form_guides/downloaded"]
    
    for csv_dir in priority_dirs:
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            for csv_file in csv_files[:50]:  # Limit to first 50 files for testing
                src_path = os.path.join(csv_dir, csv_file)
                dst_path = os.path.join(unprocessed_dir, csv_file)
                
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Error copying {csv_file}: {e}")
    
    print(f"   ✅ Copied {copied_count} CSV files to unprocessed directory for web scraping")
    return copied_count

def run_enhanced_processor_with_scraping():
    """Run the enhanced processor with web scraping enabled"""
    print("\n🚀 Initializing Enhanced Comprehensive Processor with Web Scraping...")
    
    try:
        # Import the enhanced processor
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
        
        # Initialize processor in FULL mode (enables web scraping)
        processor = EnhancedComprehensiveProcessor(
            db_path="greyhound_data.db",
            processing_mode="full",  # FULL mode enables web scraping
            batch_size=10  # Smaller batches for web scraping
        )
        
        print("✅ Enhanced Comprehensive Processor initialized")
        print(f"   🌐 Web scraping: ENABLED")
        print(f"   🎯 Processing mode: FULL")
        print(f"   📦 Batch size: 10")
        print(f"   🗄️  Database: greyhound_data.db")
        
        # Process all unprocessed files with web scraping
        print("\n🔄 Starting processing with web scraping...")
        results = processor.process_all_unprocessed()
        
        # Print results
        print(f"\n📊 PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Successfully processed: {results.get('processed_count', 0)} files")
        print(f"❌ Failed to process: {results.get('failed_count', 0)} files")
        print(f"⏭️ Skipped (already processed): {results.get('skipped_count', 0)} files")
        
        # Show detailed results
        if results.get('results'):
            success_count = sum(1 for r in results['results'] if r['result'].get('status') == 'success')
            excluded_count = sum(1 for r in results['results'] if r['result'].get('status') == 'excluded')
            
            print(f"\n📈 Detailed Results:")
            print(f"   🎯 Successfully processed with race results: {success_count}")
            print(f"   🚫 Excluded (no race results found): {excluded_count}")
            
            # Show sample successful results
            successful_results = [r for r in results['results'] if r['result'].get('status') == 'success']
            if successful_results:
                print(f"\n✅ Sample successful races:")
                for result in successful_results[:3]:
                    race_info = result['result'].get('race_info', {})
                    winner = race_info.get('winner_name', 'Unknown')
                    venue = race_info.get('venue', 'Unknown')
                    race_num = race_info.get('race_number', 'Unknown')
                    date = race_info.get('race_date', 'Unknown')
                    print(f"   🏆 {venue} Race {race_num} ({date}) - Winner: {winner}")
        
        # Generate comprehensive report
        if results.get("processed_count", 0) > 0:
            print(f"\n📋 Generating comprehensive report...")
            report_path = processor.generate_comprehensive_report()
            if report_path:
                print(f"✅ Report saved to: {report_path}")
        
        # Check database status
        print(f"\n🗄️ Checking database status...")
        check_database_status()
        
        return results
        
    except Exception as e:
        print(f"❌ Error running enhanced processor: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        try:
            processor.cleanup()
            print("✅ Processor cleanup completed")
        except:
            pass

def check_database_status():
    """Check the current database status"""
    try:
        import sqlite3
        conn = sqlite3.connect('greyhound_data.db')
        cursor = conn.cursor()
        
        # Check race metadata
        cursor.execute('SELECT COUNT(*) FROM race_metadata')
        races = cursor.fetchone()[0]
        
        # Check dog race data
        cursor.execute('SELECT COUNT(*) FROM dog_race_data')
        dog_records = cursor.fetchone()[0]
        
        # Check for races with winners
        cursor.execute('SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != ""')
        races_with_winners = cursor.fetchone()[0]
        
        # Check for races with URLs (scraped successfully)
        cursor.execute('SELECT COUNT(*) FROM race_metadata WHERE url IS NOT NULL AND url != ""')
        races_with_urls = cursor.fetchone()[0]
        
        # Sample recent races
        cursor.execute('''
            SELECT race_id, venue, race_number, race_date, winner_name, url 
            FROM race_metadata 
            ORDER BY extraction_timestamp DESC 
            LIMIT 5
        ''')
        recent_races = cursor.fetchall()
        
        conn.close()
        
        print(f"   📊 Total races in database: {races}")
        print(f"   🐕 Total dog records: {dog_records}")
        print(f"   🏆 Races with winners: {races_with_winners}")
        print(f"   🌐 Races with scraped URLs: {races_with_urls}")
        
        if recent_races:
            print(f"\n   📋 Recent races:")
            for race in recent_races:
                race_id, venue, race_num, date, winner, url = race
                winner_status = "✅" if winner else "❌"
                url_status = "🌐" if url else "📝"
                print(f"     {winner_status}{url_status} {venue} R{race_num} ({date}) - {winner or 'No winner'}")
        
    except Exception as e:
        print(f"   ❌ Error checking database: {e}")

def main():
    """Main function"""
    print("🏁 ENHANCED GREYHOUND RACING PROCESSOR WITH WEB SCRAPING")
    print("=" * 70)
    print("This script will:")
    print("1. Check prerequisites (ChromeDriver, packages)")
    print("2. Setup CSV files for processing")
    print("3. Run enhanced processor with web scraping enabled")
    print("4. Collect actual race results and winners")
    print("5. Populate database with complete race data")
    print("=" * 70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return False
    
    # Setup CSV directories
    csv_count = setup_csv_directories()
    if csv_count == 0:
        print("\n❌ No CSV files found to process. Exiting.")
        return False
    
    # Confirm before proceeding
    print(f"\n🎯 Ready to process {csv_count} CSV files with web scraping.")
    print("This will:")
    print("- Access race websites to get actual results")
    print("- Take longer due to web scraping delays")
    print("- Populate database with complete race data")
    
    response = input("\nProceed with web scraping? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled by user.")
        return False
    
    # Run the enhanced processor
    start_time = time.time()
    results = run_enhanced_processor_with_scraping()
    end_time = time.time()
    
    if results:
        processing_time = end_time - start_time
        print(f"\n⏰ Total processing time: {processing_time:.1f} seconds")
        print(f"📊 Processing rate: {results.get('processed_count', 0) / (processing_time / 60):.1f} files/minute")
        
        print(f"\n🎉 Enhanced processing with web scraping completed!")
        print("Your database now contains:")
        print("- Form guide data (historical performance)")
        print("- Actual race results and winners (from web scraping)")
        print("- Track conditions and weather data")
        print("- Expert form ratings and analysis")
        print("- Comprehensive race analytics")
        
        return True
    else:
        print(f"\n❌ Processing failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
