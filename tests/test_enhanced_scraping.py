#!/usr/bin/env python3
"""
Test Enhanced Web Scraping - Quick Verification
=============================================

Quick test to verify the enhanced processor works with the fixed database schema.
"""

import os
import sys

def test_enhanced_scraping():
    """Test the enhanced processor with a few files"""
    print("🧪 Testing Enhanced Web Scraping with Fixed Database")
    print("=" * 60)
    
    try:
        # Import and initialize processor
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
        
        processor = EnhancedComprehensiveProcessor(
            db_path="greyhound_data.db",
            processing_mode="full",  # Full mode with web scraping
            batch_size=5  # Small batch for testing
        )
        
        # Process just a few files for testing
        print("\n🔄 Processing a small test batch...")
        
        # Get list of unprocessed files
        unprocessed_files = [f for f in os.listdir("./unprocessed") if f.endswith('.csv')][:5]
        
        if not unprocessed_files:
            print("❌ No unprocessed files found")
            return False
        
        print(f"📁 Found {len(unprocessed_files)} files to test")
        
        # Process each file
        success_count = 0
        for filename in unprocessed_files:
            file_path = os.path.join("./unprocessed", filename)
            print(f"\n🔄 Testing: {filename}")
            
            try:
                result = processor.process_csv_file(file_path)
                if result and result.get('status') == 'success':
                    race_info = result.get('race_info', {})
                    winner = race_info.get('winner_name', 'Unknown')
                    print(f"   ✅ SUCCESS - Winner: {winner}")
                    success_count += 1
                else:
                    status = result.get('status', 'unknown') if result else 'failed'
                    print(f"   ⚠️  Status: {status}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n📊 Test Results: {success_count}/{len(unprocessed_files)} successful")
        
        # Check database status
        print(f"\n🗄️  Checking database...")
        import sqlite3
        conn = sqlite3.connect('greyhound_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM race_metadata')
        races = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM dog_race_data')
        dogs = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != ""')
        winners = cursor.fetchone()[0]
        
        print(f"   📊 Total races: {races}")
        print(f"   🐕 Dog records: {dogs}")
        print(f"   🏆 Races with winners: {winners}")
        
        # Show recent successful races
        cursor.execute('''
            SELECT race_id, winner_name, url 
            FROM race_metadata 
            WHERE winner_name IS NOT NULL AND winner_name != ""
            ORDER BY extraction_timestamp DESC 
            LIMIT 3
        ''')
        recent_winners = cursor.fetchall()
        
        if recent_winners:
            print(f"\n   🎉 Recent successful races:")
            for race_id, winner, url in recent_winners:
                scraped_status = "🌐" if url else "📝"
                print(f"     {scraped_status} {race_id} - Winner: {winner}")
        
        conn.close()
        
        # Cleanup
        processor.cleanup()
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_scraping()
    if success:
        print(f"\n🎉 Web scraping test PASSED! Database schema is fixed.")
        print("You can now run the full processing script.")
    else:
        print(f"\n❌ Web scraping test FAILED.")
    
    sys.exit(0 if success else 1)
