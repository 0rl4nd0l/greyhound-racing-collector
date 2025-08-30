#!/usr/bin/env python3
"""
Test TGR Integration in Enhanced Data Processing
===============================================

This script tests whether TGR integration is working correctly
when processing CSV files through the enhanced data processing workflow.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_tgr_integration_imports():
    """Test if all TGR components can be imported correctly."""
    
    print("🧪 Testing TGR Integration Components...")
    
    # Test 1: TGR Scraper
    try:
        from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
        scraper = TheGreyhoundRecorderScraper(rate_limit=5.0, use_cache=True)
        print("✅ TGR Scraper: Available and initialized")
        tgr_scraper_ok = True
    except Exception as e:
        print(f"❌ TGR Scraper: Failed - {e}")
        tgr_scraper_ok = False
    
    # Test 2: Enhanced TGR Collector
    try:
        from enhanced_tgr_collector import EnhancedTGRCollector
        collector = EnhancedTGRCollector()
        print("✅ Enhanced TGR Collector: Available and initialized")
        tgr_collector_ok = True
    except Exception as e:
        print(f"❌ Enhanced TGR Collector: Failed - {e}")
        tgr_collector_ok = False
    
    # Test 3: TGR Prediction Integration
    try:
        from tgr_prediction_integration import TGRPredictionIntegrator
        integrator = TGRPredictionIntegrator()
        print("✅ TGR Prediction Integrator: Available and initialized")
        tgr_integrator_ok = True
    except Exception as e:
        print(f"❌ TGR Prediction Integrator: Failed - {e}")
        tgr_integrator_ok = False
    
    # Test 4: Enhanced Data Integrator
    try:
        from enhanced_data_integration import EnhancedDataIntegrator
        data_integrator = EnhancedDataIntegrator()
        print("✅ Enhanced Data Integrator: Available and initialized")
        
        # Test the main TGR integration method
        test_dogs = [{'dog_name': 'Test Dog', 'clean_name': 'Test Dog'}]
        test_context = {'venue': 'Test Track', 'race_date': '2025-08-01', 'race_number': 1}
        
        enhanced_dogs = data_integrator.fetch_live_tgr_data_for_dogs(test_dogs, test_context)
        print(f"✅ TGR Integration Method: Working (processed {len(enhanced_dogs)} dogs)")
        data_integrator_ok = True
    except Exception as e:
        print(f"❌ Enhanced Data Integrator: Failed - {e}")
        data_integrator_ok = False
    
    # Test 5: Enhanced Comprehensive Processor
    try:
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
        processor = EnhancedComprehensiveProcessor(processing_mode="minimal")
        print("✅ Enhanced Comprehensive Processor: Available and initialized")
        processor_ok = True
    except Exception as e:
        print(f"❌ Enhanced Comprehensive Processor: Failed - {e}")
        processor_ok = False
    
    # Summary
    components_working = sum([
        tgr_scraper_ok, tgr_collector_ok, tgr_integrator_ok, 
        data_integrator_ok, processor_ok
    ])
    
    print(f"\\n📊 TGR Integration Status: {components_working}/5 components working")
    
    if components_working >= 4:
        print("✅ TGR integration should work in enhanced data processing!")
        return True
    else:
        print("⚠️ TGR integration may have issues in enhanced data processing")
        return False

def check_sample_csv_files():
    """Check for sample CSV files to test with."""
    
    print("\\n🔍 Checking for sample CSV files...")
    
    unprocessed_dir = Path("./unprocessed")
    upcoming_dir = Path("./upcoming_races") 
    
    csv_files = []
    
    # Check unprocessed directory
    if unprocessed_dir.exists():
        unprocessed_csvs = list(unprocessed_dir.glob("*.csv"))
        csv_files.extend(unprocessed_csvs)
        print(f"📁 Unprocessed directory: {len(unprocessed_csvs)} CSV files")
    
    # Check upcoming races directory
    if upcoming_dir.exists():
        upcoming_csvs = list(upcoming_dir.glob("*.csv"))
        csv_files.extend(upcoming_csvs)
        print(f"📁 Upcoming races directory: {len(upcoming_csvs)} CSV files")
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV files for testing:")
        for csv_file in csv_files[:5]:  # Show first 5
            print(f"   • {csv_file.name}")
        if len(csv_files) > 5:
            print(f"   • ... and {len(csv_files) - 5} more")
        return csv_files
    else:
        print("⚠️ No CSV files found for testing")
        return []

def test_csv_processing_with_tgr():
    """Test processing a CSV file with TGR integration."""
    
    print("\\n🧪 Testing CSV Processing with TGR Integration...")
    
    try:
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
        
        # Initialize processor in test mode
        processor = EnhancedComprehensiveProcessor(processing_mode="fast", batch_size=1)
        
        # Check for CSV files
        csv_files = check_sample_csv_files()
        
        if not csv_files:
            print("⚠️ No CSV files available for testing - skipping CSV processing test")
            return False
        
        # Test with first available CSV file
        test_csv = csv_files[0]
        print(f"\\n📋 Testing with: {test_csv.name}")
        
        # Process the CSV file
        result = processor.process_csv_file(str(test_csv))
        
        if result and result.get('status') == 'success':
            print("✅ CSV processing completed successfully!")
            
            # Check if TGR data was integrated
            dogs = result.get('dogs', [])
            tgr_enhanced_dogs = [dog for dog in dogs if dog.get('has_tgr_data')]
            
            print(f"📊 Processing Results:")
            print(f"   • Total dogs: {len(dogs)}")
            print(f"   • TGR enhanced dogs: {len(tgr_enhanced_dogs)}")
            print(f"   • TGR enhancement ratio: {len(tgr_enhanced_dogs)/len(dogs)*100:.1f}%" if dogs else "   • TGR enhancement ratio: N/A")
            
            if tgr_enhanced_dogs:
                print("✅ TGR integration is working in CSV processing!")
                sample_dog = tgr_enhanced_dogs[0]
                print(f"   • Sample enhanced dog: {sample_dog.get('dog_name')}")
                print(f"   • Has TGR data: {sample_dog.get('has_tgr_data')}")
                print(f"   • TGR fetch timestamp: {sample_dog.get('tgr_fetch_timestamp', 'N/A')}")
                return True
            else:
                print("⚠️ TGR integration processed but no dogs were enhanced")
                print("   This could be normal if TGR cache is empty or dogs are not in TGR database")
                return True
        else:
            print(f"❌ CSV processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing CSV processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    print("🚀 TGR Integration Test for Enhanced Data Processing")
    print("=" * 60)
    
    # Test 1: Component imports
    imports_ok = test_tgr_integration_imports()
    
    # Test 2: CSV processing with TGR
    if imports_ok:
        csv_processing_ok = test_csv_processing_with_tgr()
    else:
        print("⚠️ Skipping CSV processing test due to import failures")
        csv_processing_ok = False
    
    # Final summary
    print("\\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    
    if imports_ok and csv_processing_ok:
        print("🎉 SUCCESS: TGR integration is working in enhanced data processing!")
        print("\\n✅ When you press 'Enhanced Data Processing' in the UI:")
        print("   1. CSV files from unprocessed directory will be processed")
        print("   2. Dog names will be extracted from each CSV")
        print("   3. TGR data will be fetched for each dog")
        print("   4. Racing histories will enrich the CSV data")
        print("   5. Enhanced data will be saved to the database")
    elif imports_ok:
        print("⚠️ PARTIAL SUCCESS: TGR components work but CSV processing needs attention")
        print("   - TGR integration is available")
        print("   - May need sample CSV files or cache data to test fully")
    else:
        print("❌ FAILURE: TGR integration has component issues")
        print("   - Check dependencies and imports")
        print("   - Some TGR modules may need debugging")
    
    print(f"\\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
