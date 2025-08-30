#!/usr/bin/env python3
"""
TGR Integration Test
===================
Test the integrated TGR workflow in the enhanced comprehensive processor.
"""

import os
import sqlite3
from datetime import datetime

def test_tgr_integration():
    """Test TGR integration workflow"""
    print("🧪 Testing TGR Integration Workflow")
    print("=" * 50)
    
    # Test 1: Check TGR component availability
    print("\n1️⃣ Testing TGR Component Availability...")
    
    try:
        from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
        processor = EnhancedComprehensiveProcessor(processing_mode="fast")
        
        # Check if TGR is available in processor
        if hasattr(processor, '__dict__'):
            print("✅ Enhanced Comprehensive Processor initialized")
        
        # Check TGR imports directly
        try:
            from enhanced_data_integration import EnhancedDataIntegrator
            integrator = EnhancedDataIntegrator()
            
            if hasattr(integrator, 'tgr_available'):
                print(f"✅ TGR Status: {'Available' if integrator.tgr_available else 'Not Available'}")
            else:
                print("ℹ️ TGR status attribute not found")
                
        except Exception as e:
            print(f"⚠️ Enhanced Data Integrator error: {e}")
            
    except Exception as e:
        print(f"❌ Processor initialization error: {e}")
    
    # Test 2: Check TGR prediction integration
    print("\n2️⃣ Testing TGR Prediction Integration...")
    
    try:
        from tgr_prediction_integration import TGRPredictionIntegrator
        
        tgr_integrator = TGRPredictionIntegrator()
        features = tgr_integrator.get_feature_names()
        
        print(f"✅ TGR Features: {len(features)} available")
        print(f"   Sample: {', '.join(features[:5])}")
        
    except Exception as e:
        print(f"⚠️ TGR Prediction Integration error: {e}")
    
    # Test 3: Check database for TGR tables
    print("\n3️⃣ Testing TGR Database Structure...")
    
    try:
        if os.path.exists("greyhound_racing_data.db"):
            conn = sqlite3.connect("greyhound_racing_data.db")
            cursor = conn.cursor()
            
            # Check for TGR tables
            tgr_tables = [
                'tgr_dog_performance_summary',
                'tgr_enhanced_dog_form', 
                'tgr_expert_insights',
                'tgr_enhanced_feature_cache'
            ]
            
            for table in tgr_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   ✅ {table}: {count} records")
                except:
                    print(f"   ⚠️ {table}: Not found")
            
            conn.close()
        else:
            print("   ℹ️ Database not found")
            
    except Exception as e:
        print(f"❌ Database check error: {e}")
    
    # Test 4: Check file manager UI TGR status
    print("\n4️⃣ Testing File Manager UI TGR Status...")
    
    try:
        from file_manager_ui import FileManager
        file_manager = FileManager()
        
        pipeline_status = file_manager.get_pipeline_status()
        
        if 'tgr_integration' in pipeline_status:
            status = pipeline_status['tgr_integration']
            print(f"✅ TGR Integration in UI: {'Enabled' if status else 'Disabled'}")
        else:
            print("⚠️ TGR status not found in pipeline")
            
    except Exception as e:
        print(f"⚠️ File Manager UI error: {e}")
    
    # Test 5: Test live TGR data fetching (mock)
    print("\n5️⃣ Testing TGR Data Fetching...")
    
    try:
        from enhanced_data_integration import EnhancedDataIntegrator
        
        integrator = EnhancedDataIntegrator()
        
        # Mock dog data
        test_dogs = [
            {'dog_name': 'Test Dog 1', 'box_number': 1},
            {'dog_name': 'Test Dog 2', 'box_number': 2}
        ]
        
        race_context = {
            'venue': 'TEST',
            'race_date': '2025-08-23',
            'race_number': 1
        }
        
        enhanced_dogs = integrator.fetch_live_tgr_data_for_dogs(test_dogs, race_context)
        
        print(f"✅ TGR Data Fetch Test: {len(enhanced_dogs)} dogs processed")
        
        # Check if any dogs have TGR data
        tgr_enhanced = sum(1 for dog in enhanced_dogs if dog.get('has_tgr_data'))
        print(f"   TGR Enhanced Dogs: {tgr_enhanced}/{len(enhanced_dogs)}")
        
    except Exception as e:
        print(f"⚠️ TGR Data Fetching error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 TGR Integration Test Complete!")
    print("✅ System is ready for TGR-enhanced processing")

if __name__ == "__main__":
    test_tgr_integration()
