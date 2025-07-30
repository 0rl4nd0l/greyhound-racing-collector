#!/usr/bin/env python3
"""
Test ML System V3 Implementation
"""

import os
import sys
import json
import tempfile
import csv
from datetime import datetime

# Add the current directory to sys.path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_system_v3_import():
    """Test that ML System V3 can be imported"""
    try:
        from ml_system_v3 import MLSystemV3
        from prediction_pipeline_v3 import PredictionPipelineV3
        print("✅ ML System V3 imports successful")
        return True
    except ImportError as e:
        print(f"❌ ML System V3 import failed: {e}")
        return False

def test_ml_system_initialization():
    """Test ML System V3 initialization"""
    try:
        from ml_system_v3 import MLSystemV3
        
        ml_system = MLSystemV3("test.db")
        print("✅ ML System V3 initialization successful")
        print(f"  - Model info: {ml_system.get_model_info()}")
        return True
    except Exception as e:
        print(f"❌ ML System V3 initialization failed: {e}")
        return False

def test_prediction_pipeline_initialization():
    """Test Prediction Pipeline V3 initialization"""
    try:
        from prediction_pipeline_v3 import PredictionPipelineV3
        
        pipeline = PredictionPipelineV3("test.db")
        print("✅ Prediction Pipeline V3 initialization successful")
        return True
    except Exception as e:
        print(f"❌ Prediction Pipeline V3 initialization failed: {e}")
        return False

def test_mock_prediction():
    """Test ML System V3 prediction with mock data"""
    try:
        from ml_system_v3 import MLSystemV3
        
        ml_system = MLSystemV3("test.db")
        
        # Mock dog data
        mock_dog = {
            'name': 'Test Dog',
            'box_number': 1,
            'weight': 32.5,
            'starting_price': 2.5,
            'individual_time': 25.5,
            'field_size': 8,
            'temperature': 22.0,
            'humidity': 65.0,
            'wind_speed': 12.0
        }
        
        result = ml_system.predict(mock_dog)
        print("✅ ML System V3 prediction successful")
        print(f"  - Win probability: {result['win_probability']:.3f}")
        print(f"  - Confidence: {result['confidence']:.3f}")
        print(f"  - Model info: {result['model_info']}")
        return True
    except Exception as e:
        print(f"❌ ML System V3 prediction failed: {e}")
        return False

def test_pipeline_with_mock_race_file():
    """Test full pipeline with a mock race file"""
    try:
        from prediction_pipeline_v3 import PredictionPipelineV3
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['Dog Name', 'WGT', 'SP', 'TIME'])
            writer.writerow(['1. Fast Runner', '32.5', '2.5', '25.6'])
            writer.writerow(['2. Speed Demon', '31.8', '3.2', '25.8'])
            writer.writerow(['3. Quick Flash', '33.1', '4.1', '26.2'])
            writer.writerow(['', '', '', ''])  # Empty row to test parsing
            race_file_path = f.name
        
        try:
            pipeline = PredictionPipelineV3("test.db")
            result = pipeline.predict_race_file(race_file_path)
            
            print("✅ Pipeline prediction successful")
            print(f"  - Success: {result['success']}")
            print(f"  - Predictions count: {len(result.get('predictions', []))}")
            print(f"  - Prediction method: {result.get('prediction_method')}")
            
            if result.get('predictions'):
                top_pick = result['predictions'][0]
                print(f"  - Top pick: {top_pick['dog_name']} (prob: {top_pick['win_probability']:.3f})")
            
            if result.get('quality_issues'):
                print(f"  - Quality issues: {result['quality_issues']}")
            
            return result['success']
        finally:
            # Clean up temporary file
            os.unlink(race_file_path)
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_app_integration():
    """Test that app.py can import and use ML System V3"""
    try:
        # Check if the import succeeds in app.py context
        from prediction_pipeline_v3 import PredictionPipelineV3
        from ml_system_v3 import train_new_model
        
        print("✅ App integration imports successful")
        print("  - PredictionPipelineV3 available")
        print("  - train_new_model function available")
        
        # Test the training function
        result = train_new_model('gradient_boosting')
        print(f"  - Training function result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False

def check_weather_features():
    """Check what weather and other advanced features are available"""
    print("\n🌤️ Checking Available Features:")
    
    # Check for weather integration
    weather_files = [
        'weather_api.py',
        'weather_enhanced_predictor.py', 
        'comprehensive_weather_integration.py'
    ]
    
    for file in weather_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - Weather integration available")
        else:
            print(f"  ❌ {file} - Not found")
    
    # Check for GPT integration
    gpt_files = [
        'gpt_prediction_enhancer.py',
        'gpt_enhanced_analysis.py'
    ]
    
    for file in gpt_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - GPT enhancement available")
        else:
            print(f"  ❌ {file} - Not found")
    
    # Check for comprehensive systems
    comprehensive_files = [
        'unified_predictor.py',
        'comprehensive_prediction_pipeline.py',
        'enhanced_pipeline_v2.py'
    ]
    
    for file in comprehensive_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - Advanced pipeline available")
        else:
            print(f"  ❌ {file} - Not found")

def main():
    """Run all tests"""
    print("🚀 Testing ML System V3 Implementation")
    print("=" * 50)
    
    tests = [
        test_ml_system_v3_import,
        test_ml_system_initialization,
        test_prediction_pipeline_initialization,
        test_mock_prediction,
        test_pipeline_with_mock_race_file,
        test_app_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    # Check available features
    check_weather_features()
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
