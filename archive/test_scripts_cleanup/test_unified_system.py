#!/usr/bin/env python3
"""
Unified System Test Suite
=========================

Comprehensive test suite for the unified predictor system,
time indicators, and frontend integration.

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

def test_unified_predictor_availability():
    """Test 1: Check unified predictor availability"""
    print('\n📋 TEST 1: Unified Predictor Availability')
    try:
        from unified_predictor import UnifiedPredictor, UnifiedPredictorConfig
        config = UnifiedPredictorConfig()
        predictor = UnifiedPredictor(config)
        
        print(f'✅ Unified predictor loaded successfully')
        print(f'📊 Components available: {sum(config.components_available.values())}/{len(config.components_available)}')
        
        for component, available in config.components_available.items():
            status = '✅' if available else '❌'
            print(f'   {status} {component}')
            
        return True
        
    except Exception as e:
        print(f'❌ Unified predictor failed: {e}')
        return False

def test_upcoming_races_and_time_indicators():
    """Test 2: Check upcoming races and time indicators"""
    print('\n📋 TEST 2: Upcoming Races and Time Indicators')
    try:
        from upcoming_race_browser import UpcomingRaceBrowser
        
        browser = UpcomingRaceBrowser()
        races = browser.get_upcoming_races(days_ahead=1)
        
        print(f'✅ Found {len(races)} upcoming races')
        
        if races:
            # Test time calculation logic
            now = datetime.now()
            
            for i, race in enumerate(races[:3]):  # Test first 3 races
                race_time = race.get('race_time', 'Unknown')
                print(f'\n📅 Race {i+1}: {race.get("venue", "Unknown")} R{race.get("race_number", "?")}')
                print(f'   Time: {race_time}')
                
                if race_time and race_time != 'Unknown':
                    try:
                        # Parse time (same logic as frontend)
                        if 'AM' in race_time.upper() or 'PM' in race_time.upper():
                            time_obj = datetime.strptime(race_time.upper(), '%I:%M %p').time()
                        else:
                            time_obj = datetime.strptime(race_time, '%H:%M').time()
                        
                        race_datetime = datetime.combine(now.date(), time_obj)
                        time_diff = race_datetime - now
                        minutes_until = int(time_diff.total_seconds() / 60)
                        
                        # Determine status
                        if minutes_until < -30:
                            status = 'FINISHED'
                        elif minutes_until < 0:
                            status = 'LIVE'
                        elif minutes_until <= 15:
                            status = 'SOON'
                        elif minutes_until <= 60:
                            status = 'UPCOMING'
                        else:
                            status = 'LATER'
                        
                        print(f'   ⏱️ Status: {status} ({minutes_until} minutes)')
                        
                    except Exception as e:
                        print(f'   ❌ Time parsing failed: {e}')
                else:
                    print(f'   ⚠️ No valid time data')
        else:
            print('⚠️ No races found for today')
            
        return True
        
    except ImportError as e:
        print(f'❌ UpcomingRaceBrowser not available: {e}')
        return False
    except Exception as e:
        print(f'❌ Race browser test failed: {e}')
        return False

def test_prediction_api_integration():
    """Test 3: Check prediction API integration"""
    print('\n📋 TEST 3: Prediction API Integration')
    try:
        # Check for prediction files
        predictions_dir = './predictions'
        if os.path.exists(predictions_dir):
            prediction_files = []
            for filename in os.listdir(predictions_dir):
                if (filename.startswith('prediction_') or filename.startswith('unified_prediction_')) and filename.endswith('.json'):
                    prediction_files.append(filename)
            
            print(f'✅ Found {len(prediction_files)} prediction files:')
            
            # Show file formats
            unified_count = len([f for f in prediction_files if f.startswith('unified_prediction_')])
            legacy_count = len([f for f in prediction_files if f.startswith('prediction_')])
            
            print(f'   📊 Unified format: {unified_count}')
            print(f'   📊 Legacy format: {legacy_count}')
            
            # Test reading a prediction file
            if prediction_files:
                sample_file = os.path.join(predictions_dir, prediction_files[0])
                
                try:
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f'\n📄 Sample prediction file: {prediction_files[0]}')
                    print(f'   Race info: {bool(data.get("race_info"))}')
                    print(f'   Predictions: {len(data.get("predictions", []))}')
                    print(f'   Methods used: {data.get("prediction_methods_used", [])}')
                    print(f'   Analysis version: {data.get("analysis_version", "Unknown")}')
                    
                    # Test top pick extraction
                    predictions = data.get('predictions', [])
                    if predictions:
                        top_pick = predictions[0]
                        print(f'   🏆 Top pick: {top_pick.get("dog_name", "Unknown")} (Score: {top_pick.get("final_score", 0):.3f})')
                    
                except Exception as e:
                    print(f'   ❌ Error reading prediction file: {e}')
                    return False
            
        else:
            print('⚠️ No predictions directory found')
            return False
            
        return True
        
    except Exception as e:
        print(f'❌ Prediction API test failed: {e}')
        return False

def test_end_to_end_unified_predictor():
    """Test 4: Test unified predictor end-to-end on an actual race file"""
    print('\n📋 TEST 4: End-to-End Unified Predictor Test')
    try:
        from unified_predictor import UnifiedPredictor, UnifiedPredictorConfig
        
        # Find a race file to test
        test_file = None
        for directory in ['./upcoming_races', './historical_races', './processed']:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith('.csv') and 'README' not in filename:
                        test_file = os.path.join(directory, filename)
                        break
                if test_file:
                    break
        
        if test_file:
            print(f'🎯 Testing with file: {os.path.basename(test_file)}')
            
            # Initialize unified predictor
            config = UnifiedPredictorConfig()
            predictor = UnifiedPredictor(config)
            
            # Record start time for file comparison
            start_time = datetime.now()
            
            # Run prediction
            result = predictor.predict_race_file(test_file)
            
            if result and result.get('success'):
                print(f'✅ Prediction successful!')
                print(f'   Method: {result.get("prediction_method", "Unknown")}')
                print(f'   Dogs predicted: {len(result.get("predictions", []))}')
                
                # Show top 3 predictions
                predictions = result.get('predictions', [])[:3]
                print(f'\n🏆 Top 3 predictions:')
                for i, pred in enumerate(predictions, 1):
                    dog_name = pred.get('dog_name', 'Unknown')
                    score = pred.get('final_score', pred.get('prediction_score', 0))
                    confidence = pred.get('confidence_level', 'Unknown')
                    print(f'   {i}. {dog_name} - Score: {score:.3f}, Confidence: {confidence}')
                
                # Check if prediction file was created (fixed timestamp comparison)
                prediction_files_created = []
                predictions_dir = './predictions'
                if os.path.exists(predictions_dir):
                    start_timestamp = start_time.timestamp()
                    
                    for filename in os.listdir(predictions_dir):
                        if (filename.startswith('unified_prediction_') or filename.startswith('prediction_')) and filename.endswith('.json'):
                            file_path = os.path.join(predictions_dir, filename)
                            file_mtime = os.path.getmtime(file_path)
                            
                            # Check if file was created after we started the test
                            if file_mtime > start_timestamp:
                                prediction_files_created.append(filename)
                
                if prediction_files_created:
                    print(f'\n📁 Created prediction files: {len(prediction_files_created)}')
                    for filename in prediction_files_created[:2]:  # Show first 2
                        print(f'   • {filename}')
                else:
                    print(f'\n📁 No new prediction files detected (may have used existing cache)')
                
                return True
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f'❌ Prediction failed: {error_msg}')
                return False
                
        else:
            print('⚠️ No race files found for testing')
            return False
            
    except Exception as e:
        print(f'❌ End-to-end test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_frontend_integration():
    """Test 5: Frontend integration components"""
    print('\n📋 TEST 5: Frontend Integration Components')
    try:
        # Test Flask app imports
        sys.path.append('.')
        
        # Check if Flask app can be imported
        try:
            from app import app, db_manager
            print('✅ Flask app imports successful')
            
            # Test database manager
            db_stats = db_manager.get_database_stats()
            print(f'✅ Database manager working: {db_stats.get("total_races", 0)} races')
            
        except Exception as e:
            print(f'❌ Flask app import failed: {e}')
            return False
        
        # Test prediction file reading logic
        predictions_dir = './predictions'
        if os.path.exists(predictions_dir):
            # Simulate the API logic for reading predictions
            prediction_files = []
            for filename in os.listdir(predictions_dir):
                if (filename.startswith('prediction_') or filename.startswith('unified_prediction_')) and filename.endswith('.json') and 'summary' not in filename:
                    file_path = os.path.join(predictions_dir, filename)
                    mtime = os.path.getmtime(file_path)
                    
                    # Assign priority: unified predictions (1) > others (2)
                    priority = 1 if filename.startswith('unified_prediction_') else 2
                    prediction_files.append((file_path, mtime, priority))
            
            # Sort by priority first, then by modification time
            prediction_files.sort(key=lambda x: (x[2], -x[1]))
            
            print(f'✅ Frontend prediction reading logic: {len(prediction_files)} files processed')
            
            if prediction_files:
                # Test reading the top priority file
                try:
                    with open(prediction_files[0][0], 'r') as f:
                        data = json.load(f)
                    
                    # Extract data as the frontend would
                    race_info = data.get('race_info', {})
                    predictions_list = data.get('predictions', [])
                    prediction_methods = data.get('prediction_methods_used', [])
                    
                    print(f'   📊 Top priority file: {os.path.basename(prediction_files[0][0])}')
                    print(f'   🎯 Venue: {race_info.get("venue", "Unknown")}')
                    print(f'   📅 Date: {race_info.get("date", "Unknown")}')
                    print(f'   🐕 Dogs: {len(predictions_list)}')
                    print(f'   🔧 Methods: {prediction_methods}')
                    
                except Exception as e:
                    print(f'   ❌ Error reading top priority file: {e}')
                    return False
        
        return True
        
    except Exception as e:
        print(f'❌ Frontend integration test failed: {e}')
        return False

def main():
    """Run all tests"""
    print('🧪 COMPREHENSIVE SYSTEM TEST SUITE')
    print('=' * 50)
    
    tests = [
        test_unified_predictor_availability,
        test_upcoming_races_and_time_indicators,
        test_prediction_api_integration,
        test_end_to_end_unified_predictor,
        test_frontend_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f'❌ Test {test.__name__} crashed: {e}')
            results.append(False)
    
    # Summary
    print('\n📊 TEST SUMMARY')
    print('=' * 20)
    passed = sum(results)
    total = len(results)
    
    print(f'Tests passed: {passed}/{total}')
    print(f'Success rate: {(passed/total)*100:.1f}%')
    
    if passed == total:
        print('🎉 ALL TESTS PASSED! System is ready for production.')
        return 0
    else:
        print('⚠️ Some tests failed. Please review the issues above.')
        return 1

if __name__ == '__main__':
    sys.exit(main())
