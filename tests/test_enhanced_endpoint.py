#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced single race prediction endpoint functionality
"""

import json
import time
from datetime import datetime
import os
import sys

# Add the current directory to Python path to import app modules
sys.path.append('.')

# Import the enhanced prediction functionality from the app
try:
    from app import api_predict_single_race_enhanced
    from flask import Flask, request
    
    print("✅ Successfully imported enhanced prediction endpoint")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_enhanced_prediction():
    """Test the enhanced prediction endpoint with different scenarios"""
    
    print("🚀 Testing Enhanced Single Race Prediction Endpoint")
    print("=" * 60)
    
    # Test scenarios
    test_cases = [
        {
            "name": "Test with race_filename",
            "data": {"race_filename": "test_race.csv"},
            "description": "Using direct race filename parameter"
        },
        {
            "name": "Test with race_id derivation", 
            "data": {"race_id": "test_race"},
            "description": "Using race_id that should derive filename"
        },
        {
            "name": "Test with TAREE race",
            "data": {"race_filename": "Race 11 - TAREE - 2025-08-02.csv"},
            "description": "Testing with the TAREE race file"
        },
        {
            "name": "Test missing parameters",
            "data": {},
            "description": "Testing error handling for missing parameters"
        },
        {
            "name": "Test non-existent file",
            "data": {"race_filename": "non_existent_race.csv"},
            "description": "Testing error handling for missing file"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Data: {test_case['data']}")
        
        try:
            # Create a mock Flask app and request context for testing
            app = Flask(__name__)
            with app.test_request_context(
                '/api/predict_single_race_enhanced',
                method='POST',
                json=test_case['data']
            ):
                start_time = time.time()
                
                # Call the endpoint function directly
                result = api_predict_single_race_enhanced()
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Extract the response
                if hasattr(result, 'get_json'):
                    response_data = result.get_json()
                    status_code = result.status_code
                elif isinstance(result, tuple):
                    response_data = result[0].get_json() if hasattr(result[0], 'get_json') else result[0]
                    status_code = result[1] if len(result) > 1 else 200
                else:
                    response_data = result
                    status_code = 200
                
                print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"   📊 Status code: {status_code}")
                
                if isinstance(response_data, dict):
                    print(f"   ✅ Success: {response_data.get('success', 'unknown')}")
                    if response_data.get('success'):
                        print(f"   📈 Predictor used: {response_data.get('predictor_used', 'unknown')}")
                        predictions = response_data.get('predictions', [])
                        print(f"   🐕 Number of predictions: {len(predictions)}")
                        if predictions:
                            top_pick = predictions[0]
                            print(f"   🏆 Top pick: {top_pick.get('dog_name', 'unknown')} (Score: {top_pick.get('final_score', 0):.3f})")
                    else:
                        print(f"   ❌ Error: {response_data.get('message', 'unknown error')}")
                        print(f"   🔍 Error type: {response_data.get('error_type', 'unknown')}")
                else:
                    print(f"   📄 Raw response: {str(response_data)[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Test failed with exception: {str(e)}")
            import traceback
            print(f"   📋 Traceback: {traceback.format_exc()}")
        
        print("-" * 40)

def main():
    """Main function to run the tests"""
    print(f"🕒 Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if required files exist
    upcoming_dir = './upcoming_races'
    if not os.path.exists(upcoming_dir):
        print(f"❌ Upcoming races directory not found: {upcoming_dir}")
        return
    
    files_in_dir = os.listdir(upcoming_dir)
    csv_files = [f for f in files_in_dir if f.endswith('.csv')]
    
    print(f"📁 Found {len(csv_files)} CSV files in {upcoming_dir}:")
    for file in csv_files[:5]:  # Show first 5
        print(f"   - {file}")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more files")
    
    # Run the tests
    test_enhanced_prediction()
    
    print(f"\n🏁 Tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
