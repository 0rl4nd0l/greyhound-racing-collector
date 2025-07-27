#!/usr/bin/env python3

import requests
import json

def test_api_response():
    """Test the race files status API and examine confidence level data"""
    
    try:
        # Make API call
        response = requests.get('http://localhost:5002/api/race_files_status')
        
        if response.status_code == 200:
            data = response.json()
            
            print("🔍 API RESPONSE STRUCTURE")
            print("="*50)
            
            print(f"Success: {data.get('success', False)}")
            print(f"Total Predicted: {data.get('total_predicted', 0)}")
            print(f"Total Unpredicted: {data.get('total_unpredicted', 0)}")
            
            predicted_races = data.get('predicted_races', [])
            print(f"\n📋 Predicted Races Count: {len(predicted_races)}")
            
            if predicted_races:
                first_race = predicted_races[0]
                print(f"\n🏁 FIRST RACE STRUCTURE:")
                print(f"Race Name: {first_race.get('race_name', 'N/A')}")
                print(f"Venue: {first_race.get('venue', 'N/A')}")
                print(f"Average Confidence: {first_race.get('average_confidence', 'N/A')}")
                
                # Check top pick structure
                top_pick = first_race.get('top_pick', {})
                print(f"\n🐕 TOP PICK STRUCTURE:")
                for key, value in top_pick.items():
                    print(f"  {key}: {value}")
                    
                # Check if confidence_level exists
                if 'confidence_level' in top_pick:
                    print(f"\n✅ CONFIDENCE LEVEL FOUND: '{top_pick['confidence_level']}'")
                else:
                    print(f"\n❌ CONFIDENCE LEVEL NOT FOUND IN TOP PICK")
                    print(f"Available keys in top_pick: {list(top_pick.keys())}")
                
                # Check if ML usage is detected
                ml_used = first_race.get('ml_predictions_used', False)
                print(f"\n🤖 ML USAGE DETECTED: {ml_used}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_api_response()
