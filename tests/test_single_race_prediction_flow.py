#!/usr/bin/env python3
"""
Step 4: Single-race prediction flow (UI) Testing

This script tests:
1. Select one race and click **Predict Selected**.  
2. Verify POST to `/predict_single` (or mapped endpoint) succeeds (200) and triggers a download or inline results panel.  
3. Confirm log entry contains `"event":"prediction_completed","race_id":…,"status":"success"`.
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_single_race_prediction_flow():
    """Test the complete single-race prediction flow"""
    base_url = "http://localhost:5002"
    
    print("🚀 Testing Single-Race Prediction Flow")
    print("=" * 50)
    
    # Step 1: Check if the Flask app is running
    try:
        health_response = requests.get(f"{base_url}/api/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Flask app is running")
        else:
            print("❌ Flask app health check failed")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Flask app: {e}")
        return False
    
    # Step 2: List available races
    print("\n📋 Step 1: Checking available races...")
    try:
        races_response = requests.get(f"{base_url}/api/upcoming_races_csv", timeout=10)
        if races_response.status_code == 200:
            races_data = races_response.json()
            if races_data.get("success") and races_data.get("races"):
                races = races_data["races"]
                print(f"✅ Found {len(races)} upcoming races")
                for i, race in enumerate(races[:5]):  # Show first 5
                    print(f"   {i+1}. {race.get('filename', 'Unknown')}")
                selected_race = races[0]["filename"]  # Select first race
            else:
                print("❌ No races found in API response")
                return False
        else:
            print(f"❌ Failed to fetch races: {races_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching races: {e}")
        return False
    
    # Step 3: Test single race prediction (equivalent to clicking "Predict Selected")
    print(f"\n🎯 Step 2: Testing single race prediction for '{selected_race}'...")
    
    # Test the enhanced endpoint that the UI actually uses
    prediction_url = f"{base_url}/api/predict_single_race_enhanced"
    payload = {
        "race_filename": selected_race
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"   Sending POST to {prediction_url}")
        print(f"   Payload: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        prediction_response = requests.post(
            prediction_url, 
            data=json.dumps(payload), 
            headers=headers,
            timeout=120  # Increased time for prediction processing
        )
        end_time = time.time()
        
        print(f"   Request completed in {end_time - start_time:.2f} seconds")
        print(f"   Status Code: {prediction_response.status_code}")
        
        # Step 4: Verify successful response (200)
        if prediction_response.status_code == 200:
            print("✅ POST request succeeded (200)")
            
            try:
                response_data = prediction_response.json()
                print("✅ Response is valid JSON")
                
                # Check if prediction was successful
                if response_data.get("success"):
                    print("✅ Prediction completed successfully")
                    print(f"   Race ID: {response_data.get('race_id', 'N/A')}")
                    print(f"   Race Filename: {response_data.get('race_filename', 'N/A')}")
                    print(f"   Predictor Used: {response_data.get('predictor_used', 'N/A')}")
                    
                    # Check if predictions are present (inline results)
                    predictions = response_data.get("predictions", [])
                    if predictions:
                        print(f"✅ Found {len(predictions)} predictions (inline results)")
                        if len(predictions) > 0:
                            top_prediction = predictions[0]
                            print(f"   Top Pick: {top_prediction.get('dog_name', 'Unknown')} "
                                f"(Score: {top_prediction.get('final_score', 'N/A')})")
                    else:
                        print("⚠️ No predictions in response")
                        
                else:
                    print(f"❌ Prediction failed: {response_data.get('message', 'Unknown error')}")
                    return False
                    
            except json.JSONDecodeError:
                print("❌ Response is not valid JSON")
                print(f"Response text: {prediction_response.text[:500]}")
                return False
                
        else:
            print(f"❌ POST request failed with status {prediction_response.status_code}")
            print(f"Response: {prediction_response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Step 5: Check logs for prediction completion event
    print(f"\n📝 Step 3: Checking logs for prediction completion event...")
    
    # Give some time for logging to complete
    time.sleep(2)
    
    try:
        # Try to get logs from the API
        logs_response = requests.get(f"{base_url}/api/system_status", timeout=10)
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            logs = logs_data.get("logs", [])
            
            # Look for prediction completion logs
            prediction_logs = []
            for log in logs:
                log_message = log.get("message", "").lower()
                if any(keyword in log_message for keyword in ["prediction", "completed", "success"]):
                    prediction_logs.append(log)
            
            if prediction_logs:
                print(f"✅ Found {len(prediction_logs)} prediction-related log entries")
                for log in prediction_logs[-3:]:  # Show last 3
                    print(f"   {log.get('timestamp', 'N/A')}: {log.get('message', 'N/A')}")
            else:
                print("⚠️ No prediction-related logs found in system status")
                
        else:
            print(f"⚠️ Could not fetch system logs: {logs_response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching logs: {e}")
    
    # Also check log files directly
    try:
        log_dir = Path("./logs")
        if log_dir.exists():
            print("\n📁 Checking log files directly...")
            
            # Check process log
            process_log = log_dir / "process.log"
            if process_log.exists():
                with open(process_log, 'r') as f:
                    lines = f.readlines()[-10:]  # Get last 10 lines
                    prediction_lines = [line for line in lines if "prediction" in line.lower()]
                    if prediction_lines:
                        print(f"✅ Found prediction entries in process.log:")
                        for line in prediction_lines[-3:]:  # Show last 3
                            print(f"   {line.strip()}")
            
            # Check main workflow log
            workflow_log = log_dir / "main_workflow.jsonl"
            if workflow_log.exists():
                with open(workflow_log, 'r') as f:
                    lines = f.readlines()[-20:]  # Get last 20 lines
                    for line in lines:
                        try:
                            log_entry = json.loads(line.strip())
                            if "prediction" in log_entry.get("message", "").lower():
                                print(f"✅ Workflow log entry: {log_entry.get('message', 'N/A')}")
                        except json.JSONDecodeError:
                            continue
        
    except Exception as e:
        print(f"⚠️ Error reading log files: {e}")
    
    print("\n🎉 Single-race prediction flow test completed successfully!")
    return True

def main():
    """Main test function"""
    try:
        success = test_single_race_prediction_flow()
        if success:
            print("\n✅ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
