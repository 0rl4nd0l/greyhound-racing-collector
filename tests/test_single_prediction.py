#!/usr/bin/env python3
"""
Test script for single race prediction API endpoint
"""

import time

import requests


def test_prediction_api():
    """Test the prediction API with various scenarios"""

    base_url = "http://localhost:5002"
    endpoint = "/api/predict_single_race_enhanced"

    print("🧪 Testing Single Race Prediction API")
    print("=" * 50)

    # Test 1: Valid request
    print("\n1. Testing valid request...")
    test_data = {
        "race_filename": "Race 6 - CASINO - 2025-07-31.csv",
        "force_rerun": False,
    }

    try:
        start_time = time.time()
        response = requests.post(f"{base_url}{endpoint}", json=test_data, timeout=60)
        duration = time.time() - start_time

        print(f"   Status Code: {response.status_code}")
        print(f"   Duration: {duration:.2f}s")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("   ✅ Prediction successful!")
                prediction_info = result.get("prediction_summary", {})
                print(
                    f"   📊 Dogs analyzed: {prediction_info.get('total_dogs', 'N/A')}"
                )
                top_pick = prediction_info.get("top_pick", {})
                print(
                    f"   🏆 Top pick: {top_pick.get('dog_name', 'N/A')} ({top_pick.get('prediction_score', 0):.3f})"
                )
            else:
                print(
                    f"   ❌ Prediction failed: {result.get('message', 'Unknown error')}"
                )
        else:
            print(f"   ❌ HTTP Error: {response.text}")

    except requests.exceptions.Timeout:
        print("   ⏰ Request timed out (60s)")
    except requests.exceptions.ConnectionError:
        print("   🔌 Connection error - is Flask app running?")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Missing filename
    print("\n2. Testing missing filename...")
    try:
        response = requests.post(f"{base_url}{endpoint}", json={}, timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 400:
            result = response.json()
            print(f"   ✅ Expected 400 error: {result.get('message')}")
        else:
            print(f"   ❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Non-existent file
    print("\n3. Testing non-existent file...")
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            json={"race_filename": "NonExistent.csv"},
            timeout=10,
        )
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 404:
            result = response.json()
            print(f"   ✅ Expected 404 error: {result.get('message')}")
        else:
            print(f"   ❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 4: Quick status check
    print("\n4. Testing system status...")
    try:
        response = requests.get(f"{base_url}/api/system_status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("   ✅ System status OK")
            print(
                f"   📊 DB stats: {status.get('db_stats', {}).get('total_races', 'N/A')} races"
            )
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status error: {e}")


if __name__ == "__main__":
    test_prediction_api()
