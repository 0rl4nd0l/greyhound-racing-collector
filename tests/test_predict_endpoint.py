#!/usr/bin/env python3
"""
Test script for the /api/predict_single_race endpoint
"""

import json
import subprocess
import sys
import time

import requests


def start_flask_app():
    """Start the Flask app in background"""
    try:
        # Start Flask app
        process = subprocess.Popen(
            ["python3", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
    except Exception as e:
        print(f"Failed to start Flask app: {e}")
        return None


def test_predict_single_race():
    """Test the predict_single_race endpoint"""
    url = "http://localhost:5002/api/predict_single_race"

    # Test data - using one of the sample CSVs
    test_data = {"race_filename": "Race 6 - CASINO - 2025-07-31.csv"}

    headers = {"Content-Type": "application/json"}

    print("🧪 Testing /api/predict_single_race endpoint")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(test_data, indent=2)}")
    print("-" * 50)

    try:
        # Send POST request
        response = requests.post(url, json=test_data, headers=headers, timeout=120)

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        # Try to parse JSON response
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")

            # Check if the response indicates success
            if response.status_code == 200:
                if response_json.get("success"):
                    print("✅ SUCCESS: Endpoint returned 200 OK with success: true")
                    print(
                        "✅ CONFIRMED: Pipeline no longer throws '0 races recognised' errors"
                    )
                    return True
                else:
                    print("❌ FAILURE: Endpoint returned 200 but success: false")
                    print(
                        f"Error message: {response_json.get('message', 'No message provided')}"
                    )
                    return False
            else:
                print(f"❌ FAILURE: Endpoint returned {response.status_code}")
                if "error" in response_json:
                    print(f"Error: {response_json['error']}")
                return False

        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
            print("❌ FAILURE: Could not decode JSON response")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ FAILURE: Could not connect to Flask server")
        print("Make sure the Flask app is running on localhost:5002")
        return False
    except requests.exceptions.Timeout:
        print("❌ FAILURE: Request timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ FAILURE: Unexpected error: {e}")
        return False


def main():
    """Main test execution"""
    print("🚀 Starting smoke test for /api/predict_single_race endpoint")
    print("=" * 60)

    # Check if Flask app is already running
    try:
        response = requests.get("http://localhost:5002/api/health", timeout=5)
        print("✅ Flask app is already running")
        app_process = None
    except:
        print("🔧 Starting Flask app...")
        app_process = start_flask_app()
        if not app_process:
            print("❌ Failed to start Flask app")
            sys.exit(1)

        # Wait for Flask app to start
        print("⏳ Waiting for Flask app to initialize...")
        time.sleep(10)

        # Check if app started successfully
        try:
            response = requests.get("http://localhost:5002/api/health", timeout=5)
            print("✅ Flask app started successfully")
        except:
            print("❌ Flask app failed to start properly")
            if app_process:
                app_process.terminate()
            sys.exit(1)

    try:
        # Run the actual test
        success = test_predict_single_race()

        if success:
            print("\n" + "=" * 60)
            print("🎉 SMOKE TEST PASSED!")
            print("✅ /api/predict_single_race endpoint is working correctly")
            print("✅ Pipeline no longer throws '0 races recognised' errors")
            print("✅ Response format: 200 OK with success: true")
        else:
            print("\n" + "=" * 60)
            print("❌ SMOKE TEST FAILED!")
            print("❌ Check the error messages above for details")

    finally:
        # Clean up - terminate Flask app if we started it
        if app_process:
            print("\n🔧 Cleaning up - stopping Flask app...")
            app_process.terminate()
            try:
                app_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                app_process.kill()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
