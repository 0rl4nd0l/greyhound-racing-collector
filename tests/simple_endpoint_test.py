#!/usr/bin/env python3
"""
Simple endpoint test to verify Flask app accessibility
"""

import requests
import time
import sys

def test_endpoints():
    """Test basic endpoints for accessibility"""
    import os
    base_url = f"http://127.0.0.1:{os.environ.get('DEFAULT_PORT', '5002')}"
    
    endpoints = [
        "/api/health",
        "/api/enable-explain-analyze", 
        "/ws?type=ping",
        "/api/ml-predict",
        "/api/races",
        "/"
    ]
    
    print(f"🧪 Testing Flask endpoints at {base_url}")
    print("=" * 50)
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"Testing: {endpoint}... ", end="")
            
            if endpoint == "/api/ml-predict":
                # POST request with sample data
                data = {
                    "race_id": "test_race",
                    "dogs": [
                        {"name": "Test Dog 1", "stats": {"wins": 5, "races": 10}},
                        {"name": "Test Dog 2", "stats": {"wins": 3, "races": 8}}
                    ]
                }
                response = requests.post(url, json=data, timeout=10)
            else:
                # GET request
                response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {response.status_code} - OK")
                # Print first 100 chars of response for health endpoint
                if endpoint == "/api/health":
                    try:
                        json_resp = response.json()
                        print(f"   Status: {json_resp.get('status', 'unknown')}")
                    except:
                        print(f"   Response: {response.text[:100]}...")
                        
            elif response.status_code == 403:
                print(f"🚫 {response.status_code} - Forbidden")
                print(f"   Error: {response.text[:200]}...")
            elif response.status_code == 404:
                print(f"❓ {response.status_code} - Not Found")
            elif response.status_code == 500:
                print(f"💥 {response.status_code} - Internal Server Error")
                print(f"   Error: {response.text[:200]}...")
            else:
                print(f"⚠️  {response.status_code} - {response.reason}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error - Server not running?")
        except requests.exceptions.Timeout:
            print("⏰ Timeout - Server not responding")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Endpoint test completed")

if __name__ == "__main__":
    test_endpoints()
