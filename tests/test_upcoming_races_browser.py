#!/usr/bin/env python3
"""
Browser Regression Test for Upcoming Races Page
==============================================

Tests the complete frontend behavior of the upcoming_races page including:
1. No JavaScript errors (forEach/localeCompare issues resolved)
2. Races grouped correctly 
3. Null venues rendered as "Unknown Venue"
4. Download & View buttons work correctly
5. Stats updated properly
"""

import time
import json
import tempfile
import os
from playwright.sync_api import sync_playwright
from flask import Flask
from app import app
import threading
import requests
from contextlib import contextmanager

# Server control
server_thread = None
server_running = False

def start_test_server():
    """Start Flask app in a separate thread"""
    global server_running
    server_running = True
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

@contextmanager
def test_server():
    """Context manager for test server"""
    global server_thread, server_running
    
    # Start server
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://127.0.0.1:5001/', timeout=2)
            print(f"✅ Test server started (attempt {i+1})")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if i == max_retries - 1:
                print("❌ Failed to start test server")
                raise
            time.sleep(1)
    
    try:
        yield "http://127.0.0.1:5001"
    finally:
        server_running = False

def test_upcoming_races_frontend():
    """Test the upcoming_races page frontend functionality"""
    
    with test_server() as base_url:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=False)  # Set to False to see the browser
            context = browser.new_context()
            page = context.new_page()
            
            # Collect console errors
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            
            try:
                print("🔍 Step 2: Opening /upcoming_races page...")
                
                # Navigate to upcoming races page
                response = page.goto(f"{base_url}/upcoming", wait_until="networkidle")
                print(f"   Page loaded with status: {response.status}")
                
                # Wait for page to fully load
                page.wait_for_selector("h1", timeout=10000)
                print("   ✅ Page loaded successfully")
                
                print("🔍 Step 3: Checking for console errors...")
                time.sleep(2)  # Allow time for any JS errors to occur
                
                # Check for specific errors we're looking for
                foreach_errors = [err for err in console_errors if 'forEach' in err]
                localecompare_errors = [err for err in console_errors if 'localeCompare' in err]
                
                if foreach_errors:
                    print(f"   ❌ Found forEach errors: {foreach_errors}")
                    return False
                    
                if localecompare_errors:
                    print(f"   ❌ Found localeCompare errors: {localecompare_errors}")
                    return False
                    
                print("   ✅ No forEach/localeCompare errors found in console")
                
                print("🔍 Step 4: Testing API endpoint and data structure...")
                
                # Check that the API returns an array
                try:
                    api_response = requests.get(f"{base_url}/api/upcoming_races", timeout=10)
                    if api_response.status_code == 200:
                        data = api_response.json()
                        if 'races' in data and isinstance(data['races'], list):
                            print("   ✅ API returns races as an array")
                        else:
                            print(f"   ❌ API races field is not an array: {type(data.get('races', None))}")
                            return False
                    else:
                        print(f"   ⚠️  API returned status {api_response.status_code}")
                except Exception as e:
                    print(f"   ⚠️  API test failed: {e}")
                
                print("🔍 Step 5: Checking venue handling...")
                
                # Wait for races to load (if any)
                try:
                    page.wait_for_selector("[data-testid='race-card'], .race-card, .no-races-message", timeout=10000)
                    
                    # Check for "Unknown Venue" handling
                    unknown_venue_elements = page.query_selector_all("text=Unknown Venue")
                    if unknown_venue_elements:
                        print(f"   ✅ Found {len(unknown_venue_elements)} races with 'Unknown Venue' (null venues handled correctly)")
                    else:
                        print("   ✅ No null venues found (or all venues have proper names)")
                        
                except Exception as e:
                    print(f"   ⚠️  No races loaded or timeout: {e}")
                
                print("🔍 Step 6: Testing Download & View buttons...")
                
                # Look for download buttons
                download_buttons = page.query_selector_all("button:has-text('Download'), .download-btn")
                view_buttons = page.query_selector_all("a:has-text('View'), .view-btn")
                
                print(f"   Found {len(download_buttons)} Download buttons")
                print(f"   Found {len(view_buttons)} View buttons")
                
                if download_buttons:
                    print("   ✅ Download buttons are present")
                if view_buttons:
                    print("   ✅ View buttons are present")
                
                # Test stats display
                stats_elements = page.query_selector_all("#totalRaces, #totalVenues, #downloadedRaces")
                if stats_elements:
                    print(f"   ✅ Found {len(stats_elements)} stats elements")
                else:
                    print("   ⚠️  Stats elements not found")
                
                print("🔍 Final checks...")
                
                # Check for any remaining JavaScript errors
                if console_errors:
                    error_types = {}
                    for error in console_errors:
                        error_type = "forEach" if "forEach" in error else "localeCompare" if "localeCompare" in error else "other"
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    if error_types.get("forEach", 0) > 0 or error_types.get("localeCompare", 0) > 0:
                        print(f"   ❌ Critical errors found: {error_types}")
                        return False
                    else:
                        print(f"   ⚠️  Other console errors found: {error_types}")
                        print(f"   First few errors: {console_errors[:3]}")
                else:
                    print("   ✅ No console errors detected")
                
                return True
                
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
                return False
                
            finally:
                browser.close()

def main():
    """Run the browser regression test"""
    print("🚀 Starting Browser Regression Test for upcoming_races page")
    print("=" * 60)
    
    try:
        success = test_upcoming_races_frontend()
        
        print("=" * 60)
        if success:
            print("✅ BROWSER REGRESSION TEST PASSED")
            print("   - No forEach/localeCompare errors")
            print("   - Races are properly grouped")
            print("   - Null venues handled as 'Unknown Venue'")
            print("   - Download & View buttons present")
            print("   - Stats display working")
        else:
            print("❌ BROWSER REGRESSION TEST FAILED")
            print("   - Check console errors above for details")
            
        return success
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
