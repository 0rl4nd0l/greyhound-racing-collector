#!/usr/bin/env python3
"""
Quick test to check if the URL construction fix works
"""

import requests
from bs4 import BeautifulSoup
import re

def test_url_construction():
    """Test URL construction for expert-form pages"""
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Test URLs that were failing
    test_urls = [
        "https://www.thedogs.com.au/racing/casino/2025-07-15/2/richmond-bakery-maiden?trial=false",
        "https://www.thedogs.com.au/racing/mandurah/2025-07-15/4/all-torque-engineering-pbd?trial=false"
    ]
    
    print("🔍 Testing URL Construction Fix")
    print("=" * 60)
    
    for race_url in test_urls:
        print(f"\n🔄 Testing: {race_url}")
        
        # Remove query parameters and add expert-form
        base_race_url = race_url.split('?')[0]  # Remove ?trial=false etc.
        expert_form_url = f"{base_race_url}/expert-form"
        
        print(f"   🔗 Expert-form URL: {expert_form_url}")
        
        try:
            response = session.get(expert_form_url, timeout=10)
            print(f"   📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Check for CSV export form
                csv_form = None
                for form in soup.find_all('form'):
                    if form.find('input', {'name': 'export_csv'}) or form.find('button', {'name': 'export_csv'}):
                        csv_form = form
                        break
                
                if csv_form:
                    print(f"   ✅ Found CSV export form!")
                    
                    # Try to extract download URL
                    form_action = csv_form.get('action')
                    form_method = csv_form.get('method', 'GET').upper()
                    
                    # Build form data
                    form_data = {}
                    
                    # Get all form inputs
                    for input_elem in csv_form.find_all(['input', 'select', 'textarea']):
                        name = input_elem.get('name')
                        if name:
                            input_type = input_elem.get('type', 'text')
                            
                            if input_type == 'checkbox':
                                if input_elem.get('checked'):
                                    form_data[name] = input_elem.get('value', 'on')
                            elif input_type == 'radio':
                                if input_elem.get('checked'):
                                    form_data[name] = input_elem.get('value', '')
                            elif input_type == 'submit':
                                pass
                            elif input_type == 'hidden':
                                form_data[name] = input_elem.get('value', '')
                            else:
                                form_data[name] = input_elem.get('value', '')
                    
                    # Add the CSV export parameter
                    form_data['export_csv'] = 'true'
                    
                    # Determine the target URL
                    if form_action:
                        if form_action.startswith('/'):
                            target_url = f"https://www.thedogs.com.au{form_action}"
                        elif form_action.startswith('http'):
                            target_url = form_action
                        else:
                            target_url = f"https://www.thedogs.com.au/{form_action}"
                    else:
                        target_url = expert_form_url
                    
                    print(f"   🎯 Submitting to: {target_url}")
                    
                    # Submit the form
                    if form_method == 'POST':
                        form_response = session.post(target_url, data=form_data, timeout=10)
                    else:
                        form_response = session.get(target_url, params=form_data, timeout=10)
                    
                    print(f"   📊 Form response: {form_response.status_code}")
                    
                    if form_response.status_code == 200:
                        download_url = form_response.text.strip()
                        if download_url.startswith('http'):
                            print(f"   ✅ Got download URL: {download_url}")
                            
                            # Try to download the CSV
                            csv_response = session.get(download_url, timeout=10)
                            print(f"   📊 CSV download: {csv_response.status_code}")
                            
                            if csv_response.status_code == 200:
                                lines = csv_response.text.split('\n')
                                print(f"   📄 CSV lines: {len(lines)}")
                                print(f"   📋 Sample header: {lines[0][:80] if lines else 'No content'}")
                                print(f"   🎯 SUCCESS: CSV downloaded successfully!")
                            else:
                                print(f"   ❌ CSV download failed")
                        else:
                            print(f"   ❌ Unexpected response: {download_url[:100]}")
                    else:
                        print(f"   ❌ Form submission failed")
                else:
                    print(f"   ❌ No CSV export form found")
            else:
                print(f"   ❌ Expert-form page not accessible")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_url_construction()
