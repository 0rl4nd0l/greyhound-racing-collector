#!/usr/bin/env python3
"""
Test script to follow the CSV export redirect and download the actual CSV
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

def test_csv_download():
    """Test downloading CSV via the export URL"""
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Test URLs
    base_url = "https://www.thedogs.com.au"
    expert_form_url = "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f/expert-form"
    
    print("🔍 Testing Full CSV Download Process")
    print("=" * 60)
    
    try:
        # Step 1: Get the expert form page
        print("📋 Step 1: Getting expert form page...")
        response = session.get(expert_form_url, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Failed to access expert form page: {response.status_code}")
            return
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the form with CSV export
        form = None
        for f in soup.find_all('form'):
            if f.find('input', {'name': 'export_csv'}) or f.find('button', {'name': 'export_csv'}):
                form = f
                break
        
        if not form:
            print("❌ Could not find form with CSV export")
            return
            
        # Extract form data
        form_action = form.get('action')
        form_method = form.get('method', 'GET').upper()
        
        # Build form data
        form_data = {}
        
        # Get all form inputs
        for input_elem in form.find_all(['input', 'select', 'textarea']):
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
                target_url = base_url + form_action
            elif form_action.startswith('http'):
                target_url = form_action
            else:
                target_url = urljoin(expert_form_url, form_action)
        else:
            target_url = expert_form_url
        
        print(f"✅ Found form, submitting to: {target_url}")
        
        # Step 2: Submit the form to get the download URL
        print("📋 Step 2: Submitting form to get download URL...")
        
        if form_method == 'POST':
            response = session.post(target_url, data=form_data, timeout=30)
        else:
            response = session.get(target_url, params=form_data, timeout=30)
        
        print(f"📊 Form response status: {response.status_code}")
        print(f"📊 Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"📊 Content-Length: {len(response.content)} bytes")
        
        # The response should contain the download URL
        download_url = response.text.strip()
        
        if not download_url.startswith('http'):
            print(f"❌ Expected download URL, got: {download_url}")
            return
            
        print(f"🔗 Download URL: {download_url}")
        
        # Step 3: Download the actual CSV
        print("📋 Step 3: Downloading CSV file...")
        
        csv_response = session.get(download_url, timeout=30)
        print(f"📊 CSV response status: {csv_response.status_code}")
        print(f"📊 CSV Content-Type: {csv_response.headers.get('content-type', 'unknown')}")
        print(f"📊 CSV Content-Length: {len(csv_response.content)} bytes")
        
        # Check if this is CSV content
        content_type = csv_response.headers.get('content-type', '').lower()
        if csv_response.status_code == 200:
            
            # Save the CSV
            filename = "race_form_guide.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_response.text)
            
            print(f"✅ SUCCESS: Downloaded CSV to {filename}")
            print(f"💾 File size: {len(csv_response.text)} characters")
            
            # Show sample content
            lines = csv_response.text.split('\n')
            print(f"📄 CSV has {len(lines)} lines")
            
            if len(lines) > 0:
                print("📋 Sample CSV content:")
                for i, line in enumerate(lines[:10]):  # Show first 10 lines
                    print(f"  {i+1}: {line}")
                
                if len(lines) > 10:
                    print("  ... (truncated)")
                    
            # Basic validation
            if len(lines) > 1 and ',' in lines[0]:
                print("✅ CSV format validation: PASSED")
            else:
                print("⚠️ CSV format validation: QUESTIONABLE")
                
        else:
            print(f"❌ Failed to download CSV: {csv_response.status_code}")
            print(f"📄 Response preview: {csv_response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error during CSV download test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_download()
