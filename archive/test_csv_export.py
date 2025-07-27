#!/usr/bin/env python3
"""
Test script to properly submit the expert form for CSV export
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

def test_csv_export():
    """Test submitting the expert form for CSV export"""
    
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
    
    # Test URL from your example
    base_url = "https://www.thedogs.com.au"
    expert_form_url = "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f/expert-form"
    
    print("🔍 Testing CSV Export via Form Submission")
    print("=" * 60)
    print(f"🌐 URL: {expert_form_url}")
    
    try:
        # First, get the page to extract form data
        response = session.get(expert_form_url, timeout=30)
        print(f"📊 Initial page status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed to access page: {response.status_code}")
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
            
        print("✅ Found form with CSV export capability")
        
        # Extract form data
        form_action = form.get('action')
        form_method = form.get('method', 'GET').upper()
        
        print(f"📋 Form action: {form_action}")
        print(f"📋 Form method: {form_method}")
        
        # Build form data
        form_data = {}
        
        # Get all form inputs
        for input_elem in form.find_all(['input', 'select', 'textarea']):
            name = input_elem.get('name')
            if name:
                input_type = input_elem.get('type', 'text')
                
                if input_type == 'checkbox':
                    # Only include if checked
                    if input_elem.get('checked'):
                        form_data[name] = input_elem.get('value', 'on')
                elif input_type == 'radio':
                    # Only include if checked
                    if input_elem.get('checked'):
                        form_data[name] = input_elem.get('value', '')
                elif input_type == 'submit':
                    # Don't include submit buttons by default
                    pass
                elif input_type == 'hidden':
                    form_data[name] = input_elem.get('value', '')
                else:
                    # Regular inputs
                    form_data[name] = input_elem.get('value', '')
        
        # Add the CSV export parameter
        form_data['export_csv'] = 'true'
        
        print(f"📊 Form data: {form_data}")
        
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
        
        print(f"🎯 Target URL: {target_url}")
        
        # Make the request
        print(f"\n🚀 Submitting form with {form_method} method...")
        
        if form_method == 'POST':
            response = session.post(target_url, data=form_data, timeout=30)
        else:
            response = session.get(target_url, params=form_data, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"📊 Content-Length: {len(response.content)} bytes")
        
        # Check if this is CSV content
        content_type = response.headers.get('content-type', '').lower()
        if 'csv' in content_type or 'text/csv' in content_type:
            print("✅ SUCCESS: Received CSV content!")
            
            # Save a sample
            sample_content = response.text[:1000]
            print(f"📄 Sample CSV content:\n{sample_content}")
            
            # Try to save the full CSV
            filename = "test_export.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"💾 Saved CSV to: {filename}")
            
        elif response.status_code == 200 and 'html' in content_type:
            print("⚠️ Received HTML instead of CSV - checking for download links...")
            
            # Parse the response to see if there are download links
            result_soup = BeautifulSoup(response.content, 'html.parser')
            download_links = result_soup.find_all('a', href=re.compile(r'csv|download', re.IGNORECASE))
            
            if download_links:
                print(f"🔗 Found {len(download_links)} potential download links:")
                for link in download_links:
                    href = link.get('href')
                    text = link.text.strip()
                    print(f"  • {text}: {href}")
            else:
                print("❌ No download links found in response")
                
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print(f"📄 Response preview: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error during CSV export test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_export()
