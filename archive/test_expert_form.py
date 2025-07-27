#!/usr/bin/env python3
"""
Test script to examine expert-form pages for CSV download capabilities
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

def test_expert_form_page():
    """Test the expert-form page structure and CSV download options"""
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Test URL from your example
    base_url = "https://www.thedogs.com.au"
    expert_form_url = "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f/expert-form?expert_form%5Bsort_by%5D=&expert_form%5Bsort_dir%5D="
    
    print("🔍 Testing Expert Form Page Structure")
    print("=" * 60)
    print(f"🌐 URL: {expert_form_url}")
    
    try:
        response = session.get(expert_form_url, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed to access page: {response.status_code}")
            return
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("\n🔍 LOOKING FOR CSV/EXPORT ELEMENTS:")
        print("-" * 40)
        
        # Look for CSV download elements
        csv_elements = soup.find_all(['a', 'button', 'input'], 
                                   attrs={'value': re.compile(r'csv|export', re.IGNORECASE)})
        
        # Also check for links with CSV in href
        csv_links = soup.find_all('a', href=re.compile(r'csv|export', re.IGNORECASE))
        csv_elements.extend(csv_links)
        
        # Check for any text containing "csv" or "export"
        csv_text_elements = soup.find_all(text=re.compile(r'csv|export', re.IGNORECASE))
        
        if csv_elements:
            print(f"✅ Found {len(csv_elements)} CSV-related elements:")
            for i, element in enumerate(csv_elements):
                print(f"  {i+1}. {element.name}: {element.get('href') or element.get('value') or element.text[:50]}")
        else:
            print("❌ No CSV-related elements found")
            
        if csv_text_elements:
            print(f"\n📝 Found {len(csv_text_elements)} text mentions of CSV/export:")
            for i, text in enumerate(csv_text_elements[:5]):  # Show first 5
                print(f"  {i+1}. {text.strip()[:100]}")
        
        # Look for forms that might be relevant
        print("\n🔍 CHECKING FORMS:")
        print("-" * 40)
        
        forms = soup.find_all('form')
        print(f"Found {len(forms)} forms on the page")
        
        for i, form in enumerate(forms):
            action = form.get('action')
            method = form.get('method', 'GET')
            print(f"  Form {i+1}: {method} {action}")
            
            # Check form inputs
            inputs = form.find_all(['input', 'button', 'select'])
            for input_elem in inputs:
                input_type = input_elem.get('type', 'text')
                input_name = input_elem.get('name', '')
                input_value = input_elem.get('value', '')
                
                if 'csv' in input_name.lower() or 'export' in input_name.lower() or \
                   'csv' in input_value.lower() or 'export' in input_value.lower():
                    print(f"    📋 Relevant input: {input_type} name='{input_name}' value='{input_value}'")
        
        # Check for specific CSS selector mentioned
        print("\n🔍 CHECKING SPECIFIC SELECTOR:")
        print("-" * 40)
        
        specific_element = soup.select_one("body > div.page__layout > div.content-section-tight.expert-form__preferences__form > form > fieldset > div > div:nth-child(8)")
        if specific_element:
            print(f"✅ Found specific element: {specific_element.name}")
            print(f"   Content: {specific_element.text[:100]}")
            print(f"   HTML: {str(specific_element)[:200]}")
        else:
            print("❌ Could not find specific selector element")
            
        # Check for any tables with form data
        print("\n🔍 CHECKING FOR FORM DATA TABLES:")
        print("-" * 40)
        
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables on the page")
        
        for i, table in enumerate(tables):
            headers = table.find_all('th')
            if headers:
                header_text = [th.text.strip() for th in headers[:5]]  # First 5 headers
                print(f"  Table {i+1} headers: {header_text}")
                
                # Check if this looks like form guide data
                form_guide_keywords = ['dog', 'name', 'weight', 'time', 'trainer', 'jockey', 'barrier', 'box']
                if any(keyword in ' '.join(header_text).lower() for keyword in form_guide_keywords):
                    print(f"    🎯 This looks like form guide data!")
                    
                    # Get a sample row
                    first_row = table.find('tr')
                    if first_row:
                        cells = first_row.find_all(['td', 'th'])
                        if len(cells) > 1:
                            row_data = [cell.text.strip() for cell in cells[:5]]
                            print(f"    Sample row: {row_data}")
        
        # Try some common CSV export patterns
        print("\n🔍 TESTING COMMON CSV EXPORT PATTERNS:")
        print("-" * 40)
        
        export_patterns = [
            expert_form_url + "&export=csv",
            expert_form_url + "&format=csv",
            expert_form_url.replace('expert-form', 'expert-form.csv'),
            expert_form_url + ".csv",
            expert_form_url.replace('?expert_form', '.csv?expert_form')
        ]
        
        for pattern in export_patterns:
            try:
                test_response = session.head(pattern, timeout=10)
                content_type = test_response.headers.get('content-type', '').lower()
                print(f"  {pattern}")
                print(f"    Status: {test_response.status_code}, Content-Type: {content_type}")
                
                if test_response.status_code == 200 and ('csv' in content_type or 'text' in content_type):
                    print(f"    🎯 POTENTIAL CSV DOWNLOAD FOUND!")
                    # Try to get a sample
                    content_response = session.get(pattern, timeout=10)
                    if content_response.status_code == 200:
                        sample_content = content_response.text[:500]
                        print(f"    Sample content: {sample_content}")
                        
            except Exception as e:
                print(f"    ❌ Error testing pattern: {e}")
                
        print("\n" + "=" * 60)
        print("🏁 Expert Form Page Analysis Complete")
        
    except Exception as e:
        print(f"❌ Error accessing expert form page: {e}")

if __name__ == "__main__":
    test_expert_form_page()
