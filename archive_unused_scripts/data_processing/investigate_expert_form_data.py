#!/usr/bin/env python3
"""
Expert Form Data Investigation Script
===================================

This script investigates the expert-form pages on thedogs.com.au to determine
what additional data fields are available beyond the standard form guide CSV.

The goal is to identify enrichment opportunities for race and dog data.
"""

import csv
import io
import json
import re
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class ExpertFormInvestigator:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        print("🔍 Expert Form Data Investigator initialized")
    
    def investigate_expert_form_structure(self, race_url):
        """Investigate the structure and available data in expert-form pages"""
        
        print(f"\n🔍 INVESTIGATING EXPERT FORM DATA")
        print("=" * 60)
        print(f"🌐 Race URL: {race_url}")
        
        # Construct expert-form URL
        base_race_url = race_url.split('?')[0]
        expert_form_url = f"{base_race_url}/expert-form"
        
        print(f"🔗 Expert Form URL: {expert_form_url}")
        
        try:
            # Get the expert-form page
            response = self.session.get(expert_form_url, timeout=30)
            if response.status_code != 200:
                print(f"❌ Failed to access expert-form page: {response.status_code}")
                return
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Analyze the page structure
            self.analyze_page_structure(soup)
            
            # Try to download the CSV data
            csv_data = self.attempt_csv_download(expert_form_url, soup)
            
            if csv_data:
                self.analyze_csv_structure(csv_data)
            
            # Look for additional data fields not in CSV
            self.find_additional_data_fields(soup)
            
        except Exception as e:
            print(f"❌ Error investigating expert form: {e}")
    
    def analyze_page_structure(self, soup):
        """Analyze the HTML structure to identify data sections"""
        print(f"\n📊 PAGE STRUCTURE ANALYSIS")
        print("-" * 40)
        
        # Find all sections with class containing 'expert' or 'form'
        expert_sections = soup.find_all(['div', 'section'], class_=re.compile(r'expert|form', re.I))
        print(f"Found {len(expert_sections)} expert/form related sections")
        
        for i, section in enumerate(expert_sections[:10]):  # Limit to first 10
            classes = section.get('class', [])
            print(f"  Section {i+1}: {' '.join(classes)}")
            
            # Check for data attributes
            data_attrs = {k: v for k, v in section.attrs.items() if k.startswith('data-')}
            if data_attrs:
                print(f"    Data attributes: {data_attrs}")
        
        # Look for forms
        forms = soup.find_all('form')
        print(f"\nFound {len(forms)} forms:")
        
        for i, form in enumerate(forms):
            action = form.get('action', '')
            method = form.get('method', 'GET')
            print(f"  Form {i+1}: {method.upper()} {action}")
            
            # Check for hidden inputs with data
            hidden_inputs = form.find_all('input', {'type': 'hidden'})
            for inp in hidden_inputs:
                name = inp.get('name', '')
                value = inp.get('value', '')
                if name and len(str(value)) > 10:  # Look for substantial data
                    print(f"    Hidden input '{name}': {str(value)[:50]}...")
        
        # Look for data tables
        tables = soup.find_all('table')
        print(f"\nFound {len(tables)} tables:")
        
        for i, table in enumerate(tables):
            headers = table.find_all('th')
            if headers:
                header_texts = [th.get_text().strip() for th in headers]
                print(f"  Table {i+1}: {header_texts}")
                
                # Check if this looks like performance data
                performance_keywords = ['time', 'speed', 'rating', 'points', 'score', 'margin', 'odds']
                if any(keyword in ' '.join(header_texts).lower() for keyword in performance_keywords):
                    print(f"    🎯 Potential performance data table!")
                    
                    # Get sample rows
                    rows = table.find_all('tr')[1:3]  # Skip header, get first 2 data rows
                    for j, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        cell_data = [cell.get_text().strip() for cell in cells]
                        print(f"    Row {j+1}: {cell_data}")
    
    def attempt_csv_download(self, expert_form_url, soup):
        """Attempt to download CSV data using form submission"""
        print(f"\n📥 CSV DOWNLOAD ATTEMPT")
        print("-" * 40)
        
        # Find the CSV export form
        csv_form = None
        for form in soup.find_all('form'):
            if form.find('input', {'name': 'export_csv'}) or form.find('button', {'name': 'export_csv'}):
                csv_form = form
                break
        
        if not csv_form:
            print("❌ No CSV export form found")
            return None
        
        print("✅ Found CSV export form")
        
        # Extract form data
        form_action = csv_form.get('action', expert_form_url)
        if form_action and not form_action.startswith('http'):
            form_action = f"{self.base_url}{form_action}"
        elif not form_action:
            form_action = expert_form_url
        form_method = csv_form.get('method', 'GET').upper()
        
        # Build form data
        form_data = {}
        
        # Get all input fields
        inputs = csv_form.find_all(['input', 'select', 'textarea'])
        for inp in inputs:
            name = inp.get('name')
            if not name:
                continue
            
            input_type = inp.get('type', 'text').lower()
            
            if input_type == 'hidden':
                form_data[name] = inp.get('value', '')
            elif input_type == 'text':
                form_data[name] = inp.get('value', '')
            elif input_type == 'checkbox' and inp.get('checked'):
                form_data[name] = inp.get('value', 'on')
            elif input_type == 'radio' and inp.get('checked'):
                form_data[name] = inp.get('value', '')
            elif input_type == 'submit' and name == 'export_csv':
                form_data[name] = 'true'
        
        # Add CSV export parameter
        form_data['export_csv'] = 'true'
        
        print(f"Form method: {form_method}")
        print(f"Form action: {form_action}")
        print(f"Form data keys: {list(form_data.keys())}")
        
        try:
            # Submit the form
            if form_method == 'POST':
                response = self.session.post(form_action, data=form_data, timeout=30)
            else:
                response = self.session.get(form_action, params=form_data, timeout=30)
            
            print(f"Response status: {response.status_code}")
            print(f"Content type: {response.headers.get('content-type', '')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                
                if 'csv' in content_type or 'text/plain' in content_type or 'download' in content_type:
                    print("✅ Successfully downloaded CSV data!")
                    return response.text
                elif 'text/html' in content_type:
                    # Check if HTML contains CSV data
                    if 'content-disposition' in response.headers:
                        print("✅ CSV data in HTML response with content-disposition!")
                        return response.text
                    else:
                        print("⚠️ Got HTML response instead of CSV")
                        # Sometimes CSV is embedded in HTML, let's check
                        if '\n' in response.text and ',' in response.text:
                            lines = response.text.split('\n')
                            csv_like_lines = [line for line in lines if line.count(',') > 3]
                            if len(csv_like_lines) > 5:
                                print("🔍 Found CSV-like data embedded in HTML")
                                return '\n'.join(csv_like_lines)
                else:
                    # Handle any other content type that might contain CSV data
                    if '\n' in response.text and ',' in response.text:
                        print("🔍 Found CSV-like data in non-standard content type")
                        return response.text
            
            return None
            
        except Exception as e:
            print(f"❌ Error downloading CSV: {e}")
            return None
    
    def analyze_csv_structure(self, csv_data):
        """Analyze the structure of downloaded CSV data"""
        print(f"\n📊 CSV DATA ANALYSIS")
        print("-" * 40)
        
        try:
            # Clean up the CSV data
            lines = csv_data.strip().split('\n')
            
            # Find the actual CSV content (skip HTML if present)
            csv_start = 0
            for i, line in enumerate(lines):
                if ',' in line and not line.strip().startswith('<'):
                    csv_start = i
                    break
            
            clean_csv = '\n'.join(lines[csv_start:])
            
            # Check if this is a redirect URL instead of actual CSV
            if len(lines) == 1 and lines[0].startswith('https://'):
                redirect_url = lines[0]
                print(f"🔄 Found redirect URL, following it: {redirect_url}")
                
                # Follow the redirect
                try:
                    redirect_response = self.session.get(redirect_url, timeout=30)
                    if redirect_response.status_code == 200:
                        print(f"✅ Successfully followed redirect")
                        clean_csv = redirect_response.text
                    else:
                        print(f"❌ Failed to follow redirect: {redirect_response.status_code}")
                        return
                except Exception as e:
                    print(f"❌ Error following redirect: {e}")
                    return
            
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(clean_csv))
            rows = list(csv_reader)
            
            if not rows:
                print("❌ No CSV data found")
                return
            
            headers = rows[0]
            print(f"✅ Found CSV with {len(headers)} columns and {len(rows)-1} data rows")
            print(f"📋 Headers: {headers}")
            
            # Analyze each column
            print(f"\n📊 COLUMN ANALYSIS:")
            print("-" * 40)
            
            for i, header in enumerate(headers):
                # Get sample values from this column
                values = []
                for row in rows[1:6]:  # First 5 data rows
                    if i < len(row):
                        values.append(row[i])
                
                print(f"  {i+1:2d}. {header:20} | Sample values: {values}")
                
                # Analyze data type
                non_empty_values = [v for v in values if v and v.strip()]
                if non_empty_values:
                    # Check if numeric
                    numeric_count = sum(1 for v in non_empty_values if self.is_numeric(v))
                    if numeric_count == len(non_empty_values):
                        print(f"      → Type: NUMERIC")
                    elif any(keyword in header.lower() for keyword in ['date', 'time']):
                        print(f"      → Type: DATE/TIME")
                    else:
                        print(f"      → Type: TEXT")
            
            # Show sample data rows
            print(f"\n📋 SAMPLE DATA ROWS:")
            print("-" * 40)
            for i, row in enumerate(rows[1:4]):  # Show first 3 data rows
                print(f"  Row {i+1}: {row}")
            
        except Exception as e:
            print(f"❌ Error analyzing CSV: {e}")
    
    def find_additional_data_fields(self, soup):
        """Look for additional data fields not available in standard CSV"""
        print(f"\n🔍 ADDITIONAL DATA FIELDS SEARCH")
        print("-" * 40)
        
        # Look for JavaScript data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = script.string
                
                # Look for JSON data structures
                json_matches = re.findall(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, dict) and len(data) > 2:
                            print(f"🔍 Found JSON data structure:")
                            for key, value in list(data.items())[:5]:  # Show first 5 items
                                print(f"  {key}: {str(value)[:50]}...")
                    except json.JSONDecodeError:
                        continue
                
                # Look for specific data patterns
                patterns = {
                    'ratings': r'rating["\']?\s*:\s*([0-9.]+)',
                    'speeds': r'speed["\']?\s*:\s*([0-9.]+)',
                    'times': r'time["\']?\s*:\s*([0-9.]+)',
                    'weights': r'weight["\']?\s*:\s*([0-9.]+)',
                    'odds': r'odds["\']?\s*:\s*([0-9.]+)',
                    'margins': r'margin["\']?\s*:\s*([0-9.]+)'
                }
                
                for data_type, pattern in patterns.items():
                    matches = re.findall(pattern, content, re.I)
                    if matches:
                        print(f"📊 Found {data_type} data: {matches[:5]}...")  # Show first 5
        
        # Look for data-* attributes
        elements_with_data = soup.find_all(attrs={'data-dog': True})
        elements_with_data.extend(soup.find_all(attrs={'data-race': True}))
        elements_with_data.extend(soup.find_all(attrs={'data-time': True}))
        elements_with_data.extend(soup.find_all(attrs={'data-rating': True}))
        elements_with_data.extend(soup.find_all(attrs={'data-speed': True}))
        
        if elements_with_data:
            print(f"\n📊 DATA ATTRIBUTES FOUND:")
            print("-" * 40)
            for elem in elements_with_data[:10]:  # Show first 10
                data_attrs = {k: v for k, v in elem.attrs.items() if k.startswith('data-')}
                print(f"  {elem.name}: {data_attrs}")
        
        # Look for microdata or structured data
        microdata = soup.find_all(attrs={'itemtype': True})
        if microdata:
            print(f"\n📊 MICRODATA FOUND:")
            print("-" * 40)
            for item in microdata[:5]:  # Show first 5
                itemtype = item.get('itemtype', '')
                print(f"  Type: {itemtype}")
                
                props = item.find_all(attrs={'itemprop': True})
                for prop in props[:3]:  # Show first 3 properties
                    prop_name = prop.get('itemprop', '')
                    prop_value = prop.get_text().strip()
                    print(f"    {prop_name}: {prop_value}")
    
    def is_numeric(self, value):
        """Check if a value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def investigate_multiple_races(self, race_urls):
        """Investigate multiple race URLs to find patterns"""
        print(f"\n🔍 INVESTIGATING MULTIPLE RACES")
        print("=" * 60)
        
        for i, race_url in enumerate(race_urls[:3]):  # Limit to 3 races
            print(f"\n--- RACE {i+1} ---")
            self.investigate_expert_form_structure(race_url)
            
            if i < len(race_urls) - 1:
                print(f"\n⏸️  Waiting 2 seconds before next race...")
                import time
                time.sleep(2)

def main():
    """Main function to run the investigation"""
    investigator = ExpertFormInvestigator()
    
    # Test with some example race URLs
    test_races = [
        "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f",
        # Add more test URLs as needed
    ]
    
    investigator.investigate_multiple_races(test_races)
    
    print(f"\n🏁 INVESTIGATION COMPLETE")
    print("=" * 60)
    print("Summary of findings will help determine what additional data")
    print("can be extracted to enrich race and dog analysis.")

if __name__ == "__main__":
    main()
