#!/usr/bin/env python3
"""
Enhanced Expert Form Scraper for thedogs.com.au
===============================================

This script provides comprehensive data extraction from expert-form pages on thedogs.com.au,
capturing rich performance metrics, sectional times, ratings, and other advanced analytics
that are not available in standard form guide CSVs.

Features:
- Extract enhanced CSV data with all available columns
- Parse embedded JSON data from JavaScript variables
- Extract sectional times and performance breakdowns
- Capture detailed ratings, speeds, odds, and margins
- Extract microdata and structured semantic information
- Handle redirects and various content types
- Save enriched data for enhanced ML modeling

Author: AI Assistant
Date: July 25, 2025
"""

import os
import sys
import requests
import time
import random
import json
import re
import csv
import io
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any

class EnhancedExpertFormScraper:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        self.enhanced_data_dir = "./enhanced_expert_data"
        self.csv_data_dir = "./enhanced_expert_data/csv"
        self.json_data_dir = "./enhanced_expert_data/json"
        self.database_path = "./databases/comprehensive_greyhound_data.db"
        
        # Create directories
        os.makedirs(self.enhanced_data_dir, exist_ok=True)
        os.makedirs(self.csv_data_dir, exist_ok=True)
        os.makedirs(self.json_data_dir, exist_ok=True)
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Venue mapping
        self.venue_map = {
            'angle-park': 'AP_K',
            'sandown': 'SAN',
            'warrnambool': 'WAR',
            'bendigo': 'BEN',
            'geelong': 'GEE',
            'ballarat': 'BAL',
            'horsham': 'HOR',
            'traralgon': 'TRA',
            'dapto': 'DAPT',
            'wentworth-park': 'WPK',
            'albion-park': 'APWE',
            'cannington': 'CANN',
            'the-meadows': 'MEA',
            'healesville': 'HEA',
            'sale': 'SAL',
            'richmond': 'RICH',
            'murray-bridge': 'MURR',
            'gawler': 'GAWL',
            'mount-gambier': 'MOUNT',
            'northam': 'NOR',
            'mandurah': 'MAND',
            'casino': 'CASO',
            'goulburn': 'GOULBURN',
            'warragul': 'WARRAGUL',
            'temora': 'TEMORA',
            'gunnedah': 'GUNNEDAH',
            'hobart': 'HOBT',
            'ladbrokes-q-straight': 'LADBROKES-Q-STRAIGHT',
            'gardens': 'GRDN'
        }
        
        print("🏁 Enhanced Expert Form Scraper initialized")
        print(f"📂 Enhanced data directory: {self.enhanced_data_dir}")
        print(f"📊 CSV data directory: {self.csv_data_dir}")
        print(f"📋 JSON data directory: {self.json_data_dir}")
    
    def extract_comprehensive_race_data(self, race_url: str) -> Dict[str, Any]:
        """Extract comprehensive race data from expert form page"""
        print(f"\n🔍 EXTRACTING COMPREHENSIVE DATA")
        print("=" * 60)
        print(f"🌐 Race URL: {race_url}")
        
        # Construct expert-form URL
        base_race_url = race_url.split('?')[0]
        expert_form_url = f"{base_race_url}/expert-form"
        
        try:
            # Get the expert-form page
            response = self.session.get(expert_form_url, timeout=30)
            if response.status_code != 200:
                print(f"❌ Failed to access expert-form page: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract race information
            race_info = self.extract_race_info(soup, race_url)
            if not race_info:
                print("❌ Could not extract race information")
                return {}
            
            # Build comprehensive data structure
            comprehensive_data = {
                'race_info': race_info,
                'expert_form_url': expert_form_url,
                'extraction_timestamp': datetime.now().isoformat(),
                'csv_data': None,
                'embedded_json': {},
                'sectional_data': {},
                'performance_tables': [],
                'data_attributes': {},
                'microdata': {},
                'additional_metrics': {}
            }
            
            # Extract CSV data
            csv_data = self.extract_enhanced_csv_data(expert_form_url, soup)
            if csv_data:
                comprehensive_data['csv_data'] = csv_data
                print("✅ Enhanced CSV data extracted")
            
            # Extract embedded JSON data
            embedded_json = self.extract_embedded_json_data(soup)
            if embedded_json:
                comprehensive_data['embedded_json'] = embedded_json
                print(f"✅ Extracted {len(embedded_json)} JSON data structures")
            
            # Extract sectional times and performance data
            performance_data = self.extract_performance_data(soup)
            if performance_data:
                comprehensive_data.update(performance_data)
                print("✅ Performance data extracted")
            
            # Extract data attributes
            data_attributes = self.extract_data_attributes(soup)
            if data_attributes:
                comprehensive_data['data_attributes'] = data_attributes
                print(f"✅ Extracted {len(data_attributes)} data attributes")
            
            # Extract microdata
            microdata = self.extract_microdata(soup)
            if microdata:
                comprehensive_data['microdata'] = microdata
                print(f"✅ Extracted {len(microdata)} microdata items")
            
            # Extract additional metrics
            additional_metrics = self.extract_additional_metrics(soup)
            if additional_metrics:
                comprehensive_data['additional_metrics'] = additional_metrics
                print(f"✅ Extracted additional metrics")
            
            return comprehensive_data
            
        except Exception as e:
            print(f"❌ Error extracting comprehensive data: {e}")
            return {}
    
    def extract_race_info(self, soup: BeautifulSoup, race_url: str) -> Dict[str, str]:
        """Extract basic race information"""
        try:
            # Extract race number from URL
            url_parts = race_url.split('/')
            race_number = None
            venue = None
            date = None
            
            # Find race number in URL
            for i, part in enumerate(url_parts):
                if part.isdigit() and i > 0:
                    race_number = part
                    break
            
            # Extract venue from URL
            for venue_key, venue_code in self.venue_map.items():
                if venue_key in race_url:
                    venue = venue_code
                    break
            
            # Extract date from URL
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', race_url)
            if date_match:
                date_obj = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                date = date_obj.strftime('%d %B %Y')
            
            # Try to extract additional info from page content
            race_name = None
            grade = None
            distance = None
            
            # Extract race name
            title_selectors = ['h1', '.race-title', '.race-name', '.page-title']
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element:
                    race_name = element.get_text(strip=True)
                    break
            
            # Extract grade and distance from race name or other elements
            if race_name:
                grade_match = re.search(r'(Grade \d+|Maiden|Free For All|FFA|Group \d+)', race_name, re.I)
                if grade_match:
                    grade = grade_match.group(1)
                
                distance_match = re.search(r'(\d+m)', race_name)
                if distance_match:
                    distance = distance_match.group(1)
            
            if race_number and venue and date:
                return {
                    'race_number': race_number,
                    'venue': venue,
                    'date': date,
                    'race_name': race_name or '',
                    'grade': grade or '',
                    'distance': distance or '',
                    'race_id': f"{venue}_{race_number}_{datetime.strptime(date, '%d %B %Y').strftime('%Y-%m-%d')}"
                }
            
            return {}
            
        except Exception as e:
            print(f"❌ Error extracting race info: {e}")
            return {}
    
    def extract_enhanced_csv_data(self, expert_form_url: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract enhanced CSV data using form submission with comprehensive handling"""
        print(f"📥 Extracting enhanced CSV data...")
        
        try:
            # Find the CSV export form
            csv_form = None
            for form in soup.find_all('form'):
                if (form.find('input', {'name': 'export_csv'}) or 
                    form.find('button', {'name': 'export_csv'}) or
                    'csv' in str(form).lower() or
                    'export' in str(form).lower()):
                    csv_form = form
                    break
            
            if not csv_form:
                print("⚠️ No CSV export form found")
                return None
            
            # Extract form data
            form_action = csv_form.get('action', expert_form_url)
            if form_action and not form_action.startswith('http'):
                if form_action.startswith('/'):
                    form_action = f"{self.base_url}{form_action}"
                else:
                    form_action = f"{self.base_url}/{form_action}"
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
            
            # Submit the form
            if form_method == 'POST':
                response = self.session.post(form_action, data=form_data, timeout=30)
            else:
                response = self.session.get(form_action, params=form_data, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Form submission failed: {response.status_code}")
                return None
            
            # Handle different response types
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if response is a redirect URL
            response_text = response.text.strip()
            if (len(response_text.split('\n')) == 1 and 
                response_text.startswith('http')):
                print(f"🔄 Following redirect: {response_text}")
                redirect_response = self.session.get(response_text, timeout=30)
                if redirect_response.status_code == 200:
                    response_text = redirect_response.text
                else:
                    print(f"❌ Failed to follow redirect: {redirect_response.status_code}")
                    return None
            
            # Parse CSV data
            return self.parse_csv_data(response_text)
            
        except Exception as e:
            print(f"❌ Error extracting CSV data: {e}")
            return None
    
    def parse_csv_data(self, csv_content: str) -> Dict[str, Any]:
        """Parse CSV content and analyze structure"""
        try:
            # Clean up the CSV data
            lines = csv_content.strip().split('\n')
            
            # Find the actual CSV content (skip HTML if present)
            csv_start = 0
            for i, line in enumerate(lines):
                if ',' in line and not line.strip().startswith('<'):
                    csv_start = i
                    break
            
            clean_csv = '\n'.join(lines[csv_start:])
            
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(clean_csv))
            rows = list(csv_reader)
            
            if not rows:
                print("❌ No CSV data found")
                return {}
            
            headers = rows[0]
            data_rows = rows[1:]
            
            # Analyze CSV structure
            csv_analysis = {
                'headers': headers,
                'row_count': len(data_rows),
                'column_count': len(headers),
                'sample_data': data_rows[:3] if data_rows else [],
                'column_analysis': {}
            }
            
            # Analyze each column
            for i, header in enumerate(headers):
                values = []
                for row in data_rows[:10]:  # Analyze first 10 rows
                    if i < len(row):
                        values.append(row[i])
                
                non_empty_values = [v for v in values if v and v.strip()]
                
                column_info = {
                    'sample_values': values,
                    'non_empty_count': len(non_empty_values),
                    'data_type': 'unknown'
                }
                
                if non_empty_values:
                    # Determine data type
                    numeric_count = sum(1 for v in non_empty_values if self.is_numeric(v))
                    if numeric_count == len(non_empty_values):
                        column_info['data_type'] = 'numeric'
                    elif any(keyword in header.lower() for keyword in ['date', 'time']):
                        column_info['data_type'] = 'datetime'
                    else:
                        column_info['data_type'] = 'text'
                
                csv_analysis['column_analysis'][header] = column_info
            
            print(f"✅ Parsed CSV: {len(headers)} columns, {len(data_rows)} rows")
            return csv_analysis
            
        except Exception as e:
            print(f"❌ Error parsing CSV data: {e}")
            return {}
    
    def extract_embedded_json_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON data embedded in JavaScript variables"""
        embedded_data = {}
        
        try:
            scripts = soup.find_all('script')
            for i, script in enumerate(scripts):
                if script.string:
                    content = script.string
                    
                    # Look for various JSON patterns
                    patterns = {
                        'race_data': r'raceData\s*[=:]\s*(\{.*?\});?$',
                        'dog_data': r'dogData\s*[=:]\s*(\[.*?\]);?$',
                        'performance_data': r'performanceData\s*[=:]\s*(\{.*?\});?$',
                        'ratings_data': r'ratingsData\s*[=:]\s*(\{.*?\});?$',
                        'json_objects': r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                    }
                    
                    for data_type, pattern in patterns.items():
                        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                        for match in matches:
                            try:
                                data = json.loads(match)
                                if isinstance(data, (dict, list)) and data:
                                    key = f"{data_type}_{len(embedded_data)}"
                                    embedded_data[key] = data
                            except json.JSONDecodeError:
                                continue
                    
                    # Look for specific data patterns in arrays
                    array_patterns = {
                        'ratings': r'rating["\']?\s*:\s*([0-9.]+)',
                        'speeds': r'speed["\']?\s*:\s*([0-9.]+)',
                        'times': r'time["\']?\s*:\s*([0-9.]+)',
                        'weights': r'weight["\']?\s*:\s*([0-9.]+)',
                        'odds': r'odds["\']?\s*:\s*([0-9.]+)',
                        'margins': r'margin["\']?\s*:\s*([0-9.]+)'
                    }
                    
                    for data_type, pattern in array_patterns.items():
                        matches = re.findall(pattern, content, re.I)
                        if matches:
                            key = f"{data_type}_values"
                            embedded_data[key] = [float(m) for m in matches if self.is_numeric(m)]
            
            return embedded_data
            
        except Exception as e:
            print(f"❌ Error extracting embedded JSON: {e}")
            return {}
    
    def extract_performance_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract sectional times and performance breakdowns from tables"""
        performance_data = {
            'sectional_data': {},
            'performance_tables': [],
            'track_records': {},
            'speed_data': {}
        }
        
        try:
            # Find all tables
            tables = soup.find_all('table')
            
            for i, table in enumerate(tables):
                headers = table.find_all('th')
                if not headers:
                    continue
                
                header_texts = [th.get_text().strip().lower() for th in headers]
                
                # Check for sectional times tables
                sectional_keywords = ['400m', '500m', '600m', '700m', 'sectional', 'split']
                if any(keyword in ' '.join(header_texts) for keyword in sectional_keywords):
                    sectional_data = self.parse_sectional_table(table, header_texts)
                    if sectional_data:
                        performance_data['sectional_data'][f'table_{i}'] = sectional_data
                
                # Check for performance/rating tables
                performance_keywords = ['rating', 'speed', 'time', 'points', 'score', 'odds', 'margin']
                if any(keyword in ' '.join(header_texts) for keyword in performance_keywords):
                    table_data = self.parse_performance_table(table, header_texts)
                    if table_data:
                        performance_data['performance_tables'].append({
                            'table_id': i,
                            'headers': header_texts,
                            'data': table_data
                        })
                
                # Check for track record tables
                if 'record' in ' '.join(header_texts) or 'track' in ' '.join(header_texts):
                    record_data = self.parse_track_record_table(table, header_texts)
                    if record_data:
                        performance_data['track_records'][f'table_{i}'] = record_data
            
            return performance_data
            
        except Exception as e:
            print(f"❌ Error extracting performance data: {e}")
            return {}
    
    def parse_sectional_table(self, table: BeautifulSoup, headers: List[str]) -> Dict[str, Any]:
        """Parse sectional times table"""
        try:
            sectional_data = {
                'headers': headers,
                'dogs': []
            }
            
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells[:len(headers)]):
                        if i < len(headers):
                            value = cell.get_text().strip()
                            row_data[headers[i]] = value
                    
                    if row_data:
                        sectional_data['dogs'].append(row_data)
            
            return sectional_data
            
        except Exception as e:
            print(f"❌ Error parsing sectional table: {e}")
            return {}
    
    def parse_performance_table(self, table: BeautifulSoup, headers: List[str]) -> List[Dict[str, str]]:
        """Parse performance/rating table"""
        try:
            table_data = []
            
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells[:len(headers)]):
                        if i < len(headers):
                            value = cell.get_text().strip()
                            row_data[headers[i]] = value
                    
                    if row_data:
                        table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            print(f"❌ Error parsing performance table: {e}")
            return []
    
    def parse_track_record_table(self, table: BeautifulSoup, headers: List[str]) -> Dict[str, Any]:
        """Parse track record table"""
        try:
            record_data = {
                'headers': headers,
                'records': []
            }
            
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells[:len(headers)]):
                        if i < len(headers):
                            value = cell.get_text().strip()
                            row_data[headers[i]] = value
                    
                    if row_data:
                        record_data['records'].append(row_data)
            
            return record_data
            
        except Exception as e:
            print(f"❌ Error parsing track record table: {e}")
            return {}
    
    def extract_data_attributes(self, soup: BeautifulSoup) -> Dict[str, List[Dict[str, str]]]:
        """Extract data-* attributes from HTML elements"""
        data_attributes = {}
        
        try:
            # Look for elements with data attributes
            attribute_types = ['data-dog', 'data-race', 'data-time', 'data-rating', 'data-speed', 'data-odds', 'data-margin']
            
            for attr_type in attribute_types:
                elements = soup.find_all(attrs={attr_type: True})
                if elements:
                    data_attributes[attr_type] = []
                    for elem in elements:
                        elem_data = {
                            'tag': elem.name,
                            'text': elem.get_text().strip(),
                            'attributes': {k: v for k, v in elem.attrs.items() if k.startswith('data-')}
                        }
                        data_attributes[attr_type].append(elem_data)
            
            return data_attributes
            
        except Exception as e:
            print(f"❌ Error extracting data attributes: {e}")
            return {}
    
    def extract_microdata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract microdata and structured data"""
        microdata = {}
        
        try:
            # Look for microdata
            microdata_elements = soup.find_all(attrs={'itemtype': True})
            
            for i, elem in enumerate(microdata_elements):
                itemtype = elem.get('itemtype', '')
                item_data = {
                    'itemtype': itemtype,
                    'properties': {}
                }
                
                # Find all properties within this item
                props = elem.find_all(attrs={'itemprop': True})
                for prop in props:
                    prop_name = prop.get('itemprop', '')
                    prop_value = prop.get_text().strip()
                    
                    # Also check for content attribute
                    if prop.get('content'):
                        prop_value = prop.get('content')
                    
                    item_data['properties'][prop_name] = prop_value
                
                microdata[f'item_{i}'] = item_data
            
            return microdata
            
        except Exception as e:
            print(f"❌ Error extracting microdata: {e}")
            return {}
    
    def extract_additional_metrics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract additional metrics and analysis data"""
        additional_metrics = {}
        
        try:
            # Look for track condition information
            track_condition_patterns = [
                r'track[:\s]+([a-z]+)',
                r'condition[:\s]+([a-z]+)',
                r'going[:\s]+([a-z]+)'
            ]
            
            page_text = soup.get_text().lower()
            for pattern in track_condition_patterns:
                matches = re.findall(pattern, page_text, re.I)
                if matches:
                    additional_metrics['track_conditions'] = list(set(matches))
                    break
            
            # Look for weather information
            weather_patterns = [
                r'weather[:\s]+([a-z\s]+)',
                r'temperature[:\s]+(\d+)',
                r'wind[:\s]+([a-z\s]+)'
            ]
            
            for pattern in weather_patterns:
                matches = re.findall(pattern, page_text, re.I)
                if matches:
                    key = pattern.split('[')[0]
                    additional_metrics[key] = matches
            
            # Look for prize money
            prize_patterns = [
                r'\$([0-9,]+)',
                r'prize[:\s]+\$([0-9,]+)',
                r'total[:\s]+\$([0-9,]+)'
            ]
            
            for pattern in prize_patterns:
                matches = re.findall(pattern, page_text, re.I)
                if matches:
                    additional_metrics['prize_money'] = matches
                    break
            
            return additional_metrics
            
        except Exception as e:
            print(f"❌ Error extracting additional metrics: {e}")
            return {}
    
    def is_numeric(self, value: str) -> bool:
        """Check if a value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def save_comprehensive_data(self, comprehensive_data: Dict[str, Any]) -> bool:
        """Save comprehensive data to files"""
        try:
            race_info = comprehensive_data.get('race_info', {})
            if not race_info:
                print("❌ No race info found, cannot save data")
                return False
            
            race_id = race_info.get('race_id', 'unknown')
            venue = race_info.get('venue', 'unknown')
            race_number = race_info.get('race_number', 'unknown')
            date = race_info.get('date', 'unknown')
            
            # Create filename base
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"{venue}_Race{race_number}_{date.replace(' ', '_')}_{timestamp}"
            
            # Save comprehensive JSON data
            json_filepath = os.path.join(self.json_data_dir, f"{filename_base}_comprehensive.json")
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_data, f, indent=2, default=str)
            
            print(f"✅ Saved comprehensive data: {json_filepath}")
            
            # Save CSV data separately if available
            csv_data = comprehensive_data.get('csv_data')
            if csv_data and csv_data.get('headers'):
                csv_filepath = os.path.join(self.csv_data_dir, f"{filename_base}_enhanced.csv")
                
                headers = csv_data['headers']
                rows = []
                
                # Add header row
                rows.append(headers)
                
                # Add sample data if available
                sample_data = csv_data.get('sample_data', [])
                for row in sample_data:
                    if len(row) == len(headers):
                        rows.append(row)
                
                # Write CSV file
                with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                
                print(f"✅ Saved enhanced CSV: {csv_filepath}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving comprehensive data: {e}")
            return False
    
    def process_race_url(self, race_url: str) -> bool:
        """Process a single race URL and extract all comprehensive data"""
        print(f"\n🏁 PROCESSING RACE")
        print("=" * 60)
        print(f"🌐 URL: {race_url}")
        
        try:
            # Extract comprehensive data
            comprehensive_data = self.extract_comprehensive_race_data(race_url)
            
            if not comprehensive_data:
                print("❌ No comprehensive data extracted")
                return False
            
            # Save the data
            success = self.save_comprehensive_data(comprehensive_data)
            
            if success:
                print("✅ Race processing completed successfully")
                return True
            else:
                print("❌ Failed to save race data")
                return False
                
        except Exception as e:
            print(f"❌ Error processing race: {e}")
            return False
    
    def process_multiple_races(self, race_urls: List[str]) -> Dict[str, Any]:
        """Process multiple race URLs and return summary"""
        print(f"\n🏁 PROCESSING MULTIPLE RACES")
        print("=" * 60)
        print(f"📊 Total races to process: {len(race_urls)}")
        
        results = {
            'total_races': len(race_urls),
            'successful': 0,
            'failed': 0,
            'processed_races': [],
            'failed_races': [],
            'start_time': datetime.now().isoformat()
        }
        
        for i, race_url in enumerate(race_urls):
            print(f"\n--- RACE {i+1}/{len(race_urls)} ---")
            
            try:
                success = self.process_race_url(race_url)
                
                if success:
                    results['successful'] += 1
                    results['processed_races'].append(race_url)
                else:
                    results['failed'] += 1
                    results['failed_races'].append(race_url)
                
                # Add delay between requests
                if i < len(race_urls) - 1:
                    delay = random.uniform(2, 5)
                    print(f"⏸️  Waiting {delay:.1f} seconds before next race...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"❌ Error processing race {i+1}: {e}")
                results['failed'] += 1
                results['failed_races'].append(race_url)
        
        results['end_time'] = datetime.now().isoformat()
        results['success_rate'] = results['successful'] / results['total_races'] * 100 if results['total_races'] > 0 else 0
        
        # Save summary
        summary_filepath = os.path.join(self.enhanced_data_dir, f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎯 PROCESSING COMPLETE")
        print("=" * 60)
        print(f"📊 Results: {results['successful']}/{results['total_races']} successful ({results['success_rate']:.1f}%)")
        print(f"📁 Summary saved: {summary_filepath}")
        
        return results

def main():
    """Main function for testing"""
    scraper = EnhancedExpertFormScraper()
    
    # Test with a sample race URL
    test_races = [
        "https://www.thedogs.com.au/racing/richmond-straight/2025-07-10/4/ladbrokes-bitches-only-maiden-final-f",
        # Add more test URLs as needed
    ]
    
    if test_races:
        # Process single race for testing
        scraper.process_race_url(test_races[0])
        
        # Uncomment to process multiple races
        # scraper.process_multiple_races(test_races)
    else:
        print("⚠️ No test race URLs provided")

if __name__ == "__main__":
    main()
