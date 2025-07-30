#!/usr/bin/env python3
# LEGACY: This script is archived and no longer in active use.
# WARNING: This script may be incompatible with current database schema.
# Do not run without consulting the archive documentation.
"""
Test Script to Check What the Scraper Actually Finds
===================================================

This script tests the website structure to understand what's available.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import re

def test_website_structure():
    """Test what we can find on the website"""
    
    # Test with today's date
    today = date.today()
    date_str = today.strftime('%Y-%m-%d')
    base_url = f"https://www.thedogs.com.au/racing/{date_str}"
    
    print(f"🔍 Testing website structure for {date_str}")
    print(f"📍 URL: {base_url}")
    print("=" * 60)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        # Get the racing page
        response = session.get(base_url, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed to access website: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check what links are available
        print("\n🔗 FOUND LINKS:")
        links = soup.find_all('a', href=True)
        racing_links = []
        
        for link in links:
            href = link.get('href')
            if href and '/racing/' in href and date_str in href:
                racing_links.append(href)
                print(f"  • {href}")
        
        print(f"\n📋 Total racing links found: {len(racing_links)}")
        
        if racing_links:
            # Test one race page
            test_url = racing_links[0]
            if not test_url.startswith('http'):
                test_url = f"https://www.thedogs.com.au{test_url}"
            
            print(f"\n🔍 Testing individual race page: {test_url}")
            
            race_response = session.get(test_url, timeout=30)
            print(f"📊 Race page status: {race_response.status_code}")
            
            if race_response.status_code == 200:
                race_soup = BeautifulSoup(race_response.content, 'html.parser')
                
                # Look for CSV-related elements
                print("\n📄 LOOKING FOR CSV ELEMENTS:")
                csv_elements = race_soup.find_all(['a', 'button', 'div'], text=re.compile(r'csv|export|download', re.IGNORECASE))
                
                if csv_elements:
                    for element in csv_elements:
                        print(f"  • {element.name}: {element.get_text(strip=True)}")
                        if element.get('href'):
                            print(f"    Link: {element.get('href')}")
                else:
                    print("  ❌ No CSV-related elements found")
                
                # Check for form guide table
                print("\n📋 LOOKING FOR FORM GUIDE DATA:")
                tables = race_soup.find_all('table')
                print(f"  Found {len(tables)} tables")
                
                # Look for dog names
                dog_elements = race_soup.find_all(text=re.compile(r'\d+\.\s*[A-Z]', re.IGNORECASE))
                if dog_elements:
                    print(f"  Found {len(dog_elements)} potential dog entries")
                    for dog in dog_elements[:3]:  # Show first 3
                        print(f"    • {dog.strip()}")
                
                # Check page title
                title = race_soup.find('title')
                if title:
                    print(f"\n📝 Page title: {title.get_text(strip=True)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def check_existing_csv_structure():
    """Check what the existing CSV files look like"""
    import os
    
    print("\n" + "=" * 60)
    print("🔍 CHECKING EXISTING CSV STRUCTURE")
    print("=" * 60)
    
    processed_dir = "./form_guides/processed"
    if os.path.exists(processed_dir):
        csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        print(f"📊 Found {len(csv_files)} existing CSV files")
        
        if csv_files:
            # Check first file
            sample_file = os.path.join(processed_dir, csv_files[0])
            print(f"\n📄 Sample file: {csv_files[0]}")
            
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"  Lines: {len(lines)}")
                    print(f"  Header: {lines[0].strip()}")
                    if len(lines) > 1:
                        print(f"  Sample data: {lines[1].strip()}")
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
    else:
        print("❌ No processed directory found")

if __name__ == "__main__":
    test_website_structure()
    check_existing_csv_structure()
