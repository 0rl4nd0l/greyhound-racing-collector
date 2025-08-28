#!/usr/bin/env python3
"""
TGR HTML Structure Inspector
===========================

This script examines the actual HTML structure of TGR pages to understand
how individual dog racing histories are presented.
"""

import logging
from bs4 import BeautifulSoup
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_tgr_page():
    """Inspect a specific TGR page to understand the HTML structure."""
    
    # Use a known working TGR URL
    test_url = 'https://www.thegreyhoundrecorder.com.au/form-guides/ascot-park/long-form/244701/1/'
    
    logger.info(f"📊 Inspecting TGR page structure: {test_url}")
    
    try:
        response = requests.get(test_url)
        time.sleep(2)  # Rate limiting
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all text that mentions "Mayfield Star"
            logger.info("\n🔍 Searching for 'Mayfield Star' mentions...")
            
            mayfield_elements = soup.find_all(string=lambda text: text and 'mayfield star' in text.lower())
            
            for i, element in enumerate(mayfield_elements):
                logger.info(f"\nMayfield Star mention #{i+1}:")
                logger.info(f"  Text: {element.strip()}")
                
                # Get the parent element
                parent = element.parent if hasattr(element, 'parent') else None
                if parent:
                    logger.info(f"  Parent tag: <{parent.name}> with classes: {parent.get('class', [])}")
                    
                    # Look for siblings that might contain racing data
                    logger.info("  Next siblings:")
                    for j, sibling in enumerate(parent.find_next_siblings()):
                        if j >= 5:  # Limit to first 5 siblings
                            break
                        sibling_text = sibling.get_text(strip=True)
                        if len(sibling_text) > 0:
                            logger.info(f"    {j+1}. <{sibling.name}> {sibling.get('class', [])}: {sibling_text[:100]}...")
                    
                    # Look for child elements
                    logger.info("  Child elements:")
                    children = parent.find_all() if hasattr(parent, 'find_all') else []
                    for j, child in enumerate(children[:10]):  # Limit to first 10 children
                        child_text = child.get_text(strip=True)
                        if len(child_text) > 0:
                            logger.info(f"    {j+1}. <{child.name}> {child.get('class', [])}: {child_text[:100]}...")
            
            # Look for patterns that might indicate racing history
            logger.info("\n🏁 Looking for racing history patterns...")
            
            # Check for date patterns
            import re
            date_patterns = [
                r'\d{2}/\d{2}/\d{2}',  # DD/MM/YY
                r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # 25th Jul
            ]
            
            for pattern in date_patterns:
                matches = soup.find_all(string=re.compile(pattern))
                logger.info(f"\nDate pattern '{pattern}' found {len(matches)} times:")
                for i, match in enumerate(matches[:5]):  # Show first 5
                    logger.info(f"  {i+1}. {match.strip()}")
                    parent = match.parent if hasattr(match, 'parent') else None
                    if parent:
                        logger.info(f"     In: <{parent.name}> {parent.get('class', [])}")
            
            # Look for specific racing-related structures
            logger.info("\n🏆 Looking for racing data structures...")
            
            # Find all list items
            list_items = soup.find_all('li')
            racing_list_items = []
            
            for li in list_items:
                text = li.get_text().lower()
                if any(word in text for word in ['mayfield star', 'salted caramel', 'ballarat', 'bendigo', '1st', '2nd', '3rd']):
                    racing_list_items.append(li)
            
            logger.info(f"Found {len(racing_list_items)} list items with racing content:")
            for i, li in enumerate(racing_list_items[:10]):  # Show first 10
                logger.info(f"  {i+1}. {li.get_text(strip=True)[:150]}...")
            
            # Find all table rows
            table_rows = soup.find_all('tr')
            racing_table_rows = []
            
            for tr in table_rows:
                text = tr.get_text().lower()
                if any(word in text for word in ['mayfield star', 'salted caramel', 'ballarat', 'bendigo']):
                    racing_table_rows.append(tr)
            
            logger.info(f"Found {len(racing_table_rows)} table rows with racing content:")
            for i, tr in enumerate(racing_table_rows[:5]):  # Show first 5
                cells = tr.find_all(['td', 'th'])
                logger.info(f"  {i+1}. {len(cells)} cells: {[cell.get_text(strip=True) for cell in cells[:6]]}")
            
            # Look for specific div/section structures
            logger.info("\n📋 Looking for structured sections...")
            
            divs_with_classes = soup.find_all('div', class_=True)
            for div in divs_with_classes[:20]:  # Check first 20 divs with classes
                text = div.get_text().lower()
                if 'mayfield star' in text or any(venue in text for venue in ['ballarat', 'bendigo', 'sandown']):
                    logger.info(f"Relevant div with class {div.get('class')}: {text[:100]}...")
            
        else:
            logger.error(f"Failed to fetch page: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error inspecting page: {e}")

def main():
    """Main function."""
    inspect_tgr_page()

if __name__ == "__main__":
    main()
