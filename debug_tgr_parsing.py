#!/usr/bin/env python3
"""
Debug TGR Parsing
=================

This script debugs why the TGR scraper is finding 0 races for every dog
even though it successfully fetches 62 meetings.
"""

import logging
from src.collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_tgr_parsing():
    """Debug the TGR parsing to see what's going wrong."""
    
    logger.info("🔍 Debugging TGR parsing issues...")
    
    # Initialize the scraper
    scraper = TheGreyhoundRecorderScraper(rate_limit=2.0, use_cache=True)
    
    # Get form guides
    logger.info("📋 Fetching form guides...")
    form_guides = scraper.fetch_form_guides()
    
    logger.info(f"📊 Found {len(form_guides.get('meetings', []))} meetings")
    
    # Test with a known dog name that should exist
    test_dog = "Prime Of Life"
    
    # Try the first few meetings
    for i, meeting in enumerate(form_guides.get('meetings', [])[:3]):
        logger.info(f"\n🎯 Testing meeting {i+1}: {meeting.get('meeting_title', 'Unknown')}")
        logger.info(f"   URL: {meeting.get('long_form_url', 'No URL')}")
        
        if not meeting.get('long_form_url'):
            logger.warning("   ❌ No URL available for this meeting")
            continue
            
        # Fetch the page content
        meeting_url = meeting['long_form_url']
        if not meeting_url.startswith('http'):
            meeting_url = f"https://www.thegreyhoundrecorder.com.au{meeting_url}"
            
        logger.info(f"   🌐 Accessing: {meeting_url}")
        
        soup = scraper._get(meeting_url)
        if not soup:
            logger.warning("   ❌ Failed to fetch page content")
            continue
            
        # Look for the dog name on the page
        page_text = soup.get_text().upper()
        if test_dog.upper() in page_text:
            logger.info(f"   ✅ Found '{test_dog}' mentioned on this page!")
            
            # Find all instances of the dog name
            dog_mentions = soup.find_all(string=lambda text: text and test_dog.lower() in text.lower())
            logger.info(f"   📍 Found {len(dog_mentions)} mentions of '{test_dog}'")
            
            for j, mention in enumerate(dog_mentions[:5]):  # Show first 5
                logger.info(f"      {j+1}. '{mention.strip()}'")
                
                # Show the parent element structure
                parent = mention.parent if hasattr(mention, 'parent') else None
                if parent:
                    logger.info(f"         Parent: <{parent.name}> class={parent.get('class', [])}")
                    
                    # Show siblings
                    if hasattr(parent, 'find_next_siblings'):
                        siblings = list(parent.find_next_siblings())[:3]
                        logger.info(f"         Next {len(siblings)} siblings:")
                        for k, sibling in enumerate(siblings):
                            sibling_text = sibling.get_text(strip=True)
                            logger.info(f"           {k+1}. <{sibling.name}> {sibling.get('class', [])}: {sibling_text[:50]}...")
            
            # Look for table structures
            tables = soup.find_all('table')
            logger.info(f"   📊 Found {len(tables)} tables on the page")
            
            for j, table in enumerate(tables):
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header + data rows
                    header_row = rows[0]
                    headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
                    
                    logger.info(f"      Table {j+1}: {len(rows)} rows, headers: {headers}")
                    
                    # Check if any row contains our dog
                    for k, row in enumerate(rows[1:]):  # Skip header
                        row_text = row.get_text().upper()
                        if test_dog.upper() in row_text:
                            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                            logger.info(f"         Row {k+2}: {cells}")
            
            # We found the dog on this page, so let's stop here for detailed analysis
            break
        else:
            logger.info(f"   ❌ '{test_dog}' not found on this page")
    
    # Now test the existing fetch_enhanced_dog_data method to see what it returns
    logger.info(f"\n🧪 Testing existing fetch_enhanced_dog_data method for '{test_dog}'...")
    
    enhanced_data = scraper.fetch_enhanced_dog_data(test_dog)
    logger.info(f"📊 Result: {len(enhanced_data.get('form_entries', []))} race entries found")
    
    if enhanced_data.get('form_entries'):
        for entry in enhanced_data['form_entries']:
            logger.info(f"   📋 Entry: {entry}")
    else:
        logger.warning("   ❌ No form entries found - parsing is broken")

def main():
    """Main function."""
    debug_tgr_parsing()

if __name__ == "__main__":
    main()
