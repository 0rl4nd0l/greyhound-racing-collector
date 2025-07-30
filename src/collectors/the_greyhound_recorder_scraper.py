#!/usr/bin/env python3
"""
Scraper for The Greyhound Recorder (thegreyhoundrecorder.com.au)
=============================================================

This scraper is designed to handle the CloudFront protection and parse the 
race calendar to extract upcoming meetings and races.

Author: AI Assistant
Date: July 30, 2025
"""

import logging
import time
from pathlib import Path
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import Dict, List, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import json

BASE_URL = "https://www.thegreyhoundrecorder.com.au"

class TheGreyhoundRecorderScraper:
    """
    Scraper for The Greyhound Recorder, handling CloudFront protection and parsing.
    """

    def __init__(self, rate_limit: float = 2.0, cache_dir: str = ".tgr_cache", use_cache: bool = True):
        self.rate_limit = rate_limit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        
        # Configure requests session with retry logic and a realistic user agent
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.logger.info(
            f"TheGreyhoundRecorderScraper initialized with rate limit: {rate_limit}s, cache: '{cache_dir}', caching: {use_cache}"
        )

    def _get(self, url: str) -> BeautifulSoup | None:
        """Internal helper for making rate-limited GET requests with caching."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.html"
        
        if self.use_cache and cache_file.exists():
            self.logger.debug(f"Loading cached content for: {url}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return BeautifulSoup(f.read(), 'html.parser')
        
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self.logger.debug(f"Requesting URL: {url}")
        self.last_request_time = time.time()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if "Request blocked" in response.text:
                self.logger.error(f"CloudFront blocked request to {url}")
                return None
            
            if self.use_cache:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            return BeautifulSoup(response.text, 'html.parser')
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def fetch_race_calendar(self) -> Dict[str, Any]:
        """Fetches the main race calendar page."""
        self.logger.info("Fetching race calendar...")
        url = f"{BASE_URL}/"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_race_calendar(soup)

    def _parse_race_calendar(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parses the race calendar to extract meeting information."""
        self.logger.debug("Parsing race calendar soup.")
        calendar_data = {"meetings": []}
        
        # Placeholder parsing logic based on a typical calendar structure
        # This will be refined after inspecting the actual HTML
        meeting_links = soup.select("a[href*='/form-guides/']")
        for link in meeting_links:
            href = link.get('href', '')
            meeting_name = link.text.strip()
            if href and meeting_name:
                calendar_data["meetings"].append({
                    "meeting_name": meeting_name,
                    "meeting_url": href
                })
        
        return calendar_data
        
    def fetch_form_guides(self) -> Dict[str, Any]:
        """Fetches the form guides page for race meetings."""
        self.logger.info("Fetching form guides...")
        url = f"{BASE_URL}/form-guides/"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_form_guides(soup)
        
    def _parse_form_guides(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parses the form guides page to extract detailed meeting and race information."""
        self.logger.debug("Parsing form guides soup.")
        form_guides_data = {"meetings": []}
        
        # Parse HTML structure directly to extract meeting information
        meeting_lists = soup.find_all('div', class_='meeting-list')
        
        for meeting_list in meeting_lists:
            # Extract the date from the title
            date_title = meeting_list.find('h2', class_='meeting-list__title')
            if not date_title:
                continue
                
            date_text = date_title.get_text(strip=True)
            
            # Extract individual meetings for this date
            meeting_rows = meeting_list.find_all('div', class_='meeting-row')
            
            for meeting_row in meeting_rows:
                meeting_title_elem = meeting_row.find('h3', class_='meeting-row__title')
                if not meeting_title_elem:
                    continue
                    
                meeting_title = meeting_title_elem.get_text(strip=True)
                
                # Extract the long form link (most detailed)
                long_form_link = meeting_row.find('a', href=re.compile(r'/long-form/'))
                if not long_form_link:
                    continue
                    
                long_form_url = long_form_link.get('href', '')
                
                # Also extract fields and short form links for completeness
                fields_link = meeting_row.find('a', href=re.compile(r'/fields/'))
                short_form_link = meeting_row.find('a', href=re.compile(r'/short-form/'))
                
                fields_url = fields_link.get('href', '') if fields_link else ''
                short_form_url = short_form_link.get('href', '') if short_form_link else ''
                
                # Parse venue and state from meeting title
                venue_info = self._parse_venue_info(meeting_title)
                
                meeting_data = {
                    'date': date_text,
                    'meeting_title': meeting_title,
                    'venue': venue_info['venue'],
                    'state': venue_info['state'],
                    'country': venue_info['country'],
                    'long_form_url': long_form_url,
                    'short_form_url': short_form_url,
                    'fields_url': fields_url,
                    'meeting_id': self._extract_meeting_id(long_form_url)
                }
                
                form_guides_data['meetings'].append(meeting_data)
                
        self.logger.info(f"Extracted {len(form_guides_data['meetings'])} meetings from form guides")
        return form_guides_data
        
    def _extract_meeting_id(self, url: str) -> int | None:
        """Extracts the meeting ID from a form guide URL."""
        match = re.search(r'/(\d+)/', url)
        return int(match.group(1)) if match else None

    def _parse_venue_info(self, meeting_title: str) -> Dict[str, str | None]:
        """Parses venue, state, and country from the meeting title."""
        venue_info = {
            'venue': None,
            'state': None,
            'country': None
        }
        
        # Use regex to extract details from titles like "Ballarat (VIC)"
        match = re.match(r'(.*?) \((.*?)\)', meeting_title)
        if match:
            venue_info['venue'] = match.group(1).strip()
            location = match.group(2).strip()
            
            if len(location) == 3 and location.isalpha():
                venue_info['state'] = location
                venue_info['country'] = self._get_country_from_state(location)
            else:
                venue_info['country'] = location
        else:
            venue_info['venue'] = meeting_title
        
        return venue_info

    def _get_country_from_state(self, state: str) -> str | None:
        """Determines the country from an Australian state code."""
        aus_states = ['VIC', 'NSW', 'QLD', 'WA', 'SA', 'TAS', 'NT']
        if state.upper() in aus_states:
            return 'AUS'
        return None
        
    def _extract_nuxt_data(self, soup: BeautifulSoup) -> Dict[str, Any] | None:
        """Extracts JSON data from Nuxt.js application."""
        try:
            # Find the script tag containing the JSON data
            script_tag = soup.find('script', {'id': '__NUXT_DATA__', 'type': 'application/json'})
            
            if script_tag and script_tag.string:
                # Parse the JSON data
                json_data = json.loads(script_tag.string)
                return json_data
            else:
                self.logger.warning("Could not find __NUXT_DATA__ script tag")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error extracting Nuxt data: {e}")
            return None
            
    def _parse_nuxt_data(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses the extracted Nuxt.js JSON data to extract race meetings."""
        form_guides_data = {"meetings": []}
        
        try:
            # The JSON structure is complex - we need to navigate through it
            # Based on the data structure, we'll look for menu items or meeting data
            
            # This is a simplified parser - in reality, we'd need to understand
            # the full data structure to extract race meetings properly
            if isinstance(json_data, list) and len(json_data) > 1:
                data_section = json_data[1] if len(json_data) > 1 else {}
                
                # Look for menu or navigation data that might contain form guide links
                for key, value in data_section.items():
                    if 'menu' in key.lower() or 'guide' in key.lower():
                        self.logger.debug(f"Found potential menu/guide data in key: {key}")
                        
            # For now, return a basic structure
            # TODO: Implement proper JSON data parsing based on actual structure
            form_guides_data["meetings"] = []
            
        except Exception as e:
            self.logger.error(f"Error parsing Nuxt.js data: {e}")
        
        return form_guides_data
    
    def fetch_long_form_race_data(self, long_form_url: str) -> Dict[str, Any]:
        """Fetches detailed race data from a long form URL."""
        self.logger.info(f"Fetching long form race data: {long_form_url}")
        
        # Construct full URL
        full_url = f"{BASE_URL}{long_form_url}" if long_form_url.startswith('/') else long_form_url
        
        soup = self._get(full_url)
        if not soup:
            return {}
            
        return self._parse_long_form_race_data(soup, long_form_url)
    
    def _parse_long_form_race_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Parses the long form race data page."""
        self.logger.debug(f"Parsing long form race data from: {url}")
        
        race_data = {
            'url': url,
            'meeting_info': {},
            'races': []
        }
        
        # This is a placeholder - we'll need to examine the actual long form structure
        # to implement proper parsing
        # For now, we'll return the basic structure
        
        self.logger.debug(f"Extracted race data structure for: {url}")
        return race_data
    
    def fetch_all_meetings_with_races(self) -> Dict[str, Any]:
        """Fetches all meetings from form guides and their detailed race data."""
        self.logger.info("Fetching all meetings with detailed race data...")
        
        # First get the list of meetings
        form_guides = self.fetch_form_guides()
        
        enhanced_meetings = []
        
        for meeting in form_guides.get('meetings', []):
            long_form_url = meeting.get('long_form_url')
            if not long_form_url:
                self.logger.warning(f"No long form URL for meeting: {meeting.get('meeting_title')}")
                continue
                
            # Fetch detailed race data for this meeting
            race_data = self.fetch_long_form_race_data(long_form_url)
            
            # Combine meeting info with race data
            enhanced_meeting = {
                **meeting,
                'race_data': race_data
            }
            
            enhanced_meetings.append(enhanced_meeting)
            
            # Rate limiting between requests
            time.sleep(self.rate_limit)
        
        self.logger.info(f"Successfully fetched detailed data for {len(enhanced_meetings)} meetings")
        
        return {
            'meetings': enhanced_meetings,
            'total_meetings': len(enhanced_meetings)
        }
