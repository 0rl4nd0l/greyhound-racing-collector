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

# TODO: Make these configurable
BASE_URL = "https://fasttrack.grv.org.au"


class FastTrackScraper:
    """
    Scraper for FastTrack greyhound racing data.

    Provides methods to fetch and parse individual dog and race pages.

    TODO:
    - Implement async batching for fetching multiple URLs concurrently.
    - Add robust retry logic with exponential backoff for handling transient network errors.
    - Integrate proxy rotation to avoid rate limiting or IP bans.
    - Use a more persistent caching mechanism (e.g., requests-cache with SQLite backend).
    """

    def __init__(self, rate_limit: float = 1.0, cache_dir: str = ".ft_cache", use_cache: bool = True):
        """
        Initialize the scraper.

        Args:
            rate_limit (float): Seconds to wait between requests to avoid overwhelming the server.
            cache_dir (str): Directory to store cached responses.
            use_cache (bool): Whether to use caching for requests.
        """
        self.rate_limit = rate_limit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        
        # Configure requests session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set a realistic user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.logger.info(
            f"FastTrackScraper initialized with rate limit: {rate_limit}s, cache: '{cache_dir}', caching: {use_cache}"
        )

    def fetch_dog(self, dog_id: int) -> dict:
        """
        Download and parse a single dog profile.

        Args:
            dog_id (int): The unique identifier for the dog.

        Returns:
            dict: A normalized dictionary containing the dog's profile information.
                  Returns an empty dict if the page cannot be fetched or parsed.
        """
        self.logger.info(f"Fetching dog profile for ID: {dog_id}")
        url = f"{BASE_URL}/Dog/Form/{dog_id}"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_dog(soup)

    def fetch_watchdog_form_guides(self) -> dict:
        """
        Download and parse the main Watchdog form guides page to get a list of all upcoming meetings.

        Returns:
            dict: A normalized dictionary containing a list of upcoming race meetings.
                  Returns an empty dict if the page cannot be fetched or parsed.
        """
        self.logger.info("Fetching Watchdog form guides...")
        url = f"{BASE_URL}/Watchdog/FormGuides"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_watchdog_form_guides(soup)

    def fetch_race_meeting(self, meeting_id: int) -> dict:
        """
        Download and parse a race meeting page.

        Args:
            meeting_id (int): The unique identifier for the race meeting.

        Returns:
            dict: A normalized dictionary containing the meeting's race information.
                  Returns an empty dict if the page cannot be fetched or parsed.
        """
        self.logger.info(f"Fetching race meeting for ID: {meeting_id}")
        url = f"{BASE_URL}/RaceField/ViewRaces/{meeting_id}"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_race_meeting(soup)
        
    def fetch_race(self, meeting_id: int, race_id: int) -> dict:
        """
        Download and parse a specific race result from a meeting.

        Args:
            meeting_id (int): The unique identifier for the race meeting.
            race_id (int): The unique identifier for the race.

        Returns:
            dict: A normalized dictionary containing the race's results and details.
                  Returns an empty dict if the page cannot be fetched or parsed.
        """
        self.logger.info(f"Fetching race result for meeting {meeting_id}, race {race_id}")
        url = f"{BASE_URL}/RaceField/ViewRaces/{meeting_id}?raceId={race_id}"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_race(soup)

    def _get(self, url: str):
        """
        Internal helper for making rate-limited GET requests with caching.

        Args:
            url (str): The URL to fetch.

        Returns:
            A BeautifulSoup object of the page content, or None on error.
        """
        # Generate cache filename based on URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.html"
        
        # Check cache first if enabled
        if self.use_cache and cache_file.exists():
            self.logger.debug(f"Loading cached content for: {url}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_content = f.read()
                return BeautifulSoup(cached_content, 'html.parser')
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {url}: {e}")
        
        # Enforce rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self.logger.debug(f"Requesting URL: {url}")
        self.last_request_time = time.time()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Cache the response if caching is enabled
            if self.use_cache:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    self.logger.debug(f"Cached response for: {url}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache response for {url}: {e}")
            
            return BeautifulSoup(response.text, 'html.parser')
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def _parse_dog(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Parses the HTML of a dog's profile page.

        Args:
            soup: A BeautifulSoup object of the dog's profile page.

        Returns:
            A dictionary of the dog's details.
        """
        self.logger.debug("Parsing dog profile soup.")
        dog_details = {}

        # Example: Extracting dog's name
        name_tag = soup.select_one(".dog-profile-header h1")
        if name_tag:
            dog_details['name'] = name_tag.text.strip()

        # Add more parsing logic for other details like career summary, recent form, etc.

        return dog_details

    def _parse_watchdog_form_guides(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Parses the HTML of the Watchdog form guides page to extract upcoming meetings.

        Args:
            soup: A BeautifulSoup object of the Watchdog form guides page.

        Returns:
            A dictionary containing a list of upcoming race meetings.
        """
        self.logger.debug("Parsing Watchdog form guides soup.")
        watchdog_data = {}
        
        meetings = []
        
        # Look for meeting links - these are typically formatted as venue/date combinations
        for link in soup.select("a[href*='/Meeting/Details/']"):
            href = link.get('href', '')
            meeting_id_match = re.search(r'/Meeting/Details/(\d+)', href)
            if meeting_id_match:
                meeting_id = int(meeting_id_match.group(1))
                meeting_text = link.text.strip()
                
                meetings.append({
                    'meeting_id': meeting_id,
                    'meeting_name': meeting_text,
                    'meeting_url': href
                })
        
        # Also look for direct race field links
        for link in soup.select("a[href*='/RaceField/ViewRaces/']"):
            href = link.get('href', '')
            meeting_id_match = re.search(r'/RaceField/ViewRaces/(\d+)', href)
            if meeting_id_match:
                meeting_id = int(meeting_id_match.group(1))
                meeting_text = link.text.strip()
                
                # Check if we already have this meeting ID
                existing = next((m for m in meetings if m['meeting_id'] == meeting_id), None)
                if not existing:
                    meetings.append({
                        'meeting_id': meeting_id,
                        'meeting_name': meeting_text,
                        'meeting_url': href
                    })
        
        watchdog_data['meetings'] = meetings
        return watchdog_data

    def _parse_race_meeting(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Parses the HTML of a race meeting page.

        Args:
            soup: A BeautifulSoup object of the race meeting page.

        Returns:
            A dictionary of the meeting's details, including a list of races.
        """
        self.logger.debug("Parsing race meeting soup.")
        meeting_data = {}
        
        races = []
        for link in soup.select("a[href*='/RaceField/ViewRaces/']"):
            href = link.get('href', '')
            match = re.search(r'raceId=(\d+)', href)
            if match:
                races.append({
                    'race_id': int(match.group(1)),
                    'race_name': link.text.strip()
                })
        
        meeting_data['races'] = races
        return meeting_data

    def _parse_race(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Parses the HTML of a race result page.

        Args:
            soup: A BeautifulSoup object of the race result page.

        Returns:
            A dictionary of the race's details.
        """
        self.logger.debug("Parsing race result soup.")
        race_data = {}

        # Extract race metadata
        header = soup.select_one("h1")
        if header:
            header_text = header.text.strip()
            parts = header_text.split(' ')
            race_data['venue'] = parts[0]
            race_data['race_date'] = parts[-1]
        
        race_details_table = soup.select_one("table.race-details")
        if race_details_table:
            for row in race_details_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) == 2:
                    key = cells[0].text.strip().lower().replace(' ','_')
                    value = cells[1].text.strip()
                    race_data[key] = value

        # Extract results from the main table
        results = []
        for row in soup.select("tr.ReportRaceDogLine"):
            try:
                dog_data = {}
                box_elem = row.select_one(".ReportRaceDogRugNumber")
                name_elem = row.select_one("a[href*='/Dog/Form?id=']")
                
                if box_elem and name_elem:
                    dog_data['box_number'] = box_elem.text.strip()
                    dog_data['dog_name'] = name_elem.text.strip()
                    results.append(dog_data)
            except Exception as e:
                self.logger.warning(f"Error parsing dog row: {e}")
                continue

        race_data['results'] = results

        return race_data

