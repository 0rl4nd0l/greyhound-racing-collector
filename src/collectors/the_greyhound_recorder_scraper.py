#!/usr/bin/env python3
"""
Scraper for The Greyhound Recorder (thegreyhoundrecorder.com.au)
=============================================================

This scraper is designed to handle the CloudFront protection and parse the
race calendar to extract upcoming meetings and races.

Author: AI Assistant
Date: July 30, 2025
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.thegreyhoundrecorder.com.au"


class TheGreyhoundRecorderScraper:
    """
    Scraper for The Greyhound Recorder, handling CloudFront protection and parsing.
    """

    def __init__(
        self,
        rate_limit: float = 2.0,
        cache_dir: str = ".tgr_cache",
        use_cache: bool = True,
    ):
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
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        self.logger.info(
            f"TheGreyhoundRecorderScraper initialized with rate limit: {rate_limit}s, cache: '{cache_dir}', caching: {use_cache}"
        )

    def _get(self, url: str) -> BeautifulSoup | None:
        """Internal helper for making rate-limited GET requests with caching."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.html"

        if self.use_cache and cache_file.exists():
            self.logger.debug(f"Loading cached content for: {url}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return BeautifulSoup(f.read(), "html.parser")

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
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(response.text)

            return BeautifulSoup(response.text, "html.parser")

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
            href = link.get("href", "")
            meeting_name = link.text.strip()
            if href and meeting_name:
                calendar_data["meetings"].append(
                    {"meeting_name": meeting_name, "meeting_url": href}
                )

        return calendar_data

    def fetch_form_guides(self) -> Dict[str, Any]:
        """Fetches the form guides page for race meetings."""
        self.logger.info("Fetching form guides...")
        url = f"{BASE_URL}/form-guides/"
        soup = self._get(url)
        if not soup:
            return {}
        return self._parse_form_guides(soup)
    
    def fetch_enhanced_dog_data(self, dog_name: str, limit_days: int = 365) -> Dict[str, Any]:
        """Fetch enhanced data for a specific dog from TGR."""
        self.logger.info(f"Fetching enhanced TGR data for dog: {dog_name}")
        
        enhanced_data = {
            'dog_name': dog_name,
            'form_entries': [],
            'performance_summary': {},
            'venue_analysis': {},
            'distance_analysis': {},
            'recent_comments': [],
            'expert_insights': []
        }
        
        try:
            # Search for dog in form guides
            form_guides = self.fetch_form_guides()
            
            # Process each meeting to find races with this dog
            for meeting in form_guides.get('meetings', []):
                if meeting.get('long_form_url'):
                    race_data = self._fetch_race_details(meeting['long_form_url'])
                    dog_entries = self._extract_dog_entries(race_data, dog_name)
                    enhanced_data['form_entries'].extend(dog_entries)
            
            # Calculate performance metrics
            enhanced_data['performance_summary'] = self._calculate_performance_metrics(
                enhanced_data['form_entries']
            )
            
            # Analyze venue and distance performance
            enhanced_data['venue_analysis'] = self._analyze_venue_performance(
                enhanced_data['form_entries']
            )
            enhanced_data['distance_analysis'] = self._analyze_distance_performance(
                enhanced_data['form_entries']
            )
            
            # Extract recent comments and insights
            enhanced_data['recent_comments'] = self._extract_recent_comments(
                enhanced_data['form_entries']
            )
            
            self.logger.info(f"Enhanced data collected: {len(enhanced_data['form_entries'])} races")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch enhanced dog data: {e}")
        
        return enhanced_data
    
    def _fetch_race_details(self, race_url: str) -> Dict[str, Any]:
        """Extract race details and individual dog racing histories from TGR."""
        
        if not race_url.startswith('http'):
            race_url = f"{BASE_URL}{race_url}"
        
        soup = self._get(race_url)
        if not soup:
            return {}
        
        race_details = {
            'url': race_url,
            'date': None,
            'venue': None,
            'race_number': None,
            'grade': None,
            'distance': None,
            'field_size': 0,
            'dogs': [],
            'race_result': {},
            'expert_comments': []
        }
        
        try:
            # Extract race title/heading from the meeting heading
            heading = soup.find(class_='form-guide-meeting__heading')
            if heading:
                heading_text = heading.get_text(strip=True)
                self.logger.debug(f"Found heading: {heading_text}")
                
                # Parse venue, date, race number from heading
                # Format: "Murray Bridge Form Guide (Race 1) - 3rd Aug 2025"
                if 'Form Guide' in heading_text:
                    parts = heading_text.split('Form Guide')
                    if len(parts) >= 2:
                        race_details['venue'] = parts[0].strip()
                        
                        race_part = parts[1]
                        if 'Race' in race_part and ')' in race_part:
                            # Extract race number from (Race X)
                            race_match = re.search(r'Race (\d+)', race_part)
                            if race_match:
                                race_details['race_number'] = int(race_match.group(1))
                            
                            # Extract date from the end
                            if '-' in race_part:
                                date_part = race_part.split('-')[-1].strip()
                                race_details['date'] = date_part
            
            # Extract individual dog racing histories from form-guide-long-form-selection sections
            dog_sections = soup.find_all(class_='form-guide-long-form-selection')
            
            for section in dog_sections:
                # Skip vacant boxes
                if 'form-guide-long-form-selection--vacant' in section.get('class', []):
                    continue
                
                # Extract dog name from the header
                header = section.find(class_='form-guide-long-form-selection__header')
                if not header:
                    continue
                
                dog_name_elem = header.find(class_='form-guide-long-form-selection__header-name')
                if not dog_name_elem:
                    continue
                
                dog_name = dog_name_elem.get_text(strip=True)
                self.logger.debug(f"Processing dog: {dog_name}")
                
                # Extract the racing history table for this dog
                history_table = section.find('table', class_='form-guide-selection-results')
                if history_table:
                    dog_history = self._extract_dog_racing_history(history_table, dog_name)
                    if dog_history:
                        # Add the dog with their racing history
                        dog_entry = {
                            'dog_name': dog_name,
                            'racing_history': dog_history,
                            'total_races': len(dog_history)
                        }
                        race_details['dogs'].append(dog_entry)
                        race_details['field_size'] += 1
                        
                        self.logger.debug(f"Extracted {len(dog_history)} races for {dog_name}")
            
            self.logger.debug(f"Extracted {race_details['field_size']} dogs with racing histories")
            
        except Exception as e:
            self.logger.error(f"Error parsing race details from {race_url}: {e}")
        
        return race_details
    
    def _extract_dog_racing_history(self, table, dog_name: str) -> List[Dict[str, Any]]:
        """Extract individual race history from a dog's history table."""
        
        racing_history = []
        
        try:
            # Get table headers to understand column structure
            headers = []
            header_row = table.find('thead')
            if header_row:
                header_cells = header_row.find_all('th')
                headers = [cell.get_text(strip=True) for cell in header_cells]
            
            self.logger.debug(f"Table headers for {dog_name}: {headers}")
            
            # Process each race row in the table
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 5:  # Need minimum data
                        continue
                    
                    race_entry = {
                        'dog_name': dog_name,
                        'race_date': None,
                        'finish_position': None,
                        'box_number': None,
                        'margin': None,
                        'track': None,
                        'distance': None,
                        'grade': None,
                        'individual_time': None,
                        'winning_time': None,
                        'best_time_of_night': None,
                        'sectional_time': None,
                        'in_run': None,
                        'weight': None,
                        'starting_price': None,
                        'winner_second': None
                    }
                    
                    # Map cells to fields based on typical TGR structure:
                    # Date, Fin, Box, Mgn, Trk, Dis, Grd, Time, Win T, Bon, Sect, In Run, Weight, SP, Winner/Second
                    
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        
                        if i == 0:  # Date
                            race_entry['race_date'] = cell_text
                        elif i == 1:  # Fin (Finishing position)
                            # Extract numeric position from text like "1st", "2nd", etc.
                            pos_match = re.search(r'(\d+)', cell_text)
                            if pos_match:
                                race_entry['finish_position'] = int(pos_match.group(1))
                        elif i == 2:  # Box
                            # Remove parentheses from "(1)" format
                            box_text = cell_text.replace('(', '').replace(')', '')
                            if box_text.isdigit():
                                race_entry['box_number'] = int(box_text)
                        elif i == 3:  # Margin
                            try:
                                race_entry['margin'] = float(cell_text)
                            except:
                                race_entry['margin'] = cell_text
                        elif i == 4:  # Track
                            race_entry['track'] = cell_text
                        elif i == 5:  # Distance
                            race_entry['distance'] = cell_text
                        elif i == 6:  # Grade
                            race_entry['grade'] = cell_text
                        elif i == 7:  # Time (individual time)
                            try:
                                race_entry['individual_time'] = float(cell_text)
                            except:
                                pass
                        elif i == 8:  # Win T (winning time)
                            try:
                                race_entry['winning_time'] = float(cell_text)
                            except:
                                pass
                        elif i == 9:  # BON (Best of night)
                            try:
                                race_entry['best_time_of_night'] = float(cell_text)
                            except:
                                pass
                        elif i == 10:  # Sect (sectional)
                            try:
                                race_entry['sectional_time'] = float(cell_text)
                            except:
                                pass
                        elif i == 11:  # In Run
                            race_entry['in_run'] = cell_text
                        elif i == 15:  # Weight (usually around position 15)
                            try:
                                race_entry['weight'] = float(cell_text)
                            except:
                                pass
                        elif i == 16:  # SP (Starting Price)
                            race_entry['starting_price'] = cell_text
                        elif i == 17:  # Winner/Second
                            race_entry['winner_second'] = cell_text
                    
                    # Only add race if we have meaningful data
                    if race_entry['race_date'] and race_entry['finish_position']:
                        racing_history.append(race_entry)
            
        except Exception as e:
            self.logger.debug(f"Error extracting racing history for {dog_name}: {e}")
        
        return racing_history
    
    def _parse_dog_entry_row_fixed(self, row) -> Dict[str, Any]:
        """Fixed dog entry parsing that works with actual TGR table structure."""
        
        try:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:  # Need at least dog name + some data
                return None
            
            dog_entry = {
                'box_number': None,
                'dog_name': None,
                'trainer': None,
                'weight': None,
                'recent_form': [],
                'last_start': {},
                'comments': None,
                'odds': None,
                'rating': None,
                'early_speed': None
            }
            
            # Based on the table structure we found:
            # ['Rug/Greyhound (box)', 'Form', 'Comment', 'Early Speed', 'Rtg']
            
            for i, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                
                if i == 0:  # Dog name (and possibly box)
                    # Could be "Dog Name" or "Dog Name (box)"
                    if '(' in cell_text and ')' in cell_text:
                        # Extract dog name and box number
                        parts = cell_text.split('(')
                        dog_entry['dog_name'] = parts[0].strip()
                        
                        box_part = parts[1].replace(')', '').strip()
                        try:
                            dog_entry['box_number'] = int(box_part)
                        except:
                            pass
                    else:
                        dog_entry['dog_name'] = cell_text
                
                elif i == 1:  # Form
                    # Form could be like "3105" or "1234"
                    if cell_text.isdigit() and len(cell_text) >= 2:
                        # Convert form string to list of positions
                        dog_entry['recent_form'] = list(cell_text)
                
                elif i == 2:  # Comment
                    if cell_text and len(cell_text) > 3:
                        dog_entry['comments'] = cell_text
                
                elif i == 3:  # Early Speed
                    try:
                        dog_entry['early_speed'] = float(cell_text)
                    except:
                        pass
                
                elif i == 4:  # Rating
                    try:
                        dog_entry['rating'] = int(cell_text)
                    except:
                        pass
            
            # Look for additional data in nested elements
            links = row.find_all('a')
            for link in links:
                href = link.get('href', '')
                if '/greyhound/' in href:
                    dog_entry['profile_url'] = href
            
            return dog_entry if dog_entry['dog_name'] else None
            
        except Exception as e:
            self.logger.debug(f"Error parsing dog entry row: {e}")
            return None
    
    def _parse_dog_entry_row(self, row) -> Dict[str, Any]:
        """Backwards compatibility wrapper for old method name."""
        return self._parse_dog_entry_row_fixed(row)
    
    def _extract_dog_entries(self, race_data: Dict[str, Any], target_dog_name: str) -> List[Dict[str, Any]]:
        """Extract entries for a specific dog from race data."""
        entries = []
        target_name_clean = target_dog_name.upper().strip()
        
        for dog in race_data.get('dogs', []):
            if dog.get('dog_name', '').upper().strip() == target_name_clean:
                # Enhance with race metadata
                enhanced_entry = {
                    **dog,
                    'race_date': race_data.get('date'),
                    'venue': race_data.get('venue'),
                    'race_number': race_data.get('race_number'),
                    'grade': race_data.get('grade'),
                    'distance': race_data.get('distance'),
                    'field_size': race_data.get('field_size'),
                    'race_url': race_data.get('url'),
                    'expert_comments': race_data.get('expert_comments', [])
                }
                entries.append(enhanced_entry)
        
        return entries
    
    def _calculate_performance_metrics(self, form_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not form_entries:
            return {}
        
        metrics = {
            'total_starts': len(form_entries),
            'wins': 0,
            'places': 0,
            'win_percentage': 0.0,
            'place_percentage': 0.0,
            'average_position': 0.0,
            'best_position': 8,
            'consistency_score': 0.0,
            'recent_form_trend': 'stable',
            'grade_progression': [],
            'distance_versatility': 0
        }
        
        try:
            positions = []
            recent_positions = []
            grades = set()
            distances = set()
            
            for i, entry in enumerate(form_entries):
                # Extract position from recent form or last start
                position = None
                if entry.get('recent_form'):
                    # Get most recent position
                    for form_pos in entry['recent_form']:
                        if form_pos.isdigit():
                            position = int(form_pos)
                            break
                
                if position:
                    positions.append(position)
                    if i < 5:  # Recent form (last 5 races)
                        recent_positions.append(position)
                    
                    if position == 1:
                        metrics['wins'] += 1
                    if position <= 3:
                        metrics['places'] += 1
                
                # Track grades and distances for versatility
                if entry.get('grade'):
                    grades.add(entry['grade'])
                if entry.get('distance'):
                    distances.add(entry['distance'])
            
            if positions:
                metrics['average_position'] = sum(positions) / len(positions)
                metrics['best_position'] = min(positions)
                metrics['win_percentage'] = (metrics['wins'] / len(positions)) * 100
                metrics['place_percentage'] = (metrics['places'] / len(positions)) * 100
                
                # Calculate consistency (lower variance = higher consistency)
                if len(positions) > 1:
                    variance = sum((p - metrics['average_position']) ** 2 for p in positions) / len(positions)
                    metrics['consistency_score'] = max(0, 100 - variance)
            
            # Analyze form trend
            if len(recent_positions) >= 3:
                early_avg = sum(recent_positions[-3:]) / 3
                recent_avg = sum(recent_positions[:3]) / 3
                
                if recent_avg < early_avg - 0.5:
                    metrics['recent_form_trend'] = 'improving'
                elif recent_avg > early_avg + 0.5:
                    metrics['recent_form_trend'] = 'declining'
            
            metrics['grade_progression'] = list(grades)
            metrics['distance_versatility'] = len(distances)
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _analyze_venue_performance(self, form_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by venue."""
        venue_stats = {}
        
        for entry in form_entries:
            venue = entry.get('venue')
            if not venue:
                continue
            
            if venue not in venue_stats:
                venue_stats[venue] = {
                    'starts': 0,
                    'wins': 0,
                    'places': 0,
                    'positions': [],
                    'win_rate': 0.0,
                    'place_rate': 0.0,
                    'average_position': 0.0
                }
            
            venue_stats[venue]['starts'] += 1
            
            # Extract position data
            if entry.get('recent_form'):
                for form_pos in entry['recent_form']:
                    if form_pos.isdigit():
                        position = int(form_pos)
                        venue_stats[venue]['positions'].append(position)
                        if position == 1:
                            venue_stats[venue]['wins'] += 1
                        if position <= 3:
                            venue_stats[venue]['places'] += 1
                        break
        
        # Calculate rates and averages
        for venue, stats in venue_stats.items():
            if stats['positions']:
                stats['average_position'] = sum(stats['positions']) / len(stats['positions'])
                stats['win_rate'] = (stats['wins'] / len(stats['positions'])) * 100
                stats['place_rate'] = (stats['places'] / len(stats['positions'])) * 100
        
        return venue_stats
    
    def _analyze_distance_performance(self, form_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by distance."""
        distance_stats = {}
        
        for entry in form_entries:
            distance = entry.get('distance')
            if not distance:
                continue
            
            # Normalize distance (extract numeric value)
            distance_key = distance
            try:
                distance_match = re.search(r'(\d+)', distance)
                if distance_match:
                    distance_key = f"{distance_match.group(1)}m"
            except:
                pass
            
            if distance_key not in distance_stats:
                distance_stats[distance_key] = {
                    'starts': 0,
                    'wins': 0,
                    'places': 0,
                    'positions': [],
                    'win_rate': 0.0,
                    'place_rate': 0.0,
                    'average_position': 0.0
                }
            
            distance_stats[distance_key]['starts'] += 1
            
            # Extract position data
            if entry.get('recent_form'):
                for form_pos in entry['recent_form']:
                    if form_pos.isdigit():
                        position = int(form_pos)
                        distance_stats[distance_key]['positions'].append(position)
                        if position == 1:
                            distance_stats[distance_key]['wins'] += 1
                        if position <= 3:
                            distance_stats[distance_key]['places'] += 1
                        break
        
        # Calculate rates and averages
        for distance, stats in distance_stats.items():
            if stats['positions']:
                stats['average_position'] = sum(stats['positions']) / len(stats['positions'])
                stats['win_rate'] = (stats['wins'] / len(stats['positions'])) * 100
                stats['place_rate'] = (stats['places'] / len(stats['positions'])) * 100
        
        return distance_stats
    
    def _extract_recent_comments(self, form_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract recent comments and expert insights."""
        comments = []
        
        for entry in form_entries[:10]:  # Last 10 races
            # Individual dog comments
            if entry.get('comments'):
                comments.append({
                    'type': 'dog_comment',
                    'race_date': entry.get('race_date'),
                    'venue': entry.get('venue'),
                    'text': entry['comments'],
                    'source': 'form_guide'
                })
            
            # Expert race comments
            for expert_comment in entry.get('expert_comments', []):
                if expert_comment and len(expert_comment) > 50:
                    comments.append({
                        'type': 'expert_insight',
                        'race_date': entry.get('race_date'),
                        'venue': entry.get('venue'),
                        'text': expert_comment,
                        'source': 'expert_analysis'
                    })
        
        # Sort by recency
        comments.sort(key=lambda x: x.get('race_date', ''), reverse=True)
        
        return comments[:20]  # Return most recent 20 comments
    
    def store_enhanced_data_to_db(self, enhanced_data: Dict[str, Any], db_path: str = "greyhound_racing_data.db"):
        """Store enhanced TGR data to database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            dog_name = enhanced_data['dog_name']
            
            # Store form entries
            for entry in enhanced_data['form_entries']:
                cursor.execute("""
                    INSERT OR IGNORE INTO gr_dog_form 
                    (dog_name, race_date, venue, grade, distance, box_number, 
                     recent_form, weight, comments, odds, race_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    dog_name,
                    entry.get('race_date'),
                    entry.get('venue'),
                    entry.get('grade'),
                    entry.get('distance'),
                    entry.get('box_number'),
                    json.dumps(entry.get('recent_form', [])),
                    entry.get('weight'),
                    entry.get('comments'),
                    entry.get('odds'),
                    entry.get('race_url')
                ])
            
            # Store performance summary
            cursor.execute("""
                INSERT OR REPLACE INTO tgr_dog_performance_summary
                (dog_name, performance_data, venue_analysis, distance_analysis, 
                 last_updated, total_entries)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                dog_name,
                json.dumps(enhanced_data['performance_summary']),
                json.dumps(enhanced_data['venue_analysis']),
                json.dumps(enhanced_data['distance_analysis']),
                datetime.now().isoformat(),
                len(enhanced_data['form_entries'])
            ])
            
            # Store recent comments
            for comment in enhanced_data['recent_comments']:
                cursor.execute("""
                    INSERT OR IGNORE INTO tgr_expert_insights
                    (dog_name, comment_type, race_date, venue, comment_text, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    dog_name,
                    comment['type'],
                    comment.get('race_date'),
                    comment.get('venue'),
                    comment['text'],
                    comment['source']
                ])
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored enhanced TGR data for {dog_name}: "
                           f"{len(enhanced_data['form_entries'])} entries, "
                           f"{len(enhanced_data['recent_comments'])} comments")
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced TGR data: {e}")

    def _parse_form_guides(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parses the form guides page to extract detailed meeting and race information."""
        self.logger.debug("Parsing form guides soup.")
        form_guides_data = {"meetings": []}

        # Parse HTML structure directly to extract meeting information
        meeting_lists = soup.find_all("div", class_="meeting-list")

        for meeting_list in meeting_lists:
            # Extract the date from the title
            date_title = meeting_list.find("h2", class_="meeting-list__title")
            if not date_title:
                continue

            date_text = date_title.get_text(strip=True)

            # Extract individual meetings for this date
            meeting_rows = meeting_list.find_all("div", class_="meeting-row")

            for meeting_row in meeting_rows:
                meeting_title_elem = meeting_row.find("h3", class_="meeting-row__title")
                if not meeting_title_elem:
                    continue

                meeting_title = meeting_title_elem.get_text(strip=True)

                # Extract the long form link (most detailed)
                long_form_link = meeting_row.find("a", href=re.compile(r"/long-form/"))
                if not long_form_link:
                    continue

                long_form_url = long_form_link.get("href", "")

                # Also extract fields and short form links for completeness
                fields_link = meeting_row.find("a", href=re.compile(r"/fields/"))
                short_form_link = meeting_row.find(
                    "a", href=re.compile(r"/short-form/")
                )

                fields_url = fields_link.get("href", "") if fields_link else ""
                short_form_url = (
                    short_form_link.get("href", "") if short_form_link else ""
                )

                # Parse venue and state from meeting title
                venue_info = self._parse_venue_info(meeting_title)

                meeting_data = {
                    "date": date_text,
                    "meeting_title": meeting_title,
                    "venue": venue_info["venue"],
                    "state": venue_info["state"],
                    "country": venue_info["country"],
                    "long_form_url": long_form_url,
                    "short_form_url": short_form_url,
                    "fields_url": fields_url,
                    "meeting_id": self._extract_meeting_id(long_form_url),
                }

                form_guides_data["meetings"].append(meeting_data)

        self.logger.info(
            f"Extracted {len(form_guides_data['meetings'])} meetings from form guides"
        )
        return form_guides_data

    def _extract_meeting_id(self, url: str) -> int | None:
        """Extracts the meeting ID from a form guide URL."""
        match = re.search(r"/(\d+)/", url)
        return int(match.group(1)) if match else None

    def _parse_venue_info(self, meeting_title: str) -> Dict[str, str | None]:
        """Parses venue, state, and country from the meeting title."""
        venue_info = {"venue": None, "state": None, "country": None}

        # Use regex to extract details from titles like "Ballarat (VIC)"
        match = re.match(r"(.*?) \((.*?)\)", meeting_title)
        if match:
            venue_info["venue"] = match.group(1).strip()
            location = match.group(2).strip()

            if len(location) == 3 and location.isalpha():
                venue_info["state"] = location
                venue_info["country"] = self._get_country_from_state(location)
            else:
                venue_info["country"] = location
        else:
            venue_info["venue"] = meeting_title

        return venue_info

    def _get_country_from_state(self, state: str) -> str | None:
        """Determines the country from an Australian state code."""
        aus_states = ["VIC", "NSW", "QLD", "WA", "SA", "TAS", "NT"]
        if state.upper() in aus_states:
            return "AUS"
        return None

    def _extract_nuxt_data(self, soup: BeautifulSoup) -> Dict[str, Any] | None:
        """Extracts JSON data from Nuxt.js application."""
        try:
            # Find the script tag containing the JSON data
            script_tag = soup.find(
                "script", {"id": "__NUXT_DATA__", "type": "application/json"}
            )

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
                    if "menu" in key.lower() or "guide" in key.lower():
                        self.logger.debug(
                            f"Found potential menu/guide data in key: {key}"
                        )

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
        full_url = (
            f"{BASE_URL}{long_form_url}"
            if long_form_url.startswith("/")
            else long_form_url
        )

        soup = self._get(full_url)
        if not soup:
            return {}

        return self._parse_long_form_race_data(soup, long_form_url)

    def _parse_long_form_race_data(
        self, soup: BeautifulSoup, url: str
    ) -> Dict[str, Any]:
        """Parses the long form race data page."""
        self.logger.debug(f"Parsing long form race data from: {url}")

        race_data = {"url": url, "meeting_info": {}, "races": []}

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

        for meeting in form_guides.get("meetings", []):
            long_form_url = meeting.get("long_form_url")
            if not long_form_url:
                self.logger.warning(
                    f"No long form URL for meeting: {meeting.get('meeting_title')}"
                )
                continue

            # Fetch detailed race data for this meeting
            race_data = self.fetch_long_form_race_data(long_form_url)

            # Combine meeting info with race data
            enhanced_meeting = {**meeting, "race_data": race_data}

            enhanced_meetings.append(enhanced_meeting)

            # Rate limiting between requests
            time.sleep(self.rate_limit)

        self.logger.info(
            f"Successfully fetched detailed data for {len(enhanced_meetings)} meetings"
        )

        return {"meetings": enhanced_meetings, "total_meetings": len(enhanced_meetings)}
    
    def get_race_insights(self, venue: str, race_date: str, race_number: int) -> Dict[str, Any]:
        """Get race-level insights for a specific race (placeholder implementation)."""
        self.logger.info(f"Getting race insights for {venue} Race {race_number} on {race_date}")
        
        # This is a placeholder implementation as the method didn't exist
        # In a real implementation, this would scrape race-level insights from TGR
        
        # For now, return a basic structure indicating this is a placeholder
        return {
            'venue': venue,
            'race_date': race_date,
            'race_number': race_number,
            'insights': {
                'track_conditions': 'Unknown',
                'weather_impact': 'Unknown',
                'expert_tips': [],
                'race_preview': None
            },
            'data_source': 'tgr_placeholder',
            'timestamp': datetime.now().isoformat(),
            'status': 'placeholder_implementation'
        }
