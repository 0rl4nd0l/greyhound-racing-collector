#!/usr/bin/env python3
"""
Enhanced Greyhound Results Scraper with Website Navigation
Navigates through thedogs.com.au to find race results instead of guessing URLs
"""

import os
import re
import pandas as pd
import asyncio
import json
from datetime import datetime, timedelta
import sys

# Add the assistant_bridge_repo to path to use the enhanced scraper
sys.path.append('/Users/orlandolee/assistant_bridge_repo')
from web_scraper import EnhancedWebScraper

class GreyhoundResultsNavigator:
    def __init__(self):
        self.scraper = EnhancedWebScraper(delay=3, use_playwright=True)
        
        # Map location codes to possible track names
        self.location_map = {
            'SAN': ['sandown', 'sandown-park', 'sandown park'],
            'MEA': ['meadows', 'the-meadows', 'the meadows'], 
            'WPK': ['wentworth-park', 'wentworth', 'wentworth park'],
            'W_PK': ['wentworth-park', 'wentworth', 'wentworth park'],
            'AP_K': ['angle-park', 'angle park', 'albion-park'],  # Check both angle and albion
            'APTH': ['albion-park', 'albion park'],
            'APWE': ['albion-park', 'albion park'], 
            'BAL': ['ballarat'],
            'BEN': ['bendigo'],
            'CANN': ['cannington'],
            'CASO': ['casino'],
            'DAPT': ['dapto'],
            'GEE': ['geelong'],
            'GAWL': ['gawler'],
            'HOR': ['horsham'],
            'RICH': ['richmond', 'richmond-straight'],  # Add richmond-straight
            'SAL': ['sale'],
            'TRA': ['traralgon'],
            'WAR': ['warrnambool'],
            'NOR': ['northam'],
            'TEMA': ['temora'],
            # Add missing venue codes from failed races
            'HOBT': ['hobart'],
            'QOT': ['ladbrokes-q-straight', 'ladbrokes-q1-lakeside', 'the-gardens'],  # Queensland tracks
            'GUNN': ['gunnedah'],
            'CAPA': ['capalaba'],
            'ROCK': ['rockhampton'],
            'DARW': ['darwin'],
            'MURR': ['murray-bridge', 'murray-bridge-straight'],
            'MOUNT': ['mount-gambier'],
            'MAND': ['mandurah'],
            'SHEP': ['shepparton'],
            'WARR': ['warragul'],
            'GEELONG': ['geelong'],
            'TRARALGON': ['traralgon']
        }
        
        self.base_url = "https://www.thedogs.com.au"
        
    def extract_race_details(self, filename):
        """Extract race number, location, and date from filename"""
        # Handle filenames with (1), (2) etc. suffixes
        clean_filename = re.sub(r'\s*\(\d+\)\.csv$', '.csv', filename)
        
        pattern = r"Race (\d+) - (\w+) - (\d{2} \w+ \d{4})\.csv"
        match = re.match(pattern, clean_filename)
        if not match:
            raise ValueError(f"Filename {filename} does not match expected format")
        
        race_number = int(match.group(1))
        location = match.group(2)
        date_str = match.group(3)
        date = datetime.strptime(date_str, "%d %B %Y")
        return race_number, location, date

    async def navigate_to_date_page(self, target_date):
        """Navigate to the specific date page using view-source to get raw HTML"""
        date_str = target_date.strftime('%Y-%m-%d')
        # Use view-source: prefix to get raw HTML
        url = f"view-source:{self.base_url}/racing/{date_str}"
        print(f"🌐 Accessing view-source for date page: {url}")
        
        try:
            result = await self.scraper.scrape_url(url, extract_images=False, force_playwright=True)
            
            if 'error' not in result and result.get('page_text'):
                # The view-source content will be the raw HTML
                html_content = result['page_text']
                print(f"   📝 Raw HTML content length: {len(html_content)} chars")
                
                # Check if this looks like HTML source with racing content
                if any(keyword in html_content.lower() for keyword in ['racing', 'href=', '<html', 'meeting', 'race']):
                    print(f"   ✅ Found raw HTML source for {date_str}")
                    # Store the raw HTML in the result for parsing
                    result['raw_html'] = html_content
                    return url, result
                else:
                    print(f"   ⚠️  Content doesn't look like HTML source")
                    
        except Exception as e:
            print(f"   ❌ Error with {url}: {e}")
            
            # Fallback: try without view-source prefix
            print(f"   🔄 Fallback: trying regular page access")
            fallback_url = f"{self.base_url}/racing/{date_str}"
            try:
                result = await self.scraper.scrape_url(fallback_url, extract_images=False, force_playwright=True)
                if 'error' not in result and result.get('page_text'):
                    print(f"   ✅ Fallback successful")
                    return fallback_url, result
            except Exception as fallback_e:
                print(f"   ❌ Fallback also failed: {fallback_e}")
            
            return None, None
        
        print(f"❌ Could not find racing page for {date_str}")
        return None, None

    async def find_track_and_race(self, page_content, location, target_date, race_number):
        """Find links to specific track and race by parsing HTML and matching exact patterns"""
        print(f"🔍 Looking for {location} Race {race_number} on {target_date.strftime('%Y-%m-%d')}")
        
        # Get possible track names for this location
        track_names = self.location_map.get(location, [location.lower()])
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Debug: Show what we're looking for
        print(f"   🔍 Looking for venue code '{location}' mapped to: {track_names}")
        
        # Extract all href links from the HTML (accounting for view-source spacing)
        # Pattern matches: href = " /path/to/link "
        href_pattern = r'href\s*=\s*["\']\s*([^"\'>]+?)\s*["\']'
        all_links = re.findall(href_pattern, page_content, re.IGNORECASE)
        
        print(f"   Found {len(all_links)} total links in HTML")
        
        # First, let's see what racing venues are available on this date
        racing_venues = set()
        for link in all_links:
            if '/racing/' in link.lower() and date_str in link:
                # Extract venue name from pattern /racing/{venue}/{date}/...
                parts = link.split('/')
                if len(parts) >= 4 and parts[1] == 'racing':
                    venue = parts[2]
                    racing_venues.add(venue)
        
        print(f"   Available racing venues on {date_str}: {sorted(racing_venues)}")
        
        # Look for racing links that match our criteria
        # Pattern: /racing/{track_name}/{date}/{race_number}/...
        matching_links = []
        
        for link in all_links:
            link_lower = link.lower()
            
            # Check if it's a racing link with our date
            if '/racing/' in link_lower and date_str in link:
                # Check if it contains any of our track names
                for track_name in track_names:
                    track_pattern = f'/racing/{track_name}/{date_str}/{race_number}/'
                    if track_pattern in link_lower:
                        matching_links.append(link)
                        print(f"   ✅ Found exact match: {link}")
                        break
                
                # Also check with original location code in case the mapping isn't perfect
                location_pattern = f'/racing/{location.lower()}/{date_str}/{race_number}/'
                if location_pattern in link_lower:
                    matching_links.append(link)
                    print(f"   ✅ Found location match: {link}")
                
                # Additional fallback: check for partial track name matches
                for track_name in track_names:
                    if f'/{track_name}/' in link_lower and f'/{race_number}/' in link_lower:
                        matching_links.append(link)
                        print(f"   ✅ Found partial match: {link}")
                        break
        
        # If we found exact matches, prioritize those
        if matching_links:
            print(f"   🏆 Found {len(matching_links)} exact race matches")
            # Convert relative links to absolute
            final_links = []
            for link in matching_links:
                if link.startswith('/'):
                    final_links.append(f"{self.base_url}{link}")
                else:
                    final_links.append(link)
            return final_links
        
        # If no exact matches, look for broader patterns
        print(f"   🔍 No exact matches, looking for broader patterns...")
        
        broader_matches = []
        for link in all_links:
            link_lower = link.lower()
            
            # Look for any racing link with our date and track
            if ('/racing/' in link_lower and date_str in link and 
                any(track in link_lower for track in track_names + [location.lower()])):
                broader_matches.append(link)
        
        print(f"   Found {len(broader_matches)} broader matches")
        
        # Convert relative links to absolute
        final_links = []
        for link in broader_matches:
            if link.startswith('/'):
                final_links.append(f"{self.base_url}{link}")
            else:
                final_links.append(link)
        
        # Show some examples
        if final_links:
            print(f"   Example links: {final_links[:3]}")
        
        return final_links

    async def extract_race_info(self, content):
        """Extract additional race information from page content"""
        race_info = {
            'field_size': '',
            'race_grade': '',
            'prize_money': '',
            'track_condition': '',
            'weather': '',
            'margins': ''
        }
        
        try:
            # Extract field size by counting finishers
            finisher_pattern = r'\b(\d+)(?:st|nd|rd|th)\s+[A-Za-z]'
            finisher_matches = re.findall(finisher_pattern, content, re.IGNORECASE)
            if finisher_matches:
                race_info['field_size'] = max([int(x) for x in finisher_matches])
            
            # Extract race grade from various patterns
            grade_patterns = [
                r'(\d+rd/\d+th Grade)',
                r'(\d+st/\d+nd Grade)',
                r'(Grade \d+)',
                r'(Mixed \d+/\d+)',
                r'(Maiden)',
                r'(Free For All)',
                r'(Open)',
                r'(Listed)',
                r'(Group \d+)'
            ]
            
            for pattern in grade_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    race_info['race_grade'] = match.group(1)
                    break
            
            # Extract prize money
            prize_patterns = [
                r'PRIZE MONEY\s+\$([\d,]+)',
                r'Prize Money:\s+\$([\d,]+)',
                r'\$([\d,]+)\s+total',
                r'Prize:\s+\$([\d,]+)'
            ]
            
            for pattern in prize_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    prize_str = match.group(1).replace(',', '')
                    try:
                        race_info['prize_money'] = float(prize_str)
                    except ValueError:
                        pass
                    break
            
            # Extract track condition
            condition_patterns = [
                r'Track Condition:\s*([A-Za-z]+)',
                r'Condition:\s*([A-Za-z]+)',
                r'Track:\s*([A-Za-z]+)',
                r'\b(Good|Fast|Slow|Heavy|Wet|Dry)\b'
            ]
            
            for pattern in condition_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    condition = match.group(1).title()
                    if condition in ['Good', 'Fast', 'Slow', 'Heavy', 'Wet', 'Dry']:
                        race_info['track_condition'] = condition
                        print(f"    🏁 Found track condition: {condition}")
                        break
            
            # Extract weather (basic pattern)
            weather_patterns = [
                r'Weather:\s*([A-Za-z\s\d°C]+)',
                r'Conditions:\s*([A-Za-z\s\d°C]+)',
                r'\b(\d+°C\s+[A-Za-z]+)',
                r'\b(Clear|Sunny|Cloudy|Rainy|Overcast)\b'
            ]
            
            for pattern in weather_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    race_info['weather'] = match.group(1).strip()
                    print(f"    🌤️  Found weather: {race_info['weather']}")
                    break
            
            # Debug: Check for any weather/condition related text
            if not race_info['track_condition'] and not race_info['weather']:
                weather_keywords = ['weather', 'condition', 'track', 'temperature', 'clear', 'sunny', 'rain']
                found_keywords = [word for word in weather_keywords if word in content.lower()]
                if found_keywords:
                    print(f"    🔍 Found weather-related keywords: {found_keywords[:5]}")
            
            # Extract margins (cumulative from winner)
            margin_patterns = [
                r'MGN\s+([\d\.\s]+)',
                r'Margin[s]?:\s*([\d\.\sL]+)',
                r'([\d\.]+L\s+[\d\.]+L\s+[\d\.]+L)'
            ]
            
            for pattern in margin_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    race_info['margins'] = match.group(1).strip()
                    break
            
        except Exception as e:
            print(f"    ⚠️  Error extracting race info: {e}")
        
        return race_info

    async def search_for_race_results(self, links, location, target_date, race_number):
        """Search through candidate links for the specific race"""
        print(f"🔍 Searching for Race {race_number} results...")
        
        track_names = self.location_map.get(location, [location.lower()])
        
        for i, link in enumerate(links[:15], 1):  # Check first 15 links
            print(f"   Checking link {i}: {link}")
            
            try:
                result = await self.scraper.scrape_url(link, extract_images=False, force_playwright=True)
                
                if 'error' not in result and result.get('page_text'):
                    text = result['page_text'].lower()
                    
                    # Check if this page contains our race
                    race_indicators = [
                        f"race {race_number}",
                        f"race{race_number}",
                        f"r{race_number}",
                        f"#{race_number}"
                    ]
                    
                    track_found = any(track in text for track in track_names)
                    date_found = any(date_str in text for date_str in [
                        target_date.strftime('%Y-%m-%d'),
                        target_date.strftime('%d-%m-%Y'),
                        target_date.strftime('%d %B %Y'),
                        target_date.strftime('%d/%m/%Y')
                    ])
                    race_found = any(indicator in text for indicator in race_indicators)
                    
                    if track_found and date_found and race_found:
                        print(f"   ✅ Found matching page!")
                        return link, result
                    
                    # Also check if this page has race results that we can extract
                    if 'result' in text and 'winner' in text:
                        # Try to extract race results to see if it matches
                        race_results = await self.extract_results_from_content(result['page_text'], link)
                        if race_results:
                            print(f"   ✅ Found race results on this page!")
                            return link, result
                
                await asyncio.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"   ❌ Error checking link {link}: {e}")
                continue
        
        print("   ❌ No matching race page found")
        return None, None

    async def extract_results_from_content(self, content, url):
        """Extract race results from page content"""
        if not content:
            return None
            
        try:
            text = content.lower()
            
            # Check if this looks like a results page
            result_indicators = ['winner', 'first', '1st', 'result', 'finish', 'placing', 'time', 'margin']
            if not any(indicator in text for indicator in result_indicators):
                return None
            
            # Try to extract first 4 finishers using patterns
            # Based on actual race page structure: "1st Canya Supersonic 16.95 T: Nathan Summers"
            
            # Extract all 4 finishers and their trainers
            placings = {}
            trainers = {}
            
            for position in ['1st', '2nd', '3rd', '4th']:
                # Pattern to extract both dog name and trainer
                # "1st Dog Name time T: Trainer Name R/T:"
                trainer_patterns = [
                    rf'{position}\s+([A-Za-z][A-Za-z\s]+?)\s+[\d\.]+\s+T:\s*([A-Za-z][A-Za-z\s\'-]+?)\s+R/T:',
                    rf'{position}\s+([A-Za-z][A-Za-z\s]+?)\s+T:\s*([A-Za-z][A-Za-z\s\'-]+?)\s+R/T:',
                ]
                
                dog_name = None
                trainer_name = None
                
                # First try to extract both dog and trainer
                for pattern in trainer_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        potential_dog = match[0].strip()
                        potential_trainer = match[1].strip()
                        
                        # Clean up the extracted text
                        potential_dog = re.sub(r'[^\w\s]', ' ', potential_dog).strip()
                        potential_dog = ' '.join(potential_dog.split())
                        
                        potential_trainer = re.sub(r'[^\w\s\'-]', ' ', potential_trainer).strip()
                        potential_trainer = ' '.join(potential_trainer.split())
                        
                        if (len(potential_dog) > 3 and len(potential_dog) < 50 and
                            len(potential_trainer) > 3 and len(potential_trainer) < 50):
                            if (re.match(r'^[a-zA-Z].*[a-zA-Z]$', potential_dog) and
                                re.match(r'^[a-zA-Z].*[a-zA-Z]$', potential_trainer)):
                                dog_name = potential_dog
                                trainer_name = potential_trainer
                                break
                    
                    if dog_name and trainer_name:
                        break
                
                # If trainer extraction failed, try dog name only
                if not dog_name:
                    dog_patterns = [
                        # Pattern for: "1st Dog Name time T: Trainer Name"
                        rf'{position}\s+([A-Za-z][A-Za-z\s]+?)\s+[\d\.]+\s+T:',
                        # Pattern for: "1st Dog Name T: Trainer"
                        rf'{position}\s+([A-Za-z][A-Za-z\s]+?)\s+T:',
                        # Pattern for just "1st Dog Name" with word boundaries
                        rf'{position}\s+([A-Za-z][A-Za-z\s]{{2,25}})(?=\s|$)',
                    ]
                    
                    for pattern in dog_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                        for match in matches:
                            potential_name = match.strip()
                            # Clean up the extracted text
                            potential_name = re.sub(r'[^\w\s]', ' ', potential_name).strip()
                            potential_name = ' '.join(potential_name.split())  # Remove extra spaces
                            
                            if len(potential_name) > 3 and len(potential_name) < 50:
                                # Make sure it looks like a dog name (not just numbers or HTML)
                                if re.match(r'^[a-zA-Z].*[a-zA-Z]$', potential_name):
                                    dog_name = potential_name
                                    break
                        
                        if dog_name:
                            break
                
                if dog_name:
                    placings[position] = dog_name
                    if trainer_name:
                        trainers[position] = trainer_name
            
            # Get winner (1st place)
            winner = placings.get('1st')
            
            # Extract time if possible
            time_patterns = [
                r'time[:\s]*(\d+\.\d+)',
                r'(\d+\.\d+)\s*sec',
                r'(\d{2}\.\d{2})'
            ]
            
            race_time = None
            for pattern in time_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        race_time = float(match.group(1))
                        break
                    except ValueError:
                        continue
            
            if winner:
                # Extract additional race information
                race_info = await self.extract_race_info(content)
                
                return {
                    'winner': winner,
                    'placings': [placings.get('2nd'), placings.get('3rd'), placings.get('4th')],
                    'first': placings.get('1st'),
                    'second': placings.get('2nd'),
                    'third': placings.get('3rd'),
                    'fourth': placings.get('4th'),
                    'time': race_time,
                    'margins': race_info.get('margins', ''),
                    'field_size': race_info.get('field_size', ''),
                    'race_grade': race_info.get('race_grade', ''),
                    'prize_money': race_info.get('prize_money', ''),
                    'track_condition': race_info.get('track_condition', ''),
                    'weather': race_info.get('weather', ''),
                    'first_trainer': trainers.get('1st', ''),
                    'second_trainer': trainers.get('2nd', ''),
                    'third_trainer': trainers.get('3rd', ''),
                    'fourth_trainer': trainers.get('4th', ''),
                    'source_url': url
                }
            
            return None
            
        except Exception as e:
            print(f"    ❌ Error extracting results: {e}")
            return None

    async def fetch_race_results(self, race_number, location, date):
        """Navigate through the website to find race results"""
        print(f"🏁 Searching for Race {race_number} at {location} on {date.strftime('%Y-%m-%d')}")
        
        # Step 1: Navigate to date-specific page
        date_url, date_page = await self.navigate_to_date_page(date)
        if not date_page:
            return None
        
        # Step 2: Find links to the specific track and race
        # Use the raw HTML content from view-source for better link parsing
        html_content = date_page.get('raw_html', date_page.get('page_text', ''))
        print(f"   🗓️  HTML content sample (first 500 chars): {html_content[:500]}...")
        candidate_links = await self.find_track_and_race(html_content, location, date, race_number)
        
        if not candidate_links:
            print("   ❌ No candidate links found")
            return None
        
        # Step 3: Search through candidate links for the race results
        race_url, race_page = await self.search_for_race_results(candidate_links, location, date, race_number)
        
        if not race_page:
            return None
        
        # Step 4: Extract results from the race page
        race_results = await self.extract_results_from_content(race_page['page_text'], race_url)
        
        if race_results:
            print(f"   ✅ Found winner: {race_results['winner']}")
            return race_results
        else:
            print("   ❌ Could not extract race results from page")
            return None

    async def process_form_guides(self, folder_path):
        """Process all form guide CSVs and generate results"""
        print(f"🚀 Processing form guides from: {folder_path}")
        results_data = []
        
        # Check for existing results to avoid reprocessing
        results_csv_path = os.path.join(folder_path, 'navigator_race_results.csv')
        processed_races = set()
        
        if os.path.exists(results_csv_path):
            try:
                existing_df = pd.read_csv(results_csv_path)
                processed_races = set(existing_df['filename'].tolist())
                print(f"📋 Found {len(processed_races)} already processed races")
                # Load existing data to append to
                results_data = existing_df.to_dict('records')
            except Exception as e:
                print(f"⚠️  Could not read existing results: {e}")
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'result' not in f.lower()]
        print(f"📁 Found {len(csv_files)} CSV files to process")
        
        # Filter for races from 2022 onwards (excluding already processed ones)
        recent_files = []
        for filename in csv_files:
            try:
                race_number, location, date = self.extract_race_details(filename)
                if date.year >= 2022 and filename not in processed_races:
                    recent_files.append((filename, date))
            except:
                continue
        
        # Sort by date (most recent first)
        recent_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📁 Found {len(recent_files)} unprocessed races from 2022 onwards")
        
        for i, (filename, date) in enumerate(recent_files, 1):
            print(f"\n📄 Processing {i}/{len(recent_files)}: {filename}")
            
            try:
                # We already have the date from the filtering step
                race_number, location, _ = self.extract_race_details(filename)
                print(f"   🏁 Race {race_number} at {location} on {date.strftime('%Y-%m-%d')}")
                
                # Fetch results by navigating the website
                results = await self.fetch_race_results(race_number, location, date)
                
                if results:
                    # Add new result to existing data
                    new_result = {
                        'race_id': f"R{race_number:03d}_{date.strftime('%Y-%m-%d')}_{location}",
                        'race_date': date.strftime('%Y-%m-%d'),
                        'venue': location,
                        'race_number': race_number,
                        'first': results.get('first', ''),
                        'second': results.get('second', ''),
                        'third': results.get('third', ''),
                        'fourth': results.get('fourth', ''),
                        'time': results.get('time', ''),
                        'margins': results.get('margins', ''),
                        'field_size': results.get('field_size', ''),
                        'race_grade': results.get('race_grade', ''),
                        'prize_money': results.get('prize_money', ''),
                        'track_condition': results.get('track_condition', ''),
                        'weather': results.get('weather', ''),
                        'first_trainer': results.get('first_trainer', ''),
                        'second_trainer': results.get('second_trainer', ''),
                        'third_trainer': results.get('third_trainer', ''),
                        'fourth_trainer': results.get('fourth_trainer', ''),
                        'source_url': results.get('source_url', ''),
                        'filename': filename
                    }
                    results_data.append(new_result)
                    
                    # Show all finishers
                    finishers = [results.get('first'), results.get('second'), results.get('third'), results.get('fourth')]
                    finishers_str = ' | '.join([f"{i+1}: {name}" for i, name in enumerate(finishers) if name])
                    print(f"   ✅ Successfully processed: {finishers_str}")
                else:
                    print(f"   ❌ No results found for {filename}")
                
                # Delay between races to be respectful
                await asyncio.sleep(8)
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                continue
        
        # Create results CSV
        if results_data:
            print(f"\n💾 Preparing to save {len(results_data)} race results...")
            results_df = pd.DataFrame(results_data)
            results_csv_path = os.path.join(folder_path, 'navigator_race_results.csv')
            print(f"   📂 Saving to: {results_csv_path}")
            results_df.to_csv(results_csv_path, index=False)
            print(f"   ✅ CSV file written successfully")
            print(f"   📊 Final record count: {len(results_df)} races")
            print(f"\n💾 Results saved to: {results_csv_path}")
            print(f"📊 Successfully processed {len(results_data)} races")
        else:
            print(f"\n⚠️ No results were successfully extracted")
        
        return results_data

async def main():
    """Main function"""
    scraper = GreyhoundResultsNavigator()
    
    # Process the form guides
    folder_path = './form_guides'
    results = await scraper.process_form_guides(folder_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"NAVIGATION SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"Total races processed: {len(results)}")
    
    if results:
        print(f"Results found:")
        for result in results:
            finishers = [result.get('first'), result.get('second'), result.get('third'), result.get('fourth')]
            finishers_str = ' | '.join([f"{i+1}: {name}" for i, name in enumerate(finishers) if name])
            print(f"  - {result['race_id']}: {finishers_str}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
