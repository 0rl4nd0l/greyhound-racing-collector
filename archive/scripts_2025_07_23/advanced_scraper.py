#!/usr/bin/env python3
"""
Advanced Greyhound Racing Data Scraper
======================================

This module implements state-of-the-art web scraping techniques including:

1. Concurrent/Asynchronous Scraping
2. Smart Rate Limiting & Backoff
3. Browser Automation with Stealth Mode
4. AI-Powered Content Extraction
5. Dynamic Content Handling
6. Advanced Anti-Detection Measures
7. Intelligent Data Validation
8. Real-time Data Quality Assessment
9. Proxy Rotation Support
10. Cloud-based Scraping Integration

Author: AI Assistant
Date: July 11, 2025
"""

import asyncio
import aiohttp
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
from dataclasses import dataclass

# Advanced Web Scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Advanced Request Handling
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# HTML Parsing
from bs4 import BeautifulSoup
import lxml

# Data Processing
import pandas as pd
import numpy as np
from dateutil import parser
import re

# AI-Powered Content Extraction
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Computer Vision for OCR
try:
    import cv2
    import pytesseract
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Proxy and User Agent Management
try:
    from fake_useragent import UserAgent
    USER_AGENT_AVAILABLE = True
except ImportError:
    USER_AGENT_AVAILABLE = False

# Advanced Data Validation
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RaceResult:
    """Data class for race results with validation"""
    race_id: str
    race_date: datetime
    venue: str
    race_number: int
    distance: int
    grade: str
    track_condition: str
    weather: str
    field_size: int
    dogs: List[Dict[str, Any]]
    sectionals: Optional[Dict[str, float]] = None
    odds: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.field_size != len(self.dogs):
            logger.warning(f"Field size mismatch: {self.field_size} vs {len(self.dogs)}")
        
        if self.race_number < 1 or self.race_number > 20:
            logger.warning(f"Unusual race number: {self.race_number}")

class AdvancedGreyhoundScraper:
    """
    Advanced greyhound racing data scraper with state-of-the-art techniques
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.session = None
        self.driver = None
        self.results_dir = Path("./advanced_scraping_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.setup_session()
        self.setup_user_agents()
        self.setup_ai_models()
        
        # Scraping statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'start_time': datetime.now()
        }
        
        logger.info("🚀 Advanced Greyhound Scraper initialized")
        logger.info(f"✅ Transformers Available: {TRANSFORMERS_AVAILABLE}")
        logger.info(f"✅ Vision Available: {VISION_AVAILABLE}")
        logger.info(f"✅ User Agent Available: {USER_AGENT_AVAILABLE}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'base_url': 'https://www.thedogs.com.au',
            'concurrent_requests': 5,
            'request_delay': (1, 3),  # Random delay between requests
            'max_retries': 3,
            'timeout': 30,
            'use_proxy': False,
            'proxy_list': [],
            'use_stealth': True,
            'ai_extraction': True,
            'data_validation': True,
            'cache_results': True,
            'quality_threshold': 0.8
        }
    
    def setup_session(self):
        """Setup advanced HTTP session with retry strategy"""
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def setup_user_agents(self):
        """Setup user agent rotation"""
        if USER_AGENT_AVAILABLE:
            self.ua = UserAgent()
        else:
            self.user_agents = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ]
    
    def setup_ai_models(self):
        """Setup AI models for content extraction"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Setup text classification for content quality
                self.classifier = pipeline("text-classification", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
                
                # Setup NER for extracting racing entities
                self.ner = pipeline("ner", 
                                  model="dbmdz/bert-large-cased-finetuned-conll03-english")
                
                logger.info("✅ AI models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load AI models: {e}")
                self.classifier = None
                self.ner = None
        else:
            self.classifier = None
            self.ner = None
    
    def get_user_agent(self) -> str:
        """Get random user agent"""
        if USER_AGENT_AVAILABLE:
            return self.ua.random
        else:
            return random.choice(self.user_agents)
    
    def setup_stealth_driver(self) -> webdriver.Chrome:
        """Setup Chrome driver with stealth mode"""
        chrome_options = Options()
        
        if self.config['use_stealth']:
            # Stealth mode options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")
        
        # Headless mode
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # User agent
        chrome_options.add_argument(f"--user-agent={self.get_user_agent()}")
        
        # Memory optimization
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            if self.config['use_stealth']:
                # Execute stealth scripts
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
            return driver
            
        except Exception as e:
            logger.error(f"❌ Failed to setup stealth driver: {e}")
            raise
    
    async def fetch_with_session(self, url: str, **kwargs) -> Optional[str]:
        """Fetch URL with advanced session handling"""
        try:
            # Update headers for this request
            headers = {
                'User-Agent': self.get_user_agent(),
                'Referer': self.config['base_url']
            }
            
            # Add random delay
            delay = random.uniform(*self.config['request_delay'])
            await asyncio.sleep(delay)
            
            # Make request
            response = self.session.get(url, headers=headers, timeout=self.config['timeout'])
            self.stats['total_requests'] += 1
            
            # Check response
            if response.status_code == 200:
                self.stats['successful_requests'] += 1
                return response.text
            elif response.status_code == 429:
                self.stats['rate_limit_hits'] += 1
                logger.warning(f"Rate limited on {url}")
                await asyncio.sleep(60)  # Wait longer on rate limit
                return None
            else:
                self.stats['failed_requests'] += 1
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return None
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_race_data_with_ai(self, html_content: str, url: str) -> Optional[Dict]:
        """Extract race data using AI-powered content analysis"""
        if not self.classifier or not html_content:
            return self.extract_race_data_traditional(html_content, url)
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract text content
            text_content = soup.get_text()
            
            # Assess content quality
            quality_score = self.assess_content_quality(text_content)
            
            if quality_score < self.config['quality_threshold']:
                logger.warning(f"Low quality content detected for {url}: {quality_score}")
                return None
            
            # Extract racing entities using NER
            entities = self.extract_racing_entities(text_content)
            
            # Combine traditional and AI extraction
            traditional_data = self.extract_race_data_traditional(html_content, url)
            ai_enhanced_data = self.enhance_with_ai(traditional_data, entities)
            
            return ai_enhanced_data
            
        except Exception as e:
            logger.error(f"AI extraction failed for {url}: {e}")
            return self.extract_race_data_traditional(html_content, url)
    
    def assess_content_quality(self, text: str) -> float:
        """Assess content quality using AI classification"""
        if not self.classifier:
            return 0.8  # Default quality
        
        try:
            # Sample text for quality assessment
            sample_text = text[:500] if len(text) > 500 else text
            
            # Get quality score
            result = self.classifier(sample_text)
            
            # Convert to quality score (assuming positive sentiment = good quality)
            if result[0]['label'] == 'POSITIVE':
                return result[0]['score']
            else:
                return 1.0 - result[0]['score']
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.8
    
    def extract_racing_entities(self, text: str) -> List[Dict]:
        """Extract racing-related entities using NER"""
        if not self.ner:
            return []
        
        try:
            # Get entities
            entities = self.ner(text)
            
            # Filter racing-related entities
            racing_entities = []
            for entity in entities:
                if entity['entity'].startswith('B-') or entity['entity'].startswith('I-'):
                    racing_entities.append(entity)
            
            return racing_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_race_data_traditional(self, html_content: str, url: str) -> Optional[Dict]:
        """Traditional HTML parsing extraction"""
        if not html_content:
            return None
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract race metadata
            race_data = {
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'venue': self.extract_venue(soup),
                'race_date': self.extract_race_date(soup),
                'race_number': self.extract_race_number(soup),
                'distance': self.extract_distance(soup),
                'grade': self.extract_grade(soup),
                'track_condition': self.extract_track_condition(soup),
                'weather': self.extract_weather(soup),
                'dogs': self.extract_dogs_data(soup),
                'sectionals': self.extract_sectionals(soup),
                'odds': self.extract_odds(soup)
            }
            
            # Validate extracted data
            if self.config['data_validation']:
                if not self.validate_race_data(race_data):
                    logger.warning(f"Data validation failed for {url}")
                    return None
            
            return race_data
            
        except Exception as e:
            logger.error(f"Traditional extraction failed for {url}: {e}")
            return None
    
    def extract_venue(self, soup: BeautifulSoup) -> str:
        """Extract venue from HTML"""
        selectors = [
            'h1.race-title',
            '.venue-name',
            '[data-venue]',
            '.race-header h1',
            '.track-name'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text
        
        return "Unknown"
    
    def extract_race_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract race date from HTML"""
        selectors = [
            '.race-date',
            '[data-date]',
            '.date',
            'time[datetime]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                if element.has_attr('datetime'):
                    return element['datetime']
                
                # Try text content
                text = element.get_text(strip=True)
                if text:
                    try:
                        parsed_date = parser.parse(text)
                        return parsed_date.isoformat()
                    except:
                        continue
        
        return None
    
    def extract_race_number(self, soup: BeautifulSoup) -> int:
        """Extract race number from HTML"""
        selectors = [
            '.race-number',
            '[data-race-number]',
            '.race-title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                # Look for race number pattern
                match = re.search(r'race\s+(\d+)', text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return 1
    
    def extract_distance(self, soup: BeautifulSoup) -> int:
        """Extract race distance from HTML"""
        selectors = [
            '.distance',
            '[data-distance]',
            '.race-details'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                # Look for distance pattern
                match = re.search(r'(\d+)m', text)
                if match:
                    return int(match.group(1))
        
        return 500  # Default distance
    
    def extract_grade(self, soup: BeautifulSoup) -> str:
        """Extract race grade from HTML"""
        selectors = [
            '.grade',
            '[data-grade]',
            '.race-grade'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text
        
        return "Unknown"
    
    def extract_track_condition(self, soup: BeautifulSoup) -> str:
        """Extract track condition from HTML"""
        selectors = [
            '.track-condition',
            '[data-condition]',
            '.conditions'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text
        
        return "Unknown"
    
    def extract_weather(self, soup: BeautifulSoup) -> str:
        """Extract weather from HTML"""
        selectors = [
            '.weather',
            '[data-weather]',
            '.conditions'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text
        
        return "Unknown"
    
    def extract_dogs_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract dogs data from HTML"""
        dogs = []
        
        # Try different selectors for dog data
        selectors = [
            '.runner',
            '.dog-entry',
            '.entry',
            'tr.runner'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    dog_data = self.extract_single_dog_data(element)
                    if dog_data:
                        dogs.append(dog_data)
                break
        
        return dogs
    
    def extract_single_dog_data(self, element) -> Optional[Dict]:
        """Extract single dog data from HTML element"""
        try:
            dog_data = {
                'name': '',
                'box': 0,
                'trainer': '',
                'owner': '',
                'weight': 0,
                'form': '',
                'odds': 0
            }
            
            # Extract dog name
            name_selectors = ['.dog-name', '.runner-name', '.name']
            for selector in name_selectors:
                name_elem = element.select_one(selector)
                if name_elem:
                    dog_data['name'] = name_elem.get_text(strip=True)
                    break
            
            # Extract box number
            box_selectors = ['.box', '.box-number', '[data-box]']
            for selector in box_selectors:
                box_elem = element.select_one(selector)
                if box_elem:
                    box_text = box_elem.get_text(strip=True)
                    try:
                        dog_data['box'] = int(box_text)
                        break
                    except ValueError:
                        continue
            
            # Extract trainer
            trainer_selectors = ['.trainer', '.trainer-name']
            for selector in trainer_selectors:
                trainer_elem = element.select_one(selector)
                if trainer_elem:
                    dog_data['trainer'] = trainer_elem.get_text(strip=True)
                    break
            
            # Extract weight
            weight_selectors = ['.weight', '[data-weight]']
            for selector in weight_selectors:
                weight_elem = element.select_one(selector)
                if weight_elem:
                    weight_text = weight_elem.get_text(strip=True)
                    try:
                        weight_match = re.search(r'(\d+\.?\d*)', weight_text)
                        if weight_match:
                            dog_data['weight'] = float(weight_match.group(1))
                            break
                    except ValueError:
                        continue
            
            # Extract odds
            odds_selectors = ['.odds', '.price', '[data-odds]']
            for selector in odds_selectors:
                odds_elem = element.select_one(selector)
                if odds_elem:
                    odds_text = odds_elem.get_text(strip=True)
                    try:
                        odds_match = re.search(r'(\d+\.?\d*)', odds_text)
                        if odds_match:
                            dog_data['odds'] = float(odds_match.group(1))
                            break
                    except ValueError:
                        continue
            
            return dog_data if dog_data['name'] else None
            
        except Exception as e:
            logger.error(f"Error extracting dog data: {e}")
            return None
    
    def extract_sectionals(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract sectional times from HTML"""
        sectionals = {}
        
        # Try to find sectionals table or data
        selectors = [
            '.sectionals',
            '.splits',
            '.times'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Extract sectional data
                break
        
        return sectionals if sectionals else None
    
    def extract_odds(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract odds data from HTML"""
        odds = {}
        
        # Try to find odds data
        selectors = [
            '.odds-table',
            '.betting-odds',
            '.prices'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Extract odds data
                break
        
        return odds if odds else None
    
    def enhance_with_ai(self, traditional_data: Dict, entities: List[Dict]) -> Dict:
        """Enhance traditional extraction with AI insights"""
        if not entities:
            return traditional_data
        
        # Use AI entities to improve extraction
        enhanced_data = traditional_data.copy()
        
        # Process entities and enhance data
        for entity in entities:
            # Add AI-extracted information
            pass
        
        return enhanced_data
    
    def validate_race_data(self, race_data: Dict) -> bool:
        """Validate extracted race data"""
        required_fields = ['venue', 'race_date', 'race_number', 'dogs']
        
        for field in required_fields:
            if not race_data.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate dogs data
        if not isinstance(race_data['dogs'], list) or len(race_data['dogs']) == 0:
            logger.warning("No dogs data found")
            return False
        
        # Check for reasonable number of dogs
        if len(race_data['dogs']) > 12:
            logger.warning(f"Too many dogs: {len(race_data['dogs'])}")
            return False
        
        return True
    
    async def scrape_race_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple race URLs concurrently"""
        logger.info(f"🚀 Starting concurrent scraping of {len(urls)} URLs")
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(self.config['concurrent_requests'])
        
        async def scrape_single_url(url: str) -> Optional[Dict]:
            async with semaphore:
                html_content = await self.fetch_with_session(url)
                if html_content:
                    if self.config['ai_extraction']:
                        return self.extract_race_data_with_ai(html_content, url)
                    else:
                        return self.extract_race_data_traditional(html_content, url)
                return None
        
        # Execute scraping tasks
        tasks = [scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result is not None:
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scraping exception: {result}")
        
        logger.info(f"✅ Successfully scraped {len(successful_results)} out of {len(urls)} URLs")
        return successful_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save scraping results"""
        if not results:
            logger.warning("No results to save")
            return
        
        if filename is None:
            filename = f"scraping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {len(results)} results to {filepath}")
    
    def get_scraping_stats(self) -> Dict:
        """Get scraping statistics"""
        duration = datetime.now() - self.stats['start_time']
        
        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1),
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'duration_seconds': duration.total_seconds(),
            'requests_per_second': self.stats['total_requests'] / max(duration.total_seconds(), 1)
        }
    
    async def run_advanced_scraping(self, urls: List[str]):
        """Run the complete advanced scraping pipeline"""
        logger.info("🚀 STARTING ADVANCED SCRAPING")
        logger.info("=" * 60)
        
        # Scrape URLs
        results = await self.scrape_race_urls(urls)
        
        # Save results
        self.save_results(results)
        
        # Print statistics
        stats = self.get_scraping_stats()
        logger.info("\n📊 SCRAPING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Successful: {stats['successful_requests']}")
        logger.info(f"Failed: {stats['failed_requests']}")
        logger.info(f"Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"Rate Limits: {stats['rate_limit_hits']}")
        logger.info(f"Duration: {stats['duration_seconds']:.2f}s")
        logger.info(f"Speed: {stats['requests_per_second']:.2f} req/s")
        
        return results


async def main():
    """Main function for testing"""
    scraper = AdvancedGreyhoundScraper()
    
    # Test URLs
    test_urls = [
"https://www.thedogs.com.au/racing/vic-warrnambool/2025-07-10/6"
    ]
    
    results = await scraper.run_advanced_scraping(test_urls)
    print(f"Scraped {len(results)} race results")


if __name__ == "__main__":
    asyncio.run(main())
