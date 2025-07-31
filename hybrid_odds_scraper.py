#!/usr/bin/env python3
"""
Hybrid Odds Scraper for Greyhound Racing

This combines the professional sportsbook-odds-scraper library (API-based) with our custom
Selenium-based scraper as a fallback. This gives us the best of both worlds:
- Fast, reliable API access when available
- Flexible DOM scraping when APIs fail

Author: Orlando Lee
Date: July 27, 2025
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
# Our fallback Selenium scraper (simplified)
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Professional API scraper
from event_scraper import EventScraper
from scraper_exception import ScraperException


class HybridOddsScraper:
    """
    Hybrid scraper that tries API-based scraping first, then falls back to Selenium.
    """
    
    def __init__(self, use_headless: bool = True, timeout: int = 30):
        """
        Initialize the hybrid scraper.
        
        Args:
            use_headless: Whether to run Selenium in headless mode
            timeout: Timeout for web requests
        """
        self.use_headless = use_headless
        self.timeout = timeout
        self.driver = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track which method was used
        self.last_method_used = None
        self.success_stats = {
            'api_success': 0,
            'api_failure': 0,
            'selenium_success': 0,
            'selenium_failure': 0
        }
    
    def scrape_odds(self, url: str, max_retries: int = 2) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Scrape odds from the given URL using hybrid approach.
        
        Args:
            url: The race URL to scrape
            max_retries: Maximum number of retries for each method
            
        Returns:
            Tuple of (DataFrame with odds data, metadata dict)
        """
        metadata = {
            'url': url,
            'timestamp': datetime.now(),
            'method_used': None,
            'success': False,
            'error_message': None,
            'event_name': None,
            'markets_count': 0,
            'selections_count': 0
        }
        
        # Method 1: Try API-based scraper first
        self.logger.info(f"Attempting API-based scraping for: {url}")
        odds_df, api_success = self._try_api_scraper(url, max_retries)
        
        if api_success and odds_df is not None:
            self.logger.info("✅ API-based scraping successful")
            self.success_stats['api_success'] += 1
            self.last_method_used = 'api'
            metadata.update({
                'method_used': 'api',
                'success': True,
                'markets_count': odds_df['market_id'].nunique() if 'market_id' in odds_df.columns else 0,
                'selections_count': len(odds_df)
            })
            return odds_df, metadata
        else:
            self.logger.warning("❌ API-based scraping failed, trying Selenium fallback")
            self.success_stats['api_failure'] += 1
        
        # Method 2: Fallback to Selenium scraper
        self.logger.info("Attempting Selenium-based scraping")
        odds_df, selenium_success = self._try_selenium_scraper(url, max_retries)
        
        if selenium_success and odds_df is not None:
            self.logger.info("✅ Selenium-based scraping successful")
            self.success_stats['selenium_success'] += 1
            self.last_method_used = 'selenium'
            metadata.update({
                'method_used': 'selenium',
                'success': True,
                'markets_count': len(odds_df['market'].unique()) if 'market' in odds_df.columns else 0,
                'selections_count': len(odds_df)
            })
            return odds_df, metadata
        else:
            self.logger.error("❌ Both scraping methods failed")
            self.success_stats['selenium_failure'] += 1
            metadata.update({
                'method_used': 'both_failed',
                'success': False,
                'error_message': 'Both API and Selenium methods failed'
            })
            return None, metadata
    
    def _try_api_scraper(self, url: str, max_retries: int) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        Try the professional API-based scraper.
        
        Returns:
            Tuple of (DataFrame, success_boolean)
        """
        for attempt in range(max_retries):
            try:
                scraper = EventScraper()
                result = scraper.scrape(url)
                
                if scraper.error_message:
                    self.logger.warning(f"API scraper error (attempt {attempt + 1}): {scraper.error_message}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                if scraper.odds_df is not None and len(scraper.odds_df) > 0:
                    self.logger.info(f"API scraper got {len(scraper.odds_df)} selections")
                    return scraper.odds_df, True
                    
            except Exception as e:
                self.logger.error(f"API scraper exception (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None, False
    
    def _try_selenium_scraper(self, url: str, max_retries: int) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        Try our custom Selenium-based scraper as fallback.
        
        Returns:
            Tuple of (DataFrame, success_boolean)
        """
        for attempt in range(max_retries):
            try:
                # Setup Chrome driver
                if not self._setup_driver():
                    return None, False
                
                self.logger.info(f"Loading page with Selenium (attempt {attempt + 1})")
                self.driver.get(url)
                
                # Wait for page to load
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Look for runner containers - this is a simplified version
                odds_data = self._extract_odds_with_selenium()
                
                if odds_data and len(odds_data) > 0:
                    df = pd.DataFrame(odds_data)
                    self.logger.info(f"Selenium scraper got {len(df)} selections")
                    return df, True
                
            except Exception as e:
                self.logger.error(f"Selenium scraper exception (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            finally:
                self._cleanup_driver()
        
        return None, False
    
    def _setup_driver(self) -> bool:
        """Setup Chrome WebDriver."""
        try:
            options = Options()
            if self.use_headless:
                options.add_argument('--headless')
            
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')\
            
            self.driver = webdriver.Chrome(options=options)\n            self.driver.set_page_load_timeout(self.timeout)\n            return True\n            \n        except Exception as e:\n            self.logger.error(f\"Failed to setup Chrome driver: {str(e)}\")\n            return False\n    \n    def _cleanup_driver(self):\n        \"\"\"Clean up WebDriver resources.\"\"\"\n        if self.driver:\n            try:\n                self.driver.quit()\n            except:\n                pass\n            finally:\n                self.driver = None\n    \n    def _extract_odds_with_selenium(self) -> list:\n        \"\"\"Extract odds using Selenium - simplified implementation.\"\"\"\n        odds_data = []\n        \n        try:\n            # This is a simplified version - you'd need to adapt based on current DOM structure\n            # Look for runner containers\n            runner_containers = self.driver.find_elements(By.CSS_SELECTOR, \n                \"[data-automation-id*='runner'], .runner-card, .outcome-button\")\n            \n            self.logger.info(f\"Found {len(runner_containers)} potential runner elements\")\n            \n            for i, container in enumerate(runner_containers[:20]):  # Limit for testing\n                try:\n                    # Extract runner name\n                    name_element = container.find_element(By.CSS_SELECTOR, \n                        \"[data-automation-id*='name'], .runner-name, .selection-name\")\n                    runner_name = name_element.text.strip()\n                    \n                    # Extract odds\n                    odds_element = container.find_element(By.CSS_SELECTOR,\n                        \"[data-automation-id*='price'], .price, .odds\")\n                    odds_text = odds_element.text.strip()\n                    \n                    if runner_name and odds_text:\n                        odds_data.append({\n                            'runner_name': runner_name,\n                            'odds': odds_text,\n                            'market': 'Win',\n                            'selection_id': f'selenium_{i}',\n                            'market_id': 'win_market'\n                        })\n                        \n                except (NoSuchElementException, Exception) as e:\n                    # Skip this runner if we can't extract data\n                    continue\n            \n        except Exception as e:\n            self.logger.error(f\"Error extracting odds with Selenium: {str(e)}\")\n        \n        return odds_data\n    \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get success statistics.\"\"\"\n        total_attempts = sum(self.success_stats.values())\n        \n        if total_attempts == 0:\n            return {'message': 'No scraping attempts yet'}\n        \n        api_total = self.success_stats['api_success'] + self.success_stats['api_failure']\n        selenium_total = self.success_stats['selenium_success'] + self.success_stats['selenium_failure']\n        \n        stats = {\n            'total_attempts': total_attempts,\n            'api_attempts': api_total,\n            'api_success_rate': (self.success_stats['api_success'] / api_total * 100) if api_total > 0 else 0,\n            'selenium_attempts': selenium_total,\n            'selenium_success_rate': (self.success_stats['selenium_success'] / selenium_total * 100) if selenium_total > 0 else 0,\n            'overall_success_rate': ((self.success_stats['api_success'] + self.success_stats['selenium_success']) / total_attempts * 100),\n            'last_method_used': self.last_method_used\n        }\n        \n        return stats\n    \n    def __del__(self):\n        \"\"\"Cleanup when object is destroyed.\"\"\"\n        self._cleanup_driver()\n\n\ndef demo_hybrid_scraper():\n    \"\"\"Demo the hybrid scraper.\"\"\"\n    print(\"=== Hybrid Odds Scraper Demo ===\")\n    print()\n    \n    # Initialize scraper\n    scraper = HybridOddsScraper(use_headless=True, timeout=20)\n    \n    # Test URLs - replace with current live races\n    test_urls = [\n        \"https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604\",\n        # Add more current URLs for testing\n    ]\n    \n    for url in test_urls:\n        print(f\"Testing: {url}\")\n        print(\"-\" * 80)\n        \n        # Scrape odds\n        odds_df, metadata = scraper.scrape_odds(url)\n        \n        # Display results\n        if metadata['success']:\n            print(f\"✅ Success using {metadata['method_used']} method\")\n            print(f\"   Markets: {metadata['markets_count']}\")\n            print(f\"   Selections: {metadata['selections_count']}\")\n            \n            if odds_df is not None:\n                print(\"\\n   Sample data:\")\n                print(odds_df.head().to_string())\n        else:\n            print(f\"❌ Failed: {metadata.get('error_message', 'Unknown error')}\")\n        \n        print()\n    \n    # Show statistics\n    stats = scraper.get_stats()\n    print(\"=== Scraping Statistics ===\")\n    for key, value in stats.items():\n        print(f\"{key}: {value}\")\n\n\nif __name__ == \"__main__\":\n    demo_hybrid_scraper()"
