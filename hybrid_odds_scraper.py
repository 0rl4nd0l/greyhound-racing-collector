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
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Import our ChromeDriver helper
from drivers import get_chrome_driver, setup_selenium_driver_path

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
        """Setup Chrome WebDriver using webdriver-manager."""
        try:
            # Use our drivers.py helper which includes webdriver-manager
            setup_selenium_driver_path()
            self.driver = get_chrome_driver(headless=self.use_headless)
            self.driver.set_page_load_timeout(self.timeout)

            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {str(e)}")
            return False
    
    def _cleanup_driver(self):
        """Clean up WebDriver resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            finally:
                self.driver = None
    
    def _extract_odds_with_selenium(self) -> list:
        """Extract odds using Selenium - simplified implementation."""
        odds_data = []
        
        try:
            # This is a simplified version - you'd need to adapt based on current DOM structure
            # Look for runner containers
            runner_containers = self.driver.find_elements(By.CSS_SELECTOR, \
                "[data-automation-id*='runner'], .runner-card, .outcome-button")
            
            self.logger.info(f"Found {len(runner_containers)} potential runner elements")
            
            for i, container in enumerate(runner_containers[:20]):  # Limit for testing
                try:
                    # Extract runner name
                    name_element = container.find_element(By.CSS_SELECTOR, \
                        "[data-automation-id*='name'], .runner-name, .selection-name")
                    runner_name = name_element.text.strip()
                    
                    # Extract odds
                    odds_element = container.find_element(By.CSS_SELECTOR,\
                        "[data-automation-id*='price'], .price, .odds")
                    odds_text = odds_element.text.strip()
                    
                    if runner_name and odds_text:
                        odds_data.append({
                            'runner_name': runner_name,
                            'odds': odds_text,
                            'market': 'Win',
                            'selection_id': f'selenium_{i}',
                            'market_id': 'win_market'
                        })
                        
                except (NoSuchElementException, Exception) as e:
                    # Skip this runner if we can't extract data
                    continue
            
        except Exception as e:
            self.logger.error(f"Error extracting odds with Selenium: {str(e)}")
        
        return odds_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get success statistics."""
        total_attempts = sum(self.success_stats.values())
        
        if total_attempts == 0:
            return {'message': 'No scraping attempts yet'}
        
        api_total = self.success_stats['api_success'] + self.success_stats['api_failure']
        selenium_total = self.success_stats['selenium_success'] + self.success_stats['selenium_failure']
        
        stats = {
            'total_attempts': total_attempts,
            'api_attempts': api_total,
            'api_success_rate': (self.success_stats['api_success'] / api_total * 100) if api_total > 0 else 0,
            'selenium_attempts': selenium_total,
            'selenium_success_rate': (self.success_stats['selenium_success'] / selenium_total * 100) if selenium_total > 0 else 0,
            'overall_success_rate': ((self.success_stats['api_success'] + self.success_stats['selenium_success']) / total_attempts * 100),
            'last_method_used': self.last_method_used
        }
        
        return stats
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self._cleanup_driver()

def demo_hybrid_scraper():
    """Demo the hybrid scraper."""
    print("=== Hybrid Odds Scraper Demo ===")
    print()
    
    # Initialize scraper
    scraper = HybridOddsScraper(use_headless=True, timeout=20)
    
    # Test URLs - replace with current live races
    test_urls = [
        "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604",
        # Add more current URLs for testing
    ]
    
    for url in test_urls:
        print(f"Testing: {url}")
        print("-" * 80)
        
        # Scrape odds
        odds_df, metadata = scraper.scrape_odds(url)
        
        # Display results
        if metadata['success']:
            print(f"✅ Success using {metadata['method_used']} method")
            print(f"   Markets: {metadata['markets_count']}")
            print(f"   Selections: {metadata['selections_count']}")
            
            if odds_df is not None:
                print("\n   Sample data:")
                print(odds_df.head().to_string())
        else:
            print(f"❌ Failed: {metadata.get('error_message', 'Unknown error')}")
        
        print()
    
    # Show statistics
    stats = scraper.get_stats()
    print("=== Scraping Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_hybrid_scraper()
