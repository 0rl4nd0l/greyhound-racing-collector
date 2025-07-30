#!/usr/bin/env python3
"""
Test integration of the sportsbook-odds-scraper library with our greyhound racing system.

This demonstrates how we can use the professional scraper library instead of building our own
Selenium-based scraper from scratch.
"""

import sys
import os
from datetime import datetime
from event_scraper import EventScraper

def test_sportsbet_integration():
    """Test the Sportsbet integration with a sample URL."""
    
    # Sample URLs - these would need to be current live races
    test_urls = [
        "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604",
        # Add more current URLs here when testing
    ]
    
    print("=== Sportsbook Odds Scraper Integration Test ===")
    print(f"Test started at: {datetime.now()}")
    print()
    
    for i, url in enumerate(test_urls, 1):
        print(f"Test {i}: Testing URL - {url}")
        print("-" * 80)
        
        try:
            # Create scraper instance
            scraper = EventScraper()
            
            # Attempt to scrape the URL
            result = scraper.scrape(url)
            
            # Check results
            if scraper.error_message:
                print(f"❌ Error: {scraper.error_message}")
                print("   This could be due to:")
                print("   - Event has expired/finished")
                print("   - Geo-blocking (need Australia VPN)")
                print("   - API changes by Sportsbet")
            else:
                print(f"✅ Success!")
                print(f"   Event: {scraper.event_name}")
                print(f"   Markets: {scraper.odds_df['market_id'].nunique()}")
                print(f"   Selections: {len(scraper.odds_df)}")
                
                # Show sample data
                if scraper.odds_df is not None and len(scraper.odds_df) > 0:
                    print("\n   Sample odds data:")
                    print(scraper.odds_df.head().to_string())
                    
                    # Save to CSV for analysis
                    csv_filename = f"sportsbet_test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    scraper.odds_df.to_csv(csv_filename, index=False)
                    print(f"   Data saved to: {csv_filename}")
        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
        
        print()

def get_current_race_urls():
    """
    Helper function to get current live race URLs.
    In practice, this would scrape the main greyhound racing page to find current races.
    """
    print("To test with live races, you would need to:")
    print("1. Visit https://www.sportsbet.com.au/betting/greyhound-racing")
    print("2. Find a race that's currently available for betting")
    print("3. Copy the URL (should be like: .../race-X-XXXXXXX)")
    print("4. Update the test_urls list in this script")
    print()

def compare_with_selenium_approach():
    """Compare this API-based approach with our Selenium approach."""
    
    print("=== Comparison: API vs Selenium Approach ===")
    print()
    
    print("✅ API-based Scraper (sportsbook-odds-scraper) Advantages:")
    print("   - Much faster (no browser startup)")
    print("   - More reliable (direct API calls)")
    print("   - Lower resource usage")
    print("   - Professional error handling")
    print("   - Standardized data format")
    print("   - Supports multiple sportsbooks")
    print("   - No need to reverse-engineer DOM structure")
    print()
    
    print("❌ API-based Scraper Disadvantages:")
    print("   - Depends on undocumented APIs")
    print("   - APIs can change without notice")
    print("   - May require VPN for geo-blocked content")
    print("   - Limited to supported sportsbooks")
    print()
    
    print("🔄 Our Selenium Approach:")
    print("   - More flexible for unsupported sites")
    print("   - Can handle JavaScript-heavy pages")
    print("   - Can work around geo-blocking easier")
    print("   - But slower and more fragile")
    print()

def integration_recommendations():
    """Provide recommendations for integrating this into our system."""
    
    print("=== Integration Recommendations ===")
    print()
    
    print("1. Hybrid Approach:")
    print("   - Use API scraper as primary method (faster)")
    print("   - Fall back to Selenium when API fails")
    print("   - This gives us best of both worlds")
    print()
    
    print("2. Data Integration:")
    print("   - The scraper returns pandas DataFrames")
    print("   - Easy to integrate with our existing ML pipeline")
    print("   - Standardized format across different sportsbooks")
    print()
    
    print("3. Monitoring:")
    print("   - Add monitoring for API failures")
    print("   - Alert when switching to fallback method")
    print("   - Track success rates over time")
    print()
    
    print("4. Enhancement Opportunities:")
    print("   - Extend to other sportsbooks (TAB, Ladbrokes)")
    print("   - Real-time odds monitoring")
    print("   - Historical odds tracking")
    print("   - Arbitrage opportunity detection")

if __name__ == "__main__":
    print("Sportsbook Odds Scraper Integration Analysis")
    print("=" * 50)
    print()
    
    # Run tests
    test_sportsbet_integration()
    
    # Show guidance for getting current URLs
    get_current_race_urls()
    
    # Compare approaches
    compare_with_selenium_approach()
    
    # Provide recommendations
    integration_recommendations()
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
