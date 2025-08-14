#!/usr/bin/env python3
"""
Simple test script to check if race results are available for a specific date
"""

import asyncio
import sys

sys.path.append("..")
from datetime import datetime

from greyhound_results_scraper_navigator import GreyhoundResultsNavigator


async def test_race_download():
    print("🧪 Testing race download for specific date...")

    scraper = GreyhoundResultsNavigator()

    # Test with a specific date that should have races
    test_date = datetime(2025, 7, 10).date()

    print(f"📅 Testing date: {test_date}")

    # Test with Richmond venue - this was mentioned in the page content
    test_venue = "richmond"
    test_race_number = 1

    print(f"🏁 Testing venue: {test_venue}, race: {test_race_number}")

    try:
        results = await scraper.fetch_race_results(
            race_number=test_race_number,
            location="RICH",  # Using the venue code
            date=test_date,
        )

        if results:
            print(f"✅ Success! Found results: {results}")
            return True
        else:
            print("❌ No results found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_race_download())
