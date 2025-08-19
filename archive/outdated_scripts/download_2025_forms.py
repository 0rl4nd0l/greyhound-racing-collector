#!/usr/bin/env python3
"""
Simple 2025 Form Guide Downloader
Downloads form guides from thedogs.com.au for upcoming 2025 races.
"""

import asyncio
import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests

sys.path.append("..")
from greyhound_results_scraper_navigator import GreyhoundResultsNavigator


class Simple2025FormDownloader:
    def __init__(self):
        self.base_url = "https://www.thedogs.com.au"
        self.download_dir = "unprocessed"
        self.scraper = GreyhoundResultsNavigator()

        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)

        # Keep track of existing files
        self.existing_files = set()
        if os.path.exists(self.download_dir):
            self.existing_files = set(os.listdir(self.download_dir))

        # Venue mapping for consistent naming
        self.venue_map = {
            "albion-park": "ALBION",
            "angle-park": "ANGLE",
            "ballarat": "BALLARAT",
            "bendigo": "BENDIGO",
            "bulli": "BULLI",
            "capalaba": "CAPALABA",
            "casino": "CASINO",
            "cranbourne": "CRANBOURNE",
            "dapto": "DAPTO",
            "devonport": "DEVONPORT",
            "dubbo": "DUBBO",
            "gawler": "GAWLER",
            "geelong": "GEELONG",
            "gosford": "GOSFORD",
            "healesville": "HEALESVILLE",
            "hobart": "HOBART",
            "horsham": "HORSHAM",
            "ipswich": "IPSWICH",
            "launcelon": "LAUNCELON",
            "lismore": "LISMORE",
            "mandurah": "MANDURAH",
            "meadows": "MEADOWS",
            "mildura": "MILDURA",
            "mount-gambier": "MOUNT_GAMBIER",
            "murray-bridge": "MURRAY_BRIDGE",
            "newcastle": "NEWCASTLE",
            "nowra": "NOWRA",
            "orange": "ORANGE",
            "palmerston": "PALMERSTON",
            "parkes": "PARKES",
            "penrith": "PENRITH",
            "pinjarra": "PINJARRA",
            "richmond": "RICHMOND",
            "rockhampton": "ROCKHAMPTON",
            "sale": "SALE",
            "sandown-park": "SANDOWN",
            "shepparton": "SHEPPARTON",
            "springvale": "SPRINGVALE",
            "strathalbyn": "STRATHALBYN",
            "tamworth": "TAMWORTH",
            "temora": "TEMORA",
            "the-meadows": "THE_MEADOWS",
            "traralgon": "TRARALGON",
            "townsville": "TOWNSVILLE",
            "wagga": "WAGGA",
            "wangaratta": "WANGARATTA",
            "warragul": "WARRAGUL",
            "warrnambool": "WARRNAMBOOL",
            "wentworth-park": "WENTWORTH",
            "whyalla": "WHYALLA",
            "winton": "WINTON",
            "wodonga": "WODONGA",
            "yarra-valley": "YARRA_VALLEY",
            "young": "YOUNG",
        }

    def get_recent_dates(self):
        """Get a list of recent dates to check for races"""
        dates = []
        today = datetime.now().date()

        # Check today
        dates.append(today)

        # Check yesterday
        yesterday = today - timedelta(days=1)
        dates.append(yesterday)

        # Check next 3 days
        for i in range(1, 4):
            check_date = today + timedelta(days=i)
            dates.append(check_date)

        # Check previous 3 days
        for i in range(2, 5):
            check_date = today - timedelta(days=i)
            dates.append(check_date)

        return sorted(list(set(dates)), reverse=True)

    async def find_races_for_date(self, date):
        """Find races for a specific date"""
        date_str = date.strftime("%Y-%m-%d")
        url = f"{self.base_url}/racing/{date_str}"

        print(f"🔍 Checking {date_str}...")

        try:
            result = await self.scraper.scraper.scrape_url(url, extract_images=False)

            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
                return []

            page_content = result.get("page_text", "")
            html_content = result.get("page_html", "") or result.get("raw_html", "")

            # Debug: Show what keys are available
            print(f"   🔍 Available result keys: {list(result.keys())}")
            print(f"   🔍 HTML content length: {len(html_content)}")
            print(f"   🔍 Page content length: {len(page_content)}")

            # Search in both text and HTML content
            search_content = html_content if html_content else page_content

            # Look for venue names in the content and construct race URLs
            venue_names = []
            for venue_key, venue_code in self.venue_map.items():
                if venue_key in search_content.lower() or venue_code in search_content:
                    venue_names.append(venue_key)

            print(f"   🔍 Found venues in content: {venue_names}")

            # Construct race URLs for found venues
            race_urls = []
            for venue in venue_names:
                # Try race numbers 1-12 for each venue
                for race_num in range(1, 13):
                    race_url = f"{self.base_url}/racing/{venue}/{date_str}/{race_num}"
                    race_urls.append(race_url)

            print(f"   🔍 Constructed {len(race_urls)} potential race URLs")

            # Debug: Show sample of page content and what links we found
            print(
                f"   📝 Page content sample (first 1000 chars): {page_content[:1000]}..."
            )

            if race_urls:
                race_urls = list(set(race_urls))  # Remove duplicates
                print(f"   ✅ Found {len(race_urls)} races for {date_str}")
                return race_urls
            else:
                print(f"   ⚪ No races found for {date_str}")
                return []

        except Exception as e:
            print(f"   ❌ Error checking {date_str}: {e}")
            return []

    def extract_race_info(self, race_url):
        """Extract race information from URL"""
        pattern = r"/racing/([^/]+)/(\d{4}-\d{2}-\d{2})/(\d+)"
        match = re.search(pattern, race_url)

        if match:
            venue = match.group(1)
            date_str = match.group(2)
            race_number = match.group(3)

            # Convert venue to our code
            venue_code = self.venue_map.get(venue, venue.upper())

            # Convert date
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_formatted = date_obj.strftime("%d %B %Y")

            return {
                "venue": venue,
                "venue_code": venue_code,
                "date": date_formatted,
                "race_number": race_number,
                "filename": f"Race {race_number} - {venue_code} - {date_formatted}.csv",
            }

        return None

    async def download_form_guide(self, race_url):
        """Navigate to race page and export form guide CSV"""
        race_info = self.extract_race_info(race_url)
        if not race_info:
            return False

        filename = race_info["filename"]

        # Check if we already have this file
        if filename in self.existing_files:
            print(f"   ⚪ Already have {filename}")
            return True

        print(f"   🔄 Downloading {filename}...")

        try:
            # Get the date object from the race info
            date_obj = datetime.strptime(race_info["date"], "%d %B %Y")

            # Use the navigator's method to download the race results
            results = await self.scraper.fetch_race_results(
                race_number=int(race_info["race_number"]),
                location=race_info["venue_code"],
                date=date_obj,
            )

            if results:
                # Save the results as a CSV file
                filepath = os.path.join(self.download_dir, filename)
                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Field", "Value"])
                    for key, value in results.items():
                        writer.writerow([key, value])

                self.existing_files.add(filename)
                print(f"   ✅ Downloaded {filename}")
                return True
            else:
                print(f"   ❌ Could not download CSV for {filename}")
                return False

        except Exception as e:
            print(f"   ❌ Error downloading {filename}: {e}")
            return False

    async def download_forms(self):
        """Main method to download form guides"""
        print("🚀 Starting Form Guide Download")
        print("=" * 60)

        dates = self.get_recent_dates()
        total_downloaded = 0

        for date in dates:
            race_urls = await self.find_races_for_date(date)

            for race_url in race_urls:
                if await self.download_form_guide(race_url):
                    total_downloaded += 1

                # Small delay between downloads
                time.sleep(1)

            # Delay between dates
            time.sleep(2)

        print(f"\n🎉 Download complete!")
        print(f"📊 Total new files downloaded: {total_downloaded}")
        print(f"📁 Files saved to: {self.download_dir}")


async def main():
    downloader = Simple2025FormDownloader()
    await downloader.download_forms()


if __name__ == "__main__":
    asyncio.run(main())
