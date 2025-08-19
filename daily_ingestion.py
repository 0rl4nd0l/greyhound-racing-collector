#!/usr/bin/env python3
"""
Daily Ingestion Script
======================

This script updates upcoming race data daily from thedogs.com.au.
"""

import os

from form_guide_csv_scraper import FormGuideCsvScraper


class DailyIngestion:
    def __init__(self):
        self.scraper = FormGuideCsvScraper()

    def update_upcoming_races(self):
        """Fetch and store upcoming races."""
        print("🌟 Running daily race ingestion...")
        dates = self.scraper.get_race_dates(days_back=1)[
            :-1
        ]  # Check only yesterday for updates

        for date in dates:
            print(f"🔍 Checking races for {date}...")
            race_urls = self.scraper.find_race_urls(date)

            for race_url in race_urls:
                print(f"📂 Downloading {race_url}...")
                success = self.scraper.download_csv_from_race_page(race_url)

                if success:
                    print(f"✅ Downloaded {race_url} successfully.")
                else:
                    print(f"❌ Failed to download {race_url}.")


if __name__ == "__main__":
    ingestion = DailyIngestion()
    ingestion.update_upcoming_races()
