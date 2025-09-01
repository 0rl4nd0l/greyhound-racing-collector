#!/usr/bin/env python3
"""
Direct Racing Page Scraper
==========================

Scrapes the main racing page to find all available races.
"""

import json
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from utils.http_client import get_shared_session


def scrape_all_races():
    """Scrape all races from the main racing page"""

    session = get_shared_session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )

    print("🔍 Scraping main racing page...")

    response = None
    try:
        response = session.get("https://www.thedogs.com.au/racing", timeout=15)

        if response.status_code != 200:
            print(f"❌ Failed to fetch page (Status: {response.status_code})")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        print(f"✅ Successfully fetched main racing page")
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    # Venue mapping
    venue_map = {
        "angle-park": "AP_K",
        "sandown": "SAN",
        "warrnambool": "WAR",
        "bendigo": "BEN",
        "geelong": "GEE",
        "ballarat": "BAL",
        "horsham": "HOR",
        "traralgon": "TRA",
        "dapto": "DAPT",
        "wentworth-park": "WPK",
        "albion-park": "ALBION",
        "cannington": "CANN",
        "the-meadows": "MEA",
        "healesville": "HEA",
        "sale": "SAL",
        "richmond": "RICH",
        "murray-bridge": "MURR",
        "gawler": "GAWL",
        "mount-gambier": "MOUNT",
        "northam": "NOR",
        "mandurah": "MAND",
        "gosford": "GOSF",
        "hobart": "HOBT",
        "the-gardens": "GRDN",
        "darwin": "DARW",
        "broken-hill": "BROKEN-HILL",
        "grafton": "GRAFTON",
        "dubbo": "DUBBO",
    }

    venue_name_map = {
        "AP_K": "Angle Park",
        "SAN": "Sandown",
        "WAR": "Warrnambool",
        "BEN": "Bendigo",
        "GEE": "Geelong",
        "BAL": "Ballarat",
        "HOR": "Horsham",
        "TRA": "Traralgon",
        "DAPT": "Dapto",
        "WPK": "Wentworth Park",
        "ALBION": "Albion Park",
        "CANN": "Cannington",
        "MEA": "The Meadows",
        "HEA": "Healesville",
        "SAL": "Sale",
        "RICH": "Richmond",
        "MURR": "Murray Bridge",
        "GAWL": "Gawler",
        "MOUNT": "Mount Gambier",
        "NOR": "Northam",
        "MAND": "Mandurah",
        "GOSF": "Gosford",
        "HOBT": "Hobart",
        "GRDN": "The Gardens",
        "DARW": "Darwin",
        "BROKEN-HILL": "Broken Hill",
        "GRAFTON": "Grafton",
        "DUBBO": "Dubbo",
    }

    # Find all racing links
    race_links = []
    all_links = soup.find_all("a", href=True)

    for link in all_links:
        href = link.get("href")
        if href and "/racing/" in href:
            race_links.append((link, href))

    print(f"🔍 Found {len(race_links)} potential race links")

    # Parse race information
    races = []
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    valid_dates = [today.strftime("%Y-%m-%d"), tomorrow.strftime("%Y-%m-%d")]

    processed_races = set()  # To avoid duplicates

    for link, href in race_links:
        try:
            # Parse URL parts
            url_parts = href.strip("/").split("/")
            if "racing" not in url_parts:
                continue

            racing_index = url_parts.index("racing")

            if len(url_parts) <= racing_index + 3:
                continue

            venue_slug = url_parts[racing_index + 1]
            date_part = url_parts[racing_index + 2]
            race_number = url_parts[racing_index + 3]

            # Validate race number is numeric
            if not race_number.isdigit():
                continue

            # Only include today and tomorrow's races
            if date_part not in valid_dates:
                continue

            # Create unique race key
            race_key = f"{venue_slug}_{date_part}_{race_number}"
            if race_key in processed_races:
                continue
            processed_races.add(race_key)

            # Map venue slug to venue code
            venue_code = venue_map.get(venue_slug, venue_slug.upper())
            venue_name = venue_name_map.get(
                venue_code, venue_slug.replace("-", " ").title()
            )

            # Generate more realistic estimated race time based on race number
            # Greyhound races typically start around 6 PM and run every 20-25 minutes
            base_hour = 18  # 6 PM start (more realistic for greyhound racing)
            total_minutes = (int(race_number) - 1) * 22  # 22 minutes between races
            hour = base_hour + (total_minutes // 60)
            minute = total_minutes % 60

            # Ensure we don't go past midnight
            if hour >= 24:
                hour = 23
                minute = 59

            # Convert to 12-hour format
            if hour > 12:
                race_time = f"{hour - 12}:{minute:02d} PM"
            elif hour == 12:
                race_time = f"12:{minute:02d} PM"
            elif hour == 0:
                race_time = f"12:{minute:02d} AM"
            else:
                race_time = f"{hour}:{minute:02d} AM"

            # Get link text for additional info
            link_text = link.get_text().strip()

            # Try to extract race name from URL
            race_name = None
            if len(url_parts) > racing_index + 4:
                race_name_part = url_parts[racing_index + 4]
                race_name = race_name_part.replace("-", " ").title()

            race_url = (
                href if href.startswith("http") else f"https://www.thedogs.com.au{href}"
            )

            race_info = {
                "date": date_part,
                "venue": venue_code,
                "venue_name": venue_name,
                "race_number": int(race_number),
                "race_time": race_time,
                "race_name": race_name,
                "url": race_url,
                "title": f"Race {race_number} - {venue_name} - {date_part}",
                "description": f"🕐 {race_time}",
                "source": "main_racing_page",
            }

            races.append(race_info)

        except Exception as e:
            continue

    # Sort races by date, then by venue, then by race number
    races.sort(key=lambda x: (x["date"], x["venue"], x["race_number"]))

    # Group by venue for summary
    venues = {}
    for race in races:
        venue = race["venue"]
        if venue not in venues:
            venues[venue] = []
        venues[venue].append(race)

    print(f"\n✅ Found {len(races)} races from {len(venues)} venues:")
    for venue, venue_races in sorted(venues.items()):
        venue_name = venue_races[0]["venue_name"]
        dates = set(race["date"] for race in venue_races)
        print(
            f"  {venue}: {venue_name} - {len(venue_races)} races across {len(dates)} days"
        )

    return races


def get_today_races():
    """Get races for today only"""
    all_races = scrape_all_races()
    today = datetime.now().date().strftime("%Y-%m-%d")

    today_races = [race for race in all_races if race["date"] == today]

    print(f"\n🏁 Today ({today}) races:")
    venues = {}
    for race in today_races:
        venue = race["venue"]
        if venue not in venues:
            venues[venue] = []
        venues[venue].append(race)

    for venue, venue_races in sorted(venues.items()):
        venue_name = venue_races[0]["venue_name"]
        print(f"  {venue}: {venue_name} - {len(venue_races)} races")
        for race in sorted(venue_races, key=lambda x: x["race_number"])[:3]:
            print(f'    Race {race["race_number"]} at {race["race_time"]}')

    return today_races


if __name__ == "__main__":
    races = get_today_races()
    print(f"\nTotal today races found: {len(races)}")
