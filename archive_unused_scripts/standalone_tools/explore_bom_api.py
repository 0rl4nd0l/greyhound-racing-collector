#!/usr/bin/env python3
"""
BOM API Explorer
===============

Simple script to explore available BOM weather API endpoints and find working URLs.
"""

import json

import requests


def test_bom_endpoints():
    """Test various BOM API endpoint formats"""
    print("🔍 Exploring BOM Weather API Endpoints")
    print("=" * 45)

    # Melbourne station ID
    melbourne_station_id = "086071"

    # Different API endpoint formats to try
    endpoints = [
        # Current observations format
        f"http://www.bom.gov.au/fwo/IDN60901/IDN60901.{melbourne_station_id}.json",
        f"https://www.bom.gov.au/fwo/IDN60901/IDN60901.{melbourne_station_id}.json",
        # Alternative formats
        f"http://www.bom.gov.au/fwo/IDV60901/IDV60901.{melbourne_station_id}.json",
        f"https://www.bom.gov.au/fwo/IDV60901/IDV60901.{melbourne_station_id}.json",
        # General weather data
        "http://www.bom.gov.au/fwo/IDV60901/IDV60901.95936.json",  # Melbourne (Olympic Park)
        "https://www.bom.gov.au/fwo/IDV60901/IDV60901.95936.json",
        # Try some other known working endpoints
        "http://www.bom.gov.au/fwo/IDN60901/IDN60901.94675.json",  # Sydney Observatory Hill
        "https://www.bom.gov.au/fwo/IDN60901/IDN60901.94675.json",
    ]

    working_endpoints = []

    for i, url in enumerate(endpoints, 1):
        print(f"\n{i}. Testing: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                print("   ✅ SUCCESS! This endpoint is working")

                # Try to parse JSON
                try:
                    data = response.json()
                    print(f"   📊 Data structure: {list(data.keys())}")

                    if "observations" in data:
                        obs_data = data["observations"]
                        if "header" in obs_data and obs_data["header"]:
                            station_name = obs_data["header"][0].get("name", "Unknown")
                            print(f"   📍 Station: {station_name}")

                        if "data" in obs_data and obs_data["data"]:
                            latest = obs_data["data"][0]
                            temp = latest.get("air_temp", "N/A")
                            time = latest.get("aifstime_utc", "N/A")
                            print(f"   🌡️  Latest: {temp}°C at {time}")

                    working_endpoints.append((url, data))

                except json.JSONDecodeError:
                    print("   ⚠️  Response is not JSON")

            elif response.status_code == 403:
                print("   🚫 Forbidden - May need authentication")
            elif response.status_code == 404:
                print("   ❌ Not Found - Endpoint doesn't exist")
            else:
                print(f"   ❓ Other error: {response.status_code}")

        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return working_endpoints


def try_alternative_weather_apis():
    """Try some alternative weather data sources"""
    print(f"\n" + "=" * 45)
    print("🌐 Trying Alternative Weather Sources")
    print("=" * 45)

    # Try some public weather APIs that don't require authentication
    alternatives = [
        # Open-Meteo (free, no API key)
        "https://api.open-meteo.com/v1/forecast?latitude=-37.8136&longitude=144.9631&current_weather=true",
        # WeatherAPI (free tier, but needs key - just test endpoint)
        "https://api.weatherapi.com/v1/current.json?q=Melbourne",
        # OpenWeatherMap (free tier, but needs key - just test endpoint)
        "https://api.openweathermap.org/data/2.5/weather?q=Melbourne",
    ]

    for i, url in enumerate(alternatives, 1):
        print(f"\n{i}. Testing: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                print("   ✅ SUCCESS!")
                try:
                    data = response.json()

                    # Parse Open-Meteo format
                    if "current_weather" in data:
                        current = data["current_weather"]
                        temp = current.get("temperature", "N/A")
                        wind = current.get("windspeed", "N/A")
                        print(f"   🌡️  Temperature: {temp}°C")
                        print(f"   💨 Wind: {wind} km/h")
                    else:
                        print(f"   📊 Data keys: {list(data.keys())}")

                except json.JSONDecodeError:
                    print("   ⚠️  Response is not JSON")

            elif response.status_code == 401:
                print("   🔑 Needs API key (expected)")
            else:
                print(f"   ❓ Status: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main function"""
    working_endpoints = test_bom_endpoints()

    print(f"\n" + "=" * 45)
    print(f"📋 Summary")
    print("=" * 45)

    if working_endpoints:
        print(f"✅ Found {len(working_endpoints)} working BOM endpoints:")
        for url, data in working_endpoints:
            print(f"   • {url}")
    else:
        print("❌ No working BOM endpoints found")
        print("   This might be due to:")
        print("   • Changed API endpoints")
        print("   • Authentication requirements")
        print("   • Network restrictions")

    # Try alternatives
    try_alternative_weather_apis()

    print(f"\n💡 Recommendations:")
    if working_endpoints:
        print("   • Use the working BOM endpoints found above")
        print("   • Update the weather service with correct URLs")
    else:
        print("   • Consider using Open-Meteo API (free, no key required)")
        print("   • Register for WeatherAPI.com (free tier available)")
        print("   • Check BOM website for updated API documentation")


if __name__ == "__main__":
    main()
