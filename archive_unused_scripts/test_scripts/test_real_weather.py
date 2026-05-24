#!/usr/bin/env python3
"""
Real Weather API Test
====================

Test script to fetch actual weather data from BOM API for Melbourne (Sandown venue).
This will show current weather conditions for greyhound racing.

Author: AI Assistant  
Date: July 25, 2025
"""

import json
from datetime import datetime

import requests

from weather_api_service import (
    BOMWeatherService,
    VenueLocation,
    WeatherCondition,
    WeatherData,
)


def test_real_bom_api():
    """Test the real BOM API with Melbourne data"""
    print("🌤️  Testing Real BOM Weather API")
    print("=" * 50)

    # Melbourne (Olympic Park) station ID for Sandown venue
    melbourne_station_id = "086071"

    # BOM current weather API URL
    url = f"http://www.bom.gov.au/fwo/IDN60901/IDN60901.{melbourne_station_id}.json"

    try:
        print(f"\n1. Fetching data from BOM API...")
        print(f"   URL: {url}")

        # Make the API request
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        print(f"   ✓ API Response: {response.status_code}")

        # Parse the JSON response
        data = response.json()

        print(f"\n2. Parsing BOM data structure...")

        # Check data structure
        if "observations" in data and "data" in data["observations"]:
            observations = data["observations"]["data"]
            print(f"   ✓ Found {len(observations)} weather observations")

            if observations:
                latest = observations[0]  # Most recent observation

                print(f"\n3. Latest Weather Data for Melbourne:")
                print(f"   📍 Station: {data['observations']['header'][0]['name']}")
                print(f"   🕐 Time: {latest.get('aifstime_utc', 'Unknown')}")
                print(f"   🌡️  Temperature: {latest.get('air_temp', 'N/A')}°C")
                print(
                    f"   💨 Wind: {latest.get('wind_spd_kmh', 'N/A')} km/h {latest.get('wind_dir', 'N/A')}"
                )
                print(f"   💧 Humidity: {latest.get('rel_hum', 'N/A')}%")
                print(f"   🌧️  Rain (since 9am): {latest.get('rain_trace', 'N/A')}mm")
                print(f"   📊 Pressure: {latest.get('press_qnh', 'N/A')} hPa")
                print(f"   👁️  Visibility: {latest.get('vis_km', 'N/A')} km")
                print(f"   ☁️  Weather: {latest.get('weather', 'N/A')}")

                return latest
            else:
                print("   ✗ No observation data available")
                return None
        else:
            print("   ✗ Unexpected data structure from BOM API")
            print(f"   Available keys: {list(data.keys())}")
            return None

    except requests.RequestException as e:
        print(f"   ✗ API Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"   ✗ JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return None


def test_weather_service_integration():
    """Test the full weather service with real API"""
    print(f"\n4. Testing Weather Service Integration...")

    try:
        # Initialize the weather service
        weather_service = BOMWeatherService()

        print(f"   ✓ Weather service initialized")

        # Test getting current weather for Melbourne (Sandown)
        print(f"   🔄 Fetching weather for Sandown (Melbourne)...")

        weather_data = weather_service.get_current_weather("SANDOWN")

        if weather_data:
            print(f"   ✓ Weather data retrieved successfully!")
            print(f"   📍 Location: {weather_data.location}")
            print(f"   🌡️  Temperature: {weather_data.temperature:.1f}°C")
            print(f"   💧 Humidity: {weather_data.humidity:.1f}%")
            print(
                f"   💨 Wind: {weather_data.wind_speed:.1f} km/h {weather_data.wind_direction}"
            )
            print(f"   🌧️  Precipitation: {weather_data.precipitation:.1f}mm")
            print(f"   ☁️  Condition: {weather_data.condition.value}")
            print(f"   📊 Confidence: {weather_data.confidence:.2f}")

            # Calculate weather adjustment factor
            adjustment = weather_service.calculate_weather_adjustment_factor(
                weather_data, "SAN"
            )
            print(f"\n5. Weather Impact Analysis:")
            print(f"   🎯 Adjustment Factor: {adjustment:.3f}")

            if adjustment < 0.95:
                impact = "Challenging conditions - expect slower times"
            elif adjustment > 1.01:
                impact = "Favorable conditions - expect faster times"
            else:
                impact = "Neutral conditions - minimal impact"

            print(f"   📈 Impact: {impact}")

            # Racing recommendations
            print(f"\n6. Racing Implications:")
            if weather_data.temperature > 28:
                print(f"   ⚠️  Hot weather - dogs may tire more quickly")
            elif weather_data.temperature < 12:
                print(f"   ❄️  Cold weather - dogs may be slower to warm up")
            else:
                print(f"   ✅ Temperature suitable for racing")

            if weather_data.precipitation > 2:
                print(f"   🌧️  Wet track conditions - favor inside runners")
            elif weather_data.precipitation > 0:
                print(f"   💧 Light moisture - minimal track impact")
            else:
                print(f"   ☀️  Dry track conditions")

            if weather_data.wind_speed > 20:
                print(f"   💨 Strong winds - significant impact on times")
            elif weather_data.wind_speed > 15:
                print(f"   🌪️  Moderate winds - some impact expected")
            else:
                print(f"   🍃 Light winds - minimal impact")

            return weather_data
        else:
            print(f"   ✗ Failed to retrieve weather data")
            return None

    except Exception as e:
        print(f"   ✗ Weather service error: {e}")
        return None


def main():
    """Main test function"""
    # Test raw BOM API first
    raw_data = test_real_bom_api()

    # Test integrated weather service
    weather_data = test_weather_service_integration()

    print(f"\n" + "=" * 50)
    print(f"🎯 Real Weather API Test Complete!")

    if weather_data:
        print(f"\n✅ Success! Weather integration is working with real BOM data.")
        print(f"   The system can now provide real-time weather data for")
        print(f"   greyhound racing predictions at all configured venues.")
    else:
        print(f"\n❌ Issues detected. Check API connectivity and data parsing.")


if __name__ == "__main__":
    main()
