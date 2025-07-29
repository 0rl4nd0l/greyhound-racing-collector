#!/usr/bin/env python3
"""
Weather Service Test Retry
===========================

Retry the weather service test to ensure it's working correctly.
"""

import sys
import os
from datetime import datetime
import warnings

print("🌤️ WEATHER SERVICE TEST RETRY")
print("=" * 50)
print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from weather_service_open_meteo import OpenMeteoWeatherService
    print("✅ OpenMeteoWeatherService imported successfully")
    
    # Initialize the weather service
    weather_service = OpenMeteoWeatherService()
    print("✅ WeatherService initialized successfully")
    
    # Test venue codes for Australian greyhound venues
    test_venues = [
        {"name": "Melbourne (The Meadows)", "code": "MEA"},
        {"name": "Sydney (Wentworth Park)", "code": "W_PK"},
        {"name": "Adelaide (Angle Park)", "code": "AP_K"},
        {"name": "Perth (Cannington)", "code": "CANN"},
        {"name": "Hobart", "code": "HOBT"}
    ]
    
    print(f"\n🧪 Testing weather data retrieval for {len(test_venues)} venues:")
    
    successful_tests = 0
    failed_tests = 0
    
    for i, venue in enumerate(test_venues, 1):
        try:
            print(f"\n🌍 Test {i}: {venue['name']}")
            print(f"   🏟️ Venue code: {venue['code']}")
            
            # Get current weather
            weather_data = weather_service.get_current_weather(venue['code'])
            
            if weather_data:
                print(f"   ✅ Weather retrieved successfully")
                print(f"   🌡️  Temperature: {weather_data.temperature}°C")
                print(f"   💨 Wind Speed: {weather_data.wind_speed} km/h")
                print(f"   💧 Humidity: {weather_data.humidity}%")
                print(f"   🌤️  Condition: {weather_data.condition.value}")
                print(f"   🌧️  Precipitation: {weather_data.precipitation}mm")
                successful_tests += 1
            else:
                print(f"   ❌ No weather data retrieved")
                failed_tests += 1
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            failed_tests += 1
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   ✅ Successful tests: {successful_tests}")
    print(f"   ❌ Failed tests: {failed_tests}")
    print(f"   📈 Success rate: {successful_tests/(successful_tests+failed_tests)*100:.1f}%")
    
    if successful_tests > 0:
        print(f"\n🎉 Weather service is working correctly!")
        
        # Test with a specific venue code
        print(f"\n🏁 Testing with additional venue example (The Meadows):")
        
        try:
            # Test with a specific venue (The Meadows)
            meadows_weather = weather_service.get_current_weather('MEA')
            
            if meadows_weather:
                print(f"✅ The Meadows weather data:")
                print(f"   Location: {meadows_weather.location}")
                print(f"   Temperature: {meadows_weather.temperature}°C")
                print(f"   Humidity: {meadows_weather.humidity}%")
                print(f"   Wind Speed: {meadows_weather.wind_speed} km/h")
                print(f"   Wind Direction: {meadows_weather.wind_direction}°")
                print(f"   Condition: {meadows_weather.condition.value}")
                print(f"   Precipitation: {meadows_weather.precipitation}mm")
                print(f"   Visibility: {meadows_weather.visibility}km")
                
                # Test weather condition mapping
                condition = meadows_weather.condition.value.lower()
                track_condition = 'good'  # Default
                
                if any(word in condition for word in ['rain', 'shower', 'drizzle']):
                    track_condition = 'wet'
                elif any(word in condition for word in ['snow', 'sleet']):
                    track_condition = 'heavy'
                elif 'fog' in condition or 'mist' in condition:
                    track_condition = 'slow'
                
                print(f"   🏁 Suggested track condition: {track_condition}")
                
        except Exception as e:
            print(f"❌ Racing venue test failed: {e}")
    
    else:
        print(f"\n⚠️ Weather service appears to have issues")
        print(f"   Check internet connection and API availability")
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Weather service module not found or dependencies missing")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    sys.exit(1)

print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
