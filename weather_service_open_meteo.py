#!/usr/bin/env python3
"""
Open-Meteo Weather API Integration Service
==========================================

Weather service for greyhound racing system using Open-Meteo API (free, no API key required).
Provides real-time and forecast weather data for race venues to enhance predictions.

Key Features:
- Real-time weather data collection
- 7-day weather forecasts
- Historical weather data
- Venue-specific weather mapping
- Weather impact modeling
- No API key required

Author: AI Assistant
Date: July 25, 2025
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WeatherCondition(Enum):
    """Standard weather conditions based on WMO weather codes"""

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    STORM = "storm"
    FOG = "fog"
    MIST = "mist"


@dataclass
class WeatherData:
    """Weather data structure"""

    location: str
    timestamp: datetime
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: int  # degrees
    pressure: float
    condition: WeatherCondition
    precipitation: float
    visibility: float
    confidence: float = 0.95


@dataclass
class VenueLocation:
    """Venue location mapping with coordinates"""

    venue_code: str
    venue_name: str
    city: str
    state: str
    latitude: float
    longitude: float
    timezone: str


class OpenMeteoWeatherService:
    """Open-Meteo Weather Service (Free, No API Key Required)"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.cache_duration = 300  # 5 minutes cache
        self.request_delay = 1.0  # Rate limiting (be respectful)
        self.last_request_time = 0
        self.cache = {}

        # Base API URL
        self.base_url = "https://api.open-meteo.com/v1"

        # Initialize venue mappings
        self.venue_locations = self._initialize_venue_locations()

        # Initialize database tables
        self._initialize_weather_tables()

    def _initialize_venue_locations(self) -> Dict[str, VenueLocation]:
        """Initialize venue to coordinate mappings"""
        venues = {
            # Existing venues with updated coordinates
            "AP_K": VenueLocation(
                venue_code="AP_K",
                venue_name="Angle Park",
                city="Adelaide",
                state="SA",
                latitude=-34.8468,
                longitude=138.5390,
                timezone="Australia/Adelaide",
            ),
            "SAN": VenueLocation(
                venue_code="SAN",
                venue_name="Sandown",
                city="Melbourne",
                state="VIC",
                latitude=-37.95639,
                longitude=145.16028,
                timezone="Australia/Melbourne",
            ),
            "W_PK": VenueLocation(
                venue_code="W_PK",
                venue_name="Wentworth Park",
                city="Sydney",
                state="NSW",
                latitude=-33.8721,
                longitude=151.1949,
                timezone="Australia/Sydney",
            ),
            "MEA": VenueLocation(
                venue_code="MEA",
                venue_name="The Meadows",
                city="Melbourne",
                state="VIC",
                latitude=-37.68222,
                longitude=144.95278,
                timezone="Australia/Melbourne",
            ),
            "DAPT": VenueLocation(
                venue_code="DAPT",
                venue_name="Dapto",
                city="Wollongong",
                state="NSW",
                latitude=-34.4989,
                longitude=150.7947,
                timezone="Australia/Sydney",
            ),
            "HOBT": VenueLocation(
                venue_code="HOBT",
                venue_name="Hobart",
                city="Hobart",
                state="TAS",
                latitude=-42.8826,
                longitude=147.3257,
                timezone="Australia/Hobart",
            ),
            "BAL": VenueLocation(
                venue_code="BAL",
                venue_name="Ballarat",
                city="Ballarat",
                state="VIC",
                latitude=-37.5622,
                longitude=143.8503,
                timezone="Australia/Melbourne",
            ),
            "BEN": VenueLocation(
                venue_code="BEN",
                venue_name="Bendigo",
                city="Bendigo",
                state="VIC",
                latitude=-36.7581,
                longitude=144.2789,
                timezone="Australia/Melbourne",
            ),
            # Additional venues with coordinates from your data
            "WAR": VenueLocation(
                venue_code="WAR",
                venue_name="Warrnambool",
                city="Warrnambool",
                state="VIC",
                latitude=-38.37788,
                longitude=142.46684,
                timezone="Australia/Melbourne",
            ),
            "MOUNT": VenueLocation(
                venue_code="MOUNT",
                venue_name="Mount Gambier",
                city="Mount Gambier",
                state="SA",
                latitude=-37.84622,
                longitude=140.80213,
                timezone="Australia/Adelaide",
            ),
            "GOSF": VenueLocation(
                venue_code="GOSF",
                venue_name="Gosford",
                city="Gosford",
                state="NSW",
                latitude=-33.41417,
                longitude=151.34111,
                timezone="Australia/Sydney",
            ),
            "APTH": VenueLocation(
                venue_code="APTH",
                venue_name="Albion Park",
                city="Brisbane",
                state="QLD",
                latitude=-27.43965,
                longitude=153.04612,
                timezone="Australia/Brisbane",
            ),
            "APWE": VenueLocation(
                venue_code="APWE",
                venue_name="Albion Park",
                city="Brisbane",
                state="QLD",
                latitude=-27.43965,
                longitude=153.04612,
                timezone="Australia/Brisbane",
            ),
            "TEMA": VenueLocation(
                venue_code="TEMA",
                venue_name="Temora",
                city="Temora",
                state="NSW",
                latitude=-34.45255,
                longitude=147.54527,
                timezone="Australia/Sydney",
            ),
            # Approximate coordinates for remaining venues based on city locations
            "HEA": VenueLocation(
                venue_code="HEA",
                venue_name="Healesville",
                city="Healesville",
                state="VIC",
                latitude=-37.6500,
                longitude=145.5167,
                timezone="Australia/Melbourne",
            ),
            "SAL": VenueLocation(
                venue_code="SAL",
                venue_name="Sale",
                city="Sale",
                state="VIC",
                latitude=-38.1000,
                longitude=147.0667,
                timezone="Australia/Melbourne",
            ),
            "GEE": VenueLocation(
                venue_code="GEE",
                venue_name="Geelong",
                city="Geelong",
                state="VIC",
                latitude=-38.1499,
                longitude=144.3617,
                timezone="Australia/Melbourne",
            ),
            "CASO": VenueLocation(
                venue_code="CASO",
                venue_name="Casino",
                city="Casino",
                state="NSW",
                latitude=-28.8667,
                longitude=153.0500,
                timezone="Australia/Sydney",
            ),
            "GAWL": VenueLocation(
                venue_code="GAWL",
                venue_name="Gawler",
                city="Gawler",
                state="SA",
                latitude=-34.6167,
                longitude=138.7333,
                timezone="Australia/Adelaide",
            ),
            "HOR": VenueLocation(
                venue_code="HOR",
                venue_name="Horsham",
                city="Horsham",
                state="VIC",
                latitude=-36.7167,
                longitude=142.2000,
                timezone="Australia/Melbourne",
            ),
            "MURR": VenueLocation(
                venue_code="MURR",
                venue_name="Murray Bridge",
                city="Murray Bridge",
                state="SA",
                latitude=-35.1167,
                longitude=139.2667,
                timezone="Australia/Adelaide",
            ),
            "RICH": VenueLocation(
                venue_code="RICH",
                venue_name="Richmond",
                city="Richmond",
                state="NSW",
                latitude=-33.6000,
                longitude=150.7500,
                timezone="Australia/Sydney",
            ),
            "TRA": VenueLocation(
                venue_code="TRA",
                venue_name="Traralgon",
                city="Traralgon",
                state="VIC",
                latitude=-38.1833,
                longitude=146.5333,
                timezone="Australia/Melbourne",
            ),
            "MAND": VenueLocation(
                venue_code="MAND",
                venue_name="Mandurah",
                city="Mandurah",
                state="WA",
                latitude=-32.5269,
                longitude=115.7219,
                timezone="Australia/Perth",
            ),
            "SHEP": VenueLocation(
                venue_code="SHEP",
                venue_name="Shepparton",
                city="Shepparton",
                state="VIC",
                latitude=-36.3833,
                longitude=145.4000,
                timezone="Australia/Melbourne",
            ),
            "WARR": VenueLocation(
                venue_code="WARR",
                venue_name="Warragul",
                city="Warragul",
                state="VIC",
                latitude=-38.1667,
                longitude=145.9333,
                timezone="Australia/Melbourne",
            ),
            "NOR": VenueLocation(
                venue_code="NOR",
                venue_name="Northam",
                city="Northam",
                state="WA",
                latitude=-31.6500,
                longitude=116.6667,
                timezone="Australia/Perth",
            ),
            "GUNN": VenueLocation(
                venue_code="GUNN",
                venue_name="Gunnedah",
                city="Gunnedah",
                state="NSW",
                latitude=-30.9833,
                longitude=150.2500,
                timezone="Australia/Sydney",
            ),
            "CAPA": VenueLocation(
                venue_code="CAPA",
                venue_name="Capalaba",
                city="Brisbane",
                state="QLD",
                latitude=-27.5333,
                longitude=153.2000,
                timezone="Australia/Brisbane",
            ),
            "ROCK": VenueLocation(
                venue_code="ROCK",
                venue_name="Rockhampton",
                city="Rockhampton",
                state="QLD",
                latitude=-23.3833,
                longitude=150.5167,
                timezone="Australia/Brisbane",
            ),
            "DARW": VenueLocation(
                venue_code="DARW",
                venue_name="Darwin",
                city="Darwin",
                state="NT",
                latitude=-12.4634,
                longitude=130.8456,
                timezone="Australia/Darwin",
            ),
            "GRDN": VenueLocation(
                venue_code="GRDN",
                venue_name="The Gardens",
                city="Darwin",
                state="NT",
                latitude=-12.4634,
                longitude=130.8456,
                timezone="Australia/Darwin",
            ),
            "CANN": VenueLocation(
                venue_code="CANN",
                venue_name="Cannington",
                city="Perth",
                state="WA",
                latitude=-32.0167,
                longitude=115.9333,
                timezone="Australia/Perth",
            ),
            "DUB": VenueLocation(
                venue_code="DUB",
                venue_name="Dubbo",
                city="Dubbo",
                state="NSW",
                latitude=-32.2433,
                longitude=148.6019,
                timezone="Australia/Sydney",
            ),
            "Q1L": VenueLocation(
                venue_code="Q1L",
                venue_name="Ladbrokes Q1 Lakeside",
                city="Brisbane",
                state="QLD",
                latitude=-27.4705,
                longitude=153.0260,
                timezone="Australia/Brisbane",
            ),
        }
        return venues

    def _initialize_weather_tables(self):
        """Initialize weather-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update weather data table to handle Open-Meteo data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_data_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    race_time DATETIME,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction INTEGER,
                    pressure REAL,
                    condition TEXT,
                    precipitation REAL,
                    visibility REAL,
                    weather_code INTEGER,
                    data_source TEXT DEFAULT 'OPEN_METEO',
                    collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.95,
                    UNIQUE(venue_code, race_date, race_time)
                )
            """
            )

            conn.commit()
            conn.close()
            logger.info("Open-Meteo weather database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing weather tables: {e}")
            raise

    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _weather_code_to_condition(self, weather_code: int) -> WeatherCondition:
        """Convert WMO weather code to our weather condition enum"""
        try:
            # WMO Weather interpretation codes
            if weather_code == 0:
                return WeatherCondition.CLEAR
            elif weather_code in [1, 2]:
                return WeatherCondition.PARTLY_CLOUDY
            elif weather_code == 3:
                return WeatherCondition.CLOUDY
            elif weather_code in [45, 48]:
                return WeatherCondition.FOG
            elif weather_code in [51, 53, 55]:
                return WeatherCondition.LIGHT_RAIN
            elif weather_code in [61, 63]:
                return WeatherCondition.RAIN
            elif weather_code in [65, 80, 81]:
                return WeatherCondition.HEAVY_RAIN
            elif weather_code in [95, 96, 99]:
                return WeatherCondition.STORM
            else:
                return WeatherCondition.CLOUDY  # Default fallback
        except:
            return WeatherCondition.CLOUDY

    def get_current_weather(self, venue_code: str) -> Optional[WeatherData]:
        """Get current weather for a venue using Open-Meteo API"""
        try:
            if venue_code not in self.venue_locations:
                logger.error(f"Unknown venue code: {venue_code}")
                return None

            venue = self.venue_locations[venue_code]

            # Check cache first
            cache_key = f"{venue_code}_current"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data

            # Rate limit API requests
            self._rate_limit()

            # Open-Meteo current weather API
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": venue.latitude,
                "longitude": venue.longitude,
                "current_weather": "true",
                "hourly": "temperature_2m,relativehumidity_2m,precipitation,pressure_msl,visibility",
                "timezone": "auto",
                "forecast_days": 1,
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse current weather
            if "current_weather" in data:
                current = data["current_weather"]
                hourly = data.get("hourly", {})

                # Get current hour data for additional details
                current_time = datetime.fromisoformat(current["time"])

                # Find the current hour in hourly data
                humidity = 50.0  # Default if not available
                precipitation = 0.0
                pressure = 1013.25
                visibility = 10.0

                if hourly and "time" in hourly:
                    current_hour_str = current_time.strftime("%Y-%m-%dT%H:00")
                    try:
                        hour_index = hourly["time"].index(current_hour_str)
                        if (
                            "relativehumidity_2m" in hourly
                            and len(hourly["relativehumidity_2m"]) > hour_index
                        ):
                            humidity = float(
                                hourly["relativehumidity_2m"][hour_index] or 50.0
                            )
                        if (
                            "precipitation" in hourly
                            and len(hourly["precipitation"]) > hour_index
                        ):
                            precipitation = float(
                                hourly["precipitation"][hour_index] or 0.0
                            )
                        if (
                            "pressure_msl" in hourly
                            and len(hourly["pressure_msl"]) > hour_index
                        ):
                            pressure = float(
                                hourly["pressure_msl"][hour_index] or 1013.25
                            )
                        if (
                            "visibility" in hourly
                            and len(hourly["visibility"]) > hour_index
                        ):
                            visibility = (
                                float(hourly["visibility"][hour_index] or 10000.0)
                                / 1000.0
                            )  # Convert m to km
                    except (ValueError, IndexError):
                        pass  # Use defaults

                weather_data = WeatherData(
                    location=venue.venue_name,
                    timestamp=current_time,
                    temperature=float(current["temperature"]),
                    humidity=humidity,
                    wind_speed=float(current["windspeed"]),
                    wind_direction=int(current["winddirection"]),
                    pressure=pressure,
                    condition=self._weather_code_to_condition(
                        int(current["weathercode"])
                    ),
                    precipitation=precipitation,
                    visibility=visibility,
                    confidence=0.95,
                )

                # Cache the result
                self.cache[cache_key] = (weather_data, time.time())

                # Store in database
                self._store_weather_data(weather_data)

                return weather_data

            return None

        except requests.RequestException as e:
            logger.error(f"Error fetching current weather for {venue_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting current weather: {e}")
            return None

    def get_forecast_weather(
        self, venue_code: str, target_date: str
    ) -> Optional[WeatherData]:
        """Get weather forecast for a specific date and venue"""
        try:
            if venue_code not in self.venue_locations:
                logger.error(f"Unknown venue code: {venue_code}")
                return None

            venue = self.venue_locations[venue_code]

            # Check cache first
            cache_key = f"{venue_code}_{target_date}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data

            # Rate limit API requests
            self._rate_limit()

            # Open-Meteo forecast API
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": venue.latitude,
                "longitude": venue.longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,weathercode",
                "hourly": "relativehumidity_2m,pressure_msl",
                "timezone": "auto",
                "start_date": target_date,
                "end_date": target_date,
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse daily forecast
            if "daily" in data and data["daily"]["time"]:
                daily = data["daily"]

                # Average temperature from min/max
                temp_min = float(daily["temperature_2m_min"][0] or 15.0)
                temp_max = float(daily["temperature_2m_max"][0] or 25.0)
                temperature = (temp_min + temp_max) / 2

                # Get hourly humidity for midday (approximate)
                humidity = 60.0  # Default
                pressure = 1013.25

                if "hourly" in data and data["hourly"]["time"]:
                    hourly = data["hourly"]
                    # Use midday values (index around 12)
                    midday_index = min(12, len(hourly["time"]) - 1)
                    if (
                        "relativehumidity_2m" in hourly
                        and len(hourly["relativehumidity_2m"]) > midday_index
                    ):
                        humidity = float(
                            hourly["relativehumidity_2m"][midday_index] or 60.0
                        )
                    if (
                        "pressure_msl" in hourly
                        and len(hourly["pressure_msl"]) > midday_index
                    ):
                        pressure = float(
                            hourly["pressure_msl"][midday_index] or 1013.25
                        )

                weather_data = WeatherData(
                    location=venue.venue_name,
                    timestamp=datetime.strptime(target_date, "%Y-%m-%d"),
                    temperature=temperature,
                    humidity=humidity,
                    wind_speed=float(daily["windspeed_10m_max"][0] or 10.0),
                    wind_direction=int(daily["winddirection_10m_dominant"][0] or 180),
                    pressure=pressure,
                    condition=self._weather_code_to_condition(
                        int(daily["weathercode"][0] or 0)
                    ),
                    precipitation=float(daily["precipitation_sum"][0] or 0.0),
                    visibility=10.0,  # Default visibility for forecast
                    confidence=0.85,  # Lower confidence for forecast
                )

                # Cache the result
                self.cache[cache_key] = (weather_data, time.time())

                # Store in database
                self._store_weather_data(weather_data)

                return weather_data

            return None

        except Exception as e:
            logger.error(
                f"Error getting forecast weather for {venue_code} on {target_date}: {e}"
            )
            return None

    def _store_weather_data(self, weather_data: WeatherData):
        """Store weather data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find venue code from location
            venue_code = None
            for code, venue in self.venue_locations.items():
                if venue.venue_name == weather_data.location:
                    venue_code = venue.venue_code
                    break

            if not venue_code:
                logger.warning(
                    f"Could not find venue code for location: {weather_data.location}"
                )
                return

            cursor.execute(
                """
                INSERT OR REPLACE INTO weather_data_v2 
                (venue_code, race_date, race_time, temperature, humidity, wind_speed, 
                 wind_direction, pressure, condition, precipitation, visibility, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    venue_code,
                    weather_data.timestamp.strftime("%Y-%m-%d"),
                    weather_data.timestamp.isoformat(),
                    weather_data.temperature,
                    weather_data.humidity,
                    weather_data.wind_speed,
                    weather_data.wind_direction,
                    weather_data.pressure,
                    weather_data.condition.value,
                    weather_data.precipitation,
                    weather_data.visibility,
                    weather_data.confidence,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing weather data: {e}")

    def calculate_weather_adjustment_factor(
        self, weather_data: WeatherData, venue_code: str
    ) -> float:
        """Calculate weather-based adjustment factor for predictions"""
        try:
            if not weather_data:
                return 1.0

            adjustment_factor = 1.0

            # Temperature adjustments
            if weather_data.temperature < 10:
                adjustment_factor *= 0.95  # Cold weather slightly slows times
            elif weather_data.temperature > 30:
                adjustment_factor *= 0.92  # Hot weather can significantly slow times

            # Humidity adjustments
            if weather_data.humidity > 80:
                adjustment_factor *= 0.96  # High humidity affects performance
            elif weather_data.humidity < 30:
                adjustment_factor *= 0.98  # Very dry conditions

            # Wind adjustments
            if weather_data.wind_speed > 20:
                adjustment_factor *= 0.94  # Strong wind significantly affects racing
            elif weather_data.wind_speed > 15:
                adjustment_factor *= 0.97  # Moderate wind

            # Precipitation adjustments
            if weather_data.precipitation > 5:
                adjustment_factor *= 0.90  # Rain significantly affects track
            elif weather_data.precipitation > 0:
                adjustment_factor *= 0.95  # Light rain

            # Condition-specific adjustments
            condition_adjustments = {
                WeatherCondition.HEAVY_RAIN: 0.85,
                WeatherCondition.RAIN: 0.92,
                WeatherCondition.LIGHT_RAIN: 0.96,
                WeatherCondition.STORM: 0.80,
                WeatherCondition.FOG: 0.94,
                WeatherCondition.CLEAR: 1.02,
                WeatherCondition.PARTLY_CLOUDY: 1.01,
            }

            if weather_data.condition in condition_adjustments:
                adjustment_factor *= condition_adjustments[weather_data.condition]

            # Ensure adjustment factor stays within reasonable bounds
            adjustment_factor = max(0.75, min(1.15, adjustment_factor))

            return adjustment_factor

        except Exception as e:
            logger.error(f"Error calculating weather adjustment factor: {e}")
            return 1.0

    def get_weather_for_race(
        self, venue_code: str, race_date: str, race_time: str = None
    ) -> Optional[WeatherData]:
        """Get weather data for a specific race"""
        try:
            # Parse race date
            race_datetime = datetime.strptime(race_date, "%Y-%m-%d")
            today = datetime.now().date()

            if race_datetime.date() <= today:
                # For current or recent races, get current weather
                weather_data = self.get_current_weather(venue_code)
            else:
                # For future races, get forecast
                weather_data = self.get_forecast_weather(venue_code, race_date)

            return weather_data

        except Exception as e:
            logger.error(f"Error getting weather for race: {e}")
            return None

    def get_available_venues(self) -> List[Dict]:
        """Get list of available venues with weather support"""
        return [
            {
                "venue_code": venue.venue_code,
                "venue_name": venue.venue_name,
                "city": venue.city,
                "state": venue.state,
                "latitude": venue.latitude,
                "longitude": venue.longitude,
            }
            for venue in self.venue_locations.values()
        ]


def main():
    """Test the Open-Meteo weather service"""
    print("🌤️  Testing Open-Meteo Weather Service")
    print("=" * 50)

    # Initialize the service
    weather_service = OpenMeteoWeatherService()

    # Test current weather for Melbourne (Sandown)
    print("\n1. Getting current weather for Melbourne (Sandown)...")
    weather_data = weather_service.get_current_weather("SANDOWN")

    if weather_data:
        print(f"   ✅ Success!")
        print(f"   📍 Location: {weather_data.location}")
        print(f"   🌡️  Temperature: {weather_data.temperature:.1f}°C")
        print(f"   💧 Humidity: {weather_data.humidity:.1f}%")
        print(
            f"   💨 Wind: {weather_data.wind_speed:.1f} km/h @ {weather_data.wind_direction}°"
        )
        print(f"   🌧️  Precipitation: {weather_data.precipitation:.1f}mm")
        print(f"   ☁️  Condition: {weather_data.condition.value}")
        print(f"   📊 Pressure: {weather_data.pressure:.1f} hPa")

        # Calculate adjustment factor
        adjustment = weather_service.calculate_weather_adjustment_factor(
            weather_data, "SAN"
        )
        print(f"\n   🎯 Racing Impact:")
        print(f"   📈 Adjustment Factor: {adjustment:.3f}")

        if adjustment < 0.95:
            print(f"   ⚠️  Challenging conditions - expect slower times")
        elif adjustment > 1.01:
            print(f"   🏃 Favorable conditions - expect faster times")
        else:
            print(f"   ✅ Neutral conditions")

    else:
        print("   ❌ Failed to get weather data")

    # Test forecast
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n2. Getting forecast for {tomorrow}...")
    forecast_data = weather_service.get_forecast_weather("SANDOWN", tomorrow)

    if forecast_data:
        print(f"   ✅ Forecast retrieved!")
        print(f"   🌡️  Expected Temperature: {forecast_data.temperature:.1f}°C")
        print(f"   ☁️  Expected Condition: {forecast_data.condition.value}")
        print(f"   🌧️  Expected Precipitation: {forecast_data.precipitation:.1f}mm")
    else:
        print("   ❌ Failed to get forecast data")

    print(f"\n" + "=" * 50)
    print("🎯 Open-Meteo Weather Service Test Complete!")


if __name__ == "__main__":
    main()
