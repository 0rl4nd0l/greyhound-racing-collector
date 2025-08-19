#!/usr/bin/env python3
"""
BOM Weather API Integration Service
==================================

Weather service for greyhound racing system using Bureau of Meteorology (BOM) API.
Provides real-time and historical weather data for race venues to enhance predictions.

Key Features:
- Real-time weather data collection
- Historical weather pattern analysis
- Venue-specific weather mapping
- Weather impact modeling
- Cache system for efficiency

Author: AI Assistant
Date: July 25, 2025
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    """Standard weather conditions"""

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
    wind_direction: str
    pressure: float
    condition: WeatherCondition
    precipitation: float
    visibility: float
    uv_index: Optional[float] = None
    confidence: float = 1.0


@dataclass
class VenueLocation:
    """Venue location mapping"""

    venue_code: str
    venue_name: str
    city: str
    state: str
    latitude: float
    longitude: float
    bom_station_id: str
    timezone: str


class BOMWeatherService:
    """Bureau of Meteorology Weather Service"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.cache_duration = 300  # 5 minutes cache
        self.request_delay = 1.0  # Rate limiting
        self.last_request_time = 0
        self.cache = {}

        # Initialize venue mappings
        self.venue_locations = self._initialize_venue_locations()

        # Initialize database tables
        self._initialize_weather_tables()

    def _initialize_venue_locations(self) -> Dict[str, VenueLocation]:
        """Initialize venue to BOM station mappings"""
        venues = {
            "ANGLE_PARK": VenueLocation(
                venue_code="AP_K",
                venue_name="Angle Park",
                city="Adelaide",
                state="SA",
                latitude=-34.8468,
                longitude=138.5390,
                bom_station_id="023000",  # Adelaide (West Terrace)
                timezone="Australia/Adelaide",
            ),
            "SANDOWN": VenueLocation(
                venue_code="SAN",
                venue_name="Sandown",
                city="Melbourne",
                state="VIC",
                latitude=-37.9451,
                longitude=145.1320,
                bom_station_id="086071",  # Melbourne (Olympic Park)
                timezone="Australia/Melbourne",
            ),
            "WENTWORTH_PARK": VenueLocation(
                venue_code="WPK",
                venue_name="Wentworth Park",
                city="Sydney",
                state="NSW",
                latitude=-33.8721,
                longitude=151.1949,
                bom_station_id="066062",  # Sydney (Observatory Hill)
                timezone="Australia/Sydney",
            ),
            "THE_MEADOWS": VenueLocation(
                venue_code="MEA",
                venue_name="The Meadows",
                city="Melbourne",
                state="VIC",
                latitude=-37.9911,
                longitude=145.0964,
                bom_station_id="086071",  # Melbourne (Olympic Park)
                timezone="Australia/Melbourne",
            ),
            "DAPTO": VenueLocation(
                venue_code="DAPT",
                venue_name="Dapto",
                city="Wollongong",
                state="NSW",
                latitude=-34.4989,
                longitude=150.7947,
                bom_station_id="068072",  # Wollongong (University)
                timezone="Australia/Sydney",
            ),
            "HOBART": VenueLocation(
                venue_code="HOBT",
                venue_name="Hobart",
                city="Hobart",
                state="TAS",
                latitude=-42.8826,
                longitude=147.3257,
                bom_station_id="094029",  # Hobart (Ellerslie Road)
                timezone="Australia/Hobart",
            ),
            "GOSFORD": VenueLocation(
                venue_code="GOSF",
                venue_name="Gosford",
                city="Gosford",
                state="NSW",
                latitude=-33.4269,
                longitude=151.3428,
                bom_station_id="061412",  # Gosford
                timezone="Australia/Sydney",
            ),
            "CANNINGTON": VenueLocation(
                venue_code="CANN",
                venue_name="Cannington",
                city="Perth",
                state="WA",
                latitude=-32.0168,
                longitude=115.9398,
                bom_station_id="009021",  # Perth Airport
                timezone="Australia/Perth",
            ),
            "BALLARAT": VenueLocation(
                venue_code="BAL",
                venue_name="Ballarat",
                city="Ballarat",
                state="VIC",
                latitude=-37.5622,
                longitude=143.8503,
                bom_station_id="089002",  # Ballarat
                timezone="Australia/Melbourne",
            ),
            "BENDIGO": VenueLocation(
                venue_code="BEN",
                venue_name="Bendigo",
                city="Bendigo",
                state="VIC",
                latitude=-36.7581,
                longitude=144.2789,
                bom_station_id="081123",  # Bendigo
                timezone="Australia/Melbourne",
            ),
            "DUBBO": VenueLocation(
                venue_code="DUB",
                venue_name="Dubbo",
                city="Dubbo",
                state="NSW",
                latitude=-32.2433,
                longitude=148.6019,
                bom_station_id="065070",  # Dubbo Airport AWS
                timezone="Australia/Sydney",
            ),
            "Q1_LAKESIDE": VenueLocation(
                venue_code="Q1L",
                venue_name="Ladbrokes Q1 Lakeside",
                city="Brisbane",
                state="QLD",
                latitude=-27.4705,
                longitude=153.0260,
                bom_station_id="040913",  # Brisbane Airport
                timezone="Australia/Brisbane",
            ),
            # Add more venues as needed
        }

        return venues

    def _initialize_weather_tables(self):
        """Initialize weather-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Weather data table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    race_time DATETIME,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    pressure REAL,
                    condition TEXT,
                    precipitation REAL,
                    visibility REAL,
                    uv_index REAL,
                    data_source TEXT DEFAULT 'BOM_API',
                    collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 1.0,
                    UNIQUE(venue_code, race_date, race_time)
                )
            """
            )

            # Weather impact analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_impact_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    weather_condition TEXT NOT NULL,
                    temperature_range TEXT NOT NULL,
                    humidity_range TEXT NOT NULL,
                    wind_range TEXT NOT NULL,
                    avg_winning_time REAL,
                    time_variance REAL,
                    favorite_strike_rate REAL,
                    avg_winning_margin REAL,
                    track_bias_impact TEXT,
                    sample_size INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(venue_code, weather_condition, temperature_range, humidity_range, wind_range)
                )
            """
            )

            # Weather forecast cache
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_forecast_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    forecast_date DATE NOT NULL,
                    forecast_data TEXT NOT NULL,
                    cache_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    UNIQUE(venue_code, forecast_date)
                )
            """
            )

            conn.commit()
            conn.close()
            logger.info("Weather database tables initialized successfully")

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

    def _get_cache_key(self, venue_code: str, target_date: str) -> str:
        """Generate cache key for weather data"""
        return f"{venue_code}_{target_date}"

    def _get_cached_weather(
        self, venue_code: str, target_date: str
    ) -> Optional[WeatherData]:
        """Retrieve cached weather data"""
        cache_key = self._get_cache_key(venue_code, target_date)

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
            else:
                # Cache expired
                del self.cache[cache_key]

        return None

    def _cache_weather(
        self, venue_code: str, target_date: str, weather_data: WeatherData
    ):
        """Cache weather data"""
        cache_key = self._get_cache_key(venue_code, target_date)
        self.cache[cache_key] = (weather_data, time.time())

    def get_current_weather(self, venue_code: str) -> Optional[WeatherData]:
        """Get current weather for a venue"""
        try:
            if venue_code not in self.venue_locations:
                logger.error(f"Unknown venue code: {venue_code}")
                return None

            venue = self.venue_locations[venue_code]

            # Check cache first
            today = datetime.now().strftime("%Y-%m-%d")
            cached_data = self._get_cached_weather(venue_code, today)
            if cached_data:
                return cached_data

            # Rate limit API requests
            self._rate_limit()

            # BOM API endpoint for current weather
            url = f"http://www.bom.gov.au/fwo/IDN60901/IDN60901.{venue.bom_station_id}.json"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse BOM response
            weather_data = self._parse_bom_current_weather(data, venue)

            if weather_data:
                # Cache the result
                self._cache_weather(venue_code, today, weather_data)

                # Store in database
                self._store_weather_data(weather_data)

            return weather_data

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
            cached_data = self._get_cached_weather(venue_code, target_date)
            if cached_data:
                return cached_data

            # Check database cache
            db_cached = self._get_forecast_from_db(venue_code, target_date)
            if db_cached:
                return db_cached

            # Rate limit API requests
            self._rate_limit()

            # BOM API endpoint for forecast
            url = f"http://www.bom.gov.au/fwo/IDN11060/IDN11060.{venue.bom_station_id}.json"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse BOM forecast response
            weather_data = self._parse_bom_forecast_weather(data, venue, target_date)

            if weather_data:
                # Cache the result
                self._cache_weather(venue_code, target_date, weather_data)

                # Store in database
                self._store_forecast_data(venue_code, target_date, data)
                self._store_weather_data(weather_data)

            return weather_data

        except requests.RequestException as e:
            logger.error(
                f"Error fetching forecast weather for {venue_code} on {target_date}: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting forecast weather: {e}")
            return None

    def _parse_bom_current_weather(
        self, data: dict, venue: VenueLocation
    ) -> Optional[WeatherData]:
        """Parse BOM current weather response"""
        try:
            observations = data.get("observations", {}).get("data", [])
            if not observations:
                return None

            # Get most recent observation
            latest = observations[0]

            # Extract weather data
            temperature = float(latest.get("air_temp", 0) or 0)
            humidity = float(latest.get("rel_hum", 0) or 0)
            wind_speed = float(latest.get("wind_spd_kmh", 0) or 0)
            wind_direction = latest.get("wind_dir", "N") or "N"
            pressure = float(latest.get("press_qnh", 1013.25) or 1013.25)

            # Determine weather condition
            condition = self._determine_weather_condition(latest)

            # Parse timestamp
            timestamp_str = latest.get("aifstime_utc", "")
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else datetime.now()
            )

            return WeatherData(
                location=venue.venue_name,
                timestamp=timestamp,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                pressure=pressure,
                condition=condition,
                precipitation=float(latest.get("rain_trace", 0) or 0),
                visibility=float(latest.get("vis_km", 10) or 10),
                confidence=0.95,  # High confidence for current data
            )

        except Exception as e:
            logger.error(f"Error parsing BOM current weather data: {e}")
            return None

    def _parse_bom_forecast_weather(
        self, data: dict, venue: VenueLocation, target_date: str
    ) -> Optional[WeatherData]:
        """Parse BOM forecast weather response"""
        try:
            forecasts = data.get("forecasts", [])
            if not forecasts:
                return None

            # Find forecast for target date
            target_forecast = None
            for forecast in forecasts:
                forecast_date = forecast.get("date", "")
                if forecast_date == target_date:
                    target_forecast = forecast
                    break

            if not target_forecast:
                return None

            # Extract forecast data
            temp_min = float(target_forecast.get("temp_min", 15) or 15)
            temp_max = float(target_forecast.get("temp_max", 25) or 25)
            temperature = (temp_min + temp_max) / 2  # Average temperature

            # Estimate other values from forecast text and historical patterns
            humidity = self._estimate_humidity_from_forecast(target_forecast)
            wind_speed = self._estimate_wind_from_forecast(target_forecast)
            wind_direction = target_forecast.get("wind_direction", "N") or "N"

            condition = self._determine_forecast_condition(target_forecast)
            precipitation = float(target_forecast.get("rain_amount_max", 0) or 0)

            timestamp = datetime.strptime(target_date, "%Y-%m-%d")

            return WeatherData(
                location=venue.venue_name,
                timestamp=timestamp,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                pressure=1013.25,  # Standard pressure for forecast
                condition=condition,
                precipitation=precipitation,
                visibility=10.0,  # Assume good visibility unless otherwise indicated
                confidence=0.75,  # Lower confidence for forecast data
            )

        except Exception as e:
            logger.error(f"Error parsing BOM forecast weather data: {e}")
            return None

    def _determine_weather_condition(self, observation: dict) -> WeatherCondition:
        """Determine weather condition from BOM observation"""
        try:
            weather_text = observation.get("weather", "").lower()
            cloud_coverage = observation.get("cloud", "")
            precipitation = float(observation.get("rain_trace", 0) or 0)

            if precipitation > 10:
                return WeatherCondition.HEAVY_RAIN
            elif precipitation > 2:
                return WeatherCondition.RAIN
            elif precipitation > 0:
                return WeatherCondition.LIGHT_RAIN
            elif "storm" in weather_text:
                return WeatherCondition.STORM
            elif "fog" in weather_text:
                return WeatherCondition.FOG
            elif "mist" in weather_text or "haze" in weather_text:
                return WeatherCondition.MIST
            elif cloud_coverage and "8" in cloud_coverage:
                return WeatherCondition.OVERCAST
            elif cloud_coverage and any(x in cloud_coverage for x in ["6", "7"]):
                return WeatherCondition.CLOUDY
            elif cloud_coverage and any(x in cloud_coverage for x in ["3", "4", "5"]):
                return WeatherCondition.PARTLY_CLOUDY
            else:
                return WeatherCondition.CLEAR

        except Exception:
            return WeatherCondition.CLEAR

    def _determine_forecast_condition(self, forecast: dict) -> WeatherCondition:
        """Determine weather condition from BOM forecast"""
        try:
            forecast_text = forecast.get("short_text", "").lower()
            precipitation = float(forecast.get("rain_amount_max", 0) or 0)

            if precipitation > 10:
                return WeatherCondition.HEAVY_RAIN
            elif precipitation > 2:
                return WeatherCondition.RAIN
            elif precipitation > 0:
                return WeatherCondition.LIGHT_RAIN
            elif any(word in forecast_text for word in ["storm", "thunder"]):
                return WeatherCondition.STORM
            elif any(word in forecast_text for word in ["fog", "foggy"]):
                return WeatherCondition.FOG
            elif any(word in forecast_text for word in ["overcast", "heavy cloud"]):
                return WeatherCondition.OVERCAST
            elif any(word in forecast_text for word in ["cloudy", "cloud"]):
                return WeatherCondition.CLOUDY
            elif any(word in forecast_text for word in ["partly", "few cloud"]):
                return WeatherCondition.PARTLY_CLOUDY
            else:
                return WeatherCondition.CLEAR

        except Exception:
            return WeatherCondition.CLEAR

    def _estimate_humidity_from_forecast(self, forecast: dict) -> float:
        """Estimate humidity from forecast information"""
        try:
            forecast_text = forecast.get("short_text", "").lower()

            if any(word in forecast_text for word in ["humid", "muggy"]):
                return 80.0
            elif any(word in forecast_text for word in ["dry", "arid"]):
                return 30.0
            elif any(word in forecast_text for word in ["rain", "shower", "storm"]):
                return 75.0
            else:
                return 55.0  # Default moderate humidity

        except Exception:
            return 55.0

    def _estimate_wind_from_forecast(self, forecast: dict) -> float:
        """Estimate wind speed from forecast information"""
        try:
            forecast_text = forecast.get("short_text", "").lower()

            if any(word in forecast_text for word in ["strong wind", "gale", "gusty"]):
                return 25.0
            elif any(word in forecast_text for word in ["windy", "breezy"]):
                return 15.0
            elif any(word in forecast_text for word in ["light wind", "calm"]):
                return 5.0
            else:
                return 10.0  # Default moderate wind

        except Exception:
            return 10.0

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
                INSERT OR REPLACE INTO weather_data 
                (venue_code, race_date, race_time, temperature, humidity, wind_speed, 
                 wind_direction, pressure, condition, precipitation, visibility, 
                 uv_index, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    weather_data.uv_index,
                    weather_data.confidence,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing weather data: {e}")

    def _store_forecast_data(
        self, venue_code: str, target_date: str, forecast_data: dict
    ):
        """Store forecast data in cache table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            expires_at = datetime.now() + timedelta(
                hours=6
            )  # Forecasts expire after 6 hours

            cursor.execute(
                """
                INSERT OR REPLACE INTO weather_forecast_cache 
                (venue_code, forecast_date, forecast_data, expires_at)
                VALUES (?, ?, ?, ?)
            """,
                (
                    venue_code,
                    target_date,
                    json.dumps(forecast_data),
                    expires_at.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing forecast cache: {e}")

    def _get_forecast_from_db(
        self, venue_code: str, target_date: str
    ) -> Optional[WeatherData]:
        """Retrieve forecast from database cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT forecast_data, expires_at 
                FROM weather_forecast_cache 
                WHERE venue_code = ? AND forecast_date = ?
            """,
                (venue_code, target_date),
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                forecast_data_json, expires_at_str = result
                expires_at = datetime.fromisoformat(expires_at_str)

                if datetime.now() < expires_at:
                    # Cache is still valid
                    forecast_data = json.loads(forecast_data_json)
                    venue = self.venue_locations[venue_code]
                    return self._parse_bom_forecast_weather(
                        forecast_data, venue, target_date
                    )

            return None

        except Exception as e:
            logger.error(f"Error retrieving forecast from database: {e}")
            return None

    def get_weather_for_race(
        self, venue_code: str, race_date: str, race_time: str = None
    ) -> Optional[WeatherData]:
        """Get weather data for a specific race"""
        try:
            # Parse race date
            race_datetime = datetime.strptime(race_date, "%Y-%m-%d")
            today = datetime.now().date()

            if race_datetime.date() <= today:
                # For current or past races, try to get actual weather data
                weather_data = self.get_current_weather(venue_code)
            else:
                # For future races, get forecast
                weather_data = self.get_forecast_weather(venue_code, race_date)

            return weather_data

        except Exception as e:
            logger.error(f"Error getting weather for race: {e}")
            return None

    def bulk_update_weather_for_upcoming_races(
        self, race_list: List[Dict]
    ) -> Dict[str, WeatherData]:
        """Update weather data for multiple upcoming races"""
        results = {}

        try:
            # Group races by venue and date to minimize API calls
            venue_date_map = {}
            for race in race_list:
                venue = race.get("venue")
                race_date = race.get("race_date")
                if venue and race_date:
                    if venue not in venue_date_map:
                        venue_date_map[venue] = set()
                    venue_date_map[venue].add(race_date)

            # Use ThreadPoolExecutor for concurrent requests
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_key = {}

                for venue, dates in venue_date_map.items():
                    for date in dates:
                        key = f"{venue}_{date}"
                        future = executor.submit(self.get_weather_for_race, venue, date)
                        future_to_key[future] = key

                # Collect results
                for future in future_to_key:
                    key = future_to_key[future]
                    try:
                        weather_data = future.result(timeout=30)
                        if weather_data:
                            results[key] = weather_data
                    except Exception as e:
                        logger.error(f"Error getting weather for {key}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in bulk weather update: {e}")
            return results

    def get_weather_impact_analysis(self, venue_code: str) -> Dict:
        """Get weather impact analysis for a venue"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT weather_condition, temperature_range, humidity_range, wind_range,
                       avg_winning_time, time_variance, favorite_strike_rate, 
                       avg_winning_margin, track_bias_impact, sample_size
                FROM weather_impact_analysis 
                WHERE venue_code = ?
            """,
                (venue_code,),
            )

            results = cursor.fetchall()
            conn.close()

            analysis = {}
            for row in results:
                condition_key = f"{row[0]}_{row[1]}_{row[2]}_{row[3]}"
                analysis[condition_key] = {
                    "weather_condition": row[0],
                    "temperature_range": row[1],
                    "humidity_range": row[2],
                    "wind_range": row[3],
                    "avg_winning_time": row[4],
                    "time_variance": row[5],
                    "favorite_strike_rate": row[6],
                    "avg_winning_margin": row[7],
                    "track_bias_impact": row[8],
                    "sample_size": row[9],
                }

            return analysis

        except Exception as e:
            logger.error(f"Error getting weather impact analysis: {e}")
            return {}

    def calculate_weather_adjustment_factor(
        self, weather_data: WeatherData, venue_code: str
    ) -> float:
        """Calculate weather-based adjustment factor for predictions"""
        try:
            if not weather_data:
                return 1.0  # No adjustment if no weather data

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

    def get_available_venues(self) -> List[Dict]:
        """Get list of available venues with weather support"""
        return [
            {
                "venue_code": venue.venue_code,
                "venue_name": venue.venue_name,
                "city": venue.city,
                "state": venue.state,
                "bom_station_id": venue.bom_station_id,
            }
            for venue in self.venue_locations.values()
        ]

    def test_api_connection(self) -> Dict[str, bool]:
        """Test API connection for all venues"""
        results = {}

        for venue_code in self.venue_locations.keys():
            try:
                weather_data = self.get_current_weather(venue_code)
                results[venue_code] = weather_data is not None
            except Exception as e:
                logger.error(f"API test failed for {venue_code}: {e}")
                results[venue_code] = False

        return results


# Weather analysis utilities
class WeatherAnalyzer:
    """Weather impact analysis utilities"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path

    def analyze_weather_impact_on_performance(self, venue_code: str = None) -> Dict:
        """Analyze weather impact on race performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query to get race results with weather data
            where_clause = "WHERE rm.venue = ?" if venue_code else ""
            params = [venue_code] if venue_code else []

            query = f"""
                SELECT 
                    rm.venue,
                    wd.temperature,
                    wd.humidity,
                    wd.wind_speed,
                    wd.condition,
                    wd.precipitation,
                    rm.winner_odds,
                    rm.winner_margin,
                    COUNT(*) as race_count
                FROM race_metadata rm
                JOIN weather_data wd ON rm.venue = wd.venue_code 
                    AND rm.race_date = wd.race_date
                {where_clause}
                GROUP BY rm.venue, wd.condition, 
                    CASE 
                        WHEN wd.temperature < 15 THEN 'cold'
                        WHEN wd.temperature > 25 THEN 'hot'
                        ELSE 'moderate'
                    END,
                    CASE 
                        WHEN wd.humidity < 50 THEN 'low'
                        WHEN wd.humidity > 70 THEN 'high'
                        ELSE 'moderate'
                    END
                HAVING COUNT(*) >= 5
                ORDER BY rm.venue, wd.condition
            """

            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()

            analysis = {}
            for row in results:
                venue = row[0]
                if venue not in analysis:
                    analysis[venue] = []

                analysis[venue].append(
                    {
                        "temperature": row[1],
                        "humidity": row[2],
                        "wind_speed": row[3],
                        "condition": row[4],
                        "precipitation": row[5],
                        "avg_winner_odds": row[6],
                        "avg_winner_margin": row[7],
                        "sample_size": row[8],
                    }
                )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}")
            return {}


# Example usage and testing
def main():
    """Example usage of the weather service"""
    # Initialize the weather service
    weather_service = BOMWeatherService()

    # Test API connections
    print("Testing API connections...")
    connection_results = weather_service.test_api_connection()
    for venue, status in connection_results.items():
        print(f"{venue}: {'✓' if status else '✗'}")

    # Get current weather for a venue
    print("\nGetting current weather for Angle Park...")
    current_weather = weather_service.get_current_weather("ANGLE_PARK")
    if current_weather:
        print(f"Temperature: {current_weather.temperature}°C")
        print(f"Humidity: {current_weather.humidity}%")
        print(
            f"Wind: {current_weather.wind_speed} km/h {current_weather.wind_direction}"
        )
        print(f"Condition: {current_weather.condition.value}")

    # Get forecast for future date
    print("\nGetting forecast for tomorrow...")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_weather = weather_service.get_forecast_weather("ANGLE_PARK", tomorrow)
    if forecast_weather:
        print(f"Forecast Temperature: {forecast_weather.temperature}°C")
        print(f"Forecast Condition: {forecast_weather.condition.value}")

    # Calculate weather adjustment factor
    if current_weather:
        adjustment = weather_service.calculate_weather_adjustment_factor(
            current_weather, "ANGLE_PARK"
        )
        print(f"\nWeather adjustment factor: {adjustment:.3f}")


if __name__ == "__main__":
    main()
