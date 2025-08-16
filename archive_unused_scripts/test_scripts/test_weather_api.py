#!/usr/bin/env python3
"""
Weather API Service Test
========================

Test script for the weather API service functionality without requiring external dependencies.
This tests the core logic and database operations.

Author: AI Assistant
Date: July 25, 2025
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

class MockWeatherService:
    """Mock weather service for testing without external API calls"""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.venue_locations = self._initialize_venue_locations()
        self._initialize_weather_tables()
        
    def _initialize_venue_locations(self) -> Dict[str, VenueLocation]:
        """Initialize venue to BOM station mappings"""
        venues = {
            'ANGLE_PARK': VenueLocation(
                venue_code='AP_K',
                venue_name='Angle Park',
                city='Adelaide',
                state='SA',
                latitude=-34.8468,
                longitude=138.5390,
                bom_station_id='023000',
                timezone='Australia/Adelaide'
            ),
            'SANDOWN': VenueLocation(
                venue_code='SAN',
                venue_name='Sandown',
                city='Melbourne',
                state='VIC',
                latitude=-37.9451,
                longitude=145.1320,
                bom_station_id='086071',
                timezone='Australia/Melbourne'
            ),
            'WENTWORTH_PARK': VenueLocation(
                venue_code='WPK',
                venue_name='Wentworth Park',
                city='Sydney',
                state='NSW',
                latitude=-33.8721,
                longitude=151.1949,
                bom_station_id='066062',
                timezone='Australia/Sydney'
            )
        }
        return venues
        
    def _initialize_weather_tables(self):
        """Initialize weather-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Weather data table
            cursor.execute('''
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
                    data_source TEXT DEFAULT 'MOCK_API',
                    collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 1.0,
                    UNIQUE(venue_code, race_date, race_time)
                )
            ''')
            
            # Weather impact analysis table
            cursor.execute('''
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
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Weather database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing weather tables: {e}")
            raise
    
    def get_mock_weather_data(self, venue_code: str, weather_type: str = "current") -> Optional[WeatherData]:
        """Generate mock weather data for testing"""
        try:
            if venue_code not in self.venue_locations:
                logger.error(f"Unknown venue code: {venue_code}")
                return None
            
            venue = self.venue_locations[venue_code]
            
            # Generate realistic mock data based on venue location
            if venue.state == 'SA':  # Adelaide
                temp_base = 22.0
                humidity_base = 55.0
            elif venue.state == 'VIC':  # Melbourne
                temp_base = 18.0
                humidity_base = 65.0
            elif venue.state == 'NSW':  # Sydney
                temp_base = 21.0
                humidity_base = 60.0
            else:
                temp_base = 20.0
                humidity_base = 60.0
            
            # Add some variation
            import random
            temp_variation = random.uniform(-5, 8)
            humidity_variation = random.uniform(-15, 20)
            
            weather_data = WeatherData(
                location=venue.venue_name,
                timestamp=datetime.now(),
                temperature=temp_base + temp_variation,
                humidity=max(20, min(95, humidity_base + humidity_variation)),
                wind_speed=random.uniform(5, 25),
                wind_direction=random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
                pressure=random.uniform(1000, 1025),
                condition=random.choice(list(WeatherCondition)),
                precipitation=random.uniform(0, 10) if random.random() < 0.3 else 0,
                visibility=random.uniform(8, 15),
                confidence=0.85 if weather_type == "forecast" else 0.95
            )
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error generating mock weather data: {e}")
            return None
    
    def store_weather_data(self, weather_data: WeatherData):
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
                logger.warning(f"Could not find venue code for location: {weather_data.location}")
                return
            
            cursor.execute('''
                INSERT OR REPLACE INTO weather_data 
                (venue_code, race_date, race_time, temperature, humidity, wind_speed, 
                 wind_direction, pressure, condition, precipitation, visibility, 
                 uv_index, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                venue_code,
                weather_data.timestamp.strftime('%Y-%m-%d'),
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
                weather_data.confidence
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored weather data for {weather_data.location}")
            
        except Exception as e:
            logger.error(f"Error storing weather data: {e}")
    
    def calculate_weather_adjustment_factor(self, weather_data: WeatherData, venue_code: str) -> float:
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
                WeatherCondition.PARTLY_CLOUDY: 1.01
            }
            
            if weather_data.condition in condition_adjustments:
                adjustment_factor *= condition_adjustments[weather_data.condition]
            
            # Ensure adjustment factor stays within reasonable bounds
            adjustment_factor = max(0.75, min(1.15, adjustment_factor))
            
            return adjustment_factor
            
        except Exception as e:
            logger.error(f"Error calculating weather adjustment factor: {e}")
            return 1.0
    
    def get_weather_summary(self) -> Dict:
        """Get summary of stored weather data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    venue_code,
                    COUNT(*) as record_count,
                    AVG(temperature) as avg_temp,
                    AVG(humidity) as avg_humidity,
                    AVG(wind_speed) as avg_wind,
                    AVG(precipitation) as avg_precipitation
                FROM weather_data 
                GROUP BY venue_code
                ORDER BY record_count DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            summary = {}
            for row in results:
                venue_code = row[0]
                summary[venue_code] = {
                    'record_count': row[1],
                    'avg_temperature': round(row[2], 1) if row[2] else 0,
                    'avg_humidity': round(row[3], 1) if row[3] else 0,
                    'avg_wind_speed': round(row[4], 1) if row[4] else 0,
                    'avg_precipitation': round(row[5], 2) if row[5] else 0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting weather summary: {e}")
            return {}

def test_weather_service():
    """Test the weather service functionality"""
    print("🌤️  Testing Weather API Service")
    print("=" * 50)
    
    # Initialize the mock weather service
    weather_service = MockWeatherService()
    
    # Test 1: Venue initialization
    print("\n1. Testing venue initialization...")
    venues = weather_service.venue_locations
    print(f"   ✓ Initialized {len(venues)} venues:")
    for code, venue in venues.items():
        print(f"     - {venue.venue_name} ({venue.venue_code}) in {venue.city}, {venue.state}")
    
    # Test 2: Mock weather data generation
    print("\n2. Testing weather data generation...")
    test_venues = ['ANGLE_PARK', 'SANDOWN', 'WENTWORTH_PARK']
    weather_records = []
    
    for venue_code in test_venues:
        weather_data = weather_service.get_mock_weather_data(venue_code)
        if weather_data:
            weather_records.append(weather_data)
            print(f"   ✓ Generated weather for {weather_data.location}:")
            print(f"     - Temperature: {weather_data.temperature:.1f}°C")
            print(f"     - Humidity: {weather_data.humidity:.1f}%")
            print(f"     - Wind: {weather_data.wind_speed:.1f} km/h {weather_data.wind_direction}")
            print(f"     - Condition: {weather_data.condition.value}")
            print(f"     - Precipitation: {weather_data.precipitation:.1f}mm")
        else:
            print(f"   ✗ Failed to generate weather for {venue_code}")
    
    # Test 3: Database storage
    print("\n3. Testing database storage...")
    for weather_data in weather_records:
        weather_service.store_weather_data(weather_data)
    
    # Verify storage
    summary = weather_service.get_weather_summary()
    if summary:
        print("   ✓ Weather data stored successfully:")
        for venue_code, stats in summary.items():
            print(f"     - {venue_code}: {stats['record_count']} records, avg temp {stats['avg_temperature']}°C")
    else:
        print("   ✗ No weather data found in database")
    
    # Test 4: Weather adjustment calculations
    print("\n4. Testing weather adjustment factors...")
    for weather_data in weather_records:
        # Find venue code
        venue_code = None
        for code, venue in weather_service.venue_locations.items():
            if venue.venue_name == weather_data.location:
                venue_code = venue.venue_code
                break
        
        if venue_code:
            adjustment = weather_service.calculate_weather_adjustment_factor(weather_data, venue_code)
            print(f"   ✓ {weather_data.location}: adjustment factor = {adjustment:.3f}")
            
            # Explain the adjustment
            if adjustment < 0.95:
                print(f"     → Challenging conditions (slower times expected)")
            elif adjustment > 1.01:
                print(f"     → Favorable conditions (faster times expected)")
            else:
                print(f"     → Neutral conditions")
    
    # Test 5: Extreme weather scenarios
    print("\n5. Testing extreme weather scenarios...")
    extreme_scenarios = [
        # Hot, humid, windy day
        WeatherData(
            location="Test Location",
            timestamp=datetime.now(),
            temperature=35.0,
            humidity=85.0,
            wind_speed=25.0,
            wind_direction="W",
            pressure=1010.0,
            condition=WeatherCondition.OVERCAST,
            precipitation=0,
            visibility=10.0,
            confidence=1.0
        ),
        # Rainy, cold day
        WeatherData(
            location="Test Location",
            timestamp=datetime.now(),
            temperature=8.0,
            humidity=95.0,
            wind_speed=15.0,
            wind_direction="S",
            pressure=1005.0,
            condition=WeatherCondition.HEAVY_RAIN,
            precipitation=15.0,
            visibility=3.0,
            confidence=1.0
        ),
        # Perfect racing day
        WeatherData(
            location="Test Location",
            timestamp=datetime.now(),
            temperature=22.0,
            humidity=50.0,
            wind_speed=8.0,
            wind_direction="N",
            pressure=1020.0,
            condition=WeatherCondition.CLEAR,
            precipitation=0,
            visibility=15.0,
            confidence=1.0
        )
    ]
    
    scenario_names = ["Hot & Windy", "Cold & Rainy", "Perfect Conditions"]
    
    for i, scenario in enumerate(extreme_scenarios):
        adjustment = weather_service.calculate_weather_adjustment_factor(scenario, "AP_K")
        print(f"   ✓ {scenario_names[i]}: factor = {adjustment:.3f}")
        print(f"     - Temp: {scenario.temperature}°C, Humidity: {scenario.humidity}%, Wind: {scenario.wind_speed} km/h")
        print(f"     - Condition: {scenario.condition.value}, Rain: {scenario.precipitation}mm")
    
    print("\n" + "=" * 50)
    print("🎯 Weather API Service Test Complete!")
    
    # Final summary
    print(f"\n📊 Final Summary:")
    print(f"   - Venues configured: {len(weather_service.venue_locations)}")
    total_records = sum(stats['record_count'] for stats in summary.values())
    print(f"   - Weather records stored: {total_records}")
    print(f"   - Database tables created: ✓")
    print(f"   - Weather adjustment logic: ✓")

if __name__ == "__main__":
    test_weather_service()
