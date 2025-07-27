#!/usr/bin/env python3
"""
Weather Data Updater for Historical Races
==========================================

This script retroactively adds weather data to all existing races in the database,
regardless of their processing status. It enriches historical race data with:
- Weather conditions at race time
- Temperature, humidity, wind data
- Weather adjustment factors for performance analysis
- Complete weather context for better predictions

Author: AI Assistant
Date: July 25, 2025
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
from typing import Dict, List, Optional

# Import the weather service
try:
    from weather_service_open_meteo import OpenMeteoWeatherService
except ImportError:
    print("❌ Error: weather_service_open_meteo module not found")
    sys.exit(1)

class WeatherDataUpdater:
    """Updates existing race records with weather data"""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.weather_service = None
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        
        # Initialize weather service
        try:
            self.weather_service = OpenMeteoWeatherService(db_path)
            print("✅ Weather service initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize weather service: {e}")
            sys.exit(1)
    
    def get_races_needing_weather_update(self) -> List[Dict]:
        """Get all races that need weather data updates"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get races without weather data or with incomplete weather data
            query = """
                SELECT race_id, venue, race_date, race_number
                FROM race_metadata 
                WHERE weather_condition IS NULL 
                   OR weather_condition = ''
                   OR weather_adjustment_factor IS NULL
                ORDER BY race_date DESC, venue, race_number
            """
            
            races_df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert to list of dictionaries
            races = races_df.to_dict('records')
            
            print(f"📊 Found {len(races)} races needing weather updates")
            return races
            
        except Exception as e:
            print(f"❌ Error getting races: {e}")
            return []
    
    def collect_weather_for_race(self, race: Dict) -> Optional[Dict]:
        """Collect weather data for a specific race"""
        try:
            venue_code = race['venue']
            race_date = race['race_date']
            
            # Convert date to string format if needed
            if isinstance(race_date, str):
                race_date_str = race_date
            else:
                race_date_str = race_date.strftime('%Y-%m-%d')
            
            print(f"   🌤️ Fetching weather for {venue_code} on {race_date_str}...")
            
            # Get weather data from OpenMeteo service
            weather_data = self.weather_service.get_weather_for_race(venue_code, race_date_str)
            
            if weather_data:
                # Convert WeatherData object to dictionary for database storage
                weather_dict = {
                    'weather_condition': weather_data.condition.value,
                    'temperature': weather_data.temperature,
                    'humidity': weather_data.humidity,
                    'wind_speed': weather_data.wind_speed,
                    'wind_direction': weather_data.wind_direction,
                    'pressure': weather_data.pressure,
                    'precipitation': weather_data.precipitation,
                    'visibility': weather_data.visibility,
                    'weather_location': weather_data.location,
                    'weather_timestamp': weather_data.timestamp.isoformat()
                }
                
                # Calculate weather adjustment factor
                adjustment_factor = self.weather_service.calculate_weather_adjustment_factor(
                    weather_data, venue_code
                )
                weather_dict['weather_adjustment_factor'] = adjustment_factor
                
                print(f"   ✅ Weather collected: {weather_data.condition.value}, {weather_data.temperature:.1f}°C")
                print(f"   🎯 Adjustment factor: {adjustment_factor:.3f}")
                
                return weather_dict
            else:
                print(f"   ⚠️ No weather data available for {venue_code} on {race_date_str}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error collecting weather for race {race.get('race_id', 'unknown')}: {e}")
            return None
    
    def update_race_weather_data(self, race_id: str, weather_data: Dict) -> bool:
        """Update race metadata with weather data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the race_metadata table with weather data
            update_query = """
                UPDATE race_metadata 
                SET weather_condition = ?,
                    temperature = ?,
                    humidity = ?,
                    wind_speed = ?,
                    wind_direction = ?,
                    pressure = ?,
                    precipitation = ?,
                    visibility = ?,
                    weather_location = ?,
                    weather_timestamp = ?,
                    weather_adjustment_factor = ?
                WHERE race_id = ?
            """
            
            cursor.execute(update_query, (
                weather_data['weather_condition'],
                weather_data['temperature'],
                weather_data['humidity'],
                weather_data['wind_speed'],
                weather_data['wind_direction'],
                weather_data['pressure'],
                weather_data['precipitation'],
                weather_data['visibility'],
                weather_data['weather_location'],
                weather_data['weather_timestamp'],
                weather_data['weather_adjustment_factor'],
                race_id
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error updating database for race {race_id}: {e}")
            return False
    
    def process_race_batch(self, races: List[Dict], batch_size: int = 10) -> None:
        """Process races in batches to avoid overwhelming the API"""
        total_races = len(races)
        
        for i in range(0, total_races, batch_size):
            batch = races[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_races + batch_size - 1) // batch_size
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} races)")
            
            for race in batch:
                race_id = race['race_id']
                venue = race['venue']
                race_date = race['race_date']
                race_number = race['race_number']
                
                print(f"\n🏁 Processing: {venue} Race {race_number} on {race_date}")
                
                # Collect weather data
                weather_data = self.collect_weather_for_race(race)
                
                if weather_data:
                    # Update database
                    if self.update_race_weather_data(race_id, weather_data):
                        self.processed_count += 1
                        print(f"   ✅ Updated race {race_id} with weather data")
                    else:
                        self.failed_count += 1
                        print(f"   ❌ Failed to update database for race {race_id}")
                else:
                    self.skipped_count += 1
                    print(f"   ⏭️ Skipped race {race_id} - no weather data available")
                
                # Rate limiting to be respectful to the API
                time.sleep(1.5)
            
            # Longer pause between batches
            if i + batch_size < total_races:
                print(f"   ⏸️ Pausing for 5 seconds between batches...")
                time.sleep(5)
    
    def update_all_races(self, batch_size: int = 10) -> None:
        """Main method to update all races with weather data"""
        print("🌤️ WEATHER DATA UPDATER FOR HISTORICAL RACES")
        print("=" * 60)
        
        # Get races needing weather updates
        races = self.get_races_needing_weather_update()
        
        if not races:
            print("✅ All races already have weather data!")
            return
        
        print(f"\n🎯 Found {len(races)} races to update")
        print(f"📦 Processing in batches of {batch_size}")
        
        # Confirm before proceeding
        response = input(f"\n❓ Proceed with updating {len(races)} races? (y/N): ")
        if response.lower() != 'y':
            print("❌ Update cancelled by user")
            return
        
        start_time = time.time()
        
        # Process all races
        self.process_race_batch(races, batch_size)
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🏁 WEATHER UPDATE COMPLETE")
        print("=" * 60)
        print(f"✅ Successfully updated: {self.processed_count} races")
        print(f"❌ Failed to update: {self.failed_count} races")
        print(f"⏭️ Skipped (no data): {self.skipped_count} races")
        print(f"⏱️ Total time: {duration:.1f} seconds")
        print(f"⚡ Average: {duration/len(races):.1f} seconds per race")
        
        if self.processed_count > 0:
            print(f"\n🎉 Weather enrichment successful!")
            print(f"📊 Database now contains weather context for {self.processed_count} additional races")
    
    def get_weather_coverage_stats(self) -> Dict:
        """Get statistics on weather data coverage"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Total races
            total_races_query = "SELECT COUNT(*) as total FROM race_metadata"
            total_races = pd.read_sql_query(total_races_query, conn)['total'].iloc[0]
            
            # Races with weather data
            with_weather_query = """
                SELECT COUNT(*) as with_weather 
                FROM race_metadata 
                WHERE weather_condition IS NOT NULL 
                AND weather_condition != ''
            """
            with_weather = pd.read_sql_query(with_weather_query, conn)['with_weather'].iloc[0]
            
            # Races by venue with weather data
            venue_stats_query = """
                SELECT venue, 
                       COUNT(*) as total_races,
                       SUM(CASE WHEN weather_condition IS NOT NULL AND weather_condition != '' 
                           THEN 1 ELSE 0 END) as races_with_weather
                FROM race_metadata 
                GROUP BY venue
                ORDER BY total_races DESC
            """
            venue_stats = pd.read_sql_query(venue_stats_query, conn)
            
            conn.close()
            
            coverage_percentage = (with_weather / total_races * 100) if total_races > 0 else 0
            
            return {
                'total_races': total_races,
                'races_with_weather': with_weather,
                'coverage_percentage': coverage_percentage,
                'venue_stats': venue_stats
            }
            
        except Exception as e:
            print(f"❌ Error getting coverage stats: {e}")
            return {}
    
    def show_coverage_report(self) -> None:
        """Display weather data coverage report"""
        stats = self.get_weather_coverage_stats()
        
        if not stats:
            return
        
        print(f"\n📊 WEATHER DATA COVERAGE REPORT")
        print("=" * 50)
        print(f"📈 Total races in database: {stats['total_races']:,}")
        print(f"🌤️ Races with weather data: {stats['races_with_weather']:,}")
        print(f"📊 Coverage percentage: {stats['coverage_percentage']:.1f}%")
        
        if 'venue_stats' in stats and not stats['venue_stats'].empty:
            print(f"\n🏁 Coverage by venue:")
            for _, row in stats['venue_stats'].head(10).iterrows():
                venue = row['venue']
                total = row['total_races']
                with_weather = row['races_with_weather']
                percentage = (with_weather / total * 100) if total > 0 else 0
                print(f"   {venue}: {with_weather}/{total} ({percentage:.1f}%)")

def main():
    """Main function"""
    print("🚀 Weather Data Updater Starting...")
    
    # Check if database exists
    if not os.path.exists("greyhound_racing_data.db"):
        print("❌ Error: Database file 'greyhound_racing_data.db' not found")
        sys.exit(1)
    
    # Initialize updater
    updater = WeatherDataUpdater()
    
    # Show current coverage
    updater.show_coverage_report()
    
    # Update all races
    updater.update_all_races(batch_size=5)  # Smaller batches to be API-friendly
    
    # Show final coverage
    print(f"\n" + "=" * 60)
    print(f"📊 FINAL COVERAGE REPORT")
    updater.show_coverage_report()

if __name__ == "__main__":
    main()
