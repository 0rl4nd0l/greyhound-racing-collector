#!/usr/bin/env python3
"""
Comprehensive Weather Service Test Suite
========================================

This script thoroughly tests the weather service across:
- All race venues
- Historical dates
- Current and forecast data
- Error handling
- Database operations
- Performance metrics

Author: AI Assistant
Date: July 25, 2025
"""

import sys
import time
from datetime import datetime, timedelta
from weather_service_open_meteo import OpenMeteoWeatherService, WeatherData
import sqlite3
import traceback

class WeatherServiceTester:
    """Comprehensive weather service testing class"""
    
    def __init__(self):
        self.weather_service = OpenMeteoWeatherService()
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'performance': [],
            'venue_results': {},
            'historical_results': {}
        }
        
    def print_header(self, title: str):
        """Print a formatted test section header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_test(self, test_name: str, status: str, details: str = ""):
        """Print test result with formatting"""
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   └─ {details}")
    
    def test_venue_coverage(self):
        """Test 1: Verify all venues are configured and accessible"""
        self.print_header("Test 1: Venue Coverage & Configuration")
        
        venues = self.weather_service.get_available_venues()
        expected_venues = ['AP_K', 'SAN', 'WPK', 'MEA', 'DAPT', 'HOBT', 'BAL', 'BEN']
        
        # Test venue count
        if len(venues) >= 8:
            self.print_test("Venue Count", "PASS", f"Found {len(venues)} venues")
            self.test_results['passed'] += 1
        else:
            self.print_test("Venue Count", "FAIL", f"Expected ≥8, found {len(venues)}")
            self.test_results['failed'] += 1
        
        # Test individual venue configuration
        for venue in venues:
            venue_code = venue['venue_code']
            
            # Check required fields
            required_fields = ['venue_code', 'venue_name', 'city', 'state', 'latitude', 'longitude']
            missing_fields = [field for field in required_fields if field not in venue]
            
            if not missing_fields:
                # Check coordinate validity (Australia bounds)
                lat_valid = -45 <= venue['latitude'] <= -10
                lon_valid = 110 <= venue['longitude'] <= 155
                
                if lat_valid and lon_valid:
                    self.print_test(f"Venue {venue_code}", "PASS", 
                                  f"{venue['venue_name']} in {venue['city']}, {venue['state']}")
                    self.test_results['passed'] += 1
                else:
                    self.print_test(f"Venue {venue_code}", "FAIL", 
                                  f"Invalid coordinates: {venue['latitude']}, {venue['longitude']}")
                    self.test_results['failed'] += 1
            else:
                self.print_test(f"Venue {venue_code}", "FAIL", 
                              f"Missing fields: {missing_fields}")
                self.test_results['failed'] += 1
    
    def test_current_weather_all_venues(self):
        """Test 2: Current weather data for all venues"""
        self.print_header("Test 2: Current Weather Data - All Venues")
        
        venues = ['SANDOWN', 'ANGLE_PARK', 'WENTWORTH_PARK', 'BALLARAT', 'BENDIGO', 'HOBART', 'DAPTO']
        
        for venue_code in venues:
            try:
                start_time = time.time()
                weather_data = self.weather_service.get_current_weather(venue_code)
                response_time = time.time() - start_time
                
                if weather_data:
                    # Validate data quality
                    data_valid = (
                        -10 <= weather_data.temperature <= 50 and  # Reasonable temp range
                        0 <= weather_data.humidity <= 100 and     # Valid humidity
                        0 <= weather_data.wind_speed <= 150 and   # Reasonable wind
                        weather_data.pressure > 950               # Reasonable pressure
                    )
                    
                    if data_valid:
                        self.print_test(f"Current Weather {venue_code}", "PASS", 
                                      f"{weather_data.temperature:.1f}°C, {weather_data.condition.value}")
                        self.test_results['passed'] += 1
                        
                        # Store performance metrics
                        self.test_results['performance'].append({
                            'test': f'current_{venue_code}',
                            'response_time': response_time,
                            'data_quality': 'valid'
                        })
                        
                        # Store venue results
                        self.test_results['venue_results'][venue_code] = {
                            'current_weather': 'success',
                            'temperature': weather_data.temperature,
                            'condition': weather_data.condition.value,
                            'response_time': response_time
                        }
                    else:
                        self.print_test(f"Current Weather {venue_code}", "FAIL", 
                                      f"Invalid data: T={weather_data.temperature}°C, H={weather_data.humidity}%")
                        self.test_results['failed'] += 1
                else:
                    self.print_test(f"Current Weather {venue_code}", "FAIL", "No data returned")
                    self.test_results['failed'] += 1
                    
            except Exception as e:
                self.print_test(f"Current Weather {venue_code}", "FAIL", f"Exception: {str(e)}")
                self.test_results['failed'] += 1
                self.test_results['errors'].append({
                    'test': f'current_weather_{venue_code}',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    def test_historical_weather(self):
        """Test 3: Historical weather data"""
        self.print_header("Test 3: Historical Weather Data")
        
        # Test various historical dates
        historical_dates = [
            ('2025-07-20', 'Last week'),
            ('2025-07-15', '10 days ago'),
            ('2025-07-01', 'Start of month'),
            ('2025-06-15', 'Last month'),
            ('2025-01-01', 'Start of year')
        ]
        
        test_venue = 'SANDOWN'  # Use Melbourne as test venue
        
        for date_str, description in historical_dates:
            try:
                start_time = time.time()
                weather_data = self.weather_service.get_forecast_weather(test_venue, date_str)
                response_time = time.time() - start_time
                
                if weather_data:
                    self.print_test(f"Historical {description}", "PASS", 
                                  f"{date_str}: {weather_data.temperature:.1f}°C, {weather_data.condition.value}")
                    self.test_results['passed'] += 1
                    
                    # Store historical results
                    self.test_results['historical_results'][date_str] = {
                        'status': 'success',
                        'temperature': weather_data.temperature,
                        'condition': weather_data.condition.value,
                        'response_time': response_time
                    }
                else:
                    # Historical data might not be available for very old dates
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    days_ago = (datetime.now() - date_obj).days
                    
                    if days_ago > 90:  # Open-Meteo historical limit
                        self.print_test(f"Historical {description}", "WARN", 
                                      f"{date_str}: No data (expected for dates >90 days)")
                    else:
                        self.print_test(f"Historical {description}", "FAIL", 
                                      f"{date_str}: No data returned")
                        self.test_results['failed'] += 1
                        
            except Exception as e:
                self.print_test(f"Historical {description}", "FAIL", f"Exception: {str(e)}")
                self.test_results['failed'] += 1
                self.test_results['errors'].append({
                    'test': f'historical_{date_str}',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    def test_forecast_weather(self):
        """Test 4: Future weather forecasts"""
        self.print_header("Test 4: Weather Forecasts - Future Dates")
        
        # Test various future dates
        future_dates = []
        for days_ahead in [1, 2, 3, 7, 14]:
            future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            future_dates.append((future_date, f"{days_ahead} days ahead"))
        
        test_venues = ['SANDOWN', 'ANGLE_PARK', 'WENTWORTH_PARK']
        
        for venue_code in test_venues:
            venue_forecasts = 0
            for date_str, description in future_dates:
                try:
                    weather_data = self.weather_service.get_forecast_weather(venue_code, date_str)
                    
                    if weather_data:
                        venue_forecasts += 1
                        
                except Exception as e:
                    self.test_results['errors'].append({
                        'test': f'forecast_{venue_code}_{date_str}',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
            
            if venue_forecasts >= 3:  # At least 3 forecasts should work
                self.print_test(f"Forecasts {venue_code}", "PASS", 
                              f"{venue_forecasts}/{len(future_dates)} forecasts retrieved")
                self.test_results['passed'] += 1
            else:
                self.print_test(f"Forecasts {venue_code}", "FAIL", 
                              f"Only {venue_forecasts}/{len(future_dates)} forecasts retrieved")
                self.test_results['failed'] += 1
    
    def test_weather_adjustment_calculations(self):
        """Test 5: Weather adjustment factor calculations"""
        self.print_header("Test 5: Weather Adjustment Factor Calculations")
        
        # Test various weather scenarios
        test_scenarios = [
            {
                'name': 'Perfect Conditions',
                'temp': 22.0, 'humidity': 55.0, 'wind': 8.0, 'precip': 0.0,
                'expected_range': (0.98, 1.05)
            },
            {
                'name': 'Hot Weather',
                'temp': 35.0, 'humidity': 60.0, 'wind': 12.0, 'precip': 0.0,
                'expected_range': (0.90, 0.95)
            },
            {
                'name': 'Cold Weather',
                'temp': 8.0, 'humidity': 70.0, 'wind': 15.0, 'precip': 0.0,
                'expected_range': (0.92, 0.97)
            },
            {
                'name': 'Rainy Conditions',
                'temp': 18.0, 'humidity': 85.0, 'wind': 20.0, 'precip': 8.0,
                'expected_range': (0.85, 0.92)
            },
            {
                'name': 'Extreme Weather',
                'temp': 38.0, 'humidity': 90.0, 'wind': 30.0, 'precip': 15.0,
                'expected_range': (0.75, 0.85)
            }
        ]
        
        for scenario in test_scenarios:
            try:
                # Create mock weather data
                from weather_service_open_meteo import WeatherData, WeatherCondition
                
                weather_data = WeatherData(
                    location="Test Location",
                    timestamp=datetime.now(),
                    temperature=scenario['temp'],
                    humidity=scenario['humidity'],
                    wind_speed=scenario['wind'],
                    wind_direction=180,
                    pressure=1013.25,
                    condition=WeatherCondition.CLEAR,
                    precipitation=scenario['precip'],
                    visibility=10.0
                )
                
                adjustment = self.weather_service.calculate_weather_adjustment_factor(weather_data, 'SAN')
                
                min_expected, max_expected = scenario['expected_range']
                if min_expected <= adjustment <= max_expected:
                    self.print_test(f"Adjustment {scenario['name']}", "PASS", 
                                  f"Factor: {adjustment:.3f} (expected: {min_expected}-{max_expected})")
                    self.test_results['passed'] += 1
                else:
                    self.print_test(f"Adjustment {scenario['name']}", "FAIL", 
                                  f"Factor: {adjustment:.3f} (expected: {min_expected}-{max_expected})")
                    self.test_results['failed'] += 1
                    
            except Exception as e:
                self.print_test(f"Adjustment {scenario['name']}", "FAIL", f"Exception: {str(e)}")
                self.test_results['failed'] += 1
                self.test_results['errors'].append({
                    'test': f'adjustment_{scenario["name"]}',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    def test_database_operations(self):
        """Test 6: Database storage and retrieval"""
        self.print_header("Test 6: Database Operations")
        
        try:
            # Test database connection
            conn = sqlite3.connect(self.weather_service.db_path)
            cursor = conn.cursor()
            
            # Check if weather tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'weather%'")
            tables = cursor.fetchall()
            
            if len(tables) >= 1:
                self.print_test("Database Tables", "PASS", f"Found {len(tables)} weather tables")
                self.test_results['passed'] += 1
            else:
                self.print_test("Database Tables", "FAIL", "No weather tables found")
                self.test_results['failed'] += 1
            
            # Test data insertion and retrieval
            cursor.execute("SELECT COUNT(*) FROM weather_data_v2")
            record_count = cursor.fetchone()[0]
            
            if record_count > 0:
                self.print_test("Database Records", "PASS", f"Found {record_count} weather records")
                self.test_results['passed'] += 1
                
                # Test data quality
                cursor.execute("""
                    SELECT venue_code, temperature, humidity, condition 
                    FROM weather_data_v2 
                    ORDER BY collection_timestamp DESC 
                    LIMIT 5
                """)
                recent_records = cursor.fetchall()
                
                valid_records = 0
                for record in recent_records:
                    venue, temp, humidity, condition = record
                    if venue and temp and humidity and condition:
                        valid_records += 1
                
                if valid_records == len(recent_records):
                    self.print_test("Data Quality", "PASS", f"All {len(recent_records)} recent records valid")
                    self.test_results['passed'] += 1
                else:
                    self.print_test("Data Quality", "FAIL", 
                                  f"Only {valid_records}/{len(recent_records)} records valid")
                    self.test_results['failed'] += 1
            else:
                self.print_test("Database Records", "FAIL", "No weather records found")
                self.test_results['failed'] += 1
            
            conn.close()
            
        except Exception as e:
            self.print_test("Database Operations", "FAIL", f"Exception: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append({
                'test': 'database_operations',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def test_performance_metrics(self):
        """Test 7: Performance and response times"""
        self.print_header("Test 7: Performance Metrics")
        
        if self.test_results['performance']:
            response_times = [perf['response_time'] for perf in self.test_results['performance']]
            
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            
            # Performance benchmarks
            if avg_response < 5.0:  # Average under 5 seconds
                self.print_test("Average Response Time", "PASS", f"{avg_response:.2f}s")
                self.test_results['passed'] += 1
            else:
                self.print_test("Average Response Time", "FAIL", f"{avg_response:.2f}s (>5s)")
                self.test_results['failed'] += 1
            
            if max_response < 10.0:  # Max under 10 seconds
                self.print_test("Maximum Response Time", "PASS", f"{max_response:.2f}s")
                self.test_results['passed'] += 1
            else:
                self.print_test("Maximum Response Time", "FAIL", f"{max_response:.2f}s (>10s)")
                self.test_results['failed'] += 1
            
            print(f"   📊 Performance Summary:")
            print(f"      • Average: {avg_response:.2f}s")
            print(f"      • Range: {min_response:.2f}s - {max_response:.2f}s")
            print(f"      • Total Tests: {len(response_times)}")
        else:
            self.print_test("Performance Metrics", "FAIL", "No performance data collected")
            self.test_results['failed'] += 1
    
    def test_error_handling(self):
        """Test 8: Error handling and edge cases"""
        self.print_header("Test 8: Error Handling & Edge Cases")
        
        # Test invalid venue codes
        try:
            invalid_weather = self.weather_service.get_current_weather('INVALID_VENUE')
            if invalid_weather is None:
                self.print_test("Invalid Venue Handling", "PASS", "Correctly returned None")
                self.test_results['passed'] += 1
            else:
                self.print_test("Invalid Venue Handling", "FAIL", "Should return None for invalid venue")
                self.test_results['failed'] += 1
        except Exception as e:
            self.print_test("Invalid Venue Handling", "FAIL", f"Exception: {str(e)}")
            self.test_results['failed'] += 1
        
        # Test invalid dates
        try:
            invalid_forecast = self.weather_service.get_forecast_weather('SANDOWN', '2025-13-45')
            if invalid_forecast is None:
                self.print_test("Invalid Date Handling", "PASS", "Correctly handled invalid date")
                self.test_results['passed'] += 1
            else:
                self.print_test("Invalid Date Handling", "FAIL", "Should handle invalid dates gracefully")
                self.test_results['failed'] += 1
        except Exception:
            # Exception is acceptable for invalid dates
            self.print_test("Invalid Date Handling", "PASS", "Exception raised for invalid date (acceptable)")
            self.test_results['passed'] += 1
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        self.print_header("COMPREHENSIVE TEST REPORT")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        success_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   ✅ Passed: {self.test_results['passed']}")
        print(f"   ❌ Failed: {self.test_results['failed']}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        if self.test_results['performance']:
            avg_time = sum(p['response_time'] for p in self.test_results['performance']) / len(self.test_results['performance'])
            print(f"   ⚡ Average Response: {avg_time:.2f}s")
        
        # Venue summary
        print(f"\n🏟️ Venue Test Results:")
        for venue, results in self.test_results['venue_results'].items():
            status = "✅" if results['current_weather'] == 'success' else "❌"
            print(f"   {status} {venue}: {results.get('temperature', 'N/A')}°C, {results.get('condition', 'N/A')}")
        
        # Error summary
        if self.test_results['errors']:
            print(f"\n🐛 Errors Encountered ({len(self.test_results['errors'])}):")
            for i, error in enumerate(self.test_results['errors'][:5], 1):  # Show first 5 errors
                print(f"   {i}. {error['test']}: {error['error']}")
            if len(self.test_results['errors']) > 5:
                print(f"   ... and {len(self.test_results['errors']) - 5} more errors")
        
        # Final assessment
        print(f"\n🎯 Final Assessment:")
        if success_rate >= 90:
            print(f"   🌟 EXCELLENT: Weather service is working at {success_rate:.1f}% success rate")
            print(f"   ✅ Ready for production use in greyhound racing predictions")
        elif success_rate >= 75:
            print(f"   👍 GOOD: Weather service is working at {success_rate:.1f}% success rate")
            print(f"   ⚠️  Minor issues to address before full deployment")
        elif success_rate >= 50:
            print(f"   ⚠️  PARTIAL: Weather service working at {success_rate:.1f}% success rate")
            print(f"   🔧 Significant issues need to be resolved")
        else:
            print(f"   ❌ POOR: Weather service only working at {success_rate:.1f}% success rate")
            print(f"   🚨 Major issues need immediate attention")
        
        return success_rate >= 75
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 COMPREHENSIVE WEATHER SERVICE TEST SUITE")
        print("🌤️  Testing all venues, dates, and functionality")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run all test modules
            self.test_venue_coverage()
            self.test_current_weather_all_venues()
            self.test_historical_weather()
            self.test_forecast_weather()
            self.test_weather_adjustment_calculations()
            self.test_database_operations()
            self.test_performance_metrics()
            self.test_error_handling()
            
            # Generate final report
            return self.generate_final_report()
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Test suite failed with exception: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main test execution"""
    tester = WeatherServiceTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
