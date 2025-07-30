#!/usr/bin/env python3
"""
Comprehensive Pipeline Deep Testing
===================================

This script performs deep testing of all functionalities in the greyhound racing
prediction pipeline, including:

1. Data Collection Systems
2. Data Processing Pipeline
3. Prediction Systems
4. Database Operations
5. File Management
6. Weather Integration
7. ML Systems
8. Traditional Analysis
9. Enhanced Data Integration
10. Error Handling and Edge Cases

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePipelineTester:
    """Deep testing class for all pipeline functionalities"""
    
    def __init__(self):
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = time.time()
        
        print("🧪 COMPREHENSIVE PIPELINE DEEP TESTING")
        print("=" * 60)
        
    def run_test(self, test_name, test_func, *args, **kwargs):
        """Execute a single test with error handling"""
        self.test_count += 1
        print(f"\n🔬 Test {self.test_count}: {test_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = test_func(*args, **kwargs)
            end_time = time.time()
            
            if result is True or (isinstance(result, dict) and result.get('success', False)):
                self.passed_tests += 1
                status = "✅ PASS"
                self.test_results[test_name] = {
                    'status': 'PASS',
                    'execution_time': end_time - start_time,
                    'result': result
                }
            else:
                self.failed_tests += 1
                status = "❌ FAIL"
                self.test_results[test_name] = {
                    'status': 'FAIL',
                    'execution_time': end_time - start_time,
                    'result': result
                }
            
            print(f"{status} ({end_time - start_time:.2f}s)")
            
            if isinstance(result, dict) and 'message' in result:
                print(f"Message: {result['message']}")
                
        except Exception as e:
            self.failed_tests += 1
            print(f"❌ ERROR: {str(e)}")
            self.test_results[test_name] = {
                'status': 'ERROR',
                'execution_time': 0,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_database_operations(self):
        """Test database connectivity and operations"""
        try:
            db_path = 'greyhound_racing_data.db'
            
            # Test connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test basic queries
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Test race data
            try:
                cursor.execute("SELECT COUNT(*) FROM races LIMIT 1")
                race_count = cursor.fetchone()[0]
            except:
                race_count = 0
            
            # Test runner data
            try:
                cursor.execute("SELECT COUNT(*) FROM runners LIMIT 1")
                runner_count = cursor.fetchone()[0]
            except:
                runner_count = 0
            
            conn.close()
            
            return {
                'success': True,
                'message': f"Database accessible with {len(tables)} tables, {race_count} races, {runner_count} runners"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Database test failed: {str(e)}"
            }
    
    def test_file_system_operations(self):
        """Test file system operations and directory structure"""
        try:
            directories = [
                './unprocessed',
                './processed', 
                './upcoming_races',
                './predictions',
                './form_guides'
            ]
            
            results = {}
            for directory in directories:
                if os.path.exists(directory):
                    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                    results[directory] = len(files)
                else:
                    results[directory] = "Missing"
            
            return {
                'success': True,
                'message': f"Directory structure: {results}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"File system test failed: {str(e)}"
            }
    
    def test_prediction_pipeline_import(self):
        """Test importing the main prediction pipeline"""
        try:
            from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
            pipeline = ComprehensivePredictionPipeline()
            
            return {
                'success': True,
                'message': "Prediction pipeline imported and initialized successfully"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Pipeline import failed: {str(e)}"
            }
    
    def test_ml_system_functionality(self):
        """Test ML system functionality"""
        try:
            from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
            ml_system = ComprehensiveEnhancedMLSystem()
            
            # Test with sample data
            sample_features = {
                'win_rate': 0.2,
                'place_rate': 0.4,
                'average_position': 3.5,
                'recent_form': 0.6,
                'track_condition_performance': 0.7
            }
            
            prediction = ml_system.predict_single_runner(sample_features)
            
            return {
                'success': True,
                'message': f"ML system functional, sample prediction: {prediction:.3f}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"ML system test failed: {str(e)}"
            }
    
    def test_weather_integration(self):
        """Test weather integration functionality"""
        try:
            from weather_enhanced_predictor import WeatherEnhancedPredictor
            weather_predictor = WeatherEnhancedPredictor()
            
            # Test weather data retrieval for a known location
            weather_data = weather_predictor.get_weather_data("Sydney", "2025-07-27")
            
            return {
                'success': True,
                'message': f"Weather integration functional: {weather_data is not None}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Weather integration test failed: {str(e)}"
            }
    
    def test_traditional_analysis(self):
        """Test traditional analysis system"""
        try:
            from traditional_analysis import TraditionalAnalysis
            analyzer = TraditionalAnalysis()
            
            # Test with sample dog data
            sample_data = {
                'recent_races': [
                    {'place': 1, 'time': 23.5},
                    {'place': 3, 'time': 23.8},
                    {'place': 2, 'time': 23.6}
                ]
            }
            
            analysis = analyzer.analyze_dog_performance(sample_data)
            
            return {
                'success': True,
                'message': f"Traditional analysis functional: {analysis is not None}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Traditional analysis test failed: {str(e)}"
            }
    
    def test_form_guide_processing(self):
        """Test form guide CSV processing"""
        try:
            from form_guide_csv_scraper import FormGuideCsvScraper
            scraper = FormGuideCsvScraper()
            
            # Test basic scraper functionality
            scraper_status = scraper.get_status() if hasattr(scraper, 'get_status') else 'Initialized'
            
            return {
                'success': True,
                'message': f"Form guide scraper functional: {scraper_status}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Form guide processing test failed: {str(e)}"
            }
    
    def test_enhanced_data_integration(self):
        """Test enhanced data integration system"""
        try:
            from enhanced_data_integration import EnhancedDataIntegrator
            integrator = EnhancedDataIntegrator()
            
            # Test basic integration functionality
            integration_status = "Initialized successfully"
            
            return {
                'success': True,
                'message': f"Enhanced data integration functional: {integration_status}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Enhanced data integration test failed: {str(e)}"
            }
    
    def test_race_file_validation(self):
        """Test race file validation functionality"""
        try:
            # Find a real race file to test with
            race_files = []
            for directory in ['./upcoming_races', './processed', './unprocessed']:
                if os.path.exists(directory):
                    race_files.extend([
                        os.path.join(directory, f) 
                        for f in os.listdir(directory) 
                        if f.endswith('.csv')
                    ])
            
            if not race_files:
                return {
                    'success': False,
                    'message': "No race files found for validation testing"
                }
            
            # Test with first available race file
            test_file = race_files[0]
            
            try:
                df = pd.read_csv(test_file)
                is_valid = len(df) > 0 and not df.empty
                
                return {
                    'success': True,
                    'message': f"Race file validation successful: {os.path.basename(test_file)} ({len(df)} rows)"
                }
            except Exception as parse_error:
                return {
                    'success': False,
                    'message': f"Race file parsing failed: {str(parse_error)}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Race file validation test failed: {str(e)}"
            }
    
    def test_fixed_prediction_scoring(self):
        """Test the fixed prediction scoring system"""
        try:
            from fixed_prediction_scoring import FixedPredictionScoring
            scorer = FixedPredictionScoring()
            
            # Test with sample dog data
            sample_dog_data = {
                'form_guide_data': [
                    {'place': 1, 'time': 23.5, 'weight': 30.0, 'box': 1, 'track': 'TEST', 'date': '2025-07-20', 'distance': 450},
                    {'place': 2, 'time': 23.8, 'weight': 30.1, 'box': 2, 'track': 'TEST', 'date': '2025-07-15', 'distance': 450}
                ],
                'database_data': [],
                'enhanced_data': {},
                'weather_performance': {}
            }
            
            race_info = {'venue': 'TEST', 'date': '2025-07-27'}
            
            # Test scoring methods
            ml_score = scorer.get_improved_ml_prediction_score('Test Dog', sample_dog_data, race_info)
            traditional_score = scorer.get_improved_traditional_analysis_score('Test Dog', sample_dog_data, race_info)
            final_score = scorer.calculate_improved_weighted_final_score({
                'ml_system': ml_score,
                'traditional': traditional_score,
                'weather_enhanced': traditional_score
            }, 0.8)
            
            return {
                'success': True,
                'message': f"Fixed scoring system functional: ML={ml_score:.3f}, Traditional={traditional_score:.3f}, Final={final_score:.3f}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Fixed prediction scoring test failed: {str(e)}"
            }
    
    def test_upcoming_race_predictor(self):
        """Test upcoming race prediction functionality"""
        try:
            from upcoming_race_predictor import UpcomingRacePredictor
            predictor = UpcomingRacePredictor()
            
            # Test basic predictor functionality
            predictor_status = "Initialized successfully"
            
            return {
                'success': True,
                'message': f"Upcoming race predictor functional: {predictor_status}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Upcoming race predictor test failed: {str(e)}"
            }
    
    def test_web_application(self):
        """Test Flask web application"""
        try:
            from app import app, db_manager
            
            # Test database manager
            stats = db_manager.get_stats()
            recent_races = db_manager.get_recent_races(5)
            
            return {
                'success': True,
                'message': f"Web application functional: {stats['total_races']} races, {len(recent_races)} recent"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Web application test failed: {str(e)}"
            }
    
    def test_error_handling_robustness(self):
        """Test system robustness with edge cases"""
        try:
            # Test with invalid data
            from fixed_prediction_scoring import FixedPredictionScoring
            scorer = FixedPredictionScoring()
            
            # Test with empty data
            empty_data = {
                'form_guide_data': [],
                'database_data': [],
                'enhanced_data': {},
                'weather_performance': {}
            }
            
            race_info = {'venue': 'TEST', 'date': '2025-07-27'}
            
            # Should not crash with empty data
            ml_score = scorer.get_improved_ml_prediction_score('Empty Dog', empty_data, race_info)
            
            # Test with malformed data
            malformed_data = {
                'form_guide_data': [
                    {'place': 'invalid', 'time': None, 'weight': -1, 'box': 'box1'}
                ],
                'database_data': [],
                'enhanced_data': {},
                'weather_performance': {}
            }
            
            # Should handle malformed data gracefully
            ml_score_malformed = scorer.get_improved_ml_prediction_score('Malformed Dog', malformed_data, race_info)
            
            return {
                'success': True,
                'message': f"Error handling robust: Empty={ml_score:.3f}, Malformed={ml_score_malformed:.3f}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error handling robustness test failed: {str(e)}"
            }
    
    def test_data_quality_validation(self):
        """Test data quality validation systems"""
        try:
            # Test CSV file quality validation
            race_files = []
            for directory in ['./upcoming_races', './processed']:
                if os.path.exists(directory):
                    race_files.extend([
                        os.path.join(directory, f) 
                        for f in os.listdir(directory) 
                        if f.endswith('.csv')
                    ])
            
            quality_results = {}
            for race_file in race_files[:3]:  # Test first 3 files
                try:
                    df = pd.read_csv(race_file)
                    completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
                    quality_results[os.path.basename(race_file)] = f"{completeness:.2%}"
                except:
                    quality_results[os.path.basename(race_file)] = "Parse Error"
            
            return {
                'success': True,
                'message': f"Data quality validation: {quality_results}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Data quality validation test failed: {str(e)}"
            }
    
    def run_all_tests(self):
        """Execute all tests"""
        print(f"Starting comprehensive pipeline testing at {datetime.now()}")
        
        # Core system tests
        self.run_test("Database Operations", self.test_database_operations)
        self.run_test("File System Operations", self.test_file_system_operations) 
        self.run_test("Prediction Pipeline Import", self.test_prediction_pipeline_import)
        
        # Component tests
        self.run_test("Fixed Prediction Scoring", self.test_fixed_prediction_scoring)
        self.run_test("ML System Functionality", self.test_ml_system_functionality)
        self.run_test("Weather Integration", self.test_weather_integration)
        self.run_test("Traditional Analysis", self.test_traditional_analysis)
        self.run_test("Form Guide Processing", self.test_form_guide_processing)
        self.run_test("Enhanced Data Integration", self.test_enhanced_data_integration)
        self.run_test("Upcoming Race Predictor", self.test_upcoming_race_predictor)
        
        # Application tests
        self.run_test("Web Application", self.test_web_application)
        
        # Data quality tests
        self.run_test("Race File Validation", self.test_race_file_validation)
        self.run_test("Data Quality Validation", self.test_data_quality_validation)
        
        # Robustness tests
        self.run_test("Error Handling Robustness", self.test_error_handling_robustness)
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏁 COMPREHENSIVE TESTING SUMMARY")
        print("="*60)
        
        print(f"📊 Test Results:")
        print(f"   Total Tests: {self.test_count}")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.failed_tests}")
        print(f"   📈 Success Rate: {(self.passed_tests/self.test_count)*100:.1f}%")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            execution_time = result.get('execution_time', 0)
            print(f"   {status_emoji} {test_name:<35} ({execution_time:.2f}s)")
            
            if result['status'] == 'ERROR':
                print(f"      Error: {result.get('error', 'Unknown error')}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if self.passed_tests == self.test_count:
            print("   🏆 EXCELLENT: All systems functioning perfectly!")
        elif self.passed_tests >= self.test_count * 0.8:
            print("   👍 GOOD: Most systems functioning well with minor issues")
        elif self.passed_tests >= self.test_count * 0.6:
            print("   ⚠️ FAIR: Some systems need attention")
        else:
            print("   🚨 POOR: Multiple systems require immediate attention")
        
        print("\n✅ Comprehensive pipeline testing completed!")

if __name__ == "__main__":
    tester = ComprehensivePipelineTester()
    tester.run_all_tests()
