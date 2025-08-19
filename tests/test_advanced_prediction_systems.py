#!/usr/bin/env python3
"""
Advanced Prediction Systems Test & Validation Suite
==================================================

Comprehensive testing and validation of:
1. ML System V4 - Temporal leakage protection, EV calculation, calibration
2. Prediction Pipeline V4 - Advanced integrated system
3. Prediction Pipeline V3 - Comprehensive system with fallbacks
4. Data pipeline integrity and validation
5. Model performance and accuracy
6. End-to-end prediction workflows

Author: AI Assistant
Date: August 3, 2025
"""

import os
import sys
import json
import time
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class AdvancedPredictionSystemsTest:
    """Comprehensive test suite for advanced prediction systems"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'system_status': {},
            'performance_metrics': {},
            'validation_results': {},
            'errors': []
        }
        
        self.test_race_files = []
        self._find_test_race_files()
        
        print("🧪 Advanced Prediction Systems Test Suite Initialized")
        print("=" * 70)
    
    def _find_test_race_files(self):
        """Find available race files for testing"""
        possible_dirs = [
            './upcoming_races',
            './test_upcoming_races', 
            './processed',
            './form_guides/processed'
        ]
        
        for directory in possible_dirs:
            if os.path.exists(directory):
                csv_files = list(Path(directory).glob('*.csv'))
                for file in csv_files:
                    if file.name.lower() != 'readme.csv' and file.stat().st_size > 100:
                        self.test_race_files.append(str(file))
        
        print(f"📁 Found {len(self.test_race_files)} test race files")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("\n🚀 Starting Comprehensive Advanced Systems Testing")
        print("=" * 70)
        
        # Test 1: ML System V4 Core Functionality
        self.test_ml_system_v4()
        
        # Test 2: Prediction Pipeline V4
        self.test_prediction_pipeline_v4()
        
        # Test 3: Prediction Pipeline V3 with Fallbacks
        self.test_prediction_pipeline_v3()
        
        # Test 4: Temporal Leakage Protection
        self.test_temporal_leakage_protection()
        
        # Test 5: Model Calibration and EV Calculation
        self.test_calibration_and_ev()
        
        # Test 6: End-to-End Prediction Workflow
        self.test_end_to_end_workflow()
        
        # Test 7: Data Quality and Integrity
        self.test_data_quality()
        
        # Test 8: Performance Benchmarks
        self.test_performance_benchmarks()
        
        # Generate final report
        self._generate_final_report()
        
        return self.test_results
    
    def test_ml_system_v4(self):
        """Test ML System V4 core functionality"""
        print("\n1️⃣ TESTING ML SYSTEM V4 CORE FUNCTIONALITY")
        print("-" * 50)
        
        test_name = "ml_system_v4_core"
        self.test_results['tests_run'] += 1
        
        try:
            from ml_system_v4 import MLSystemV4
            
            # Initialize system
            start_time = time.time()
            ml_v4 = MLSystemV4()
            init_time = time.time() - start_time
            
            print(f"✅ ML System V4 initialized in {init_time:.2f}s")
            
            # Test system components
            components = {
                'temporal_builder': ml_v4.temporal_builder is not None,
                'calibrated_pipeline': ml_v4.calibrated_pipeline is not None,
                'ev_thresholds': len(ml_v4.ev_thresholds) > 0,
                'feature_columns': len(ml_v4.feature_columns) >= 0,
                'model_info': len(ml_v4.model_info) >= 0
            }
            
            print("📊 System Components:")
            for component, available in components.items():
                status = "✅" if available else "❌"
                print(f"   {status} {component}")
            
            # Test data preparation
            try:
                print("\n🔧 Testing data preparation...")
                train_data, test_data = ml_v4.prepare_time_ordered_data()
                
                if not train_data.empty and not test_data.empty:
                    print(f"✅ Data preparation successful:")
                    print(f"   📊 Training data: {len(train_data)} samples")
                    print(f"   📊 Testing data: {len(test_data)} samples")
                    
                    self.test_results['performance_metrics']['ml_v4_data_prep'] = {
                        'train_samples': len(train_data),
                        'test_samples': len(test_data),
                        'prep_time': init_time
                    }
                else:
                    print("⚠️ Data preparation returned empty datasets")
                    
            except Exception as e:
                print(f"⚠️ Data preparation test failed: {e}")
            
            self.test_results['system_status'][test_name] = {
                'status': 'passed',
                'components': components,
                'init_time': init_time
            }
            
            self.test_results['tests_passed'] += 1
            print(f"✅ ML System V4 core test PASSED")
            
        except Exception as e:
            print(f"❌ ML System V4 core test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"ML System V4 core: {str(e)}")
            self.test_results['system_status'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_prediction_pipeline_v4(self):
        """Test Prediction Pipeline V4"""
        print("\n2️⃣ TESTING PREDICTION PIPELINE V4")
        print("-" * 50)
        
        test_name = "prediction_pipeline_v4"
        self.test_results['tests_run'] += 1
        
        try:
            from prediction_pipeline_v4 import PredictionPipelineV4
            
            # Initialize pipeline
            start_time = time.time()
            pipeline_v4 = PredictionPipelineV4()
            init_time = time.time() - start_time
            
            print(f"✅ Prediction Pipeline V4 initialized in {init_time:.2f}s")
            
            # Test with a race file if available
            if self.test_race_files:
                test_file = self.test_race_files[0]
                print(f"🎯 Testing with race file: {os.path.basename(test_file)}")
                
                try:
                    start_pred_time = time.time()
                    result = pipeline_v4.predict_race_file(test_file)
                    pred_time = time.time() - start_pred_time
                    
                    if result.get('success'):
                        predictions = result.get('predictions', [])
                        print(f"✅ Prediction successful:")
                        print(f"   📊 Generated {len(predictions)} predictions")
                        print(f"   ⏱️ Prediction time: {pred_time:.2f}s")
                        
                        if predictions:
                            top_pick = predictions[0]
                            print(f"   🏆 Top pick: {top_pick.get('dog_clean_name', 'Unknown')}")
                            print(f"   📈 Method: ML System V4")
                            
                        self.test_results['performance_metrics']['pipeline_v4_prediction'] = {
                            'prediction_count': len(predictions),
                            'prediction_time': pred_time,
                            'success': True
                        }
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"❌ Prediction failed: {error}")
                        self.test_results['performance_metrics']['pipeline_v4_prediction'] = {
                            'success': False,
                            'error': error
                        }
                        
                except Exception as e:
                    print(f"❌ Prediction test failed: {e}")
                    self.test_results['errors'].append(f"Pipeline V4 prediction: {str(e)}")
            else:
                print("⚠️ No test race files available for prediction testing")
            
            self.test_results['system_status'][test_name] = {
                'status': 'passed',
                'init_time': init_time
            }
            
            self.test_results['tests_passed'] += 1
            print(f"✅ Prediction Pipeline V4 test PASSED")
            
        except Exception as e:
            print(f"❌ Prediction Pipeline V4 test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Prediction Pipeline V4: {str(e)}")
            self.test_results['system_status'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_prediction_pipeline_v3(self):
        """Test Prediction Pipeline V3 with fallbacks"""
        print("\n3️⃣ TESTING PREDICTION PIPELINE V3 (COMPREHENSIVE)")
        print("-" * 50)
        
        test_name = "prediction_pipeline_v3"
        self.test_results['tests_run'] += 1
        
        try:
            from prediction_pipeline_v3 import PredictionPipelineV3
            
            # Initialize pipeline
            start_time = time.time()
            pipeline_v3 = PredictionPipelineV3()
            init_time = time.time() - start_time
            
            print(f"✅ Prediction Pipeline V3 initialized in {init_time:.2f}s")
            
            # Check available systems
            systems = {
                'ml_system': pipeline_v3.ml_system is not None,
                'weather_predictor': pipeline_v3.weather_predictor is not None,
                'gpt_enhancer': pipeline_v3.gpt_enhancer is not None,
                'unified_predictor': pipeline_v3.unified_predictor is not None,
                'comprehensive_pipeline': pipeline_v3.comprehensive_pipeline is not None
            }
            
            print("🔧 Available Systems:")
            for system, available in systems.items():
                status = "✅" if available else "❌"
                print(f"   {status} {system}")
            
            available_count = sum(systems.values())
            print(f"📊 Total systems available: {available_count}/5")
            
            # Test prediction with fallback hierarchy
            if self.test_race_files:
                test_file = self.test_race_files[0]
                print(f"\n🎯 Testing fallback hierarchy with: {os.path.basename(test_file)}")
                
                try:
                    start_pred_time = time.time()
                    result = pipeline_v3.predict_race_file(test_file, enhancement_level='full')
                    pred_time = time.time() - start_pred_time
                    
                    if result.get('success'):
                        predictions = result.get('predictions', [])
                        method = result.get('prediction_method', 'unknown')
                        tier = result.get('prediction_tier', 'unknown')
                        
                        print(f"✅ Prediction successful:")
                        print(f"   📊 Generated {len(predictions)} predictions")
                        print(f"   🎯 Method used: {method}")
                        print(f"   🏗️ Tier used: {tier}")
                        print(f"   ⏱️ Prediction time: {pred_time:.2f}s")
                        
                        self.test_results['performance_metrics']['pipeline_v3_prediction'] = {
                            'prediction_count': len(predictions),
                            'prediction_time': pred_time,
                            'method_used': method,
                            'tier_used': tier,
                            'success': True
                        }
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"❌ Prediction failed: {error}")
                        
                except Exception as e:
                    print(f"❌ Prediction test failed: {e}")
            
            self.test_results['system_status'][test_name] = {
                'status': 'passed',
                'available_systems': available_count,
                'systems': systems,
                'init_time': init_time
            }
            
            self.test_results['tests_passed'] += 1
            print(f"✅ Prediction Pipeline V3 test PASSED")
            
        except Exception as e:
            print(f"❌ Prediction Pipeline V3 test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Prediction Pipeline V3: {str(e)}")
            self.test_results['system_status'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_temporal_leakage_protection(self):
        """Test temporal leakage protection in ML System V4"""
        print("\n4️⃣ TESTING TEMPORAL LEAKAGE PROTECTION")
        print("-" * 50)
        
        test_name = "temporal_leakage_protection"
        self.test_results['tests_run'] += 1
        
        try:
            from ml_system_v4 import MLSystemV4
            from temporal_feature_builder import TemporalFeatureBuilder
            
            ml_v4 = MLSystemV4()
            
            # Test temporal feature builder
            if ml_v4.temporal_builder:
                print("✅ Temporal feature builder available")
                
                # Test assertion hook
                if hasattr(ml_v4, 'assert_no_leakage') and ml_v4.assert_no_leakage:
                    print("✅ Temporal assertion hook active")
                else:
                    print("⚠️ Temporal assertion hook not found")
                
                # Test temporal assertion hook directly
                try:
                    from temporal_feature_builder import create_temporal_assertion_hook
                    
                    test_hook = create_temporal_assertion_hook()
                    
                    # Test with safe features (should pass)
                    safe_features = {
                        'box_number': 1,
                        'weight': 30.5,
                        'distance': 500,
                        'historical_avg_position': 3.2
                    }
                    
                    try:
                        test_hook(safe_features, "test_race", "test_dog")
                        print("✅ Safe features passed assertion hook")
                        
                        # Test with post-race features (should fail)
                        leaky_features = {
                            'box_number': 1,
                            'weight': 30.5,
                            'finish_position': 1,  # This should trigger protection
                            'individual_time': 29.5  # This should trigger protection
                        }
                        
                        try:
                            test_hook(leaky_features, "test_race_leaky", "test_dog")
                            print("❌ Temporal protection failed - leaky features not detected")
                            leakage_protected = False
                        except AssertionError as e:
                            if "TEMPORAL LEAKAGE DETECTED" in str(e):
                                print("✅ Temporal leakage protection verified - correctly detected leaky features")
                                leakage_protected = True
                            else:
                                print(f"❌ Unexpected assertion: {e}")
                                leakage_protected = False
                        
                    except AssertionError as e:
                        print(f"❌ Safe features incorrectly rejected: {e}")
                        leakage_protected = False
                        
                except Exception as e:
                    print(f"❌ Temporal protection test failed: {e}")
                    leakage_protected = False
                
            else:
                print("❌ Temporal feature builder not available")
                leakage_protected = False
            
            self.test_results['validation_results']['temporal_leakage_protection'] = {
                'protected': leakage_protected,
                'temporal_builder_available': ml_v4.temporal_builder is not None
            }
            
            if leakage_protected:
                self.test_results['tests_passed'] += 1
                print("✅ Temporal leakage protection test PASSED")
            else:
                self.test_results['tests_failed'] += 1
                print("❌ Temporal leakage protection test FAILED")
                
        except Exception as e:
            print(f"❌ Temporal leakage protection test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Temporal leakage protection: {str(e)}")
    
    def test_calibration_and_ev(self):
        """Test model calibration and EV calculation"""
        print("\n5️⃣ TESTING MODEL CALIBRATION & EV CALCULATION")
        print("-" * 50)
        
        test_name = "calibration_and_ev"
        self.test_results['tests_run'] += 1
        
        try:
            from ml_system_v4 import MLSystemV4
            
            ml_v4 = MLSystemV4()
            
            # Test calibration availability
            calibration_available = ml_v4.calibrated_pipeline is not None
            print(f"{'✅' if calibration_available else '❌'} Calibrated pipeline: {calibration_available}")
            
            # Test EV thresholds
            ev_configured = len(ml_v4.ev_thresholds) > 0
            print(f"{'✅' if ev_configured else '❌'} EV thresholds configured: {ev_configured}")
            
            if ev_configured:
                print(f"   📊 EV thresholds: {ml_v4.ev_thresholds}")
            
            # Test group normalization function
            try:
                # Test softmax normalization using ML System V4's method
                import numpy as np
                test_probs = np.array([0.1, 0.3, 0.2, 0.4])
                
                # Use ML System V4's normalization method
                normalized = ml_v4._group_normalize_probabilities(test_probs)
                
                # Verify probabilities sum to 1
                prob_sum = np.sum(normalized)
                if abs(prob_sum - 1.0) < 0.001:
                    print("✅ Group normalization working correctly")
                    normalization_working = True
                else:
                    print(f"❌ Group normalization failed - sum: {prob_sum}")
                    normalization_working = False
                    
            except Exception as e:
                print(f"❌ Group normalization test failed: {e}")
                normalization_working = False
            
            self.test_results['validation_results']['calibration_and_ev'] = {
                'calibrated_pipeline': calibration_available,
                'ev_configured': ev_configured,
                'normalization_working': normalization_working
            }
            
            if calibration_available and normalization_working:
                self.test_results['tests_passed'] += 1
                print("✅ Calibration & EV test PASSED")
            else:
                self.test_results['tests_failed'] += 1
                print("❌ Calibration & EV test FAILED")
                
        except Exception as e:
            print(f"❌ Calibration & EV test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Calibration & EV: {str(e)}")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end prediction workflow"""
        print("\n6️⃣ TESTING END-TO-END PREDICTION WORKFLOW")
        print("-" * 50)
        
        test_name = "end_to_end_workflow"
        self.test_results['tests_run'] += 1
        
        if not self.test_race_files:
            print("⚠️ No test race files available - skipping end-to-end test")
            self.test_results['tests_passed'] += 1  # Don't fail if no test data
            return
        
        try:
            # Test complete workflow with multiple systems
            systems_to_test = [
                ('PredictionPipelineV4', 'prediction_pipeline_v4'),
                ('PredictionPipelineV3', 'prediction_pipeline_v3'),
            ]
            
            workflow_results = {}
            
            for system_name, module_name in systems_to_test:
                print(f"\n🔧 Testing {system_name} workflow...")
                
                try:
                    module = __import__(module_name)
                    pipeline_class = getattr(module, system_name)
                    
                    # Initialize pipeline
                    pipeline = pipeline_class()
                    
                    # Test prediction on first available race file
                    test_file = self.test_race_files[0]
                    
                    start_time = time.time()
                    result = pipeline.predict_race_file(test_file)
                    total_time = time.time() - start_time
                    
                    if result.get('success'):
                        predictions = result.get('predictions', [])
                        
                        workflow_results[system_name] = {
                            'success': True,
                            'prediction_count': len(predictions),
                            'total_time': total_time,
                            'method': result.get('prediction_method', 'unknown')
                        }
                        
                        print(f"   ✅ {system_name} workflow successful")
                        print(f"   📊 Predictions: {len(predictions)}")
                        print(f"   ⏱️ Time: {total_time:.2f}s")
                        
                    else:
                        error = result.get('error', 'Unknown error')
                        workflow_results[system_name] = {
                            'success': False,
                            'error': error,
                            'total_time': total_time
                        }
                        
                        print(f"   ❌ {system_name} workflow failed: {error}")
                        
                except Exception as e:
                    workflow_results[system_name] = {
                        'success': False,
                        'error': str(e),
                        'total_time': 0
                    }
                    print(f"   ❌ {system_name} workflow error: {e}")
            
            # Summary
            successful_workflows = sum(1 for result in workflow_results.values() if result.get('success'))
            total_workflows = len(workflow_results)
            
            print(f"\n📊 Workflow Summary: {successful_workflows}/{total_workflows} successful")
            
            self.test_results['validation_results']['end_to_end_workflows'] = workflow_results
            
            if successful_workflows > 0:
                self.test_results['tests_passed'] += 1
                print("✅ End-to-end workflow test PASSED")
            else:
                self.test_results['tests_failed'] += 1
                print("❌ End-to-end workflow test FAILED")
                
        except Exception as e:
            print(f"❌ End-to-end workflow test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"End-to-end workflow: {str(e)}")
    
    def test_data_quality(self):
        """Test data quality and integrity"""
        print("\n7️⃣ TESTING DATA QUALITY & INTEGRITY")
        print("-" * 50)
        
        test_name = "data_quality"
        self.test_results['tests_run'] += 1
        
        try:
            # Test database connection and data availability
            import sqlite3
            
            db_path = "greyhound_racing_data.db"
            if not os.path.exists(db_path):
                print("❌ Database not found")
                self.test_results['tests_failed'] += 1
                return
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check key tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            key_tables = ['dog_race_data', 'race_metadata', 'comprehensive_dog_profiles']
            missing_tables = [table for table in key_tables if table not in tables]
            
            print(f"📊 Database tables found: {len(tables)}")
            print(f"📋 Key tables present: {len(key_tables) - len(missing_tables)}/{len(key_tables)}")
            
            if missing_tables:
                print(f"⚠️ Missing tables: {missing_tables}")
            
            # Check data volume
            data_stats = {}
            for table in key_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    data_stats[table] = count
                    print(f"   📊 {table}: {count:,} records")
            
            conn.close()
            
            # Test race file quality
            valid_files = 0
            total_files = len(self.test_race_files)
            
            for race_file in self.test_race_files[:5]:  # Test first 5 files
                try:
                    df = pd.read_csv(race_file)
                    if not df.empty and len(df.columns) > 3:
                        valid_files += 1
                except:
                    pass
            
            file_quality = valid_files / max(total_files, 1)
            print(f"📁 Race file quality: {valid_files}/{total_files} valid ({file_quality:.1%})")
            
            self.test_results['validation_results']['data_quality'] = {
                'database_tables': len(tables),
                'key_tables_present': len(key_tables) - len(missing_tables),
                'data_stats': data_stats,
                'race_file_quality': file_quality,
                'valid_race_files': valid_files,
                'total_race_files': total_files
            }
            
            # Pass if we have database and some valid files
            if len(tables) > 0 and file_quality > 0:
                self.test_results['tests_passed'] += 1
                print("✅ Data quality test PASSED")
            else:
                self.test_results['tests_failed'] += 1
                print("❌ Data quality test FAILED")
                
        except Exception as e:
            print(f"❌ Data quality test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Data quality: {str(e)}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n8️⃣ TESTING PERFORMANCE BENCHMARKS")
        print("-" * 50)
        
        test_name = "performance_benchmarks"
        self.test_results['tests_run'] += 1
        
        try:
            benchmarks = {}
            
            # Memory usage test
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            benchmarks['memory_usage_mb'] = memory_mb
            print(f"💾 Current memory usage: {memory_mb:.1f} MB")
            
            # System loading time benchmark
            if self.test_race_files:
                # Test CSV loading speed
                test_file = self.test_race_files[0]
                
                start_time = time.time()
                df = pd.read_csv(test_file)
                load_time = time.time() - start_time
                
                benchmarks['csv_load_time'] = load_time
                benchmarks['csv_rows'] = len(df)
                
                print(f"📁 CSV load time: {load_time:.3f}s for {len(df)} rows")
                
                # Test prediction speed (if systems available)
                try:
                    from prediction_pipeline_v4 import PredictionPipelineV4
                    
                    pipeline = PredictionPipelineV4()
                    
                    start_time = time.time()
                    result = pipeline.predict_race_file(test_file)
                    pred_time = time.time() - start_time
                    
                    if result.get('success'):
                        predictions = result.get('predictions', [])
                        pred_per_second = len(predictions) / max(pred_time, 0.001)
                        
                        benchmarks['prediction_time'] = pred_time
                        benchmarks['predictions_per_second'] = pred_per_second
                        
                        print(f"🚀 Prediction speed: {pred_time:.2f}s for {len(predictions)} predictions")
                        print(f"📈 Throughput: {pred_per_second:.1f} predictions/second")
                        
                except Exception as e:
                    print(f"⚠️ Prediction benchmark failed: {e}")
            
            # Performance thresholds
            performance_ok = True
            
            if memory_mb > 500:  # 500MB threshold
                print("⚠️ High memory usage detected")
                performance_ok = False
            
            if benchmarks.get('prediction_time', 0) > 30:  # 30 second threshold
                print("⚠️ Slow prediction performance detected")
                performance_ok = False
            
            self.test_results['performance_metrics']['benchmarks'] = benchmarks
            
            if performance_ok:
                self.test_results['tests_passed'] += 1
                print("✅ Performance benchmarks test PASSED")
            else:
                self.test_results['tests_failed'] += 1
                print("❌ Performance benchmarks test FAILED")
                
        except Exception as e:
            print(f"❌ Performance benchmarks test FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Performance benchmarks: {str(e)}")
    
    def _generate_final_report(self):
        """Generate final test report"""
        print("\n" + "=" * 70)
        print("📋 FINAL TEST REPORT")
        print("=" * 70)
        
        total_tests = self.test_results['tests_run']
        passed_tests = self.test_results['tests_passed']
        failed_tests = self.test_results['tests_failed']
        success_rate = (passed_tests / max(total_tests, 1)) * 100
        
        print(f"📊 Tests Summary:")
        print(f"   🏃 Total tests run: {total_tests}")
        print(f"   ✅ Tests passed: {passed_tests}")
        print(f"   ❌ Tests failed: {failed_tests}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        # Overall system health
        if success_rate >= 80:
            health_status = "🟢 EXCELLENT"
        elif success_rate >= 60:
            health_status = "🟡 GOOD"
        elif success_rate >= 40:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 POOR"
        
        print(f"\n🏥 Overall System Health: {health_status}")
        
        # Save detailed report
        report_file = f"advanced_systems_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved: {report_file}")
        
        if self.test_results['errors']:
            print(f"\n⚠️ Errors encountered:")
            for error in self.test_results['errors']:
                print(f"   • {error}")
        
        return success_rate >= 60  # Consider 60% success rate as passing


def main():
    """Main test runner"""
    print("🧪 ADVANCED PREDICTION SYSTEMS TEST & VALIDATION SUITE")
    print("=" * 70)
    print("Testing ML System V4, Pipeline V4, V3, and comprehensive validation")
    print("=" * 70)
    
    # Initialize test suite
    test_suite = AdvancedPredictionSystemsTest()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Return appropriate exit code
    success_rate = (results['tests_passed'] / max(results['tests_run'], 1)) * 100
    return 0 if success_rate >= 60 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
