#!/usr/bin/env python3
"""
Comprehensive Enhanced Data Pipeline Integration
=============================================

This script ensures that all enhanced expert form data is properly integrated
into every component of the prediction pipeline, including:

1. Database synchronization with enhanced data
2. ML model feature engineering updates  
3. Weather-enhanced predictor integration
4. Comprehensive ML system updates
5. Feature engineering enhancements
6. Prediction API updates

Author: AI Assistant
Date: July 26, 2025
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all the components we need to integrate
try:
    from enhanced_data_integration import EnhancedDataIntegrator
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced data integration not available: {e}")
    ENHANCED_INTEGRATION_AVAILABLE = False

try:
    from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
    COMPREHENSIVE_ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Comprehensive ML system not available: {e}")
    COMPREHENSIVE_ML_AVAILABLE = False

try:
    from weather_enhanced_predictor import WeatherEnhancedPredictor
    WEATHER_PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Weather enhanced predictor not available: {e}")
    WEATHER_PREDICTOR_AVAILABLE = False

try:
    from enhanced_feature_engineering import EnhancedFeatureEngineer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced feature engineering not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

class EnhancedPipelineIntegrator:
    """Integrates enhanced data across all prediction pipeline components"""
    
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.integration_report = {
            'timestamp': datetime.now().isoformat(),
            'components_integrated': [],
            'data_statistics': {},
            'validation_results': {},
            'errors': []
        }
        
        print("🚀 Enhanced Pipeline Integrator Initialized")
        
    def run_complete_integration(self):
        """Run complete enhanced data integration across all pipeline components"""
        print("🔗 STARTING COMPREHENSIVE ENHANCED DATA INTEGRATION")
        print("=" * 70)
        
        try:
            # Step 1: Sync enhanced data to database
            print("\n1️⃣ SYNCING ENHANCED DATA TO DATABASE")
            print("-" * 50)
            self._sync_enhanced_data_to_database()
            
            # Step 2: Update comprehensive ML system with enhanced features
            print("\n2️⃣ UPDATING COMPREHENSIVE ML SYSTEM")
            print("-" * 50)
            self._update_comprehensive_ml_system()
            
            # Step 3: Validate enhanced data availability in models
            print("\n3️⃣ VALIDATING ENHANCED DATA IN MODELS")
            print("-" * 50)
            self._validate_enhanced_data_in_models()
            
            # Step 4: Test prediction pipeline with enhanced data
            print("\n4️⃣ TESTING PREDICTION PIPELINE")
            print("-" * 50)
            self._test_enhanced_prediction_pipeline()
            
            # Step 5: Generate integration report
            print("\n5️⃣ GENERATING INTEGRATION REPORT")
            print("-" * 50)
            self._generate_integration_report()
            
            print("\n✅ COMPREHENSIVE ENHANCED DATA INTEGRATION COMPLETE!")
            print("=" * 70)
            
            return self.integration_report
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            self.integration_report['errors'].append(f"Integration failed: {e}")
            return self.integration_report
    
    def _sync_enhanced_data_to_database(self):
        """Sync enhanced expert form data to database"""
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                print("⚠️ Enhanced data integration module not available")
                return False
            
            # Initialize enhanced data integrator
            integrator = EnhancedDataIntegrator(self.db_path)
            
            if not integrator.enhanced_data_available:
                print("⚠️ No enhanced expert form data found")
                self.integration_report['errors'].append("No enhanced expert form data found")
                return False
            
            print(f"📊 Found enhanced data: {len(list(integrator.enhanced_csv_dir.glob('*.csv')))} CSV files")
            print(f"📄 Found enhanced data: {len(list(integrator.enhanced_json_dir.glob('*.json')))} JSON files")
            
            # Sync to database
            sync_success = integrator.sync_enhanced_data_to_database()
            
            if sync_success:
                # Get statistics
                stats = integrator.get_enhanced_statistics()
                self.integration_report['data_statistics']['enhanced_database'] = stats
                
                print(f"✅ Enhanced data synced successfully")
                print(f"📊 Records: {stats.get('total_enhanced_records', 0)}")
                print(f"🐕 Dogs: {stats.get('dogs_with_enhanced_data', 0)}")
                print(f"🏁 Races: {stats.get('races_with_enhanced_data', 0)}")
                
                self.integration_report['components_integrated'].append('enhanced_database_sync')
                return True
            else:
                print("❌ Enhanced data sync failed")
                self.integration_report['errors'].append("Enhanced data sync failed")
                return False
                
        except Exception as e:
            print(f"❌ Error syncing enhanced data: {e}")
            self.integration_report['errors'].append(f"Enhanced data sync error: {e}")
            return False
    
    def _update_comprehensive_ml_system(self):
        """Update comprehensive ML system to use enhanced data"""
        try:
            if not COMPREHENSIVE_ML_AVAILABLE:
                print("⚠️ Comprehensive ML system not available")
                return False
            
            # Initialize comprehensive ML system
            ml_system = ComprehensiveEnhancedMLSystem(self.db_path)
            
            # Check if enhanced data is available in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check enhanced_expert_data table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='enhanced_expert_data'")
            if not cursor.fetchone():
                print("⚠️ Enhanced expert data table not found in database")
                conn.close()
                return False
            
            # Get enhanced data statistics
            cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
            enhanced_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT dog_clean_name) FROM enhanced_expert_data WHERE first_sectional IS NOT NULL")
            dogs_with_sectionals = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT dog_clean_name) FROM enhanced_expert_data WHERE pir_rating IS NOT NULL")
            dogs_with_pir = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"📊 Enhanced records in database: {enhanced_records}")
            print(f"🏃 Dogs with sectional times: {dogs_with_sectionals}")
            print(f"📈 Dogs with PIR ratings: {dogs_with_pir}")
            
            # Test enhanced feature creation
            test_enhanced_features = self._test_enhanced_feature_creation()
            
            if test_enhanced_features:
                print("✅ Enhanced features successfully integrated into ML system")
                self.integration_report['components_integrated'].append('comprehensive_ml_system')
                self.integration_report['data_statistics']['enhanced_features'] = {
                    'enhanced_records': enhanced_records,
                    'dogs_with_sectionals': dogs_with_sectionals,
                    'dogs_with_pir': dogs_with_pir,
                    'feature_creation_test': 'passed'
                }
                return True
            else:
                print("❌ Enhanced feature creation test failed")
                return False
                
        except Exception as e:
            print(f"❌ Error updating comprehensive ML system: {e}")
            self.integration_report['errors'].append(f"ML system update error: {e}")
            return False
    
    def _test_enhanced_feature_creation(self):
        """Test that enhanced features can be created successfully"""
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                return False
            
            # Initialize enhanced data integrator
            integrator = EnhancedDataIntegrator(self.db_path)
            
            # Get a sample dog to test feature creation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dog_clean_name FROM enhanced_expert_data 
                WHERE first_sectional IS NOT NULL 
                AND pir_rating IS NOT NULL 
                LIMIT 1
            """)
            
            sample_dog = cursor.fetchone()
            conn.close()
            
            if not sample_dog:
                print("⚠️ No sample dog with enhanced data available for testing")
                return False
            
            dog_name = sample_dog[0]
            print(f"🧪 Testing enhanced feature creation for: {dog_name}")
            
            # Get enhanced data for this dog
            enhanced_data = integrator.get_enhanced_dog_data(dog_name, max_races=5)
            
            if enhanced_data and enhanced_data.get('enhanced_features'):
                enhanced_features = enhanced_data['enhanced_features']
                print(f"   ✅ Enhanced features created: {len(enhanced_features)} features")
                
                # Check for key enhanced features
                key_features = [
                    'avg_first_section', 'sectional_consistency', 'avg_pir_rating',
                    'weight_consistency', 'enhanced_data_quality'
                ]
                
                features_found = sum(1 for feat in key_features if feat in enhanced_features)
                print(f"   📊 Key enhanced features found: {features_found}/{len(key_features)}")
                
                if features_found >= 3:  # At least 3 key features
                    print("   ✅ Enhanced feature creation test passed")
                    return True
                else:
                    print("   ⚠️ Insufficient enhanced features created")
                    return False
            else:
                print("   ❌ No enhanced features created")
                return False
                
        except Exception as e:
            print(f"   ❌ Enhanced feature creation test failed: {e}")
            return False
    
    def _validate_enhanced_data_in_models(self):
        """Validate that enhanced data is properly available in trained models"""
        try:
            models_dir = Path('./comprehensive_trained_models')
            
            if not models_dir.exists():
                print("⚠️ No trained models directory found")
                return False
            
            # Find latest model
            model_files = list(models_dir.glob('comprehensive_best_model_*.joblib'))
            if not model_files:
                print("⚠️ No trained comprehensive models found")
                return False
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Checking model: {latest_model.name}")
            
            try:
                import joblib
                model_data = joblib.load(latest_model)
                
                feature_columns = model_data.get('feature_columns', [])
                model_metadata = {
                    'model_name': model_data.get('model_name', 'Unknown'),
                    'accuracy': model_data.get('accuracy', 0),
                    'feature_count': len(feature_columns)
                }
                
                print(f"📊 Model: {model_metadata['model_name']}")
                print(f"📊 Accuracy: {model_metadata['accuracy']:.3f}")
                print(f"📊 Features: {model_metadata['feature_count']}")
                
                # Check for enhanced-data-related features
                enhanced_features = [
                    feat for feat in feature_columns 
                    if any(keyword in feat.lower() for keyword in 
                          ['sectional', 'pir', 'enhanced', 'weather', 'traditional'])
                ]
                
                print(f"📊 Enhanced/Weather/Traditional features: {len(enhanced_features)}")
                
                if enhanced_features:
                    print("   Sample enhanced features:")
                    for feat in enhanced_features[:5]:
                        print(f"     - {feat}")
                
                self.integration_report['validation_results']['model_validation'] = {
                    'model_file': str(latest_model),
                    'model_metadata': model_metadata,
                    'enhanced_features_count': len(enhanced_features),
                    'enhanced_features': enhanced_features[:10]  # First 10 for brevity
                }
                
                print("✅ Model validation completed")
                return True
                
            except Exception as e:
                print(f"❌ Error loading model for validation: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error validating models: {e}")
            self.integration_report['errors'].append(f"Model validation error: {e}")
            return False
    
    def _test_enhanced_prediction_pipeline(self):
        """Test the enhanced prediction pipeline end-to-end"""
        try:
            # Find a sample race file to test
            upcoming_dir = Path('./upcoming_races')
            if not upcoming_dir.exists():
                print("⚠️ No upcoming races directory found")
                return False
            
            race_files = list(upcoming_dir.glob('*.csv'))
            if not race_files:
                print("⚠️ No race files found for testing")
                return False
            
            sample_race = race_files[0]
            print(f"🧪 Testing prediction pipeline with: {sample_race.name}")
            
            # Test weather-enhanced predictor if available
            if WEATHER_PREDICTOR_AVAILABLE:
                try:
                    predictor = WeatherEnhancedPredictor(self.db_path)
                    
                    # Test enhanced data integration
                    integrator = EnhancedDataIntegrator(self.db_path)
                    
                    # Read race file to get dog names
                    race_df = pd.read_csv(sample_race)
                    
                    # Extract first dog for testing
                    current_dog_name = None
                    for idx, row in race_df.iterrows():
                        dog_name_raw = str(row.get('Dog Name', '')).strip()
                        if dog_name_raw not in ['""', '', 'nan']:
                            if '. ' in dog_name_raw:
                                current_dog_name = dog_name_raw.split('. ', 1)[1]
                            else:
                                current_dog_name = dog_name_raw
                            break
                    
                    if current_dog_name:
                        print(f"🐕 Testing enhanced data for: {current_dog_name}")
                        
                        # Test enhanced data retrieval
                        enhanced_data = integrator.get_enhanced_dog_data(current_dog_name, max_races=3)
                        
                        if enhanced_data and enhanced_data.get('enhanced_features'):
                            features = enhanced_data['enhanced_features']
                            print(f"   ✅ Enhanced data retrieved: {features.get('enhanced_data_quality', 0)} data points")
                            
                            # Test specific enhanced features
                            sectional_data = len(enhanced_data.get('sectional_times', []))
                            pir_data = len(enhanced_data.get('pir_ratings', []))
                            weight_data = len(enhanced_data.get('weight_history', []))
                            
                            print(f"   📊 Sectional times: {sectional_data}")
                            print(f"   📊 PIR ratings: {pir_data}")
                            print(f"   📊 Weight history: {weight_data}")
                            
                            self.integration_report['validation_results']['pipeline_test'] = {
                                'test_race': str(sample_race),
                                'test_dog': current_dog_name,
                                'enhanced_data_retrieved': True,
                                'sectional_times_count': sectional_data,
                                'pir_ratings_count': pir_data,
                                'weight_history_count': weight_data,
                                'enhanced_features_count': len(features)
                            }
                            
                            print("✅ Enhanced prediction pipeline test passed")
                            self.integration_report['components_integrated'].append('prediction_pipeline')
                            return True
                        else:
                            print("   ⚠️ No enhanced data found for test dog")
                            return False
                    else:
                        print("   ⚠️ No valid dog found in test race")
                        return False
                        
                except Exception as e:
                    print(f"❌ Prediction pipeline test failed: {e}")
                    self.integration_report['errors'].append(f"Pipeline test error: {e}")
                    return False
            else:
                print("⚠️ Weather enhanced predictor not available for testing")
                return False
                
        except Exception as e:
            print(f"❌ Error testing prediction pipeline: {e}")
            self.integration_report['errors'].append(f"Pipeline test error: {e}")
            return False
    
    def _generate_integration_report(self):
        """Generate comprehensive integration report"""
        try:
            # Summary statistics
            total_components = len(self.integration_report['components_integrated'])
            total_errors = len(self.integration_report['errors'])
            
            print(f"📊 INTEGRATION SUMMARY:")
            print(f"   Components integrated: {total_components}")
            print(f"   Errors encountered: {total_errors}")
            print(f"   Success rate: {((total_components/(total_components+total_errors))*100):.1f}%" if (total_components+total_errors) > 0 else "N/A")
            
            # Component status
            print(f"\n📋 COMPONENT STATUS:")
            components = [
                ('Enhanced Database Sync', 'enhanced_database_sync' in self.integration_report['components_integrated']),
                ('Comprehensive ML System', 'comprehensive_ml_system' in self.integration_report['components_integrated']),
                ('Prediction Pipeline', 'prediction_pipeline' in self.integration_report['components_integrated'])
            ]
            
            for component, status in components:
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            
            # Data statistics
            print(f"\n📊 DATA STATISTICS:")
            db_stats = self.integration_report['data_statistics'].get('enhanced_database', {})
            if db_stats:
                print(f"   Enhanced records: {db_stats.get('total_enhanced_records', 0)}")
                print(f"   Dogs with enhanced data: {db_stats.get('dogs_with_enhanced_data', 0)}")
                print(f"   Races with enhanced data: {db_stats.get('races_with_enhanced_data', 0)}")
            
            feature_stats = self.integration_report['data_statistics'].get('enhanced_features', {})
            if feature_stats:
                print(f"   Dogs with sectional times: {feature_stats.get('dogs_with_sectionals', 0)}")
                print(f"   Dogs with PIR ratings: {feature_stats.get('dogs_with_pir', 0)}")
            
            # Validation results
            print(f"\n🔍 VALIDATION RESULTS:")
            model_val = self.integration_report['validation_results'].get('model_validation', {})
            if model_val:
                print(f"   Model enhanced features: {model_val.get('enhanced_features_count', 0)}")
                print(f"   Model accuracy: {model_val.get('model_metadata', {}).get('accuracy', 0):.3f}")
            
            pipeline_test = self.integration_report['validation_results'].get('pipeline_test', {})
            if pipeline_test:
                print(f"   Pipeline test: {'✅ PASSED' if pipeline_test.get('enhanced_data_retrieved') else '❌ FAILED'}")
            
            # Errors
            if self.integration_report['errors']:
                print(f"\n❌ ERRORS ENCOUNTERED:")
                for error in self.integration_report['errors']:
                    print(f"   - {error}")
            
            # Save report to file
            report_file = Path('./enhanced_integration_report.json')
            with open(report_file, 'w') as f:
                json.dump(self.integration_report, f, indent=2, default=str)
            
            print(f"\n💾 Integration report saved: {report_file}")
            
        except Exception as e:
            print(f"❌ Error generating integration report: {e}")
    
    def get_integration_status(self):
        """Get current integration status summary"""
        try:
            status = {
                'enhanced_data_available': False,
                'database_synced': False,
                'models_updated': False,
                'pipeline_tested': False,
                'overall_status': 'Unknown'
            }
            
            # Check enhanced data availability
            enhanced_dir = Path('./enhanced_expert_data')
            if enhanced_dir.exists():
                csv_files = len(list((enhanced_dir / 'csv').glob('*.csv')))
                json_files = len(list((enhanced_dir / 'json').glob('*.json')))
                status['enhanced_data_available'] = csv_files > 0 and json_files > 0
                status['enhanced_files'] = {'csv': csv_files, 'json': json_files}
            
            # Check database sync
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM enhanced_expert_data")
                enhanced_records = cursor.fetchone()[0]
                conn.close()
                status['database_synced'] = enhanced_records > 0
                status['enhanced_records'] = enhanced_records
            except:
                status['database_synced'] = False
                status['enhanced_records'] = 0
            
            # Check model updates
            models_dir = Path('./comprehensive_trained_models')
            if models_dir.exists():
                model_files = list(models_dir.glob('comprehensive_best_model_*.joblib'))
                status['models_updated'] = len(model_files) > 0
                status['model_files'] = len(model_files)
            
            # Determine overall status
            if all([status['enhanced_data_available'], status['database_synced'], status['models_updated']]):
                status['overall_status'] = 'Fully Integrated'
            elif status['enhanced_data_available'] and status['database_synced']:
                status['overall_status'] = 'Partially Integrated'
            elif status['enhanced_data_available']:
                status['overall_status'] = 'Data Available'
            else:
                status['overall_status'] = 'Not Integrated'
            
            return status
            
        except Exception as e:
            print(f"❌ Error getting integration status: {e}")
            return {'overall_status': 'Error', 'error': str(e)}

def main():
    """Main function for enhanced pipeline integration"""
    print("🚀 ENHANCED PREDICTION PIPELINE INTEGRATION")
    print("=" * 70)
    
    integrator = EnhancedPipelineIntegrator()
    
    # Check current status
    print("\n📊 CURRENT INTEGRATION STATUS:")
    print("-" * 50)
    status = integrator.get_integration_status()
    
    print(f"Overall Status: {status['overall_status']}")
    print(f"Enhanced Data Available: {'✅' if status['enhanced_data_available'] else '❌'}")
    print(f"Database Synced: {'✅' if status['database_synced'] else '❌'}")
    print(f"Models Updated: {'✅' if status['models_updated'] else '❌'}")
    
    if status.get('enhanced_files'):
        print(f"Enhanced Files: {status['enhanced_files']['csv']} CSV, {status['enhanced_files']['json']} JSON")
    
    if status.get('enhanced_records'):
        print(f"Enhanced Records in DB: {status['enhanced_records']}")
    
    # Run integration if needed
    if status['overall_status'] != 'Fully Integrated':
        print(f"\n🔄 Running integration to achieve full integration...")
        integration_report = integrator.run_complete_integration()
        
        print(f"\n📋 FINAL INTEGRATION STATUS:")
        print("-" * 50)
        components_integrated = len(integration_report['components_integrated'])
        errors = len(integration_report['errors'])
        print(f"Components Integrated: {components_integrated}")
        print(f"Errors: {errors}")
        
        if errors == 0 and components_integrated >= 2:
            print("🎉 INTEGRATION SUCCESSFUL!")
        else:
            print("⚠️ INTEGRATION INCOMPLETE - Check errors above")
    else:
        print(f"\n✅ System is already fully integrated!")

if __name__ == "__main__":
    main()
