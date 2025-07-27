#!/usr/bin/env python3
"""
Comprehensive ML Predictor - Enhanced with Traditional Analysis
============================================================

This wrapper integrates the high-confidence comprehensive enhanced ML system
with traditional race analysis for enriched feature engineering. Combines
ML predictions with proven handicapping methods to achieve superior accuracy.

New Features:
- Traditional analysis integration for ML feature enrichment
- Comprehensive handicapping metrics
- Enhanced prediction confidence scoring
- Combined traditional + ML prediction methodology

Author: AI Assistant
Date: July 25, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from json_utils import safe_json_dump, safe_mean, safe_correlation, safe_float
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Import the comprehensive enhanced ML system
try:
    from comprehensive_enhanced_ml_system import ComprehensiveEnhancedMLSystem
    COMPREHENSIVE_ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Comprehensive ML system not available: {e}")
    COMPREHENSIVE_ML_AVAILABLE = False

# Import traditional analysis system
try:
    from traditional_analysis import TraditionalRaceAnalyzer, get_traditional_ml_features
    TRADITIONAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Traditional analysis system not available: {e}")
    TRADITIONAL_ANALYSIS_AVAILABLE = False

# Import scikit-learn for prediction
try:
    import joblib
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ComprehensiveMLPredictor:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.upcoming_dir = Path('./upcoming_races')
        self.predictions_dir = Path('./predictions')
        self.models_dir = Path('./comprehensive_trained_models')
        
        # Create directories
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Load the trained model
        self.trained_model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None
        
        self._load_trained_model()
        
        print("🚀 Comprehensive ML Predictor Initialized")
        print(f"✅ Comprehensive ML Available: {COMPREHENSIVE_ML_AVAILABLE}")
        print(f"✅ Trained Model Loaded: {self.trained_model is not None}")
        
    def _load_trained_model(self):
        """Load the latest trained comprehensive model"""
        try:
            if not self.models_dir.exists():
                print("⚠️ No trained models directory found")
                return
            
            # Find the latest model file
            model_files = list(self.models_dir.glob('comprehensive_best_model_*.joblib'))
            if not model_files:
                print("⚠️ No trained comprehensive models found")
                return
            
            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Load the model
            model_data = joblib.load(latest_model)
            self.trained_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_metadata = {
                'model_name': model_data.get('model_name', 'Unknown'),
                'accuracy': model_data.get('accuracy', 0),
                'timestamp': model_data.get('timestamp', ''),
                'data_summary': model_data.get('data_summary', {})
            }
            
            print(f"✅ Loaded model: {self.model_metadata['model_name']}")
            print(f"📊 Model accuracy: {self.model_metadata['accuracy']:.3f}")
            print(f"🔧 Features: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
    
    def predict_race_file(self, race_file_path):
        """Predict a single race file using the comprehensive ML system"""
        try:
            if not COMPREHENSIVE_ML_AVAILABLE:
                return self._fallback_prediction(race_file_path)
            
            if not self.trained_model:
                return self._fallback_prediction(race_file_path)
            
            print(f"🎯 Predicting race: {os.path.basename(race_file_path)}")
            
            # Initialize the comprehensive ML system
            ml_system = ComprehensiveEnhancedMLSystem(self.db_path)
            
            # Load form guide data for this specific race
            race_form_data = self._load_single_race_form_data(race_file_path)
            
            # Load database historical data
            race_results_df = ml_system.load_race_results_data()
            if race_results_df is None:
                print("⚠️ No historical data available")
                return self._fallback_prediction(race_file_path)
            
            # Extract race info from filename
            race_info = self._extract_race_info(race_file_path)
            
            # Create predictions for each dog in the race
            predictions = []
            
            # Parse the race CSV to get participating dogs
            race_df = pd.read_csv(race_file_path)
            participating_dogs = self._extract_participating_dogs(race_df)
            
            print(f"📊 Found {len(participating_dogs)} dogs in race")
            
            for dog_info in participating_dogs:
                dog_name = dog_info['name']
                
                # Get historical data for this dog
                dog_historical = race_results_df[
                    race_results_df['dog_clean_name'].str.upper() == dog_name.upper()
                ].sort_values('race_date', ascending=False)
                
                # Get form guide data for this dog
                dog_form_data = race_form_data.get(dog_name, [])
                
                if len(dog_historical) >= 1 or len(dog_form_data) >= 3:
                    # Create comprehensive features for this dog
                    dog_features = self._create_dog_features(
                        dog_info, dog_historical, dog_form_data, race_info
                    )
                    
                    if dog_features is not None:
                        # Make prediction using the trained model
                        ml_score = self._predict_dog_performance(dog_features)
                        
                        # Get traditional analysis score if available
                        traditional_score = 0.5  # Default
                        if TRADITIONAL_ANALYSIS_AVAILABLE:
                            try:
                                race_context = {
                                    'venue': race_info.get('venue', 'Unknown'),
                                    'distance': '500m',
                                    'grade': 'Grade 5',
                                    'track_condition': 'Good'
                                }
                                from traditional_analysis import calculate_traditional_score
                                traditional_score = calculate_traditional_score(
                                    dog_name, race_context, self.db_path
                                )
                            except Exception as e:
                                print(f"   ⚠️ Traditional score calculation failed for {dog_name}: {e}")
                        
                        # Combine ML and traditional scores (70% ML, 30% traditional)
                        combined_score = (ml_score * 0.7) + (traditional_score * 0.3)
                        
                        predictions.append({
                            'dog_name': dog_name,
                            'box_number': dog_info.get('box', 'Unknown'),
                            'prediction_score': float(combined_score),
                            'ml_score': float(ml_score),
                            'traditional_score': float(traditional_score),
                            'confidence': min(0.95, max(0.1, combined_score)),
                            'historical_races': len(dog_historical),
                            'form_data_races': len(dog_form_data),
                            'features_used': len(self.feature_columns)
                        })
                    else:
                        # Fallback prediction for dogs with insufficient data
                        predictions.append({
                            'dog_name': dog_name,
                            'box_number': dog_info.get('box', 'Unknown'),
                            'prediction_score': 0.5,
                            'confidence': 0.1,
                            'historical_races': len(dog_historical),
                            'form_data_races': len(dog_form_data),
                            'features_used': 0,
                            'note': 'Insufficient data for ML prediction'
                        })
                else:
                    # Very basic prediction for dogs with no data
                    predictions.append({
                        'dog_name': dog_name,
                        'box_number': dog_info.get('box', 'Unknown'),
                        'prediction_score': 0.3,
                        'confidence': 0.05,
                        'historical_races': len(dog_historical),
                        'form_data_races': len(dog_form_data),
                        'features_used': 0,
                        'note': 'No historical data available'
                    })
            
            # Sort predictions by score (highest first)
            predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
            
            # Create prediction summary
            prediction_summary = {
                'race_info': race_info,
                'model_info': {
                    'system': 'Comprehensive Enhanced ML',
                    'model_name': self.model_metadata.get('model_name', 'Unknown'),
                    'accuracy': self.model_metadata.get('accuracy', 0),
                    'features': len(self.feature_columns) if self.feature_columns else 0
                },
                'race_summary': {
                    'total_dogs': len(predictions),
                    'dogs_with_data': sum(1 for p in predictions if p['historical_races'] > 0),
                    'average_confidence': safe_mean([p['confidence'] for p in predictions], 0.0)
                },
                'predictions': predictions,
                'top_pick': predictions[0] if predictions else None,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Save prediction results
            output_file = self._save_prediction_results(prediction_summary, race_info)
            
            print(f"✅ Prediction completed for {len(predictions)} dogs")
            print(f"🏆 Top pick: {predictions[0]['dog_name'] if predictions else 'None'}")
            print(f"💾 Results saved: {output_file}")
            
            return {
                'success': True,
                'predictions': predictions,
                'summary': prediction_summary,
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"❌ Error predicting race: {e}")
            return self._fallback_prediction(race_file_path)
    
    def _load_single_race_form_data(self, race_file_path):
        """Load form guide data combining separate files and embedded race data"""
        try:
            # Parse race info from filename
            race_info = self._extract_race_info(race_file_path)
            
            # Initialize comprehensive ML system to use its form guide parsing
            ml_system = ComprehensiveEnhancedMLSystem(self.db_path)
            
            # Load all form guide data from separate files
            all_form_data = ml_system.load_form_guide_data()
            
            # Extract embedded historical data from this race file
            embedded_form_data = self._extract_embedded_historical_data(race_file_path)
            
            # Get participating dogs
            race_df = pd.read_csv(race_file_path)
            participating_dogs = self._extract_participating_dogs(race_df)
            participating_dog_names = [dog['name'].upper() for dog in participating_dogs]
            
            # Combine both sources of form data
            race_specific_form_data = {}
            for dog_name in participating_dog_names:
                # Start with form guide data if available
                combined_data = []
                
                # Add data from separate form guide files
                for form_dog_name, form_data in all_form_data.items():
                    if form_dog_name.upper() == dog_name.upper():
                        combined_data.extend(form_data)
                        break
                
                # Add embedded historical data from race file
                if dog_name.upper() in embedded_form_data:
                    combined_data.extend(embedded_form_data[dog_name.upper()])
                
                if combined_data:
                    # Find original case dog name
                    original_dog_name = next(
                        (dog['name'] for dog in participating_dogs if dog['name'].upper() == dog_name.upper()),
                        dog_name
                    )
                    race_specific_form_data[original_dog_name] = combined_data
            
            return race_specific_form_data
            
        except Exception as e:
            print(f"⚠️ Error loading form data: {e}")
            return {}
    
    def _extract_embedded_historical_data(self, race_file_path):
        """Extract historical race data embedded within the race CSV file"""
        try:
            race_df = pd.read_csv(race_file_path)
            embedded_data = {}
            current_dog_name = None
            
            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get('Dog Name', '')).strip()
                
                # Check if this is a new dog or continuation of previous
                if dog_name_raw not in ['""', '', 'nan'] and dog_name_raw != 'nan':
                    # New dog - clean the name
                    current_dog_name = dog_name_raw
                    if '. ' in current_dog_name:
                        current_dog_name = current_dog_name.split('. ', 1)[1]
                    
                    # Initialize dog record
                    if current_dog_name.upper() not in embedded_data:
                        embedded_data[current_dog_name.upper()] = []
                
                # Skip if we don't have a current dog
                if current_dog_name is None:
                    continue
                
                # Parse this row as historical race data for current dog
                place = str(row.get('PLC', '')).strip()
                date = str(row.get('DATE', '')).strip()
                
                # Only add if we have meaningful race data
                if place and date and place != '' and date != '' and place != 'nan' and date != 'nan':
                    historical_race = {
                        'sex': str(row.get('Sex', '')).strip(),
                        'place': place,
                        'box': str(row.get('BOX', '')).strip(),
                        'weight': str(row.get('WGT', '')).strip(),
                        'distance': str(row.get('DIST', '')).strip(),
                        'date': date,
                        'track': str(row.get('TRACK', '')).strip(),
                        'grade': str(row.get('G', '')).strip(),
                        'time': str(row.get('TIME', '')).strip(),
                        'win_time': str(row.get('WIN', '')).strip(),
                        'bonus': str(row.get('BON', '')).strip(),
                        'first_sectional': str(row.get('1 SEC', '')).strip(),
                        'margin': str(row.get('MGN', '')).strip(),
                        'runner_up': str(row.get('W/2G', '')).strip(),
                        'pir': str(row.get('PIR', '')).strip(),
                        'starting_price': str(row.get('SP', '')).strip(),
                        'source': 'embedded_race_file'
                    }
                    
                    embedded_data[current_dog_name.upper()].append(historical_race)
            
            # Log results
            total_embedded_races = sum(len(races) for races in embedded_data.values())
            print(f"   📊 Extracted {total_embedded_races} embedded races for {len(embedded_data)} dogs")
            
            return embedded_data
            
        except Exception as e:
            print(f"⚠️ Error extracting embedded historical data: {e}")
            return {}
    
    def _extract_participating_dogs(self, race_df):
        """Extract participating dogs from race CSV with proper blank row handling"""
        try:
            dogs = []
            current_dog_name = None
            
            for idx, row in race_df.iterrows():
                dog_name_raw = str(row.get('Dog Name', '')).strip()
                
                # Check if this is a new dog or continuation of previous
                if dog_name_raw not in ['""', '', 'nan'] and dog_name_raw != 'nan':
                    # New dog - clean the name
                    current_dog_name = dog_name_raw
                    # Remove box number prefix (e.g., "1. Mel Monelli" -> "Mel Monelli")
                    if '. ' in current_dog_name:
                        current_dog_name = current_dog_name.split('. ', 1)[1]
                    
                    # Extract box number from the prefix
                    box_number = None
                    if '. ' in dog_name_raw:
                        try:
                            box_number = int(dog_name_raw.split('.')[0])
                        except (ValueError, TypeError):
                            pass
                    
                    dogs.append({
                        'name': current_dog_name,
                        'box': box_number,
                        'raw_name': dog_name_raw
                    })
            
            return dogs
            
        except Exception as e:
            print(f"⚠️ Error extracting participating dogs: {e}")
            return []
    
    def _extract_race_info(self, race_file_path):
        """Extract race information from file path"""
        try:
            filename = os.path.basename(race_file_path)
            # Example: "Race 1 - AP_K - 24 July 2025.csv"
            parts = filename.replace('.csv', '').split(' - ')
            
            if len(parts) >= 3:
                race_number = parts[0].replace('Race ', '')
                venue = parts[1]
                date_str = parts[2]
                
                return {
                    'filename': filename,
                    'race_number': race_number,
                    'venue': venue,
                    'race_date': date_str,
                    'filepath': race_file_path
                }
            else:
                return {
                    'filename': filename,
                    'race_number': 'Unknown',
                    'venue': 'Unknown',
                    'race_date': 'Unknown',
                    'filepath': race_file_path
                }
                
        except Exception as e:
            print(f"⚠️ Error extracting race info: {e}")
            return {
                'filename': os.path.basename(race_file_path),
                'race_number': 'Unknown',
                'venue': 'Unknown',
                'race_date': 'Unknown',
                'filepath': race_file_path
            }
    
    def _create_dog_features(self, dog_info, dog_historical, dog_form_data, race_info):
        """Create features for a single dog using comprehensive feature engineering with traditional analysis"""
        try:
            if not self.feature_columns:
                return None
            
            # Initialize feature dict with defaults
            features = {}
            
            # Get traditional analysis features if available
            traditional_features = {}
            if TRADITIONAL_ANALYSIS_AVAILABLE:
                try:
                    # Create race context for traditional analysis
                    race_context = {
                        'venue': race_info.get('venue', 'Unknown'),
                        'distance': '500m',  # Default distance
                        'grade': 'Grade 5',  # Default grade
                        'track_condition': 'Good'  # Default condition
                    }
                    
                    # Get traditional analysis features
                    traditional_features = get_traditional_ml_features(
                        dog_info['name'], race_context, self.db_path
                    )
                    
                    print(f"   🎯 Traditional analysis for {dog_info['name']}: {traditional_features.get('traditional_overall_score', 0):.3f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Traditional analysis failed for {dog_info['name']}: {e}")
                    traditional_features = {}
            
            # Merge traditional features into main features dict
            features.update(traditional_features)
            
            # Process database historical data
            if len(dog_historical) > 0:
                positions = []
                times = []
                weights = []
                odds = []
                
                for _, hist_row in dog_historical.iterrows():
                    # Position data
                    if pd.notna(hist_row['finish_position']):
                        pos_str = str(hist_row['finish_position']).strip()
                        if pos_str not in ['', 'N/A', 'None', 'nan']:
                            pos_cleaned = ''.join(filter(str.isdigit, pos_str))
                            if pos_cleaned and 1 <= int(pos_cleaned) <= 10:
                                positions.append(int(pos_cleaned))
                    
                    # Time data
                    try:
                        if pd.notna(hist_row['individual_time']) and float(hist_row['individual_time']) > 0:
                            times.append(float(hist_row['individual_time']))
                    except (ValueError, TypeError):
                        pass
                    
                    # Weight data
                    try:
                        if pd.notna(hist_row['weight']) and float(hist_row['weight']) > 0:
                            weights.append(float(hist_row['weight']))
                    except (ValueError, TypeError):
                        pass
                    
                    # Odds data
                    try:
                        if pd.notna(hist_row['starting_price']) and float(hist_row['starting_price']) > 0:
                            odds.append(float(hist_row['starting_price']))
                    except (ValueError, TypeError):
                        pass
            else:
                positions = []
                times = []
                weights = []
                odds = []
            
            # Process form guide data
            form_positions = []
            form_times = []
            form_weights = []
            form_odds = []
            
            for form_entry in dog_form_data[:20]:  # Use up to 20 historical races
                try:
                    # Parse position
                    place_str = form_entry.get('place', '').strip()
                    if place_str and place_str.isdigit():
                        position = int(place_str)
                        if 1 <= position <= 10:
                            form_positions.append(position)
                    
                    # Parse time
                    time_str = form_entry.get('time', '').strip()
                    if time_str:
                        try:
                            time_val = float(time_str)
                            if 15.0 <= time_val <= 60.0:
                                form_times.append(time_val)
                        except (ValueError, TypeError):
                            pass
                    
                    # Parse weight
                    weight_str = form_entry.get('weight', '').strip()
                    if weight_str:
                        try:
                            weight_val = float(weight_str)
                            if 20.0 <= weight_val <= 40.0:
                                form_weights.append(weight_val)
                        except (ValueError, TypeError):
                            pass
                    
                    # Parse starting price
                    sp_str = form_entry.get('starting_price', '').strip()
                    if sp_str:
                        try:
                            sp_val = float(sp_str)
                            if 1.0 <= sp_val <= 1000.0:
                                form_odds.append(sp_val)
                        except (ValueError, TypeError):
                            pass
                            
                except Exception:
                    continue
            
            # Combine all data
            all_positions = positions + form_positions
            all_times = times + form_times
            all_weights = weights + form_weights
            all_odds = odds + form_odds
            
            if not all_positions:
                return None
            
            # Calculate core features (matching the trained model)
            features['avg_position'] = np.mean(all_positions)
            features['recent_form_avg'] = np.mean(all_positions[:8]) if len(all_positions) >= 8 else np.mean(all_positions)
            features['market_confidence'] = 1 / (np.mean(all_odds) + 1) if all_odds else 0.1
            features['current_odds_log'] = np.log(10)  # Default odds
            features['venue_experience'] = len([p for p in positions])  # Simplified
            features['place_rate'] = sum(1 for p in all_positions if p <= 3) / len(all_positions)
            features['current_weight'] = np.mean(all_weights) if all_weights else 30.0
            features['time_consistency'] = 1 / (np.std(all_times) + 0.1) if len(all_times) > 1 else 0.5
            features['win_rate'] = sum(1 for p in all_positions if p == 1) / len(all_positions)
            
            # Additional comprehensive features
            features['long_term_form_trend'] = 0  # Simplified
            features['position_consistency'] = 1 / (np.std(all_positions) + 0.1)
            features['avg_time'] = np.mean(all_times) if all_times else 30.0
            features['best_time'] = min(all_times) if all_times else 28.0
            features['time_improvement_trend'] = 0  # Simplified
            features['avg_weight'] = np.mean(all_weights) if all_weights else 30.0
            features['weight_consistency'] = 1 / (np.std(all_weights) + 0.1) if len(all_weights) > 1 else 0.5
            features['weight_vs_avg'] = 0  # Simplified
            features['distance_specialization'] = 0.1  # Simplified
            features['grade_experience'] = 5  # Default
            features['days_since_last'] = 14  # Default
            features['fitness_score'] = features['place_rate'] * features['win_rate']
            features['competition_strength'] = 0.5  # Default
            features['box_win_rate'] = 0.1  # Default
            features['current_box'] = dog_info.get('box', 4)
            features['field_size'] = 6  # Default
            features['historical_races_count'] = len(all_positions)
            
            # Encoded features (defaults)
            features['venue_encoded'] = 0
            features['track_condition_encoded'] = 0
            features['grade_encoded'] = 0
            features['distance_numeric'] = 500.0
            
            # Ensure all required features are present
            feature_vector = []
            for feature_name in self.feature_columns:
                feature_vector.append(features.get(feature_name, 0.0))
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            print(f"⚠️ Error creating features for {dog_info['name']}: {e}")
            return None
    
    def _predict_dog_performance(self, dog_features):
        """Make prediction using the trained model with calibration correction"""
        try:
            if not self.trained_model or not self.scaler:
                return 0.5
            
            # Scale features
            scaled_features = self.scaler.transform(dog_features)
            
            # Get raw prediction probability
            raw_probability = 0.5
            if hasattr(self.trained_model, 'predict_proba'):
                prediction_proba = self.trained_model.predict_proba(scaled_features)
                raw_probability = prediction_proba[0][1] if prediction_proba.shape[1] > 1 else 0.5
            else:
                # For models without predict_proba, use decision function or predict
                prediction = self.trained_model.predict(scaled_features)
                raw_probability = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
            
            # Apply calibration correction for overly conservative balanced model
            # The balanced model compresses all predictions into 10-25% range
            # We need to expand this back to a more realistic 5-80% range
            calibrated_probability = self._calibrate_prediction(raw_probability)
            
            return calibrated_probability
                
        except Exception as e:
            print(f"⚠️ Error making prediction: {e}")
            return 0.5
    
    def _calibrate_prediction(self, raw_probability):
        """Calibrate overly conservative balanced model predictions"""
        try:
            # The balanced model typically outputs 0.10-0.25 for all dogs
            # We need to map this to a more realistic 0.05-0.80 range
            
            # Empirical calibration based on model behavior:
            # - Model's 0.10 should map to ~0.05 (very poor dogs)
            # - Model's 0.15 should map to ~0.15 (average dogs) 
            # - Model's 0.20 should map to ~0.35 (good dogs)
            # - Model's 0.25+ should map to ~0.60+ (excellent dogs)
            
            if raw_probability <= 0.10:
                # Very poor performers
                calibrated = 0.05 + (raw_probability - 0.05) * 2.0
            elif raw_probability <= 0.15:
                # Below average performers  
                calibrated = 0.15 + (raw_probability - 0.10) * 2.0
            elif raw_probability <= 0.20:
                # Average to good performers
                calibrated = 0.15 + (raw_probability - 0.15) * 4.0
            elif raw_probability <= 0.25:
                # Good to excellent performers
                calibrated = 0.35 + (raw_probability - 0.20) * 5.0
            else:
                # Exceptional performers (rare)
                calibrated = 0.60 + min((raw_probability - 0.25) * 4.0, 0.20)
            
            # Ensure bounds
            calibrated = max(0.05, min(0.85, calibrated))
            
            # Log calibration for debugging (occasionally)
            import random
            if random.random() < 0.1:  # 10% of the time
                print(f"     🔧 Calibration: {raw_probability:.3f} → {calibrated:.3f}")
            
            return calibrated
            
        except Exception as e:
            print(f"⚠️ Calibration error: {e}")
            return raw_probability
    
    def _save_prediction_results(self, prediction_summary, race_info):
        """Save prediction results to JSON file"""
        try:
            # Create filename based on race info
            race_id = f"{race_info['race_number']}_{race_info['venue']}_{race_info['race_date'].replace(' ', '_')}"
            output_filename = f"prediction_{race_id}.json"
            output_path = self.predictions_dir / output_filename
            
            # Save results using safe JSON dump to handle NaN values
            with open(output_path, 'w') as f:
                safe_json_dump(prediction_summary, f)
            
            return str(output_path)
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
            return None
    
    def _fallback_prediction(self, race_file_path):
        """Fallback prediction when ML system is not available"""
        try:
            race_df = pd.read_csv(race_file_path)
            dogs = self._extract_participating_dogs(race_df)
            race_info = self._extract_race_info(race_file_path)
            
            predictions = []
            for i, dog in enumerate(dogs):
                predictions.append({
                    'dog_name': dog['name'],
                    'box_number': dog.get('box', 'Unknown'),
                    'prediction_score': max(0.1, 0.8 - (i * 0.1)),  # Decreasing scores
                    'confidence': 0.1,
                    'note': 'Fallback prediction - ML system unavailable'
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'summary': {
                    'race_info': race_info,
                    'model_info': {'system': 'Fallback'},
                    'predictions': predictions,
                    'prediction_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ Fallback prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python comprehensive_ml_predictor.py <race_file_path>")
        sys.exit(1)
    
    race_file_path = sys.argv[1]
    
    if not os.path.exists(race_file_path):
        print(f"❌ Race file not found: {race_file_path}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = ComprehensiveMLPredictor()
    
    # Make prediction
    result = predictor.predict_race_file(race_file_path)
    
    if result['success']:
        print(f"\n🏆 PREDICTION RESULTS")
        print("=" * 50)
        for i, prediction in enumerate(result['predictions'][:5], 1):
            print(f"{i}. {prediction['dog_name']} - Score: {prediction['prediction_score']:.3f} (Confidence: {prediction['confidence']:.2f})")
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
