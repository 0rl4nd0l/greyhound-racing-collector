
"""
ML System V3 - Comprehensive Integrated System
==============================================

A completely integrated ML system that combines:
- Real database data analysis
- Weather-enhanced predictions
- GPT analysis integration
- Comprehensive feature engineering
- Multiple prediction methods with ensemble weighting

This is the primary prediction system with basic ML as fallback.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import sqlite3
# Try to import XGBoost, fallback gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, falling back to sklearn models only")

from traditional_analysis import TraditionalRaceAnalyzer

logger = logging.getLogger(__name__)

class MLSystemV3:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.pipeline = None
        self.feature_columns = []
        self.model_info = {}
        self.traditional_analyzer = TraditionalRaceAnalyzer(db_path)
        self._try_load_latest_model()
    
    def train_model(self, model_type='gradient_boosting'):
        """Trains a new model on the latest data from the database."""
        logger.info("🚀 Starting comprehensive ML model training...")
        
        # Load and prepare data
        data = self._load_comprehensive_data()
        if data.empty:
            logger.error("No data loaded, cannot train model.")
            return False
        
        logger.info(f"Loaded {len(data)} race records for training")
        
        # Create features and target
        features, target = self._create_comprehensive_features(data)
        if features.empty:
            logger.error("No features created, cannot train model.")
            return False
        
        logger.info(f"Created {len(features.columns)} features: {features.columns.tolist()}")
        logger.info(f"Target distribution - Wins: {target.sum()}, Losses: {len(target) - target.sum()}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Create model based on type
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, falling back to GradientBoosting")
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                }
            else:
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                }
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20],
                'model__min_samples_split': [2, 5]
            }
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'model__C': [0.1, 1, 10]
            }
        
        # Create and tune pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        self.pipeline = grid_search.best_estimator_
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        
        # Evaluate model
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        if hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            importances = self.pipeline.named_steps['model'].feature_importances_
            feature_importance = sorted(zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 Feature Importances:")
            for feature, importance in feature_importance[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
        
        logger.info(f"Model Performance:")
        logger.info(f"  Training Accuracy: {train_score:.4f}")
        logger.info(f"  Test Accuracy: {test_score:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        # Save model info
        self.feature_columns = features.columns.tolist()
        self.model_info = {
            'model_type': model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'n_features': len(features.columns),
            'n_samples': len(features),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save model
        model_path = self._save_model()
        logger.info(f"✅ Model training completed successfully! Saved to {model_path}")
        
        return True
    
    def predict(self, dog_data):
        """Makes a prediction using the trained model."""
        if not self.pipeline:
            logger.warning("No model loaded, cannot make prediction.")
            return {
                'win_probability': 0.5,
                'confidence': 0.0,
                'model_info': 'No model loaded'
            }
        
        try:
            # Extract features for this dog
            features = self._extract_features_for_prediction(dog_data)
            
            # Create DataFrame and align columns
            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
            
            # Make prediction
            win_prob = self.pipeline.predict_proba(features_df)[0, 1]
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(features)
            
            return {
                'win_probability': float(win_prob),
                'confidence': float(confidence),
                'model_info': self.model_info.get('model_type', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'win_probability': 0.5,
                'confidence': 0.0,
                'model_info': f'Prediction error: {str(e)}'
            }
    
    def get_model_info(self):
        """Returns information about the current model."""
        return self.model_info

    def _load_data_from_db(self):
        """Loads the dog race data from the SQLite database using unified schema."""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            # Load all race data with race metadata using unified schema
            query = """
            SELECT 
                drd.*,
                rm.venue, rm.race_date, rm.distance, rm.grade, rm.weather,
                rm.temperature, rm.humidity, rm.track_condition
            FROM dog_race_data drd
            LEFT JOIN race_metadata rm ON drd.race_id = rm.race_id
            WHERE drd.finish_position IS NOT NULL 
                AND drd.finish_position != '' 
                AND drd.finish_position != 'N/A'
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Loaded {len(df)} records from unified database schema.")
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()

    def _feature_engineering(self, data):
        """Creates features from the raw data."""
        # Basic features
        features = pd.DataFrame()
        features['box_number'] = data['box_number']
        features['weight'] = data['weight']
        features['starting_price'] = data['starting_price']
        features['individual_time'] = pd.to_numeric(data['individual_time'], errors='coerce')
        
        # Target variable (did the dog win?)
        target = (data['finish_position'] == 1).astype(int)

        # Fill missing values
        features.fillna(features.median(), inplace=True)

        logger.info(f"Created {len(features.columns)} features.")
        return features, target

    def _load_comprehensive_data(self):
        """Loads comprehensive data from database with race metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Join dog race data with race metadata for richer features
            query = """
            SELECT 
                d.*,
                r.venue,
                r.grade,
                r.distance,
                r.track_condition,
                r.weather,
                r.temperature,
                r.humidity,
                r.wind_speed,
                r.field_size,
                r.race_date
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            WHERE d.finish_position IS NOT NULL
            AND d.finish_position != ''
            AND d.individual_time IS NOT NULL
            AND d.individual_time != ''
            ORDER BY r.race_date DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} comprehensive race records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading comprehensive data: {e}")
            return pd.DataFrame()
    
    def _create_comprehensive_features(self, data):
        """Creates comprehensive features from race data."""
        logger.info("Creating comprehensive feature set...")
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Add traditional analysis features
        try:
            traditional_features = pd.DataFrame(index=data.index)
            for idx, row in data.iterrows():
                try:
                    dog_factors = self.traditional_analyzer.analyze_dog(row['dog_clean_name'], row.to_dict())
                    trad_features = self.traditional_analyzer._extract_ml_features(dog_factors, row.to_dict())
                    for key, value in trad_features.items():
                        traditional_features.loc[idx, key] = value
                except Exception as e:
                    logger.debug(f"Error getting traditional features for {row.get('dog_clean_name', 'unknown')}: {e}")
                    # Fill with defaults
                    default_features = {
                        'traditional_overall_score': 0.35,
                        'traditional_performance_score': 0.3,
                        'traditional_form_score': 0.3,
                        'traditional_class_score': 0.5,
                        'traditional_consistency_score': 0.3,
                        'traditional_fitness_score': 0.5,
                        'traditional_experience_score': 0.2,
                        'traditional_trainer_score': 0.5,
                        'traditional_track_condition_score': 0.5,
                        'traditional_distance_score': 0.5,
                        'traditional_confidence_level': 0.2,
                        'traditional_key_factors_count': 0,
                        'traditional_risk_factors_count': 2
                    }
                    for key, value in default_features.items():
                        traditional_features.loc[idx, key] = value
            
            # Fill any remaining NaN values
            traditional_features = traditional_features.fillna(0.3)
            features = pd.concat([features, traditional_features], axis=1)
        except Exception as e:
            logger.warning(f"Error adding traditional features: {e}. Skipping traditional analysis.")
        
        # Basic performance features
        features['box_number'] = pd.to_numeric(data['box_number'], errors='coerce')
        features['weight'] = pd.to_numeric(data['weight'], errors='coerce')
        features['starting_price'] = pd.to_numeric(data['starting_price'], errors='coerce')
        
        # Time-based features
        individual_times = pd.to_numeric(data['individual_time'], errors='coerce')
        features['individual_time'] = individual_times
        
        # Convert finish position to numeric and create target
        finish_pos = pd.to_numeric(data['finish_position'], errors='coerce')
        target = (finish_pos == 1).astype(int)
        
        # Race characteristics
        features['field_size'] = pd.to_numeric(data['field_size'], errors='coerce')
        features['temperature'] = pd.to_numeric(data['temperature'], errors='coerce')
        features['humidity'] = pd.to_numeric(data['humidity'], errors='coerce')
        features['wind_speed'] = pd.to_numeric(data['wind_speed'], errors='coerce')
        
        # Categorical features (encoded)
        venue_dummies = pd.get_dummies(data['venue'], prefix='venue', dummy_na=True)
        grade_dummies = pd.get_dummies(data['grade'], prefix='grade', dummy_na=True)
        track_condition_dummies = pd.get_dummies(data['track_condition'], prefix='track', dummy_na=True)
        
        # Limit dummy variables to prevent feature explosion
        features = pd.concat([
            features,
            venue_dummies.iloc[:, :10],  # Top 10 venues
            grade_dummies.iloc[:, :8],   # Top 8 grades
            track_condition_dummies.iloc[:, :5]  # Top 5 track conditions
        ], axis=1)
        
        # Derived features
        features['price_rank'] = features['starting_price'].groupby(data['race_id']).rank(method='dense')
        features['weight_rank'] = features['weight'].groupby(data['race_id']).rank(method='dense')
        features['time_rank_in_race'] = features['individual_time'].groupby(data['race_id']).rank(method='dense')
        features['box_advantage'] = (features['box_number'] <= 3).astype(int) - (features['box_number'] >= 6).astype(int)
        features['is_favorite'] = (features['price_rank'] == 1).astype(int)
        
        # Performance ratios
        features['weight_to_field_ratio'] = features['weight'] / features['field_size']
        features['price_to_field_ratio'] = features['starting_price'] / features['field_size']

        # Interaction features
        features['box_x_price_rank'] = features['box_number'] * features['price_rank']
        features['weight_x_box'] = features['weight'] * features['box_number']
        
        # Fill missing values with median/mode
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        
        # Remove rows where target is NaN
        valid_mask = ~target.isna()
        features = features[valid_mask]
        target = target[valid_mask]
        
        logger.info(f"Created {len(features.columns)} features from {len(features)} valid samples")
        logger.info(f"Feature columns: {features.columns.tolist()}")
        
        return features, target
    
    def _extract_features_for_prediction(self, dog_data):
        """Extracts features for a single dog prediction."""
        features = {}
        
        # Basic features with proper type conversion
        features['box_number'] = float(dog_data.get('box_number', 4))
        features['weight'] = float(dog_data.get('weight', 30.0))
        
        # Handle starting_price properly (could be string)
        starting_price = dog_data.get('starting_price', 10.0)
        if isinstance(starting_price, str):
            try:
                starting_price = float(starting_price.replace('$', '').replace(',', ''))
            except (ValueError, AttributeError):
                starting_price = 10.0
        features['starting_price'] = float(starting_price)
        
        # Handle individual_time properly (could be string)
        individual_time = dog_data.get('individual_time', 30.0)
        if isinstance(individual_time, str):
            try:
                individual_time = float(individual_time)
            except (ValueError, AttributeError):
                individual_time = 30.0
        features['individual_time'] = float(individual_time)
        
        features['field_size'] = float(dog_data.get('field_size', 8))
        features['temperature'] = float(dog_data.get('temperature', 20.0))
        features['humidity'] = float(dog_data.get('humidity', 60.0))
        features['wind_speed'] = float(dog_data.get('wind_speed', 10.0))
        
        # Derived features
        features['price_rank'] = 1.0  # Would need race context for real rank
        features['weight_rank'] = 1.0
        features['box_advantage'] = float(int(features['box_number'] <= 3))
        features['is_favorite'] = float(int(features['starting_price'] <= 3.0))
        
        # Add traditional features
        try:
            dog_stats = self.traditional_analyzer.analyze_dog(dog_data['name'], dog_data)
            traditional_features = self.traditional_analyzer._extract_ml_features(dog_stats, dog_data)
            features.update(traditional_features)
        except Exception as e:
            logger.debug(f"Error getting traditional features for {dog_data.get('name', 'unknown')}: {e}")
            # Add default traditional features
            default_traditional = {
                'traditional_overall_score': 0.35,
                'traditional_performance_score': 0.3,
                'traditional_form_score': 0.3,
                'traditional_class_score': 0.5,
                'traditional_consistency_score': 0.3,
                'traditional_fitness_score': 0.5,
                'traditional_experience_score': 0.2,
                'traditional_trainer_score': 0.5,
                'traditional_track_condition_score': 0.5,
                'traditional_distance_score': 0.5,
                'traditional_confidence_level': 0.2,
                'traditional_key_factors_count': 0,
                'traditional_risk_factors_count': 2
            }
            features.update(default_traditional)
        features['weight_to_field_ratio'] = features['weight'] / features['field_size']
        features['price_to_field_ratio'] = features['starting_price'] / features['field_size']
        
        # Categorical features - set defaults
        for col in self.feature_columns:
            if col.startswith(('venue_', 'grade_', 'track_')):
                features[col] = 0  # Default to 0 for dummy variables
        
        return features
    
    def _calculate_confidence(self, features):
        """Calculates prediction confidence based on feature quality."""
        # Simple confidence based on completeness of features
        non_zero_features = sum(1 for v in features.values() if v != 0)
        total_features = len(features)
        return min(0.95, non_zero_features / total_features)
    
    def _save_model(self):
        """Saves the trained model pipeline to a file."""
        model_dir = Path('./ml_models_v3')
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"ml_model_v3_{timestamp}.joblib"
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'model_info': self.model_info,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def _try_load_latest_model(self):
        """Attempts to load the latest trained model."""
        model_dir = Path('./ml_models_v3')
        if not model_dir.exists():
            logger.info("No model directory found, will need to train new model")
            return
        
        # Find the latest model file
        model_files = list(model_dir.glob('ml_model_v3_*.joblib'))
        if not model_files:
            logger.info("No trained models found, will need to train new model")
            return
        
        # Load the latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            model_data = joblib.load(latest_model)
            self.pipeline = model_data.get('pipeline')
            self.feature_columns = model_data.get('feature_columns', [])
            self.model_info = model_data.get('model_info', {})
            
            logger.info(f"Loaded model from {latest_model}")
            logger.info(f"Model info: {self.model_info}")
            
        except Exception as e:
            logger.error(f"Error loading model from {latest_model}: {e}")
            self.pipeline = None

# Function for frontend training calls
def train_new_model(model_type='gradient_boosting'):
    """Train a new ML model - can be called from frontend"""
    try:
        ml_system = MLSystemV3()
        success = ml_system.train_model(model_type)
        
        if success:
            return {
                'success': True,
                'message': 'Model trained successfully',
                'model_info': ml_system.get_model_info()
            }
        else:
            return {
                'success': False,
                'message': 'Model training failed'
            }
    except Exception as e:
        return {
            'success': False,
            'message': f'Training error: {str(e)}'
        }

