#!/usr/bin/env python3
"""
Debug Prediction Failure Harness
=================================

Reproduces the prediction system failure with controlled environment and detailed logging.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Set controlled environment
os.environ['TESTING'] = 'true'
os.environ['DISABLE_FEATURE_CACHE'] = '1'
os.environ['TGR_ENABLED'] = '0'
os.environ['GREYHOUND_LOOKBACK_DAYS'] = '365'

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./debug_artifacts/v4/logs.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_debug_race_data():
    """Create test race data with proper column mapping."""
    logger.info("Creating debug race data...")
    
    # Load the test CSV
    data = pd.read_csv('Race 1 - SAND - 25 August 2025.csv')
    
    # Map to expected column names
    data['dog_clean_name'] = data['Dog Name'].str.extract(r'^\d+\.\s*(.+)')[0]
    data['box_number'] = data['BOX']
    data['weight'] = data['WGT']
    data['trainer_name'] = data['TRAINER']  
    data['race_date'] = '2025-08-25'
    data['venue'] = 'SAND'
    data['grade'] = data['G'] 
    data['distance'] = data['DIST']
    data['track_condition'] = 'Good'
    data['weather'] = 'Fine'

    # Keep only mapped columns
    mapped_data = data[['dog_clean_name', 'box_number', 'weight', 'trainer_name', 
                       'race_date', 'venue', 'grade', 'distance', 'track_condition', 'weather']]
    
    # Save input data
    mapped_data.to_csv('./debug_artifacts/v4/input_race.csv', index=False)
    logger.info(f"Saved input race data: {len(mapped_data)} dogs")
    
    return mapped_data

def debug_model_pipeline():
    """Debug model pipeline selection and metadata."""
    from ml_system_v4 import MLSystemV4
    
    logger.info("=== DEBUGGING MODEL PIPELINE ===")
    
    system = MLSystemV4()
    
    # Log basic model info
    logger.info(f"Model info: {system.model_info}")
    logger.info(f"Has calibrated pipeline: {system.calibrated_pipeline is not None}")
    logger.info(f"Feature columns: {len(system.feature_columns)} cols")
    logger.info(f"Model type: {system.model_info.get('model_type', 'unknown')}")
    
    if system.calibrated_pipeline:
        logger.info(f"Pipeline type: {type(system.calibrated_pipeline).__name__}")
        
        # Check for classes_
        classes = getattr(system.calibrated_pipeline, 'classes_', None)
        logger.info(f"Classes: {classes}")
        
        # Check preprocessor
        try:
            pre = getattr(system.calibrated_pipeline, 'named_steps', {}).get('preprocessor')
            if pre is None and hasattr(system.calibrated_pipeline, 'base_estimator_'):
                pre = getattr(system.calibrated_pipeline.base_estimator_, 'named_steps', {}).get('preprocessor')
            
            if pre and hasattr(pre, 'transformers_'):
                logger.info(f"Preprocessor transformers: {len(pre.transformers_)}")
                for name, trans, cols in pre.transformers_:
                    logger.info(f"  {name}: {len(cols) if cols is not None else 0} columns")
                    if cols is not None and len(cols) < 20:
                        logger.info(f"    Columns: {list(cols)[:10]}...")
                        
                # Save transformer info
                transformer_info = {
                    'transformers': []
                }
                for name, trans, cols in pre.transformers_:
                    transformer_info['transformers'].append({
                        'name': name,
                        'type': str(type(trans)),
                        'columns': list(cols) if cols is not None else [],
                        'column_count': len(cols) if cols is not None else 0
                    })
                
                with open('./debug_artifacts/v4/transformer_info.json', 'w') as f:
                    json.dump(transformer_info, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error analyzing preprocessor: {e}")
    
    return system

def debug_feature_building(system, race_data):
    """Debug feature building process."""
    logger.info("=== DEBUGGING FEATURE BUILDING ===")
    
    race_id = 'debug_race_sand'
    
    try:
        # Build features using temporal builder
        race_features = system.build_features_for_race_with_cache(race_data, race_id)
        
        if race_features is not None and not race_features.empty:
            logger.info(f"Built {len(race_features)} feature vectors with {len(race_features.columns)} columns")
            
            # Save race features before any processing
            race_features.to_csv('./debug_artifacts/v4/race_features.csv', index=False)
            
            # Analyze feature variance
            varying_cols = []
            constant_cols = []
            
            for col in race_features.columns:
                if col not in ['race_id', 'dog_clean_name', 'target', 'target_timestamp']:
                    nunique = race_features[col].nunique()
                    if nunique > 1:
                        varying_cols.append((col, nunique))
                    else:
                        constant_cols.append((col, race_features[col].iloc[0] if len(race_features) > 0 else None))
            
            logger.info(f"Varying columns: {len(varying_cols)}")
            logger.info(f"Constant columns: {len(constant_cols)}")
            logger.info(f"Percent constant: {len(constant_cols) / max(1, len(varying_cols) + len(constant_cols)) * 100:.1f}%")
            
            # Log top varying columns
            varying_cols.sort(key=lambda x: x[1], reverse=True)
            logger.info("Top 10 varying columns:")
            for col, nunique in varying_cols[:10]:
                logger.info(f"  {col}: {nunique} unique values")
                
            # Save variance analysis
            variance_analysis = {
                'varying_columns': varying_cols,
                'constant_columns': constant_cols,
                'percent_constant': len(constant_cols) / max(1, len(varying_cols) + len(constant_cols)) * 100
            }
            
            with open('./debug_artifacts/v4/variance_analysis.json', 'w') as f:
                json.dump(variance_analysis, f, indent=2, default=str)
                
        else:
            logger.error("Feature building returned empty result")
            return None
            
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
    return race_features

def debug_prediction_pipeline(system, race_data):
    """Debug the full prediction pipeline step by step."""
    logger.info("=== DEBUGGING FULL PREDICTION PIPELINE ===")
    
    race_id = 'debug_race_sand'
    
    # Monkey patch to capture intermediate results
    original_predict_race = system.predict_race
    
    def debug_predict_race(race_data, race_id=None, market_odds=None):
        logger.info("Starting debug prediction...")
        
        # Build features
        race_features = system.build_features_for_race_with_cache(race_data, race_id)
        
        if race_features is None or race_features.empty:
            logger.error("No features generated")
            return {"success": False, "error": "No features generated"}
            
        # Prepare X_pred before reindexing
        X_pred_before = race_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
        X_pred_before.to_csv('./debug_artifacts/v4/X_pred_before_reindex.csv', index=False)
        logger.info(f"X_pred_before shape: {X_pred_before.shape}")
        
        # Get expected features from preprocessor
        cp = system.calibrated_pipeline
        pre = getattr(cp, 'named_steps', {}).get('preprocessor')
        if pre is None and hasattr(cp, 'base_estimator_'):
            pre = getattr(cp.base_estimator_, 'named_steps', {}).get('preprocessor')
            
        if pre and hasattr(pre, 'transformers_'):
            expected_cols = []
            derived_cat = set()
            derived_num = set()
            
            for name, trans, cols in pre.transformers_:
                if cols is not None:
                    expected_cols.extend(list(cols))
                    if name == 'cat':
                        derived_cat.update(list(cols))
                    elif name == 'num':
                        derived_num.update(list(cols))
            
            expected_set = set(expected_cols)
            missing_features = expected_set - set(X_pred_before.columns)
            extra_features = set(X_pred_before.columns) - expected_set
            
            logger.info(f"Expected features: {len(expected_set)}")
            logger.info(f"Missing features: {len(missing_features)} - {list(missing_features)[:10]}")
            logger.info(f"Extra features: {len(extra_features)} - {list(extra_features)[:10]}")
            
            # Reindex to expected features
            X_pred_after = X_pred_before.reindex(columns=sorted(expected_set), fill_value=0)
            X_pred_after.to_csv('./debug_artifacts/v4/X_pred_after_reindex.csv', index=False)
            logger.info(f"X_pred_after shape: {X_pred_after.shape}")
            
            # Handle categorical and numerical features
            for cat_col in derived_cat:
                if cat_col in X_pred_after.columns:
                    defaults = {"venue": "UNKNOWN", "grade": "5", "track_condition": "Good", 
                               "weather": "Fine", "trainer_name": "Unknown"}
                    default_val = defaults.get(cat_col, "Unknown")
                    X_pred_after[cat_col] = X_pred_after[cat_col].apply(
                        lambda x: default_val if (pd.isna(x) or x == 0 or x == "0") else str(x)
                    )
                    
            for num_col in derived_num:
                if num_col in X_pred_after.columns:
                    X_pred_after[num_col] = pd.to_numeric(X_pred_after[num_col], errors='coerce').fillna(0.0).astype(np.float64)
            
            # Make raw predictions
            try:
                classes = getattr(cp, 'classes_', None)
                pos_index = int(np.where(classes == 1)[0][0]) if classes is not None and 1 in classes else -1
                logger.info(f"Classes: {classes}, positive index: {pos_index}")
                
                if pos_index >= 0:
                    proba_raw = cp.predict_proba(X_pred_after)[:, pos_index]
                else:
                    proba_raw = cp.predict_proba(X_pred_after)[:, 1]  # fallback
                    
                logger.info(f"Raw probabilities: {proba_raw}")
                logger.info(f"Raw prob stats: mean={np.mean(proba_raw):.6f}, std={np.std(proba_raw):.6f}, min={np.min(proba_raw):.6f}, max={np.max(proba_raw):.6f}")
                
                # Save raw probabilities
                with open('./debug_artifacts/v4/proba_raw.json', 'w') as f:
                    json.dump({
                        'probabilities': proba_raw.tolist(),
                        'stats': {
                            'mean': float(np.mean(proba_raw)),
                            'std': float(np.std(proba_raw)),
                            'min': float(np.min(proba_raw)),
                            'max': float(np.max(proba_raw))
                        }
                    }, f, indent=2)
                
                # Test normalization
                if np.sum(proba_raw) > 0:
                    normalized_simple = proba_raw / np.sum(proba_raw)
                else:
                    normalized_simple = np.ones(len(proba_raw)) / len(proba_raw)
                
                logger.info(f"Simple normalized: {normalized_simple}")
                
                # Return debug result
                debug_result = {
                    "success": True,
                    "race_id": race_id,
                    "predictions": [],
                    "debug_info": {
                        "proba_raw": proba_raw.tolist(),
                        "normalized_simple": normalized_simple.tolist(),
                        "expected_features": len(expected_set),
                        "missing_features": len(missing_features),
                        "extra_features": len(extra_features)
                    }
                }
                
                # Create predictions
                for i, dog_name in enumerate(race_features['dog_clean_name']):
                    pred = {
                        'dog_clean_name': dog_name,
                        'box_number': int(race_features.iloc[i].get('box_number', i + 1)),
                        'win_probability': float(normalized_simple[i]),
                        'place_probability': float(min(0.95, normalized_simple[i] * 2.8)),
                        'confidence': float(min(0.95, 0.6 + 0.3 * normalized_simple[i])),
                        'ensemble_models': ['debug_model'],
                        'model_agreement': 1.0,
                        'race_id': race_id,
                        'prediction_timestamp': datetime.now().isoformat()
                    }
                    debug_result['predictions'].append(pred)
                
                # Sort by probability
                debug_result['predictions'].sort(key=lambda x: x['win_probability'], reverse=True)
                
                return debug_result
                
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"success": False, "error": str(e)}
        else:
            logger.error("No preprocessor found")
            return {"success": False, "error": "No preprocessor found"}
    
    # Run debug prediction
    result = debug_predict_race(race_data, race_id)
    
    # Save full result
    with open('./debug_artifacts/v4/prediction_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
        
    return result

def main():
    """Main debug harness."""
    logger.info("Starting prediction failure debug harness")
    
    try:
        # Create debug race data
        race_data = create_debug_race_data()
        
        # Debug model pipeline
        system = debug_model_pipeline()
        
        # Debug feature building
        race_features = debug_feature_building(system, race_data)
        
        if race_features is not None:
            # Debug full prediction pipeline
            result = debug_prediction_pipeline(system, race_data)
            
            if result.get('success'):
                logger.info("=== FINAL RESULTS ===")
                predictions = result.get('predictions', [])
                for p in predictions[:3]:
                    logger.info(f"  {p['dog_clean_name']}: win_prob={p['win_probability']:.4f}, confidence={p['confidence']:.4f}")
                
                debug_info = result.get('debug_info', {})
                logger.info(f"Raw probabilities std: {np.std(debug_info.get('proba_raw', [])):.6f}")
                
            else:
                logger.error(f"Prediction failed: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Debug harness failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
