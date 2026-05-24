#!/usr/bin/env python3
"""
ML System V4 Prediction Fix
============================

Patches to fix the prediction system failures:
1. Handle missing preprocessor in calibrated pipeline
2. Add guards against uniform predictions
3. Fix feature alignment issues
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def patch_predict_race_method():
    """Monkey patch the predict_race method to handle the missing preprocessor issue."""
    
    from ml_system_v4 import MLSystemV4
    
    original_predict_race = MLSystemV4.predict_race
    
    def patched_predict_race(self, race_data, race_id=None, market_odds=None):
        """Patched predict_race method with better error handling."""
        
        # Build features first
        try:
            race_features = self.build_features_for_race_with_cache(race_data, race_id)
            
            if race_features is None or race_features.empty:
                logger.error(f"No features generated for race {race_id}")
                return {
                    "success": False,
                    "error": "Feature building returned empty result",
                    "race_id": race_id,
                }
                
        except Exception as e:
            logger.error(f"Feature building failed for race {race_id}: {e}")
            return {
                "success": False,
                "error": f"Feature building error: {str(e)}",
                "race_id": race_id,
            }
        
        # Check for pipeline structure issues
        cp = self.calibrated_pipeline
        
        if not cp:
            logger.error("No calibrated model available")
            return {
                "success": False,
                "error": "No model loaded",
                "race_id": race_id,
            }
        
        # Check if this is a raw classifier (missing preprocessor)
        if not hasattr(cp, 'named_steps') and not hasattr(cp, 'base_estimator_'):
            logger.warning("Model is a raw classifier without preprocessor - attempting direct prediction")
            
            # Try to predict directly with available features
            X_pred = race_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], 
                                       axis=1, errors='ignore')
            
            # Check feature variance - guard against uniform predictions
            varying_cols = []
            for col in X_pred.columns:
                if X_pred[col].nunique() > 1:
                    varying_cols.append(col)
            
            percent_constant = (len(X_pred.columns) - len(varying_cols)) / max(1, len(X_pred.columns)) * 100
            
            if percent_constant > 70:
                logger.warning(f"Warning: {percent_constant:.1f}% of features are constant - predictions may be uniform")
                
            if len(varying_cols) < 3:
                logger.error(f"Only {len(varying_cols)} varying features found - insufficient for prediction")
                return {
                    "success": False,
                    "error": f"Insufficient feature variance: only {len(varying_cols)} varying columns",
                    "race_id": race_id,
                    "varying_features": varying_cols,
                    "percent_constant": percent_constant,
                }
            
            # Handle categorical features - convert to proper types
            categorical_features = ['venue', 'grade', 'track_condition', 'weather', 'trainer_name']
            numerical_features = [col for col in X_pred.columns if col not in categorical_features]
            
            # Process categorical features
            for cat_col in categorical_features:
                if cat_col in X_pred.columns:
                    defaults = {"venue": "UNKNOWN", "grade": "5", "track_condition": "Good", 
                               "weather": "Fine", "trainer_name": "Unknown"}
                    default_val = defaults.get(cat_col, "Unknown")
                    X_pred[cat_col] = X_pred[cat_col].apply(
                        lambda x: default_val if (pd.isna(x) or x == 0 or x == "0") else str(x)
                    )
            
            # Process numerical features
            for num_col in numerical_features:
                if num_col in X_pred.columns:
                    X_pred[num_col] = pd.to_numeric(X_pred[num_col], errors='coerce').fillna(0.0)
            
            # For direct prediction without preprocessor, we need to handle categorical encoding
            # This is a simplified approach - ideally we'd have the original preprocessor
            
            # One-hot encode categorical features manually
            for cat_col in categorical_features:
                if cat_col in X_pred.columns:
                    # Simple categorical encoding - replace with numeric codes
                    unique_vals = X_pred[cat_col].unique()
                    for i, val in enumerate(unique_vals):
                        X_pred[f'{cat_col}_{val}'] = (X_pred[cat_col] == val).astype(int)
                    X_pred = X_pred.drop(cat_col, axis=1)
            
            # Ensure all features are numeric
            for col in X_pred.columns:
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0.0)
            
            try:
                # Get classes and determine positive class index
                classes = getattr(cp, 'classes_', np.array([0, 1]))
                pos_index = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                
                # Make predictions
                proba_full = cp.predict_proba(X_pred)
                
                # Check if we have the right number of classes
                if proba_full.shape[1] > pos_index:
                    proba_raw = proba_full[:, pos_index]
                else:
                    proba_raw = proba_full[:, -1]  # Use last column as fallback
                
                logger.info(f"Raw probabilities stats: mean={np.mean(proba_raw):.6f}, std={np.std(proba_raw):.6f}")
                
                # Guard against uniform predictions
                if np.std(proba_raw) < 1e-6:
                    logger.warning("Raw probabilities are uniform - model may be broken")
                    return {
                        "success": False,
                        "error": "Model producing uniform probabilities",
                        "race_id": race_id,
                        "proba_std": float(np.std(proba_raw)),
                    }
                
                # Normalize probabilities
                if np.sum(proba_raw) > 0:
                    normalized_probs = proba_raw / np.sum(proba_raw)
                else:
                    normalized_probs = np.ones(len(proba_raw)) / len(proba_raw)
                
                # Create predictions
                predictions = []
                for i, dog_name in enumerate(race_features['dog_clean_name']):
                    pred = {
                        'dog_clean_name': dog_name,
                        'box_number': int(race_features.iloc[i].get('box_number', i + 1)),
                        'win_probability': float(normalized_probs[i]),
                        'place_probability': float(min(0.95, normalized_probs[i] * 2.8)),
                        'confidence': float(min(0.95, 0.6 + 0.3 * normalized_probs[i])),
                        'ensemble_models': ['patched_model'],
                        'model_agreement': 1.0,
                        'race_id': race_id,
                    }
                    predictions.append(pred)
                
                # Sort by probability
                predictions.sort(key=lambda x: x['win_probability'], reverse=True)
                
                return {
                    "success": True,
                    "race_id": race_id,
                    "predictions": predictions,
                    "patch_info": {
                        "method": "direct_classifier_prediction",
                        "varying_features": len(varying_cols),
                        "percent_constant": percent_constant,
                        "proba_std": float(np.std(proba_raw)),
                    }
                }
                
            except Exception as e:
                logger.error(f"Direct prediction failed: {e}")
                return {
                    "success": False,
                    "error": f"Direct prediction error: {str(e)}",
                    "race_id": race_id,
                }
        
        # If we have a proper pipeline, delegate to original method
        return original_predict_race(self, race_data, race_id, market_odds)
    
    # Apply the patch
    MLSystemV4.predict_race = patched_predict_race
    logger.info("Applied prediction pipeline patch for missing preprocessor issue")

def test_patched_system():
    """Test the patched system."""
    import pandas as pd
    
    # Create test data
    data = pd.read_csv('Race 1 - SAND - 25 August 2025.csv')
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

    mapped_data = data[['dog_clean_name', 'box_number', 'weight', 'trainer_name', 
                       'race_date', 'venue', 'grade', 'distance', 'track_condition', 'weather']]
    
    # Apply patch
    patch_predict_race_method()
    
    # Test the system
    from ml_system_v4 import MLSystemV4
    system = MLSystemV4()
    
    result = system.predict_race(mapped_data, 'test_patched_race')
    
    print("=== PATCHED SYSTEM TEST RESULTS ===")
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        predictions = result.get('predictions', [])
        print(f"Predictions: {len(predictions)}")
        for p in predictions[:3]:
            print(f"  {p['dog_clean_name']}: {p['win_probability']:.4f}")
            
        patch_info = result.get('patch_info', {})
        print(f"Varying features: {patch_info.get('varying_features')}")
        print(f"Probability std: {patch_info.get('proba_std', 0):.6f}")
    else:
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    test_patched_system()
