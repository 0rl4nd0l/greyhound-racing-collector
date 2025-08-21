#!/usr/bin/env python3
"""
ML Pipeline Validation Script
=============================

Test the MLSystemV4 pipeline end-to-end.
"""

import sys
import os
import json
import warnings
import traceback
from datetime import datetime
import time
import logging

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def test_ml_pipeline():
    """Test the ML pipeline functionality."""
    
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_status": "unknown",
        "issues": [],
        "recommendations": []
    }
    
    print("🧪 Testing ML Pipeline Components...")
    
    # Test 1: Import MLSystemV4
    try:
        from ml_system_v4 import MLSystemV4
        results["tests"]["import_ml_system"] = {"status": "pass", "details": "MLSystemV4 imported successfully"}
        print("✅ MLSystemV4 import successful")
    except Exception as e:
        results["tests"]["import_ml_system"] = {"status": "fail", "error": str(e)}
        results["issues"].append("Cannot import MLSystemV4")
        print(f"❌ MLSystemV4 import failed: {e}")
        return results
    
    # Test 2: Import temporal feature builder
    try:
        from temporal_feature_builder import TemporalFeatureBuilder
        results["tests"]["import_temporal_builder"] = {"status": "pass", "details": "TemporalFeatureBuilder imported successfully"}
        print("✅ TemporalFeatureBuilder import successful")
    except Exception as e:
        results["tests"]["import_temporal_builder"] = {"status": "fail", "error": str(e)}
        results["issues"].append("Cannot import TemporalFeatureBuilder")
        print(f"❌ TemporalFeatureBuilder import failed: {e}")
    
    # Test 3: Instantiate MLSystemV4
    try:
        ml_system = MLSystemV4()
        results["tests"]["instantiate_ml_system"] = {"status": "pass", "details": f"MLSystemV4 instantiated with DB: {ml_system.db_path}"}
        print(f"✅ MLSystemV4 instantiated successfully")
        print(f"   Database path: {ml_system.db_path}")
    except Exception as e:
        results["tests"]["instantiate_ml_system"] = {"status": "fail", "error": str(e), "traceback": traceback.format_exc()}
        results["issues"].append("Cannot instantiate MLSystemV4")
        print(f"❌ MLSystemV4 instantiation failed: {e}")
        return results
    
    # Test 4: Check temporal builder
    try:
        has_temporal_builder = hasattr(ml_system, 'temporal_builder') and ml_system.temporal_builder is not None
        results["tests"]["temporal_builder_check"] = {
            "status": "pass" if has_temporal_builder else "fail",
            "details": f"Temporal builder present: {has_temporal_builder}"
        }
        print(f"✅ Temporal builder check: {has_temporal_builder}")
    except Exception as e:
        results["tests"]["temporal_builder_check"] = {"status": "fail", "error": str(e)}
        print(f"❌ Temporal builder check failed: {e}")
    
    # Test 5: Try loading existing model 
    try:
        has_pipeline = ml_system.pipeline is not None or ml_system.calibrated_pipeline is not None
        results["tests"]["existing_model_check"] = {
            "status": "info",
            "details": {
                "has_pipeline": ml_system.pipeline is not None,
                "has_calibrated_pipeline": ml_system.calibrated_pipeline is not None,
                "feature_columns_count": len(ml_system.feature_columns) if ml_system.feature_columns else 0
            }
        }
        print(f"📊 Existing model check:")
        print(f"   Pipeline: {ml_system.pipeline is not None}")
        print(f"   Calibrated pipeline: {ml_system.calibrated_pipeline is not None}")
        print(f"   Feature columns: {len(ml_system.feature_columns) if ml_system.feature_columns else 0}")
    except Exception as e:
        results["tests"]["existing_model_check"] = {"status": "fail", "error": str(e)}
        print(f"❌ Existing model check failed: {e}")
    
    # Test 6: Try preparing training data (with timeout)
    try:
        print("🔄 Testing data preparation (may take a moment)...")
        start_time = time.time()
        
        # Set reasonable limit for testing
        os.environ['V4_MAX_RACES'] = '50'
        
        train_data, test_data = ml_system.prepare_time_ordered_data()
        
        duration = time.time() - start_time
        
        train_size = len(train_data) if train_data is not None else 0
        test_size = len(test_data) if test_data is not None else 0
        
        results["tests"]["data_preparation"] = {
            "status": "pass" if train_size > 0 else "fail",
            "details": {
                "train_samples": train_size,
                "test_samples": test_size,
                "duration_seconds": round(duration, 2),
                "train_races": len(train_data['race_id'].unique()) if train_size > 0 else 0,
                "test_races": len(test_data['race_id'].unique()) if test_size > 0 else 0
            }
        }
        
        if train_size > 0:
            print(f"✅ Data preparation successful")
            print(f"   Train: {train_size} samples, {results['tests']['data_preparation']['details']['train_races']} races")
            print(f"   Test: {test_size} samples, {results['tests']['data_preparation']['details']['test_races']} races")
            print(f"   Duration: {duration:.2f}s")
        else:
            print(f"⚠️ Data preparation returned empty dataset")
            results["issues"].append("No training data available")
            
    except Exception as e:
        results["tests"]["data_preparation"] = {"status": "fail", "error": str(e), "traceback": traceback.format_exc()}
        results["issues"].append(f"Data preparation failed: {str(e)}")
        print(f"❌ Data preparation failed: {e}")
    
    # Test 7: Test prediction interface
    try:
        # Test with dummy dog features
        dummy_features = {
            'starting_price': 3.5,
            'odds': 3.5,
            'box_number': 1,
            'distance': 520,
            'track_condition': 'Good',
            'venue': 'Sandown',
            'grade': 'Grade 5'
        }
        
        prediction = ml_system.predict(dummy_features)
        
        is_valid_prediction = (
            isinstance(prediction, dict) and 
            'win_probability' in prediction and 
            'confidence' in prediction and
            isinstance(prediction['win_probability'], (int, float)) and
            isinstance(prediction['confidence'], (int, float)) and
            0 <= prediction['win_probability'] <= 1 and
            0 <= prediction['confidence'] <= 1
        )
        
        results["tests"]["prediction_interface"] = {
            "status": "pass" if is_valid_prediction else "fail", 
            "details": {
                "prediction_output": prediction,
                "is_valid_format": is_valid_prediction
            }
        }
        
        if is_valid_prediction:
            print(f"✅ Prediction interface working")
            print(f"   Sample prediction: {prediction}")
        else:
            print(f"⚠️ Prediction format issue: {prediction}")
            results["issues"].append("Prediction interface returns invalid format")
            
    except Exception as e:
        results["tests"]["prediction_interface"] = {"status": "fail", "error": str(e)}
        results["issues"].append(f"Prediction interface failed: {str(e)}")
        print(f"❌ Prediction interface failed: {e}")
    
    # Determine overall status
    passed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "pass")
    total_critical_tests = 6  # Don't count info tests
    
    if passed_tests >= 5:
        results["overall_status"] = "good"
    elif passed_tests >= 3:
        results["overall_status"] = "partial"
    else:
        results["overall_status"] = "poor"
    
    # Generate recommendations
    if results["overall_status"] == "poor":
        results["recommendations"].append("HIGH: Major issues with ML pipeline - requires immediate attention")
        results["recommendations"].append("Consider retraining models or fixing data pipeline issues")
    elif results["overall_status"] == "partial":
        results["recommendations"].append("MED: Some pipeline components working but issues present")
        results["recommendations"].append("Address data preparation and model loading issues")
    else:
        results["recommendations"].append("LOW: ML pipeline appears functional")
        results["recommendations"].append("Consider performance optimization and feature engineering improvements")
    
    # Save results
    output_path = "artifacts/pipeline_validation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📋 Pipeline Validation Summary:")
    print(f"Overall status: {results['overall_status'].upper()}")
    print(f"Passed tests: {passed_tests}/{total_critical_tests}")
    print(f"Issues found: {len(results['issues'])}")
    print(f"Report saved: {output_path}")
    
    if results["issues"]:
        print("\n⚠️ Issues:")
        for issue in results["issues"]:
            print(f"   • {issue}")
    
    if results["recommendations"]:
        print("\n🎯 Recommendations:")
        for rec in results["recommendations"]:
            print(f"   • {rec}")
    
    return results

if __name__ == "__main__":
    results = test_ml_pipeline()
