#!/usr/bin/env python3
"""
Test Script for Prediction Adapters
===================================

This script demonstrates and validates the functionality of the three adapter classes:
- V3Adapter
- V3SAdapter  
- V4Adapter

It shows how all adapters return the same standardized format regardless of the
underlying prediction system being used.
"""

import logging
import os
import json
from pathlib import Path
from prediction_adapters import V3Adapter, V3SAdapter, V4Adapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adapter(adapter_class, adapter_name, race_file_path):
    """Test a specific adapter class."""
    print(f"\n{'='*60}")
    print(f"Testing {adapter_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize adapter
        adapter = adapter_class()
        print(f"✅ {adapter_name} initialized successfully")
        
        # Make prediction
        result = adapter.predict_race(race_file_path)
        
        # Validate result structure
        if validate_result_structure(result, adapter_name):
            print(f"✅ {adapter_name} returned valid standardized structure")
            print_result_summary(result, adapter_name)
        else:
            print(f"❌ {adapter_name} returned invalid structure")
            
        return result
        
    except Exception as e:
        print(f"❌ {adapter_name} failed with error: {e}")
        logger.exception(f"Error testing {adapter_name}")
        return None

def validate_result_structure(result, adapter_name):
    """Validate that the result follows the standardized structure."""
    required_keys = ['race_id', 'predictions', 'metadata']
    
    # Check top-level structure
    for key in required_keys:
        if key not in result:
            print(f"❌ {adapter_name}: Missing required key '{key}'")
            return False
    
    # Check metadata structure
    metadata = result['metadata']
    required_metadata_keys = ['timestamp', 'success', 'adapter', 'method']
    for key in required_metadata_keys:
        if key not in metadata:
            print(f"❌ {adapter_name}: Missing required metadata key '{key}'")
            return False
    
    # Check predictions structure
    predictions = result['predictions']
    if not isinstance(predictions, list):
        print(f"❌ {adapter_name}: 'predictions' should be a list")
        return False
    
    # Check individual prediction structure
    for i, pred in enumerate(predictions):
        required_pred_keys = ['dog', 'win_prob_norm', 'raw_prob']
        for key in required_pred_keys:
            if key not in pred:
                print(f"❌ {adapter_name}: Missing required prediction key '{key}' in prediction {i}")
                return False
        
        # Check probability values are reasonable
        if not (0 <= pred['win_prob_norm'] <= 1):
            print(f"❌ {adapter_name}: win_prob_norm ({pred['win_prob_norm']}) out of range [0,1] for {pred['dog']}")
            return False
        
        if not (0 <= pred['raw_prob'] <= 1):
            print(f"❌ {adapter_name}: raw_prob ({pred['raw_prob']}) out of range [0,1] for {pred['dog']}")
            return False
    
    # Check that normalized probabilities sum to approximately 1
    if predictions:
        norm_prob_sum = sum(pred['win_prob_norm'] for pred in predictions)
        if not (0.95 <= norm_prob_sum <= 1.05):  # Allow small numerical errors
            print(f"❌ {adapter_name}: Normalized probabilities sum to {norm_prob_sum:.3f}, expected ~1.0")
            return False
    
    return True

def print_result_summary(result, adapter_name):
    """Print a summary of the prediction result."""
    metadata = result['metadata']
    predictions = result['predictions']
    
    print(f"\n📊 {adapter_name} Result Summary:")
    print(f"   Race ID: {result['race_id']}")
    print(f"   Success: {metadata['success']}")
    print(f"   Method: {metadata.get('method', 'Unknown')}")
    print(f"   Total Dogs: {len(predictions)}")
    
    if metadata.get('error'):
        print(f"   Error: {metadata['error']}")
        return
    
    print(f"\n🏆 Top 3 Predictions:")
    for i, pred in enumerate(predictions[:3], 1):
        prob_norm = pred['win_prob_norm']
        prob_raw = pred['raw_prob']
        confidence = pred.get('confidence')
        conf_str = f" (conf: {confidence:.2f})" if confidence else ""
        print(f"   {i}. {pred['dog']}: {prob_norm:.3f} (raw: {prob_raw:.3f}){conf_str}")
    
    # Show additional adapter-specific information
    if 'fallback_reasons' in metadata and metadata['fallback_reasons']:
        print(f"   Fallback Reasons: {len(metadata['fallback_reasons'])} reasons")
    
    if 'calibration_applied' in metadata:
        print(f"   Calibration Applied: {metadata['calibration_applied']}")
    
    if 'temporal_leakage_protected' in metadata:
        print(f"   Temporal Leakage Protected: {metadata['temporal_leakage_protected']}")

def compare_adapters(results):
    """Compare results from different adapters."""
    print(f"\n{'='*60}")
    print("Adapter Comparison")
    print(f"{'='*60}")
    
    valid_results = {name: result for name, result in results.items() 
                    if result and result['metadata']['success']}
    
    if not valid_results:
        print("❌ No valid results to compare")
        return
    
    print(f"✅ Successfully compared {len(valid_results)} adapters")
    
    # Compare prediction counts
    print(f"\n📊 Prediction Counts:")
    for name, result in valid_results.items():
        count = len(result['predictions'])
        method = result['metadata'].get('method', 'Unknown')
        print(f"   {name}: {count} predictions (method: {method})")
    
    # Compare top prediction for each adapter
    print(f"\n🏆 Top Predictions Comparison:")
    for name, result in valid_results.items():
        if result['predictions']:
            top_pred = result['predictions'][0]
            print(f"   {name}: {top_pred['dog']} ({top_pred['win_prob_norm']:.3f})")
        else:
            print(f"   {name}: No predictions")
    
    # Show adapter-specific features
    print(f"\n🔧 Adapter Features:")
    for name, result in valid_results.items():
        metadata = result['metadata']
        features = []
        
        if metadata.get('temporal_leakage_protected'):
            features.append("Temporal Protection")
        if metadata.get('calibration_applied'):
            features.append("Calibration")
        if metadata.get('fallback_used'):
            features.append("Fallback Used")
        if 'ev_analysis' in metadata:
            features.append("EV Analysis")
            
        feature_str = ", ".join(features) if features else "Basic"
        print(f"   {name}: {feature_str}")

def find_test_race_file():
    """Find a suitable test race file."""
    # Look for race files in common locations
    possible_locations = [
        "upcoming_races",
        "test_data", 
        "sample_data",
        "."
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if file.endswith('.csv') and ('race' in file.lower() or 'dog' in file.lower()):
                    return os.path.join(location, file)
    
    return None

def create_sample_race_file():
    """Create a sample race file for testing if none exists."""
    sample_data = """Dog Name,BOX,WGT,SP,TRAINER,G,DIST,PIR
1. FAST RUNNER,1,30.5,2.50,J SMITH,G5,500,85
2. QUICK STAR,2,31.0,3.20,M JONES,G5,500,82
3. SPEEDY DOG,3,29.8,4.10,S BROWN,G5,500,78
4. RAPID HOUND,4,30.2,5.50,P WILSON,G5,500,75
5. SWIFT RACER,5,31.5,6.00,T DAVIS,G5,500,73
6. FLEET FOOT,6,30.8,7.20,R MILLER,G5,500,70
7. DASH MASTER,7,29.5,8.50,L TAYLOR,G5,500,68
8. BOLT LIGHTNING,8,30.0,12.00,K ANDERSON,G5,500,65"""
    
    test_file = "Race 1 - TEST - 04 August 2025.csv"
    
    with open(test_file, 'w') as f:
        f.write(sample_data)
    
    print(f"📁 Created sample race file: {test_file}")
    return test_file

def main():
    """Main test function."""
    print("🚀 Prediction Adapters Test Suite")
    print("=" * 60)
    
    # Find or create test race file
    race_file = find_test_race_file()
    if not race_file:
        race_file = create_sample_race_file()
    
    print(f"📁 Using test race file: {race_file}")
    
    # Test all adapters
    adapters = [
        (V3Adapter, "V3Adapter"),
        (V3SAdapter, "V3SAdapter"), 
        (V4Adapter, "V4Adapter")
    ]
    
    results = {}
    
    for adapter_class, adapter_name in adapters:
        result = test_adapter(adapter_class, adapter_name, race_file)
        results[adapter_name] = result
    
    # Compare results
    compare_adapters(results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    successful_adapters = [name for name, result in results.items() 
                          if result and result['metadata']['success']]
    
    print(f"✅ Successful Adapters: {len(successful_adapters)}/{len(adapters)}")
    for name in successful_adapters:
        print(f"   ✅ {name}")
    
    failed_adapters = [name for name, result in results.items() 
                      if not result or not result['metadata']['success']]
    
    if failed_adapters:
        print(f"❌ Failed Adapters: {len(failed_adapters)}")
        for name in failed_adapters:
            error = results[name]['metadata'].get('error', 'Unknown error') if results[name] else 'Initialization failed'
            print(f"   ❌ {name}: {error}")
    
    print(f"\n🎯 All adapters return standardized format: {race_id, predictions[{dog, win_prob_norm, raw_prob}], metadata}")
    print("✅ Test suite completed!")
    
    # Clean up sample file if we created it
    if race_file == "Race 1 - TEST - 04 August 2025.csv":
        try:
            os.remove(race_file)
            print(f"🗑️ Cleaned up sample file: {race_file}")
        except:
            pass

if __name__ == "__main__":
    main()
