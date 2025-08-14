#!/usr/bin/env python3
"""
Demo script for sanity checks integration
==========================================

This script demonstrates how to integrate sanity checks into the prediction pipeline
and shows examples of both valid and invalid predictions.
"""

import json
import logging
from pathlib import Path
from sanity_checks import SanityChecks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_sanity_checks():
    """
    Demonstrate sanity checks with various prediction scenarios.
    """
    checker = SanityChecks()
    
    print("🔍 Sanity Checks Demo")
    print("=" * 50)
    
    # Example 1: Valid predictions
    print("\n📋 Test 1: Valid Predictions")
    valid_predictions = [
        {
            'dog_name': 'Lightning Bolt',
            'win_probability': 0.35,
            'place_probability': 0.60,
            'predicted_rank': 1
        },
        {
            'dog_name': 'Thunder Strike',
            'win_probability': 0.25,
            'place_probability': 0.55,
            'predicted_rank': 2
        },
        {
            'dog_name': 'Wind Runner',
            'win_probability': 0.20,
            'place_probability': 0.45,
            'predicted_rank': 3
        },
        {
            'dog_name': 'Storm Chaser',
            'win_probability': 0.20,
            'place_probability': 0.40,
            'predicted_rank': 4
        }
    ]
    
    result = checker.validate_predictions(valid_predictions)
    print(f"✅ Validation Result: {result['total_predictions']} predictions")
    print(f"   Passed Checks: {result['passed_checks']}")
    print(f"   Failed Checks: {result['failed_checks']}")
    print(f"   Flags: {result['flags']}")
    
    # Example 2: Invalid probability ranges
    print("\n📋 Test 2: Invalid Probability Ranges")
    invalid_range_predictions = [
        {
            'dog_name': 'Speed Demon',
            'win_probability': 1.5,  # Invalid: > 1
            'place_probability': 0.60,
            'predicted_rank': 1
        },
        {
            'dog_name': 'Fast Track',
            'win_probability': -0.1,  # Invalid: < 0
            'place_probability': 0.55,
            'predicted_rank': 2
        }
    ]
    
    result = checker.validate_predictions(invalid_range_predictions)
    print(f"❌ Validation Result: {result['total_predictions']} predictions")
    print(f"   Passed Checks: {result['passed_checks']}")
    print(f"   Failed Checks: {result['failed_checks']}")
    print(f"   Flags: {result['flags']}")
    
    # Example 3: Rank misalignment
    print("\n📋 Test 3: Rank Misalignment")
    misaligned_predictions = [
        {
            'dog_name': 'Top Dog',
            'win_probability': 0.10,  # Low probability but ranked 1st
            'predicted_rank': 1
        },
        {
            'dog_name': 'Underdog',
            'win_probability': 0.90,  # High probability but ranked 2nd
            'predicted_rank': 2
        }
    ]
    
    result = checker.validate_predictions(misaligned_predictions)
    print(f"❌ Validation Result: {result['total_predictions']} predictions")
    print(f"   Passed Checks: {result['passed_checks']}")
    print(f"   Failed Checks: {result['failed_checks']}")
    print(f"   Flags: {result['flags']}")
    
    # Example 4: Duplicate ranks
    print("\n📋 Test 4: Duplicate Ranks")
    duplicate_rank_predictions = [
        {
            'dog_name': 'Racer A',
            'win_probability': 0.40,
            'predicted_rank': 1
        },
        {
            'dog_name': 'Racer B',
            'win_probability': 0.60,
            'predicted_rank': 1  # Duplicate rank
        }
    ]
    
    result = checker.validate_predictions(duplicate_rank_predictions)
    print(f"❌ Validation Result: {result['total_predictions']} predictions")
    print(f"   Passed Checks: {result['passed_checks']}")
    print(f"   Failed Checks: {result['failed_checks']}")
    print(f"   Flags: {result['flags']}")

def validate_prediction_file(file_path: str):
    """
    Validate predictions from a JSON file.
    
    Args:
        file_path: Path to the prediction JSON file
    """
    checker = SanityChecks()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        predictions = data.get('predictions', [])
        if not predictions:
            print(f"❌ No predictions found in {file_path}")
            return
        
        result = checker.validate_predictions(predictions)
        print(f"\n📄 Validation Results for {Path(file_path).name}")
        print(f"   Total Predictions: {result['total_predictions']}")
        print(f"   Passed Checks: {len(result['passed_checks'])}")
        print(f"   Failed Checks: {len(result['failed_checks'])}")
        
        if result['flags']:
            print(f"   ⚠️  Issues Found: {len(result['flags'])}")
            for flag in result['flags']:
                print(f"      - {flag}")
        else:
            print("   ✅ All validations passed!")
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {file_path}")
    except Exception as e:
        print(f"❌ Error validating file {file_path}: {e}")

def main():
    """
    Main demonstration function.
    """
    # Run demo scenarios
    demo_sanity_checks()
    
    # Check if there are any prediction files to validate
    prediction_dir = Path("predictions")
    if prediction_dir.exists():
        print(f"\n🔍 Checking for prediction files in {prediction_dir}")
        json_files = list(prediction_dir.glob("*.json"))
        
        if json_files:
            print(f"Found {len(json_files)} JSON files. Validating first 3...")
            for file_path in json_files[:3]:
                validate_prediction_file(str(file_path))
        else:
            print("No JSON prediction files found.")
    else:
        print("\n📁 No predictions directory found.")
        
    print("\n🎯 Sanity Checks Demo Complete!")

if __name__ == "__main__":
    main()
