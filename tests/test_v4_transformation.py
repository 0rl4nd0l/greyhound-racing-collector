#!/usr/bin/env python3
"""
Quick test for V4 data transformation functionality
Tests the CSV to V4 format transformation without heavy dependencies
"""

import logging
import pandas as pd
from pathlib import Path
from prediction_pipeline_v4 import PredictionPipelineV4

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_v4_transformation():
    """Test the V4 pipeline transformation without full system initialization"""
    
    print("=== Testing V4 Data Transformation ===")
    
    # Create a minimal test CSV data
    test_data = {
        'Dog Name': ['LONE SIZZLER', 'FAST RUNNER', 'QUICK DASH'],
        'BOX': [1, 2, 3],
        'WGT': [30.5, 32.0, 29.8],
        'SP': [3.50, 2.20, 4.10],
        'DATE': ['01/08/2025', '01/08/2025', '01/08/2025'],
        'TRACK': ['GOUL', 'GOUL', 'GOUL'],
        'G': ['5', '5', '5'],
        'TIME': ['29.50', '29.45', '29.60']
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    print(f"Original CSV data shape: {test_df.shape}")
    print("Original columns:", list(test_df.columns))
    
    # Create pipeline instance
    try:
        pipeline = PredictionPipelineV4()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Test the transformation method
    try:
        # Apply transformation (the method extracts race info internally)
        filename = "Race 1 - GOUL - 01 August 2025.csv"
        transformed_df = pipeline._map_csv_to_v4_format(test_df, filename)
        print(f"✅ Transformed data shape: {transformed_df.shape}")
        print("✅ Transformed columns:", list(transformed_df.columns))
        
        # Show sample of transformed data
        print("\n=== Transformation Results ===")
        for col in transformed_df.columns:
            print(f"{col}: {transformed_df[col].iloc[0]}")
        
        # Check required V4 fields are present
        required_fields = [
            'race_id', 'dog_clean_name', 'box_number', 'weight', 'starting_price',
            'trainer_name', 'venue', 'grade', 'track_condition', 'weather',
            'temperature', 'humidity', 'wind_speed', 'field_size', 'race_date', 'race_time'
        ]
        
        missing_fields = [field for field in required_fields if field not in transformed_df.columns]
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            print("✅ All required V4 fields present")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v4_transformation()
    if success:
        print("\n✅ V4 transformation test PASSED")
    else:
        print("\n❌ V4 transformation test FAILED")
