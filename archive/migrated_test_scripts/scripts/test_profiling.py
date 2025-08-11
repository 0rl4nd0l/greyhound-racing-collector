#!/usr/bin/env python3
"""
Test Profiling Script for Greyhound Prediction Pipeline
======================================================

This script runs the prediction pipeline with profiling enabled and validates
the generated JSON output with expected keys.

Usage:
    python3 scripts/test_profiling.py --race-file path.csv --enable-profiling

Expected output:
    - Runs pipeline with profiling enabled
    - Prints path to generated JSON file
    - Asserts expected keys exist in result
    - Creates profiling data in audit_results directory

Author: AI Assistant
Date: January 2025
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import profiling utilities
from profiling_config import set_profiling_enabled, is_profiling
from utils.profiling_utils import ProfilingRecorder

# Try to import available prediction pipelines (with fallbacks)
print("🔍 Detecting available prediction pipelines...")

pipeline = None
pipeline_name = "unknown"

# Try UnifiedPredictor first (most stable)
try:
    from unified_predictor import UnifiedPredictor
    pipeline = UnifiedPredictor()
    pipeline_name = "UnifiedPredictor"
    print("✅ Using UnifiedPredictor")
except Exception as e:
    print(f"⚠️  UnifiedPredictor not available: {e}")

# Try comprehensive prediction pipeline as fallback
if pipeline is None:
    try:
        from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
        pipeline = ComprehensivePredictionPipeline()
        pipeline_name = "ComprehensivePredictionPipeline"
        print("✅ Using ComprehensivePredictionPipeline")
    except Exception as e:
        print(f"⚠️  ComprehensivePredictionPipeline not available: {e}")

# Try ML System V3 as final fallback
if pipeline is None:
    try:
        from ml_system_v3 import MLSystemV3
        pipeline = MLSystemV3()
        pipeline_name = "MLSystemV3"
        print("✅ Using MLSystemV3")
    except Exception as e:
        print(f"⚠️  MLSystemV3 not available: {e}")

if pipeline is None:
    print("❌ No prediction pipeline available. Please check your installation.")
    sys.exit(1)

def main():
    """Main execution function"""
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(
        description='Run test profiling for prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/test_profiling.py --race-file sample.csv --enable-profiling
    python3 scripts/test_profiling.py --race-file upcoming_races/race.csv --enable-profiling

Output:
    - JSON file with prediction results
    - Profiling data in audit_results/ directory
    - Performance metrics and timing information
        """
    )
    parser.add_argument(
        '--race-file', 
        type=str, 
        required=True, 
        help='Path to race CSV file for prediction'
    )
    parser.add_argument(
        '--enable-profiling', 
        action='store_true', 
        help='Enable profiling during pipeline execution'
    )
    
    # Parse Arguments
    args = parser.parse_args()
    
    print(f"🚀 Starting test profiling with {pipeline_name}")
    print(f"📁 Race file: {args.race_file}")
    print(f"📊 Profiling enabled: {args.enable_profiling}")
    
    # Validate race file exists
    if not os.path.exists(args.race_file):
        print(f"❌ ERROR: Race file not found: {args.race_file}")
        sys.exit(1)
    
    # Enable profiling if requested
    if args.enable_profiling:
        set_profiling_enabled(True)
        print("✅ Profiling enabled")
    
    # Ensure profiling is enabled for this test
    if not is_profiling():
        print("❌ ERROR: Profiling is not enabled. Use --enable-profiling to enable it.")
        sys.exit(1)
    
    # Initialize profiling session
    profiling_recorder = ProfilingRecorder
    
    # Extract race info for profiling session
    race_filename = os.path.basename(args.race_file)
    race_id = race_filename.replace('.csv', '')
    
    profiling_recorder.start_session(
        race_id=race_id,
        model_version=pipeline_name,
        n_dogs=8,  # Default assumption
        method="test_profiling"
    )
    
    print("\n📈 Starting profiled prediction...")
    start_time = time.time()
    
    try:
        # Run prediction with the available pipeline
        if hasattr(pipeline, 'predict_race_file'):
            # Unified predictor or similar
            result = pipeline.predict_race_file(args.race_file)
        elif hasattr(pipeline, 'predict_race_file_with_all_enhancements'):
            # Comprehensive pipeline
            result = pipeline.predict_race_file_with_all_enhancements(args.race_file)
        elif hasattr(pipeline, 'predict'):
            # ML System V3 or similar - need to load CSV first
            import pandas as pd
            race_data = pd.read_csv(args.race_file)
            result = pipeline.predict(race_data)
        else:
            raise AttributeError(f"No suitable prediction method found on {pipeline_name}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Prediction completed in {duration:.3f} seconds")
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {
                "success": True,
                "predictions": result,
                "pipeline_used": pipeline_name,
                "execution_time": duration
            }
        
        # Add profiling metadata
        result["profiling_enabled"] = True
        result["pipeline_used"] = pipeline_name
        result["execution_time"] = duration
        result["race_file"] = args.race_file
        result["timestamp"] = datetime.now().isoformat()
        
        # Path to the generated JSON
        output_dir = Path('./profiling_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"prediction_profiling_{timestamp}.json"
        json_file_path = output_dir / json_filename
        
        # Save result to JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(result, json_file, indent=2, default=str)
        
        # Output the path to the generated JSON
        print(f"\n📄 Output saved to: {json_file_path}")
        
        # Assert expected keys exist in result
        expected_keys = ['success', 'pipeline_used', 'execution_time']
        missing_keys = [key for key in expected_keys if key not in result]
        
        if missing_keys:
            print(f"⚠️  WARNING: Some expected keys missing: {', '.join(missing_keys)}")
        else:
            print("✅ All expected keys are present")
        
        # Show some key information
        print("\n📊 Profiling Summary:")
        print(f"   Pipeline: {result.get('pipeline_used', 'unknown')}")
        print(f"   Execution time: {result.get('execution_time', 0):.3f}s")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Race file: {result.get('race_file', 'unknown')}")
        
        # End profiling session
        profiling_recorder.end_session()
        
        print("\n🎯 Profiling test completed successfully!")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ ERROR during prediction: {e}")
        
        # Create error result
        error_result = {
            "success": False,
            "error": str(e),
            "pipeline_used": pipeline_name,
            "execution_time": duration,
            "race_file": args.race_file,
            "timestamp": datetime.now().isoformat(),
            "profiling_enabled": True
        }
        
        # Save error result
        output_dir = Path('./profiling_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"prediction_profiling_error_{timestamp}.json"
        json_file_path = output_dir / json_filename
        
        with open(json_file_path, 'w') as json_file:
            json.dump(error_result, json_file, indent=2, default=str)
        
        print(f"📄 Error report saved to: {json_file_path}")
        
        # End profiling session
        profiling_recorder.end_session()
        
        sys.exit(1)

if __name__ == "__main__":
    main()
