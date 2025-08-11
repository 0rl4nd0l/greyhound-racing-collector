#!/usr/bin/env python3
"""
Step 7: Run full system smoke test for PredictionPipelineV3

This script executes `PredictionPipelineV3.predict_race_file()` on both a fixture
and a real live race file to verify:
- Weather-enhanced predictor executes (log shows ✔)
- Output probabilities come from real ML (no "Fallback prediction" text)
- Capture JSON output to `docs/sample_output_after_fix.json`
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Setup logging to capture system status
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('smoke_test_prediction_pipeline_v3.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main smoke test execution"""
    logger.info("🚀 Starting Step 7: Full System Smoke Test for PredictionPipelineV3")
    
    # Import PredictionPipelineV3
    try:
        from prediction_pipeline_v3 import PredictionPipelineV3
        logger.info("✅ Successfully imported PredictionPipelineV3")
    except ImportError as e:
        logger.error(f"❌ Failed to import PredictionPipelineV3: {e}")
        return False
    
    # Initialize the pipeline
    try:
        pipeline = PredictionPipelineV3()
        logger.info("✅ Successfully initialized PredictionPipelineV3")
    except Exception as e:
        logger.error(f"❌ Failed to initialize PredictionPipelineV3: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # Test files to use
    test_files = [
        # Test fixture
        {
            "path": "tests/fixtures/test_race.csv",
            "type": "fixture",
            "description": "Test fixture file"
        },
        # Real race file from processed data
        {
            "path": "processed/completed/Race 1 - GARD - 04 July 2025.csv",
            "type": "real_race",
            "description": "Real race file from processed data"
        }
    ]
    
    results = {}
    
    for test_file in test_files:
        file_path = test_file["path"]
        file_type = test_file["type"]
        description = test_file["description"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {description}: {file_path}")
        logger.info(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ File not found: {file_path} - Skipping")
            continue
        
        # Run prediction
        try:
            logger.info(f"🔄 Running prediction on {file_path}")
            result = pipeline.predict_race_file(file_path)
            
            if result.get("success", False):
                logger.info("✅ Prediction completed successfully!")
                
                # Check for weather-enhanced predictor execution
                prediction_tier = result.get("prediction_tier", "unknown")
                logger.info(f"🎯 Prediction tier used: {prediction_tier}")
                
                if prediction_tier == "weather_enhanced":
                    logger.info("✅ Weather-enhanced predictor executed successfully!")
                elif prediction_tier == "comprehensive_pipeline":
                    logger.info("✅ Comprehensive pipeline executed (includes weather enhancement)!")
                else:
                    logger.warning(f"⚠️ Used fallback tier: {prediction_tier}")
                
                # Check for real ML output (no fallback text)
                predictions = result.get("predictions", [])
                has_fallback_text = any(
                    "Fallback prediction" in str(pred) 
                    for pred in predictions
                )
                
                if not has_fallback_text:
                    logger.info("✅ Output uses real ML predictions (no 'Fallback prediction' text)")
                else:
                    logger.warning("⚠️ Output contains fallback prediction text")
                
                # Log prediction summary
                logger.info(f"📊 Generated {len(predictions)} predictions")
                if predictions:
                    top_prediction = predictions[0]
                    logger.info(f"🥇 Top prediction: {top_prediction.get('dog_name', 'Unknown')} "
                              f"(Win prob: {top_prediction.get('win_probability', 0):.3f})")
                
                # Store result
                results[file_type] = {
                    "file_path": file_path,
                    "success": True,
                    "prediction_tier": prediction_tier,
                    "num_predictions": len(predictions),
                    "weather_enhanced": prediction_tier in ["weather_enhanced", "comprehensive_pipeline"],
                    "real_ml_output": not has_fallback_text,
                    "result": result
                }
                
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Prediction failed: {error_msg}")
                results[file_type] = {
                    "file_path": file_path,
                    "success": False,
                    "error": error_msg,
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"❌ Exception during prediction: {e}")
            logger.error(traceback.format_exc())
            results[file_type] = {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "exception": traceback.format_exc()
            }
    
    # Generate summary report
    logger.info(f"\n{'='*60}")
    logger.info("📋 SMOKE TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    total_tests = len(results)
    
    logger.info(f"✅ Successful tests: {successful_tests}/{total_tests}")
    
    weather_enhanced_count = sum(1 for r in results.values() 
                                if r.get("weather_enhanced", False))
    logger.info(f"🌤️ Weather-enhanced predictions: {weather_enhanced_count}/{successful_tests}")
    
    real_ml_count = sum(1 for r in results.values() 
                       if r.get("real_ml_output", False))
    logger.info(f"🤖 Real ML output (no fallback): {real_ml_count}/{successful_tests}")
    
    # Save detailed results to docs folder
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    output_file = docs_dir / "sample_output_after_fix.json"
    
    # Prepare output data
    output_data = {
        "smoke_test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "weather_enhanced_count": weather_enhanced_count,
                "real_ml_output_count": real_ml_count
            }
        },
        "test_results": results
    }
    
    # Save to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"💾 Detailed results saved to: {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    # Final status
    if successful_tests == total_tests and weather_enhanced_count > 0 and real_ml_count > 0:
        logger.info("🎉 SMOKE TEST PASSED - All requirements met!")
        return True
    else:
        logger.warning("⚠️ SMOKE TEST PARTIAL - Some requirements not fully met")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
