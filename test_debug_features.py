#!/usr/bin/env python3
"""
Debug Test Script for Enhanced Pipeline V2
==========================================

This script runs the enhanced pipeline with debug logging enabled to investigate
why certain features are showing zero values.
"""

import logging
import sys
from pathlib import Path
from enhanced_pipeline_v2 import EnhancedPipelineV2

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_feature_extraction.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 Starting debug analysis of feature extraction...")
    
    # Initialize the enhanced pipeline
    pipeline = EnhancedPipelineV2()
    
    # Test file path
    test_file = Path("upcoming_races/Race_1_-_TAREE_-_2025-07-26.csv")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"📋 Analyzing race file: {test_file}")
    
    # Run prediction with debug logging
    result = pipeline.predict_race_file(str(test_file))
    
    if result['success']:
        logger.info("✅ Prediction completed successfully")
        logger.info(f"📊 Total dogs analyzed: {len(result['predictions'])}")
        
        # Print summary of key features for each dog
        logger.info("\n🔍 FEATURE ANALYSIS SUMMARY:")
        logger.info("=" * 80)
        
        for i, prediction in enumerate(result['predictions']):
            dog_name = prediction['dog_name']
            logger.info(f"\n🐕 {i+1}. {dog_name} (Box {prediction['box_number']})")
            logger.info(f"   Prediction Score: {prediction['prediction_score']}")
            logger.info(f"   Confidence: {prediction['confidence_level']} ({prediction['confidence_score']:.3f})")
            logger.info(f"   Reasoning: {prediction['reasoning']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 Check the debug_feature_extraction.log file for detailed feature extraction logs")
        
    else:
        logger.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
