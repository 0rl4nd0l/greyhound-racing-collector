
import logging
import sys
from pathlib import Path
from enhanced_pipeline_v2 import EnhancedPipelineV2

# Redirect stdout and stderr to a log file
sys.stdout = open('debug_output.log', 'w')
sys.stderr = sys.stdout

# Set up basic logging to also go to the file
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Pipeline Debug Run ---")
    pipeline = EnhancedPipelineV2()
    test_file = Path("upcoming_races/Race_1_-_TAREE_-_2025-07-26.csv")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    logger.info(f"Analyzing race file: {test_file}")
    result = pipeline.predict_race_file(str(test_file))
    
    if result.get('success'):
        logger.info("--- Prediction Completed Successfully ---")
        # Pretty print the final result
        import json
        logger.info(json.dumps(result, indent=4))
    else:
        logger.error(f"--- Prediction Failed ---")
        logger.error(result.get('error', 'Unknown error'))

    logger.info("--- Debug Run Finished ---")

if __name__ == "__main__":
    main()

