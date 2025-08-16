#!/usr/bin/env python3

import sys
import os
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from prediction_pipeline_v3 import PredictionPipelineV3
from profiling_config import set_profiling_enabled

# Enable profiling
set_profiling_enabled(True)

# Load pipeline
pipeline = PredictionPipelineV3()

# Sample race file (existence assumed)
race_file = 'archive/corrupt_or_legacy_race_files/20250730162231_upcoming_race_sample.csv'

# Run prediction
result = pipeline.predict_race_file(race_file, enhancement_level='full')

# Print the result
print(json.dumps(result, indent=2))

