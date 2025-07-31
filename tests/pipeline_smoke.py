
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
from prediction_pipeline_v3 import PredictionPipelineV3
from unified_predictor import UnifiedPredictor

def main():
    results = {}
    test_race_file = "sample_data/test_race_1.csv"

    for predictor_class in [
        UnifiedPredictor,
        PredictionPipelineV3,
    ]:
        predictor_name = predictor_class.__name__
        results[predictor_name] = {}

        try:
            start_time = time.time()
            predictor = predictor_class()
            prediction = predictor.predict_race_file(test_race_file)
            end_time = time.time()

            results[predictor_name]["status"] = "pass"
            results[predictor_name]["execution_time"] = end_time - start_time
            results[predictor_name]["win_prob"] = (
                prediction["predictions"][0]["win_probability"]
                if "win_probability" in prediction["predictions"][0]
                else None
            )
            results[predictor_name]["place_prob"] = (
                prediction["predictions"][0]["place_probability"]
                if "place_probability" in prediction["predictions"][0]
                else None
            )
            results[predictor_name]["predicted_rank"] = (
                prediction["predictions"][0]["predicted_rank"]
                if "predicted_rank" in prediction["predictions"][0]
                else None
            )
            results[predictor_name]["warnings"] = None
        except Exception as e:
            results[predictor_name]["status"] = "fail"
            results[predictor_name]["error"] = str(e)
            results[predictor_name]["stack_trace"] = traceback.format_exc()

    with open("tests/results/pipeline_smoke.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

