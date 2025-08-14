"""
Project Description for AI Coding Copilot:
This Python script implements the Prediction Assistant for greyhound race winner prediction.
- **Purpose**: Predict winners for a new race by scoring greyhounds based on metrics and patterns from the Analysis Assistant's 'analysis_output.json', generated from up to 900 historical form guides and 'navigator_race_results.csv'.
- **Input**:
  - New form guide CSV (e.g., './form_guides/new_race.csv') with fields: Dog Name, Sex, PLC, BOX, WGT, DIST, TIME, DATE, TRACK, G, WIN, BON, 1 SEC, MGN, W/2G, PIR, SP.
  - 'analysis_output.json' with metrics (e.g., time_per_meter, sectional_efficiency), win rates (box, sex, venue-distance, trainer), feature importance, and patterns.
- **Output**: Ranked list of greyhounds with confidence scores and key factors (e.g., high sectional efficiency).
- **Key Features**:
  - Calculates metrics for each greyhound in the new race (e.g., time per meter, PIR trends).
  - Scores greyhounds using feature importance and win rates from Analysis JSON.
  - Uses GPT-4o-mini for cost-efficient prediction, as metrics are pre-computed.
  - Handles folder-based CSV input, bypassing OpenAI Playground's CSV upload limitation.
- **Dependencies**: Python 3.8+, pandas, numpy, openai.
- **Setup**: Set OPENAI_API_KEY, place new form guide in './form_guides/', ensure 'analysis_output.json' exists.
- **Cost**: ~$0.14 for 100 runs (5,000 input tokens, 1,000 output tokens per run, GPT-4o-mini: $0.15/1M input, $0.60/1M output).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# File paths
FORM_GUIDES_DIR = "./form_guides/"  # Directory for new form guide CSV
ANALYSIS_JSON = "../ultra_insights.json"  # Ultra Insights analysis output


def clean_dog_name(name):
    """
    Clean dog names for consistent matching
    - Input: name (str or None), e.g., "CANYA ALL CLASS"
    - Output: cleaned name (str), e.g., "canyaallclass"
    """
    return (
        "".join(c for c in name if c.isalnum()).lower()
        if isinstance(name, str)
        else name
    )


def parse_pir(pir):
    """
    Parse PIR (Position in Running)
    - Input: pir (str or None), e.g., "111"
    - Output: list of positions [start, mid, finish], e.g., [1, 1, 1] or [nan, nan, nan]
    """
    return (
        [int(p) for p in pir]
        if isinstance(pir, str) and pir.isdigit()
        else [np.nan] * 3
    )


def get_weight_bracket(weight):
    """
    Determine weight bracket based on ultra_insights.json categories
    """
    if weight < 27.5:
        return "Light"
    elif weight < 30.0:
        return "Medium_Light"
    elif weight < 32.0:
        return "Medium"
    elif weight < 34.0:
        return "Medium_Heavy"
    else:
        return "Heavy"


def get_weight_bracket_performance(ultra_insights, weight_bracket):
    """
    Get win rate for weight bracket from ultra_insights.json
    """
    for bracket in ultra_insights["weight_impact"]["weight_bracket_performance"]:
        if bracket["weight_bracket"] == weight_bracket:
            return bracket["win_rate"]
    return 0.15  # Default win rate


def calculate_metrics(form_data):
    """
    Calculate metrics for a greyhound based on ultra_insights.json features
    - Input: form_data (pandas DataFrame), subset for one greyhound
    - Output: dict of metrics matching ultra_insights feature importance
    """
    recent_races = form_data.sort_values("DATE", ascending=False).head(5)

    # Calculate relative time (compared to average race time)
    avg_time = recent_races["TIME"].mean()
    relative_time = (
        avg_time / recent_races["DIST"].mean() if recent_races["DIST"].mean() > 0 else 0
    )

    # Calculate early speed from sectional times
    early_speed = (
        (recent_races["DIST"] / recent_races["1 SEC"]).mean()
        if recent_races["1 SEC"].mean() > 0
        else 0
    )

    # Calculate relative weight (compared to average weight in dataset)
    avg_weight = recent_races["WGT"].mean()
    relative_weight = avg_weight / 31.0  # Normalize against typical greyhound weight

    # Calculate average speed
    avg_speed = (recent_races["DIST"] / recent_races["TIME"]).mean()

    metrics = {
        "relative_time": relative_time,
        "early_speed": early_speed,
        "relative_weight": relative_weight,
        "individual_time_numeric": recent_races["TIME"].mean(),
        "avg_speed": avg_speed,
        "box_number": recent_races["BOX"].iloc[0] if len(recent_races) > 0 else 1,
        "sectional_1_numeric": recent_races["1 SEC"].mean(),
        "recent_form_score": np.average(
            recent_races["PLC"].fillna(8),
            weights=[0.5, 0.3, 0.2, 0.1, 0.05][: len(recent_races)],
        ),
        "sp_mean": recent_races["SP"].mean(),
    }
    return metrics


def run_prediction_assistant(form_guide_path):
    """
    Run Prediction Assistant using ultra_insights.json
    - Input: form_guide_path (str), path to new race CSV
    - Output: list of dicts, ranked predictions with confidence scores
    """
    # Load Ultra Insights JSON
    with open(ANALYSIS_JSON, "r") as f:
        ultra_insights = json.load(f)

    # Load new form guide
    form_data = pd.read_csv(form_guide_path)
    form_data["dog_clean"] = form_data["Dog Name"].apply(clean_dog_name)

    # Calculate scores for each greyhound
    predictions = []
    for dog in sorted(form_data["dog_clean"].unique()):  # Sort for deterministic order
        dog_data = form_data[form_data["dog_clean"] == dog]
        metrics = calculate_metrics(dog_data)

        # Score based on ultra_insights feature importance
        score = 0
        for fi in ultra_insights["ultra_predictive_model"]["feature_importance"]:
            feature = fi["feature"]
            if feature in metrics and not np.isnan(metrics[feature]):
                # Apply feature importance directly
                score += fi["importance"] * metrics[feature]

        # Adjust for weight bracket performance
        weight = dog_data["WGT"].iloc[0] if len(dog_data) > 0 else 30.0
        weight_bracket = get_weight_bracket(weight)
        weight_performance = get_weight_bracket_performance(
            ultra_insights, weight_bracket
        )
        score += weight_performance * 0.15

        # Adjust for speed analysis insights
        if "avg_speed" in metrics:
            # Bonus for high speed (winners average higher early speed)
            if (
                metrics["early_speed"]
                > ultra_insights["speed_analysis"]["early_speed_analysis"][
                    "winners_avg_early_speed"
                ]
            ):
                score += 0.1

        # Add prediction
        top_feature = max(
            ultra_insights["ultra_predictive_model"]["feature_importance"],
            key=lambda x: x["importance"],
        )["feature"]
        predictions.append(
            {
                "greyhound_name": dog_data["Dog Name"].iloc[0],
                "confidence": score,
                "key_factors": f"Strong {top_feature} ({metrics.get(top_feature, 0):.4f})",
                "weight_bracket": weight_bracket,
                "predicted_speed": metrics.get("avg_speed", 0),
            }
        )

    # Normalize confidence scores
    total_score = sum(p["confidence"] for p in predictions)
    for p in predictions:
        p["confidence"] = (
            p["confidence"] / total_score if total_score > 0 else 1 / len(predictions)
        )

    # Return ranked predictions
    return sorted(predictions, key=lambda x: x["confidence"], reverse=True)


# Main execution
if __name__ == "__main__":
    # Check if ultra_insights.json exists in parent directory
    if not os.path.exists("../ultra_insights.json"):
        print("Error: ultra_insights.json not found in parent directory.")
        print(
            "Please ensure ultra_insights.json is in the greyhound_racing_collector directory."
        )
        exit(1)

    # Example: Predict for a new race
    new_form_guide = "./form_guides/example_race.csv"
    if os.path.exists(new_form_guide):
        predictions = run_prediction_assistant(new_form_guide)
        print("Predictions:", json.dumps(predictions, indent=2))
    else:
        print(f"Error: {new_form_guide} not found.")
        print("Available form guides:")
        if os.path.exists("./form_guides/"):
            for file in os.listdir("./form_guides/"):
                if file.endswith(".csv"):
                    print(f"  - {file}")
