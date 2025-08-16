#!/usr/bin/env python3
"""
JSON Utilities Module
====================

Safe utility functions for JSON operations and data processing.
"""

import json
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd


def safe_json_dump(data: Any, filepath: str, indent: int = 2) -> bool:
    """Safely dump data to JSON file with error handling"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"❌ Error writing JSON file {filepath}: {e}")
        return False


def safe_mean(values: List[Union[int, float]]) -> float:
    """Calculate mean with safe handling of invalid values"""
    try:
        if not values:
            return 0.0

        # Filter out None, NaN, and invalid values
        clean_values = []
        for val in values:
            if val is not None and not pd.isna(val):
                try:
                    float_val = float(val)
                    if not np.isinf(float_val):
                        clean_values.append(float_val)
                except (ValueError, TypeError):
                    continue

        return np.mean(clean_values) if clean_values else 0.0
    except Exception:
        return 0.0


def safe_correlation(x: List[Union[int, float]], y: List[Union[int, float]]) -> float:
    """Calculate correlation with safe handling of invalid values"""
    try:
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        # Clean data
        clean_pairs = []
        for i in range(len(x)):
            if (
                x[i] is not None
                and y[i] is not None
                and not pd.isna(x[i])
                and not pd.isna(y[i])
            ):
                try:
                    x_val = float(x[i])
                    y_val = float(y[i])
                    if not np.isinf(x_val) and not np.isinf(y_val):
                        clean_pairs.append((x_val, y_val))
                except (ValueError, TypeError):
                    continue

        if len(clean_pairs) < 2:
            return 0.0

        x_clean, y_clean = zip(*clean_pairs)
        correlation = np.corrcoef(x_clean, y_clean)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with safe error handling"""
    try:
        if value is None or pd.isna(value):
            return default

        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in ["nan", "none", "null"]:
                return default

        float_val = float(value)
        return float_val if not np.isinf(float_val) else default
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int with safe error handling"""
    try:
        return int(safe_float(value, default))
    except (ValueError, TypeError):
        return default


def clean_data_dict(data: dict) -> dict:
    """Clean dictionary by replacing NaN and None values"""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_data_dict(value)
        elif isinstance(value, list):
            cleaned[key] = [
                clean_data_dict(item) if isinstance(item, dict) else safe_float(item)
                for item in value
            ]
        else:
            cleaned[key] = (
                safe_float(value) if isinstance(value, (int, float)) else value
            )
    return cleaned


def format_prediction_output(prediction_data: dict) -> dict:
    """Format prediction data for consistent output"""
    formatted = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "race_info": prediction_data.get("race_info", {}),
        "predictions": [],
        "model_info": prediction_data.get("model_info", {}),
        "performance_metrics": prediction_data.get("performance_metrics", {}),
    }

    for pred in prediction_data.get("predictions", []):
        formatted_pred = {
            "dog_name": pred.get("dog_name", ""),
            "box_number": safe_int(pred.get("box_number")),
            "win_probability": safe_float(pred.get("win_probability")),
            "place_probability": safe_float(pred.get("place_probability")),
            "predicted_position": safe_int(pred.get("predicted_position")),
            "confidence_score": safe_float(pred.get("confidence_score")),
            "recommendation": pred.get("recommendation", "NEUTRAL"),
        }
        formatted["predictions"].append(formatted_pred)

    return formatted
