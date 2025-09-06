#!/usr/bin/env python3
"""
JSON utilities for handling NaN values safely in prediction generation.

This module provides utilities to ensure that NaN values are properly
handled when writing JSON files, preventing the creation of invalid JSON.
"""

import json
import math

import numpy as np


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that safely handles NaN, Infinity, and other
    problematic values by converting them to null.
    """

    def encode(self, obj):
        """Override encode to handle NaN values"""
        return super().encode(self._sanitize_value(obj))

    def iterencode(self, obj, _one_shot=False):
        """Override iterencode to handle NaN values"""
        return super().iterencode(self._sanitize_value(obj), _one_shot=_one_shot)

    def _sanitize_value(self, obj):
        """Recursively sanitize values, replacing NaN/Infinity with None"""
        if isinstance(obj, dict):
            return {key: self._sanitize_value(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_value(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_value(item) for item in obj)
        elif isinstance(obj, (float, int)):
            # Handle NaN and Infinity
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.floating):
            # Handle numpy floating point types
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.integer):
            # Handle numpy integer types
            return int(obj)
        elif hasattr(obj, "item") and callable(getattr(obj, "item")):
            # Handle numpy scalars
            item_val = obj.item()
            if isinstance(item_val, (float, int)) and (
                math.isnan(item_val) or math.isinf(item_val)
            ):
                return None
            return item_val
        else:
            return obj


def safe_json_dump(obj, fp, **kwargs):
    """
    Safe version of json.dump that handles NaN values.

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        **kwargs: Additional arguments passed to json.dump
    """
    # Ensure we use our safe encoder
    kwargs["cls"] = SafeJSONEncoder
    # Ensure ASCII is False for better Unicode support
    kwargs.setdefault("ensure_ascii", False)
    # Pretty print by default
    kwargs.setdefault("indent", 2)

    return json.dump(obj, fp, **kwargs)


def safe_json_dumps(obj, **kwargs):
    """
    Safe version of json.dumps that handles NaN values.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string with NaN values converted to null
    """
    # Ensure we use our safe encoder
    kwargs["cls"] = SafeJSONEncoder
    # Ensure ASCII is False for better Unicode support
    kwargs.setdefault("ensure_ascii", False)
    # Pretty print by default
    kwargs.setdefault("indent", 2)

    return json.dumps(obj, **kwargs)


def safe_float(value, default=None):
    """
    Convert a value to float, handling NaN/Infinity cases safely.

    Args:
        value: Value to convert
        default: Default value to return if conversion fails or value is NaN/inf

    Returns:
        Float value or default if problematic
    """
    try:
        if value is None:
            return default

        float_val = float(value)

        if math.isnan(float_val) or math.isinf(float_val):
            return default

        return float_val
    except (ValueError, TypeError, OverflowError):
        return default


def safe_mean(values, default=None):
    """
    Calculate mean of values, handling empty lists and NaN results safely.

    Args:
        values: List/array of values
        default: Default value to return if calculation fails

    Returns:
        Mean value or default if problematic
    """
    try:
        if not values:
            return default

        # Filter out None/NaN values
        clean_values = []
        for val in values:
            if val is not None:
                try:
                    float_val = float(val)
                    if not (math.isnan(float_val) or math.isinf(float_val)):
                        clean_values.append(float_val)
                except (ValueError, TypeError):
                    continue

        if not clean_values:
            return default

        mean_val = sum(clean_values) / len(clean_values)

        if math.isnan(mean_val) or math.isinf(mean_val):
            return default

        return mean_val
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def safe_correlation(x_values, y_values, default=None):
    """
    Calculate correlation coefficient safely, handling edge cases.

    Args:
        x_values: First array of values
        y_values: Second array of values
        default: Default value to return if calculation fails

    Returns:
        Correlation coefficient or default if problematic
    """
    try:
        if not x_values or not y_values or len(x_values) != len(y_values):
            return default

        # Clean and align the data
        clean_pairs = []
        for x, y in zip(x_values, y_values):
            try:
                x_float = float(x) if x is not None else None
                y_float = float(y) if y is not None else None

                if (
                    x_float is not None
                    and y_float is not None
                    and not math.isnan(x_float)
                    and not math.isnan(y_float)
                    and not math.isinf(x_float)
                    and not math.isinf(y_float)
                ):
                    clean_pairs.append((x_float, y_float))
            except (ValueError, TypeError):
                continue

        if len(clean_pairs) < 2:  # Need at least 2 points for correlation
            return default

        x_clean = [pair[0] for pair in clean_pairs]
        y_clean = [pair[1] for pair in clean_pairs]

        # Calculate correlation manually to avoid numpy dependency
        n = len(x_clean)
        sum_x = sum(x_clean)
        sum_y = sum(y_clean)
        sum_xy = sum(x * y for x, y in clean_pairs)
        sum_x2 = sum(x * x for x in x_clean)
        sum_y2 = sum(y * y for y in y_clean)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt(
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        )

        if denominator == 0:
            return default

        correlation = numerator / denominator

        if math.isnan(correlation) or math.isinf(correlation):
            return default

        return correlation
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return default


# Example usage and testing
if __name__ == "__main__":
    # Test data with problematic values
    test_data = {
        "normal_value": 1.5,
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "negative_inf": float("-inf"),
        "nested": {
            "list_with_nan": [1, 2, float("nan"), 4],
            "normal_list": [1, 2, 3, 4],
        },
        "numpy_nan": np.nan if "np" in globals() else None,
    }

    print("Original data (problematic):")
    print(f"NaN value: {test_data['nan_value']}")
    print(f"Inf value: {test_data['inf_value']}")

    print("\nSafe JSON output:")
    safe_json = safe_json_dumps(test_data)
    print(safe_json)

    print("\nSafe utility functions:")
    print(f"safe_float(NaN): {safe_float(float('nan'), 'default')}")
    print(f"safe_mean([1,2,NaN,4]): {safe_mean([1, 2, float('nan'), 4], 'default')}")
    print(
        f"safe_correlation([1,2],[NaN,4]): {safe_correlation([1, 2], [float('nan'), 4], 'default')}"
    )
