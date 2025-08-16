"""
File naming utilities for prediction outputs.
Provides standardized filename generation for prediction results.
"""

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple


def build_prediction_filename(
    race_id: str, ts: Optional[datetime] = None, method: str = "ml"
) -> str:
    """
    Build a standardized prediction filename.

    Args:
        race_id: The race identifier (e.g., "Race_01_MAN_2025-01-15")
        ts: Timestamp for the prediction (defaults to current time)
        method: Prediction method identifier (e.g., "ml", "ensemble", "v3")

    Returns:
        str: Standardized filename in format: prediction_<race_id>_<method>_<YYYYMMDD_HHMMSS>.json

    Examples:
        >>> build_prediction_filename("Race_01_MAN_2025-01-15", method="ml")
        'prediction_Race_01_MAN_2025-01-15_ml_20250115_143022.json'
        >>> build_prediction_filename("Race_05_ALB_2025-01-16", method="ensemble")
        'prediction_Race_05_ALB_2025-01-16_ensemble_20250115_143022.json'
    """
    if ts is None:
        ts = datetime.now()

    # Clean race_id: remove any .csv extension and sanitize
    clean_race_id = race_id.replace(".csv", "").replace(".json", "")

    # Sanitize race_id and method to ensure they're filesystem-safe
    clean_race_id = sanitize_filename_component(clean_race_id)
    clean_method = sanitize_filename_component(method)

    # Format timestamp as YYYYMMDD_HHMMSS
    timestamp_str = ts.strftime("%Y%m%d_%H%M%S")

    # Build the standardized filename
    filename = f"prediction_{clean_race_id}_{clean_method}_{timestamp_str}.json"

    return filename


def sanitize_filename_component(component: str) -> str:
    """
    Sanitize a filename component to be filesystem-safe.

    Args:
        component: String component to sanitize

    Returns:
        str: Sanitized component safe for use in filenames
    """
    # Remove or replace problematic characters
    # Keep alphanumeric, hyphens, underscores, and periods
    sanitized = re.sub(r"[^\w\-_.]", "_", component)

    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_{2,}", "_", sanitized)

    # Strip leading/trailing underscores and periods
    sanitized = sanitized.strip("_.")

    return sanitized


def parse_prediction_filename(filename: str) -> dict:
    """
    Parse a standardized prediction filename to extract components.

    Args:
        filename: The prediction filename to parse

    Returns:
        dict: Dictionary containing parsed components:
            - race_id: str
            - method: str
            - timestamp: datetime
            - is_valid: bool

    Examples:
        >>> parse_prediction_filename("prediction_Race_01_MAN_2025-01-15_ml_20250115_143022.json")
        {
            'race_id': 'Race_01_MAN_2025-01-15',
            'method': 'ml',
            'timestamp': datetime(2025, 1, 15, 14, 30, 22),
            'is_valid': True
        }
    """
    # Expected pattern: prediction_<race_id>_<method>_<YYYYMMDD_HHMMSS>.json
    # Since race_id can contain underscores, we need to be careful with parsing
    pattern = r"^prediction_(.+)_([^_]+)_(\d{8}_\d{6})\.json$"

    match = re.match(pattern, filename)

    if not match:
        return {"race_id": "", "method": "", "timestamp": None, "is_valid": False}

    race_id, method, timestamp_str = match.groups()

    # Additional validation to ensure race_id and method are not empty
    # Also check for race_id starting/ending with underscore (indicates malformed filename)
    if (
        not race_id.strip()
        or not method.strip()
        or race_id.startswith("_")
        or race_id.endswith("_")
    ):
        return {
            "race_id": race_id,
            "method": method,
            "timestamp": None,
            "is_valid": False,
        }

    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return {
            "race_id": race_id,
            "method": method,
            "timestamp": None,
            "is_valid": False,
        }

    return {
        "race_id": race_id,
        "method": method,
        "timestamp": timestamp,
        "is_valid": True,
    }


def is_valid_prediction_filename(filename: str) -> bool:
    """
    Check if a filename follows the standardized prediction naming convention.

    Args:
        filename: The filename to validate

    Returns:
        bool: True if filename follows the standard format
    """
    return parse_prediction_filename(filename)["is_valid"]


def extract_race_id_from_csv_filename(csv_filename: str) -> str:
    """
    Extract race ID from a CSV filename for use in prediction filenames.

    Args:
        csv_filename: The CSV filename (e.g., "Race_01_MAN_2025-01-15.csv")

    Returns:
        str: Race ID suitable for prediction filename
    """
    # Remove .csv extension if present
    race_id = csv_filename.replace(".csv", "")

    # If filename starts with path, get just the basename
    race_id = os.path.basename(race_id)

    return sanitize_filename_component(race_id)


def ensure_unique_filename(directory: str, filename: str) -> str:
    """
    Ensure filename is unique in the given directory by adding a suffix if needed.

    Args:
        directory: Directory where the file will be saved
        filename: Proposed filename

    Returns:
        str: Unique filename (may have suffix added)
    """
    if not os.path.exists(directory):
        return filename

    full_path = os.path.join(directory, filename)

    if not os.path.exists(full_path):
        return filename

    # File exists, need to make it unique
    base_name, ext = os.path.splitext(filename)
    counter = 1

    while True:
        new_filename = f"{base_name}_{counter:03d}{ext}"
        new_full_path = os.path.join(directory, new_filename)

        if not os.path.exists(new_full_path):
            return new_filename

        counter += 1

        # Safety check to prevent infinite loop
        if counter > 999:
            # Use timestamp microseconds for uniqueness
            timestamp_suffix = datetime.now().strftime("%f")
            return f"{base_name}_{timestamp_suffix}{ext}"


def get_filename_for_race_id(
    race_id: str, search_paths: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the filename for a given race_id by searching in specified directories.
    
    This function prevents duplication of filename search logic across endpoints
    by providing a centralized way to locate race files based on race_id.
    
    Args:
        race_id: The race identifier to search for
        search_paths: List of directories to search in. If None, uses default paths.
                     Default paths are ['./upcoming_races', './historical_races']
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (filename, full_path) if found, (None, None) if not found
        
    Examples:
        >>> get_filename_for_race_id("Race_01_MAN_2025-01-15")
        ("Race_01_MAN_2025-01-15.csv", "./upcoming_races/Race_01_MAN_2025-01-15.csv")
        
        >>> get_filename_for_race_id("NonExistent", ["./custom_dir"])
        (None, None)
    """
    if search_paths is None:
        search_paths = ["./upcoming_races", "./historical_races"]
    
    # Generate possible filename patterns that might match the race_id
    possible_filenames = _generate_filename_patterns(race_id)
    
    # Search in each directory
    for directory in search_paths:
        if not os.path.exists(directory):
            continue
            
        # First try exact matches
        for filename_candidate in possible_filenames:
            candidate_path = os.path.join(directory, filename_candidate)
            if os.path.exists(candidate_path):
                return filename_candidate, candidate_path
        
        # If no exact match, try partial matches
        filename, full_path = _search_partial_matches(directory, race_id)
        if filename:
            return filename, full_path
    
    return None, None


def _generate_filename_patterns(race_id: str) -> List[str]:
    """
    Generate possible filename patterns for a given race_id.
    
    Args:
        race_id: The race identifier
        
    Returns:
        List[str]: List of possible filename patterns
    """
    # Clean the race_id first
    clean_race_id = race_id.replace(".csv", "").replace(".json", "")
    
    patterns = [
        f"{clean_race_id}.csv",
        f"Race {clean_race_id}.csv",
        f"Race_{clean_race_id}.csv",
        f"Race-{clean_race_id}.csv",
    ]
    
    # Add patterns with spaces replaced by underscores and vice versa
    if " " in clean_race_id:
        underscore_version = clean_race_id.replace(" ", "_")
        patterns.extend([
            f"{underscore_version}.csv",
            f"Race_{underscore_version}.csv",
            f"Race-{underscore_version}.csv",
        ])
    
    if "_" in clean_race_id:
        space_version = clean_race_id.replace("_", " ")
        patterns.extend([
            f"{space_version}.csv",
            f"Race {space_version}.csv",
            f"Race_{space_version}.csv",
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)
    
    return unique_patterns


def _search_partial_matches(directory: str, race_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Search for files that contain the race_id as a substring.
    
    Args:
        directory: Directory to search in
        race_id: Race identifier to search for
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (filename, full_path) if found, (None, None) if not found
    """
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and race_id in filename:
                full_path = os.path.join(directory, filename)
                return filename, full_path
    except (OSError, PermissionError):
        # Directory access issues, skip silently
        pass
    
    return None, None


def get_race_id_from_filename(filename: str) -> str:
    """
    Extract race_id from a filename, handling various formats.
    
    This is the inverse operation of get_filename_for_race_id.
    
    Args:
        filename: The filename to extract race_id from
        
    Returns:
        str: Extracted race_id
        
    Examples:
        >>> get_race_id_from_filename("Race_01_MAN_2025-01-15.csv")
        "Race_01_MAN_2025-01-15"
        
        >>> get_race_id_from_filename("Race 5 - GEE - 22 July 2025.csv")
        "Race_5_-_GEE_-_22_July_2025"
    """
    # Remove path components
    basename = os.path.basename(filename)
    
    # Remove file extension
    race_id = os.path.splitext(basename)[0]
    
    # Remove common prefixes
    if race_id.startswith("Race "):
        race_id = race_id[5:]  # Remove "Race " prefix
    elif race_id.startswith("Race_"):
        race_id = race_id[5:]  # Remove "Race_" prefix
    elif race_id.startswith("Race-"):
        race_id = race_id[5:]  # Remove "Race-" prefix
    
    # Sanitize the result
    return sanitize_filename_component(race_id)
