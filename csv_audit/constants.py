"""
Constants and configuration for CSV audit operations.

This module defines the canonical header list, outcome columns used for leakage tests,
and other configuration constants for consistent CSV processing across all modules.
"""

from typing import List, Set

# Canonical header list (in order)
CANONICAL_HEADERS = [
    "ID",
    "TIMESTAMP",
    "USER_ID", 
    "SESSION_ID",
    "PLC",
    "TIME",
    "BON",
    "SCORE",
    "OUTCOME",
    "STATUS",
    "CATEGORY",
    "SUBCATEGORY",
    "NOTES",
    "METADATA"
]

# Required outcome columns used for leakage tests
OUTCOME_COLUMNS = {
    "PLC",    # Primary Leakage Column
    "TIME",   # Time-based outcome
    "BON",    # Bonus/Binary Outcome Normalized
    "SCORE",  # Score-based outcome
    "OUTCOME" # General outcome column
}

# CSV processing configuration
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"

# Typical dog-block length range (6-8 records per block)
DOG_BLOCK_MIN_LENGTH = 6
DOG_BLOCK_MAX_LENGTH = 8
DOG_BLOCK_TYPICAL_LENGTH = 7


def is_header_compliant(cols: List[str]) -> bool:
    """
    Check if provided column list matches the canonical header specification.
    
    Args:
        cols: List of column names to validate
        
    Returns:
        bool: True if headers match canonical order and content, False otherwise
    """
    if not isinstance(cols, list):
        return False
    
    # Convert to list if needed and strip whitespace
    normalized_cols = [str(col).strip() for col in cols]
    
    # Check exact match with canonical headers
    return normalized_cols == CANONICAL_HEADERS


def is_outcome_col(col: str) -> bool:
    """
    Check if a column is designated as an outcome column for leakage tests.
    
    Args:
        col: Column name to check
        
    Returns:
        bool: True if column is an outcome column, False otherwise
    """
    if not isinstance(col, str):
        return False
    
    return col.strip().upper() in OUTCOME_COLUMNS


def get_canonical_headers() -> List[str]:
    """
    Get a copy of the canonical headers list.
    
    Returns:
        List[str]: Copy of canonical headers in order
    """
    return CANONICAL_HEADERS.copy()


def get_outcome_columns() -> Set[str]:
    """
    Get a copy of the outcome columns set.
    
    Returns:
        Set[str]: Copy of outcome columns set
    """
    return OUTCOME_COLUMNS.copy()


def is_valid_dog_block_length(length: int) -> bool:
    """
    Check if a block length is within the typical dog-block range.
    
    Args:
        length: Length of the block to validate
        
    Returns:
        bool: True if length is within typical range (6-8), False otherwise
    """
    return DOG_BLOCK_MIN_LENGTH <= length <= DOG_BLOCK_MAX_LENGTH
