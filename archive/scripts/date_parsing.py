"""
Date parsing utilities for flexible format handling.

This module provides a helper function to parse dates in multiple formats
with proper fallback mechanisms as required by the system.
"""

from datetime import datetime


def parse_date_flexible(date_str):
    """
    Parse a date string with flexible format support.

    Attempts parsing with the primary format '%d %B %Y' first,
    then falls back to '%Y-%m-%d' if that raises ValueError.
    Always returns the result formatted as '%Y-%m-%d'.

    Args:
        date_str (str): The date string to parse

    Returns:
        str: Formatted date string in '%Y-%m-%d' format

    Raises:
        ValueError: If both parsing attempts fail

    Examples:
        >>> parse_date_flexible("25 July 2025")
        '2025-07-25'
        >>> parse_date_flexible("2025-07-25")
        '2025-07-25'
    """
    if not date_str or str(date_str).strip() == "":
        raise ValueError("Empty date string provided")

    date_str = str(date_str).strip()

    try:
        # Primary format: '%d %B %Y' (e.g., "25 July 2025")
        date_obj = datetime.strptime(date_str, "%d %B %Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Fallback format: '%Y-%m-%d' (e.g., "2025-07-25")
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Unable to parse date '{date_str}' with formats '%d %B %Y' or '%Y-%m-%d'"
            )
