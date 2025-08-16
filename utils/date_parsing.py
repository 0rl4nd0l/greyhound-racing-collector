"""
Date parsing utilities for flexible format handling.

This module provides a helper function to parse dates in multiple formats
with proper fallback mechanisms as required by the system.
Also provides utilities for historical race filtering.
"""

from datetime import datetime, date, timedelta


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


def is_historical(race_date):
    """
    Check if a race date is historical (before today).
    
    When --historical mode is set, this function determines if a race date
    should be considered for processing. Any date < today is considered historical.
    
    Args:
        race_date (str|datetime|date): The race date to check
    
    Returns:
        bool: True if the date is before today, False otherwise
        
    Examples:
        >>> is_historical("2024-01-01")
        True
        >>> is_historical(datetime(2024, 1, 1))
        True
        >>> is_historical(date.today() + timedelta(days=1))
        False
    """
    today = date.today()
    
    # Handle different input types
    if isinstance(race_date, str):
        try:
            # Try to parse the date string
            parsed_date_str = parse_date_flexible(race_date)
            race_date_obj = datetime.strptime(parsed_date_str, "%Y-%m-%d").date()
        except ValueError:
            # If parsing fails, assume it's not historical (safer default)
            return False
    elif isinstance(race_date, datetime):
        race_date_obj = race_date.date()
    elif isinstance(race_date, date):
        race_date_obj = race_date
    else:
        # Unknown type, assume not historical
        return False
    
    # Return True if race date is before today
    return race_date_obj < today
