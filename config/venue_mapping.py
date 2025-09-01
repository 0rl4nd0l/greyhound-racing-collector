#!/usr/bin/env python3
"""
Comprehensive Venue Mapping Configuration
==========================================

This module provides a comprehensive mapping of Australian greyhound racing venues
from various formats to standardized venue codes.

The mapping handles:
- Full venue names to abbreviations
- Alternative spellings and formats  
- Legacy codes and variations
- Track-specific identifiers (e.g., K track at Albion Park)
"""

# Comprehensive venue mapping for Australian greyhound racing tracks
VENUE_MAPPING = {
    # New South Wales
    "DAPTO": "DAPT",
    "DAPT": "DAPT",
    "BULLI": "RICH",  # Richmond (Bulli track)
    "RICHMOND": "RICH",
    "RICH": "RICH",
    "RICHMOND-STRAIGHT": "RICH",
    "WENTWORTH PARK": "WPK",
    "WENTWORTH_PARK": "WPK",
    "WPK": "WPK",
    "W_PK": "WPK",  # Alternative format
    "DUBBO": "DUBBO",
    "DUBO": "DUBBO",  # Alternative spelling
    "DUB": "DUBBO",
    "GOULBURN": "GOUL",
    "GOUL": "GOUL",
    "GOSFORD": "GOSFORD",
    "NOWRA": "NOWRA",
    "NOW": "NOWRA",
    "MAITLAND": "MAITLAND",
    "TAREE": "TAREE",
    "TAR": "TAREE",
    "GRAFTON": "GRAF",
    "GRAF": "GRAF",
    "MUSWELLBROOK": "MUSWELLBROOK",
    "BROKEN HILL": "BH",
    "BROKEN-HILL": "BH",
    "BH": "BH",
    # Victoria
    "THE GARDENS": "GRDN",  # The Gardens (Geelong)
    "GARDENS": "GRDN",
    "GRDN": "GRDN",
    "GARD": "GRDN",  # Legacy mapping
    "GEELONG": "GEE",
    "GEE": "GEE",
    "HEALESVILLE": "HEA",
    "HEA": "HEA",
    "BENDIGO": "BEN",
    "BEN": "BEN",
    "BALLARAT": "BAL",
    "BAL": "BAL",
    "SHEPPARTON": "SHEP",
    "SHEP": "SHEP",
    "SHE": "SHEP",  # Alternative abbreviation
    "WARRNAMBOOL": "WAR",
    "WARR": "WAR",
    "WAR": "WAR",
    "WARRAGUL": "WRGL",  # Warragul is different from Warrnambool
    "WRGL": "WRGL",
    "SANDOWN": "SAN",
    "SAN": "SAN",
    "MEADOWS": "MEA",
    "THE_MEADOWS": "MEA",
    "MEA": "MEA",
    "CRANBOURNE": "CRAN",
    "CRAN": "CRAN",
    # Queensland
    "ALBION PARK": "AP",
    "ALBION_PARK": "AP",
    "AP": "AP",
    "AP_K": "AP_K",  # Albion Park K track
    "AP/K": "AP_K",  # Alternative format
    "IPSWICH": "IPS",
    "IPS": "IPS",
    "QOT": "QOT",  # Queensland Oaks Track
    "QST": "QOT",  # Alternative for QOT
    "Q1L": "QOT",  # Q1 Lakeside
    "LADBROKES_Q_STRAIGHT": "QOT",
    "LADBROKES-Q-STRAIGHT": "QOT",
    "LADBROKES-Q1-LAKESIDE": "QOT",
    "LADBROKES-Q2-PARKLANDS": "QOT",
    "CAPALABA": "CAPA",
    "CAPA": "CAPA",
    "CAP": "CAPA",
    "ROCKHAMPTON": "ROCK",
    "ROCK": "ROCK",
    "TOWNSVILLE": "TWN",
    "TWN": "TWN",
    "CASINO": "CASO",
    "CASO": "CASO",
    "CAS": "CASO",
    # South Australia
    "MURRAY BRIDGE": "MURR",
    "MURRAY_BRIDGE": "MURR",
    "MURRAY-BRIDGE-STRAIGHT": "MURR",
    "MURRAY": "MURR",
    "MURR": "MURR",
    "ANGLE PARK": "APWE",
    "APWE": "APWE",
    "GAWLER": "GAWL",
    "GAWL": "GAWL",
    "MOUNT GAMBIER": "MT_G",
    "MOUNT": "MT_G",
    "MT_G": "MT_G",
    # Western Australia
    "CANNINGTON": "CANN",
    "CANN": "CANN",
    "MANDURAH": "MAND",
    "MAND": "MAND",
    "NORTHAM": "NOR",
    "NOR": "NOR",
    # Tasmania
    "HOBART": "HOBT",
    "HOBT": "HOBT",
    "HOB": "HOBT",
    "LAUNCESTON": "LAU",
    "LAUNCESTON": "LCTN",  # Alternative
    "LAU": "LAU",
    "LCTN": "LAU",
    "BRIGHTON": "BRIGH",
    "BRIGH": "BRIGH",
    # ACT/Other
    "TEMORA": "TEM",
    "TEM": "TEM",
    "TEMA": "TEM",  # Alternative
    "WAGGA": "WAG",
    "WAG": "WAG",
    "WAGA": "WAG",  # Alternative
    "SALE": "SAL",
    "SAL": "SAL",
    "DARWIN": "DARW",
    "DARW": "DARW",
    "DAR": "DARW",
    "GUNNEDAH": "GUNN",
    "GUNN": "GUNN",
    "HORSHAM": "HOR",
    "HOR": "HOR",
    "BALLINA": "MBS",  # More Beach Side
    "MBS": "MBS",
    "TRARALGON": "TRA",
    "TRA": "TRA",
    "WODONGA": "WOL",
    "WOL": "WOL",
    # Special/Test venues
    "UNKNOWN": "UNKNOWN",
    "TEST_VEN": "TEST_VEN",
    "RACE": "RACE",  # Generic race venue
    "__R_": "UNKNOWN",  # Corrupted data
}

# Reverse mapping for standardized codes to full names
VENUE_CODE_TO_NAME = {
    "DAPT": "Dapto",
    "RICH": "Richmond",
    "WPK": "Wentworth Park",
    "DUBBO": "Dubbo",
    "GOUL": "Goulburn",
    "GOSFORD": "Gosford",
    "NOWRA": "Nowra",
    "MAITLAND": "Maitland",
    "TAREE": "Taree",
    "GRAF": "Grafton",
    "MUSWELLBROOK": "Muswellbrook",
    "BH": "Broken Hill",
    "GRDN": "The Gardens",
    "GEE": "Geelong",
    "HEA": "Healesville",
    "BEN": "Bendigo",
    "BAL": "Ballarat",
    "SHEP": "Shepparton",
    "WAR": "Warrnambool",
    "WRGL": "Warragul",
    "SAN": "Sandown",
    "MEA": "The Meadows",
    "CRAN": "Cranbourne",
    "AP": "Albion Park",
    "AP_K": "Albion Park (K)",
    "IPS": "Ipswich",
    "QOT": "Queensland Oaks Track",
    "CAPA": "Capalaba",
    "ROCK": "Rockhampton",
    "TWN": "Townsville",
    "CASO": "Casino",
    "MURR": "Murray Bridge",
    "APWE": "Angle Park",
    "GAWL": "Gawler",
    "MT_G": "Mount Gambier",
    "CANN": "Cannington",
    "MAND": "Mandurah",
    "NOR": "Northam",
    "HOBT": "Hobart",
    "LAU": "Launceston",
    "BRIGH": "Brighton",
    "TEM": "Temora",
    "WAG": "Wagga",
    "SAL": "Sale",
    "DARW": "Darwin",
    "GUNN": "Gunnedah",
    "HOR": "Horsham",
    "MBS": "Ballina",
    "TRA": "Traralgon",
    "WOL": "Wodonga",
    "UNKNOWN": "Unknown",
    "TEST_VEN": "Test Venue",
    "RACE": "Generic Race",
}

# States mapping for venues
VENUE_STATE_MAPPING = {
    "DAPT": "NSW",
    "RICH": "NSW",
    "WPK": "NSW",
    "DUBBO": "NSW",
    "GOUL": "NSW",
    "GOSFORD": "NSW",
    "NOWRA": "NSW",
    "MAITLAND": "NSW",
    "TAREE": "NSW",
    "GRAF": "NSW",
    "MUSWELLBROOK": "NSW",
    "BH": "NSW",
    "GRDN": "VIC",
    "GEE": "VIC",
    "HEA": "VIC",
    "BEN": "VIC",
    "BAL": "VIC",
    "SHEP": "VIC",
    "WAR": "VIC",
    "WRGL": "VIC",
    "SAN": "VIC",
    "MEA": "VIC",
    "CRAN": "VIC",
    "AP": "QLD",
    "AP_K": "QLD",
    "IPS": "QLD",
    "QOT": "QLD",
    "CAPA": "QLD",
    "ROCK": "QLD",
    "TWN": "QLD",
    "CASO": "NSW",  # Casino is actually in NSW
    "MURR": "SA",
    "APWE": "SA",
    "GAWL": "SA",
    "MT_G": "SA",
    "CANN": "WA",
    "MAND": "WA",
    "NOR": "WA",
    "HOBT": "TAS",
    "LAU": "TAS",
    "BRIGH": "TAS",
    "TEM": "NSW",
    "WAG": "NSW",
    "SAL": "VIC",
    "DARW": "NT",
    "GUNN": "NSW",
    "HOR": "VIC",
    "MBS": "NSW",
    "TRA": "VIC",
    "WOL": "VIC",
}


def normalize_venue(venue: str) -> str:
    """
    Normalize a venue name/code to the standard format.

    Args:
        venue: Venue name or code in any format

    Returns:
        Standardized venue code, or original if not found
    """
    if not venue:
        return "UNKNOWN"

    venue_upper = venue.strip().upper()
    return VENUE_MAPPING.get(venue_upper, venue_upper)


def get_venue_full_name(venue_code: str) -> str:
    """
    Get the full venue name from a venue code.

    Args:
        venue_code: Standardized venue code

    Returns:
        Full venue name, or the code if not found
    """
    return VENUE_CODE_TO_NAME.get(venue_code, venue_code)


def get_venue_state(venue_code: str) -> str:
    """
    Get the state for a venue code.

    Args:
        venue_code: Standardized venue code

    Returns:
        State abbreviation, or 'UNKNOWN' if not found
    """
    return VENUE_STATE_MAPPING.get(venue_code, "UNKNOWN")


def get_all_venues():
    """Get all supported venues as a list of (code, name, state) tuples."""
    venues = []
    for code, name in VENUE_CODE_TO_NAME.items():
        state = get_venue_state(code)
        venues.append((code, name, state))
    return sorted(venues)


if __name__ == "__main__":
    # Print all venues for reference
    print("Australian Greyhound Racing Venues:")
    print("=" * 50)
    for code, name, state in get_all_venues():
        print(f"{code:8} | {name:20} | {state}")
