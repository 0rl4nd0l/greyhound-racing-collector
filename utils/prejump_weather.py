"""Pre-jump, sidecar-only weather collection for upcoming race artifacts.

This module intentionally does not write to the database. It returns metadata
that can be persisted in the CSV sidecar and later accepted only if the sidecar
capture timestamp is before the race jump time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from utils.http_client import get_shared_session


@dataclass(frozen=True)
class VenueWeatherLocation:
    venue_code: str
    venue_name: str
    latitude: float
    longitude: float
    timezone: str


VENUE_WEATHER_LOCATIONS: dict[str, VenueWeatherLocation] = {
    "AP_K": VenueWeatherLocation("AP_K", "Angle Park", -34.8468, 138.5390, "Australia/Adelaide"),
    "AP-K": VenueWeatherLocation("AP_K", "Angle Park", -34.8468, 138.5390, "Australia/Adelaide"),
    "ANGLE-PARK": VenueWeatherLocation("AP_K", "Angle Park", -34.8468, 138.5390, "Australia/Adelaide"),
    "BAL": VenueWeatherLocation("BAL", "Ballarat", -37.5622, 143.8503, "Australia/Melbourne"),
    "BALLARAT": VenueWeatherLocation("BAL", "Ballarat", -37.5622, 143.8503, "Australia/Melbourne"),
    "BEN": VenueWeatherLocation("BEN", "Bendigo", -36.7581, 144.2789, "Australia/Melbourne"),
    "BENDIGO": VenueWeatherLocation("BEN", "Bendigo", -36.7581, 144.2789, "Australia/Melbourne"),
    "BROKEN-HILL": VenueWeatherLocation("BROKEN-HILL", "Broken Hill", -31.9539, 141.4539, "Australia/Broken_Hill"),
    "BULLI": VenueWeatherLocation("BULLI", "Bulli", -34.3333, 150.9167, "Australia/Sydney"),
    "CANN": VenueWeatherLocation("CANN", "Cannington", -32.0167, 115.9333, "Australia/Perth"),
    "CANNINGTON": VenueWeatherLocation("CANN", "Cannington", -32.0167, 115.9333, "Australia/Perth"),
    "CAPA": VenueWeatherLocation("CAPA", "Capalaba", -27.5333, 153.2000, "Australia/Brisbane"),
    "CAPALABA": VenueWeatherLocation("CAPA", "Capalaba", -27.5333, 153.2000, "Australia/Brisbane"),
    "CASO": VenueWeatherLocation("CASO", "Casino", -28.8667, 153.0500, "Australia/Sydney"),
    "CASINO": VenueWeatherLocation("CASO", "Casino", -28.8667, 153.0500, "Australia/Sydney"),
    "DAPT": VenueWeatherLocation("DAPT", "Dapto", -34.4989, 150.7947, "Australia/Sydney"),
    "DAPTO": VenueWeatherLocation("DAPT", "Dapto", -34.4989, 150.7947, "Australia/Sydney"),
    "DARW": VenueWeatherLocation("DARW", "Darwin", -12.4634, 130.8456, "Australia/Darwin"),
    "DARWIN": VenueWeatherLocation("DARW", "Darwin", -12.4634, 130.8456, "Australia/Darwin"),
    "DUB": VenueWeatherLocation("DUB", "Dubbo", -32.2433, 148.6019, "Australia/Sydney"),
    "DUBBO": VenueWeatherLocation("DUB", "Dubbo", -32.2433, 148.6019, "Australia/Sydney"),
    "GAWL": VenueWeatherLocation("GAWL", "Gawler", -34.6167, 138.7333, "Australia/Adelaide"),
    "GAWLER": VenueWeatherLocation("GAWL", "Gawler", -34.6167, 138.7333, "Australia/Adelaide"),
    "GEE": VenueWeatherLocation("GEE", "Geelong", -38.1499, 144.3617, "Australia/Melbourne"),
    "GEELONG": VenueWeatherLocation("GEE", "Geelong", -38.1499, 144.3617, "Australia/Melbourne"),
    "GOSF": VenueWeatherLocation("GOSF", "Gosford", -33.4142, 151.3411, "Australia/Sydney"),
    "GOSFORD": VenueWeatherLocation("GOSF", "Gosford", -33.4142, 151.3411, "Australia/Sydney"),
    "GRDN": VenueWeatherLocation("GRDN", "The Gardens", -32.8810, 151.7280, "Australia/Sydney"),
    "THE-GARDENS": VenueWeatherLocation("GRDN", "The Gardens", -32.8810, 151.7280, "Australia/Sydney"),
    "GUNN": VenueWeatherLocation("GUNN", "Gunnedah", -30.9833, 150.2500, "Australia/Sydney"),
    "GUNNEDAH": VenueWeatherLocation("GUNN", "Gunnedah", -30.9833, 150.2500, "Australia/Sydney"),
    "HEA": VenueWeatherLocation("HEA", "Healesville", -37.6500, 145.5167, "Australia/Melbourne"),
    "HEALESVILLE": VenueWeatherLocation("HEA", "Healesville", -37.6500, 145.5167, "Australia/Melbourne"),
    "HOBT": VenueWeatherLocation("HOBT", "Hobart", -42.8826, 147.3257, "Australia/Hobart"),
    "HOBART": VenueWeatherLocation("HOBT", "Hobart", -42.8826, 147.3257, "Australia/Hobart"),
    "HOR": VenueWeatherLocation("HOR", "Horsham", -36.7167, 142.2000, "Australia/Melbourne"),
    "HORSHAM": VenueWeatherLocation("HOR", "Horsham", -36.7167, 142.2000, "Australia/Melbourne"),
    "MAND": VenueWeatherLocation("MAND", "Mandurah", -32.5269, 115.7219, "Australia/Perth"),
    "MANDURAH": VenueWeatherLocation("MAND", "Mandurah", -32.5269, 115.7219, "Australia/Perth"),
    "MEA": VenueWeatherLocation("MEA", "The Meadows", -37.6822, 144.9528, "Australia/Melbourne"),
    "THE-MEADOWS": VenueWeatherLocation("MEA", "The Meadows", -37.6822, 144.9528, "Australia/Melbourne"),
    "MOUNT": VenueWeatherLocation("MOUNT", "Mount Gambier", -37.8462, 140.8021, "Australia/Adelaide"),
    "MOUNT-GAMBIER": VenueWeatherLocation("MOUNT", "Mount Gambier", -37.8462, 140.8021, "Australia/Adelaide"),
    "MURR": VenueWeatherLocation("MURR", "Murray Bridge", -35.1167, 139.2667, "Australia/Adelaide"),
    "MURRAY-BRIDGE": VenueWeatherLocation("MURR", "Murray Bridge", -35.1167, 139.2667, "Australia/Adelaide"),
    "MURRAY-BRIDGE-STRAIGHT": VenueWeatherLocation("MURR", "Murray Bridge", -35.1167, 139.2667, "Australia/Adelaide"),
    "NOR": VenueWeatherLocation("NOR", "Northam", -31.6500, 116.6667, "Australia/Perth"),
    "NORTHAM": VenueWeatherLocation("NOR", "Northam", -31.6500, 116.6667, "Australia/Perth"),
    "NOWRA": VenueWeatherLocation("NOWRA", "Nowra", -34.8750, 150.6000, "Australia/Sydney"),
    "RICH": VenueWeatherLocation("RICH", "Richmond", -33.6000, 150.7500, "Australia/Sydney"),
    "RICHMOND": VenueWeatherLocation("RICH", "Richmond", -33.6000, 150.7500, "Australia/Sydney"),
    "ROCK": VenueWeatherLocation("ROCK", "Rockhampton", -23.3833, 150.5167, "Australia/Brisbane"),
    "ROCKHAMPTON": VenueWeatherLocation("ROCK", "Rockhampton", -23.3833, 150.5167, "Australia/Brisbane"),
    "SAL": VenueWeatherLocation("SAL", "Sale", -38.1000, 147.0667, "Australia/Melbourne"),
    "SALE": VenueWeatherLocation("SAL", "Sale", -38.1000, 147.0667, "Australia/Melbourne"),
    "SAN": VenueWeatherLocation("SAN", "Sandown", -37.9564, 145.1603, "Australia/Melbourne"),
    "SANDOWN": VenueWeatherLocation("SAN", "Sandown", -37.9564, 145.1603, "Australia/Melbourne"),
    "SHEP": VenueWeatherLocation("SHEP", "Shepparton", -36.3833, 145.4000, "Australia/Melbourne"),
    "SHEPPARTON": VenueWeatherLocation("SHEP", "Shepparton", -36.3833, 145.4000, "Australia/Melbourne"),
    "TAREE": VenueWeatherLocation("TAREE", "Taree", -31.9100, 152.4600, "Australia/Sydney"),
    "TARE": VenueWeatherLocation("TAREE", "Taree", -31.9100, 152.4600, "Australia/Sydney"),
    "TEMORA": VenueWeatherLocation("TEMORA", "Temora", -34.4526, 147.5453, "Australia/Sydney"),
    "TEMA": VenueWeatherLocation("TEMORA", "Temora", -34.4526, 147.5453, "Australia/Sydney"),
    "TRA": VenueWeatherLocation("TRA", "Traralgon", -38.1833, 146.5333, "Australia/Melbourne"),
    "TRARALGON": VenueWeatherLocation("TRA", "Traralgon", -38.1833, 146.5333, "Australia/Melbourne"),
    "TWN": VenueWeatherLocation("TWN", "Townsville", -19.2589, 146.8169, "Australia/Brisbane"),
    "TOWNSVILLE": VenueWeatherLocation("TWN", "Townsville", -19.2589, 146.8169, "Australia/Brisbane"),
    "WAR": VenueWeatherLocation("WAR", "Warrnambool", -38.3779, 142.4668, "Australia/Melbourne"),
    "WARRNAMBOOL": VenueWeatherLocation("WAR", "Warrnambool", -38.3779, 142.4668, "Australia/Melbourne"),
    "WRGL": VenueWeatherLocation("WRGL", "Warragul", -38.1667, 145.9333, "Australia/Melbourne"),
    "WARRAGUL": VenueWeatherLocation("WRGL", "Warragul", -38.1667, 145.9333, "Australia/Melbourne"),
    "WPK": VenueWeatherLocation("WPK", "Wentworth Park", -33.8721, 151.1949, "Australia/Sydney"),
    "WENTWORTH-PARK": VenueWeatherLocation("WPK", "Wentworth Park", -33.8721, 151.1949, "Australia/Sydney"),
}
DEFAULT_PREJUMP_DISPLAY_TIMEZONE = "Australia/Melbourne"

WEATHER_CODE_LABELS = {
    0: "Clear",
    1: "Partly Cloudy",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Light Rain",
    53: "Light Rain",
    55: "Light Rain",
    56: "Light Rain",
    57: "Light Rain",
    61: "Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Rain",
    67: "Heavy Rain",
    71: "Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow",
    80: "Rain",
    81: "Rain",
    82: "Heavy Rain",
    85: "Snow",
    86: "Heavy Snow",
    95: "Storm",
    96: "Storm",
    99: "Storm",
}


def venue_weather_location(venue: Any) -> VenueWeatherLocation | None:
    text = str(venue or "").strip().upper()
    if not text:
        return None
    if text in VENUE_WEATHER_LOCATIONS:
        return VENUE_WEATHER_LOCATIONS[text]
    variants = {
        text,
        text.replace("_", "-"),
        text.replace(" ", "-"),
        text.replace("_", " "),
        text.replace("-", " "),
    }
    for variant in variants:
        key = "-".join(str(variant).upper().replace("_", "-").split())
        if key in VENUE_WEATHER_LOCATIONS:
            return VENUE_WEATHER_LOCATIONS[key]
    return None


def _prejump_display_timezone(value: Any = None) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_PREJUMP_DISPLAY_TIMEZONE


def _parse_race_datetime(
    race_date: Any,
    race_time: Any,
    timezone_name: str,
    *,
    source_timezone: Any = None,
) -> datetime | None:
    date_text = str(race_date or "").strip()[:10]
    time_text = str(race_time or "").strip().upper().replace(".", ":")
    if not date_text or not time_text:
        return None
    display_timezone = _prejump_display_timezone(source_timezone)
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(time_text, fmt).time()
            parsed_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            display_dt = datetime.combine(
                parsed_date,
                parsed_time,
                tzinfo=ZoneInfo(display_timezone),
            )
            return display_dt.astimezone(ZoneInfo(timezone_name))
        except ValueError:
            continue
    return None


def _nearest_hour_index(times: list[Any], race_dt: datetime) -> int | None:
    best: tuple[float, int] | None = None
    for idx, item in enumerate(times):
        try:
            dt = datetime.fromisoformat(str(item))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=race_dt.tzinfo)
        delta = abs((dt - race_dt).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, idx)
    if best is None:
        return None
    # Hourly forecast should be close to the jump; otherwise fail closed.
    return best[1] if best[0] <= 90 * 60 else None


def _at(values: Mapping[str, Any], key: str, idx: int) -> Any:
    series = values.get(key)
    if not isinstance(series, list) or idx >= len(series):
        return None
    return series[idx]


def collect_open_meteo_weather_metadata(
    race_info: Mapping[str, Any],
    *,
    session: Any = None,
) -> dict[str, Any]:
    """Return source-backed weather metadata for a race, or a rejection payload."""

    venue = race_info.get("venue") or race_info.get("venue_name")
    race_date = race_info.get("date") or race_info.get("race_date")
    race_time = race_info.get("race_time") or race_info.get("jump_time")
    race_time_timezone = (
        race_info.get("race_time_timezone")
        or race_info.get("jump_time_timezone")
        or race_info.get("display_timezone")
    )
    location = venue_weather_location(venue)
    rejected: list[str] = []
    if location is None:
        return {
            "rejected_weather_track_metadata_sources": ["weather_venue_not_mapped"],
        }
    race_dt = _parse_race_datetime(
        race_date,
        race_time,
        location.timezone,
        source_timezone=race_time_timezone,
    )
    if race_dt is None:
        return {
            "rejected_weather_track_metadata_sources": ["weather_race_time_unparseable"],
        }

    params = {
        "latitude": f"{location.latitude:.5f}",
        "longitude": f"{location.longitude:.5f}",
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "pressure_msl",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "visibility",
            ]
        ),
        "timezone": location.timezone,
        "start_date": race_dt.date().isoformat(),
        "end_date": race_dt.date().isoformat(),
    }
    source_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    client = session or get_shared_session()
    response = None
    try:
        response = client.get(source_url, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "weather_track_metadata_source": "open_meteo_forecast_api",
            "weather_track_metadata_source_url": source_url,
            "rejected_weather_track_metadata_sources": [
                f"weather_source_fetch_failed:{type(exc).__name__}"
            ],
        }
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    hourly = payload.get("hourly") if isinstance(payload, Mapping) else None
    if not isinstance(hourly, Mapping) or not isinstance(hourly.get("time"), list):
        return {
            "weather_track_metadata_source": "open_meteo_forecast_api",
            "weather_track_metadata_source_url": source_url,
            "rejected_weather_track_metadata_sources": ["weather_source_missing_hourly"],
        }
    idx = _nearest_hour_index(hourly["time"], race_dt)
    if idx is None:
        return {
            "weather_track_metadata_source": "open_meteo_forecast_api",
            "weather_track_metadata_source_url": source_url,
            "rejected_weather_track_metadata_sources": ["weather_source_no_near_jump_hour"],
        }
    weather_code = _at(hourly, "weather_code", idx)
    if weather_code is None:
        rejected.append("weather_code_missing")
        condition = None
    else:
        try:
            condition = WEATHER_CODE_LABELS.get(int(weather_code), "Cloudy")
        except (TypeError, ValueError):
            condition = None
            rejected.append("weather_code_unparseable")
    if not condition:
        return {
            "weather_track_metadata_source": "open_meteo_forecast_api",
            "weather_track_metadata_source_url": source_url,
            "rejected_weather_track_metadata_sources": rejected or ["weather_condition_missing"],
        }

    detail = {
        "provider": "open-meteo",
        "venue_code": location.venue_code,
        "venue_name": location.venue_name,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": location.timezone,
        "race_time_display_timezone": _prejump_display_timezone(race_time_timezone),
        "race_time_venue_timezone": location.timezone,
        "race_time_venue_local": race_dt.isoformat(),
        "forecast_time": str(hourly["time"][idx]),
        "weather_code": weather_code,
        "temperature_2m": _at(hourly, "temperature_2m", idx),
        "relative_humidity_2m": _at(hourly, "relative_humidity_2m", idx),
        "precipitation": _at(hourly, "precipitation", idx),
        "pressure_msl": _at(hourly, "pressure_msl", idx),
        "wind_speed_10m": _at(hourly, "wind_speed_10m", idx),
        "wind_direction_10m": _at(hourly, "wind_direction_10m", idx),
        "visibility": _at(hourly, "visibility", idx),
    }
    return {
        "weather": condition,
        "weather_condition": condition,
        "weather_track_metadata_source": "open_meteo_forecast_api",
        "weather_track_metadata_source_url": source_url,
        "weather_track_metadata_is_leakage_safe": True,
        "weather_track_metadata_detail": detail,
    }
