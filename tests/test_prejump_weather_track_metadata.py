from utils.csv_metadata import (
    build_safe_weather_track_metadata_payload,
    normalize_track_condition_text,
)
from utils.prejump_weather import (
    collect_open_meteo_weather_metadata,
    venue_weather_location,
)


class FakeWeatherResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def close(self):
        return None


class FakeWeatherSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeWeatherResponse(self.payload)


def _weather_payload():
    return {
        "hourly": {
            "time": ["2026-06-18T18:00", "2026-06-18T19:00", "2026-06-18T20:00"],
            "weather_code": [3, 0, 3],
            "temperature_2m": [10.1, 9.2, 8.8],
            "relative_humidity_2m": [90, 95, 96],
            "precipitation": [0, 0, 0],
            "pressure_msl": [1028.5, 1028.9, 1029.1],
            "wind_speed_10m": [5.2, 6.3, 5.8],
            "wind_direction_10m": [80, 87, 90],
            "visibility": [30000, 35080, 34000],
        }
    }


def test_track_condition_rejects_race_title_promo_text():
    assert normalize_track_condition_text("Good") == "Good"
    assert normalize_track_condition_text("to the Dogs Book Sale") is None

    payload = build_safe_weather_track_metadata_payload(
        {"track_condition": "to the Dogs Book Sale"},
        source_url="https://www.thedogs.com.au/racing/wentworth-park/2026-06-18/9/example",
    )

    assert payload["track_condition"] is None
    assert payload["weather_track_metadata_is_leakage_safe"] is False
    assert "track_condition_missing_or_placeholder" in payload[
        "rejected_weather_track_metadata_sources"
    ]


def test_venue_weather_location_accepts_ap_k_and_nowra():
    assert venue_weather_location("AP_K").venue_name == "Angle Park"
    assert venue_weather_location("NOWRA").venue_name == "Nowra"
    assert venue_weather_location("LADBROKES-Q1-LAKESIDE").venue_name == (
        "Ladbrokes Q1 Lakeside"
    )
    assert venue_weather_location("ladbrokes q2 parklands").venue_code == "Q2"
    assert venue_weather_location("QOT").timezone == "Australia/Brisbane"


def test_weather_metadata_interprets_thedogs_display_time_for_wa():
    session = FakeWeatherSession(_weather_payload())

    metadata = collect_open_meteo_weather_metadata(
        {
            "date": "2026-06-18",
            "venue": "MAND",
            "race_time": "9:12 PM",
        },
        session=session,
    )

    assert metadata["weather"] == "Clear"
    assert metadata["weather_track_metadata_is_leakage_safe"] is True
    detail = metadata["weather_track_metadata_detail"]
    assert detail["race_time_display_timezone"] == "Australia/Melbourne"
    assert detail["race_time_venue_timezone"] == "Australia/Perth"
    assert detail["race_time_venue_local"].startswith("2026-06-18T19:12:00+08:00")
    assert detail["forecast_time"] == "2026-06-18T19:00"
