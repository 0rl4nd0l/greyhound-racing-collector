from datetime import datetime
from zoneinfo import ZoneInfo

from utils.prejump_sportsbet import collect_sportsbet_track_metadata


class FakeSportsbetResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def close(self):
        return None


class FakeSportsbetSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeSportsbetResponse(self.payload)


def _sale_r9_event(**overrides):
    event = {
        "id": 10597250,
        "classId": "4",
        "className": "Greyhound Racing",
        "competitionName": "Sale",
        "raceNumber": 9,
        "startTime": int(
            datetime(
                2026, 6, 17, 13, 57, tzinfo=ZoneInfo("Australia/Melbourne")
            ).timestamp()
        ),
        "type": "greyhound",
        "distance": "435",
        "trackStatus": "Good",
    }
    event.update(overrides)
    return event


def test_sportsbet_track_metadata_accepts_matched_pre_race_event():
    session = FakeSportsbetSession([_sale_r9_event()])

    metadata = collect_sportsbet_track_metadata(
        {
            "date": "2026-06-17",
            "venue": "SAL",
            "race_number": "9",
            "race_time": "1:57 PM",
            "url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/example",
        },
        session=session,
    )

    assert metadata["track_condition"] == "Good"
    assert metadata["weather_track_metadata_source"] == "sportsbet_pre_race_page"
    assert metadata["weather_track_metadata_is_leakage_safe"] is True
    assert metadata["weather_track_metadata_detail"]["event_id"] == 10597250
    assert session.calls


def test_sportsbet_track_metadata_rejects_mismatched_jump_time():
    session = FakeSportsbetSession([_sale_r9_event()])

    metadata = collect_sportsbet_track_metadata(
        {
            "date": "2026-06-17",
            "venue": "SAL",
            "race_number": "9",
            "race_time": "12:57 PM",
            "url": "https://www.thedogs.com.au/racing/sale/2026-06-17/9/example",
        },
        session=session,
    )

    assert "track_condition" not in metadata
    assert metadata["rejected_weather_track_metadata_sources"] == [
        "sportsbet_matching_pre_race_event_not_found"
    ]
