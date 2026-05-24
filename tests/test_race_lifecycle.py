from datetime import datetime
from zoneinfo import ZoneInfo

from utils.race_lifecycle import (
    JUMPED_PENDING_RESULTS,
    RESULTED,
    STALE_FORM_GUIDE,
    UPCOMING_NOT_JUMPED,
    classify_race_file,
)


NOW = datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne"))


FORM_GUIDE = """Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP
1. KAIA GOT SQUARE,B,1,4,29.1,450,2026-03-23,BAL,Grade 5,25.698,25.698,25.485,6.64,2.0,Kid Luck,1,7.5
"",B,3,2,28.9,425,2026-03-18,BEN,Grade 5,24.417,24.329,24.186,,1.5,BASEMENT,3,12.0
"""


def write_csv(tmp_path, filename, content=FORM_GUIDE):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_past_form_guide_with_historical_plc_time_is_stale_not_resulted(tmp_path):
    path = write_csv(tmp_path, "Race 12 - SAL - 2026-03-29.csv")

    lifecycle = classify_race_file(path, now=NOW)

    assert lifecycle.status == STALE_FORM_GUIDE
    assert lifecycle.race_date == "2026-03-29"
    assert lifecycle.venue == "SAL"
    assert lifecycle.race_number == 12
    assert lifecycle.has_official_result is False
    assert lifecycle.is_live_target is False


def test_future_file_without_result_is_live_upcoming(tmp_path):
    path = write_csv(tmp_path, "Race 1 - MEA - 2026-05-22.csv")

    lifecycle = classify_race_file(path, now=NOW)

    assert lifecycle.status == UPCOMING_NOT_JUMPED
    assert lifecycle.is_live_target is True


def test_today_file_without_jump_time_is_not_live_safe(tmp_path):
    path = write_csv(tmp_path, "Race 1 - AP_K - 2026-05-21.csv")

    lifecycle = classify_race_file(path, now=NOW)

    assert lifecycle.status == JUMPED_PENDING_RESULTS
    assert lifecycle.status_reason == "today_without_jump_time_not_live_safe"
    assert lifecycle.is_live_target is False


def test_today_file_with_future_jump_time_is_live_upcoming(tmp_path):
    path = write_csv(
        tmp_path,
        "Race_1_WPK_2026-05-21.csv",
        "Race Name,Venue,Race Date,Race Time,Race Number,Dog Name,Box\n"
        "Race One,WPK,2026-05-21,18:30,1,Runner One,1\n",
    )

    lifecycle = classify_race_file(path, now=NOW)

    assert lifecycle.status == UPCOMING_NOT_JUMPED
    assert lifecycle.jump_time == "18:30"


def test_explicit_official_result_overrides_date_status(tmp_path):
    path = write_csv(
        tmp_path,
        "Race_1_WPK_2026-05-22.csv",
        "Race Name,Venue,Race Date,Race Time,Race Number,winner_name\n"
        "Race One,WPK,2026-05-22,18:30,1,Runner One\n",
    )

    lifecycle = classify_race_file(path, now=NOW)

    assert lifecycle.status == RESULTED
    assert lifecycle.has_official_result is True
    assert lifecycle.is_live_target is False
