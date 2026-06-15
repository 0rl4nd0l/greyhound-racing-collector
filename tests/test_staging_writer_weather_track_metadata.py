import json

from ingestion.staging_writer import parse_race_csv_for_staging


def _write_basic_race_csv(path):
    path.write_text(
        "Dog Name,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,1 SEC,track_condition,weather\n"
        "1. Fast Dog,1,1,30.5,350,2026-06-01,TEST,5,19.20,5.43,Heavy,Rain\n",
        encoding="utf-8",
    )


def _write_sidecar(path, **overrides):
    payload = {
        "race_url": "https://www.thedogs.com.au/racing/test/2026-06-01/1/test-race",
        "metadata_is_leakage_safe": True,
        "race_info": {
            "url": "https://www.thedogs.com.au/racing/test/2026-06-01/1/test-race",
            "race_number": 1,
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
        },
    }
    payload.update(overrides)
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_parse_race_csv_for_staging_uses_safe_sidecar_weather_track(tmp_path):
    race_file = tmp_path / "Race 1 - TEST - 2026-06-01.csv"
    _write_basic_race_csv(race_file)
    _write_sidecar(race_file, track_condition="Slow", weather="Showers")

    meta, _dogs = parse_race_csv_for_staging(str(race_file))

    assert meta.track_condition == "Slow"
    assert meta.weather == "Showers"


def test_parse_race_csv_for_staging_rejects_placeholders_and_ignores_csv_weather_track(
    tmp_path,
):
    race_file = tmp_path / "Race 1 - TEST - 2026-06-01.csv"
    _write_basic_race_csv(race_file)
    _write_sidecar(race_file, track_condition="Unknown", weather="N/A")

    meta, _dogs = parse_race_csv_for_staging(str(race_file))

    assert meta.track_condition is None
    assert meta.weather is None


def test_parse_race_csv_for_staging_does_not_use_result_or_odds_weather_track_fields(
    tmp_path,
):
    race_file = tmp_path / "Race 1 - TEST - 2026-06-01.csv"
    _write_basic_race_csv(race_file)
    _write_sidecar(
        race_file,
        winner_track_condition="Fast",
        odds_weather="Fine",
        result_weather="Clear",
    )

    meta, _dogs = parse_race_csv_for_staging(str(race_file))

    assert meta.track_condition is None
    assert meta.weather is None
