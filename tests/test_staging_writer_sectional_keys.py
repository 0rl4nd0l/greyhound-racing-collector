from ingestion.staging_writer import parse_race_csv_for_staging


def test_parse_race_csv_for_staging_accepts_one_sec_sectional_header(tmp_path):
    race_file = tmp_path / "Race 1 - TEST - 2026-06-01.csv"
    race_file.write_text(
        "Dog Name,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,1 SEC\n"
        "1. Fast Dog,1,1,30.5,350,2026-06-01,TEST,5,19.20,5.43\n",
        encoding="utf-8",
    )

    _, dogs = parse_race_csv_for_staging(str(race_file))

    assert dogs[0]["weight"] == 30.5
    assert dogs[0]["sectional_1st"] == "5.43"
