import json

from scripts.audit_non_box_feature_quality import (
    csv_delimiter_for_source,
    read_source_csv_for_reconstruction,
)


def test_reconstruction_prefers_verified_pipe_sidecar_delimiter(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    csv_path.write_text(
        "Dog Name|Sex|PLC|BOX|WGT|DIST|DATE|TRACK|G|TIME|WIN|BON|1 SEC|MGN|W/2G|PIR|SP\n"
        "1. Dog One|D|1|1|30.0|400|2026-05-01|TEST|5|22.1|22.1|22.0|6.1|0.1|Other Dog|111|2.0\n",
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "Race 1 - TEST - 2026-06-03.csv.metadata.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "normalized_delimiter": "|",
                "metadata_is_leakage_safe": True,
                "target_distance": "400m",
                "target_distance_source": "canonical_pre_race_page",
            }
        ),
        encoding="utf-8",
    )

    assert csv_delimiter_for_source(csv_path) == "|"
    df = read_source_csv_for_reconstruction(csv_path)

    assert list(df.columns)[:4] == ["Dog Name", "Sex", "PLC", "BOX"]
    assert df.loc[0, "Dog Name"] == "1. Dog One"
    assert int(df.loc[0, "DIST"]) == 400


def test_reconstruction_falls_back_to_first_line_delimiter_count(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    csv_path.write_text(
        "Dog Name|Sex|PLC|BOX|WGT|DIST|DATE|TRACK|G|TIME|WIN|BON|1 SEC|MGN|W/2G|PIR|SP\n"
        "1. Dog, With Comma|D|1|1|30.0|400|2026-05-01|TEST|5|22.1|22.1|22.0|6.1|0.1|Other Dog|111|2.0\n",
        encoding="utf-8",
    )

    assert csv_delimiter_for_source(csv_path) == "|"
    df = read_source_csv_for_reconstruction(csv_path)

    assert df.shape == (1, 17)
    assert df.loc[0, "Dog Name"] == "1. Dog, With Comma"
