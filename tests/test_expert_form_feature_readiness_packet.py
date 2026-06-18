import json
from pathlib import Path

from scripts import build_expert_form_feature_readiness_packet as packet


def _sidecar(path: Path, *, race_number: int = 1, dog_name: str = "Alpha Runner") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "race_info": {
            "date": "2026-06-17",
            "race_time": "4:00 PM",
            "venue": "CAPALABA",
            "race_number": str(race_number),
        },
        "expert_form_metadata": {
            "schema_version": "thedogs_expert_form_metadata_v1",
            "source": "thedogs_expert_form_page",
            "source_url": f"https://www.thedogs.com.au/racing/capalaba/2026-06-17/{race_number}/test/expert-form",
            "captured_at": "2026-06-17T05:00:00Z",
            "metadata_is_leakage_safe": True,
            "runner_count": 1,
            "rejected_reasons": [],
            "runners": [
                {
                    "dog_name": dog_name,
                    "grade": "5",
                    "career": {"starts": 10, "wins": 3, "seconds": 1, "thirds": 2},
                    "track_distance": {
                        "starts": 4,
                        "wins": 2,
                        "seconds": 0,
                        "thirds": 1,
                        "best_time": 19.8,
                    },
                    "win_percent": 30.0,
                    "place_percent": 60.0,
                    "prize_money": 12345.0,
                    "greyhound": {"sex": "D", "sire": "Sire", "dam": "Dam"},
                    "trainer": {"name": "Trainer", "district": "- Test"},
                    "winning_distance_counts": {"<400": 2, "400+": 1},
                    "box_history": {"1": {"starts": 2, "wins": 1, "places": 1}},
                    "best_win_times_other_tracks": [
                        {"track": "QST", "distance": "350m", "time": 18.8}
                    ],
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _feature_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dog_name": "Alpha Runner",
            "expert_form_metadata_from_sidecar": True,
            "expert_form_career_starts": 10,
            "expert_form_career_wins": 3,
            "expert_form_track_distance_starts": 4,
            "expert_form_win_percent": 30.0,
        }
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _schema(path: Path, *, include_expert_feature: bool = False) -> None:
    features = ["field_size", "box_number"]
    if include_expert_feature:
        features.append("expert_form_career_starts")
    path.write_text(
        json.dumps(
            {
                "schema_version": "test_schema",
                "feature_columns": features,
                "categorical_features": [],
                "numeric_or_boolean_features": features,
            }
        ),
        encoding="utf-8",
    )


def test_expert_form_readiness_packet_requires_source_coverage(tmp_path):
    schema = tmp_path / "schema.json"
    _schema(schema)

    report = packet.build_report(
        artifact_roots=[tmp_path / "artifacts"],
        schema_path=schema,
        min_source_races=1,
        min_source_runner_rows=1,
        min_shadow_feature_rows=1,
    )

    assert report["final_status"] == packet.FINAL_SOURCE_LOW
    assert "safe_expert_form_sidecar_races_below_min" in report["blockers"]


def test_expert_form_readiness_packet_blocks_on_schema_gap_after_coverage(tmp_path):
    root = tmp_path / "artifacts"
    _sidecar(root / "Race 1 - CAPALABA - 2026-06-17.csv.metadata.json")
    _feature_rows(root / "shadow_feature_rows.json")
    schema = tmp_path / "schema.json"
    _schema(schema)

    report = packet.build_report(
        artifact_roots=[root],
        schema_path=schema,
        min_source_races=1,
        min_source_runner_rows=1,
        min_shadow_feature_rows=1,
    )

    assert report["final_status"] == packet.FINAL_SCHEMA_TRIAL
    assert report["activation_allowed"] is False
    assert report["coverage_summary"]["safe_source_runner_rows"] == 1
    assert report["coverage_summary"]["safe_shadow_feature_rows"] == 1
    assert report["schema_gap_count"] > 0

