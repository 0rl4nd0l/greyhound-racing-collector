import json
from pathlib import Path

from scripts import run_expert_form_schema_trial_ablation_report_only as packet


def _schema(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "test_schema",
                "feature_columns": ["field_size", "box_number"],
                "categorical_features": [],
                "numeric_or_boolean_features": ["field_size", "box_number"],
                "feature_families": {},
            }
        ),
        encoding="utf-8",
    )


def _feature_rows(path: Path, *, races: int = 2) -> None:
    rows = []
    for race_number in range(1, races + 1):
        race_id = f"Race {race_number} - TEST - 2026-06-17"
        for box in (1, 2, 3, 4):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": "2026-06-17",
                    "dog_name": f"Dog {race_number}-{box}",
                    "box_number": box,
                    "field_size": 4,
                    "expert_form_metadata_from_sidecar": True,
                    "expert_form_career_starts": race_number + box,
                    "expert_form_career_wins": box == 1,
                    "expert_form_track_distance_starts": box,
                    "expert_form_win_percent": 25.0 if box == 1 else 0.0,
                    "expert_form_grade": "5",
                }
            )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _official_rows(path: Path, *, races: int = 2) -> None:
    lines = []
    for race_number in range(1, races + 1):
        race_id = f"Race {race_number} - TEST - 2026-06-17"
        winner_box = 1 if race_number % 2 else 2
        for box in (1, 2, 3, 4):
            lines.append(
                json.dumps(
                    {
                        "race_id": race_id,
                        "race_date": "2026-06-17",
                        "venue": "TEST",
                        "race_number": race_number,
                        "dog_name": f"Dog {race_number}-{box}",
                        "box_number": box,
                        "finish_position": 1 if box == winner_box else box + 1,
                        "is_winner": box == winner_box,
                        "source": "thedogs_official_result_page",
                        "source_url": f"https://www.thedogs.com.au/racing/test/2026-06-17/{race_number}/race?trial=false",
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_schema_trial_adds_expert_form_features_without_mutating_canonical(tmp_path):
    schema_path = tmp_path / "schema.json"
    _schema(schema_path)
    before = schema_path.read_text(encoding="utf-8")

    schema = json.loads(before)
    trial, diff_rows = packet.build_trial_schema(schema)

    assert schema_path.read_text(encoding="utf-8") == before
    assert trial["canonical_schema_mutation"] is False
    assert len(trial["feature_columns"]) == 2 + len(packet.EXPERT_FORM_FEATURES)
    assert sum(row["trial_action"] == "added_report_only" for row in diff_rows) == len(
        packet.EXPERT_FORM_FEATURES
    )


def test_schema_trial_packet_blocks_when_labels_missing(tmp_path):
    schema_path = tmp_path / "schema.json"
    rows_path = tmp_path / "rows.json"
    _schema(schema_path)
    _feature_rows(rows_path, races=3)

    report = packet.build_report(
        schema_path=schema_path,
        expert_feature_rows_path=rows_path,
        official_result_paths=[],
        min_train_races=1,
        min_holdout_races=1,
    )

    assert report["final_status"] == packet.FINAL_LABELS_MISSING
    assert report["activation_allowed"] is False
    assert report["coverage_summary"]["labeled_races"] == 0
    assert "labeled_temporal_split_below_min" in report["blockers"]


def test_schema_trial_packet_runs_report_only_ablation_when_labeled(tmp_path):
    schema_path = tmp_path / "schema.json"
    rows_path = tmp_path / "rows.json"
    official_path = tmp_path / "official.jsonl"
    _schema(schema_path)
    _feature_rows(rows_path, races=6)
    _official_rows(official_path, races=6)

    report = packet.build_report(
        schema_path=schema_path,
        expert_feature_rows_path=rows_path,
        base_feature_rows_path=rows_path,
        official_result_paths=[official_path],
        min_train_races=2,
        min_holdout_races=1,
    )

    assert report["ablation_status"] == "RUN"
    assert report["control_metrics"]["status"] == "EVALUATED"
    assert report["trial_metrics"]["status"] == "EVALUATED"
    assert report["gate_decision"]["status"] in {"PASS", "FAIL"}
    assert report["thresholds"]["min_slice_races"] == packet.DEFAULT_MIN_SLICE_RACES
    assert report["slice_regression"]
    assert {row["slice_type"] for row in report["slice_regression"]} >= {"venue"}
    assert report["no_write_guarantees"]["canonical_schema_mutation"] is False
