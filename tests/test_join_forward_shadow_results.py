import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import join_forward_shadow_results as joiner


def _prediction(
    race_id: str,
    dog_name: str,
    box: int,
    probability: float,
    rank: int,
) -> dict:
    return {
        "race_id": race_id,
        "dog_name": dog_name,
        "box": box,
        "shadow_rf_uncalibrated_probability": probability,
        "shadow_rf_calibrated_probability": probability,
        "predicted_rank": rank,
        "calibration_method": "power_gamma_2.4",
        "model_version": "test-shadow",
        "model_source": "artifacts/test/model.joblib",
        "tgr_enabled": False,
        "output_mode": "shadow_only",
    }


def _official(
    box: int,
    dog_name: str,
    finish_position: int | None,
    status: str | None = None,
) -> dict:
    return {
        "box_number": box,
        "dog_name": dog_name,
        "finish_position": finish_position,
        "status": status,
    }


def _result_html(rows: list[dict]) -> str:
    body = []
    for row in rows:
        position = row["status"] or (
            f"{row['finish_position']}st"
            if row["finish_position"] == 1
            else f"{row['finish_position']}th"
            if row["finish_position"] is not None
            else ""
        )
        body.append(
            "<tr class=\"race-runner\">"
            f"<td class=\"race-runners__finish-position\">{position}</td>"
            f"<td class=\"race-runners__box\"><input name=\"rug_{row['box_number']}\" /></td>"
            f"<td class=\"race-runners__name\">{row['dog_name']}</td>"
            "</tr>"
        )
    return "<table class=\"race-runners--result\">" + "".join(body) + "</table>"


def _write_shadow_run(root: Path, races: dict[str, list[dict]]) -> Path:
    shadow_dir = root / "shadow"
    shadow_dir.mkdir()
    manifest = {
        "prediction_rows": sum(len(rows) for rows in races.values()),
        "race_count": len(races),
        "calibration_method": "power_gamma_2.4",
        "all_missing_train_policy": "quarantine_feature",
        "tgr_enabled": False,
        "output_mode": "shadow_only",
    }
    (shadow_dir / "shadow_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (shadow_dir / "shadow_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "race_id",
            "dog_name",
            "box",
            "shadow_rf_uncalibrated_probability",
            "shadow_rf_calibrated_probability",
            "predicted_rank",
            "calibration_method",
            "model_version",
            "model_source",
            "tgr_enabled",
            "output_mode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rows in races.values():
            for row in rows:
                writer.writerow(row)

    for index, race_id in enumerate(races, start=1):
        source = shadow_dir / "eligible_inputs" / f"source_{index:04d}"
        source.mkdir(parents=True)
        csv_path = source / f"{race_id}.csv"
        csv_path.write_text("Dog Name|BOX\n", encoding="utf-8")
        race_number = race_id.split(" - ")[0].replace("Race ", "")
        venue = race_id.split(" - ")[1]
        race_date = race_id.split(" - ")[2]
        metadata = {
            "filename": csv_path.name,
            "race_url": f"https://www.thedogs.com.au/racing/test/{race_date}/{race_number}/test?trial=false",
            "race_info": {
                "date": race_date,
                "venue": venue,
                "race_number": race_number,
                "url": f"https://www.thedogs.com.au/racing/test/{race_date}/{race_number}/test?trial=false",
            },
            "metadata_is_leakage_safe": True,
            "runner_completeness_after_canonical_alignment": {
                "status": "COMPLETE",
                "runner_count": len(races[race_id]),
                "boxes": [row["box"] for row in races[race_id]],
                "participants": [
                    {"box_number": row["box"], "dog_name": row["dog_name"]}
                    for row in races[race_id]
                ],
            },
            "canonical_runner_alignment": {
                "schema_version": "canonical_runner_alignment_v1",
                "status": "aligned",
                "reason": None,
                "canonical_runner_set_status": "available",
                "canonical_runner_count": len(races[race_id]),
                "prediction_runner_count": len(races[race_id]),
                "remapped_participants": [],
                "dropped_participants": [],
            },
        }
        csv_path.with_name(csv_path.name + ".metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
    return shadow_dir


def test_normalized_identity_strips_badges_and_punctuation_without_fuzzy_matching():
    assert joiner.normalize_result_identity_name("Shank's Pony NBT") == (
        joiner.normalize_result_identity_name("Shanks Pony")
    )
    assert joiner.normalize_result_identity_name("Alpha Runner") != (
        joiner.normalize_result_identity_name("Alfa Runner")
    )


def test_classify_safe_join_allows_extra_official_scratch_and_unplaced_nonwinners():
    predictions = [
        _prediction("Race 5 - TRA - 2026-06-08", "Minter Blinder", 1, 0.6, 1),
        _prediction("Race 5 - TRA - 2026-06-08", "Shanks Pony", 5, 0.3, 2),
        _prediction("Race 5 - TRA - 2026-06-08", "Little Thief", 8, 0.1, 3),
    ]
    official_rows = [
        _official(1, "Minter Blinder", 1),
        _official(5, "Shank's Pony", 2),
        _official(8, "Little Thief NBT", None),
        _official(2, "King Cherry NBT", None, "SCR"),
    ]

    report = joiner.classify_result_identity_join(
        race_id="Race 5 - TRA - 2026-06-08",
        prediction_rows=predictions,
        official_rows=official_rows,
    )

    assert report["status"] == "SAFE_EXACT_BOX_AND_NAME_MATCH"
    assert report["winner_box"] == 1
    assert report["allowed_extra_scratched_official_boxes"] == [
        {"box": 2, "dog_name": "King Cherry NBT", "status": "SCR"}
    ]


def test_classify_join_rejects_fuzzy_name_mismatch():
    predictions = [_prediction("Race 1 - TEST - 2026-06-08", "Alpha Runner", 1, 1.0, 1)]
    official_rows = [_official(1, "Alfa Runner", 1)]

    report = joiner.classify_result_identity_join(
        race_id="Race 1 - TEST - 2026-06-08",
        prediction_rows=predictions,
        official_rows=official_rows,
    )

    assert report["status"] == "UNSAFE_QUARANTINED"
    assert "dog_name_mismatch_after_exact_badge_stripping" in report["identity_errors"]


def test_join_forward_shadow_results_writes_partial_artifact_with_stub_fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(joiner, "ROOT", tmp_path)
    monkeypatch.setattr(
        joiner,
        "DEFAULT_PROTECTED_PATHS",
        (tmp_path / "greyhound_racing_data.db",),
    )
    races = {
        "Race 1 - TEST - 2026-06-08": [
            _prediction("Race 1 - TEST - 2026-06-08", "Alpha Runner", 1, 0.7, 1),
            _prediction("Race 1 - TEST - 2026-06-08", "Bravo Runner", 2, 0.3, 2),
        ],
        "Race 2 - TEST - 2026-06-08": [
            _prediction("Race 2 - TEST - 2026-06-08", "Charlie Runner", 1, 0.55, 1),
            _prediction("Race 2 - TEST - 2026-06-08", "Delta Runner", 2, 0.45, 2),
        ],
    }
    shadow_dir = _write_shadow_run(tmp_path, races)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_result_join_20260608T120000+1000"
    )

    def fetch_html(url: str) -> dict:
        if "/1/" in url:
            return {
                "url": url,
                "final_url": url,
                "status_code": 200,
                "text": _result_html(
                    [
                        _official(1, "Alpha Runner", 1),
                        _official(2, "Bravo Runner NBT", None),
                    ]
                ),
                "error": None,
            }
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "text": "<html>No results yet</html>",
            "error": None,
        }

    result = joiner.join_forward_shadow_results(
        shadow_run_dir=shadow_dir,
        output_dir=output_dir,
        current_time=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        fetch_html=fetch_html,
        verify_db=False,
    )

    assert result["verdict"] == "PARTIAL_JOIN_PENDING_MORE_RESULTS"
    assert result["safe_joined_race_count"] == 1
    assert result["pending_race_count"] == 1
    assert result["unsafe_match_count"] == 0
    assert (output_dir / "joined_shadow_predictions.jsonl").read_text(encoding="utf-8").count("\n") == 2
    metrics = json.loads((output_dir / "shadow_forward_metrics.json").read_text(encoding="utf-8"))
    assert metrics["top1"] == 1.0
    identity_report = json.loads(
        (output_dir / "identity_match_report.json").read_text(encoding="utf-8")
    )
    assert identity_report["race_attempts"][0]["prejump_runner_alignment"][
        "canonical_runner_alignment_status"
    ] == "aligned"
    assert json.loads((output_dir / "pending_results.json").read_text(encoding="utf-8"))[
        "pending_race_count"
    ] == 1


def test_join_forward_shadow_results_quarantines_malformed_prediction_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(joiner, "ROOT", tmp_path)
    monkeypatch.setattr(
        joiner,
        "DEFAULT_PROTECTED_PATHS",
        (tmp_path / "greyhound_racing_data.db",),
    )
    shadow_dir = _write_shadow_run(
        tmp_path,
        {
            "Race 1 - TEST - 2026-06-08": [
                _prediction("Race 1 - TEST - 2026-06-08", "Alpha Runner", 1, 0.7, 1),
                _prediction("Race 1 - TEST - 2026-06-08", "Bravo Runner", 2, 0.3, 2),
            ]
        },
    )
    with (shadow_dir / "shadow_predictions.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Race 1 - TEST - 2026-06-08",
                "Broken Runner",
                "",
                "0.1",
                "0.1",
                "",
                "power_gamma_2.4",
                "test-shadow",
                "artifacts/test/model.joblib",
                "False",
                "shadow_only",
            ]
        )

    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_result_join_20260608T121500+1000"
    )

    result = joiner.join_forward_shadow_results(
        shadow_run_dir=shadow_dir,
        output_dir=output_dir,
        current_time=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        fetch_html=lambda _url: {
            "url": _url,
            "final_url": _url,
            "status_code": 200,
            "text": _result_html(
                [
                    _official(1, "Alpha Runner", 1),
                    _official(2, "Bravo Runner", 2),
                ]
            ),
            "error": None,
        },
        verify_db=False,
    )

    assert result["verdict"] == "BLOCKED_IDENTITY_MATCH_FAILURE"
    assert result["safe_joined_race_count"] == 0
    assert result["unsafe_match_count"] == 1
    assert result["malformed_prediction_row_count"] == 1
    assert (output_dir / "joined_shadow_predictions.jsonl").read_text(encoding="utf-8") == ""
    malformed = json.loads(
        (output_dir / "malformed_prediction_rows.json").read_text(encoding="utf-8")
    )
    assert malformed["malformed_prediction_row_count"] == 1
    assert malformed["malformed_prediction_rows"][0]["dog_name"] == "Broken Runner"
    identity_report = json.loads(
        (output_dir / "identity_match_report.json").read_text(encoding="utf-8")
    )
    assert identity_report["summary"]["malformed_prediction_row_count"] == 1
    assert identity_report["summary"]["safe_joined_race_count"] == 0
    unsafe = json.loads((output_dir / "unsafe_result_matches.json").read_text(encoding="utf-8"))
    assert unsafe["unsafe_match_count"] == 1
    assert unsafe["unsafe_result_matches"][0]["reason"] == ["malformed_prediction_rows_for_race"]


def test_join_forward_shadow_results_quarantines_all_malformed_race_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(joiner, "ROOT", tmp_path)
    monkeypatch.setattr(
        joiner,
        "DEFAULT_PROTECTED_PATHS",
        (tmp_path / "greyhound_racing_data.db",),
    )
    malformed_race_id = "Race 2 - TEST - 2026-06-08"
    shadow_dir = _write_shadow_run(
        tmp_path,
        {
            "Race 1 - TEST - 2026-06-08": [
                _prediction("Race 1 - TEST - 2026-06-08", "Alpha Runner", 1, 0.7, 1),
                _prediction("Race 1 - TEST - 2026-06-08", "Bravo Runner", 2, 0.3, 2),
            ],
            malformed_race_id: [],
        },
    )
    with (shadow_dir / "shadow_predictions.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                malformed_race_id,
                "Broken Runner",
                "",
                "0.1",
                "0.1",
                "",
                "power_gamma_2.4",
                "test-shadow",
                "artifacts/test/model.joblib",
                "False",
                "shadow_only",
            ]
        )

    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_result_join_20260608T122000+1000"
    )
    fetched_urls: list[str] = []

    def fetch_html(url: str) -> dict:
        fetched_urls.append(url)
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "text": _result_html(
                [
                    _official(1, "Alpha Runner", 1),
                    _official(2, "Bravo Runner", 2),
                ]
            ),
            "error": None,
        }

    result = joiner.join_forward_shadow_results(
        shadow_run_dir=shadow_dir,
        output_dir=output_dir,
        current_time=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
        fetch_html=fetch_html,
        verify_db=False,
    )

    assert result["verdict"] == "PARTIAL_JOIN_PENDING_MORE_RESULTS"
    assert result["safe_joined_race_count"] == 1
    assert result["unsafe_match_count"] == 1
    assert (output_dir / "joined_shadow_predictions.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert not any("/2/" in url for url in fetched_urls)
    unsafe = json.loads((output_dir / "unsafe_result_matches.json").read_text(encoding="utf-8"))
    assert unsafe["unsafe_result_matches"][0]["race_id"] == malformed_race_id
    assert unsafe["unsafe_result_matches"][0]["reason"] == ["malformed_prediction_rows_for_race"]


def test_unique_default_output_dir_adds_suffix_for_existing_generated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(joiner, "ROOT", tmp_path)
    generated_at = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    existing = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_result_join_20260608T120000+0000"
    )
    existing.mkdir(parents=True)

    output_dir = joiner.unique_default_output_dir(
        tmp_path / "artifacts/full_evidence_orchestration_20260525",
        generated_at,
    )

    assert output_dir.name == "forward_shadow_result_join_20260608T120000+0000_001"


def test_explicit_output_dir_still_fails_when_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(joiner, "ROOT", tmp_path)
    monkeypatch.setattr(joiner, "DEFAULT_PROTECTED_PATHS", (tmp_path / "greyhound_racing_data.db",))
    shadow_dir = _write_shadow_run(
        tmp_path,
        {
            "Race 1 - TEST - 2026-06-08": [
                _prediction("Race 1 - TEST - 2026-06-08", "Alpha Runner", 1, 1.0, 1)
            ]
        },
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_result_join_20260608T120000+1000"
    )
    output_dir.mkdir(parents=True)

    try:
        joiner.join_forward_shadow_results(
            shadow_run_dir=shadow_dir,
            output_dir=output_dir,
            fetch_html=lambda _url: {"text": "", "status_code": 200, "error": None},
            verify_db=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected explicit existing output_dir to fail closed")
