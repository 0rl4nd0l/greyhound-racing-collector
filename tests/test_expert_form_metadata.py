import json
from pathlib import Path

from scripts import run_shadow_non_tgr_rf_evaluation as shadow_eval
from utils.csv_metadata import build_csv_download_provenance_payload
from utils.expert_form_metadata import build_expert_form_metadata_payload
from utils.runner_completeness import analyze_csv_text_runner_completeness


EXPERT_FORM_HTML = """
<div class="layout--sidebar--expert">
  <div class="expert-form-runner__main-container">
    <div class="expert-form-runner__details__dog__name">Pop Said Yes<span>(FFA)</span></div>
    <div class="trainer-info"><span>Trainer:</span><span class="trainer-name">Corey mutton</span><span class="trainer-district">- Palm beach</span></div>
    <div class="owner-info"><span>Owner:</span><span>Corey Mutton</span></div>
  </div>
  <div class="expert-form-dog--details">
    <div class="expert-form-cell"><h4>GREYHOUND</h4><div class="expert-form-cell__content">Brindle<br>D<br>00:00 15 November 2023<br>S:<br>Mr. America<br>D:<br>Tin Tin Speckle</div></div>
    <div class="expert-form-cell"><h4>CAREER</h4><div class="expert-form-cell__content">Career:<br>24: 14-4-2<br>TD:<br>15: 9-3-1<br>Win:<br>58% / 83%<br>P/M:<br>$40,686</div></div>
    <div class="expert-form-cell"><h4>TRACK/DIST</h4><div class="expert-form-cell__content">Best Time:<br>19.40<br>15/04/26<br>Best 1st Split:<br>0.00</div></div>
    <div class="expert-form-cell"><h4>BEST WIN TIMES - OTHER TRACKS</h4><div class="expert-form-cell__content_best__wins">QST<br>350m<br>18.74<br>14:22 01 April 2026</div></div>
  </div>
  <div class="box-history"><span>&lt;400</span><span>400+</span><span>500+</span><span>600+</span><span>700+</span><span>12</span><span>1</span><span>1</span><span>0</span><span>0</span></div>
  <div class="box-history"><span>Starts</span><span>5</span><span>2</span><span>2</span><span>1</span><span>6</span><span>3</span><span>2</span><span>3</span><span>Wins</span><span>3</span><span>1</span><span>0</span><span>1</span><span>4</span><span>2</span><span>1</span><span>2</span><span>Places</span><span>1</span><span>0</span><span>2</span><span>0</span><span>1</span><span>1</span><span>1</span><span>0</span></div>
</div>
"""


def _race_info():
    return {
        "date": "2026-06-17",
        "race_time": "2:39 PM",
        "venue": "CAPALABA",
        "race_number": "9",
        "distance": "366m",
        "grade": "Free For All",
    }


def test_expert_form_metadata_accepts_prejump_source_backed_stats():
    payload = build_expert_form_metadata_payload(
        EXPERT_FORM_HTML,
        race_info=_race_info(),
        source_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos/expert-form",
        captured_at="2026-06-17T04:38:00Z",
    )

    assert payload["metadata_is_leakage_safe"] is True
    assert payload["runner_count"] == 1
    runner = payload["runners"][0]
    assert runner["dog_name"] == "Pop Said Yes"
    assert runner["trainer"]["name"] == "Corey mutton"
    assert runner["owner"] == "Corey Mutton"
    assert runner["career"] == {
        "starts": 24,
        "wins": 14,
        "seconds": 4,
        "thirds": 2,
    }
    assert runner["track_distance"]["starts"] == 15
    assert runner["track_distance"]["best_time"] == 19.40
    assert runner["win_percent"] == 58.0
    assert runner["place_percent"] == 83.0
    assert runner["prize_money"] == 40686.0
    assert runner["best_win_times_other_tracks"][0]["track"] == "QST"
    assert runner["winning_distance_counts"]["<400"] == 12
    assert runner["box_history"]["1"]["starts"] == 5
    assert runner["box_history"]["5"]["wins"] == 4


def test_expert_form_metadata_rejects_post_jump_capture_time():
    payload = build_expert_form_metadata_payload(
        EXPERT_FORM_HTML,
        race_info=_race_info(),
        source_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos/expert-form",
        captured_at="2026-06-17T04:40:00Z",
    )

    assert payload["metadata_is_leakage_safe"] is False
    assert "expert_form_metadata_captured_at_not_before_jump" in payload["rejected_reasons"]
    assert payload["runners"] == []


def test_csv_sidecar_preserves_expert_form_metadata(tmp_path: Path):
    race_file = tmp_path / "Race 9 - CAPALABA - 2026-06-17.csv"
    content = (
        "Dog Name|Sex|PLC|BOX|WGT|DIST|DATE|TRACK|G|TIME|WIN|BON|1 SEC|MGN|W/2G|PIR|SP\n"
        "4. Pop Said Yes|D|1|4|38.6|350|2026-06-11|QST|OPEN|18.95|18.74|18.72|4.06|3.0|Flagman Franky|652|3.0\n"
    )
    expert_form_metadata = build_expert_form_metadata_payload(
        EXPERT_FORM_HTML,
        race_info=_race_info(),
        source_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos/expert-form",
        captured_at="2026-06-17T04:38:00Z",
    )

    payload = build_csv_download_provenance_payload(
        filepath=race_file,
        race_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos?trial=false",
        csv_info={"type": "GET", "url": "https://www.thedogs.com.au/export.csv"},
        content=content,
        completeness=analyze_csv_text_runner_completeness(content).as_dict(),
        race_info={
            **_race_info(),
            "metadata_is_leakage_safe": True,
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
            "expert_form_metadata": expert_form_metadata,
        },
        filename=race_file.name,
        allow_generic_fields=True,
    )

    assert payload["expert_form_metadata"]["metadata_is_leakage_safe"] is True
    assert payload["expert_form_metadata"]["runners"][0]["career"]["starts"] == 24
    json.dumps(payload)


def test_live_feature_rows_flatten_safe_expert_form_sidecar_metadata(
    tmp_path: Path,
    monkeypatch,
):
    race_file = tmp_path / "Race 9 - CAPALABA - 2026-06-17.csv"
    content = (
        "Dog Name|BOX|DATE|TRACK|DIST|G|TIME|WGT|PLC\n"
        "4. Pop Said Yes|4|2026-06-11|QST|350|OPEN|18.95|38.6|1\n"
    )
    race_file.write_text(content, encoding="utf-8")
    expert_form_metadata = build_expert_form_metadata_payload(
        EXPERT_FORM_HTML,
        race_info=_race_info(),
        source_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos/expert-form",
        captured_at="2026-06-17T04:38:00Z",
    )
    sidecar = build_csv_download_provenance_payload(
        filepath=race_file,
        race_url="https://www.thedogs.com.au/racing/capalaba/2026-06-17/9/just-greyhound-photos?trial=false",
        csv_info={"type": "GET", "url": "https://www.thedogs.com.au/export.csv"},
        content=content,
        completeness=analyze_csv_text_runner_completeness(content).as_dict(),
        race_info={
            **_race_info(),
            "metadata_is_leakage_safe": True,
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
            "expert_form_metadata": expert_form_metadata,
        },
        filename=race_file.name,
        allow_generic_fields=True,
    )
    race_file.with_name(race_file.name + ".metadata.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    class DummyConnection:
        def close(self):
            return None

    monkeypatch.setattr(shadow_eval, "sqlite_ro", lambda _path: DummyConnection())
    monkeypatch.setattr(shadow_eval, "load_db_history", lambda _connection: {})

    rows = shadow_eval.build_live_feature_rows(
        input_paths=[race_file],
        schema={"feature_columns": ["field_size", "box_number"]},
        db_path=Path("unused.db"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["expert_form_metadata_from_sidecar"] is True
    assert row["expert_form_trainer_name"] == "Corey mutton"
    assert row["expert_form_owner"] == "Corey Mutton"
    assert row["expert_form_career_starts"] == 24
    assert row["expert_form_career_wins"] == 14
    assert row["expert_form_track_distance_starts"] == 15
    assert row["expert_form_win_percent"] == 58.0
    assert row["expert_form_prize_money"] == 40686.0
    assert row["expert_form_track_distance_best_time"] == 19.40
    assert row["expert_form_best_other_track_min_time"] == 18.74
    assert row["expert_form_distance_wins_under_400"] == 12
    assert row["expert_form_current_box_starts"] == 1
    assert row["expert_form_current_box_wins"] == 1
    assert row["expert_form_current_box_places"] == 0
