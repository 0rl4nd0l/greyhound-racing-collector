import csv
from pathlib import Path

import pytest

import scripts.run_target_metadata_fetch_parse as target_metadata_fetch_parse
from scripts.run_target_metadata_fetch_parse import (
    APPROVAL_TEXT,
    _assert_report_output_dir_safe,
    build_target_metadata_packet,
    parse_target_metadata_from_text,
)


def _artifact_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "full_evidence_orchestration_20260525" / "target-metadata-test"


class _FakeResponse:
    def __init__(self, *, status_code=200, text="", url=""):
        self.status_code = status_code
        self.text = text
        self.url = url


class _FakeHttpClient:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def _manifest(tmp_path: Path, *, rows: list[dict] | None = None) -> Path:
    path = tmp_path / "manifest.csv"
    fieldnames = [
        "manifest_index",
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "race_name",
        "thedogs_url",
        "current_distance",
        "current_grade",
        "current_race_time",
        "current_start_datetime",
        "winner_name",
        "data_source",
        "results_status",
        "winner_source",
        "dog_rows",
        "finish_rows",
        "box_rows",
        "winner_rows",
        "needs_distance_parse",
        "needs_grade_parse",
        "allowed_current_action",
    ]
    default_rows = [
        {
            "manifest_index": "1",
            "race_id": "GARD_2025-06-27_9",
            "race_date": "2025-06-27",
            "venue": "GARD",
            "race_number": "9",
            "race_name": "",
            "thedogs_url": (
                "https://www.thedogs.com.au/racing/the-gardens/2025-06-27/9/"
                "results?trial=false"
            ),
            "current_distance": "",
            "current_grade": "",
            "current_race_time": "",
            "current_start_datetime": "",
            "winner_name": "Fascinate John",
            "data_source": "official_reverify_lookup_packet",
            "results_status": "resulted",
            "winner_source": "thedogs_official",
            "dog_rows": "8",
            "finish_rows": "8",
            "box_rows": "8",
            "winner_rows": "1",
            "needs_distance_parse": "True",
            "needs_grade_parse": "True",
            "allowed_current_action": "none_without_explicit_fetch_parse_approval",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or default_rows)
    return path


def _html() -> str:
    return """
    <html>
      <head><title>The Gardens Race 9</title></head>
      <body>
        <dl>
          <dt>Distance</dt><dd>515m</dd>
          <dt>Grade</dt><dd>Mixed 4/5</dd>
          <dt>Start Time</dt><dd>7:13 PM</dd>
        </dl>
      </body>
    </html>
    """


def test_parse_target_metadata_from_rendered_text():
    parsed = parse_target_metadata_from_text(
        "Distance\n515m\nGrade\nMixed 4/5\nStart Time\n7:13 PM",
        race_date="2025-06-27",
    )

    assert parsed["parse_status"] == "METADATA_PARSED"
    assert parsed["parsed_distance"] == "515m"
    assert parsed["parsed_grade"] == "Mixed 4/5"
    assert parsed["parsed_race_time"] == "7:13 PM"
    assert parsed["parsed_start_datetime"] == "2025-06-27T19:13:00"


def test_fixture_mode_parses_metadata_without_fetch_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path)
    manifest = _manifest(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "the-gardens_2025-06-27_9.html").write_text(
        _html(),
        encoding="utf-8",
    )

    packet = build_target_metadata_packet(
        manifest_path=manifest,
        output_dir=_artifact_output_dir(tmp_path),
        fixture_dir=fixture_dir,
        expected_races=1,
    )

    result = packet["results"][0]
    assert packet["writes_performed"]["official_fetch"] is False
    assert packet["summary"]["official_fetch_attempted_count"] == 0
    assert packet["summary"]["safe_for_metadata_review_count"] == 1
    assert result["parse_status"] == "METADATA_PARSED"
    assert result["parsed_distance"] == "515m"
    assert result["parsed_grade"] == "Mixed 4/5"
    assert result["parsed_race_time"] == "7:13 PM"
    assert result["parsed_start_datetime"] == "2025-06-27T19:13:00"
    assert result["safe_for_metadata_review"] is True
    assert (_artifact_output_dir(tmp_path) / "target_metadata_fetch_parse_packet.json").exists()
    assert (_artifact_output_dir(tmp_path) / "target_metadata_candidates.csv").exists()
    assert (_artifact_output_dir(tmp_path) / "SUMMARY.md").exists()


def test_real_fetch_mode_requires_exact_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="official_fetch_requires_exact_approval"):
        build_target_metadata_packet(
            manifest_path=_manifest(tmp_path),
            output_dir=_artifact_output_dir(tmp_path),
            http_client=_FakeHttpClient(_FakeResponse(text=_html())),
            expected_races=1,
        )


def test_approved_real_fetch_mode_uses_manifest_url_and_marks_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path)
    manifest = _manifest(tmp_path)
    url = (
        "https://www.thedogs.com.au/racing/the-gardens/2025-06-27/9/"
        "results?trial=false"
    )
    client = _FakeHttpClient(_FakeResponse(text=_html(), url=url))

    packet = build_target_metadata_packet(
        manifest_path=manifest,
        output_dir=_artifact_output_dir(tmp_path),
        approve_fetch_parse=APPROVAL_TEXT,
        http_client=client,
        expected_races=1,
    )

    assert client.requests[0][0] == url
    assert packet["approval"]["approval_text_matched"] is True
    assert packet["writes_performed"]["official_fetch"] is True
    assert packet["writes_performed"]["db_write"] is False
    assert packet["writes_performed"]["label_write"] is False
    assert packet["writes_performed"]["metadata_write"] is False
    assert packet["summary"]["official_fetch_attempted_count"] == 1
    assert packet["results"][0]["safe_for_metadata_review"] is True


def test_expected_race_count_mismatch_fails_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path)
    output_dir = _artifact_output_dir(tmp_path)
    with pytest.raises(ValueError, match="expected_races_mismatch"):
        build_target_metadata_packet(
            manifest_path=_manifest(tmp_path),
            output_dir=output_dir,
            fixture_dir=tmp_path,
            expected_races=2,
        )

    assert not output_dir.exists()


def test_absolute_output_outside_repo_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path / "repo")

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        _assert_report_output_dir_safe(
            tmp_path / "outside" / "artifacts" / "full_evidence_orchestration_20260525"
        )


def test_in_repo_non_artifact_output_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_metadata_fetch_parse, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        _assert_report_output_dir_safe(tmp_path / "reports" / "target-metadata-test")
