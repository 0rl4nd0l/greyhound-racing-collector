import json
from pathlib import Path

import pytest

import scripts.run_official_race_number_lookup_dry_run as lookup
from scripts.run_official_race_number_lookup_dry_run import build_race_number_lookup_packet


class _FakeResponse:
    def __init__(self, *, status_code=200, text="", url="https://example.test/result"):
        self.status_code = status_code
        self.text = text
        self.url = url


class _FakeHttp:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        for needle, response in self.responses.items():
            if needle in url:
                return response
        return _FakeResponse(status_code=404, text="missing", url=url)


def _result_markup(*, race_number: int, distance: str, grade: str, winner: str) -> str:
    return f"""
    <html>
      <title>thedogs - Angle Park 09 September 2025 Race {race_number}</title>
      <body>
        R{race_number} Fixture {grade} {distance}
        <table class="race-runners race-runners--result">
          <tr class="accordion__anchor race-runner">
            <td class="race-runners__finish-position">1st</td>
            <td class="race-runners__box"><sprite-svg name="rug_4"></sprite-svg></td>
            <td class="race-runners__name"><a>{winner} Nbt T: Trainer Name</a></td>
          </tr>
          <tr class="accordion__anchor race-runner">
            <td class="race-runners__finish-position">2nd</td>
            <td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>
            <td class="race-runners__name"><a>Other Runner</a></td>
          </tr>
        </table>
      </body>
    </html>
    """


def _queue(path: Path) -> Path:
    row = {
        "schema_version": "expanded_historical_official_reverify_candidate_v1",
        "legacy_race_id": "AMBIGUOUS|APK20250909342MTG5W",
        "legacy_runner_rows": 8,
        "lookup_status": "PARSE_BLOCKED",
        "lookup_key": None,
        "partial_lookup_key": {
            "venue": "AP_K",
            "race_date": "2025-09-09",
            "race_number": None,
            "target_distance": 342.0,
            "selected_metadata_grade": "TG5+W",
        },
        "blockers": [
            "identity_resolution_needs_official_race_number_lookup",
            "race_number_missing",
        ],
        "identity_key": "AMBIGUOUS|APK20250909342MTG5W",
        "race_date": "2025-09-09",
        "venue": "AP_K",
        "race_number": None,
        "target_distance": 342.0,
        "selected_metadata_grade": "TG5+W",
        "winner_name": "Charlie'S Harley",
        "winner_key": "dog:CHARLIESHARLEY",
        "label_safety_precheck_reasons": [],
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": False,
            "snapshot_mutation": False,
            "manifest_mutation": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
            "betting_decision": False,
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_race_number_lookup_resolves_single_official_winner_distance_match(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(lookup, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/packet"
    http = _FakeHttp(
        {
            "angle-park/2025-09-09/1/results": _FakeResponse(
                text=_result_markup(
                    race_number=1,
                    distance="342m",
                    grade="TG5+W",
                    winner="Wrong Winner",
                ),
                url="https://www.thedogs.com.au/racing/angle-park/2025-09-09/1/results",
            ),
            "angle-park/2025-09-09/2/results": _FakeResponse(
                text=_result_markup(
                    race_number=2,
                    distance="530m",
                    grade="TG5+W",
                    winner="Charlie'S Harley",
                ),
                url="https://www.thedogs.com.au/racing/angle-park/2025-09-09/2/results",
            ),
            "angle-park/2025-09-09/3/results": _FakeResponse(
                text=_result_markup(
                    race_number=3,
                    distance="342m",
                    grade="TG5+W",
                    winner="Charlie's Harley",
                ),
                url="https://www.thedogs.com.au/racing/angle-park/2025-09-09/3/results",
            ),
        }
    )

    packet = build_race_number_lookup_packet(
        queue_path=_queue(tmp_path / "queue.jsonl"),
        output_dir=output_dir,
        http_client=http,
        max_race_number=3,
    )

    assert packet["schema_version"] == "official_race_number_lookup_dry_run_v1"
    assert packet["status"] == "REPORT_ONLY"
    assert packet["summary"]["resolved_count"] == 1
    assert packet["summary"]["unresolved_count"] == 0
    assert packet["writes_performed"]["label_write"] is False
    resolved = packet["resolved_queue_rows"][0]
    assert resolved["lookup_status"] == "PARSE_READY"
    assert resolved["lookup_key"] == {
        "venue": "AP_K",
        "race_date": "2025-09-09",
        "race_number": 3,
    }
    assert resolved["blockers"] == []
    assert resolved["original_blockers"] == [
        "identity_resolution_needs_official_race_number_lookup",
        "race_number_missing",
    ]
    assert resolved["race_number_resolution"]["label_write_approved"] is False
    assert (output_dir / "official_race_number_lookup_packet.json").exists()
    assert (output_dir / "resolved_official_reverify_queue.jsonl").exists()
    rows = (output_dir / "resolved_official_reverify_queue.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"lookup_status": "PARSE_READY"' in rows


def test_race_number_lookup_keeps_multiple_matches_unresolved(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(lookup, "ROOT", tmp_path)
    http = _FakeHttp(
        {
            "angle-park/2025-09-09/1/results": _FakeResponse(
                text=_result_markup(
                    race_number=1,
                    distance="342m",
                    grade="TG5+W",
                    winner="Charlie'S Harley",
                ),
                url="https://www.thedogs.com.au/racing/angle-park/2025-09-09/1/results",
            ),
            "angle-park/2025-09-09/2/results": _FakeResponse(
                text=_result_markup(
                    race_number=2,
                    distance="342m",
                    grade="TG5+W",
                    winner="Charlie's Harley",
                ),
                url="https://www.thedogs.com.au/racing/angle-park/2025-09-09/2/results",
            ),
        }
    )

    packet = build_race_number_lookup_packet(
        queue_path=_queue(tmp_path / "queue.jsonl"),
        output_dir=tmp_path / "artifacts/full_evidence_orchestration_20260525/packet",
        http_client=http,
        max_race_number=2,
    )

    assert packet["summary"]["resolved_count"] == 0
    assert packet["summary"]["unresolved_count"] == 1
    assert packet["summary"]["resolution_status_counts"] == {
        "MULTIPLE_OFFICIAL_WINNER_DISTANCE_MATCHES_REVIEW_REQUIRED": 1
    }
    assert packet["resolved_queue_rows"] == []


def test_race_number_lookup_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/packet"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        lookup._assert_report_output_dir_safe(outside, root=tmp_path)


def test_race_number_lookup_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        lookup._assert_report_output_dir_safe(tmp_path / "reports/packet", root=tmp_path)
