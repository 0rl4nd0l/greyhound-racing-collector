import json
from pathlib import Path

from scripts.run_official_reverify_lookup_dry_run import build_lookup_packet, main


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


def _result_markup(position_rows):
    rows = []
    for item in position_rows:
        if len(item) == 3:
            box, position, dog_name = item
        else:
            box, position = item
            dog_name = f"Runner {box}"
        rows.append(
            f"""
            <tr class="accordion__anchor race-runner">
              <td class="race-runners__finish-position">{position}th</td>
              <td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td>
              <td class="race-runners__name"><a>{dog_name}</a></td>
            </tr>
            """
        )
    return '<table class="race-runners race-runners--result">' + "\n".join(rows) + "</table>"


def _queue(path: Path) -> Path:
    rows = [
        {
            "legacy_race_id": "R001_2025-02-18_AP_K",
            "legacy_runner_rows": 2,
            "lookup_status": "PARSE_READY",
            "lookup_key": {"venue": "AP_K", "race_number": 1, "race_date": "2025-02-18"},
        },
        {
            "legacy_race_id": "GEE_5_22_July_2025",
            "legacy_runner_rows": 3,
            "lookup_status": "PARSE_READY",
            "lookup_key": {"venue": "GEE", "race_number": 5, "race_date": "2025-07-22"},
        },
        {
            "legacy_race_id": "UNKNOWN_1_22_July_2025",
            "legacy_runner_rows": 8,
            "lookup_status": "PARSE_READY",
            "lookup_key": {"venue": "UNKNOWN", "race_number": 1, "race_date": "2025-07-22"},
        },
        {
            "legacy_race_id": "NOT_PARSEABLE",
            "legacy_runner_rows": 6,
            "lookup_status": "PARSE_BLOCKED",
            "lookup_key": None,
            "blockers": ["legacy_race_id_not_parseable"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_dry_run_lookup_separates_parse_ready_from_label_write_ready(tmp_path):
    http = _FakeHttp(
        {
            "angle-park/2025-02-18/1": _FakeResponse(
                text=_result_markup([(2, 1), (1, 2)]),
                url="https://www.thedogs.com.au/racing/angle-park/2025-02-18/1/results",
            ),
            "geelong/2025-07-22/5": _FakeResponse(
                text=_result_markup([(8, 1)]),
                url="https://www.thedogs.com.au/racing/geelong/2025-07-22/5/results",
            ),
        }
    )

    packet = build_lookup_packet(
        queue_path=_queue(tmp_path / "queue.jsonl"),
        output_dir=tmp_path / "packet",
        http_client=http,
        max_candidates=None,
    )

    assert packet["status"] == "REPORT_ONLY"
    assert packet["writes_performed"] == {
        "db_write": False,
        "label_write": False,
        "official_fetch": True,
        "snapshot_mutation": False,
        "manifest_mutation": False,
        "model_training": False,
        "registry_mutation": False,
        "promotion": False,
        "betting_decision": False,
    }
    assert packet["summary"]["queue_rows_seen"] == 4
    assert packet["summary"]["official_fetch_attempted_count"] == 2
    assert packet["summary"]["result_parse_ready_count"] == 2
    assert packet["summary"]["label_write_ready_count"] == 1
    assert packet["summary"]["label_write_skip_reason_counts"] == {
        "legacy_lookup_parse_blocked": 1,
        "official_positions_incomplete_for_legacy_runner_count": 1,
        "venue_slug_missing": 1,
    }

    results = {row["legacy_race_id"]: row for row in packet["results"]}
    assert results["R001_2025-02-18_AP_K"]["label_write_ready"] is True
    assert results["R001_2025-02-18_AP_K"]["positions"] == [
        {"box_number": 2, "finish_position": 1, "dog_name": "Runner 2"},
        {"box_number": 1, "finish_position": 2, "dog_name": "Runner 1"},
    ]
    assert results["GEE_5_22_July_2025"]["result_parse_ready"] is True
    assert results["GEE_5_22_July_2025"]["label_write_ready"] is False
    assert results["UNKNOWN_1_22_July_2025"]["lookup_status"] == "VENUE_SLUG_MISSING"
    assert results["NOT_PARSEABLE"]["lookup_status"] == "QUEUE_PARSE_BLOCKED"


def test_dry_run_lookup_cli_writes_report_and_json(tmp_path):
    http_fixture = tmp_path / "fixtures"
    http_fixture.mkdir()
    (http_fixture / "angle-park_2025-02-18_1.html").write_text(
        _result_markup([(2, 1), (1, 2)]),
        encoding="utf-8",
    )
    queue = _queue(tmp_path / "queue.jsonl")
    output_dir = tmp_path / "lookup"

    exit_code = main(
        [
            "--queue",
            str(queue),
            "--output-dir",
            str(output_dir),
            "--max-candidates",
            "1",
            "--fixture-dir",
            str(http_fixture),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "official_reverify_lookup_packet.json").read_text())
    assert payload["summary"]["queue_rows_seen"] == 1
    assert payload["summary"]["label_write_ready_count"] == 1
    assert (output_dir / "report.md").exists()


def test_dry_run_lookup_rejects_protected_repo_output_path(tmp_path):
    queue = _queue(tmp_path / "queue.jsonl")
    protected = Path(__file__).resolve().parents[1] / "artifacts" / "prediction_snapshots"

    try:
        build_lookup_packet(
            queue_path=queue,
            output_dir=protected,
            http_client=_FakeHttp({}),
            max_candidates=1,
        )
    except ValueError as exc:
        assert str(exc) == "protected_output_dir:artifacts/prediction_snapshots"
    else:
        raise AssertionError("expected protected output path rejection")


def test_dry_run_lookup_requires_legacy_runner_count_for_label_write_ready(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "legacy_race_id": "R001_2025-02-18_AP_K",
                "lookup_status": "PARSE_READY",
                "lookup_key": {
                    "venue": "AP_K",
                    "race_number": 1,
                    "race_date": "2025-02-18",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    http = _FakeHttp(
        {
            "angle-park/2025-02-18/1": _FakeResponse(
                text=_result_markup([(2, 1), (1, 2)]),
                url="https://www.thedogs.com.au/racing/angle-park/2025-02-18/1/results",
            ),
        }
    )

    packet = build_lookup_packet(
        queue_path=queue,
        output_dir=tmp_path / "packet",
        http_client=http,
    )

    assert packet["summary"]["result_parse_ready_count"] == 1
    assert packet["summary"]["label_write_ready_count"] == 0
    assert packet["summary"]["label_write_skip_reason_counts"] == {
        "legacy_runner_count_missing": 1,
    }
