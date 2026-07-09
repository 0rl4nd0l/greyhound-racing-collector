import importlib.util
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_ingest_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ingest_results_for_date.py"
    spec = importlib.util.spec_from_file_location(
        "_ingest_results_for_date_for_tests", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_sportsbet_result_text_extracts_top_four_boxes():
    module = _load_ingest_module()
    text = """
    Warragul
    18:15
    R4 Barn Function Area
    6,8,3,4
    """

    parsed = module.parse_sportsbet_result_text(text)

    assert parsed[4]["time"] == "18:15"
    assert parsed[4]["race_name"] == "Barn Function Area"
    assert parsed[4]["boxes"] == [6, 8, 3, 4]


def test_parse_participants_from_csv_uses_only_prefixed_runner_rows(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 4 - WRGL - 2026-05-21.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Dog Name,PLC,TIME",
                "1. Rumour Not Fact,3,22.51",
                ",1,22.31",
                "2. Mr. Fahrenheit,5,22.81",
                "2. Mr. Fahrenheit,6,22.92",
            ]
        ),
        encoding="utf-8",
    )

    participants = module.parse_participants_from_csv(csv_path)

    assert participants == [
        {"box_number": 1, "dog_name": "Rumour Not Fact"},
        {"box_number": 2, "dog_name": "Mr. Fahrenheit"},
    ]


def test_parse_thedogs_result_text_matches_ordinals_to_local_runners():
    module = _load_ingest_module()
    participants = [
        {"box_number": 1, "dog_name": "Rumour Not Fact"},
        {"box_number": 2, "dog_name": "Mr. Fahrenheit"},
        {"box_number": 3, "dog_name": "Tootsie"},
        {"box_number": 4, "dog_name": "Offensive Lady"},
        {"box_number": 5, "dog_name": "Socks And Slides"},
        {"box_number": 6, "dog_name": "Rhine Star"},
        {"box_number": 7, "dog_name": "Chorus Line"},
        {"box_number": 8, "dog_name": "Valve Bounce"},
    ]
    text = """
    Results
    1st
    6. Rhine Star
    2nd
    8. Valve Bounce
    3rd
    3. Tootsie
    """

    positions = module.parse_thedogs_result_text(text, participants)

    assert positions == {6: 1, 8: 2, 3: 3}


def test_parse_thedogs_result_html_preserves_unknown_official_boxes():
    module = _load_ingest_module()
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>
        <td class="race-runners__name"><a>Poppy Florence</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">2nd</td>
        <td class="race-runners__box"><sprite-svg name="rug_6"></sprite-svg></td>
        <td class="race-runners__name"><a>Dalair Milo</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_9"></sprite-svg></td>
        <td class="race-runners__name"><a>Deuces</a></td>
      </tr>
    </table>
    """

    positions = module.parse_thedogs_result_html(markup)

    assert positions == {2: 1, 6: 2}


def test_parse_thedogs_result_html_runner_rows_include_official_names():
    module = _load_ingest_module()
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_7"></sprite-svg></td>
        <td class="race-runners__name"><a>7. She's Driven 19.73 T: Ron Schadow R/T: GM</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">2nd</td>
        <td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>
        <td class="race-runners__name"><a>No Idea</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_9"></sprite-svg></td>
        <td class="race-runners__name"><a>Reserve Runner</a></td>
      </tr>
    </table>
    """

    rows = module.parse_thedogs_result_html_runner_rows(markup)

    assert rows == [
        {
            "box_number": 7,
            "finish_position": 1,
            "dog_name": "She's Driven",
            "status": None,
        },
        {
            "box_number": 2,
            "finish_position": 2,
            "dog_name": "No Idea",
            "status": None,
        },
        {
            "box_number": 9,
            "finish_position": None,
            "dog_name": "Reserve Runner",
            "status": "SCR",
        },
    ]


def test_thedogs_result_html_preserves_non_finish_statuses(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_6"></sprite-svg></td>
        <td class="race-runners__name"><a>Foxtrot Runner</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">FELL</td>
        <td class="race-runners__box"><sprite-svg name="rug_5"></sprite-svg></td>
        <td class="race-runners__name"><a>Echo Runner</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">L/SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_9"></sprite-svg></td>
        <td class="race-runners__name"><a>Reserve Runner</a></td>
      </tr>
    </table>
    """

    result = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
    )._result_from_html(candidate, "https://www.thedogs.test/race", markup)

    assert result is not None
    assert result.positions_by_box == {6: 1}
    assert result.terminal_status_by_box == {5: "FELL", 9: "L/SCR"}


def test_thedogs_result_html_remaps_promoted_reserve_from_box_note(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_6"></sprite-svg></td>
        <td class="race-runners__name"><a>Foxtrot Runner</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">2nd</td>
        <td class="race-runners__box"><sprite-svg name="rug_10"></sprite-svg></td>
        <td class="race-runners__name"><a>Hotel Runner 19.50 (from box 8)</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_8"></sprite-svg></td>
        <td class="race-runners__name"><a>Original Box Eight</a></td>
      </tr>
    </table>
    """

    result = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
    )._result_from_html(candidate, "https://www.thedogs.test/race", markup)

    assert result is not None
    assert result.positions_by_box == {6: 1, 8: 2}
    assert result.terminal_status_by_box == {}
    assert result.reserve_box_remappings == [
        {
            "original_box_number": 10,
            "target_box_number": 8,
            "official_dog_name": "Hotel Runner 19.50 (from box 8)",
            "cleaned_official_dog_name": "Hotel Runner",
            "expected_dog_name": "Hotel Runner",
            "source": "thedogs_result_from_box_note",
        }
    ]
    assert result.ignored_terminal_status_rows == [
        {
            "box_number": 8,
            "status": "SCR",
            "dog_name": "Original Box Eight",
            "reason": "replaced_by_promoted_reserve_from_box_note",
        }
    ]
    assert module.result_validation_error(candidate, result) is None


def test_thedogs_result_html_remaps_promoted_reserve_with_nbt_suffix(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_6"></sprite-svg></td>
        <td class="race-runners__name"><a>Foxtrot Runner</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">2nd</td>
        <td class="race-runners__box"><sprite-svg name="rug_9"></sprite-svg></td>
        <td class="race-runners__name"><a>Delta Runner NBT (from box 4)</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">3rd</td>
        <td class="race-runners__box"><sprite-svg name="rug_10"></sprite-svg></td>
        <td class="race-runners__name"><a>Hotel Runner 19.50 (from box 8)</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_4"></sprite-svg></td>
        <td class="race-runners__name"><a>Original Box Four</a></td>
      </tr>
      <tr class="accordion__anchor race-runner race-runner--scratched">
        <td class="race-runners__finish-position">SCR</td>
        <td class="race-runners__box"><sprite-svg name="rug_8"></sprite-svg></td>
        <td class="race-runners__name"><a>Original Box Eight</a></td>
      </tr>
    </table>
    """

    result = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
    )._result_from_html(candidate, "https://www.thedogs.test/race", markup)

    assert result is not None
    assert result.positions_by_box == {6: 1, 4: 2, 8: 3}
    assert result.terminal_status_by_box == {}
    assert result.reserve_box_remappings == [
        {
            "original_box_number": 9,
            "target_box_number": 4,
            "official_dog_name": "Delta Runner NBT (from box 4)",
            "cleaned_official_dog_name": "Delta Runner",
            "expected_dog_name": "Delta Runner",
            "source": "thedogs_result_from_box_note",
        },
        {
            "original_box_number": 10,
            "target_box_number": 8,
            "official_dog_name": "Hotel Runner 19.50 (from box 8)",
            "cleaned_official_dog_name": "Hotel Runner",
            "expected_dog_name": "Hotel Runner",
            "source": "thedogs_result_from_box_note",
        },
    ]
    assert result.ignored_terminal_status_rows == [
        {
            "box_number": 4,
            "status": "SCR",
            "dog_name": "Original Box Four",
            "reason": "replaced_by_promoted_reserve_from_box_note",
        },
        {
            "box_number": 8,
            "status": "SCR",
            "dog_name": "Original Box Eight",
            "reason": "replaced_by_promoted_reserve_from_box_note",
        },
    ]
    assert module.result_validation_error(candidate, result) is None


def test_thedogs_result_html_rejects_promoted_reserve_name_mismatch(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    markup = """
    <table class="race-runners race-runners--result">
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">1st</td>
        <td class="race-runners__box"><sprite-svg name="rug_6"></sprite-svg></td>
        <td class="race-runners__name"><a>Foxtrot Runner</a></td>
      </tr>
      <tr class="accordion__anchor race-runner">
        <td class="race-runners__finish-position">2nd</td>
        <td class="race-runners__box"><sprite-svg name="rug_10"></sprite-svg></td>
        <td class="race-runners__name"><a>Wrong Runner 19.50 (from box 8)</a></td>
      </tr>
    </table>
    """

    result = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
    )._result_from_html(candidate, "https://www.thedogs.test/race", markup)

    assert result is not None
    assert result.positions_by_box == {6: 1, 10: 2}
    assert result.reserve_box_remappings == []
    assert result.rejected_reserve_box_remappings == [
        {
            "original_box_number": 10,
            "target_box_number": 8,
            "official_dog_name": "Wrong Runner 19.50 (from box 8)",
            "cleaned_official_dog_name": "Wrong Runner",
            "expected_dog_name": "Hotel Runner",
            "reason": "promoted_reserve_name_mismatch",
        }
    ]
    assert (
        module.result_validation_error(candidate, result)
        == "result_boxes_not_in_participants:10"
    )


def test_promoted_reserve_remap_rejects_duplicate_target_box(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    remap = module.remap_promoted_reserve_runner_rows(
        [
            {
                "box_number": 9,
                "dog_name": "Hotel Runner 19.50 (from box 8)",
                "finish_position": 2,
                "status": None,
            },
            {
                "box_number": 10,
                "dog_name": "Hotel Runner 19.51 (from box 8)",
                "finish_position": 3,
                "status": None,
            },
        ],
        candidate.participants,
    )

    assert remap["remappings"] == []
    assert [row["reason"] for row in remap["rejected_remappings"]] == [
        "duplicate_promoted_reserve_target_box",
        "duplicate_promoted_reserve_target_box",
    ]
    assert [row["box_number"] for row in remap["rows"]] == [9, 10]


def test_result_validation_rejects_official_boxes_outside_local_participants(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    candidate.participants = [
        {"box_number": 4, "dog_name": "Cobra Beach"},
        {"box_number": 6, "dog_name": "Dalair Milo"},
        {"box_number": 7, "dog_name": "More Drama"},
        {"box_number": 9, "dog_name": "Deuces"},
        {"box_number": 10, "dog_name": "Spring Orchid"},
    ]
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/gunnedah/2026-05-27/2",
        positions_by_box={2: 1, 6: 2, 8: 3, 4: 5, 7: 7},
        raw_order=[2, 6, 8, 4, 7],
    )

    error = module.result_validation_error(candidate, result)

    assert error == "result_boxes_not_in_participants:2,8"


def test_result_validation_rejects_local_subset_without_first_place(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/gunnedah/2026-05-27/2",
        positions_by_box={6: 2, 4: 5, 7: 7},
        raw_order=[6, 4, 7],
    )

    error = module.result_validation_error(candidate, result)

    assert error == "missing_first_place_result"


def test_result_validation_allows_official_dead_heat_below_first_place(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/hobart/2026-06-11/1",
        positions_by_box={6: 1, 8: 2, 1: 3, 4: 3, 2: 5},
        raw_order=[6, 8, 1, 4, 2],
    )

    assert module.result_validation_error(candidate, result) is None
    assert result.winner_box == 6


def test_result_validation_rejects_non_competition_rank_positions(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/cannington/2026-06-13/7",
        positions_by_box={4: 1, 5: 2, 6: 2, 7: 2, 2: 3, 8: 3, 3: 6},
        raw_order=[4, 5, 6, 7, 2, 8, 3],
    )

    assert module.result_validation_error(candidate, result) == (
        "finish_positions_not_competition_ranked"
    )


def test_result_validation_rejects_duplicate_first_place(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/hobart/2026-06-11/1",
        positions_by_box={6: 1, 8: 1, 1: 3, 4: 4, 2: 5},
        raw_order=[6, 8, 1, 4, 2],
    )

    assert module.result_validation_error(result=result, candidate=candidate) == (
        "duplicate_first_place_results"
    )


def test_source_result_diagnostic_preserves_failed_attempt_context(tmp_path):
    module = _load_ingest_module()
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/gunnedah/2026-05-31/3",
        positions_by_box={6: 2, 4: 5, 7: 7},
        raw_order=[6, 4, 7],
        error="missing_first_place_result",
    )

    diagnostic = module._source_result_diagnostic(result)

    assert diagnostic == {
        "source": "thedogs_official",
        "status": "resulted",
        "source_url": "https://www.thedogs.com.au/racing/gunnedah/2026-05-31/3",
        "error": "missing_first_place_result",
        "raw_order": [6, 4, 7],
        "terminal_statuses": [],
        "reserve_box_remappings": [],
        "ignored_terminal_status_rows": [],
        "rejected_reserve_box_remappings": [],
        "positions": [
            {"box_number": 6, "finish_position": 2},
            {"box_number": 4, "finish_position": 5},
            {"box_number": 7, "finish_position": 7},
        ],
    }


def test_source_result_diagnostic_preserves_official_dog_names_by_box(tmp_path):
    module = _load_ingest_module()
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/taree/2026-06-13/7",
        positions_by_box={2: 1, 9: 2},
        raw_order=[2, 9],
        dog_names_by_box={2: "Riverside Levi", 9: "Reserve Runner"},
    )

    diagnostic = module._source_result_diagnostic(result)

    assert diagnostic["dog_names_by_box"] == {
        "2": "Riverside Levi",
        "9": "Reserve Runner",
    }
    assert diagnostic["positions"] == [
        {
            "box_number": 2,
            "finish_position": 1,
            "dog_name": "Riverside Levi",
        },
        {
            "box_number": 9,
            "finish_position": 2,
            "dog_name": "Reserve Runner",
        },
    ]


def test_dry_run_result_summary_preserves_official_source_and_positions(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4/results",
        positions_by_box={6: 1, 8: 2},
        raw_order=[6, 8],
    )
    db_path, conn = _make_ingest_db(tmp_path)
    try:
        summary = module.write_result(conn, candidate, result, [result], dry_run=True)
    finally:
        conn.close()

    assert db_path.exists()
    assert summary["race_id"] == "Race 4 - WRGL - 2026-05-21"
    assert summary["source"] == "thedogs_official"
    assert summary["source_url"].endswith("/results")
    assert summary["winner_box"] == 6
    assert summary["winner_name"] == "Foxtrot Runner"
    assert summary["positions"] == [
        {"box_number": 6, "finish_position": 1, "dog_name": "Foxtrot Runner"},
        {"box_number": 8, "finish_position": 2, "dog_name": "Hotel Runner"},
    ]
    assert summary["participant_source"] == "csv"


def _make_ingest_db(tmp_path):
    db_path = tmp_path / "results.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_time TEXT,
            start_datetime TEXT,
            sportsbet_url TEXT,
            results_status TEXT DEFAULT 'pending',
            winner_name TEXT,
            winner_odds REAL,
            winner_source TEXT,
            scraping_attempts INTEGER DEFAULT 0,
            last_scraped_at TEXT,
            extraction_timestamp TEXT,
            actual_field_size INTEGER,
            field_size INTEGER,
            url TEXT,
            parse_confidence REAL,
            data_quality_note TEXT,
            data_source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            placing INTEGER,
            scraped_finish_position TEXT,
            extraction_timestamp TEXT,
            data_source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_odds (
            race_id TEXT,
            market_type TEXT,
            box_number INTEGER,
            odds_decimal REAL,
            is_current INTEGER,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return db_path, conn


def _candidate(module, tmp_path):
    csv_path = tmp_path / "Race 4 - WRGL - 2026-05-21.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Dog Name",
                "1. Alpha Runner",
                "2. Bravo Runner",
                "3. Charlie Runner",
                "4. Delta Runner",
                "5. Echo Runner",
                "6. Foxtrot Runner",
                "7. Golf Runner",
                "8. Hotel Runner",
            ]
        ),
        encoding="utf-8",
    )
    participants = module.parse_participants_from_csv(csv_path)
    return module.RaceCandidate(
        race_id="Race 4 - WRGL - 2026-05-21",
        venue="WRGL",
        race_number=4,
        race_date="2026-05-21",
        race_time="16:30",
        start_datetime=None,
        sportsbet_url=(
            "https://www.sportsbet.com.au/betting/greyhound-racing/"
            "australia-nz/warragul/race-4"
        ),
        csv_path=csv_path,
        participants=participants,
        lifecycle_status="jumped_pending_results",
    )


def test_gawl_venue_code_maps_to_gawler_result_slug(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 1 - GAWL - 2026-05-31.csv"
    csv_path.write_text("Dog Name\n1. Alpha Runner\n", encoding="utf-8")
    candidate = module.RaceCandidate(
        race_id="Race 1 - GAWL - 2026-05-31",
        venue="GAWL",
        race_number=1,
        race_date="2026-05-31",
        race_time="10:00",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=[{"box_number": 1, "dog_name": "Alpha Runner"}],
        lifecycle_status="jumped_pending_results",
    )

    assert candidate.thedogs_slug == "gawler"
    assert candidate.sportsbet_slug == "gawler"


def test_darw_venue_code_maps_to_darwin_result_slug(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 1 - DARW - 2026-05-31.csv"
    csv_path.write_text("Dog Name\n1. Alpha Runner\n", encoding="utf-8")
    candidate = module.RaceCandidate(
        race_id="Race 1 - DARW - 2026-05-31",
        venue="DARW",
        race_number=1,
        race_date="2026-05-31",
        race_time="10:00",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=[{"box_number": 1, "dog_name": "Alpha Runner"}],
        lifecycle_status="jumped_pending_results",
    )

    assert candidate.thedogs_slug == "darwin"
    assert candidate.sportsbet_slug == "darwin"


def test_canonical_thedogs_url_overrides_unknown_snapshot_venue_code(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 11 - TRA - 2026-06-01.csv"
    csv_path.write_text("Dog Name\n1. Alpha Runner\n", encoding="utf-8")
    candidate = module.RaceCandidate(
        race_id="Race 11 - TRA - 2026-06-01",
        venue="TRA",
        race_number=11,
        race_date="2026-06-01",
        race_time="17:48",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=[{"box_number": 1, "dog_name": "Alpha Runner"}],
        lifecycle_status="jumped_pending_results",
        canonical_thedogs_url=(
            "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
            "sportsbet-supporters-of-having-a-crack?trial=false"
        ),
    )

    assert candidate.thedogs_slug == "traralgon"
    assert candidate.sportsbet_slug == "traralgon"


def test_thedogs_fetch_tries_canonical_race_result_url_first(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 11 - TRA - 2026-06-01.csv"
    csv_path.write_text("Dog Name\n1. Alpha Runner\n", encoding="utf-8")
    candidate = module.RaceCandidate(
        race_id="Race 11 - TRA - 2026-06-01",
        venue="TRA",
        race_number=11,
        race_date="2026-06-01",
        race_time="17:48",
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=[{"box_number": 1, "dog_name": "Alpha Runner"}],
        lifecycle_status="jumped_pending_results",
        canonical_thedogs_url=(
            "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
            "sportsbet-supporters-of-having-a-crack?trial=false"
        ),
    )

    class Session:
        def get(self, *_args, **_kwargs):
            raise AssertionError("URL list construction should not need HTTP")

    fetcher = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
        http_session=Session(),
    )

    urls = fetcher._result_urls(candidate)

    assert urls[:4] == [
        "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
        "sportsbet-supporters-of-having-a-crack/results?trial=false",
        "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
        "sportsbet-supporters-of-having-a-crack/results",
        "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
        "sportsbet-supporters-of-having-a-crack?trial=false",
        "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/"
        "sportsbet-supporters-of-having-a-crack",
    ]
    assert "https://www.thedogs.com.au/racing/traralgon/2026-06-01/11/results?trial=false" in urls


def _four_runner_csv() -> str:
    return "\n".join(
        [
            "Dog Name",
            "1. Alpha Runner",
            "2. Bravo Runner",
            "3. Charlie Runner",
            "4. Delta Runner",
        ]
    )


FOUR_PARTICIPANTS = [
    {"box_number": 1, "dog_name": "Alpha Runner"},
    {"box_number": 2, "dog_name": "Bravo Runner"},
    {"box_number": 3, "dog_name": "Charlie Runner"},
    {"box_number": 4, "dog_name": "Delta Runner"},
]


def test_thedogs_fetch_success_uses_resulted_status(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Body:
        text = "Results\n1st\n6. Foxtrot Runner\n2nd\n8. Hotel Runner\n"

    class Driver:
        title = "Race results"

        def get(self, url):
            self.current_url = url

        def find_element(self, *_args):
            return Body()

    result = module.TheDogsResultFetcher(Driver(), wait_seconds=0).fetch(candidate)

    assert result.source == "thedogs_official"
    assert result.status == "resulted"
    assert result.positions_by_box == {6: 1, 8: 2}


def test_thedogs_fetch_discovers_public_result_route_before_selenium(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Response:
        def __init__(self, url, text, status_code=200):
            self.url = url
            self.text = text
            self.status_code = status_code

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **_kwargs):
            self.urls.append(url)
            if url.endswith("/racing/warragul/2026-05-21?trial=false"):
                return Response(
                    url,
                    """
                    <a href="/racing/warragul/2026-05-21/4/barn-function-area?trial=false">R4</a>
                    """,
                )
            return Response(
                url,
                """
                <html><title>thedogs - Warragul 21 May 2026 Race 4</title>
                <body>
                <table>
                <tr><td>1st</td><td>Foxtrot Runner</td></tr>
                <tr><td>2nd</td><td>Hotel Runner</td></tr>
                </table>
                </body></html>
                """,
            )

    class Driver:
        def get(self, _url):
            raise AssertionError("Selenium should not be used when public HTML succeeds")

    session = Session()

    result = module.TheDogsResultFetcher(
        Driver(),
        wait_seconds=0,
        http_session=session,
    ).fetch(candidate)

    assert session.urls[0] == "https://www.thedogs.com.au/racing/warragul/2026-05-21?trial=false"
    assert session.urls[1] == (
        "https://www.thedogs.com.au/racing/warragul/2026-05-21/4/"
        "barn-function-area?trial=false"
    )
    assert result.source == "thedogs_official"
    assert result.status == "resulted"
    assert result.source_url.endswith("/barn-function-area?trial=false")
    assert result.positions_by_box == {6: 1, 8: 2}


def test_thedogs_fetch_tries_public_result_url_after_forbidden_meeting_discovery(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Response:
        def __init__(self, url, text, status_code=200):
            self.url = url
            self.text = text
            self.status_code = status_code

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **_kwargs):
            self.urls.append(url)
            if url.endswith("/racing/warragul/2026-05-21?trial=false"):
                return Response(url, "403 Forbidden", status_code=403)
            if url.endswith("/racing/warragul/2026-05-21/4/results?trial=false"):
                return Response(
                    url,
                    """
                    <html><body>
                    Results
                    1st
                    6. Foxtrot Runner
                    2nd
                    8. Hotel Runner
                    </body></html>
                    """,
                )
            return Response(url, "", status_code=404)

    class Driver:
        def get(self, _url):
            raise AssertionError("Selenium should not be used when public HTML succeeds")

    session = Session()

    result = module.TheDogsResultFetcher(
        Driver(),
        wait_seconds=0,
        http_session=session,
    ).fetch(candidate)

    assert session.urls[:2] == [
        "https://www.thedogs.com.au/racing/warragul/2026-05-21?trial=false",
        "https://www.thedogs.com.au/racing/warragul/2026-05-21/4/results?trial=false",
    ]
    assert result.source == "thedogs_official"
    assert result.status == "resulted"
    assert result.positions_by_box == {6: 1, 8: 2}


def test_thedogs_http_403_is_reported_without_selenium_fallback(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Response:
        url = "https://www.thedogs.com.au/racing/warragul/2026-05-21?trial=false"
        text = "403 Forbidden"
        status_code = 403

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    class Driver:
        def get(self, _url):
            raise AssertionError("Blocked official access should remain auditable")

    result = module.TheDogsResultFetcher(
        Driver(),
        wait_seconds=0,
        http_session=Session(),
    ).fetch(candidate)

    assert result.source == "thedogs_official"
    assert result.status == "error"
    assert result.error == "thedogs_403_forbidden"
    assert result.positions_by_box == {}


def test_thedogs_result_table_without_positions_does_not_parse_form_history(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    markup = """
    <html><body>
      <table class="race-runners--result">
        <tr class="race-runner">
          <td class="race-runners__finish-position">OFF</td>
          <td class="race-runners__box"></td>
          <td class="race-runners__name">Foxtrot Runner</td>
        </tr>
      </table>
      <section>
        Results
        1st
        6. Foxtrot Runner
        2nd
        8. Hotel Runner
      </section>
    </body></html>
    """

    result = module.TheDogsResultFetcher(
        driver=None,
        wait_seconds=0,
    )._result_from_html(candidate, "https://www.thedogs.test/race", markup)

    assert result is not None
    assert result.source == "thedogs_official"
    assert result.status == "error"
    assert result.error == "thedogs_result_table_without_strict_positions"
    assert result.positions_by_box == {}


def test_thedogs_http_404_is_reported_without_selenium_fallback(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Response:
        def __init__(self, url, text="", status_code=200):
            self.url = url
            self.text = text
            self.status_code = status_code

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **_kwargs):
            self.urls.append(url)
            if url.endswith("/racing/warragul/2026-05-21?trial=false"):
                return Response(url, "<html><body>No race links yet</body></html>")
            return Response(url, "Not Found", status_code=404)

    class Driver:
        def get(self, _url):
            raise AssertionError("HTTP 404 should remain the auditable official error")

    session = Session()

    result = module.TheDogsResultFetcher(
        Driver(),
        wait_seconds=0,
        http_session=session,
    ).fetch(candidate)

    assert "https://www.thedogs.com.au/racing/warragul/2026-05-21/4/results?trial=false" in session.urls
    assert result.source == "thedogs_official"
    assert result.status == "error"
    assert result.error == "thedogs_http_404"
    assert result.positions_by_box == {}
    assert result.attempted_urls
    assert {
        "url": "https://www.thedogs.com.au/racing/warragul/2026-05-21/4/results?trial=false",
        "final_url": "https://www.thedogs.com.au/racing/warragul/2026-05-21/4/results?trial=false",
        "status_code": 404,
        "error": "thedogs_http_404",
    } in result.attempted_urls
    diagnostic = module._source_result_diagnostic(result)
    assert diagnostic["attempted_urls"] == result.attempted_urls


def test_thedogs_public_http_client_is_stateless(monkeypatch):
    module = _load_ingest_module()
    observed = {}

    class Cookies:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    class Response:
        status_code = 200
        text = "ok"
        url = "https://example.test/result"

    class Session:
        def __init__(self):
            self.trust_env = True
            self.cookies = Cookies()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            observed["url"] = url
            observed["trust_env"] = self.trust_env
            observed["cookies_cleared"] = self.cookies.cleared
            observed["request_cookies"] = kwargs.get("cookies")
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)

    response = module._StatelessPublicHttpClient().get(
        "https://www.thedogs.com.au/racing/warragul/2026-05-21?trial=false",
        cookies={"session": "should-not-be-sent"},
    )

    assert response.status_code == 200
    assert observed == {
        "url": "https://www.thedogs.com.au/racing/warragul/2026-05-21?trial=false",
        "trust_env": False,
        "cookies_cleared": True,
        "request_cookies": {},
    }


def test_sportsbet_fetcher_marks_top_four_as_partial(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Body:
        text = "Warragul\n18:15\nR4 Barn Function Area\n6,8,3,4\n"

    class Driver:
        current_url = "https://www.sportsbet.com.au/results/2026-05-21/racing/greyhound-racing-4/warragul-123"

        def get(self, url):
            self.current_url = url

        def find_element(self, *_args):
            return Body()

    fetcher = module.SportsbetResultFetcher(Driver(), "2026-05-21", wait_seconds=0)
    fetcher.category_links = {"warragul": Driver.current_url}

    result = fetcher.fetch(candidate)

    assert result.source == "sportsbet_results_top4"
    assert result.status == "partial_sportsbet_results"
    assert result.positions_by_box == {6: 1, 8: 2, 3: 3, 4: 4}


def test_result_validation_rejects_boxes_outside_frozen_participants(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    result = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://example.test/results",
        positions_by_box={6: 1, 8: 2, 9: 3},
        raw_order=[6, 8, 9],
    )

    error = module.result_validation_error(candidate, result)

    assert error == "result_boxes_not_in_participants:9"


def test_result_validation_uses_frozen_participant_reason_for_snapshot_candidates(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)
    candidate.participant_source = "snapshot"
    result = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://example.test/results",
        positions_by_box={6: 1, 8: 2, 9: 3},
        raw_order=[6, 8, 9],
    )

    error = module.result_validation_error(candidate, result)

    assert error == "result_boxes_not_in_frozen_participants:9"


def test_write_sportsbet_fallback_records_partial_status_and_thedogs_error(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    conn.execute(
        """
        INSERT INTO live_odds
            (race_id, market_type, box_number, odds_decimal, is_current, timestamp)
        VALUES (?, 'win', 6, 3.2, 1, '2026-05-21T16:29:00')
        """,
        (candidate.race_id,),
    )
    conn.commit()

    official_error = module.SourceResult(
        source="thedogs_official",
        status="error",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4",
        positions_by_box={},
        raw_order=[],
        error="thedogs_403_forbidden",
    )
    fallback = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://www.sportsbet.com.au/results/2026-05-21/racing/greyhound-racing-4/warragul-123",
        positions_by_box={6: 1, 8: 2, 3: 3, 4: 4},
        raw_order=[6, 8, 3, 4],
    )

    module.write_result(conn, candidate, fallback, [official_error, fallback], dry_run=False)
    conn.commit()

    row = conn.execute(
        """
        SELECT winner_name, winner_odds, winner_source, results_status, data_quality_note
        FROM race_metadata
        WHERE race_id = ?
        """,
        (candidate.race_id,),
    ).fetchone()
    dog_rows = conn.execute(
        "SELECT box_number, finish_position, data_source FROM dog_race_data ORDER BY box_number"
    ).fetchall()
    conn.close()

    assert db_path.exists()
    assert row[0] == "Foxtrot Runner"
    assert row[1] == 3.2
    assert row[2] == "sportsbet_results_top4"
    assert row[3] == "partial_sportsbet_results"
    assert "thedogs_official:thedogs_403_forbidden" in row[4]
    assert dog_rows[5] == (6, 1, "sportsbet_results_top4")
    assert dog_rows[0] == (1, None, "sportsbet_results_top4")


def test_write_official_result_uses_resulted_lifecycle_status(tmp_path):
    module = _load_ingest_module()
    _db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'jumped_pending_results')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    official = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4",
        positions_by_box={6: 1, 8: 2, 3: 3, 4: 4, 1: 5, 2: 6, 5: 7, 7: 8},
        raw_order=[6, 8, 3, 4, 1, 2, 5, 7],
    )

    module.write_result(conn, candidate, official, [official], dry_run=False)
    conn.commit()

    row = conn.execute(
        "SELECT winner_source, results_status, parse_confidence FROM race_metadata WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()
    conn.close()

    assert row == ("thedogs_official", "resulted", 1.0)


def test_write_result_seeds_missing_metadata_for_snapshot_candidate(tmp_path):
    module = _load_ingest_module()
    _db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    candidate.participant_source = "snapshot"
    official = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4",
        positions_by_box={6: 1, 8: 2, 3: 3, 4: 4, 1: 5, 2: 6, 5: 7, 7: 8},
        raw_order=[6, 8, 3, 4, 1, 2, 5, 7],
    )

    summary = module.write_result(conn, candidate, official, [official], dry_run=False)
    conn.commit()

    meta = conn.execute(
        "SELECT race_id, results_status, data_source FROM race_metadata WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()
    labels = conn.execute(
        "SELECT COUNT(*) FROM dog_race_data WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()[0]
    conn.close()

    assert summary["metadata_seeded"] is True
    assert meta == (candidate.race_id, "resulted", "frozen_snapshot")
    assert labels == 8


def test_dry_run_write_result_does_not_mutate_database(tmp_path):
    module = _load_ingest_module()
    _db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    conn.commit()
    result = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://example.test/results",
        positions_by_box={6: 1},
        raw_order=[6],
    )

    summary = module.write_result(conn, candidate, result, [result], dry_run=True)
    conn.commit()

    status = conn.execute(
        "SELECT results_status FROM race_metadata WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()[0]
    dog_count = conn.execute("SELECT COUNT(*) FROM dog_race_data").fetchone()[0]
    conn.close()

    assert summary["dry_run"] is True
    assert status == "pending"
    assert dog_count == 0


def test_result_ingest_main_requires_write_approval_for_label_writes(
    tmp_path, monkeypatch, capsys
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "result label writes require" in captured.err
    assert '"status": "not_approved"' in captured.out


def test_result_ingest_main_writes_structured_dry_run_report_without_candidates(
    tmp_path,
    capsys,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    report_path = tmp_path / "dry_run_report.json"

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--dry-run",
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Candidates: 0" in captured.out
    assert report["schema_version"] == "official_result_ingest_report_v1"
    assert report["status"] == "DATA_MISSING"
    assert report["dry_run"] is True
    assert report["candidate_count"] == 0
    assert report["clean_for_label_write"] is False


def test_result_ingest_main_dry_run_can_use_official_http_without_selenium(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    race_id = "Race 4 - WRGL - 2026-05-21"
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', NULL, 'pending')
        """,
        (race_id,),
    )
    conn.commit()
    conn.close()

    class Response:
        def __init__(self, url, text, status_code=200):
            self.url = url
            self.text = text
            self.status_code = status_code

    class Session:
        def get(self, url, **_kwargs):
            if url.endswith("/racing/warragul/2026-05-21?trial=false"):
                return Response(
                    url,
                    """
                    <a href="/racing/warragul/2026-05-21/4/barn-function-area?trial=false">R4</a>
                    """,
                )
            return Response(
                url,
                """
                <table class="race-runners race-runners--result">
                  <tr class="accordion__anchor race-runner">
                    <td class="race-runners__finish-position">1st</td>
                    <td class="race-runners__box"><sprite-svg name="rug_1"></sprite-svg></td>
                    <td class="race-runners__name"><a>Alpha Runner</a></td>
                  </tr>
                  <tr class="accordion__anchor race-runner">
                    <td class="race-runners__finish-position">2nd</td>
                    <td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>
                    <td class="race-runners__name"><a>Bravo Runner</a></td>
                  </tr>
                  <tr class="accordion__anchor race-runner">
                    <td class="race-runners__finish-position">3rd</td>
                    <td class="race-runners__box"><sprite-svg name="rug_3"></sprite-svg></td>
                    <td class="race-runners__name"><a>Charlie Runner</a></td>
                  </tr>
                  <tr class="accordion__anchor race-runner">
                    <td class="race-runners__finish-position">4th</td>
                    <td class="race-runners__box"><sprite-svg name="rug_4"></sprite-svg></td>
                    <td class="race-runners__name"><a>Delta Runner</a></td>
                  </tr>
                </table>
                """,
            )

    monkeypatch.setattr(module, "_StatelessPublicHttpClient", Session)
    report_path = tmp_path / "dry_run_report.json"

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--dry-run",
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Candidates: 1" in captured.out
    assert report["status"] == "SUCCESS"
    assert report["dry_run"] is True
    assert report["clean_for_label_write"] is True
    assert len(report["ingested"]) == 1
    ingested = report["ingested"][0]
    assert ingested["box_order"] == [1, 2, 3, 4]
    assert ingested["dry_run"] is True
    assert ingested["race_id"] == race_id
    assert ingested["source"] == "thedogs_official"
    assert ingested["status"] == "resulted"
    assert ingested["winner_name"] == "Alpha Runner"
    assert ingested["winner_box"] == 1
    assert ingested["source_url"].endswith(
        "/racing/warragul/2026-05-21/4/barn-function-area?trial=false"
    )
    assert ingested["positions"] == [
        {"box_number": 1, "finish_position": 1, "dog_name": "Alpha Runner"},
        {"box_number": 2, "finish_position": 2, "dog_name": "Bravo Runner"},
        {"box_number": 3, "finish_position": 3, "dog_name": "Charlie Runner"},
        {"box_number": 4, "finish_position": 4, "dog_name": "Delta Runner"},
    ]


def test_result_ingest_main_requires_clean_dry_run_report_before_label_write(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', ?, 'pending')
        """,
        (
            "Race 4 - WRGL - 2026-05-21",
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()
    report_path = tmp_path / "blocked_write_report.json"
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--write-labels-approved",
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert "clean prior --dry-run report" in captured.err
    assert report["candidate_count"] == 1
    assert report["dry_run_report_gate"]["reason"] == "missing_approved_dry_run_report"
    assert report["backup_path"] is None


def test_clean_dry_run_report_gate_accepts_matching_clean_report(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    candidate = _candidate(module, tmp_path)
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--dry-run",
        ]
    )
    write_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--write-labels-approved",
        ]
    )
    report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(dry_args),
        candidates=[candidate],
        skipped=[],
        ingested=[
            {
                "race_id": candidate.race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ],
        failed=[],
        backup_path=None,
    )
    report_path = tmp_path / "clean_dry_run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = module.validate_clean_dry_run_report(
        report_path=str(report_path),
        args=write_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        candidate_race_ids=[candidate.race_id],
    )

    assert gate["approved"] is True
    assert gate["status"] == "approved"


def test_result_ingest_main_validates_label_write_readiness_without_approval(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    race_id = "Race 4 - WRGL - 2026-05-21"
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', ?, 'pending')
        """,
        (
            race_id,
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()
    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [race_id],
    )
    assert skipped == []
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--race-id",
            race_id,
            "--dry-run",
        ]
    )
    clean_report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=Path(dry_args.snapshot_dir),
        write_approval=module.result_label_write_approved(dry_args),
        candidates=candidates,
        skipped=[],
        ingested=[
            {
                "race_id": race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4],
            }
        ],
        failed=[],
        backup_path=None,
    )
    clean_report_path = tmp_path / "clean_dry_run_report.json"
    clean_report_path.write_text(json.dumps(clean_report), encoding="utf-8")
    readiness_path = tmp_path / "readiness.json"
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--race-id",
            race_id,
            "--approved-dry-run-report",
            str(clean_report_path),
            "--validate-label-write-readiness",
            "--output",
            str(readiness_path),
        ]
    )

    captured = capsys.readouterr()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Candidates: 1" in captured.out
    assert readiness["schema_version"] == "result_label_write_readiness_validation_v1"
    assert readiness["status"] == "READY_FOR_EXPLICIT_APPROVAL"
    assert readiness["write_performed"] is False
    assert readiness["candidate_count_loaded_for_write_scope"] == 1
    assert readiness["skipped_before_write_scope_validation"] == []
    assert readiness["dry_run_report_gate"]["approved"] is True
    assert readiness["result_label_write_approval"]["approved"] is False
    assert "--write-labels-approved" in readiness["planned_command_if_approved"]


def test_label_write_readiness_reports_stale_skips_without_blocking_clean_scope(
    tmp_path,
    monkeypatch,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    race_id = "Race 4 - WRGL - 2026-05-21"
    _write_snapshot(
        snapshot_dir,
        prediction_timestamp="2026-05-21T15:00:00",
        source_file_path=str(candidate_csv),
        extra_fields={
            "snapshot_readiness": {
                "status": "NOT_READY",
                "requirements": {"runner_rows_have_probabilities": False},
            }
        },
    )
    _write_snapshot(
        snapshot_dir,
        prediction_timestamp="2026-05-21T16:00:00",
        source_file_path=str(candidate_csv),
    )
    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [race_id],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
        require_ready_snapshot=True,
    )
    assert len(candidates) == 1
    assert [item["reason"] for item in skipped] == [
        "snapshot_not_ready_for_result_labels"
    ]
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--require-ready-snapshot",
            "--race-id",
            race_id,
            "--dry-run",
        ]
    )
    clean_report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(dry_args),
        candidates=candidates,
        skipped=skipped,
        ingested=[
            {
                "race_id": race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4],
            }
        ],
        failed=[],
        backup_path=None,
    )
    clean_report_path = tmp_path / "clean_dry_run_report.json"
    clean_report_path.write_text(json.dumps(clean_report), encoding="utf-8")
    readiness_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--require-ready-snapshot",
            "--race-id",
            race_id,
            "--approved-dry-run-report",
            str(clean_report_path),
            "--validate-label-write-readiness",
        ]
    )
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)

    gate = module.validate_clean_dry_run_report(
        report_path=str(clean_report_path),
        args=readiness_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        candidate_race_ids=module._candidate_race_ids(candidates),
    )
    readiness = module.build_label_write_readiness_report(
        args=readiness_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(readiness_args),
        candidates=candidates,
        skipped=skipped,
        dry_run_report_gate=gate,
    )

    assert gate["approved"] is True
    assert readiness["status"] == "READY_FOR_EXPLICIT_APPROVAL"
    assert readiness["skipped_before_write_scope_validation_by_reason"] == {
        "snapshot_not_ready_for_result_labels": 1
    }
    assert readiness["skipped_count_before_write_scope_validation"] == 1
    assert readiness["approval_required"] is True
    assert readiness["write_performed"] is False


def test_result_ingest_label_write_readiness_rejects_scope_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    race_id = "Race 4 - WRGL - 2026-05-21"
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', ?, 'pending')
        """,
        (
            race_id,
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()
    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [race_id],
    )
    assert skipped == []
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--race-id",
            race_id,
            "--dry-run",
        ]
    )
    clean_report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=Path(dry_args.snapshot_dir),
        write_approval=module.result_label_write_approved(dry_args),
        candidates=candidates,
        skipped=[],
        ingested=[
            {
                "race_id": race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4],
            }
        ],
        failed=[],
        backup_path=None,
    )
    clean_report_path = tmp_path / "clean_dry_run_report.json"
    clean_report_path.write_text(json.dumps(clean_report), encoding="utf-8")
    readiness_path = tmp_path / "readiness.json"
    monkeypatch.delenv("APPROVE_RESULT_LABEL_WRITE", raising=False)

    rc = module.main(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--approved-dry-run-report",
            str(clean_report_path),
            "--validate-label-write-readiness",
            "--output",
            str(readiness_path),
        ]
    )

    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert readiness["status"] == "NOT_READY"
    assert readiness["write_performed"] is False
    assert readiness["dry_run_report_gate"]["approved"] is False
    assert readiness["dry_run_report_gate"]["reason"] == "report_scope_mismatch"


def test_clean_dry_run_report_gate_rejects_official_incomplete_positions(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    candidate = _candidate(module, tmp_path)
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--dry-run",
        ]
    )
    write_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--write-labels-approved",
        ]
    )
    report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(dry_args),
        candidates=[candidate],
        skipped=[],
        ingested=[
            {
                "race_id": candidate.race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4, 5, 6, 7],
            }
        ],
        failed=[],
        backup_path=None,
    )
    report_path = tmp_path / "incomplete_official_dry_run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = module.validate_clean_dry_run_report(
        report_path=str(report_path),
        args=write_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        candidate_race_ids=[candidate.race_id],
    )

    assert report["clean_for_label_write"] is False
    assert report["label_write_blockers"] == [
        {
            "race_id": candidate.race_id,
            "source": "thedogs_official",
            "status": "resulted",
            "reason": "label_write_requires_complete_official_result_positions",
            "expected_box_count": 8,
            "result_box_count": 7,
            "missing_result_boxes": [8],
            "unexpected_result_boxes": [],
        }
    ]
    assert gate["approved"] is False
    assert gate["reason"] == "report_not_clean_for_label_write"


def test_clean_dry_run_report_gate_reports_official_terminal_statuses(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    candidate = _candidate(module, tmp_path)
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--dry-run",
        ]
    )
    write_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--write-labels-approved",
        ]
    )
    report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(dry_args),
        candidates=[candidate],
        skipped=[],
        ingested=[
            {
                "race_id": candidate.race_id,
                "dry_run": True,
                "source": "thedogs_official",
                "status": "resulted",
                "box_order": [1, 2, 3, 4, 6, 7, 8],
                "terminal_statuses": {"5": "FELL"},
            }
        ],
        failed=[],
        backup_path=None,
    )
    report_path = tmp_path / "terminal_status_dry_run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = module.validate_clean_dry_run_report(
        report_path=str(report_path),
        args=write_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        candidate_race_ids=[candidate.race_id],
    )

    assert report["clean_for_label_write"] is False
    assert report["label_write_blockers"] == [
        {
            "race_id": candidate.race_id,
            "source": "thedogs_official",
            "status": "resulted",
            "reason": "label_write_requires_terminal_status_support",
            "terminal_statuses": {"5": "FELL"},
        }
    ]
    assert gate["approved"] is False
    assert gate["reason"] == "report_not_clean_for_label_write"


def test_clean_dry_run_report_gate_rejects_partial_sportsbet_label_source(
    tmp_path,
):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    candidate = _candidate(module, tmp_path)
    dry_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--dry-run",
        ]
    )
    write_args = module.build_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--date",
            "2026-05-21",
            "--upcoming-dir",
            str(upcoming_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--write-labels-approved",
        ]
    )
    report = module._build_report(
        args=dry_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        write_approval=module.result_label_write_approved(dry_args),
        candidates=[candidate],
        skipped=[],
        ingested=[
            {
                "race_id": candidate.race_id,
                "dry_run": True,
                "source": "sportsbet_results_top4",
                "status": "partial_sportsbet_results",
            }
        ],
        failed=[],
        backup_path=None,
    )
    report_path = tmp_path / "partial_sportsbet_dry_run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = module.validate_clean_dry_run_report(
        report_path=str(report_path),
        args=write_args,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        candidate_race_ids=[candidate.race_id],
    )

    assert report["clean_for_label_write"] is False
    assert report["label_write_blockers"] == [
        {
            "race_id": candidate.race_id,
            "source": "sportsbet_results_top4",
            "status": "partial_sportsbet_results",
            "reason": "label_write_requires_complete_official_result",
        }
    ]
    assert gate["approved"] is False
    assert gate["reason"] == "report_not_clean_for_label_write"


def test_backup_db_creates_readable_pre_write_copy(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()

    backup_path = module.backup_db(db_path)

    assert backup_path.exists()
    assert sqlite3.connect(backup_path).execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_load_candidates_skips_today_race_before_jump(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '18:30', ?, 'pending')
        """,
        (
            "Race 4 - WRGL - 2026-05-21",
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "race_not_jumped:upcoming_not_jumped",
        }
    ]


def test_load_candidates_keeps_race_metadata_candidates_working(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', ?, 'pending')
        """,
        (
            "Race 4 - WRGL - 2026-05-21",
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
    )

    assert skipped == []
    assert len(candidates) == 1
    assert candidates[0].race_id == "Race 4 - WRGL - 2026-05-21"
    assert candidates[0].participants == FOUR_PARTICIPANTS


def test_require_ready_snapshot_blocks_csv_only_result_candidates(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '16:30', NULL, 'pending')
        """,
        ("Race 4 - WRGL - 2026-05-21",),
    )
    conn.commit()
    conn.close()
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
        require_ready_snapshot=True,
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "ready_prejump_snapshot_required",
        }
    ]


def _write_snapshot(
    snapshot_dir: Path,
    *,
    race_id: str = "Race 4 - WRGL - 2026-05-21",
    venue: str = "WRGL",
    race_number: int = 4,
    race_date: str = "2026-05-21",
    jump_time: str | None = "16:30",
    prediction_timestamp: str = "2026-05-21T16:00:00",
    source_file_path: str | None = None,
    snapshot_state: str = "pre_jump_feature_freeze",
    is_pre_jump_snapshot: bool = True,
    extra_fields: dict | None = None,
) -> Path:
    path = snapshot_dir / race_date / venue / f"{prediction_timestamp.replace(':', '')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema_version": "prediction_snapshot_v1",
        "race_id": race_id,
        "stable_race_key": f"{race_date}|{venue}|{race_number}",
        "race_date": race_date,
        "venue": venue,
        "race_number": race_number,
        "jump_time": jump_time,
        "jump_datetime": f"{race_date}T{jump_time}:00+10:00" if jump_time else None,
        "source_file_path": source_file_path,
        "lifecycle_status": "upcoming_not_jumped",
        "snapshot_state": snapshot_state,
        "is_pre_jump_snapshot": is_pre_jump_snapshot,
        "prediction_timestamp": prediction_timestamp,
        "feature_freeze_timestamp": prediction_timestamp,
        "model_version": "test-model",
        "snapshot_readiness": {"status": "READY", "requirements": {}},
        "predictions": [
            {
                "dog_name": "Alpha Runner",
                "box_number": 1,
                "win_prob_norm": 0.4,
                "predicted_rank": 1,
            },
            {
                "dog_name": "Bravo Runner",
                "box_number": 2,
                "win_prob_norm": 0.3,
                "predicted_rank": 2,
            },
            {
                "dog_name": "Charlie Runner",
                "box_number": 3,
                "win_prob_norm": 0.2,
                "predicted_rank": 3,
            },
            {
                "dog_name": "Delta Runner",
                "box_number": 4,
                "win_prob_norm": 0.1,
                "predicted_rank": 4,
            },
        ],
    }
    if extra_fields:
        snapshot.update(extra_fields)
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_load_candidates_falls_back_to_frozen_snapshot_when_metadata_missing(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, source_file_path=str(candidate_csv))

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert skipped == []
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.race_id == "Race 4 - WRGL - 2026-05-21"
    assert candidate.venue == "WRGL"
    assert candidate.race_number == 4
    assert candidate.race_date == "2026-05-21"
    assert candidate.race_time == "16:30"
    assert candidate.lifecycle_status == "jumped_pending_results"
    assert candidate.participants == FOUR_PARTICIPANTS
    assert candidate.participant_source == "snapshot"
    assert candidate.sportsbet_slug == "warragul"


def test_require_ready_snapshot_prefers_frozen_snapshot_over_csv_metadata_row(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES ('Race 4 - WRGL - 2026-05-21', 'WRGL', 4, '2026-05-21', '16:30',
                'https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4',
                'pending')
        """
    )
    conn.commit()
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, source_file_path=str(candidate_csv))

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
        require_ready_snapshot=True,
    )

    assert skipped == []
    assert len(candidates) == 1
    assert candidates[0].race_id == "Race 4 - WRGL - 2026-05-21"
    assert candidates[0].participant_source == "snapshot"
    assert candidates[0].participants == FOUR_PARTICIPANTS
    assert candidates[0].sportsbet_slug == "warragul"


def test_snapshot_fallback_rejects_current_csv_participant_mismatch(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(
        _four_runner_csv() + "\n5. Echo Runner\n",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, source_file_path=str(candidate_csv))

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "snapshot_csv_participant_mismatch",
            "snapshot_boxes": [1, 2, 3, 4],
            "csv_boxes": [1, 2, 3, 4, 5],
        }
    ]


def test_snapshot_fallback_requires_result_free_pre_jump_snapshot(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        prediction_timestamp="2026-05-21T15:00:00",
        source_file_path=str(candidate_csv),
        is_pre_jump_snapshot=False,
    )
    _write_snapshot(
        snapshot_dir,
        prediction_timestamp="2026-05-21T16:00:00",
        source_file_path=str(candidate_csv),
        extra_fields={"winner_name": "Alpha Runner"},
    )

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert candidates == []
    assert [item["reason"] for item in skipped] == [
        "not_frozen_pre_jump_snapshot",
        "snapshot_unreadable_or_not_result_free:ValueError",
    ]


def test_snapshot_fallback_requires_ready_snapshot_for_result_labels(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        source_file_path=str(candidate_csv),
        extra_fields={
            "snapshot_readiness": {
                "status": "NOT_READY",
                "requirements": {"final_runner_set_verified": False},
            }
        },
    )

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "snapshot_not_ready_for_result_labels",
            "runner_completeness": None,
            "snapshot_readiness": {
                "status": "NOT_READY",
                "requirements": {"final_runner_set_verified": False},
            },
        }
    ]


def test_load_candidates_uses_latest_snapshot_for_jump_time_guard(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        jump_time=None,
        prediction_timestamp="2026-05-21T15:00:00",
        source_file_path=str(candidate_csv),
    )
    _write_snapshot(
        snapshot_dir,
        jump_time="18:30",
        prediction_timestamp="2026-05-21T16:00:00",
        source_file_path=str(candidate_csv),
    )

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "race_not_jumped:upcoming_not_jumped",
        }
    ]


def test_frozen_snapshot_rescues_incomplete_metadata_row_without_jump_time(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES ('Race 4 - WRGL - 2026-05-21', 'WRGL', 4, '2026-05-21', NULL, NULL, 'pending')
        """
    )
    conn.commit()
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        jump_time=None,
        source_file_path=str(candidate_csv),
    )

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert len(candidates) == 1
    assert candidates[0].race_id == "Race 4 - WRGL - 2026-05-21"
    assert candidates[0].lifecycle_status == "jumped_pending_results"
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "race_not_jumped:upcoming_not_jumped",
        }
    ]


def test_frozen_snapshot_rescue_preserves_metadata_sportsbet_url(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    sportsbet_url = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
        "q1-lakeside/race-4-10524017"
    )
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'LADBROKES-Q1-LAKESIDE', 4, '2026-05-21', NULL, ?, 'pending')
        """,
        ("Race 4 - LADBROKES-Q1-LAKESIDE - 2026-05-21", sportsbet_url),
    )
    conn.commit()
    conn.close()
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - LADBROKES-Q1-LAKESIDE - 2026-05-21.csv"
    candidate_csv.write_text(_four_runner_csv(), encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        race_id="Race 4 - LADBROKES-Q1-LAKESIDE - 2026-05-21",
        venue="LADBROKES-Q1-LAKESIDE",
        source_file_path=str(candidate_csv),
    )

    candidates, _skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
        snapshot_dir=snapshot_dir,
    )

    assert len(candidates) == 1
    assert candidates[0].participant_source == "snapshot"
    assert candidates[0].sportsbet_url == sportsbet_url
    assert candidates[0].sportsbet_slug == "q1-lakeside"
