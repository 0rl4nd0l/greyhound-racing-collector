from utils.runner_completeness import (
    analyze_csv_text_runner_completeness,
    align_csv_text_to_canonical_final_runner_set,
    extract_canonical_runner_set_from_html,
    verify_final_runner_set,
)


def _source_report(participants):
    return {
        "schema_version": "runner_completeness_v1",
        "status": "COMPLETE",
        "runner_count": len(participants),
        "min_complete_runners": 4,
        "boxes": [row["box_number"] for row in participants],
        "dog_names": [row["dog_name"] for row in participants],
        "participants": participants,
        "duplicate_boxes": [],
        "duplicate_dog_names": [],
        "invalid_runner_rows": 0,
        "reasons": [],
    }


def _runner_row(box, name, *, scratched=False, into_box=None):
    classes = "accordion__anchor race-runner"
    if scratched:
        classes += " race-runner--scratched"
    into = f"(into box {into_box})" if into_box is not None else ""
    odds = "SCR" if scratched else "N/A"
    return f"""
    <tr class="{classes}">
      <td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td>
      <td class="race-runners__name">
        <div class="race-runners__name__dog">{name}
          <span class="race-runners__name__time">24.00</span>
          <span class="race-runners__name__box">{into}</span>
        </div>
      </td>
      <td class="race-runners__odds">{odds}</td>
    </tr>
    """


def _race_page(*rows):
    return f"""
    <html><body>
      <div class="race-header"><span class="race-box__number">R4</span></div>
      <table class="race-runners"><tbody>{''.join(rows)}</tbody></table>
    </body></html>
    """


def test_runner_completeness_extracts_full_embedded_form_field():
    content = "\n".join(
        [
            "Dog Name,PLC,BOX,DATE",
            "1. Alpha Runner,3,1,2026-05-20",
            '"" ,4,2,2026-05-15',
            "2. Bravo Runner,2,2,2026-05-20",
            "3. Charlie Runner,1,3,2026-05-20",
            "4. Delta Runner,5,4,2026-05-20",
        ]
    )

    report = analyze_csv_text_runner_completeness(content)

    assert report.status == "COMPLETE"
    assert report.boxes == [1, 2, 3, 4]
    assert report.runner_count == 4


def test_runner_completeness_flags_partial_field_like_shep():
    content = "\n".join(
        [
            "Dog Name,PLC,BOX,DATE",
            "2. Shima Lexie,6,3,2026-05-04",
            '"" ,8,8,2025-12-09',
            "4. Sekiro,3,4,2026-05-18",
        ]
    )

    report = analyze_csv_text_runner_completeness(content)

    assert report.status == "INCOMPLETE"
    assert report.boxes == [2, 4]
    assert "runner_count_below_min:2<4" in report.reasons


def test_comprehensive_collector_skips_blank_continuation_rows(tmp_path):
    from comprehensive_form_data_collector import ComprehensiveFormDataCollector

    csv_path = tmp_path / "Race 1 - SHEP - 2026-05-25.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Dog Name,PLC,BOX,DATE",
                "2. Shima Lexie,6,3,2026-05-04",
                '"" ,8,8,2025-12-09',
                "4. Sekiro,3,4,2026-05-18",
            ]
        ),
        encoding="utf-8",
    )
    collector = ComprehensiveFormDataCollector.__new__(ComprehensiveFormDataCollector)

    dogs = collector._identify_dogs_needing_enhancement(str(csv_path))

    assert dogs == ["Sekiro", "Shima Lexie"]


def test_final_runner_verifier_accepts_matching_canonical_active_boxes():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(3, "Charlie Runner"),
            _runner_row(4, "Delta Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )

    report = verify_final_runner_set(
        _source_report(
            [
                {"box_number": 1, "dog_name": "Alpha Runner"},
                {"box_number": 2, "dog_name": "Bravo Runner"},
                {"box_number": 3, "dog_name": "Charlie Runner"},
                {"box_number": 4, "dog_name": "Delta Runner"},
            ]
        ),
        canonical,
    )

    assert report["final_runner_set_status"] == "verified"
    assert report["canonical_active_boxes"] == [1, 2, 3, 4]
    assert report["mismatch_reason"] is None


def test_final_runner_verifier_treats_source_reserve_as_non_active_by_default():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(9, "Reserve Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )

    report = verify_final_runner_set(
        _source_report(
            [
                {"box_number": 1, "dog_name": "Alpha Runner"},
                {"box_number": 2, "dog_name": "Bravo Runner"},
                {"box_number": 9, "dog_name": "Reserve Runner"},
            ]
        ),
        canonical,
    )

    assert report["final_runner_set_status"] == "verified"
    assert report["source_reserve_boxes"] == [9]
    assert report["source_active_boxes"] == [1, 2]
    assert report["mismatch_reason"] is None


def test_final_runner_verifier_rejects_explicit_source_reserve_active_box():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(9, "Reserve Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )
    source = _source_report(
        [
            {"box_number": 1, "dog_name": "Alpha Runner"},
            {"box_number": 2, "dog_name": "Bravo Runner"},
            {"box_number": 9, "dog_name": "Reserve Runner"},
        ]
    )
    source["active_boxes"] = [1, 2, 9]

    report = verify_final_runner_set(source, canonical)

    assert report["final_runner_set_status"] == "mismatch"
    assert report["source_reserve_boxes"] == [9]
    assert "source_extra_active_boxes:9" in report["mismatch_reason"]


def test_final_runner_verifier_rejects_canonical_active_missing_from_csv():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(3, "Charlie Runner"),
            _runner_row(4, "Delta Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )

    report = verify_final_runner_set(
        _source_report(
            [
                {"box_number": 1, "dog_name": "Alpha Runner"},
                {"box_number": 2, "dog_name": "Bravo Runner"},
                {"box_number": 3, "dog_name": "Charlie Runner"},
            ]
        ),
        canonical,
    )

    assert report["final_runner_set_status"] == "mismatch"
    assert "source_missing_active_boxes:4" in report["mismatch_reason"]


def test_final_runner_verifier_reports_reserve_replacing_scratched_runner_without_repair():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(3, "Charlie Runner"),
            _runner_row(4, "Scratched Runner", scratched=True),
            _runner_row(9, "Reserve Runner", into_box=4),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )

    report = verify_final_runner_set(
        _source_report(
            [
                {"box_number": 1, "dog_name": "Alpha Runner"},
                {"box_number": 2, "dog_name": "Bravo Runner"},
                {"box_number": 3, "dog_name": "Charlie Runner"},
                {"box_number": 4, "dog_name": "Scratched Runner"},
                {"box_number": 9, "dog_name": "Reserve Runner"},
            ]
        ),
        canonical,
    )

    assert canonical["final_runner_boxes"] == [1, 2, 3, 4]
    assert canonical["scratched_boxes"] == [4]
    assert canonical["reserve_boxes"] == [9]
    assert report["final_runner_set_status"] == "mismatch"
    assert "box_4_name_mismatch" in report["mismatch_reason"]


def test_canonical_alignment_promotes_reserve_and_drops_scratched_runner():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(3, "Charlie Runner"),
            _runner_row(4, "Scratched Runner", scratched=True),
            _runner_row(9, "Reserve Runner", into_box=4),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )
    source_csv = "\n".join(
        [
            "Dog Name|Sex|PLC|BOX|DATE",
            "1. Alpha Runner|D|1|1|2026-05-01",
            "|D|2|2|2026-04-20",
            "2. Bravo Runner|D|1|2|2026-05-01",
            "3. Charlie Runner|D|1|3|2026-05-01",
            "4. Scratched Runner|D|1|4|2026-05-01",
            "9. Reserve Runner|D|1|8|2026-05-01",
            "|D|2|7|2026-04-20",
        ]
    )

    aligned_csv, alignment = align_csv_text_to_canonical_final_runner_set(
        source_csv,
        canonical,
        source="memory.csv",
    )
    aligned_report = analyze_csv_text_runner_completeness(aligned_csv)
    verification = verify_final_runner_set(aligned_report.as_dict(), canonical)

    assert alignment["status"] == "aligned"
    assert alignment["prediction_runner_count"] == 4
    assert alignment["dropped_participants"] == [
        {"box_number": 4, "dog_name": "Scratched Runner"}
    ]
    assert alignment["remapped_participants"] == [
        {
            "dog_name": "Reserve Runner",
            "source_box_number": 9,
            "final_box_number": 4,
            "original_box_number": 9,
        }
    ]
    assert "4. Reserve Runner" in aligned_csv
    assert "4. Scratched Runner" not in aligned_csv
    assert "9. Reserve Runner" not in aligned_csv
    assert aligned_report.boxes == [1, 2, 3, 4]
    assert verification["final_runner_set_status"] == "verified"


def test_canonical_alignment_drops_unpromoted_reserves_before_prediction():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
            _runner_row(9, "Reserve Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )
    source_csv = "\n".join(
        [
            "Dog Name|Sex|PLC|BOX|DATE",
            "1. Alpha Runner|D|1|1|2026-05-01",
            "2. Bravo Runner|D|1|2|2026-05-01",
            "9. Reserve Runner|D|1|8|2026-05-01",
        ]
    )

    aligned_csv, alignment = align_csv_text_to_canonical_final_runner_set(
        source_csv,
        canonical,
    )
    aligned_report = analyze_csv_text_runner_completeness(
        aligned_csv,
        min_complete_runners=2,
    )

    assert alignment["status"] == "aligned"
    assert alignment["prediction_runner_count"] == 2
    assert alignment["dropped_participants"] == [
        {"box_number": 9, "dog_name": "Reserve Runner"}
    ]
    assert "9. Reserve Runner" not in aligned_csv
    assert aligned_report.boxes == [1, 2]


def test_canonical_alignment_fails_closed_when_final_runner_missing_from_source():
    canonical = extract_canonical_runner_set_from_html(
        _race_page(
            _runner_row(1, "Alpha Runner"),
            _runner_row(2, "Bravo Runner"),
        ),
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )
    source_csv = "\n".join(
        [
            "Dog Name|Sex|PLC|BOX|DATE",
            "1. Alpha Runner|D|1|1|2026-05-01",
        ]
    )

    aligned_csv, alignment = align_csv_text_to_canonical_final_runner_set(
        source_csv,
        canonical,
    )

    assert aligned_csv == source_csv
    assert alignment["status"] == "not_aligned"
    assert alignment["reason"] == "canonical_participant_missing_from_source_csv"
    assert alignment["missing_canonical_participants"] == [
        {
            "box_number": 2,
            "dog_name": "Bravo Runner",
            "original_box_number": None,
        }
    ]


def test_final_runner_verifier_fails_closed_when_canonical_page_unavailable():
    canonical = extract_canonical_runner_set_from_html(
        "<html><body>No runners yet</body></html>",
        source_url="https://www.thedogs.com.au/racing/test/2026-05-27/4/example",
    )

    report = verify_final_runner_set(
        _source_report([{"box_number": 1, "dog_name": "Alpha Runner"}]),
        canonical,
    )

    assert report["final_runner_set_status"] == "unavailable"
    assert report["mismatch_reason"] == "no_race_runner_rows"
