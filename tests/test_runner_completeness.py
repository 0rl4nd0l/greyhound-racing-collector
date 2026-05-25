from utils.runner_completeness import analyze_csv_text_runner_completeness


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
