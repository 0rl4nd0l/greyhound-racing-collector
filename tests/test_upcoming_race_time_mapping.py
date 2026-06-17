import json
from datetime import datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from upcoming_race_browser import UpcomingRaceBrowser
from utils.race_lifecycle import JUMPED_PENDING_RESULTS, UPCOMING_NOT_JUMPED, classify_race_file
from utils.csv_metadata import normalize_target_grade


class FakeResponse:
    status_code = 200
    content = b"<html><body></body></html>"
    text = "<html><body></body></html>"

    def close(self):
        pass


def _browser(monkeypatch):
    browser = UpcomingRaceBrowser()
    monkeypatch.setattr(browser.session, "get", lambda *args, **kwargs: FakeResponse())
    return browser


def _links(html):
    soup = BeautifulSoup(html, "html.parser")
    return [(a, a["href"]) for a in soup.find_all("a", href=True)]


def test_live_scrape_maps_times_by_each_canonical_race_url(monkeypatch):
    browser = _browser(monkeypatch)
    links = _links(
        """
        <a href="/racing/horsham/2026-05-26/12/sportsbet-more-places">R12</a>
        <a href="/racing/horsham/2026-05-26/11/horsham-doors-and-glass">R11</a>
        """
    )
    monkeypatch.setattr(browser, "_find_race_links_fast", lambda soup, date: links)

    def exact_time(url):
        if "/11/" in url:
            return "5:40 PM"
        if "/12/" in url:
            return "6:05 PM"
        return None

    monkeypatch.setattr(browser, "_scrape_race_time_from_page", exact_time)

    races = browser._scrape_live_races_for_date("2026-05-26")
    by_race = {int(race["race_number"]): race for race in races}

    assert by_race[11]["race_time"] == "5:40 PM"
    assert by_race[12]["race_time"] == "6:05 PM"
    assert by_race[11]["race_time_mapping_status"] == "exact_url_match"
    assert by_race[12]["race_time_source"] == "canonical_race_url"
    assert all(race.get("time_source") != "estimated" for race in races)


def test_missing_canonical_race_time_is_flagged_not_estimated(monkeypatch):
    browser = _browser(monkeypatch)
    links = _links(
        '<a href="/racing/horsham/2026-05-26/12/sportsbet-more-places">R12</a>'
    )
    monkeypatch.setattr(browser, "_find_race_links_fast", lambda soup, date: links)
    monkeypatch.setattr(browser, "_scrape_race_time_from_page", lambda url: None)

    races = browser._scrape_live_races_for_date("2026-05-26")

    assert len(races) == 1
    assert races[0]["race_number"] == "12"
    assert races[0]["race_time"] is None
    assert races[0]["race_time_mapping_status"] == "missing_race_time"


def test_canonical_url_race_number_wins_over_visible_text():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        '<a href="/racing/horsham/2026-05-26/11/horsham-doors-and-glass">'
        "R12 6:05 PM</a>",
        "html.parser",
    )
    link = soup.find("a")

    race = browser.extract_race_info_from_link(
        link,
        link["href"],
        "2026-05-26",
    )

    assert race["race_number"] == "11"
    assert race["url"].endswith("/racing/horsham/2026-05-26/11/horsham-doors-and-glass")


def test_lifecycle_uses_corrected_sidecar_jump_time(tmp_path):
    path = tmp_path / "Race 12 - HOR - 2026-05-26.csv"
    path.write_text(
        "Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP\n"
        "1. Runner One,D,1,1,30.0,410,2026-05-01,HOR,Grade 5,23.1,23.1,23.0,,1.0,Other Dog,1,2.0\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "Race 12 - HOR - 2026-05-26.csv.metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "form_guide_download_provenance_v1",
                "race_info": {
                    "date": "2026-05-26",
                    "venue": "HOR",
                    "race_number": 12,
                    "race_time": "6:05 PM",
                    "race_time_source": "canonical_race_url",
                    "race_time_mapping_status": "exact_url_match",
                },
            }
        ),
        encoding="utf-8",
    )

    before_jump = datetime(2026, 5, 26, 18, 4, tzinfo=ZoneInfo("Australia/Melbourne"))
    after_jump = datetime(2026, 5, 26, 18, 6, tzinfo=ZoneInfo("Australia/Melbourne"))

    assert classify_race_file(path, now=before_jump).status == UPCOMING_NOT_JUMPED
    assert classify_race_file(path, now=after_jump).status == JUMPED_PENDING_RESULTS


def test_csv_provenance_writer_preserves_exact_race_time(tmp_path):
    path = tmp_path / "Race 4 - TOWNSVILLE - 2026-05-26.csv"
    path.write_text(
        "Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP\n"
        "1. Runner One,D,1,1,30.0,380,2026-05-01,TVLE,6,22.1,22.1,21.9,,1.0,Other Dog,1,2.0\n",
        encoding="utf-8",
    )
    browser = UpcomingRaceBrowser()
    browser._write_csv_provenance(
        str(path),
        race_url="https://www.thedogs.com.au/racing/townsville/2026-05-26/4/example",
        csv_info="https://www.thedogs.com.au/racing/townsville/2026-05-26/4/example/export-expert-form",
        content=path.read_text(encoding="utf-8"),
        completeness=type("Completeness", (), {"as_dict": lambda self: {"status": "COMPLETE"}})(),
        race_info={
            "date": "2026-05-26",
            "venue": "TOWNSVILLE",
            "race_number": "4",
            "race_time": "8:15 PM",
            "race_time_source": "canonical_race_url",
            "race_time_mapping_status": "exact_url_match",
        },
    )

    sidecar = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
    assert sidecar["race_info"]["race_time"] == "8:15 PM"
    assert sidecar["race_info"]["race_time_source"] == "canonical_race_url"
    assert sidecar["race_info"]["race_time_mapping_status"] == "exact_url_match"

    before_jump = datetime(2026, 5, 26, 20, 14, tzinfo=ZoneInfo("Australia/Melbourne"))
    lifecycle = classify_race_file(path, now=before_jump)

    assert lifecycle.status == UPCOMING_NOT_JUMPED
    assert lifecycle.jump_time == "20:15"


def test_canonical_pre_race_page_distance_and_grade_are_safe_metadata():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <section class="race-card">
              <dl>
                <dt>Race Distance</dt><dd>520m</dd>
                <dt>Race Grade</dt><dd>Grade 5</dd>
              </dl>
            </section>
            <table>
              <tr><th>PLC</th><th>TIME</th><th>BON</th></tr>
              <tr><td>1</td><td>29.90</td><td>29.80</td></tr>
            </table>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata["target_distance"] == "520m"
    assert metadata["target_distance_source"] == "canonical_pre_race_page"
    assert metadata["target_grade"] == "Grade 5"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"
    assert metadata["metadata_is_leakage_safe"] is True


def test_current_race_header_distance_and_grade_are_safe_metadata():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <div class="race-header">
              <div class="race-box__number">R10</div>
              <div class="race-header__info">
                <div class="race-header__info__name">Annual Pest Management Pathways H</div>
                <div class="race-header__info__grade">P5 366m</div>
              </div>
            </div>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/capalaba/2026-05-27/10/annual-pest-management-pathways-h?trial=false",
    )

    assert metadata["target_distance"] == "366m"
    assert metadata["target_distance_source"] == "canonical_pre_race_page"
    assert metadata["target_grade"] == "P5"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"
    assert metadata["metadata_is_leakage_safe"] is True


def test_current_race_header_accepts_ordinal_grade():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R4</div>
          <div class="race-header__info__grade">3rd/4th Grade 330m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/temora/2026-05-27/4/choppa-s-concreting?trial=false",
    )

    assert metadata["target_distance"] == "330m"
    assert metadata["target_grade"] == "3rd/4th Grade"


def test_current_race_header_title_class_words_are_safe_metadata():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R7</div>
          <div class="race-header__info__name">Ladbrokes Thunderbolt Heat 1</div>
          <div class="race-header__info__grade">457m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/goulburn/2026-05-29/7/ladbrokes-thunderbolt-heat-1?trial=false",
    )

    assert metadata["target_distance"] == "457m"
    assert metadata["target_grade"] == "Heat"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_target_grade_normalizes_official_shorthand_classes():
    assert normalize_target_grade("NG1-4") == "NG1-4"
    assert normalize_target_grade("M1/M2/M3") == "M1/M2/M3"
    assert normalize_target_grade("M5") == "M5"
    assert normalize_target_grade("PM") == "PM"
    assert normalize_target_grade("PM 390m") == "PM"
    assert normalize_target_grade("8:12 PM") is None
    assert normalize_target_grade("R/W") == "R/W"
    assert normalize_target_grade("RW") == "R/W"
    assert normalize_target_grade("N/P") == "N/P"
    assert normalize_target_grade("N-P PBD Stake PR2 Division2") == "N/P"
    assert normalize_target_grade("0-3 Win") == "0-3 Win"
    assert normalize_target_grade("1 - 4 Win") == "1-4 Win"
    assert normalize_target_grade("Best 8") == "Best 8"
    assert normalize_target_grade("Invitational") == "Invitational"
    assert normalize_target_grade("No Grade") == "No Grade"
    assert normalize_target_grade("Non Graded") == "Non Graded"
    assert normalize_target_grade("Special Event") == "Special Event"
    assert normalize_target_grade("Other") == "Other"
    assert normalize_target_grade("Other 515m") == "Other"
    assert normalize_target_grade("Other Dog") is None


def test_current_race_header_accepts_non_graded_class():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R5</div>
          <div class="race-header__info__name">Interior Constructions</div>
          <div class="race-header__info__grade">Non Graded 525m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/wagga/2026-05-29/5/interior-constructions?trial=false",
    )

    assert metadata["target_distance"] == "525m"
    assert metadata["target_grade"] == "Non Graded"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_current_race_header_accepts_special_event_class():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R4</div>
          <div class="race-header__info__name">Grv Distance Racing</div>
          <div class="race-header__info__grade">Special Event 680m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/geelong/2026-05-29/4/grv-distance-racing?trial=false",
    )

    assert metadata["target_distance"] == "680m"
    assert metadata["target_grade"] == "Special Event"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_current_race_header_accepts_other_class():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R8</div>
          <div class="race-header__info__name">Hst Tree Services Division1</div>
          <div class="race-header__info__grade">Other 515m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/launceston/2026-06-08/8/hst-tree-services-division1?trial=false",
    )

    assert metadata["target_distance"] == "515m"
    assert metadata["target_grade"] == "Other"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_current_race_header_accepts_official_shorthand_grade_codes():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R7</div>
          <div class="race-header__info__name">Ladbrokes Thunderbolt Heat 1</div>
          <div class="race-header__info__grade">NG1-4 457m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/goulburn/2026-05-29/7/ladbrokes-thunderbolt-heat-1?trial=false",
    )

    assert metadata["target_distance"] == "457m"
    assert metadata["target_grade"] == "NG1-4"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_current_race_title_accepts_np_class_when_grade_area_has_distance_only():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R9</div>
          <div class="race-header__info__name">Thedogssa N/P PBD Stake PR2 Division2</div>
          <div class="race-header__info__grade">342m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/angle-park/2026-06-02/9/thedogssa-n-p-pbd-stake-pr2-division2?trial=false",
    )

    assert metadata["target_distance"] == "342m"
    assert metadata["target_grade"] == "N/P"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_current_race_title_accepts_explicit_win_class_when_grade_area_has_distance_only():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R10</div>
          <div class="race-header__info__name">Greyhound Clubs NSW 1-3 Win</div>
          <div class="race-header__info__grade">350m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/goulburn/2026-05-29/10/greyhound-clubs-nsw-1-3-win?trial=false",
    )

    assert metadata["target_distance"] == "350m"
    assert metadata["target_grade"] == "1-3 Win"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_structured_current_race_title_class_words_are_safe_metadata():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <script type="application/ld+json">
          {
            "url": "https://www.thedogs.com.au/racing/goulburn/2026-05-29/11/goulburn-greyhounds-as-pets-masters",
            "name": "Goulburn Greyhounds As Pets Masters",
            "race_distance": "457m"
          }
        </script>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/goulburn/2026-05-29/11/goulburn-greyhounds-as-pets-masters",
    )

    assert metadata["target_distance"] == "457m"
    assert metadata["target_grade"] == "Masters"
    assert metadata["target_grade_source"] == "canonical_pre_race_page"


def test_generic_current_race_title_without_class_still_has_no_target_grade():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R3</div>
          <div class="race-header__info__name">Greyhounds Make Great Pets</div>
          <div class="race-header__info__grade">352m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/2026-05-29/3/greyhounds-make-great-pets?trial=false",
    )

    assert metadata["target_distance"] == "352m"
    assert "target_grade" not in metadata


def test_structured_target_metadata_requires_current_race_tie():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <script type="application/ld+json">
              {
                "url": "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
                "race_distance": "525m",
                "race_grade": "Grade 5"
              }
            </script>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata["target_distance"] == "525m"
    assert metadata["target_grade"] == "Grade 5"

    unrelated = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/8/other",
    )
    assert unrelated == {}


def test_ambiguous_multi_race_headers_are_rejected():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <div class="race-header">
          <div class="race-box__number">R7</div>
          <div class="race-header__info__grade">Grade 5 525m</div>
        </div>
        <div class="race-header">
          <div class="race-box__number">R8</div>
          <div class="race-header__info__grade">Maiden 400m</div>
        </div>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata == {}


def test_result_table_distance_and_grade_are_rejected():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <table class="results-table">
              <tr><th>Distance</th><td>525m</td></tr>
              <tr><th>Grade</th><td>Grade 5</td></tr>
              <tr><th>PLC</th><td>1</td></tr>
            </table>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata == {}


def test_unavailable_target_metadata_fails_closed():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup("<html><body><h1>Race 7</h1></body></html>", "html.parser")

    metadata = browser._extract_safe_target_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata == {}


def test_canonical_pre_race_page_weather_track_are_safe_metadata():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <section class="race-card">
              <dl>
                <dt>Track Condition</dt><dd>Soft</dd>
                <dt>Weather</dt><dd>Overcast</dd>
              </dl>
            </section>
            <table class="runner-form">
              <tr><th>Track</th><td>Good</td></tr>
              <tr><th>Weather</th><td>Fine</td></tr>
            </table>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_weather_track_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata["track_condition"] == "Soft"
    assert metadata["weather"] == "Overcast"
    assert metadata["weather_condition"] == "Overcast"
    assert metadata["weather_track_metadata_source"] == "canonical_pre_race_page"
    assert metadata["weather_track_metadata_is_leakage_safe"] is True


def test_result_table_weather_track_metadata_is_rejected():
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <table class="results-table">
              <tr><th>Track Condition</th><td>Soft</td></tr>
              <tr><th>Weather</th><td>Overcast</td></tr>
            </table>
          </body>
        </html>
        """,
        "html.parser",
    )

    metadata = browser._extract_safe_weather_track_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    assert metadata == {}


def test_csv_provenance_writer_records_extracted_weather_track_metadata(tmp_path):
    path = tmp_path / "Race 7 - MEA - 2026-05-21.csv"
    path.write_text(
        "Dog Name,Box\n"
        "1. Runner One,1\n",
        encoding="utf-8",
    )
    browser = UpcomingRaceBrowser()
    soup = BeautifulSoup(
        """
        <section class="race-card">
          <dl>
            <dt>Track Condition</dt><dd>Soft</dd>
            <dt>Weather</dt><dd>Overcast</dd>
          </dl>
        </section>
        """,
        "html.parser",
    )
    weather_track = browser._extract_safe_weather_track_metadata_from_page(
        soup,
        "https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
    )

    browser._write_csv_provenance(
        str(path),
        race_url="https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
        csv_info={"type": "direct_csv", "url": "https://example.test/export.csv"},
        content=path.read_text(encoding="utf-8"),
        completeness=type("Completeness", (), {"as_dict": lambda self: {"status": "COMPLETE"}})(),
        race_info={
            "date": "2026-05-21",
            "venue": "MEA",
            "race_number": "7",
            "race_time": "8:15 PM",
            "metadata_is_leakage_safe": True,
            **weather_track,
        },
    )

    sidecar = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
    assert sidecar["track_condition"] == "Soft"
    assert sidecar["weather"] == "Overcast"
    assert sidecar["weather_track_metadata_source"] == "canonical_pre_race_page"
    assert sidecar["weather_track_metadata_is_leakage_safe"] is True


def test_csv_provenance_writer_records_safe_target_metadata(tmp_path):
    path = tmp_path / "Race 7 - MEA - 2026-05-21.csv"
    path.write_text(
        "Dog Name,Box\n"
        "1. Runner One,1\n"
        "2. Runner Two,2\n"
        "3. Runner Three,3\n"
        "4. Runner Four,4\n",
        encoding="utf-8",
    )
    browser = UpcomingRaceBrowser()
    browser._write_csv_provenance(
        str(path),
        race_url="https://www.thedogs.com.au/racing/the-meadows/2026-05-21/7/example",
        csv_info={"type": "direct_csv", "url": "https://example.test/export.csv"},
        content=path.read_text(encoding="utf-8"),
        completeness=type("Completeness", (), {"as_dict": lambda self: {"status": "COMPLETE"}})(),
        race_info={
            "date": "2026-05-21",
            "venue": "MEA",
            "race_number": "7",
            "target_distance": "520m",
            "target_distance_source": "canonical_pre_race_page",
            "target_grade": "Grade 5",
            "target_grade_source": "canonical_pre_race_page",
            "metadata_is_leakage_safe": True,
        },
    )

    sidecar = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
    assert sidecar["target_distance"] == "520m"
    assert sidecar["target_distance_source"] == "canonical_pre_race_page"
    assert sidecar["target_grade"] == "Grade 5"
    assert sidecar["target_grade_source"] == "canonical_pre_race_page"
    assert sidecar["metadata_is_leakage_safe"] is True
