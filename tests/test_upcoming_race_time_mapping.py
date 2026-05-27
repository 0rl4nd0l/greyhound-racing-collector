import json
from datetime import datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from upcoming_race_browser import UpcomingRaceBrowser
from utils.race_lifecycle import JUMPED_PENDING_RESULTS, UPCOMING_NOT_JUMPED, classify_race_file


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
