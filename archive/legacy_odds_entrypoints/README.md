Legacy Odds Entrypoints
======================

These files are archived odds scraping entrypoints that are not part of the
current guarded Sportsbet odds capture path.

Current guarded odds flow:

- `scripts/autonomous_live_odds_capture.py`
- `odds_auto_integrator.py`
- `sportsbet_odds_integrator.py`
- `utils/prejump_sportsbet.py`

Archived files:

- `odds_scraper_system.py`: Selenium demo/legacy scraper marked deprecated in
  `docs/redundancy_matrix.md` and not referenced by active runtime imports
  during the June 18, 2026 archive audit.
