Legacy Prediction Entrypoints
============================

These files are archived compatibility or debugging entrypoints that are not
part of the current live/shadow prediction path.

Current live/shadow prediction flow:

- `scripts/shadow_autopilot_daemon.py`
- `scripts/shadow_autopilot_v1.py`
- `scripts/refresh_prejump_upcoming.py`
- `scripts/run_shadow_non_tgr_rf_evaluation.py score-live`
- `scripts/autonomous_live_odds_capture.py` for the guarded odds lane

Archived files:

- `standalone_working_predictor.py`: independent debug predictor with no active
  runtime imports found during the June 18, 2026 archive audit.
