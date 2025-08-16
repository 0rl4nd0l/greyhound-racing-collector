Ad-hoc and manual test scripts

This folder contains ad-hoc/manual test utilities and exploratory scripts that are not part of the automated test suite.

Guidelines:
- Prefer writing proper unit/integration tests under `tests/` when feasible.
- Scripts here can be run manually via `python -m tests.scripts.<script_name_without_py>` if they support module execution, or `python tests/scripts/<script>.py` directly.
- Keep side effects safe and non-destructive. Avoid writing to production data.
- If a script becomes stable and broadly useful, consider converting it into an automated test or moving it into a proper tool/CLI.

Recently moved from repo root:
- concurrency_test.py
- debug_upcoming_races.py
- preview_data.py
- reproduce_json_error.py
- debug_call_chain.py
- run_batch_tests.py
- backtest_ballarat.py
- ev_backtest_sanity_check.py
- step5_validation_comprehensive_test.py
- step6_temporal_leakage_test.py

