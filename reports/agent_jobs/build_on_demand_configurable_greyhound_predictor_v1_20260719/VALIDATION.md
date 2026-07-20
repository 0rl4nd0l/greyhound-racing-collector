# Validation

- Focused on-demand suite: `27 passed`.
- Combined on-demand, receipt, artifact scorer, and PR #46 dependency suites:
  `371 passed`.
- Independent dependency-only rerun: `344 passed`.
- Ruff check: pass.
- Ruff format check: pass.
- `py_compile`: pass.
- `git diff --check`: pass.
- Exact packaged command surface, `uv run scripts/predict_race_now.py --help`:
  exit 0 without ad hoc dependencies.
- GitHub PR #53 checks on implementation head
  `f9c732c94cc48e3069340e4fe7062a06eb2d4639`: hardening (2), backend Python
  3.11, comprehensive tests, Playwright, and Cypress all passed.
- Fixture CLI: canonical stdout, `PREDICTION_READY`, research-only, no production
  persistence.
- PR #46 known leak remains unfixed and blocks merge; its writer path was not run.
- Live browser/network behavior: not exercised because no live prediction was
  authorized.
