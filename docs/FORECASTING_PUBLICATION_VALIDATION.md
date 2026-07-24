# Forecasting publication candidate validation

## Exact identity

| Item | Value |
| --- | --- |
| Candidate product commit | `e54bc58599d8e536ab918b26991ecbe97c023fe6` |
| Candidate product tree | `05d2c363959b9e1eb5a0a75d1940d06c9373e73f` |
| Normalized candidate commit | `e9e49816a33f751b680e3473f44b0c1cbc6f9775` |
| Normalized candidate tree | `07983c5d7169e9ffe8bf9e0ddf2294678974b2f7` |
| Canonical base commit | `02f59c147a7702a13eee154161a4be950aeb0e60` |
| Canonical base tree | `92fda0ab3eaee8de8bf6b66bea73a2d3e57aaac3` |
| Recovered completed source | `a9b16fd906838d1a5b0a7215b9401d2f31b28eae` |
| Recovered source tree | `91bd7a8a5cc426c1f87f3a56cae2f3655eadf9cb` |

The candidate was constructed as one direct child of the verified canonical base.
It deliberately excludes the recovered experiment's sandbox-control deletions and
`.gitignore` change. The only reviewed divergence from its product blobs is the
fail-closed operational symlink repair and its regression test. The normalized
candidate contains only mechanical Black/isort changes to added Python files;
the inherited canonical `app.py` remains unformatted and was not rewritten.
All retained validation evidence below applies to normalized candidate
`e9e49816a33f751b680e3473f44b0c1cbc6f9775` and tree
`07983c5d7169e9ffe8bf9e0ddf2294678974b2f7`; `e54bc585` is retained as
the pre-normalization product identity and changed-file selection anchor.

## Commands and results

| Command | Exit | Result / immutable output hash |
| --- | ---: | --- |
| `git diff --check origin/master...e54bc585` | 0 | clean |
| `git fsck --full --no-dangling` | 0 | clean |
| `PYTHONPYCACHEPREFIX=/tmp/greyhound-publication-pycache git diff --name-only origin/master...e54bc585 -- '*.py' \| xargs python3 -m py_compile` | 0 | output log SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `git diff --diff-filter=A --name-only origin/master...e54bc585 -- '*.py' \| xargs uv run --no-project --with black black --check` | 0 | output log SHA-256 `3b0060419dacfa6cf2755e94795dd038a846ed001813aeca11a9d39bb0d8e39d` |
| `git diff --diff-filter=A --name-only origin/master...e54bc585 -- '*.py' \| xargs uv run --no-project --with isort isort --check-only` | 0 | output log SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `uv run --no-project --with ruff ruff check --select E9,F63,F7,F82 race_collection tests/race_collection scripts/import_legacy_v4_bundle.py scripts/verify_legacy_v4_loader.py` | 0 | configured fatal-error selection clean |
| JSON parse of all `config/race_collection*.schema.json` files | 0 | three schemas valid |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase7_operational.py -k operational_path_rejects_symlink_run_escape_and_loop` | 0 | `1 passed, 55 deselected`; log SHA-256 `a06ca3399c90efeedaa1540b8fe768fde85074b384ca978711828b846e98309a` |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase4_model_serving.py -k real_flask_canonical_route_and_every_sealed_evidence_adapter` | 0 | `1 passed, 57 deselected`; log SHA-256 `2d6ae44916e8ad9e662b40dc106561da025320888fb79ba4fb2f8725b37e1bb8` |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/test_predict_market_form_residual.py tests/test_daily_race_ingest_shadow_orchestrator.py tests/test_strict_win_odds_fixture_capture.py` | 0 | `472 passed`; log SHA-256 `af314386b4e402619f963723aee3738b01707f9e6cbf608928d3723b66d0390d` |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_operations.py -k 'populated_schema_17_migrates_forward_to_latest_without_data_loss or populated_schema_27_partial_day_migration_preserves_prefix_and_defines_suffix or schema_28_refuses_inexact_v27_progress_and_rolls_back'` | 0 | `3 passed, 21 deselected`; log SHA-256 `f65e0e1fce86e00fd00436449a6957f55ef2cc179e58640952a9ed2f4a543adf` |

## Retained-suite completion and release boundary

The complete retained `tests/race_collection` suite was run against the
normalized candidate with the declared `requirements/all.in` dependency layer:

| Command | Exit | Result / immutable output hash |
| --- | ---: | --- |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection` | 0 | `457 passed, 36 subtests passed in 4080.62s (1:08:00)`; log SHA-256 `796a9060e5e91a9aa06e3c727292d3951b16074e709ef4ea60f975abb164dc8b` |

All code-validation gates now pass. Draft publication remains pending the final
fresh read-only review of this exact candidate head. No service, database,
queue, model, deployment, or live prediction was started. Runtime proof remains
**DATA_MISSING**.

All log hashes above are recomputable from the correspondingly named files in
`docs/forecasting_validation_logs/`. The two empty successful logs are retained
intentionally: the respective checkers produced no stdout or stderr.
