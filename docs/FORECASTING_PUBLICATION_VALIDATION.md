# Forecasting publication candidate validation

## Exact identity

| Item | Value |
| --- | --- |
| Candidate product commit | `e54bc58599d8e536ab918b26991ecbe97c023fe6` |
| Candidate product tree | `05d2c363959b9e1eb5a0a75d1940d06c9373e73f` |
| Canonical base commit | `02f59c147a7702a13eee154161a4be950aeb0e60` |
| Canonical base tree | `92fda0ab3eaee8de8bf6b66bea73a2d3e57aaac3` |
| Recovered completed source | `a9b16fd906838d1a5b0a7215b9401d2f31b28eae` |
| Recovered source tree | `91bd7a8a5cc426c1f87f3a56cae2f3655eadf9cb` |

The candidate was constructed as one direct child of the verified canonical base.
It deliberately excludes the recovered experiment's sandbox-control deletions and
`.gitignore` change. The only reviewed divergence from its product blobs is the
fail-closed operational symlink repair and its regression test.

## Commands and results

| Command | Exit | Result / immutable output hash |
| --- | ---: | --- |
| `git diff --check origin/master...e54bc585` | 0 | clean |
| `git fsck --full --no-dangling` | 0 | clean |
| `uv run --no-project --with ruff ruff check --select E9,F63,F7,F82 race_collection tests/race_collection scripts/import_legacy_v4_bundle.py scripts/verify_legacy_v4_loader.py` | 0 | configured fatal-error selection clean |
| JSON parse of all `config/race_collection*.schema.json` files | 0 | three schemas valid |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase7_operational.py -k operational_path_rejects_symlink_run_escape_and_loop` | 0 | `1 passed, 55 deselected`; log SHA-256 `9ca6255c1dbe9cd442a0ad569a77b371a38178af6a84b506b9182b659d74bceb` |
| `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase4_model_serving.py -k real_flask_canonical_route_and_every_sealed_evidence_adapter` | 0 | `1 passed, 57 deselected`; log SHA-256 `2d6ae44916e8ad9e662b40dc106561da025320888fb79ba4fb2f8725b37e1bb8` |
| Current-master overlap: `tests/test_predict_market_form_residual.py tests/test_daily_race_ingest_shadow_orchestrator.py tests/test_strict_win_odds_fixture_capture.py` | 0 | `472 passed`; log SHA-256 `af314386b4e402619f963723aee3738b01707f9e6cbf608928d3723b66d0390d` |
| Populated-schema upgrades: v17, v27, and inexact-v27 rollback tests | 0 | `3 passed, 21 deselected`; log SHA-256 `859cb3c722b838e55799ad3505e76703c453087b00371831a85ba67df2e884ce` |

## Incomplete gates

The complete retained `tests/race_collection` suite was run with the declared
`requirements/all.in` dependency layer and deliberately interrupted with exit
130 after `348 passed in 955.13s`. It had entered
`test_public_authenticated_499_race_evaluation_cannot_be_sealed`, which creates
an additional real 499-race SQLite baseline; this is neither a test failure nor
a runtime activation. Its partial log SHA-256 is
`b2cb5a949eb3b5574464d01fe43426517b6b5e8c8d7c65415595c33507931fef`.

The Phase 7 operational/runtime-adapter pair was likewise deliberately
interrupted with exit 130 after `37 passed, 27 subtests passed in 181.86s` while
constructing a large authenticated forward-evaluation fixture. Its partial log
SHA-256 is `5415d9c4010d9780b04d7f96e2ba69cf56e237a6ac07f468244a57a167c77971`.

Therefore this candidate is **PARTIAL, not ready for a draft PR**. No service,
database, queue, model, deployment, or live prediction was started. Runtime
proof remains **DATA_MISSING**.
