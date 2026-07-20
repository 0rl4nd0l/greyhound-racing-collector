# Validation

- Exact task-card validation: PASS.
- Allowlist diff check before implementation commit: PASS.
- Command-focused suite: 54 passed.
- Relevant collector/scorer regression suite: 521 passed, 1 skipped.
- Ruff on changed Python files: PASS.
- Python compile on changed Python files: PASS.
- `git diff --check`: PASS.
- Real receipt compatibility probe: PASS, 16 exact WIN/PLACE rows, SQLite URI
  `mode=ro` plus `PRAGMA query_only=ON`.
- Independent adversarial review: code findings resolved; live proof contract
  blocker confirmed independently twice.
- Full legacy suite: attempted with dependencies. The 2143-test run exposed
  baseline failures and its self-mutating `tests/test_e2e.py` collector aborted
  pytest near 35 percent. A second 2142-test run excluding only that file
  reproduced the failures and was terminated by the legacy browser harness at
  `tests/test_evasion_scraping.py` near 36 percent. Generated ignored fixtures
  were removed. See `full-pytest.log`.
- Optional live prediction: NOT RUN, `BLOCKED_TASK_CONTRACT`.

No target outcomes, model mutations, threshold changes, prediction persistence,
lock mutation, service/timer change, deployment, GitHub mutation, merge,
promotion, betting, or cohort cutoff occurred.
