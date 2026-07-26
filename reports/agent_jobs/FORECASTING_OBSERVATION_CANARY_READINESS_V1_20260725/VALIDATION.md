# Validation

Validated implementation commit
`c56783af1a9a40bcb39a2c4a46fc07bd8fd33f50`, tree
`9c8e1279a54c673d9704efabb71cea1d73045123`, from canonical commit
`17f7b605b9f81c5a08a88fea8835aadf291cbfe7`.

## Focused repaired paths

Command:

```text
uv run --no-project --python 3.11 --with-requirements requirements/all.in \
  python -m pytest -q --noconftest \
  tests/race_collection/test_operator.py \
  tests/race_collection/test_phase7_runtime_adapter.py \
  tests/race_collection/test_phase7_operational.py::Phase7OperationalTests::test_release_manifest_and_generic_unit_validation \
  tests/race_collection/test_phase7_operational.py::Phase7OperationalTests::test_backup_and_restore_require_integrity_not_command_success
```

Result: exit 0; `23 passed, 13 subtests passed in 7.02s`.

Additional accepted-head results:

- Runtime adapter file: exit 0; `18 passed in 5.15s`.
- Operator and operations focus: exit 0; `27 passed in 5.62s`.
- Full Phase 7 operational file: exit 0; `57 passed, 40 subtests passed
  in 455.38s`.
- The one complete-suite failure, run alone once and then in a 20-run loop:
  21/21 passes.
- Exact preceding Phase 7 operational file plus that test: exit 0;
  `58 passed, 40 subtests passed in 445.58s`.

## Complete-suite disclosure

Command:

```text
uv run --no-project --python 3.11 --with-requirements requirements/all.in \
  python -m pytest -q --noconftest tests/race_collection
```

First result: exit 1; `1 failed, 477 passed, 40 subtests passed in 5590.77s
(1:33:10)`. The sole failure was
`test_checked_in_adapter_resumes_real_prefix_through_main_once`, where the
sanitized service boundary returned fail-closed code 69. It did not reproduce
in the focused checks above.

Second result: stopped by owner for time with SIGINT, exit 130. Approximately
72% had completed with no failure observed when stop was requested; the
terminal summary recorded `353 passed in 1251.80s (0:20:51)`, or 353/478
collected tests. This is not a complete passing-suite result.

Authoritative full-suite GitHub CI confirmation remains required before merge.

## Static and integrity gates

- Black `--check --diff` on the eight changed Python files: exit 0;
  `8 files would be left unchanged`.
- Flake8 on the eight changed Python files with
  `--select=E9,F63,F7,F82`: exit 0; count `0`.
- isort `--check-only --diff` on the eight changed Python files: exit 0.
- `py_compile` on the eight changed Python files and executable operator
  wrapper: exit 0.
- Draft 2020-12 meta-schema validation of
  `config/race_collection_runtime_input.schema.json`: exit 0;
  `schema-valid`.
- Operator wrapper `--help`: exit 0.
- `git diff --check` from canonical base: exit 0.
- Changed paths compared with the task-card allowlist: exit 0;
  `changed-paths-valid`.

A read-only repository-wide Black probe was not used as the gate because it
reported pre-existing formatting drift outside this task (`794 files would be
reformatted`, four could not be reformatted). It made no edits. The relevant
changed-file formatting gate is green.

## Runtime and data boundary

No service, timer, database migration, production write, deployment,
prediction, result handling, training, evaluation, promotion, model selection,
betting, legacy shutdown, or live-odds-capture action ran.
