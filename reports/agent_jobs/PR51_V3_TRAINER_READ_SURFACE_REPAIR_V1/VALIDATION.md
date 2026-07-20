# Validation

## Local gates

- Compile: PASS.
- Ruff on builder and focused test file: PASS.
- Focused tests: PASS, 84/84.
- Coverage: PASS, builder 81% with branches (1184 statements, 171 missed; 518
  branches, 132 partial).
- Exact-set/adversarial: PASS, including an unexpected thirteenth file and the
  complete path/type/link/declaration/length/hash matrix.
- Complete ten-file attacker view: PASS with zero dog-token/digest, source-path,
  sealed alignment-key, cross-race identity and development/OOT intersections.
- Two clean real builds: PASS, byte-identical by `diff -qr`.
- Domain hashes: PASS and descriptor-bound across 19 artifacts.
- Diagnostic isolation: PASS; trainer aggregate before and after diagnostic build
  is `97967ab3...4e31` and byte-identical.
- Full diff review and `git diff --check`: PASS; no raw or large data.

## Repository suite

Attempted with the documented `requirements.txt` environment using:

`uv run --no-project --with-requirements requirements.txt --with pytest --with pytest-cov pytest --junitxml=/tmp/pr51-trainer-surface-full.xml`

Collection stops before tests with exit 4:
`ModuleNotFoundError: No module named 'flask_compress'` from `app.py:47` while
loading `tests/conftest.py`. This is outside the seven-file repair and is not
reported as a green broad suite. Focused and live GitHub comprehensive checks are
the bounded publication evidence.

## Output boundaries

- Outcomes opened: false.
- Jul 11-Aug 9 source status: outcome-unopened pre-race freeze only.
- Model fit/evaluation: false.
- Market cohort or edge claim: false.
- Runtime/database/service access: false.
- PR 46-48 mutation: false.
- Merge/activation/ready-for-review mutation: false.
