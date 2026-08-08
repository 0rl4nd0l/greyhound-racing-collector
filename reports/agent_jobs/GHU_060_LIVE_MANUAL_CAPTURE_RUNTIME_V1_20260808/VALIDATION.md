- `python3 -m py_compile` passed for all modified Python modules.
- Ruff passed for modified implementation and test files.
- Focused full manual tier command passed: `785 passed`.
- No-conftest mode was used as in the repository workflow; the normal host
  environment lacked application dependencies for the repository conftest.
- Tests include exact URL success, redirect, challenge, outcome, malformed
  response, invalid/missing odds, runner mismatch, timeout, no retry,
  navigation guard, parent cleanup, unchanged fixture behavior, GHU-052 seal,
  default-off binding, and worker guard.
- Classifier is recorded after the exact review commit in `RUN_OUTCOME.json`.
