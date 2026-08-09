- `python3 -m py_compile` passed for the modified Python modules.
- Ruff passed for the modified implementation and test files.
- Focused network-policy suite passed: `31 passed`.
- Exact repository `manual_prediction` tier passed: `806 passed in 42.76s`.
- Forecasting and backend classifier contracts passed (`28` and `12` tests).
- Forecasting CI contract passed: `9` workflow YAML files and `5` focused checks.
- JSON parsing, hardening guards, and `git diff --check` passed.
- No-conftest mode was used as in the repository workflow; the normal host
  environment lacked application dependencies for the repository conftest.
- Adversarial policy cases include exact navigation allow; second/different
  navigation, unknown XHR/fetch, result-like details, websocket/event-stream,
  outcome API, untrusted host, unreviewed path, query-bearing asset, non-GET,
  and unknown resource denial; reviewed CSS/JS/image/font asset allow.
- The real browser/source path was not attempted, per the task boundary.
