## Corrective pass summary

PR #128 now uses a pure fail-closed live browser network allowlist. It permits
only the exact canonical race `GET` document navigation and query-free `GET`
stylesheet, script, image, or font assets on the exact canonical host under
`/assets/`. XHR, fetch, websocket, event-stream, API, result-like, subframe,
unknown, and unclassified requests abort. No odds API endpoint is allowed
because repository evidence did not prove one safe to admit.

The existing exact `goto`, GHU-051 executor/process/timeout/runner binding,
GHU-052 media/parser/sealing semantics, default-off deployment, and production
permission safety remain unchanged.

## Original implementation context

GHU-059 stopped because GHU-051 had only a fixture child and no bounded live
manual runtime. This change adds one versioned live Playwright child behind the
existing GHU-051 executor, parent-side exact readiness runner binding, and an
explicit request-scoped CLI with no discovery, retry, fallback, or substitution.

GHU-052 keeps its sealing and outcome-rejection semantics; it gains only the
distinct live JSON media/parser identity. The default-off GHU-056 package now
hash-binds the reviewed live entrypoint and child while remaining disabled.

## Safety and claims boundary

- Research-only, canonical=false, Phase-7 excluded.
- No scoring/model/DB/history/result/autonomous/betting/promotion changes.
- No live browser/source attempt, deployment, installation, activation,
  restart, lock manipulation, or merge was performed.
- This proves controlled-fixture traversal only; real live-source success and
  real pre-jump prediction remain the next ticket.

## Validation

- Exact base: `0a67f95ea06effa04609faabe0103fe2e69ff94e`.
- Corrective source head: `8bcb9cd574549871b5f6de71edd4a62e4a2a0cd7`.
- Classifier: `manual_prediction`, `ci_contract_changed=true`.
- Focused network-policy suite: `31 passed`.
- Exact manual tier: `806 passed`.
- Forecasting/backend classifier contracts: passed.
- YAML/JSON/schema, Ruff, compile, hardening, and diff checks: passed.
- Ruff, compile, diff-check, parent runner binding, GHU-052 sealing, default-off
  deployment binding, process cleanup, redirect/challenge/malformed/odds/
  timeout/result/no-retry tests passed.
