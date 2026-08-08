## Summary

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

- Exact base: `1c937b53491787f1e54b16d235f7536af48c3c85`.
- Classifier: `manual_prediction`, `ci_contract_changed=true`.
- Focused manual tier: `785 passed`.
- Ruff, compile, diff-check, parent runner binding, GHU-052 sealing, default-off
  deployment binding, process cleanup, redirect/challenge/malformed/odds/
  timeout/result/no-retry tests passed.
