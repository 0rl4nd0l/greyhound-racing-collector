# Manual independent capture V1 contract

`manual-independent-capture-v1` is a future exact-race, research-only capture
boundary. This change defines the boundary; it does not implement capture,
browser/process control, scoring, UI, deployment, or runtime integration.

Every accepted configuration and terminal artifact is versioned, exact-field,
canonically serializable, and validated by
`src/predictor/manual_independent_capture.py`. JSON schemas live under
`configs/prediction/manual-independent-capture-v1/`.

## Authority matrix

| Surface | Allowed | Forbidden |
|---|---|---|
| Reads | One exact canonical TheDogs race; declared pre-jump source bytes; explicit research model and manual config bytes | Autonomous shared lock; canonical DB/history/`live_odds`; forward corpus; collector requests/state; result evidence; Phase 7; service/timer state |
| Writes | `<operations_root>/manual-independent-capture-v1/runs`, its fixed `browser-profile`, and `manual-capture.lock` | Every protected read surface and every path outside the isolated manual root |
| Lock | Exactly one manual process lock; at most one manual run | Inspecting, waiting on, acquiring, replacing, bypassing, or mutating the autonomous shared lock |
| Browser | Fixed manual-only profile | Autonomous/shared browser profile or process authority |
| Downstream | Research-only review after terminal bundle closure | Canonical evidence, Phase 7 admission/example status, training, promotion, EV, staking, or betting |

`authority_matrix()` is the machine-readable equivalent. Configuration validation
requires the caller to supply all protected path categories out of band and
rejects lexical or resolved overlap. Protected locators are not embedded in the
manual process configuration.

## Configuration and paths

The example config fixes these derived paths:

```text
<operations_root>/manual-independent-capture-v1/
├── browser-profile/
├── runs/
└── manual-capture.lock
```

The policy values are constants: one concurrent manual run, one capture
attempt, no retry, and no replay. Minimum pre-jump margin, one absolute hard
timeout, and one cancellation cleanup grace are bounded integers. The
configuration cannot claim canonical or Phase 7 authority.

## Terminal artifact

Every record binds:

- request and run UUIDs plus the canonical request hash;
- the exact canonical TheDogs URL, stable race identity, and scheduled start
  whose declared calendar date must equal the URL-bound race date, plus the
  configured margin (except `EXACT_RACE_INVALID`, which preserves only the
  rejected URL and has no selected identity);
- readiness, one absolute deadline, cancellation cleanup deadline, capture,
  source, closure, and terminal timestamps with exact ordering and margin
  arithmetic; cancellation grace cannot extend the absolute deadline;
- attempt count `1` and source attempt count `0` or `1` according to the
  failure class;
- strictly ordered runner identity and positive finite decimal odds;
- source commit/tree bound to trusted caller-supplied expectations, config,
  model, request, race, runner-set, odds, source bytes/timestamps/content class,
  explicit target/same/future-outcome exclusion, and artifact member
  bytes/hashes;
- `research_only=true`, `canonical=false`, `phase7_excluded=true`,
  `phase7_eligible=false`, and
  `phase7_exclusion_reason="manual_research_only_noncanonical"`;
- closed-bundle assertions that outcome, Phase 7, and canonical-write access
  did not occur, with downstream admissibility fixed to
  `research_only_noncanonical_phase7_excluded`.

Canonical JSON uses sorted keys, compact separators, and one trailing newline.
Duplicate keys, unknown/missing fields, non-finite values, unsafe member paths,
member/hash drift, race or request disagreement, replayed IDs/hashes, a late
non-timeout artifact, and conflicting terminal states are rejected.

## Terminal vocabulary

`CAPTURE_READY` has no failure code. All other codes have one fixed status:

| Status | Failure codes |
|---|---|
| `BLOCKED` | `MANUAL_BUSY`, `EXACT_RACE_INVALID`, `INSUFFICIENT_PREJUMP_MARGIN`, `FEATURE_BLOCKED`, `SCORING_BLOCKED` |
| `FAILED` | `SOURCE_TIMEOUT`, `SOURCE_MALFORMED`, `IDENTITY_MISMATCH`, `RUNNER_SET_MISMATCH`, `ODDS_INVALID`, `PROCESS_REAP_UNCONFIRMED` |
| `CANCELLED` | `CANCELLED` |
| `TIMED_OUT` | `TIMED_OUT` |

Readiness blockers perform zero source attempts. Source/identity/runner/odds
failures and feature/scoring blockers bind exactly one. Cancellation, timeout,
and reap-unconfirmed records bind whether the one source attempt had begun.
Only `CAPTURE_READY`, `FEATURE_BLOCKED`, and `SCORING_BLOCKED` may retain closed
capture evidence; no terminal artifact can contain probability or outcome
fields.

## Claims boundary and next ticket

Acceptance supports only: `manual-independent-capture-v1 contract ready for
implementation`. It does not support capture success, prediction readiness,
runtime isolation, model quality, canonical coverage, Phase 7 eligibility,
training, promotion, EV, staking, or betting.

GHU-051 may begin only after the exact GHU-050 commit/tree is accepted, focused
contract/schema tests and CI pass, and exact-diff review confirms the authority
matrix, manual-only path/profile/lock ownership, one-attempt timing and terminal
vocabulary, provenance bindings, outcome exclusion, and structural Phase 7
exclusion without a blocking finding.
