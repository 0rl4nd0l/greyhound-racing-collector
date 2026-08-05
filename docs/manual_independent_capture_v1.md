# Manual independent capture V1 contract

`manual-independent-capture-v1` is an exact-race, research-only capture
boundary. GHU-050 defines and validates the artifact contract. GHU-051 adds a
fixture-only executor for its lock, path, process, timeout, cancellation, and
terminal-artifact controls. It does not provide a live fetch implementation,
scoring, UI, deployment, or runtime integration.

Every accepted configuration and terminal artifact is versioned, exact-field,
canonically serializable, and validated by
`src/predictor/manual_independent_capture.py`. JSON schemas live under
`configs/prediction/manual-independent-capture-v1/`.

## Authority matrix

| Surface | Allowed | Forbidden |
|---|---|---|
| Reads | One exact canonical TheDogs race; declared pre-jump source bytes; explicit research model and manual config bytes | Autonomous shared lock, browser profile, current index, and evidence root; canonical DB/history/`live_odds`/prediction bundles; shared model storage; forward corpus; collector requests/state; result evidence; Phase 7 artifacts/operations database; service/timer state |
| Writes | `<operations_root>/manual-independent-capture-v1/runs`, its fixed `browser-profile`, and `manual-capture.lock` | Every protected read surface and every path outside the isolated manual root |
| Lock | Exactly one manual process lock; at most one manual run | Inspecting, waiting on, acquiring, replacing, bypassing, or mutating the autonomous shared lock |
| Browser | Fixed manual-only profile | Reading, writing, reusing, or controlling the autonomous/shared browser profile or processes |
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

## GHU-051 fixture executor

`src/predictor/manual_independent_capture_executor.py` accepts one exact
canonical TheDogs URL and its complete bound race identity. It validates the
GHU-050 configuration, acquires only `manual-capture.lock` without waiting,
uses the fixed `browser-profile`, creates one UUID run directory beneath
`runs`, and launches at most one caller-supplied reviewed fixture command in a
new process session. The manual child environment contains only the exact race
identity and the fixed manual profile/run locations; the API has no database
argument, persistence-capable object, autonomous lock locator, retry, discovery,
or race-substitution surface.

The executor rechecks the configured pre-jump margin as its final wall-clock
action before launch. A second invocation emits `MANUAL_BUSY` without launching
a child. Cancellation and timeout use monotonic absolute deadlines, signal the
whole process group with TERM and then KILL when necessary, reap the leader,
and confirm that the process group is absent. PID, PGID, signal escalation, and
reap proof are returned as `ManualCaptureExecution`; unconfirmed cleanup emits
`PROCESS_REAP_UNCONFIRMED` and can never emit capture success.

Fixture stdout is an exact-field, bounded child record. The parent accepts only
the requested URL/race hash, one pre-jump source class, and a GHU-050-valid
runner/odds set. It writes fixed members beneath the UUID run directory and
writes `terminal.json` last, only after `validate_terminal_artifact` accepts the
complete byte/hash binding. Tests—not production configuration—supply the
fixture command. There is intentionally no CLI or default live/browser command.

## Terminal artifact

Every record binds:

- request and run UUIDs plus the canonical request hash;
- the exact canonical TheDogs URL, stable race identity, and scheduled start
  whose declared calendar date must equal the URL-bound race date, plus the
  configured margin (except `EXACT_RACE_INVALID`, which preserves only the
  rejected URL and has no selected identity);
- readiness, one absolute deadline, cancellation cleanup deadline, capture,
  source, closure, and terminal timestamps with exact ordering and margin
  arithmetic; source timestamps cannot predate readiness, cancellation grace
  cannot extend the absolute deadline, and unreaped-process failure is emitted
  exactly at its cleanup deadline;
- attempt count `1` and source attempt count `0` or `1` according to the
  failure class;
- strictly ordered runners with trimmed, control-free display/native IDs,
  uppercase ASCII identity, and finite decimal odds greater than `1`;
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

Source and artifact member locations are closed vocabularies, not user-selected
filenames. Each source content class and artifact role maps to exactly one fixed
relative path. Therefore outcome aliases such as `winners`, `placings`, or
`finishing-order` cannot be represented even when they evade a word blacklist.

Validation requires trusted caller expectations for the run UUID, request UUID,
canonical request hash, source commit, source tree, explicit research-model,
runner-set and odds hashes, and the complete expected source manifest. The
odds hash covers the capture timestamp plus the ordered box/odds rows, so
trusted odds cannot be moved to a different claimed instant. Each source row
binds its bytes/hash/timestamp/class to the exact selected race URL and
race-identity hash. Validation also requires mutable caller-owned replay inventories; a
successful validation records all three identifiers before returning, so reuse
through the same persisted inventory is terminally rejected.
All timestamps use exactly `YYYY-MM-DDTHH:MM:SS±HH:MM`; `Z` and fractional
forms are excluded so schema and runtime accept one deterministic vocabulary.
`EXACT_RACE_INVALID` is accepted only when the requested URL is not already an
exact canonical TheDogs race URL.

Canonical JSON uses sorted keys, compact separators, and one trailing newline.
Duplicate keys, unknown/missing fields, non-finite values, unsafe or unlisted
member paths, member/hash drift, race or request
disagreement, replayed IDs/hashes, a late non-timeout artifact, and conflicting
terminal states are rejected.

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

GHU-051 acceptance supports only: `the isolated executor is fixture-proven to
perform one exact-race attempt without canonical or autonomous state access`.
It does not support live-source/browser success, prediction readiness, deployed
runtime isolation, model quality, canonical coverage, Phase 7 eligibility,
training, promotion, EV, staking, or betting.

GHU-052 may begin only from an accepted exact GHU-051 executor diff and must add
the separately reviewed immutable evidence seal and tamper/identity validation.
The GHU-051 process result or a passing fixture alone is not sealed GHU-052
evidence and is not scoreable, canonical, or Phase 7-admissible.
