# Manual independent capture V1 contract

`manual-independent-capture-v1` is an exact-race, research-only capture
boundary. GHU-050 defines and validates the artifact contract. GHU-051 adds the
executor for its lock, path, process, timeout, cancellation, and
terminal-artifact controls; its original fixture child protocol remains
unchanged. GHU-052 adds an immutable atomic evidence seal and read-only
verifier. GHU-060 adds one versioned live child behind the same executor, plus
an explicit default-off deployment binding. The lane still has no scoring,
canonical, Phase 7, or autonomous-runtime authority.

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

## GHU-051 executor and GHU-060 live child

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

Child stdout is an exact-field, bounded record. The parent accepts only
the requested URL/race hash, one pre-jump source class, final URL equal to the
requested exact race, HTTP status `200`, a control-free content type, and a
GHU-050-valid runner/odds set. The returned execution binds those response
values and the exact source-body hash for the immediate GHU-052 seal. It writes
fixed members beneath the UUID run directory and writes `terminal.json` last,
only after `validate_terminal_artifact` accepts the complete byte/hash binding.
The legacy fixture schema is still accepted. The live child uses
`manual_independent_capture_child_live_v1` and additionally binds the exact
readiness runner set supplied by the parent; it cannot substitute a race or
runner.

`src/predictor/manual_live_capture_child.py` is the only live source adapter.
It receives only the parent-exported URL, race identity/hash, dedicated manual
profile, and run directory. It opens one persistent manual Playwright context,
performs one `goto` to that exact canonical TheDogs URL, rejects redirects,
non-200 responses, login/challenge markers, result selectors, and malformed
runner rows, and closes the context on every terminal path. It reads only the
fixed runner-row selectors and explicit decimal WIN odds. It emits a canonical
runner/odds sidecar with media type
`application/vnd.greyhound.manual-live+json`; GHU-052 recognizes that media
through the distinct `manual_live_json_odds_v1` parser identity while retaining
the same strict odds, source-hash, timing, and outcome rejection semantics.
There is no discovery, retry, fallback, directory scan, autonomous state read,
or result/outcome request.

`src/predictor/manual_live_capture.py` is an explicit request-scoped wrapper,
not a service mode. It requires one caller-supplied canonical race document
and protected-path inventory, then delegates to the existing executor. The
deployment generator records hashes and entrypoints for this wrapper and child
but keeps the generated service and `MANUAL_RESEARCH_ENABLED=0` default-off.

## GHU-052 immutable evidence bundle

GHU-052 was refreshed against merged GHU-051 master
`47e76063cfa14d697a4f4805f75aeaf9d597762e` / tree
`5cc7625500e0d84979de365e5155b45ef28df6af`. The older planning ticket named
shared capture and scoring validators from its then-current base. The accepted
implementation instead consumes only the isolated GHU-051 execution and the
existing pure GHU-050 validator. It does not import a canonical receipt,
database, autonomous collector, forward-corpus, scoring, or browser surface.

`src/predictor/manual_independent_capture_sealer.py` accepts only
`CAPTURE_READY` with one source attempt, no cancellation, and cleanup proving
the leader reaped and process group absent. It re-reads canonical
`terminal.json`, revalidates the exact config, protected-path set, trusted
commit/tree/model/request/runner/odds expectations, and requires the run
directory to be the UUID child of the configured isolated `runs` root. The
actual source response must bind the same exact race URL and bytes, status 200,
an allowed content type for its source class, and no target outcome-shaped
JSON/CSV/HTML field. The fixture child protocol is
`manual_independent_capture_child_fixture_v2`; its source bytes carry the odds
that the sealer parses through a closed, versioned parser vocabulary. The live
child's normalized sidecar is likewise bounded source bytes; it uses the
distinct `manual_live_json_odds_v1` parser identity and is not an unrestricted
page-body capture.

The final directory is:

```text
<run>/sealed-evidence/<bundle-sha256>/
├── bundle.json
├── manifest.json
├── normalized/odds.json
├── producer/terminal.json
└── source/raw.bin
```

Raw source bytes are preserved verbatim. Normalized odds are canonical JSON and
retain the exact ordered runner set, capture/source timestamps, runner hash, and
odds hash. The bundle also binds the parser identity and a canonical hash of the
box/name/odds rows re-derived from those preserved bytes; a source/envelope odds
disagreement fails closed. `bundle.json` binds race URL/identity,
venue/date/race/jump, capture
start/end and final margin, one-attempt proof, response metadata and hash,
cleanup status, GHU-050 config/model/producer identities, and SHA-256 identities
for the executor, sealer, and all four versioned schemas. Both JSON schemas
reject unknown fields and retain `research_only=true`, `canonical=false`, and
the explicit Phase 7 exclusion.

Publication takes a per-run isolated seal lock, removes only safe stale staging
directories, writes every fixed member with exclusive no-follow opens and file
`fsync`, writes the manifest last, makes the staged tree read-only, fsyncs its
directories, and renames the complete staged directory within the same parent
filesystem. It then fsyncs the parent directory. Readers accept only the final
hash-named directory, require the exact closed member set, canonical JSON, and
all manifest/member/relationship hashes, and reject symlinks, partial output,
tampering, path escape, outcome leakage, or identity drift.

An exact concurrent or later replay verifies and returns the identical bundle.
Any disagreement fails closed and never replaces the existing directory.

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

GHU-060 acceptance supports only controlled-fixture proof of the live child
through the existing executor and sealer. It does not prove real live-source
success, a real pre-jump prediction, deployed runtime isolation, model quality,
canonical coverage, Phase 7 eligibility, training, promotion, EV, staking, or
betting.

GHU-053 may begin only from the accepted merged GHU-052 head and its reviewed
exact tree, with the read-only verifier as its sole capture input. It must retain
the current no-SQLite, no-result, research-only, canonical-false and Phase-7-
excluded boundary; the presence of a sealed bundle is not scoring authority.
