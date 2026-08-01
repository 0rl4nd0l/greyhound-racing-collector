# Greyhound Operator UI V1 product, evidence, and authority contract

Status: authoritative product contract for Operator UI releases R1–R5.

## 1. Product boundary

The product is a private, research-only forecasting console. Its authoritative
application stack is the existing Flask/Jinja/static application rooted at
`app.py`, `templates/`, and `static/`. `fastapi_app/main.py` is secondary.
`frontend/`, `tgr_dashboard_server.py`, and a new frontend framework are not
product authority.

Every page and exported view MUST display:

> **RESEARCH ONLY — NOT FOR BETTING**

Every surface containing fixture or illustrative values MUST also display,
adjacent to those values:

> **PROTOTYPE DATA**

The UI MUST bind only to an explicitly configured private interface, use an
approved local/private access path, and require application authentication. It
MUST NOT be publicly bound or exposed. A private network location is not
authentication.

The UI observes repository and server-owned evidence. It does not own the
collector, browser, shared lock, canonical racing database/history, model
registry, training, deployment, or promotion. Collector code remains the sole
browser/shared-lock/capture/append authority. When current receipt reuse cannot
satisfy the selected mode, the predictor may synchronously invoke one
`scripts/shadow_autopilot_daemon.py capture-one` child process group.
`capture-one` is a collector-owned entrypoint using the same canonical browser,
daemon lock, validated Sportsbet capture, and append-only `live_odds`
implementation as the scheduled collector; it is not a second collector or
browser. The unchanged background timer is never interactive transport. UI/API
code owns no lock/browser/capture implementation and never directly signals or
manipulates collector or browser children. Existing Flask mutation routes are
not precedent for this product.

The plan's named `greyhound_PROJECT.md` and `greyhound_EVIDENCE.md` are absent.
The current equivalents used here are
`docs/CANONICAL_RACE_FORECASTING_SPEC.md`,
`docs/race_evidence_inventory.md`,
`docs/on_demand_race_prediction.md`,
`race_collection/manual_prediction_collector_request.py`,
`scripts/predict_race_now.py`, and `scripts/shadow_autopilot_daemon.py`.

## 2. Exact authority model

Role names, fields, hidden controls, client checks, session possession, and a
private bind address are not runtime authorization. Each request requires
server-side authentication and authorization for its exact operation.

| Level | Plan name | Permitted authority | Forbidden authority |
|---|---|---|---|
| 1 | Viewer | Read dashboard, races, evidence, corpus, models, health, and audit. | Any mutation. |
| 2 | Operator | After every R3 security and job gate, run one exact manual research prediction. | Service control, lock access, direct capture, retry, training, or promotion. |
| 3 | Researcher | Inspect evaluations and prepare frozen or draft, non-executing experiment specifications. | Training without separate authority or any runtime activation. |
| 4 | Owner review | Record or issue the separate owner decision authorizing a deployment, or review a qualifying promotion package. | Background or automatic promotion. |

R1 and R2 viewing map to Level 1; R3 prediction maps to Level 2; R4 draft
specifications map to Level 3; deployment and promotion review decisions map to
Level 4. A higher level does not inherit an unrelated operational capability.
Level 4 creates no in-UI service, deploy, activation, training, or promotion
button. R5 execution and promotion remain deferred and require separate future
authority.

The current primary Flask app has no proven authn, authz, role enforcement, or
CSRF protection. R3 MUST NOT expose a POST until all such gates in section 9
are implemented and verified.

## 3. Evidence envelope and observation rules

Every operational status, count, timestamp, version, probability, link, and
progress event MUST have an envelope containing:

- `source_kind`, immutable `source_identity`, content hash, and a server-owned
  allowlisted `source_locator`;
- `observed_at` or `generated_at` in UTC, server observation time, computed
  `age_seconds`, and the named policy below;
- availability (`present`, `missing`, `unreadable`, or `error`), schema and
  integrity result, and all reference hashes;
- exact race, runner, source, and time identity when race-scoped; and
- the single narrow `supported_claim`.

The browser supplies no filesystem path. Display source time and zone plus UTC
and AEST for race-scoped values. Hashes establish byte identity only; they do
not establish truth, quality, freshness, readiness, or authority. Unknown,
missing, stale, malformed, unreadable, errored, or divergent evidence MUST NOT
render as healthy, current, successful, empty, or numeric zero.

These policies are exhaustive; an adapter may not invent or defer a threshold:

- **P-DEPLOY-60:** a repository/deployment/installed-config observation is
  fresh for `age_seconds <= 60`. All named hashes must match.
- **P-COLLECTOR-FULL-DYNAMIC:** observe the installed
  `shadow-autopilot.timer` and `shadow-autopilot.service`. Derive `G`, the
  maximum seconds between consecutive activations over the timer's complete
  repeating schedule, from normalized `OnUnitInactiveSec` (generated default
  `15min`); derive `A` from timer `AccuracySec` (generated default `30s`) and
  `T` from service `TimeoutStartSec`. The generator's full-daemon command
  default is 840 seconds and its `TimeoutStartSec` is
  `max(timeout + 60, timeout * 4)`. Missing, non-finite, or non-positive `G`,
  `A`, or `T` is `DATA_MISSING`. Use only the server-configured full-daemon
  `shadow_autopilot_daemon_runtime/state.json` and the current full-daemon
  output's `daemon_run_report.json`. Validate their `run_id`, lifecycle status,
  and applicable timestamps inside this lane. A completed lane is fresh
  through its completion/generation time plus `G + A`; an active lane is fresh
  through its `generated_at + T`; after its applicable deadline it is `STALE`.
- **P-COLLECTOR-ODDS-DYNAMIC:** observe the installed
  `shadow-autopilot-odds-capture.timer` and
  `shadow-autopilot-odds-capture.service`. Derive `G` from the normalized
  `OnCalendar` schedule over its complete repeat (the generated schedule runs
  every minute except 02/17/32/47), `A` from `AccuracySec` (generated default
  `15s`), and `T` from `TimeoutStartSec`. The generator's odds-only command
  default is 600 seconds and its `TimeoutStartSec` is
  `max(timeout + 60, timeout * 2)`. Apply the same missing/non-positive rule
  and completed (`generated_at + G + A`) or active (`generated_at + T`) deadline
  rule within this lane. Use only the server-configured
  `shadow_autopilot_daemon_runtime/odds_capture_state.json`, the current
  odds-only output's `odds_capture_only_daemon_report.json`, and that report's
  `odds_capture_refresh_report` evidence (or its same-run
  `odds_capture_refresh_report.json` source where applicable). Validate
  `run_id`, lifecycle status, report/source path, and timestamps only among
  evidence belonging to that odds-only lifecycle.
- **P-COLLECTOR-AGGREGATE:** expose full-daemon and odds-only lane status
  separately. The lanes have distinct state, run IDs, and lifecycles and MUST
  NOT be required to share a run or cycle ID. A fresh lane plus a missing,
  stale, invalid, or unavailable other required lane is partial/degraded/
  unavailable according to that other lane's own evidence, never `DIVERGENT`
  merely because lane identities differ. `DIVERGENT` is reserved for
  conflicting evidence within one lane or for a shared deployed/source
  identity mismatch. Overall `HEALTHY` requires every lane required by the
  approved plan to be fresh and integrity-valid and all installed/deployed
  source identities to be consistent. Enabled/active unit state alone is
  never healthy evidence. Full-daemon `last_odds_capture_*` observations may
  be displayed as a cross-lane link, but never prove equality with the current
  odds-only run.
- **P-UPCOMING-300-PREJUMP:** listing and selected-race evidence is fresh only
  when `age_seconds <= 300` and server observation time is strictly before
  scheduled jump. Selected evidence, runner evidence, and their hashes must
  agree; at or after jump it is expired.
- **P-CURRENT-INDEX-1200:** predictor discovery reads only the collector-owned
  fixed packet
  `shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json`
  at its server-configured locator beneath the server-owned evidence root. The
  packet has schema `collector_current_race_index_v1` or
  `collector_current_race_index_v2`, canonical bytes, at most 32 races, a
  timezone-aware `source_generated_at`, and the sealed refresh-report path and
  SHA-256. Its configured maximum age is exactly 1200 seconds. Every v1 and v2
  row contains validated `date`, timezone-aware `jump_datetime`, stable
  `race_id`, bounded `race_id_aliases`, integer `race_number`, `race_time`,
  canonical TheDogs `race_url`, and non-empty `venue`; canonical URL
  date/race-number, stable ID, and race-ID uniqueness must agree. The exact row
  shape containing only those fields is v1-only; v2 additionally has the exact
  runner, provenance, source, and hash fields required by
  P-CURRENT-INDEX-V2-RUNNER-SEALED. Daemon
  `current_race_index_publish` evidence is `SKIPPED`, `REJECTED`, or
  `PUBLISHED`; only a matching `PUBLISHED` chain is usable. Fail closed with
  `CURRENT_INDEX_UNAVAILABLE`, `CURRENT_INDEX_STALE`,
  `CURRENT_INDEX_PATH_UNSAFE`, `CURRENT_INDEX_SOURCE_MISSING`,
  `CURRENT_INDEX_SOURCE_CHANGED`, `CURRENT_INDEX_SOURCE_INVALID`,
  `CURRENT_INDEX_INVALID`, `CURRENT_INDEX_SIZE_INVALID`, or
  `CURRENT_INDEX_UNBOUNDED` as applicable, including noncanonical bytes and
  oversized/unbounded input. This predictor allowance never relaxes
  P-UPCOMING-300-PREJUMP: UI selection still requires age no greater than 300
  seconds and observation strictly pre-jump. For both schemas, publication is
  an atomic canonical replacement at the sole fixed filename; safe-root,
  no-symlink, age, finite size/count/deadline, matching publication/source,
  failure, and atomic-publication rules are identical.
- **P-CURRENT-INDEX-V2-RUNNER-SEALED:** `GHU-022P` keeps that sole fixed
  filename and bounded legacy v1 reader compatibility for predictor discovery,
  but v1 is runnerless and MUST NOT source the `GHU-022` UI catalog. Schema
  `collector_current_race_index_v2` preserves every validated v1 race/time
  field and adds one non-empty canonically ordered unique final active runner
  set derived only inside the existing collector refresh/download flow from an
  accepted canonical-aligned, leakage-safe pre-race CSV/sidecar. Publication,
  bounded read, and later UI adaptation perform no new fetch, browser, scan,
  lock, caller-path resolution, or independent refresh-report interpretation.
  Each runner binds integer box, source display name, protocol-compatible
  normalized uppercase identity, explicit `ACTIVE` scratch state, and a
  source-native runner ID only when the accepted source supplies it; absence is
  explicit and IDs are never guessed. Duplicate box/normalized identity,
  empty/partial/ambiguous sets, unknown scratch state, or noncanonical ordering
  is invalid. The deterministic runner-set SHA binds the ordered set to exact
  race URL/date/venue/number/jump identity, pre-race source URL/timestamp,
  fixed-root source locators, and source byte hashes using the existing
  protocol contract where compatible. Safe fixed-root reads verify the sealed
  refresh report and runner-source bytes and fail closed on missing, changed,
  tampered, stale, unsafe, or mismatched identity. V2
  `current_race_index_publish` names v2 and carries exact packet SHA-256 and
  source/runner hashes; only `PUBLISHED` with matching packet, publication,
  source chain, and identities is usable. Atomic canonical write, finite
  sizes/counts/deadline, v1 predictor behavior, no retry/second collector, and
  no canonical database/history write remain unchanged.
- **P-CATALOG-60:** repository configs/schemas and deployed/model manifests
  are observed within 60 seconds and every expected SHA-256 matches.
- **P-BUNDLE-LIST-60:** a directory/index listing observation is fresh for 60
  seconds. For prediction bundles this means only the fixed producer index in
  P-PREDICTION-BUNDLE-SEALED, never a directory scan. Each listed bundle must
  independently pass P-IMMUTABLE-HISTORICAL; index freshness never makes its
  contents current health.
- **P-PREDICTION-BUNDLE-SEALED:** `GHU-023P` owns the producer/index/verifier
  prerequisite beneath the one server-configured private root
  `artifacts/on_demand_prediction_runs`. The sole producer-owned listing is an
  atomically replaced canonical `prediction_bundle_index_v1.json` with strict
  schema `on_demand_prediction_bundle_index_v1`; the UI/browser supplies no
  root, locator, directory, filename or path and no reader scans the root.
  The index is at most 512 KiB and 256 unique entries, ordered newest-first by
  timezone-aware `generated_at`, then ascending stable `prediction_id` as the
  deterministic tie-breaker. Each entry names only a directory matching
  `prediction_[0-9]{8}T[0-9]{12}[+-][0-9]{4}_[0-9a-f]{12}`, its stable
  `prediction_id` (a lowercase canonical UUID minted once at bundle creation),
  nullable `job_id` (null or the exact immutable UI operations-store job ID),
  generated time, terminal status,
  `blocker_stage`, manifest SHA-256 and aggregate logical bundle SHA-256.
  Index validation has a 1-second monotonic deadline and rejects duplicate or
  unknown fields/identities/statuses, noncanonical/nonfinite bytes, oversize,
  truncation and unsafe file types. Atomic publication is canonical temporary
  regular-file write, flush/fsync, same-directory replace and directory fsync;
  only producer-sealed v2 bundles are indexed. Every producer index update is
  serialized by the one fixed producer-owned sibling lock
  `prediction_bundle_index_v1.lock`, distinct from every collector/runtime
  lock. Acquisition is a descriptor-relative, no-follow, exclusive create of
  a canonical regular file under the retained root descriptor, bounded by the
  same 1-second monotonic index deadline. The producer records a fresh
  unguessable ownership token and its process identity, retains the opened
  lock descriptor, and verifies the constructed/opened name, type, device and
  inode before reading the current index or publishing its replacement.
  Contention, a pre-existing/stale lock, symlink, non-regular type, name or
  component replacement, identity mismatch, or deadline exhaustion fails
  closed without updating the index; stale locks are never inferred from age
  or process liveness and are never stolen. Release occurs in a `finally`
  path only by the acquiring producer, after publication or failure, and only
  when descriptor `fstat`, descriptor-relative name lookup and the stored
  ownership token still identify the same regular file; otherwise it leaves
  the lock untouched and reports failure. `GHU-023P` authorizes creation,
  validation and release of only this fixed lock during its producer index
  update. It grants no discovery, opening, waiting on, deletion, replacement,
  bypass or other manipulation of the live collector lock or any other lock.
  Existing/unindexed
  `on_demand_race_prediction_v1` and
  `on_demand_prediction_bundle_manifest_v1` bundles remain replay-compatible
  but catalog-ineligible and are never rewritten absent a separately
  authorized future migration.

  An indexed detail has strict schemas `on_demand_race_prediction_v2` and
  `on_demand_prediction_bundle_manifest_v2`, rejecting unknown fields. It has
  at most 32 manifest entries, 64 MiB per regular file, 256 MiB aggregate
  logical bytes, 1 MiB each for result and manifest, and a 5-second monotonic
  verification deadline. These conservative bounds cover the current finite
  output shape (canonical JSON/CSV evidence, model/config files and one sealed
  history database) while bounding the former recursive verifier.
  `bundle_manifest.json` is enumerated once as the schema-fixed control file;
  every other allowed regular file is enumerated exactly once in `files` by a
  validated single-component or slash-separated relative name. Names
  prohibit empty/dot/dot-dot components, absolute paths and platform
  separators, and duplicate canonical names are rejected. The allowed
  directory set is exactly the set of all proper parent-component prefixes
  derived from those canonical `files` names; it is not manifest-authored and
  cannot be enlarged. Under the 32-entry, byte and 5-second limits, the
  verifier opens the bundle and each derived directory once
  descriptor-relatively and performs bounded non-recursive enumeration of
  each. The observed entries must equal exactly `bundle_manifest.json`, the
  manifest regular-file set and that derived directory set at their respective
  parents. Every derived directory must be a no-follow directory and every
  manifest/control file a no-follow regular file. Any extra or missing entry,
  extra directory, symlink, device, FIFO, socket, other special file,
  traversal, duplicate, component/type/identity replacement, or enumeration
  that exceeds the fixed count/deadline bounds fails closed; no recursive or
  otherwise unbounded tree walk is permitted. The manifest itself is not
  self-hashed.
  `logical_bundle_sha256` is SHA-256 of canonical JSON bytes of exactly
  `{schema_version, prediction_id, job_id, files}`, where `files` is the
  lexicographically keyed manifest mapping of relative name to exact byte
  length and SHA-256. The index records the manifest file SHA-256 and this
  logical hash, avoiding an impossible self-hash while sealing all logical
  bundle content.

  Verification opens the configured root, indexed directory and every named
  object descriptor-relatively with no-follow semantics; each must be a
  regular file/directory on the retained root chain as applicable. It holds
  descriptors through finite reads, compares construction/open identities,
  rechecks `fstat` device/inode/type/size and root/component identities after
  reading, and fails on replacement or mutation. It independently binds the
  exact directory name, index row, result, manifest, nullable UI job identity,
  race identity (stable ID, canonical TheDogs URL, date, venue, race number
  and timezone-aware jump), config identity and canonical byte SHA-256, and
  model resolved identity/schema/manifest/artifact hashes. A valid
  `market_only_v1` model explicitly records artifact and artifact-manifest
  identity as `UNAVAILABLE_NOT_APPLICABLE` with null hashes; no other mode may
  omit required model evidence. No absolute bundle/source path is exposed or
  trusted; evidence references are validated names relative to the fixed root.

  Every terminal result has stable `prediction_id`, explicit nullable
  `job_id`, timezone-aware `generated_at`, exact terminal `status`, and
  `blocker_stage`. Status is exactly `PREDICTION_READY` or
  `PREDICTION_BLOCKED`; stage is null only for ready and otherwise exactly
  `PROTOCOL`, `VALIDATION`, or `SCORING`. A blocked result has one non-empty
  producer-owned blocker code and no probabilities. Producer-owned
  deterministic code maps known failures at the boundary where they occur;
  an unknown status, blocker code, or stage is integrity-invalid and MUST NOT
  be guessed. `PREDICTION_READY`
  alone may contain ranked finite probabilities, with unique runner/box
  identities, contiguous ranks, each probability in `[0,1]`, and canonical
  absolute sum error no greater than `1e-12`; blockers contain none.
  The read model distinguishes fixed index/root unavailable, invalid/integrity
  failure, verified protocol blocker, verified validation blocker, verified
  scoring blocker and verified success. Verified bytes receive only
  P-IMMUTABLE-HISTORICAL semantics: age is displayed and never proves current
  health, present model quality, promotion readiness or replay equivalence.
- **P-JOB-5-DEADLINE:** persisted UI job and protocol directories are scanned
  in one observation no older than 5 seconds. The checked-in deterministic
  fields are `discovery_seconds=12`, `lock_seconds=1`,
  `capture_seconds=60`, `validation_seconds=8`, `scoring_seconds=30`,
  `safety_seconds=15`, and `receipt_max_age_seconds=900`. The `capture-one`
  child timeout is `lock + capture + validation + safety`, currently `84s`.
  Fresh capture is eligible only with strictly more than `126s` to jump
  including discovery (`114s` after discovery); receipt reuse requires
  strictly more than `53s`. Request expiry is the exact target jump. The active
  phase/job deadline is the earliest applicable immutable request expiry,
  exact jump, durable job deadline, and current checked-in phase/budget limit.
  Missing, invalid, or drifted budget/config identity is `DATA_MISSING`. There
  is no response-polling wait, retry, or deadline extension. A passed deadline
  is `TIMED_OUT` or `EXPIRED`, never progress.
- **P-REPORT-24H:** a report observation is fresh for
  `age_seconds <= 86400`, with exact input population, source identity, and
  every chain hash matching.
- **P-OPS-5:** a future operations-store scan is fresh for 5 seconds and
  requires schema validation, monotonic/append-only identity, event/previous
  hash verification, and matching referenced hashes.
- **P-IMMUTABLE-HISTORICAL:** verified immutable bytes have no wall-clock
  expiry only for an explicitly historical, run- or slice-bound claim. Display
  age. Every manifest/reference hash must match. This policy can never make the
  artifact current, healthy, promotion-ready, or representative of present
  quality.

## 4. Source-to-screen contract

Each row names its owner, observation/integrity inputs, policy, behavior, and
only supported claim.

| Screen/card | Exact owner and integrity inputs | Policy and fail-closed behavior | Narrow supported claim |
|---|---|---|---|
| System/deployed identity | Git commit/tree; future generated UI deployment manifest; read-only installed unit/config content, hashes, working directory, and observation. | P-DEPLOY-60. Separate source and deployed identities. Missing manifest/input is `DATA_MISSING`; mismatch is `DIVERGENT`; stale observation is `STALE`. | Exact source and installed identities observed at the displayed time. |
| Collector — full daemon | Read-only installed `shadow-autopilot.timer`/`.service`, server-configured `shadow_autopilot_daemon_runtime/state.json`, and current full-daemon `daemon_run_report.json`, owned by `scripts/shadow_autopilot_daemon.py`. | P-COLLECTOR-FULL-DYNAMIC. Missing cadence/accuracy/timeout or required lane evidence is `DATA_MISSING`; intra-lane conflict is `DIVERGENT`; passed lane deadline is `STALE`. No UI service action. | Last internally consistent full-daemon lifecycle and whether its evidence is within that lane's derived deadline. |
| Collector — odds only | Read-only installed `shadow-autopilot-odds-capture.timer`/`.service`, server-configured `shadow_autopilot_daemon_runtime/odds_capture_state.json`, current odds-only `odds_capture_only_daemon_report.json`, and its embedded or same-run `odds_capture_refresh_report.json` evidence, owned by `scripts/shadow_autopilot_daemon.py`. | P-COLLECTOR-ODDS-DYNAMIC. Missing cadence/accuracy/timeout or required lane evidence is `DATA_MISSING`; intra-lane conflict is `DIVERGENT`; passed lane deadline is `STALE`. No UI service action. | Last internally consistent odds-only lifecycle and whether its evidence is within that lane's derived deadline. |
| Collector — aggregate | The two lane envelopes above plus matching installed/deployed source identities. | P-COLLECTOR-AGGREGATE. Show both lanes separately and preserve the unavailable/degraded state of either. Never compare their run IDs for equality. | Whether all plan-required collector lanes are individually fresh/integrity-valid under a consistent deployment identity. |
| Upcoming races | Collector-owned fixed packet `shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json` at schema `collector_current_race_index_v2`; matching v2 `current_race_index_publish`; sealed refresh-report and runner-source locators/hashes; exact packet and runner-set hashes; fixed server-owned locator and evidence root. | Validate P-CURRENT-INDEX-1200 and P-CURRENT-INDEX-V2-RUNNER-SEALED, then apply stricter P-UPCOMING-300-PREJUMP. Legacy v1 is predictor-compatible but catalog-ineligible. Read/adapt only the verified packet/publication/source chain. Browser/UI never supplies, derives, enumerates, or displays as input `--current-race-index`, evidence root, filesystem path, lock/browser/current-time argument; it never independently interprets a refresh report, browses, scans, fetches, scrapes, locks, or starts a browser. | Exact pre-jump race and ordered validated final active runner set from one verified collector v2 publication chain. |
| Model/config catalog | Finite checked-in configs and schemas resolved by `scripts/predict_race_now.py --list-configs`, plus matching frozen model/deployed manifest hashes. | P-CATALOG-60. Hash the exact observed files and normalized finite catalog; all repository, deployed, model, schema, and config identities must agree. Missing is `DATA_MISSING`; mismatch is `DIVERGENT`; invalid schema is `INVALID`. | Finite server-allowlisted model/config choices with exact byte identities. |
| Bundle list/detail | Fixed private bundle root and producer-owned `prediction_bundle_index_v1.json`; indexed strict-v2 `result.json` and `bundle_manifest.json`; exact logical, job/race/model/config and descriptor identities. | P-PREDICTION-BUNDLE-SEALED and P-BUNDLE-LIST-60; every detail independently passes the bounded descriptor-safe verifier and then only P-IMMUTABLE-HISTORICAL. Unavailable, integrity-invalid, protocol/validation/scoring blocker and verified success remain distinct. Never scan, trust/expose an absolute path, or catalog legacy/unindexed v1. | The verified terminal evidence of one producer-indexed historical prediction attempt, never current health or quality. |
| Prediction progress | Durable UI job/process identity; protocol and capture records; fixed current-index packet hash; `current_race_index_publish` status/hash; sealed source refresh-report path/SHA-256; runner-set, source, receipt, and bundle hashes; one fixed server-owned packet locator and evidence root. | P-CURRENT-INDEX-1200 and P-JOB-5-DEADLINE. Validate the complete source-to-screen chain and exact race/runner/model/config/job/process binding. Browser/UI never supplies, derives, enumerates, or displays packet/root/path/lock/browser/current-time arguments. No independent refresh-report interpretation, browsing, scan, fetch, scrape, lock, browser, or raw collector control. Probabilities require verified `PREDICTION_READY`. | Last persisted UI phase and exact packet/publication/source, protocol, capture, and scoring evidence. |
| Corpus readiness | Current matching report-only inventory and scorecard chain built by `scripts/build_race_evidence_inventory_packet.py`, including input population/source identity, exclusions, closure evidence, generated time, and chain hashes. | P-REPORT-24H. Exact input/source identities and all referenced report hashes must match. Missing/stale/mismatch blocks readiness. Raw DB counts never substitute. | Report-defined readiness and exclusions for the exact named population. |
| Model lineage/evaluation | Immutable model/config/manifest plus the matching named evaluation, rolling comparison, promotion-distance, and refinement-chain hashes and slice identities. | Artifacts use P-IMMUTABLE-HISTORICAL; a “current evaluation” listing uses P-REPORT-24H and must bind the exact model/config/source slice. Missing or mismatch is unavailable/divergent. A historical slice never becomes present quality or promotion evidence because bytes remain available. | Identity and reported evaluation for one explicitly named historical slice. |
| UI audit | No current store exists. Future separate UI operations store and its scan/integrity/reference-hash observation. | Now: `DATA_MISSING — UI audit store not implemented`. Future: P-OPS-5. Application logs do not substitute. | Future verified UI operation events only. |
| Draft experiment specifications | No current store exists. Future separate draft store with immutable corpus/model/report references, deterministic spec hash, and separate owner-decision record. | Now: `DATA_MISSING — draft store not implemented`. Future: P-OPS-5 plus P-IMMUTABLE-HISTORICAL for frozen specs. Drift/hash mismatch is `DIVERGENT`/`INVALID`. | A non-executing draft/frozen specification and its distinct review state. |

## 5. Exact race, runner, and temporal identity

The server issues a selection binding internal and source race IDs, source URL,
meeting/venue slug, local racing date, race number, scheduled jump in UTC and
source zone, distance/grade where required, and source observation identity. A
venue code alone is insufficient. Display AEST, UTC, and original source
time/zone.

The selection also binds ordered active runners: canonical/source identity,
box, exact normalized name, scratch state, and deterministic runner-set hash.
Submission re-resolves all identities server-side and rejects missing,
duplicated, changed, reordered, ambiguous, or substituted identity.

All prediction inputs must be observed strictly before jump and preserve
temporal cutoff and source timestamps. Same-day or undated history is excluded
when pre-jump safety is unprovable. Prediction execution cannot access outcomes
or results. Post-closure research may read authoritative results only after
closure, never as prediction input.

For the current-index v2 selection, each runner's source display name and
protocol-compatible normalized uppercase identity are distinct bound fields;
scratch state is exactly `ACTIVE`. A source-native runner ID is bound only when
present in the accepted source and is otherwise explicitly unavailable. The
ordered runner-set hash also binds the exact race identity and accepted
pre-race source URL/timestamp, locators, and byte hashes. No layer may infer a
missing ID, runner, scratch state, order, or source field.

## 6. Status and lifecycle vocabulary

Evidence states are `AVAILABLE/FRESH`, `STALE`,
`UNAVAILABLE/DATA_MISSING`, `INVALID/INTEGRITY_FAILED`, and `DIVERGENT`.
`WAITING` is a non-terminal job state with a finite deadline and last verified
event; it is neither collector health nor success.

UI phases may be `SUBMITTED`, `VALIDATED`, `WAITING_FOR_CLAIM`, `CLAIMED`,
`ATTEMPT_STARTED`, `RESPONSE_RECORDED`, `RECEIPT_VERIFIED`, `CONSUMED`,
`SCORING`, and terminal `PREDICTION_READY`, `FAILED`, `REJECTED`, `EXPIRED`, or
`TIMED_OUT`. They remain projections linked to exact source records.

Collector terminal statuses remain separate and exact: `RECEIPT_READY`,
`REQUEST_EXPIRED`, `RACE_NOT_FOUND`, `CAPTURE_WINDOW_CLOSED`,
`IDENTITY_MISMATCH`, and `CAPTURE_FAILED`. `RECEIPT_READY` is not prediction
success. Synchronous/public terminal classifications also include `BUSY`,
`CANCELLED`, and `INSUFFICIENT_PREJUMP_MARGIN`; these are not new V1 response
statuses. Each maps to a preserved V1 `CAPTURE_FAILED` response with a
deterministic reason, unless cancellation occurs after a receipt was already
sealed, in which case that receipt remains reusable. `BUSY` includes validated
existing-owner evidence and is immediate: there is no lock wait, steal, or
retry. At expiry or jump, terminal responses are normalized deterministically.
Where applicable, timestamps satisfy
`created_at <= claimed_at <= attempt.started_at <= responded_at <= consumed_at`.
One lifecycle is exactly one request, claim, attempt, response, optional
receipt, and consume. There is no retry or race substitution.

For R3 probability disclosure, the producer-owned v2 bundle additionally
seals the selected protocol request, claim, attempt, response, receipt,
consume, authenticated exact-receipt, and history-cutoff identities. The UI
store persists that exact producer chain with the sole claimed attempt's
response and completion events. Disclosure requires one unique durable chain
and exact equality with verified request/result/index/manifest bytes; absence,
duplication, ambiguity, or mutation withholds probabilities.

## 7. Release actions

| Release | Level | Permitted action |
|---|---:|---|
| R1 | 1 Viewer | Fixture-only navigation, filtering, drawers, desktop/mobile and accessibility review; every value is **PROTOTYPE DATA**. |
| R2 | 1 Viewer | Authenticated GET-only evidence-backed read models. |
| R3 | 2 Operator | After all gates, submit one server-issued exact race with finite server-owned model/config/odds choices and idempotency key; read/reconnect to that one job. |
| R4 | 3 Researcher | Inspect evaluation/corpus evidence and create a draft/frozen non-executing experiment specification. |
| Deployment/promotion review | 4 Owner review | Record a separate deployment decision or review a qualifying promotion package; no execution control. |
| R5 | Deferred | No executable UI action; training, activation, and promotion require a new contract and authority. |

Idempotency is actor-and-operation scoped. Retransmission with the same key
returns the same job; different inputs are rejected. One accepted job launches
at most one UI-owned fixed-argv predictor subprocess. That predictor publishes
at most one collector request and, only when capture is needed, starts at most
one collector-owned `capture-one` child process group; valid receipt reuse
starts none. Neither layer retries, launches a second predictor, acquisition,
or browser, republishes after an attempt, waits for or steals the lock, or
substitutes the race. Timeout or cancellation is terminal; the predictor
terminates and reaps its collector process group, and the UI never signals that
group directly. Restart or reconnect returns the same persisted job and may not
duplicate either process or request. This is one click to one job to one
predictor invocation, not one total operating-system process.

## 8. Forbidden controls and inputs

No release may expose or accept service/timer control; lock access or path;
browser/scraper/capture control; retry/requeue/substitution; canonical
database/history/evidence writes; arbitrary shell, command, argv, executable,
path, URL, database, root, output root, or current-time input; pre-closure
outcomes; training, fitting, persistence, registration, model pointer change,
activation, deployment, or promotion execution; EV, edge, staking, best-bet,
profitability, wagering, or betting output/action; or public/anonymous access.

The R3 worker constructs one fixed argv solely from server-owned allowlists and
never uses a shell. Its current-index and evidence-root values are fixed,
server-owned, and allowlisted; neither is browser/UI input or displayed as an
input. UI/API code never acquires the collector lock, starts a
browser, calls direct capture, writes canonical racing/history data, or
directly signals/manipulates collector or browser children. The predictor may
invoke the one fixed-argv collector-owned `capture-one` entrypoint described in
sections 1 and 7; the background timer remains unchanged and is not manual
request transport.

## 9. Release and proof gates

R1 requires fixture-only values, both mandatory labels, no operational
mutation, keyboard/focus/status accessibility, and supported desktop/mobile
layouts.

R2 requires server-side authn/authz; secure session/secrets; GET-only routes;
the separate append-only UI operations/audit store in section 10; a mandatory
access-audit append for every authenticated Level-1 operational GET before its
response is disclosed; deterministic fail-closed tests proving that an append
failure discloses no operational response; read-only/query-only canonical DB
connections; no-path allowlisted adapters; exact fresh/stale/missing/invalid/
divergent tests; accessibility, desktop/mobile, refresh/reconnect and
no-side-effect tests; exact HEAD/tree; reversible default-off feature flag; and
displayed source/deployed identity.

R3 requires R2 plus Level 2 authorization, CSRF, session rotation/expiry,
rate-limit and cross-actor isolation; durable monotonic job state and
idempotency; exact identity
revalidation; fixed argv/no-shell/no-path/no-root/no-time inputs; read-only
history DB; zero outcome leakage; no-lock/no-browser/no-capture proof; one
request/claim/attempt/response/optional receipt/consume; one predictor
invocation; stable timeouts; verified bundle before probabilities; and no
canonical, model, or runtime writes. Deterministic tests must prove
reconnect/progress follows the synchronous job, protocol, capture-one, and
scoring records rather than a
fabricated timer wait; cancellation/timeout reaps the child group without a
duplicate attempt; lock contention is immediate; the before-discovery,
after-discovery, and post-capture/reuse pre-jump margin checkpoints and
public-to-V1 status mapping are exact; timestamp ordering
`created_at <= claimed_at <= attempt.started_at <= responded_at <= consumed_at`
is preserved; and one click/job yields at most one predictor invocation, one
request, and zero or one `capture-one` child group.

Generated deployment requires separate Level 4 owner authority after
independent exact-head review: repository-generated unit/config, private bind,
secret handling, feature flag default-off, reversible disable/rollback without
evidence deletion, and matching deployed commit/tree/unit/config hashes.

The finite `repository-v1` composition is default-off. Its checked-in profile
defines exact relative locators and the one fixed generated-binding location;
the later repository-owned deployment generator must bind the deployed source,
pinned regular Python, authoritative collector DB/evidence/current-index,
producer bundle/protocol root, and separate writable Operator UI operations
root. A missing, malformed, symlinked, permission-unsafe, overlapping, or
incomplete binding/source fails startup. The application accepts no binding
path or arbitrary path, root, executable, command, lock, time, or model
location from environment, operator, API, or browser input.

One bounded live proof requires separate explicit live-action authority. It
observes exact deployed/generated identity and one natural collector cycle,
then creates one exact job from one click for one suitable exact race. It must
preserve the exact observed protocol, status, timestamps, hashes, raw
UI/protocol/bundle evidence, synchronous records, and bundle provenance;
perform no fabricated timer wait, retry/service workaround, outcome access, or
prohibited mutation; and stop after the first terminal result. On the naturally
reached path it must show at most one predictor invocation, one request, and
zero or one `capture-one` child group, as applicable. It must neither induce nor
claim contention, cancellation/timeout, margin-checkpoint, status-mapping, or
other failure branches not observed on that path. Runtime-proven may be claimed
only after valid scoring provenance.

R4 requires applicable R2 gates; P-REPORT-24H matching corpus chain; exact
lineage/evaluation evidence and claim limits; draft-only separate storage;
append-only audit; deterministic spec and immutable references; leakage checks,
non-overlapping forward/OOS windows, baseline, metrics, uncertainty, minimum
sample and exclusions; and separate owner state. It performs no training,
artifact persistence, registration, activation, promotion, or deployment.

## 10. UI operations and access-audit store

No current UI audit or draft store is claimed. Before R2 can release, a
separate append-only UI operations/audit store MUST exist. It MUST NOT be the
canonical racing database and MUST remain separate from the future Race
Collection operations/job database. Application or web-server logs do not
substitute. The same store may later carry R3/R4 UI operation events, while
draft content remains in its separate draft store.

Every authenticated Level-1 operational GET MUST append an access-audit event
before disclosing the operational response. At minimum, that event contains an
immutable event ID and schema, event time in UTC, authenticated actor identity
and level, session identifier, request identifier, route and HTTP method,
authorization decision and policy, evidence source identifiers and content/
reference hashes used for the response, deployed application/source identity
(including commit/tree/version as applicable), and response/status
classification.

Additional mutation or draft events include privacy-approved client identity,
idempotency-key hash, job/draft/race and runner-set identities, resolved
model/config/input hashes, prior/new state, status/reason, and referenced
protocol/artifact/report hashes. All events carry event and previous-event/
segment hashes and are insert-only; corrections append superseding events.
If the mandatory Level-1 access-audit append cannot complete and be confirmed,
the request fails closed with a deterministic non-operational error: the
server MUST NOT disclose the operational response or its evidence-derived
content. Mutation audit failure likewise fails closed. R2 tests MUST force
append unavailability/failure and prove both the error classification and
non-disclosure. Retention, access, backup, integrity verification, and
redaction policy must precede release.

## 11. Claims and terminal safety stops

Supported claims are limited to the narrow claim in a fresh, available,
integrity-valid envelope: observed identity; last observed collector cycle;
exact upcoming race/runner identity; finite catalog identity; verified
job/protocol phase; verified research probabilities; report-defined corpus
state; or evaluation on one named slice.

Unsupported claims include betting suitability, best bet, edge, profit,
staking, production readiness, present quality from historical bytes,
promotion readiness without the exact qualifying chain, or deployment success
without matching installed evidence. Mockup values support no operational
claim.

Stop the affected adapter, screen, action, job, or release at the first exact
blocker: corruption risk; possible outcome leakage; ambiguous/substituted
identity; required evidence that is missing, stale-for-action, malformed,
tampered, divergent, or unverifiable; an authn/authz/CSRF/session/isolation/
rate-limit/injection/traversal/public-exposure flaw; a failed exact-head,
security, API, no-shell/no-path/no-lock/no-browser/read-only/no-retry,
accessibility, responsive, reconnect, deployed-identity, natural-cycle,
rollback, or bounded-live-proof gate; or unresolved authority.

Fail closed. Never fabricate or downgrade a status, reuse stale evidence as
fresh, bypass a gate, broaden an input, retry, substitute a race, invoke an
adjacent legacy endpoint, authorize canonical writes or pre-closure outcomes,
execute training/promotion, emit betting guidance, or widen exposure.
## GHU-032C4 finite generated binding

`operator_ui_repository_binding_v1` has exact top-level `schema_version`, `profile_id`, `generator`, `deployment`, `profile_sha256`, `artifacts`, and `roots` members. The generator identity is `GHU-036-repository-v1-generator` / `operator_ui_repository_binding_generator_v1` / version `1`. Deployment identity must equal the fixed repository-v1 profile, and all five fixed prediction artifact hashes and the checked-in profile hash must match bounded, no-follow retained reads.
