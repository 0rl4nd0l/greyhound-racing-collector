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
registry, training, deployment, or promotion. The scheduled collector remains
the sole browser/shared-lock/capture authority. Existing Flask mutation routes
are not precedent for this product.

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
- **P-CATALOG-60:** repository configs/schemas and deployed/model manifests
  are observed within 60 seconds and every expected SHA-256 matches.
- **P-BUNDLE-LIST-60:** a directory/index listing observation is fresh for 60
  seconds. Each listed bundle must independently pass P-IMMUTABLE-HISTORICAL.
- **P-JOB-5-DEADLINE:** persisted UI job and protocol directories are scanned
  in one observation no older than 5 seconds. Each phase deadline is the
  earliest of the immutable request `expires_at`, target jump, job deadline,
  and phase start plus its configured limit. Collector response waiting uses
  the exact checked-in config value (currently 600 seconds, schema maximum 900);
  absent/invalid limits are `DATA_MISSING`. A passed deadline is `TIMED_OUT` or
  `EXPIRED`, never progress.
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
| Upcoming races | Collector-produced current-run `refresh_prejump_report.json` and/or `odds_capture_refresh_report.json` selected-race evidence plus validated runner identities and runner-set hash, under collector ownership. | P-UPCOMING-300-PREJUMP. Missing/ambiguous race, jump, source URL, meeting, or runner identity is `UNAVAILABLE`; disagreement is `DIVERGENT`; post-jump is expired. UI/API reads MUST NOT call `UpcomingRaceBrowser`, start a browser, scrape, capture, or acquire a lock. | Exact pre-jump race and ordered validated runner set available for selection. |
| Model/config catalog | Finite checked-in configs and schemas resolved by `scripts/predict_race_now.py --list-configs`, plus matching frozen model/deployed manifest hashes. | P-CATALOG-60. Hash the exact observed files and normalized finite catalog; all repository, deployed, model, schema, and config identities must agree. Missing is `DATA_MISSING`; mismatch is `DIVERGENT`; invalid schema is `INVALID`. | Finite server-allowlisted model/config choices with exact byte identities. |
| Bundle list/detail | Private isolated bundle `result.json` and `bundle_manifest.json` produced by `scripts/predict_race_now.py`, with every manifest entry rehashed and exact job/race binding verified. | Listing uses P-BUNDLE-LIST-60; verified detail uses P-IMMUTABLE-HISTORICAL. Tamper/missing bytes are `INVALID`/`UNAVAILABLE`. Historical bundle age never makes it a current prediction. | The verified result of one named historical prediction run. |
| Prediction progress | Future persisted UI job state plus request, claim, attempt, response, receipt, and consume records owned by `race_collection/manual_prediction_collector_request.py`; verified bundle for scoring/result. | P-JOB-5-DEADLINE. Scan all sources together; validate schemas, hashes, uniqueness, ordering, exact job/race binding, and phase deadline. Missing events remain `WAITING` only before deadline; gaps cannot be fabricated. Probabilities require verified `PREDICTION_READY`. | Last persisted UI phase and corresponding exact protocol evidence. |
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
success. One lifecycle is exactly one request, claim, attempt, response,
optional receipt, and consume. There is no retry or race substitution.

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
at most one fixed-argument subprocess and publishes at most one collector
request. Timeout is terminal and authorizes no retry.

## 8. Forbidden controls and inputs

No release may expose or accept service/timer control; lock access or path;
browser/scraper/capture control; retry/requeue/substitution; canonical
database/history/evidence writes; arbitrary shell, command, argv, executable,
path, URL, database, root, output root, or current-time input; pre-closure
outcomes; training, fitting, persistence, registration, model pointer change,
activation, deployment, or promotion execution; EV, edge, staking, best-bet,
profitability, wagering, or betting output/action; or public/anonymous access.

The R3 worker constructs one fixed argv solely from server-owned allowlists and
never uses a shell. UI/API code never acquires the collector lock, starts a
browser, calls direct capture, or writes canonical racing/history data.

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
request/claim/attempt/response/optional receipt/consume; one invocation; stable
timeouts; verified bundle before probabilities; and no canonical, model, or
runtime writes.

Generated deployment requires separate Level 4 owner authority after
independent exact-head review: repository-generated unit/config, private bind,
secret handling, feature flag default-off, reversible disable/rollback without
evidence deletion, and matching deployed commit/tree/unit/config hashes.

One bounded live proof requires separate explicit live-action authority. It
observes installed/generated identity, one natural collector cycle, then runs
one suitable exact race once. It must preserve raw UI/protocol/bundle evidence,
perform no retry/service workaround/outcome access/prohibited mutation, and
stop after the first terminal result. Runtime-proven may be claimed only after
valid scoring provenance.

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
