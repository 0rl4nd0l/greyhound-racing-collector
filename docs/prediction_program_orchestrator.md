# Greyhound prediction programme: orchestrator context

This is the coordination entrypoint for moving the project from trustworthy
pre-race collection to genuine forward predictions. It records the programme's
current state, unresolved acquisition seam, ordered next work, and authority
boundaries. It is not runtime configuration, capture authority, model evidence,
or promotion authority.

Read `prediction_program_state.json` beside this file for the exact verified
snapshot. Recheck Git, installed units, current index, processes, shared lock,
and live configuration before acting because those facts can change after the
snapshot time.

## Mission

Produce an honest prospective baseline as soon as the required inputs exist:

1. identify which fields were genuinely available before jump;
2. recover retained evidence only when source-native identity and timing remain
   provable;
3. capture unrecoverable evidence prospectively through the existing Race
   Collection Service;
4. freeze one predeclared cohort and seal predictions before results;
5. evaluate forecast quality against market and favourite baselines; and
6. investigate additional feature interactions only after the baseline.

Optional runner-state enrichment must not delay the first baseline. Unknown
data stays unknown; it is not imputed from post-race pages, names, row order, or
other weak aliases.

## Current outcome

The trustworthy prediction lifecycle is merged and the current collection
runtime is healthy, but the forward baseline is not active. At the verified
snapshot:

- `origin/master` and the deployed clean runtime both identified commit
  `c04cc378a4860dac1cbc42c66c7f79570bb1451e`;
- all eight current-master GitHub checks had completed successfully;
- both collection timers remained enabled;
- the installed service had no `--forward-baseline-config` argument;
- no dedicated forward-baseline operations database or bound configuration was
  found in the scoped runtime locations; and
- the live collector-owned index contained six eligible races across four
  venues on one racing date, with six numeric native race IDs and all 44
  native runner IDs numeric.

This proves that the native-identity path now works prospectively. The remaining
admission blocker is population across two dates: the forward cohort needs a
single fresh verified population of exactly 20 selected races spanning at least
three venues and two racing dates.

No Issue #159 forward cohort has been frozen. No Issue #159 deferred snapshot
prediction or result evaluation has run.

## What has landed

The relevant merge sequence is:

| Change | Outcome |
| --- | --- |
| PR #154 | Added the prospective source-only TheDogs five-window market-history adapter. |
| PR #157 | Isolated per-race official-result source rejection so unrelated collection continues. |
| PR #158 | Added semantic deferral matching while retaining exact raw-response hashes separately. |
| Issue #159 / PR #160 | Added feature-availability manifests, sealed evidence, and the exact 20-race forward-baseline lifecycle. |
| PR #161 | Bound pre-seal collection failures to append-only prediction quarantine. |
| PR #162 | Preserved corpus-only service operation when no baseline config is supplied. |
| PR #163 | Restored primary-only operational current-index publication and kept 20/3/2 composition at cohort freeze. |
| PR #165 | Bounded odds-priority deferrals so the primary collection path receives a natural opportunity. |
| PR #166 | Preserved TheDogs native identity through the primary refresh without adding requests. |
| PR #167 | Retained exact already-fetched primary race-page evidence and receipts. |
| PR #168 | Corrected live runner identity to numeric `data-runner-id` with matching runner URL corroboration. |
| PR #169 | Requested identity response encoding and retained the request/receipt contract without adding calls. |

The older uncommitted worktree
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-current-index-fix-20260824`
is evidence from an earlier base, not an implementation source. Its intended
identity and publication-ownership changes were superseded by PRs #163 and
#166–#169. Do not commit, rebase, or transplant that 14-file diff wholesale.

## Prediction paths and evidence classes

Keep these paths distinct when answering what the system currently predicts:

### Scheduled shadow model

The installed scheduled scorer uses a shadow-only Random Forest with a
78-column contract and 73 fitted inputs. Results, post-jump prices, future
history, and unauthorized enrichments are not scoring inputs.

### Report-only and frozen research

- `stage2_market_blend_70` is a report-only 70% normalized Sportsbet WIN / 30%
  Random Forest blend, not the active model.
- `market_form_residual_v1` remains research-only and not activated.
- Broad form/speed residual and pace-topology studies did not establish
  incremental signal over the market.
- The frozen Betfair 95/5 candidate and the later overround successor have
  separate protocols and untouched evaluation windows. Do not access, retune,
  merge, or reuse their frozen populations for Issue #159.

See `predictive_research_roadmap.md` for the research programme. That roadmap is
hypothesis inventory, not production or promotion evidence.

### Issue #159 forward baseline

This is the next genuine prospective experiment. It uses the existing
`ForwardSealedCorpus`, feature derivation, forecasting authority, deferred
snapshot prediction, and SQLite operations seams. It does not create a second
scheduler, prediction pipeline, scorer, or result path.

## Feature availability

Each candidate field must have one explicit classification:

- `READY_NOW`: independently verified as available by the race's feature
  cutoff and eligible for the production matrix;
- `FORWARD_CAPTURE`: potentially useful, but valid evidence must be gathered
  prospectively;
- `DEVELOPMENT_ONLY`: useful for research but inadmissible for production; or
- `EXCLUDED`: unavailable, unsafe, outcome-derived, or outside the experiment.

The manifest binds source schema, raw and normalized checksums, coverage,
completeness, derivation identity, timing, and finite blockers. Recursive
outcome fields and evidence received at or after the cutoff fail closed.

### Current core evidence

Depending on the sealed packet and active scorer, the core evidence includes:

- race venue, distance, grade, scheduled jump, and source identity;
- complete active runner set, box, and native runner identity;
- source-bound historical form and derived speed/context features;
- verified pre-jump Sportsbet WIN prices where present;
- collection timestamps, source paths, raw bytes, checksums, and receipts; and
- independently captured official results used only after the result barrier.

### Prospective market evidence

The TheDogs market-history design targets T-120, T-60, T-30, T-10, and T-2.
Each accepted observation requires immutable raw bytes and receipt, exact native
runner-set agreement, valid effective boxes, provider identity, and request
timing. Static `OPEN`, `LOW`, and `HIGH` values are extrema, not a temporal
trajectory.

The adapter and one-window canary are proven; that is not evidence of a full
five-window multi-race trajectory cohort.

### Optional runner-state enrichment

Prior weight, workload, health, trials, first-start lifecycle, and steward text
remain sparse or rights/schema constrained. Retrospective rows do not prove the
project possessed them before jump. Missing steward text remains `unknown`.

GRV Topaz is a Victoria-only technical candidate. The preserved rights request
is `READY_FOR_HUMAN_REVIEW`, not permission. Written route, cadence, raw
retention, derived-use, privacy, sharing, completeness, and revocation terms are
required before a live runner-state pilot. This optional lane does not block the
Issue #159 baseline. The preserved review packet is under
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-rights-review-20260822/`.

## Forward-baseline contract

The cohort must satisfy all of these conditions before any member is admitted:

- exactly 20 future ordinary races;
- at least three venues and two racing dates;
- one immutable ordered membership and cohort checksum;
- numeric source-native race and runner IDs;
- complete frozen active runner sets;
- exact race URLs, jump instants, cutoffs, and timezone;
- evidence received strictly before the feature cutoff;
- no discovery-time outcome access;
- no substitution, retry, or backfill; and
- only `READY_NOW` features in the production matrix.

Every member finishes as `ACCEPTED`, `QUARANTINED`,
`AUTHORIZATION_BLOCKED`, or `INTEGRITY_FAILED`. Every terminal member has either
an immutable prediction identity or an immutable prediction-quarantine
identity. The result barrier remains closed until all 20 members are terminal.

## Current acquisition seam

The live production flow is:

```text
systemd Race Collection Service
  -> scripts/shadow_autopilot_daemon.py
  -> scripts/shadow_autopilot_v1.py
  -> scripts/refresh_prejump_upcoming.py
  -> UpcomingRaceBrowser.download_race_csv()
  -> retained raw page / receipt / CSV / sidecar evidence
  -> current_index_metadata_selection()
  -> publish_current_race_index_after_refresh()
  -> collector_current_race_index_v2
```

PRs #163 and #165–#169 resolved the publication-owner, starvation,
raw-page-retention, transport, and source-native identity problems without
increasing acquisition calls. The operational index is now a non-empty bounded
set of individually valid races from one completed primary refresh. Odds-only
cycles cannot publish it.

The full primary path is still invoked with `--days-ahead 1 --refresh-limit 6`.
It therefore cannot present 20 races across two dates in one verified packet.
The forward service has no cross-packet candidate accumulation.

## Agreed next design: bounded horizon inventory

The next change should deepen the existing primary discovery-to-index path. It
should not raise full per-race collection from six to twenty every 15 minutes.
That naive change was estimated to add at least 56 TheDogs requests per cycle,
or about 5,376 requests per day, plus possible weather calls.

The target design is:

1. discover candidate race metadata across at least two local racing dates;
2. acquire canonical native identity only for newly observed candidates;
3. retain and reuse verified identity evidence through the established
   artifact/sidecar path;
4. keep the frequent 0–60-minute near-race form, weather, and odds refresh
   unchanged;
5. remove expired, changed, substituted, or identity-conflicting candidates;
6. publish one fresh, independently verifiable candidate population from the
   existing primary publisher;
7. enforce explicit per-cycle and per-day request budgets; and
8. preserve primary-only publication and the existing shared lock.

Before implementation, quantify discovery requests, per-new-race identity
requests, steady-state calls, worst-case calls, retry ceilings, and expiry
behaviour. If the existing endpoints cannot implement this without a new
request class or cadence increase, return the exact blocker and request
separate authority.

Completion means deterministic tests can assemble a valid 20/3/2 population
from separately observed, still-fresh, immutable candidate evidence without
recollecting full far-future feature packets. Offline success does not prove
live source coverage.

## Ordered execution plan

### Stage 1: implement bounded horizon inventory

Start from current `origin/master` in a durable clean sibling worktree. Do not
reuse the stale uncommitted d41 diff as the base. Use TDD and independently
review the exact final diff against both repository standards and this contract.

Terminal outcome: a green draft PR, or one exact acquisition/design blocker.

### Stage 2: merge and deploy

Merge only with explicit authority and green exact-head checks. Deploy the exact
merge from a clean worktree. Verify generated/installed units, timers, command
arguments, process ownership, shared lock, and one naturally completed primary
cycle. Leave any active one-shot untouched.

### Stage 3: bounded live horizon proof

Use a separately authorized request budget. No retry or substitution is implied.
Prove that one fresh collector-owned population can reach 20 races, three
venues, two dates, numeric native identities, and complete runner sets. Retain
raw bytes, receipts, hashes, request counts, and every rejection.

### Stage 4: bind baseline runtime

Only after Stage 3, create a dedicated operations database migrated through
`race_collection/migrations/0031_forward_baseline_prediction_quarantine.sql`
and one canonical `forward-baseline-capture-service-config-v1` document. Its ten
closed fields bind cohort ID, operations database, corpus/evidence/index paths,
index age and read timeout, feature cutoff, and timezone.

Add `--forward-baseline-config` to the existing service path. A below-floor
preflight must return zero-write `AWAITING_COHORT_CANDIDATES`.

### Stage 5: freeze and predict

Freeze the first exact qualifying cohort once. Capture each member through the
existing scheduler and sealed-evidence seam. Commit a deferred snapshot
prediction or explicit quarantine for every race before opening results.

### Stage 6: evaluate

Freeze the metrics and comparator population before result access. At minimum,
report coverage, quarantine rate, favourite/market baseline, top-selection
accuracy, log loss, Brier score, calibration, ranking quality, uncertainty, and
receipt-bound market-relative diagnostics. One 20-race cohort validates the
lifecycle; it does not establish ROI, edge, or promotion readiness.

### Stage 7: iterate on information, not complexity

After the baseline, prioritize market disagreement and genuine temporal market
movement. Then test opponent-adjusted ability, dynamic ratings, adjusted speed,
pace/congestion, fast-non-favourite regimes, and later rights-cleared
runner-state evidence. Compare each addition against the frozen market and
incumbent model on the same eligible population.

## Authority and preservation rules

- Treat diagnosis, implementation, commit, push, PR, merge, deployment, live
  acquisition, database creation, cohort freeze, result access, fitting,
  promotion, EV/ROI reporting, and betting as separate stages.
- Preserve dirty worktrees, untracked research artifacts, sealed evidence,
  rejection receipts, and active processes.
- Use a clean sibling worktree from current `origin/master` for publishable
  changes.
- Preserve natural shared-lock ownership. Wait for an active process rather
  than stopping, restarting, stealing, deleting, or bypassing its lock.
- Keep exact raw-byte SHA-256 integrity separate from semantic matching
  fingerprints.
- Use numeric source-native identity. Names, boxes, row order, slugs, and local
  IDs cannot replace it.
- Keep missing fields and ambiguous results visible. Quarantine them instead of
  inventing values or silently changing the denominator.
- Do not access outcomes to select the cohort, choose features, tune the model,
  or decide which failed races to omit.
- Existing scheduled predictions, the Issue #159 baseline, frozen Betfair and
  overround cohorts, report-only research, and optional runner-state pilots are
  separate evidence lanes.
- Use ordinary repository workflows and the globally installed Matt Pocock
  skills. Legacy Tenn/V2 machinery applies only when explicitly requested.

## Source map

Read these sources instead of expanding this document with duplicated detail:

- `CONTEXT.md`: canonical domain language and lifecycle boundaries.
- `docs/adr/0014-require-the-collector-owned-current-race-index.md`: operational
  index ownership versus scientific cohort composition.
- `docs/FORWARD_SEALED_CORPUS_CAPTURE.md`: sealed-corpus evidence contract.
- `docs/predictive_research_roadmap.md`: modelling hypotheses and frozen
  research protocols.
- `docs/thedogs_market_history_capture.md`: TheDogs temporal market evidence.
- `config/forward_baseline_capture_service.schema.json`: closed runtime config.
- `race_collection/service.py`: candidate preflight and baseline binding.
- `race_collection/forward_sealed_corpus.py`: exact cohort and evidence gates.
- `race_collection/forecasting.py`: prediction/result barrier.
- `race_collection/synchronous_manual_capture.py`: current-index publication and
  verification.
- `scripts/shadow_autopilot_daemon.py` and `scripts/shadow_autopilot_v1.py`:
  installed service, scheduling, lock, and collection orchestration.

The snapshot paths in `prediction_program_state.json` are evidence locators,
not permission to mutate their contents.
