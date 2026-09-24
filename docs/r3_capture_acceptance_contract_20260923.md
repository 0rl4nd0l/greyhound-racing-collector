# Minimum useful capture contract for R3

This is an offline, source-derived acceptance contract. It changes no capture
schedule, population, market baseline, feature calculation or activation. The
frozen residual prediction needs **one complete, timely, authenticated market
observation plus its exact reproducible feature inputs**. The current R3 adapter
also requires the matching PLACE field. Four snapshots are not a frozen-model
requirement; a fresh index alone is not a usable race.

Source baseline is integration commit
`89d22067a737222a053e73e02726ba352eec08e3`. The separately inspected collector
candidate remains `fe3d984a13a7446ca0af301dd736fed20bab0b96`; its execution/package
authority is separate from this worktree and the installed R3 release. This
document reports source requirements, not installed configuration or live
coverage. Only source, configuration definitions and named documentation were
read; no real race histories, target outcomes or protected study payloads were
opened.

## Required inputs and identity

| Layer | Exact minimum and rejection boundary | Source owner |
| --- | --- | --- |
| Discovery | Verified collector-owned current index, source refresh/report and publication lifecycle; exact TheDogs URL, date, venue, race number, aware jump timestamp, unambiguous aliases and runner identity. Dashboard upcoming policy labels source age over 300 seconds stale; baseline bootstrap/CLI index validation uses the frozen config's 1,200 seconds. Operational acceptance here requires at most 300 seconds. | [`_race_snapshot` implementation](../src/operator_ui/live_adapters.py), [bootstrap](../src/operator_ui/bootstrap.py), [index contract](on_demand_race_prediction.md) |
| Race and runners | Same exact race, jump and canonical box/name runner-set hash in index, admission, form/sidecar, receipt, features and output. Native source box evidence is required; page/list order alone is insufficient. No missing, extra, duplicated or conflicting active runners; scratches must be explicit and consistent with the admitted field. At least two runners. | [`discover_exact_receipt_ready`](../src/predictor/receipt_preflight.py), [`_active_capture_rows` and `_feature_packet`](../scripts/predict_market_form_residual.py), [capture validator](../scripts/autonomous_live_odds_capture.py) |
| Market | Sportsbet source URL agreeing with exact venue/race; validated `APPENDED` attempt, finite decimal prices greater than 1, fetch/append times, accepted runner rows and receipt/report/form/sidecar hashes. The residual uses normalized inverse **WIN** prices. Current `normalize_validation_receipt` additionally requires complete `accepted_place_rows` matching WIN identities; PLACE is an existing adapter requirement, not one of the model's features. | [`receipt_from_handoff` / `normalize_validation_receipt`](../src/predictor/on_demand.py), [`FEATURES` / frozen market offset](../src/predictor/market_form_residual.py) |
| Target form | Exact form CSV and adjacent sidecar; safe target distance and normalized grade; exact TheDogs race and meeting-card proof, matching date/venue/race/runner set and byte hashes. Neither cached grade nor a race-title guess substitutes for proof. | [`_sidecar_context` / `_feature_packet`](../scripts/predict_market_form_residual.py), [manual input contract](manual_live_market_form_residual_prediction.md) |
| History and features | Exact sealed database/history projection and raw/normalized forms, primary source bytes and receipts; all 16 feature keys per runner with their explicit values or nulls. History must precede target race date; same-day, target and future rows are excluded. Safe counters must show no target/post-outcome rows used. A missing key is invalid; explicit null follows frozen missingness handling. | [`build_live_feature_rows`](../scripts/run_shadow_non_tgr_rf_evaluation.py), [`_feature_packet`](../scripts/predict_market_form_residual.py), [retention contract](prospective_input_retention_preparation.md) |
| Reproducibility | Hash-bound generator source/dependency closure, schema, model/manifest, canonical config, environment lock and replay worker; exact WIN receipt and observations; feature/input seal and parent retention acceptance before configured cutoff. A later mutable DB read does not prove consumption of the retained input. | [retention implementation](../race_collection/prospective_input_retention.py), [scheduled parent](../race_collection/scheduled_input_retention.py), [retention preparation](prospective_input_retention_preparation.md) |
| Admission and closure | Active bounded R3 authority, pinned model/config, no previous race claim, receipt preflight, result-acquisition readiness, prediction dispatch, sealed-output verification, and subsequent matching official-result closure. These are additional requirements beyond collection. | [journal](../src/operator_ui/journal.py), [readiness](../src/operator_ui/journal_readiness.py), [worker](../src/operator_ui/prediction_worker.py), [results](../src/operator_ui/journal_results.py) |

The capture validator keeps WIN and PLACE in separate source fields. The
overround protocol below has its own stronger explicit raw paired-column WIN
evidence requirement; a generic historical `market_type=win` label does not
satisfy that protocol. No source class or market baseline is substituted here.

## The 16 frozen features

Order below is the exact [`FEATURES` tuple](../src/predictor/market_form_residual.py).
Values are computed by the existing
[`add_history_features`](../scripts/run_feature_recovery_execution_v1.py) over the
approved prior history assembled by `build_live_feature_rows`. This is a mapping,
not an alternative implementation.

| Feature | Required underlying information |
| --- | --- |
| `prior_start_count` | Number of accepted prior starts |
| `days_since_last_start` | Last accepted historical date and target date |
| `recent_finish_mean_3` | Finish values of last three accepted starts |
| `recent_finish_best_5` | Minimum finish value among last five |
| `recent_win_rate_5` | Last five starts; finish equal to 1 |
| `recent_place_rate_5` | Last five starts; finish in first three |
| `recent_avg_margin_5` | Available margins among last five |
| `career_win_rate` | Accepted prior starts and win indicators |
| `career_place_rate` | Accepted prior starts and first-three indicators |
| `career_avg_finish` | Available finish values in accepted prior history |
| `starts_same_venue` | Historical venue matching target venue |
| `win_rate_same_venue` | Same-venue starts and win indicators |
| `starts_same_distance` | Historical distance within **50 metres** of target distance |
| `win_rate_same_distance` | That same-distance subset and win indicators |
| `same_grade_start_count` | Historical normalized grade equal to safe target grade |
| `same_grade_win_rate` | That same-grade subset and win indicators |

All features preserve existing missingness, median imputation, missing indicators,
scaling and frozen coefficients. A source availability failure must not be
relabelled as a legitimate null or zero. Weather, track condition, sectional
speed, and successive market movements are not among these 16 inputs. Their
absence from this feature list does **not** waive source-side metadata safety
checks or another protocol's requirements. Byte replay proves reproducibility,
not completeness of the upstream historical record.

## Observation times and windows

The exact checked-in [`manual-default.json`](../configs/prediction/manual-default.json)
keeps receipt age at most **900 seconds**. Current R3 receipt preflight requires
strictly more than **53 seconds** to jump: validation 8 + scoring 30 + safety 15.
It rechecks age and margin after receipt validation, before the one-shot claim.
The standalone capture budget is distinct: lock 1 + capture 60 + validation 8 +
scoring 30 + safety 15 = 114 seconds, plus discovery 12 for total 126. R3 journal
selection is receipt-only; none of these numbers authorizes additional capture.
See [budget implementation](../race_collection/synchronous_manual_capture.py),
[preflight](../src/predictor/receipt_preflight.py) and
[worker](../src/operator_ui/prediction_worker.py).

Receipt freshness is measured from the authenticated append timestamp. Preserve
the actual fetch/source observation too: append time is not permission to
reinterpret an old observation. Scorer timelines require metadata <= feature
freeze <= feature generation <= scoring < jump, and metadata <= odds fetch <=
append <= scoring < jump. Retention additionally requires source observation <=
retention, data copy/seal and parent acceptance strictly before its configured
`jump - cutoff_seconds_before_jump`; the cutoff must precede jump. No universal
T-2 feature cutoff is specified by these components. The selected retention
configuration must state it explicitly. Sources:
[`score_from_artifacts` timeline](../scripts/predict_market_form_residual.py),
[scheduled retention](../race_collection/scheduled_input_retention.py),
[retention timing](prospective_input_retention_preparation.md).

Native planner intervals and existing receipt-validation intervals differ:

| Label | Planner can start when time is in | Timestamp accepted for that window |
| --- | --- | --- |
| T-60 | [jump-60m, jump-30m) | [jump-63m, jump-30m) |
| T-30 | [jump-30m, jump-10m) | [jump-33m, jump-10m) |
| T-10 | [jump-10m, jump-2m) | [jump-13m, jump-2m) |
| T-2 | [jump-2m, jump) | [jump-5m, jump) |

These follow `due_capture_window`, `capture_window_bounds` and
`capture_timestamp_in_window` in the
[existing collector](../scripts/autonomous_live_odds_capture.py). The 180-second
validation tolerance does not start the planner early. A T-2 receipt can be too
late for R3's 53-second margin; a T-30 receipt can become too old for the
900-second age limit. Label alone establishes neither eligibility nor readiness.
Do not credit a late completion to its original window merely because its append
is relabelled into a later one.

## Which consumer needs which snapshots

| Consumer / protocol | Required captures | Boundary |
| --- | --- | --- |
| Frozen residual generation through receipt-only R3 | One complete authenticated observation with matching WIN/PLACE rows, safe forms and sealed 16-feature inputs, still inside receipt-age and pre-jump budgets | No requirement for all four labels, and no requirement to wait until T-2. One odds snapshot **alone** is insufficient. |
| Reproducible retained-input prediction | The exact observation matched to the retained form/history/feature seal and parent acceptance before configured cutoff | A later fresh receipt cannot silently replace the retained receipt; changing history after sealing cannot change the consumer input. |
| Forward overround successor | Exactly the source-explicit WIN T-30 observation, accepted T-33 inclusive to T-10 exclusive; complete field and raw source receipt | [`forward_overround_successor_protocol.md`](forward_overround_successor_protocol.md): proposed 1,000-member protocol, `PREPARED_NOT_AUTHORIZED`, October start and future activation. T-60/T-10/T-2 are not required by this protocol and cannot substitute for T-30. |
| Sportsbet/Betfair consensus | Frozen corrected Sportsbet baseline plus outcome-free Betfair `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` projection, exact market/selection/runner identities and full-interval completeness receipts | [`sportsbet_betfair_forward_consensus_protocol.md`](sportsbet_betfair_forward_consensus_protocol.md): protected August 20–September 30 interval, blocked source boundary. None of the collector's four labels is a substitute for Betfair scheduled-off evidence. No new capture or cohort work follows here. |
| Market movement / later market comparison | A separately declared comparison must define its paired observations, baseline, cutoff and complete runner alignment | [`collect_shadow_odds_snapshots.py`](../scripts/collect_shadow_odds_snapshots.py) recognizes four capture modes, but recognition is not a scientific requirement to collect all four. No active all-four R3 requirement was found in the routed model/validator/protocol sources. |

Missing optional comparison windows cannot retrospectively invalidate or modify
an otherwise valid R3 prediction. They do make that race unavailable to any
separately frozen comparison that requires them. Missing required evidence,
late capture/seal, partial field or identity conflict rejects the candidate for
that consumer with an explicit reason; it is not backfilled into membership.
Official result delay is a separate pending closure state. Ambiguous/conflicting
results remain blocked, and neither case justifies deleting the admitted race
from its denominator. See [result adapter](../src/operator_ui/journal_results.py)
and the individual frozen protocols above.

## Operational acceptance denominators

This defines reporting requirements, not a new scheduler or a target pass rate.
Before a future authorized observation, freeze the observation interval,
calendar/discovery scope, consumer and required window policy. Report:

1. **Race opportunities:** all calendar races in the declared scope whose
   required opportunity intersects the interval, including races omitted by
   publication selection caps. Separate wholly observed opportunities from
   left/right-censored ones. Report source-unknown coverage as unknown, not zero.
2. **Eligible races:** opportunities meeting the declared population, source
   identity and timing rules, before assessing capture success. Account for
   each exclusion by reason; never define eligibility as “has a receipt.”
3. **Required windows:** sum of required opportunities per eligible race for
   the selected consumer. The existing four-window collector diagnostic has
   denominator 4N only when all four windows are observable. Residual generation
   instead has one qualifying-observation requirement per race; report attempted
   native windows separately, without treating any single label as compulsory.
4. **Attempts and valid receipts:** unique consumed race/window attempts, valid
   authenticated receipts, invalid/partial/late attempts, seen-unattempted and
   unseen opportunities. A request or odds append is not a valid receipt. Report
   `valid required observations / required observations` and
   `races with all required observations / eligible races` separately.
5. **Reproducible usable races:** eligible races simultaneously satisfying fresh
   index, exact current runner identity, timely receipt, sealed retained inputs,
   admitted generator/model/config and result-readiness configuration. Report
   `usable races / eligible races`; separately count authorized admissions,
   attempts, verified sealed predictions, pending results, verified closures and
   permanent failures. Disabled authority is a configuration/activation state,
   not fabricated input availability.
6. **Time with usable inputs:** for each eligible race, the measure of time
   within its declared admission opportunity during which all input conditions
   hold, divided by that whole opportunity duration. Also report wall-clock
   `time with at least one usable race / entire observation duration`. Include
   startup, absent/empty index and stale intervals; record sampling cadence and
   do not infer continuous availability between sparse observations. Separate
   absence of calendar opportunities from failed input availability.

Source loss, capture allowance exhaustion, age expiry, claim consumption,
activation expiry and result delay need different reasons. None may be silently
removed from the relevant opportunity, attempt or admission denominator.

## Capacity evidence and smallest outstanding decision

The existing [synthetic capacity study](/home/l4nd0/greyhound-capacity-study-20260923/docs/freshness_capacity_study_20260923.md)
uses candidate `54aeaa17b9dc3c3f7b0d1416fd9f15dd40cf1405`, not this integration or
PR #184 head. Its eight-simultaneous-race example completes 13/32 original
windows; the faster-refresh sensitivity completes 17/32. These are synthetic
scheduling completions, not native appends, verified receipts, retained bundles
or R3 admissions. Its one-attempt control and publication caps also prevent
coverage inference from a freshness pass. The study does not establish measured
live throughput, latency percentiles or a service acceptance percentage.

Concrete proposal for the next **separately authorized R3 integration acceptance**:
one explicitly eligible future race, one authenticated retained receipt and
complete input seal, one receipt-only claim, one verified output, and eventual
official closure, with every stage and exclusion recorded. Do not require all
four windows for this residual integration proof. Keep the prepared collector
rehearsal and all frozen research protocols unchanged. A recurring service still
needs an owner-selected race scope and minimum race/time coverage target; this
source review cannot invent one. Retention cutoff/history authority, exact
deployment bindings, result acquisition and journal activation also remain
explicit prerequisites, not questions that block the present offline work.
