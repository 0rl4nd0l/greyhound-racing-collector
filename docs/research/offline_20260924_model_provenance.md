# Frozen model provenance and outcome-free form reconstruction

Inspected at source commit `1570dbadd1aca25deb249c41cf288101275b0b2b` for the
isolated September 24 development programme. No deployed configuration, model,
database, service, protected outcome, or existing experiment was changed.

## Actual frozen method

The model is a race-conditional logit with a **fixed** market offset, not a
general tree ensemble. For runner i, market probability is normalized inverse
decimal WIN odds. The scorer imputes 16 form features using training medians,
adds 16 missing indicators, scales using training mean/population standard
deviation, centers each feature within race, and computes
`a_i = 0.35*tanh(x_i beta / 0.35)`. Full/half probabilities are
`softmax(log(market_i) + strength*a_i)` for strengths 1 and 0.5.
Both use one shared coefficient vector. No box, early speed, sectionals, weather,
or raw time feature enters that frozen model.
Sources: [scorer](../../src/predictor/market_form_residual.py),
[model](../../artifacts/frozen_models/market_form_residual_v1/model.json).

The 16 features are `prior_start_count`, `days_since_last_start`,
`recent_finish_mean_3`, `recent_finish_best_5`, `recent_win_rate_5`,
`recent_place_rate_5`, `recent_avg_margin_5`, `career_win_rate`,
`career_place_rate`, `career_avg_finish`, `starts_same_venue`,
`win_rate_same_venue`, `starts_same_distance`, `win_rate_same_distance`,
`same_grade_start_count`, and `same_grade_win_rate`.

The original optimizer minimizes **mean race log loss + 0.5 * ||beta||²**;
the ridge penalty is not divided by the race count. It uses deterministic zero
initialization and L-BFGS-B with maxiter 500, maxls 50, ftol 1e-12, gtol 1e-8.
All-missing training columns receive median zero; scales below 1e-12 become 1.
Source: original evaluator `fit_preprocessor`, `conditional_loss_gradient`, and
`fit_residual` in
[evaluate_market_form_residual.py](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-prospective-market-form-residual-20260716/reports/agent_jobs/prospective_market_form_residual_challenger_20260716/evaluate_market_form_residual.py).

Frozen model SHA256 is
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`.
It was fit on 678 races / 4,752 runners, June 10–July 9, 2026, including dates
that would now be historical test blocks. Therefore testing that fitted artifact
on those dates is in-sample. A new historical comparison must retrain its method
using only earlier dates. The artifact's activation fields describe its original
freeze, not verified current runtime status; no service inspection was performed.
Sources: [manifest](../../artifacts/frozen_models/market_form_residual_v1/manifest.json),
[original evaluation sample](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-prospective-market-form-residual-20260716/reports/agent_jobs/prospective_market_form_residual_challenger_20260716/evaluation_results.json).

Original development validation was June 19–25, June 27–July 2, July 4–9;
training ended June 17, June 25, July 2, respectively, with one-day embargoes.
The original final disposition was `NO_CREDIBLE_CHALLENGER`, including insufficient
venue stability. Those metrics are prior development evidence, not fresh results.
Sources: [candidate definition](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-prospective-market-form-residual-20260716/reports/agent_jobs/prospective_market_form_residual_challenger_20260716/candidate_definition.json),
[selection decision](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-prospective-market-form-residual-20260716/reports/agent_jobs/prospective_market_form_residual_challenger_20260716/selection_decision.json).

## Historical population limitations

All 678 retained races had already been inspected across 54 candidates; the
historical reconciliation declares zero untouched holdout races. Its 69
otherwise-aligned training/Sportsbet races did not qualify as additional Tier B:
the earlier model features consumed later-materialized TheDogs OPEN/LOW/HIGH
without per-race pre-jump timestamps. Four additional races had prediction or
feature freezes after jump. Do not import these old probabilities as honest
baseline predictions or use published OPEN/LOW/HIGH as decision-time prices.
Source: [historical WIN eligibility manifest](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-historical-win-eligibility-20260715/reports/agent_jobs/historical_win_eligibility_reconciliation_20260715/historical_win_eligibility_manifest_v1.json).

The strongest located form source is the raw-card acquisition development
population: May 27–July 9, 917 races / 6,456 runners across 42 source venue
labels; 573 races have official Tier-A label provenance and 344 have separate
published-history provenance. Cards were captured 60.0667–3223.4167 minutes
before jump. Label values are absent from these acquisition files.
Source: hash-bound
[development_races.csv](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-form-only-v1-acquisition-20260718/reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_races.csv).

The old feature CSV SHA256 `195bc517...b568` does not match the current tracked
v4 descriptor's `3dee15bb...e284`. It must not be silently presented as the
current canonical packet. Likewise the old README's claim of zero unexplained
legacy builder disagreements is superseded: current v4 diagnostic metadata
records 259 unexplained differences across the 530-runner overlap, with 486
history-count and 527 recency differences. These diagnostics are explicitly
non-authoritative; they do not invalidate a separately reconstructed raw-card
feature population. Sources: [v4 descriptor](../form_only_v1_reproducibility.json),
[canonical semantics](../form_only_v1_acquisition.md).

## Executed raw reconstruction

[offline_form_packet.py](../../scripts/offline_form_packet.py) pins original
acquisition metadata hashes and the current canonical parser source hash,
verifies each card and sidecar, and rebuilds features directly from retained
raw bytes. It reads no target labels. It excludes supplied protected identities
before card reads and returns the reserved 88-race out-of-time identity list
from an outcome-free identity CSV. Parent analyses must additionally apply all
other protected/reserved manifests before joining outcomes.

The first execution took 2.30 seconds and retained **868 races / 6,119 runners**.
It excluded 49 races whose original selected runner roster differed from the
complete card/sidecar roster; this conservative research loader does not infer
scratch or reserve removals. No target-or-later history row was accepted or
encountered in the retained cards. Current canonical parsing uses histories
strictly before the target date, deterministic ordering, deduplication, cap 20,
recent windows 3/5 and explicit missingness. The extra recent-versus-long
alternative uses a fixed three-start half-life weighted mean finish/margin.
This reproduces the frozen *model method* with rebuilt as-of features, not the
older frozen artifact's potentially inconsistent historical feature values.

An initial check exposed a canonical parser limitation: target distances such
as `400m` were passed to a numeric-only parser, making every target distance
appear missing. All 917 source sidecars actually carry explicit metre distances.
The isolated loader now parses exact positive integers with optional `m` suffix;
it neither modifies the production parser nor infers distance from past starts.
After this repair all 6,119 retained runners have target distance; same-distance
win rate is missing for 1,392 because no matching prior history exists.
Same-venue win rate is missing for 2,457 and same-grade win rate for 1,970.
Target grade comes from the hash-bound pre-race sidecar and current canonical
grade aliases, rather than the frozen model's older materialized feature values.
Early-speed/sectional and time-comparability work is deferred.

Validation: `python3 -m tests.test_offline_form_packet` passed four tests covering
protected-identity exclusion before raw reads, removal of target/future history
before weighting, rejection of modified source bytes, and explicit metre-suffix
parsing without guessing. Standard-library
execution was used because the inspected Python environments lacked pytest.
Full feature loading also completed successfully. Detailed included identities,
source hashes, missingness and exclusions are returned to the parent runner as
audit metadata, which should be saved with its experiment outputs.

## Bounded next-signal coverage audit

Executed [offline_signal_coverage.py](../../scripts/offline_signal_coverage.py)
on the exact prepared 332-race / 2,367-runner intersection. It projected only
race and runner identity from prepared rows, then inspected hash-bound pre-race
cards; target outcomes and comparative scores were not decoded. The retained
[coverage artifact](/home/l4nd0/greyhound-offline-research-output-20260924/signal_coverage.json)
binds prepared SHA256 `8049f6efb8dee4785f5f35ba6898884ad3b5c406b7755b92122f2f8aefd823be`.
Counts below concern 11,487 prior-start observations repeated across target
runner cards, not 11,487 distinct historical starts.

| Potential signal | Runner targets with any history | Runner targets with at least 3 | Full fields with any / at least 3 |
|---|---:|---:|---:|
| Multi-digit PIR containing only positions 1–8 | 1,102 / 2,367 | 906 / 2,367 | 124 / 332; 88 / 332 |
| Numeric first sectional (`1 SEC`), any track/distance | 2,150 / 2,367 | 1,670 / 2,367 | 247 / 332; 119 / 332 |
| Positive first sectional, same venue and distance | 878 / 2,367 | 358 / 2,367 | 31 / 332; 7 / 332 |
| Positive race time, same venue and distance | 1,246 / 2,367 | 639 / 2,367 | 60 / 332; 11 / 332 |
| Numeric margin | 2,367 / 2,367 | 2,293 / 2,367 | 332 / 332; 293 / 332 |

PIR is populated in every prior-start row, but **all 6,920 single-digit PIR
values equal that historical finish position**. Treating these values as an
independent early-position signal would manufacture information. The other
4,567 rows have multiple digits (4,563 use only digits 1–8); the retained local
documentation does not establish a consistent call-position convention.
[schema_diff_fasttrack.md](../schema_diff_fasttrack.md) calls PIR a Position In
Running code, while [fasttrack_field_map.md](../fasttrack_field_map.md) contains
both Performance Index Rating and points-in-running descriptions. These are
not sufficient evidence that the first digit represents a comparable first
call across these venues. A source-defined multi-digit-only parser is a plausible
future experiment, but direct whole-field early-speed pressure is not currently
supported with reliable semantics and broad complete-field coverage.

Margins and finishing positions each have at least three numeric prior values
for 2,293 runners. Testing margin improvement conditional on finishing position
is therefore supported by broad raw-field coverage. Track/distance-normalized
time or sectional signals are also present, with substantially sparser matching
history. Their incremental value has not been tested by this coverage audit;
no feature or model selection was made from their target outcomes.
