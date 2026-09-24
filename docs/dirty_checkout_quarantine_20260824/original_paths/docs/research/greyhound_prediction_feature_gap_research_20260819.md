# Greyhound prediction feature-gap research — 2026-08-19

## Question and boundary

What genuinely pre-race information or modelling approach might add signal beyond
the current corrected Sportsbet WIN baseline and the feature families already
tested in this repository?

This is a research note, not a modelling or collection authorization. A source
showing that a field exists, or that it is associated with racing speed, does
not show that the field improves win probabilities beyond Sportsbet. Every item
below remains a hypothesis until it passes a frozen chronological market-residual
test and, separately, prospective validation.

## What is already covered

The recent frozen residual test already included last-five speed level, best,
trend and consistency; early pace; exact track-distance speed; rest days;
finishing form; box position; and within-field speed rank/gap. It used a
race-conditional correction to normalized Sportsbet probabilities. Coverage was
thin for several history features and the result was
`NO_INCREMENTAL_FORM_SPEED_SIGNAL`; see
[`scripts/run_form_speed_market_residual_experiment.py`](../../scripts/run_form_speed_market_residual_experiment.py)
and its [report](../../artifacts/form_speed_market_residual_20260818_report_only/report.json).

Pace topology has also been tested and closed `NO_PACE_TOPOLOGY_SIGNAL`; see its
[report](../../artifacts/pace_topology_mechanism_audit_20260818_report_only/report.json).
Betfair teacher work was historically promising but remains quarantined/report-only,
not proof that a live or prospective model improves on Sportsbet; see its
[report](../../artifacts/betfair_teacher_distillation_20260818_report_only/report.json).

The repository can represent trainer, sex, sire, dam, weight, PIR and comments,
but representation is not the same as proven point-in-time coverage or current
model use; see the local [database schema](../data_dictionary/database_schema.md),
[FastTrack field map](../fasttrack_field_map.md), and [on-demand model boundary](../on_demand_race_prediction.md).

## Best remaining feature candidates

### 1. Steward-derived health and compromised-run state — highest priority

Build strictly-prior, structured features such as days since injury or stand-down,
return from a failed/qualifying trial, prior fall or pull-up, severe-check burden,
repeated slow/quick starts, and stable rail/wide-running tendency. An official GRV
[stewards report](https://fasttrack.grv.org.au/RaceField/DownloadStewardsReport/1148738901)
demonstrates that declared-weight changes, start quality, running position,
injuries and stand-down periods can be published in one source. GRV defines a
“check” as interference that can cost momentum and time, and defines veterinary
stand-downs and trials in its official
[Racing 101 glossary](https://www.grv.org.au/racing/racing-101/).

**Evidence:** the data exists and a check can contaminate an observed time.
**Hypothesis:** a model can recover latent ability by distinguishing a genuinely
slow run from a compromised run. Missing report text must mean “unknown,” not
“healthy” or “clean run.” This is more novel than re-aggregating the same raw
finish and speed fields.

### 2. Workload, recovery and weight trajectory — highest priority

Replace the single `days_since_run` value with starts and metres raced over the
prior 7/14/28 days, minimum gap, long-layoff/comeback state, and interactions with
prior injury. Add last-known weight minus the dog's rolling baseline, robust
weight z-score and rate of change, with age/sex/distance interactions. Use only a
weight proven available before the target jump; if race-day kennelling weight is
not public in time, use the previous official weight.

A 2026 original study modelled 206,686 UK runs from 12,883 greyhounds and found
that inter-race interval was associated with racing speed, with a nonlinear and
small effect relative to individual variation
([DOI 10.1016/j.tvjl.2026.106555](https://doi.org/10.1016/j.tvjl.2026.106555)).
The same study modelled sex and bodyweight, supporting interactions rather than
one pooled linear weight coefficient. Official GRV pages publish runner weight,
split, PIR, time and margin in race results; this
[FastTrack race page](https://fasttrack.grv.org.au/RaceField/ViewRaces/1149854137?raceId=1271673303)
is an example.

**Evidence:** workload interval and bodyweight are associated with measured
speed. **Hypothesis:** richer state features add information beyond market odds.
The existing residual's `days_since_run` does not test this whole family.

### 3. Public trial and sparse-history lifecycle evidence — high priority for maidens

For low-start runners, collect public qualifying/performance-trial recency,
distance, split and overall time; first-start/low-start status; exact age from
whelp date; and uncertainty due to little racing history. GRV states that public
trial times are displayed and that trials obtain split and overall times as an
indication of performance in its official
[glossary](https://www.grv.org.au/racing/racing-101/). GRNSW states that
Performance Trials are recorded in the grading system in its official
[order-of-choice explanation](https://www.grnsw.com.au/order-of-choice-explained).

**Evidence:** these are pre-race performance observations that can exist before
a runner has normal race history. **Hypothesis:** they improve maiden or sparse-
history predictions. Private trials are unavailable by definition and must not
be inferred.

### 4. Explicit track/start geometry and same-meeting track variant — medium priority

Current venue, distance and box categories can memorize locations but do not
explicitly represent distance to the first bend, bend radius, straight versus
circle starts, transitions, or surface grade. A UTS/GRNSW primary engineering
report found that bend-limited speed is a function of radius, ground shear
strength and surface grade
([UTS track-design report](https://opus.lib.uts.edu.au/bitstream/10453/124253/1/2018%2005%2010%20UTS%20Phase%20II%20Progress%20Report.pdf)).
GRNSW's current track standards page confirms that racecourse design and
construction are regulated attributes
([GRNSW tracks and minimum standards](https://www.grnsw.com.au/racing/tracks)).

A separate as-of feature could estimate that meeting's track-speed variant from
official earlier races only, normalized by track/distance and runner ability.
Never use a later race from the same meeting.

**Evidence:** geometry and surface constrain running dynamics. **Hypothesis:**
explicit geometry or an as-of track variant improves cross-track normalization
or box interactions beyond venue categories.

### 5. Trainer changes and partially pooled kennel form — medium/low priority

Use stable trainer identity, trainer-change date, and strictly-prior shrunk
trainer × track/distance effects. GRV publishes official trainer identities and
rolling track-specific tables in
[Leading Trainers](https://fasttrack.grv.org.au/Statistics/LeadingTrainers).

**Evidence:** the identity and historical counts are available before a race.
**Hypothesis:** the trainer contributes incremental information. Raw strike rate
is strongly confounded by dog quality and stable size, so this should use partial
pooling and a trainer-change design, not an unregularized leaderboard rank. It is
lower priority because the repository has already collected some expert-form
trainer fields without establishing activation value.

### 6. Pedigree/litter effects — low priority, cold-start only

Use sire, dam or litter as partially pooled priors only where the dog itself has
little history. A primary study of 42,785 Irish races estimated moderate
heritability for racing time (0.31) and adjusted racing time (0.38), but very low
heritability for race ranking (0.10)
([PubMed record and abstract](https://pubmed.ncbi.nlm.nih.gov/17550352/)). GRV
publishes sire/dam statistics in
[Leading Sire/Dam](https://fasttrack.grv.org.au/Statistics/LeadingSireDam).

**Evidence:** pedigree explains some variation in time. **Hypothesis:** it adds
winner-probability signal beyond the market. Its low ranking heritability and
high leakage risk make it unsuitable as the first experiment; offspring and
family aggregates must be constructed strictly as of each race date.

## Modelling approaches that have not been cleanly answered

### Dynamic hierarchical latent-speed model

Estimate a time-varying ability state for each dog with partial pooling,
track-distance normalization, recovery/weight state and observation noise for
compromised runs. Convert the joint predictive time distributions to race win
probabilities, then test only the residual relative to Sportsbet. This is a more
coherent use of repeated observations than fixed last-five summaries. The Irish
study reported repeatability of 0.56 for racing time but only 0.13 for ranking,
supporting time as the intermediate target
([primary abstract](https://pubmed.ncbi.nlm.nih.gov/17550352/)). A Bayesian skill
rating is one implementation family; the original TrueSkill paper handles any
number of competitors while tracking uncertainty
([Microsoft Research paper](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/)).

**Evidence:** repeated race times contain stable individual variation and the
method represents evolving uncertain ability. **Hypothesis:** it beats the
current market-residual formulation. Start here before deep learning.

### Full-order race likelihood

A Plackett–Luce model fits a probability distribution over the field's finishing
order instead of treating runner outcomes independently; the original paper was
motivated in part by bookmaker place-odds problems
([Plackett, 1975](https://academic.oup.com/jrsssc/article/24/2/193/6953554)).
It could share information between win, place and order targets while respecting
race grouping.

**Evidence:** the likelihood is mathematically appropriate for ranked outcomes.
**Hypothesis:** it improves calibration here. It should not revive economic or
exotic claims already closed by prior tests without a new frozen protocol.

### Approaches to defer

- **Deep set/attention models over the whole field:** attractive for runner
  interactions, but roughly one thousand development races is too small to make
  this the next defensible step.
- **Pre-race stress sensors or video posture:** a 525-dog Australian pilot found
  performance associations with age and box, while pre-race eye temperature
  varied by track and race number; it did not establish a deployable pre-race
  winner signal ([primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341205/)).
- **More tuning of the same form/speed aggregates:** the current frozen gate
  failed, so trying many nearby transformations on the same outcomes would raise
  selection risk without adding an independent source.

## Recommended sequence

1. Audit point-in-time coverage for steward incidents, previous official weight,
   workload windows and public trials on the existing native-ID race population.
   Stop if whole-field coverage or temporal provenance is inadequate.
2. Freeze one compact **state** family: steward/health state + workload/recovery +
   weight trajectory. Evaluate it as a Sportsbet residual with chronological
   whole-race folds and no post-result feature construction.
3. If coverage is sufficient, compare the current residual with one dynamic
   hierarchical latent-time model. Predeclare log loss, calibration, fold
   consistency and uncertainty gates.
4. Evaluate trial/lifecycle features only on a predeclared maiden or sparse-
   history slice. Do not let the slice reopen a failed all-race feature search.
5. Leave trainer, pedigree, geometry and full-order modelling as later ablations;
   do not combine all candidates into one feature search.

No cited source establishes an edge over corrected Sportsbet WIN. The most
promising next move is therefore better point-in-time **state evidence**, not a
larger generic model.
