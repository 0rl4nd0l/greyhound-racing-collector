# Historical first-sectional construction — 6 October 2026

This implements the fixed construction in the [task](../agent_tasks/historical_speed_features_20261006.md)
under the user's [working semantics](speed_feature_design_clarification_20261006.md):
`1 SEC` is the individual runner's first sectional. Source time units are assumed
to be seconds without conversion. This exploratory assumption is explicit; the
module does not authenticate a provider definition or change the older strict
measurement protocol. No empirical improvement or predictive advantage is claimed.

## Public interface and evidence responsibilities

`race_collection.historical_speed_features.build_speed_features(packet)` is pure:
no files, providers, databases, outcomes, training or mutable caller state. The
separate adapter authenticates the exact retained card, target identity, roster,
source hashes and capture receipt before supplying this packet:

| Packet member | Required content |
|---|---|
| `target` | `race_id`, ISO `date`, literal `source_track`, positive integer `distance_m`, timezone-aware `cutoff`; optional `layout_id` and `layout_era` |
| `captured_at` | Timezone-aware availability bound for the exact retained historical document, strictly before target cutoff; the adapter may use a verified capture timestamp or conservatively use its original completed seal time, declaring which |
| `roster` | Nonempty ordered list of unique, nonblank opaque runner identifiers |
| `histories` | Mapping of roster identifiers to observation lists; an absent roster member is explicitly unsupported; extra runner identifiers reject the packet |
| Each observation | `date`, literal `source_track`, integer `distance_m`, `first_sectional`; optional `layout_id`, `layout_era` and opaque SHA256 `observation_fingerprint` |

The adapter must bind the cutoff to the admitted target and ensure it is no
later than jump. The pure function checks capture versus that supplied cutoff;
it cannot authenticate either declaration. A hash-bound pre-jump capture proves
availability of those exact historical inputs by capture time. An original
publication timestamp and a provider reply are not prerequisites for this
separately scoped construction.

The target's source racing date cannot be later than the supplied cutoff's local
calendar date. An earlier meeting date remains permitted after midnight.
Historical dates cannot be later than capture converted to the cutoff timezone.
The adapter must therefore supply the source's racing-time offset with the cutoff,
rather than silently changing calendar semantics by converting it to UTC.

The schema rejects unexpected keys rather than accepting outcome or unrelated
feature fields. Shared target, roster, schema or timestamp failures raise
`FeatureRejected` with a fixed category and no echoed values. Row-level date,
context and measurement defects produce exclusion counts instead of silently
inventing data. Missing layout/era remains an explicit stability assumption.

## Fixed selection and calculation

For each roster runner, independently:

1. Require a valid ISO historical date strictly earlier than the target racing
   date. Same-day records are excluded because these inputs have day resolution.
2. Group prior observations by date before choosing a context or value. Equal
   projected observations count once; equivalent positive numeric spellings
   such as `5.8` and `5.80` are equal. Differing timing, context or layout metadata
   on one date excludes every interpretation of that date. No averaging or
   name-based reconciliation occurs.
   When supplied, the opaque observation fingerprint also participates in this
   comparison. The retained adapter supplies the existing timing-coverage
   whitelist digest, preserving conflicts in other retained timing/PIR fields
   without passing their scalar values into this module. A fingerprint is a
   duplicate-comparison token, not independent proof of source authenticity.
3. Match the exact source track string after whitespace trimming and exact
   integer distance. There is no track alias mapping, case folding, cross-track
   pooling or distance approximation. Invalid contexts remain excluded.
4. Exclude known layouts/eras differing from a known target. Require finite,
   positive individual first-sectional values; blanks and missing codes remain
   missing, and nonnumeric, nonpositive, boolean or nonfinite values are invalid.
5. Select the **most recent three usable, nonconflicting distinct prior dates**.
   Earlier usable dates may replace excluded missing/invalid/conflicting dates.
   There is no additional age threshold. If the selected set contains different
   known layouts/eras while the target is unknown, withhold that runner's
   construction; do not search for a more favorable alternate subset.
6. Return the median sectional and the **unscaled median absolute deviation**
   around that median. Preserve the three selected dates and their ages in days.
   Fewer than three usable dates yields no numeric summary.

Distinct dates are the conservative counting rule for this experiment, not
newly verified native historical event identities. Exclusions count observations:
an identical pair contributes one `DUPLICATE_OBSERVATION`, while a conflicting
pair contributes two `CONFLICTING_PRIOR_DATE` exclusions. `usable_prior_dates`
counts individually usable candidate dates before any selected-set layout
conflict; the runner status determines whether a three-start summary exists.

The output preserves every runner in roster order. `SUPPORTED` runners retain
their individual median/MAD even when another runner is unsupported. Other
statuses are `NO_RETAINED_HISTORY`, `INSUFFICIENT_COMPARABLE_HISTORY` and
`KNOWN_LAYOUT_CONFLICT`, accompanied by explicit exclusion counts.

Only when every roster runner has support and the selected known layouts/eras
are mutually compatible does the module return:

```text
field_median_gap = runner_median_first_sectional
                   - median(all roster runner medians)
```

Negative means a smaller time under the stated comparable-segment assumption.
The field center and **every** relative gap are null otherwise. `field_blocker`
distinguishes `INCOMPLETE_ROSTER_SUPPORT` from cross-runner
`KNOWN_LAYOUT_CONFLICT`; the latter preserves supported individual summaries.
An even-size field uses an overflow-safe arithmetic midpoint for its median.
No field rank, physical velocity, latent ability or model probability is inferred.

## Privacy, limitations and validation

The returned medians, MADs, gaps and selected dates/ages are private feature
output. The adapter owns private file permissions, whole-member accounting and
a separate values-free public summary. The pure function never prints values.
Known layout compatibility is enforced; unknown layout, sectional endpoint and
clock stability remain declared same-context assumptions, not certified facts.
There is no grade correction, external par, run-home, going adjustment, fitted
parameter or production activation in this version.

The focused suite uses fabricated source-shaped values only. It verifies exact
latest-three selection, retained individual support, whole-roster gap gating,
missing/invalid/nonfinite values, date/capture leakage, duplicate/conflicting
dates, literal context matching, known layout conflicts, input-order invariance
and preservation, and finite arithmetic. Tests ran with networking denied and
the filesystem read-only using the pinned Python, without repository conftests,
plugin autoload, bytecode or pytest cache:

```sh
bwrap --unshare-net --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp \
  --chdir /mnt/tenn-nvme2/tenn/greyhound-speed-features-20261006 \
  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv PYTEST_DISABLE_PLUGIN_AUTOLOAD 1 \
  /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  -B -m pytest --noconftest -c /dev/null -p no:cacheprovider -q \
  tests/test_historical_speed_features.py
```

No real retained timing rows were processed by this implementation agent.
The coordinator owns the separately bounded retained-input execution and any
coverage claim. These tests establish construction behavior only.
