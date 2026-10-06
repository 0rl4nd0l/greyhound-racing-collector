# Speed feature design following user clarification — 6 October 2026

The user explicitly clarified that TheDogs `1 SEC` is the listed runner's first
sectional and `TIME` is that runner's total time, using historical races only.
These are the **working semantics for this design**, supplied by the user; they
are not newly authenticated provider definitions. Accepting them permits useful
same-context feature design and synthetic testing now. A comprehensive provider
enquiry is not a universal prerequisite for that work.

The earlier [definition enquiry](thedogs_definition_enquiry_20261006.md) remains
**unsent and is superseded/narrowed by this clarification**. Its broad requests
are no longer the starting gate for this design. This new note records the change;
the enquiry, prior audits and frozen implementation remain unchanged. The
[retained measurement contract](/mnt/tenn-nvme2/tenn/greyhound-speed-contract-20261006/docs/research/speed_measurement_contract_20261006.md)
still identifies provenance and comparability questions. Its unqualified
provider-evidence disposition must not be mistaken for a ban on designing a
bounded experiment under the user's stated assumptions.

## Same-context time features and pars

Start with comparisons within the same track, race distance, physical layout
and applicable layout era. For a sectional, also match the start trigger,
sectional endpoint and clock convention. Under the working semantics, recent
historical `1 SEC` and `TIME` summaries can describe a runner's early and total
performance in that context. Keep missing observations explicit and deduplicate
repeated observations of the same historical participation.

A past-only robust par is a useful candidate benchmark: for example, a median
time within the declared context, with a robust spread such as median absolute
deviation or interquartile range. For an eligible observation:

```text
residual_seconds = observed_time - context_par
normalized_residual = residual_seconds / context_spread
```

Negative values mean faster than the defined par under this convention. If a
larger-is-faster score is desired, negate the normalized residual explicitly.
Prespecify minimum support, the spread estimator, any pooling/shrinkage and the
handling of zero or unstable spread. Sparse or incompatible groups should be
withheld or follow that fixed fallback, not receive an invented comparison.

A smaller raw residual in seconds across different sectional lengths or
contexts does not establish better ability: clock length, field composition and
natural spread differ. For example, `-0.15 s` versus `-0.10 s` on different
sectional lengths/distributions does not prove the first runner is the faster
breaker. A past-only percentile or standardized residual may improve
comparability, but remains performance relative to its benchmark population.
Cross-track equivalence, calibration and grade/population effects remain
hypotheses to test; normalization is not proof of superior underlying ability.

As an independent example, [GRV's Race Data explanation](https://www.grv.org.au/racedata/)
describes track-shape effects even for time-to-distance comparisons, differences
between beam and Isolynx timing, and slower finishing segments in longer races.
Root checked this primary explanatory source on October 6. It supports the need
for context-specific comparisons; it does not define TheDogs fields.

Track/grade pars are a testable extension. Grade can change the expected field
strength and spread, while grade labels can have different meanings across
jurisdictions. Control grade explicitly within its defined scope and compare
the incremental effect of grade adjustment against a track/distance/layout
benchmark. Automatically subtracting a grade par may remove useful ability
information or introduce sparse-group noise. Freeze the alternatives before
evaluating them rather than selecting a grade treatment from target outcomes.

## Total minus first sectional

Under the working semantics, a candidate remaining-time feature is:

```text
remaining_time = TIME - 1 SEC
```

This subtraction has a physical interpretation only when both values belong to
the same dog in the same historical race and source revision, use compatible
units, and share the same clock origin. The sectional endpoint must lie within
the total timed course. The difference then describes elapsed time from that
endpoint to the finish. Reject inconsistent records such as a first sectional
greater than total time; preserve missingness rather than manufacturing a value.

“Run-home” is a convenient feature name only after the actual endpoint is
established; the source may define a different run-home segment. Remaining time
cannot be compared directly across different remaining distances, course shapes
or layouts. A context-specific remaining-time par can be considered under the
same past-only rules. The residual includes the runner's condition and position
at the split and the subsequent racing circumstances; it does not isolate or
prove latent finishing power.

## Going corrections

Use going only if the source actually supplies an applicable observation or
allowance. Do not fabricate a going value from the presence of timing fields.
Retain its unit, sign convention, publication/revision time and exact scope:
track, layout, meeting or race, distance and timed segment as applicable.

If the documented convention says an allowance represents seconds added by
slower conditions, subtracting it may be appropriate. That example is conditional
on the source's convention; do not assume its sign. A qualitative going label
is not automatically a numeric time allowance. A whole-race allowance must not
be subtracted from the first sectional. Nor should it be allocated to sections
in proportion to distance without a supported segment model. Compare raw times
with any valid adjusted alternative under a frozen experiment design.

## Availability and experiment scope

Prior-race timing values are allowed inputs. For each target prediction use the
historical values and revisions actually retained before its pre-jump cutoff.
Do not substitute a later corrected value into an earlier frozen prediction.
Event occurrence and source availability are separate facts.

An exact retained pre-jump capture proves that its contents were available by
that capture time. For this scoped use of those captured historical inputs,
there is no universal need to recover an unavailable original publication
timestamp. Bind the actual document bytes to their receipt and cutoff, record
the user's working meanings and the remaining comparability assumptions, and
proceed with the practical design. The older stricter publication-metadata
protocol remains unchanged; its additional requirement does not become an
automatic gate on every separately scoped research design.

Every par, spread, grade adjustment or other fitted benchmark must also use only
information available before that target's cutoff. Apply this separately at
each historical evaluation cutoff; a benchmark computed once from the full
evaluation period would leak future information. The target race's eventual
times, finish, going correction published afterwards, or other outcomes must
not enter its own features or benchmarks. No current/future target outcomes are
needed for the design or synthetic checks described here.

A practical scoped sequence is: define the same-context raw-time baseline;
specify past-only robust pars and missingness rules; test the total-minus-split
construction under its clock/endpoint assumptions; then test grade and valid
going adjustments as separate additions. Preserve the existing proposed
three-comparable-prior-starts requirement and complete target-roster accounting
for its original hypothesis. Different coverage or pooling rules would be a
separate prespecified variant, not a silent relaxation based on results.

## Narrow remaining source questions

1. **Sectional endpoint and layout:** For the chosen track/distance/layout era,
   where does the first sectional start and finish, and do its clock origin and
   units agree with TIME? This governs comparable early segments and the meaning
   of the remaining-time subtraction.
2. **Measured or derived TIME:** Is the individual total measured directly or
   computed from winner time and beaten margin? If derived, what conversion and
   rounding apply? This affects precision and dependence between candidate
   features; it does not invalidate the user's definition of whose time it is.
3. **Going availability and applicability:** Does the source supply going or a
   timing allowance for the selected historical context, with what unit, sign,
   segment scope and availability? If absent or incompatible, omit that feature.

These questions narrow the next evidence check; no provider contact or website
access was performed for this note. No retained values, protected outcomes,
models, frozen code or services were changed.
