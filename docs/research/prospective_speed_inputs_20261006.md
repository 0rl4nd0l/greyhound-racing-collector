# Prospective frozen speed input adapter

`race_collection.prospective_speed_inputs.forecast` converts an already verified,
sealed native prediction capture into three probabilities for the same complete
field: normalized market, June period-one development `base16`, and that exact
baseline with the frozen 0.1 speed adjustment. It does not acquire, fit, inspect
results, publish a pre-jump seal, modify the original bundle, or change a service.

The candidate remains exploratory. The retained production model is recorded
from the original prediction request and is explicitly distinct from the
comparison's historical development baseline. The development artifact is
`search_v1/period1_models.json`, SHA256
`df6bd595e1905bd67a1da9e5becab10f8b5f3db95707090079421201ccf6861e`, key `base16`.
The five frozen implementation source hashes are checked against candidate
commit `af55ae9322d08d0f255f767a14e385c794205dfe` before calculation.

## Interface and responsibility

```python
forecast(reader, member, original, model_reference,
         forecast_at=actual_start_time, prior_observations=authenticated_history)
```

The caller provides a bounded hash-checking reader. `member` has the existing
seven native retained-input references (`bundle_manifest`, `admission`,
`sidecar`, `accepted_csv`, `raw_export`, `primary_page`, `primary_receipt`) plus
`comparison_inputs`, `odds_receipt` and `request`. The extra references must map
to those exact paths inside the native bundle and match its file hashes. This
accepts dynamically allocated members and does not relax or mutate the original
82-card manifest loader.

`original` must come from the independently verified native publication ledger,
with its original admission, completion time, jump time, runner-set hash and
manifest references intact. Supplying an arbitrary alleged historical seal is
not authorization. Root's lifecycle worker owns allocation checks, authoritative
ledger verification, durable exclusive attempt claims, resource limits and
actual pre-jump output publication. The adapter's `forecast_at` argument does
not prove that publication occurred before the jump. Late output must be
accounted as a failure or replay, never a prospective forecast.

For new captures, `member_from_native` directly invokes the existing
`src.predictor.future_comparison.verify_comparison` after checking the allocated
native plan hash. It binds that verifier's completion record to the exact
admission/completion bytes and derives the dynamic member from the native bundle
manifest and source sidecar. It checks original source paths against explicitly
allowed roots. The verifier reads retained forecasts and pre-race inputs; its
native `result.json` is a prediction result, not an official race outcome. No
caller-supplied success flag replaces independent native verification.

Old and current production captures can have a different feature-generator
implementation from this research checkout. They must be verified with their
original package, rather than relaxing the generator hash check. The optional
`verifier_source_reference` identifies the hash-bound original package plan.
Its frozen comparison must match the admission plan and its prediction root
must match the native bundle root. The adapter checks the package's original
source identity and every mapped source file, plus its Python binary hash.
It then starts the original verifier in a separate read-only, network-denied
process with that source's own working directory and `PYTHONPATH`. Before import,
the child checks the complete source identity again; returned admission and
completion hashes must match the exact controls already read by the adapter.
A merely byte-compatible package from another window is not substituted.

The returned private record contains baseline feature values and accepted
historical rows, exact odds/source bindings, the complete speed input packet,
selected support and benchmark memberships, all three probabilities, model and
production identities, and neutral-support dispositions. Raw source files remain
hash-bound and unmodified. The adapter never opens a native result file.

## Timing and benchmark updates

The effective common information cutoff is the earlier of the actual forecast
start and the native scheduled decision cutoff. The native complete publication
must precede that cutoff, and forecast start must precede jump. Prices must be
from the same atomic retained receipt, inside the existing two-to-ten-minute
window before jump. Receipt, comparison inputs, native entry IDs, boxes and dog
tokens must all agree. No new price request is made.

The caller may extend the history inventory with newly authenticated retained
cards under the fixed plan. Original availability timestamps are retained. The
unchanged candidate filters copies not available before this target cutoff
before deduplication and conflict handling. No alias is inferred; same-row dog
profile identity is required for cross-card joins. Target-date history remains
excluded. Repeated copies cannot increase independent history support.

## Exact baseline semantics

The adapter reproduces `scripts/offline_form_packet.py`'s original raw-card
feature mapping, rather than using the production feature matrix:

- The original `accepted_history` parser filters dates, resolves ordering,
  deduplicates normalized history and retains at most 20 starts.
- Target venue and grade use the original canonical rules. Explicit target
  distance uses the original integer-metre parser, including an `m` suffix.
- The same six feature aliases and eight-decimal canonical formatting are used.
  `recent_finish_best_5` is the minimum valid finish among the latest five
  accepted starts, not an absent canonical field.
- All 16 original features are preserved. Missing values remain missing for the
  saved median/missingness preprocessing. The saved within-field centering,
  coefficients and bounded residual transformation are reused unchanged.

The old development cohort used earlier form captures. Prospective inputs reuse
the collector's fresh native capture; the form-capture age difference is a
documented population difference, not a changed feature definition.

Speed's own literal context handling remains separate from the baseline's
historical venue canonicalization. Missing speed history receives zero direct
adjustment; whole-field normalization can still change that runner's final
probability. A wholly unsupported field is an exact baseline copy.

## Validation

Twenty focused fabricated-input tests pass in a read-only, network-denied
process. They cover exact feature mapping and rounding, historical duplication
and target-date exclusion, missing history, neutral fields, partly supported
fields, future source exclusion, replay, source/roster identity, the frozen
baseline artifact, pre-jump timing, native verifier invocation, source path
scope, original verifier source binding and exact control-byte proof. These
tests establish input-contract behaviour, not a live prospective forecast.
Root must separately verify real
retained feature parity, independent source-cell reconstruction and measured
operating overhead before activation.

The adapter reports separate baseline feature/scoring and speed calculation
timings. Runtime total latency must also include history authentication and
native verification. Native source-package hashing and the original verifier's
reads occur outside the bounded adapter reader; its read counters must not be
described as total filesystem IO.
