# Forward overround successor protocol

Status: `PREPARED_NOT_AUTHORIZED`. This document and the companion JSON are a
protocol proposal and replayable state model only. They do not authorize or
start collection, install a unit, create a cohort, score a race, or mutate the
canonical database.

## Purpose and predecessor boundary

The successor asks the same predictive question as V2 with the same frozen
`OVERROUND_STRUCTURE_TRANSFORM_V1_20260816` model bytes. It changes no model,
preprocessing, scorer contract, population hypothesis, or metric. It repairs
one protocol-mechanics defect: V2 converted any observed installed capture-code
drift into an immutable experiment-fatal sentinel, even when the drift was
observed before an invalid prediction was written and the installed hashes were
later restored.

V2 remains `BLOCKED_FORWARD_EVIDENCE` with zero metrics. Its nine predictions,
six results, exclusions, protocol, and block are never eligible for this
successor. Known V2 outcomes informed only the control-flow repair described
here; they did not select a model, parameter, cutoff, sample size, metric, or
confirmation threshold.

## Frozen population and membership

- Target: exactly 1,000 eligible races. There is no analysis-driven stopping
  rule and no administrative collection deadline.
- Start boundary: a race jump must be at or after
  `2026-10-01T00:00:00+10:00` and strictly after a future immutable activation
  receipt, and activation itself cannot precede that boundary. This excludes
  the current-master Sportsbet/Betfair confirmatory window of 2026-08-18
  through 2026-09-30 inclusive, so its races cannot enter this successor.
  Capture must also be at or after activation. August 2026 is excluded.
- Selection: the first 1,000 eligible races ordered by jump time, capture time,
  then exact race ID. Membership becomes irrevocable when the write-once
  prediction receipt is sealed. A member cannot be removed, replaced, or
  reused in another confirmatory cohort.
- Identity: exact source race ID plus box number plus exact dog name; name-only
  alignment is prohibited. The complete active runner set must agree across
  race metadata, odds, prediction, and official result evidence.
- Odds: source-explicit Sportsbet fixed decimal WIN prices proven from the raw
  paired-column row. Legacy generic `market_type=win` rows are not sufficient.
  The accepted `autonomous_prejump_t30m` capture interval is from T-33 minutes
  inclusive to T-10 minutes exclusive. Every raw price and source receipt is
  retained.
- Candidate failure before sealing does not create membership. Incomplete
  field, source unavailability, out-of-window timing, identity ambiguity, or
  WIN-provenance ambiguity is recorded as a rejected candidate and may not be
  repaired retrospectively into a member.

## Frozen comparison

For every sealed runner with Sportsbet decimal WIN odds `o_i`, the baseline is
`(1/o_i) / sum_j(1/o_j)` within the complete race field. The candidate is the
unchanged frozen linear overround-allocation transform using the model,
preprocessing, protocol, scorer contract, and `SHA256SUMS` hashes recorded in
the JSON protocol.

The primary statistic is paired race-level mean multiclass log loss,
candidate minus proportional baseline, on the identical 1,000 races. Secondary
reporting is multiclass Brier, fixed-band runner calibration and ECE, top-1
accuracy, mean winner rank, and mean reciprocal winner rank. ROI is not
computed and cannot affect membership, model, finalization, or verdict.

No interim candidate loss, baseline loss, paired delta, calibration result, or
ranking result may be computed or exposed. Operational status may show only
counts, provenance integrity, admission state, and result-closure state.

## Uncertainty and confirmation

Finalization uses 20,000 paired race bootstrap replicates with seed `20260817`
and 20,000 Australia/Melbourne race-date cluster bootstrap replicates with seed
`20260818`, both with 95% percentile intervals. It also partitions the ordered
cohort into five deterministic equal-count chronological blocks.

`FORWARD_OVERROUND_SIGNAL_CONFIRMED` requires all of:

1. candidate-minus-baseline mean log-loss delta below zero;
2. paired race-bootstrap 95% upper bound below zero;
3. race-date-cluster bootstrap 95% upper bound below zero; and
4. negative paired log-loss delta in at least four of five chronological
   blocks.

A valid complete cohort that misses any gate returns
`FORWARD_OVERROUND_SIGNAL_NOT_CONFIRMED`. Evidence-integrity failure or an
explicit operator abort returns `BLOCKED_FORWARD_EVIDENCE` with no metrics.

## Runtime admission correction

Activation must bind the protocol hash, frozen model/scorer hashes, finalizer
code hash, capture-code hash set, unit hash, and the first admission event. Each
prediction receipt binds the active admission ID and its content hash.

An observed capture-code or unit hash not in the append-only admission chain is
a temporary pre-seal admission failure:

1. halt sealing before any prediction write;
2. append a timestamped `ADMISSION_CHECK_FAILED` event containing no candidate
   outcome;
3. review the new runtime and prove the model, scorer contract, protocol
   semantics, population, cutoff, and evidence validators are unchanged;
4. append a new `ADMISSION_ACCEPTED` event whose hash chain names the previous
   admission and whose timestamp precedes every seal under the new runtime; and
5. resume with future candidates only.

The finalizer validates every member against the admission that preceded its
capture. It does not fail merely because an earlier temporary pause remains in
the journal. This is the smallest correction to V2: no sentinel is removed or
ignored, and sealed-evidence failures remain fatal.

Model, preprocessing, scorer-contract, protocol, or finalizer code drift cannot
be re-admitted during the cohort. Only reviewed capture transport and unit
changes that preserve the frozen semantics can enter the admission chain. Nor
can a seal made under an
unadmitted runtime, a changed existing receipt, an invalid member identity or
capture, or a conflicting official result. Any such event ends the experiment
without metrics.

## Result closure and finalization

Approved official results observed strictly after the member's jump append to,
but never modify, the prediction receipts. Missing not-yet-published results
are recorded only after jump, remain temporary, and cause the cohort to wait;
there is no deadline that converts a valid fixed-N cohort into a smaller scored
cohort. A permanently unobtainable or ambiguous member result requires an
explicit operator abort with no metrics.

After exactly 1,000 predictions and 1,000 matching approved results, the state
becomes `READY_TO_FINALIZE`. The one-shot finalization request freezes a
deterministic manifest of the exact prediction/result receipt pairs and is the
durable scorer-start precommit. The scorer runs only in the process that
created that precommit. A restart with a precommit but no complete metrics
receipt seals deterministic no-metrics terminal evidence instead of repeating
the scorer. A complete metrics receipt can resume publication without another
scorer execution. The metrics receipt must bind that manifest. Restart replays
event IDs and hashes idempotently; a conflicting duplicate event ID is fatal.
Consumption occurs only after the final report is sealed and the complete
cross-hash commit validates.

## State machine

| State | Permitted progress | Failure behavior |
| --- | --- | --- |
| `PREPARED_NOT_AUTHORIZED` | Separate owner authorization plus initial admission | No collection is possible from repository state alone |
| `COLLECTING` | Seal the next eligible member or reject a nonmember candidate | Unadmitted runtime changes move to `ADMISSION_PAUSED` before write |
| `ADMISSION_PAUSED` | Append a reviewed hash-chained admission | Any prediction seal is fatal with no metrics |
| `RESULT_CLOSURE` | Append approved results for the immutable 1,000 members | Conflicts or permanent ambiguity require no-metrics termination |
| `READY_TO_FINALIZE` | Request the deterministic one-shot paired evaluation | Early or incomplete requests are rejected without scoring |
| `FINALIZATION_LOCKED` | Commit a metrics receipt for the exact member manifest | Manifest or scorer integrity mismatch is fatal |
| `FINALIZED_SCORED` | No further event | Terminal; cohort consumed |
| `FINALIZED_ABORTED_NO_METRICS` | No further event | Terminal; invalid experiment evidence, not negative model evidence |

The focused synthetic tests replay a complete 1,000-member path, a resolved
pre-seal code-drift path, restart/idempotence, candidate rejection, result
conflict, early finalization, semantic-contract drift, and explicit abort. The
state machine has no collection, database, systemd, model-fitting, or scoring
implementation.

## Future authorization gate

Starting the successor requires a new explicit owner instruction and a new
runtime task that freezes the implementation and activation receipt, verifies
the canonical Sportsbet WIN capture surface, installs only the reviewed unit,
and proves no candidate was observed or sealed before activation. Until then,
the protocol remains `PREPARED_NOT_AUTHORIZED` and no successor timer or
service should exist.
