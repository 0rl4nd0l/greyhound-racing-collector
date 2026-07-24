# Greyhound Race Prediction

This context covers the collection of race evidence and the production of greyhound race predictions from sealed evidence.

## Language

**Race Collection Service**:
The unattended service that collects upcoming races, pre-jump odds, and official results. It does not train or promote models, alter completed predictions, or place bets.
_Avoid_: Automation, daemon, autopilot

**Deferred Snapshot Prediction**:
An immutable prediction computed after a race starts but before its result is collected, using only evidence sealed before jump.
_Avoid_: Pre-jump prediction, live prediction, backtest prediction

**On-demand Forecast**:
A refreshable pre-jump forecast computed from current evidence for immediate operational use. It is stored separately and never enters champion-versus-challenger evaluation.
_Avoid_: Official prediction, deferred prediction, live model

**Ordered Finish Forecast**:
A coherent probability forecast over the possible finishing orders of a race, from which win, top-N, exacta, and trifecta probabilities can be derived.
_Avoid_: Winner pick, runner ranking, place heuristic

**Forecast Quality**:
The out-of-sample accuracy and calibration of an ordered finish forecast, independent of which wagers happen to be placed.
_Avoid_: Profit, winning bet, model accuracy

**Evaluation-eligible Race**:
A resolved race with sufficient provenance and one unambiguous official finishing order. Ambiguous outcomes are quarantined and reported rather than silently omitted or forced into an artificial order.
_Avoid_: Clean race, valid result

**Training Example**:
An immutable join of sealed race evidence and a provenance-bearing official result for an evaluation-eligible race. It is the canonical input to model training and historical challenger evaluation.
_Avoid_: Historical row, current database record, processed race

**Legacy Training Corpus**:
Audited historical evidence reconstructed before reliable forward sealing existed. It may bootstrap challenger training but cannot provide authoritative model-promotion evidence.
_Avoid_: Sealed corpus, evaluation corpus, ground truth

**Wagering Strategy**:
A fixed rule that converts an ordered finish forecast and sealed market odds into zero or more proposed tickets. Its utility is evaluated separately from forecast quality.
_Avoid_: Model, prediction, recommendation

**Sealed Race Evidence**:
The immutable, provenance-bearing inference input captured before jump for one race. It excludes that race's result and all information that became available after its feature-freeze time.
_Avoid_: Current data, race file, odds data

**Racing Day**:
The set of races collected for one race date, closed after every known race reaches a terminal lifecycle state or the configured hard cutoff passes.
_Avoid_: Calendar day, nightly run

**RaceId**:
The immutable internal identity assigned when a race is discovered. Source IDs, filenames, URLs, and venue/date/race-number keys are aliases and never replace it.
_Avoid_: Filename, source race ID, race key

**Dog Run**:
One dog's participation in one historical race, uniquely identified by canonical dog and local racing date under the invariant that a dog races at most once per day.
_Avoid_: History row, form line, dog race entry

**Run Observation**:
A provenance-bearing source account of a dog run. Multiple race cards, embedded form-guide rows, and official results may observe the same run without increasing its start or win count.
_Avoid_: Run, result row, duplicate

**Provisional Dog Run**:
A dog run supported only by embedded form-guide evidence. It may supply dog-level historical features but cannot imply a complete race or finishing order and is superseded when an authoritative race entry arrives.
_Avoid_: Historical result, inferred race

**Quarantined Race**:
A collected race excluded from its racing-day prediction batch because its lifecycle or sealed evidence remained unresolved at the hard cutoff. Its exclusion always carries an explicit reason.
_Avoid_: Skipped race, missing race

**Prediction Batch**:
The once-per-racing-day computation of deferred snapshot predictions. It completes when every eligible race has either committed an immutable prediction or entered quarantine, after which result collection may begin for committed predictions.
_Avoid_: Predict all, nightly predictions

**Model Training Workflow**:
A separately invoked workflow that creates and evaluates a candidate model from historical evidence.
_Avoid_: Automated ML, evening ML

**Model Promotion**:
The auditable change that assigns a qualifying challenger as champion from a future racing day. Its approval and effective dates are distinct and visible on every affected prediction.
_Avoid_: Auto-select, latest model

**Promotion Record**:
The immutable evidence that identifies a champion change, its approval and effective dates, compared race population, promotion scorecard, artifact checksum, and rollback target.
_Avoid_: Registry update, model date

**Prediction Provenance**:
The mandatory identity and timing evidence connecting a prediction to its champion artifact, promotion record, training cutoff, evidence freeze, and computation time. Missing provenance invalidates the prediction.
_Avoid_: Model info, version string, timestamp

**Champion Model**:
The promoted immutable model bundle assigned to future racing days. It retains incumbency until a challenger passes the long-horizon promotion policy.
_Avoid_: Best model, latest model, production file

**Challenger Model**:
An immutable model bundle evaluated on the same sealed race evidence as the champion without affecting committed predictions.
_Avoid_: Test model, experimental settings

**Long-horizon Scorecard**:
The broad, coverage-aware evaluation used to decide model promotion across a minimum resolved-race sample.
_Avoid_: Overall accuracy, leaderboard

**Short-horizon Monitor**:
The recent-race evaluation used to detect drift and nominate investigation or challenger training. It cannot authorize promotion by itself.
_Avoid_: Recent winner, daily leaderboard
