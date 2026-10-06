# Independent raw-cell arithmetic verification

`race_collection/sectional_speed_oracle.py` is an independent verification module. It imports no functions or constants from the feature implementation. Root alone runs it on real retained inputs after issuing the experiment receipt.

The interface is `verify_sample(packets, outputs, reader)`. Packets are the feature module's original input packets, and outputs are its matching result objects. The reader is either a callable accepting a `{path, sha256}` reference or an object exposing `read(reference)`. Root supplies the same finite, authenticated, network-denied reader used by the real experiment. Source reads are cached locally by exact path and SHA so repeated historical copies do not cause repeated CSV reads.

Every packet must contain `target.source_card`, independently recorded before history observation construction. It binds the complete accepted CSV, source racing date and availability, complete source roster including block tokens, identity mappings and box numbers, source-binding template, and any fixed alias evidence. `target.observation_pool_scope` is `ALL_SOURCE_CARDS` for the October native-profile population and `CARD_LOCAL` for the legacy development population. The oracle compares this source roster against both the original CSV block/box roster and the target packet roster. It refuses verification without this complete inventory, even if the observation bank is empty.

The oracle independently enumerates all eligible raw history rows for every allocated source card, reconstructs original observation copies, and compares source locations and complete signatures to the adapter's observation population at every target cutoff. Omitted newest rows, omitted complete peer histories, missing-value rows or arbitrary extra source rows fail this reconciliation before arithmetic verification. This closes the earlier v1 defect where checking only surviving observation bindings could accept a self-consistent but incomplete adapter projection. The v1 defect and its fabricated reproduction remain recorded in review; v2 does not describe surviving-row checks as source completeness.

## Fixed representative sample

The seven prespecified categories are supported, one supported observation, two supported observations, an actual missing first-sectional cell, unsupported history, conflicting historical copies, and aliases. Each observed category selects the minimum SHA256 of `[race_id, runner_id]`, giving at most seven distinct runners. Missing first-sectional cells and merely unsupported benchmarks are distinct categories. Category support counts are recomputed from independently enumerated raw histories; corrupt or omitted adapter observations cannot remove a case from the sample. Categories absent from the real data are reported as absent, never supplied by fabricated substitutes.

Sampling uses neither target outcomes nor effect sizes. The oracle covers each selected runner and the complete eligible historical pool needed to establish its benchmark. It is proportionate representative verification, not a claim that every historical median in the experiment was independently recalculated.

## Source-cell checks

Every historical copy surviving that packet's temporal filter is resolved directly against its original CSV. The binding supplies `accepted_csv`, `block_token`, zero-based `block_row_index`, and `available_by`. October native-profile bindings also carry the original HTML/receipt references and identity availability; the adapter remains responsible for validating that native identity bridge.

The oracle independently verifies the CSV SHA, parses native dog blocks with the same documented label-token convention, and reads only the projected history fields. It checks original `DATE`, `TRACK`, integer `DIST`, `1 SEC`, and the complete SHA256 fingerprint of `DATE/TRACK/DIST/TIME/WIN/BON/1 SEC/PIR`. Total time and the other projected fields are fingerprint inputs, not candidate features. Placement and odds columns are neither interpreted nor emitted.

The source binding's observation and identity availability must not exceed the observation's stated availability. All allocated raw cards are read once to establish complete source enumeration. For each target, future evidence is excluded before population comparison or conflict reconciliation; no unknown path in an excluded future observation binding is followed. Later contradictory evidence cannot invalidate an earlier calculation. Source proof for alias equivalence remains the adapter's responsibility; the oracle checks the raw track and the fixed supplied alias mapping's use, not a second external source enquiry.

## Independent arithmetic

For the selected cases, the oracle independently reconstructs:

1. Cutoff and strictly prior-date inclusion.
2. Distinct dog/date observations, conflicting event dates, conflicting signatures and duplicate copies.
3. Benchmark membership excluding the runner itself.
4. One median contribution per other runner, minimum five other runners, robust centre and median absolute deviation multiplied by `1.4826`.
5. Sparse and zero-spread benchmark rejection.
6. Each standardized sectional, clipped to `[-3, 3]`, and the latest five qualifying observations.
7. Runner median standardized sectional and fixed `n/(n+3)` shrinkage.

It compares source bindings, selected observation identities/order, benchmark peer membership and history counts as well as arithmetic. Numeric comparisons use `1e-12` absolute and relative tolerance; identity, membership, parameters and dispositions use exact equality. A failed check raises a fixed safe category and must make root's execution fail rather than publish a successful partial verification.

The optional `verify_adjustment(baseline, estimates, coefficient, probabilities)` independently checks a baseline-offset softmax over the complete field. A zero coefficient or entirely zero feature vector must reproduce the baseline exactly. Otherwise an unsupported runner with zero direct speed estimate can still receive a different normalized probability because other runners' weights changed.

## Validation and limits

Fabricated tests exercise all seven sample categories, duplicate copies, omitted newest/peer/missing rows, an empty adapter observation bank, missing inventory, unsupported-without-missing distinction, changed raw cells, changed output arithmetic/membership, falsified support claims, cutoff-safe future conflicts and complete-field probability normalization. They do not open retained real timing cells or labels. Root records complete source enumeration, real arithmetic sample results, source-read consumption and absent categories alongside the executed feature outputs.

This verifier establishes the sampled source calculations. It does not independently re-prove every runner's native identity, certify source timing semantics, prove cross-track comparability, or establish predictive improvement. Those remain separate provenance, assumption and evaluation claims.
