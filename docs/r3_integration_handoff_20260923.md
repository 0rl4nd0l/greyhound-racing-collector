# Collector to R3 integration: offline handoff

This change is separate from PR #184 and the 24 September collector rehearsal.
Successful collection supplies authenticated discovery and price inputs. It does
not itself authorize prediction, retain history, configure result discovery, or
close the research journal. The new connections remain default-off.

## Source and runtime identities

| Component | Verified identity and meaning |
| --- | --- |
| Integration base | `89d22067a737222a053e73e02726ba352eec08e3`, current master at start; isolated branch `fix/r3-collector-handoff-20260923` |
| PR #184 | Open, head `fe3d984a13a7446ca0af301dd736fed20bab0b96`; not modified by this work |
| Prepared collector execution | `6e7fa6e423655df22a6769e524b18c975359b2f6`; package `access-review-20260923/release-candidate-6e7fa6e4` under the existing campaign root |
| Installed collectors | Unit working directory `/home/l4nd0/greyhound-retention-lanes-repair-20260917/source`; both services inactive and timers disabled at start |
| Installed R3 | Binding commit `5013ff039fda418f47a59373041d7bc7c4124f07`, tree `509f77cd8e9d2a357198adbc3f858df601ef8b30`; source `/home/l4nd0/greyhound-r3-release-20260916-5013ff03`; PID 149626 at start |

The candidate uses `scripts/run_freshness_service.py run-once --live-freshness`
with `bounded80-v1`. Its generated full-service command has neither result
acquisition nor retention activation. It is intentionally different from the
ordinary full collector and from installed R3 authority. No prepared command,
package or binding was edited to accommodate this change.

## Actual path and defects

1. The existing collector publishes the v2 current index after refresh, with
   hashes tying CSV/sidecar runners to the refresh, publication report and state.
   Scheduled successful acquisition separately publishes a complete exact
   receipt through `publish_scheduled_capture_receipts`. Index freshness and
   receipt validity are distinct gates.
2. `bootstrap._build_r3_services` resolves an allowlisted receipt-only request,
   performs receipt preflight and freezes `OperationalIndexProvenance` into
   `JobInput`. The journal adds activation, cutoff and result-readiness gates
   before allocating a durable job. Claim-once dispatch and verification of the
   sealed producer bundle already exist.
3. Previously, `ResultAcquisitionReadiness` required
   `daily_race_ingest_shadow_<current-run>_daemon_autopilot/` containing
   `stage2_shadow_predictions.jsonl`, `shadow_feature_rows.json`, and matching
   source CSV. The ordinary daily shadow producer creates those files; the
   bounded collector campaign does not. The result backlog also finds candidates
   through shadow predictions/features, so deleting that readiness condition
   alone would leave admitted R3 races without a demonstrated discovery route.
4. The opt-in result connection instead nominates verified durable R3 jobs and
   their sealed outputs to the **existing** official-result collector. Admission
   checks the exact configured job-store/bundle paths against the pinned full
   service and reauthenticates the same collector publication and native runner
   identities. Missing, changed, stale or mismatched prerequisites reject. Legacy
   units still use their existing shadow prerequisite; no fallback fabricates it.
5. `OfficialResultSource` reads only matching collector-owned result rows.
   Journal closure rechecks prediction and result identity, complete finishing
   order and post-prediction/post-jump observation chronology. Delays stay pending;
   ambiguity stays rejected. Prediction verification and closure machinery are
   reused rather than replaced.
6. The existing predictor previously reopened current history even when a
   retained-input bundle existed. The default-off retained consumer connection
   binds the exact bundle/manifest, uses its sealed history and forms and checks
   the actual 16-feature scorer input against its retained projection. Details
   and limitations are in [retained consumer findings](r3_retained_consumer_20260923.md).

The [capture acceptance contract](r3_capture_acceptance_contract_20260923.md)
maps each field/window/cutoff and supplies explicit race, window, receipt and
usable-time denominators. One valid observation can support residual generation;
the adapter still requires both WIN and PLACE. All-four-window coverage is a
different diagnostic. The dashboard's 300-second freshness policy differs from
the frozen config's 1,200-second maximum; the new journal handoff enforces the
stricter 300-second operational boundary without changing frozen configuration.

## Deployment and activation still needed

These are future steps, not actions performed by this PR:

1. Finish the already prepared PR #184 collector observation without modifying
   its package or allowances. A pass proves only that finite collector case.
2. Choose and approve a permanent combined source release, persistent paths and
   paired collector rollback. The bounded campaign wrapper does not become a
   recurring result-acquisition service merely by adding configuration.
3. Generate the ordinary full collector with `--skip-shadow-run`,
   `--enable-autonomous-result-capture`
   and matching `--r3-job-store <R3 operations>/jobs.sqlite3` and
   `--r3-prediction-bundles <R3 operations>/artifacts/on_demand_prediction_runs`.
   Keep both new options absent until that deployment is authorized. Make those
   exact paths readable by the collector; keep canonical result appends under
   the existing collector authority and shared lock. The ordinary generated
   full lane uses a result-candidate limit of eight per cycle (standalone maximum
   128); pending admitted jobs are not dropped by the legacy two-day lookback.
   The one-race proposal fits this bound. Legacy full-lane reporting/rejoin
   stages still exist; this change does not define a new recurring service.
4. Regenerate the R3 deployment/live-evidence binding against the exact installed
   paired collector definitions and combined source. Old service hashes cannot
   authorize the new configuration. Preserve private listener, pinned Python,
   frozen model/config/schema and existing rollback paths. Retention and R3 must
   use compatible exact environment locks: the inspected collector interpreter
   has `requests=2.34.2`, `charset-normalizer=3.5.0`; installed R3 has `2.32.4`
   and `3.4.9`. Both have Python 3.11.15, urllib3 2.5.0, certifi 2026.7.22 and
   idna 3.18. The new consumer rejects the current cross-environment pairing.
   Validate a common pinned environment for the future combined release; do not
   install dependencies into tomorrow's campaign interpreter.
5. Separately approve a future race, pre-cutoff history/form retention scope and
   retention configuration in both collection lanes. After successful parent
   acceptance, bind its exact retained bundle and manifest to the R3 generated
   profile using `src.operator_ui.deployment generate --retained-input-bindings`
   and the existing deployment arguments. Missing bindings cannot fall back to
   live history. This finite per-race mapping requires preparation after parent
   acceptance and before the retained cutoff; automatic recurring selection of
   new retained bundles is not implemented or implied.
6. Separately authorize a bounded journal activation with future cutoff and
   result observation limit. Require fresh index, exact timely receipt, accepted
   retained input and configured result discovery before its one claim. Observe
   verified sealed output and subsequent official closure as separate milestones.

Concrete next decision: one explicitly eligible future race, one authenticated
retained observation, one receipt-only R3 claim and eventual closure. No four-window
requirement is proposed for that proof. Recurring coverage still needs a declared
race scope and minimum usable-race/time target; no population or protocol is
changed by this proposal.

## Verification record

The initial real-publisher/index-reader regression failed with
`RESULT_ACQUISITION_NOT_READY` for a complete invented collector-only publication.
It passes with the configured R3 result connection. Mismatched race, runner,
runner hash, run, refresh hash, publication hash, jump, service binding, stale or
future observation and unverified index remain rejected. Altered refresh bytes
remain rejected by the native index reader.

Tested implementation commit:
`d5705e4f486bf8a8e638784d6da1552b3fe9c7bb`. Subsequent handoff/evidence edits
are documentation only. Machine summary:
[r3_integration_validation_20260923.json](r3_integration_validation_20260923.json).

Executed with the existing pinned R3 Python 3.11.15 interpreter and a test audit
guard denying network and non-fixture data reads:

| Check | Executed result |
| --- | --- |
| Affected predictor, journal, JobStore, worker, bootstrap, deployment, retention and collector suites | **956 passed, 2 deselected**, 104.90 seconds |
| Standalone exported source, real consumer/admission/result-binding and generated-package checks | **80 passed, 119 deselected**, 17.48 seconds; deselections are the explicit focused package selection |
| Local combination with PR #184 head `fe3d984a`, tree `6546b4179c91846246a000d54b4bd1bd4ce45a3f` | Automatic merge without conflicts; **341 passed, 2 deselected**, 12.05 seconds; this is an isolated uncommitted compatibility tree, not a PR merge |
| Independent verifier regression rerun | **2 passed**, 1.87 seconds |
| Source formatting/integrity | `git diff --check` passed; no frozen model, feature schema or prediction-config changes |

The two full-suite exclusions are
`test_autopilot_default_min_joined_races_matches_review_target` and
`test_autonomous_live_odds_capture_runs_before_daily_shadow_run`. Both were
independently reproduced against exact baseline `89d22067`: stale assertions
expect refresh limit 8 while baseline already uses 16. They were not silently
fixed or counted as passing. This is focused regression coverage, not a claim
that the entire repository suite passed.

Retained failed evidence includes the initial handoff/consumer regressions,
the review verifier reproductions, an earlier collector-environment startup
failure (`flask_compress`, later passed with the installed R3 interpreter),
three fixture-isolation failures corrected without relaxing the test guard,
and an initial export-harness failure missing `accuracy_program` Python modules.
The corrected export contains 1,018 regular code/config/test files plus one
internal wrapper symlink; no raw historical or outcome fixtures. Archive SHA256:
`53b7248d01171c358e539d2034d727de669cb43d914c92bdbd87de23f3e405e6`.
Logs, the export script, per-file hashes and both reviewed revisions are retained
under `/home/l4nd0/greyhound-r3-integration-evidence-20260923/`.

Synthetic transport/scoring establishes orchestration and provenance only. No
live source availability, throughput, prediction quality or eventual official
result delivery is claimed.

## Standards

Independent review found zero documented violations and zero actionable smell
findings at the initial implementation and corrective revision. Reviewers
inspected repository guidance, relevant ADRs, source and synthetic tests.

## Spec

Independent review found one P2 issue: an outer producer could reseal rejected
or chronologically invalid retention completion metadata while preserving the
pinned retained manifest. The repair shares the complete metadata validator
between consumption and independent indexed verification. Both exact attacks
now reject; the reviewer reran them. Zero unresolved Spec findings remain.

Review totals: Standards **0**; Spec **1 found and fixed, 0 unresolved**.

## Campaign preservation

The preservation comparison found identical prepared archive/plan, launch
script, installed collector/R3 unit bytes and R3 binding. Both collector
services stayed inactive, timers disabled, and R3 remained the same running PID.
Campaign ledger bytes are unchanged: **1/12 attempts, 405/48,000 requests,
721.058807/10,800 charged seconds**. No launch was armed, source recovery used,
provider contacted, real prediction made, retention/journal activated, runtime
installed or PR merged. The separate draft PR is the only publication.
