# Approved July result acquisition — execution status

## Service approval and offline integration update (11 October, 09:18 UTC)

The user subsequently approved safe collector recovery and activation of the reviewed July interface. That approval is retained at `service-activation/user-approval.json`; it supersedes the earlier restriction on service changes for this narrow recovery. Operational thread `01a11a4d-08df-7c71-b9c1-d092513b8b2b` is the sole runtime/provider owner. The audit process makes no provider requests or service changes.

The collector had failed, and automatic preparation timed out leaving an unresolved ownership token and partial package. The operator has paused the automatic recovery watcher to avoid competing recovery attempts; the receipt requires restoring it. Preserving and reconciling the interrupted ownership evidence remains underway. Prepared source code, passing tests and an initially empty request queue do not prove activation or acquisition. No new request completion is confirmed at this update.

`scripts/integrate_acquired_live_results.py` now joins actual owner completion receipts to the frozen population. It requires an externally pinned, genuinely authorized admission; checks all race membership and authority references before decoding bodies; preserves consumed intents without completion; and isolates invalid response evidence by race. Every original forecast gets a disposition. New independent identity, unchanged-field and WIN evidence can promote a race; winner or field contradictions suppress the older weak diagnostic while preserving its evidence. Losing-rank or terminal-status revisions alone do not veto a complete unchanged field with the same unique winner, and receive explicit correction events. This WIN interpretation was recorded and reviewed before new outcome acquisition.

Root validation passes **118 tests and four subtests**, with four existing optional tests skipped. Independent review separately checks the integration and projector. The projector's operationally pinned hash is unchanged. No new outcome was used to change forecasts, selectors or thresholds. The prior nine-race diagnostic remains the latest actual scorecard until accepted completions arrive.

The sole owner must publish the actual admission path/hash and a metadata-only census of **all** intents and completions, including failed and interrupted consumption. Run the integration against those artifacts, then feed its output directory to the existing frozen rerun:

```bash
python3 scripts/integrate_acquired_live_results.py \
  --benchmark-evidence /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --prior-projection /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results/archive-result-projection.json \
  --reconciliation /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results/reconciliation.json \
  --completion-manifest <actual-owner-census.json> \
  --approved-admission <actual-owner-admission.json> \
  --approved-admission-sha256 <verified-admission-sha256> \
  --output <new-integrated-directory>
python3 scripts/rerun_frozen_live_reconciliation.py \
  --original-evidence /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --reconciled-records <new-integrated-directory> \
  --output-dir <new-metrics-directory>
```

The earlier observations below are retained as historical snapshots, not current service-health claims.

## Earlier acquisition and ownership snapshot

The user approved the prepared request for up to **210 TheDogs official-result GET attempts**, one per exact race identity, at least60seconds apart, with no automatic retry. The approval is recorded in `acquisition-authority.APPROVED.json`, hash-bound to the original prepared request. The request,210-race manifest and frozen population hashes agree. The allowance is unconsumed: **zero requests and zero new official responses**.

Execution is waiting for genuine operational-owner admission. The active collector holds the campaign owner mutex for its entire lifetime, and its installed release has no result-purpose delegation hook. Read-only source admission returns `sportsbet_persistent_owner_required`. The current source's open state cannot confer the owner role. The [ownership audit](live_result_acquisition_owner_audit_20261011.md) records the actual systemd command, kernel lock holder and source controls.

Root contacted the active operational-owner thread about an already-authorized safe interface or handoff. Its separate prospective-programme work does not automatically admit this July grant. No collector was interrupted, owner tag impersonated, lock acquired or reset, old allocation reopened, source allowance replaced, or background fetcher installed. The210-race queue records `AWAITING_OWNER_ADMISSION`; all original result and forecast dispositions remain intact. The approval remains valid for its exact scope; it is not being requested again.

## Offline work completed after approval

`scripts/project_acquired_live_results.py` now converts accepted retained responses into the benchmark projection. It reads membership/access boundaries before response bodies; requires the exact authorized requested URL and unchanged final URL; verifies response hashes and post-jump acquisition time; and requires an independent canonical/OG race identity and one complete official result table. It retains source-native dog names rather than copying the original forecast roster. Original raw-response, metadata, projector and imported-parser identities are hashed.

Every official runner row, scratched reserve, source-supported reserve mapping and terminal status is preserved in derived evidence, including rejected fields. Original active-runner scratches, changed names, additions, unknown statuses, malformed/unparsed rows, winning ties, redirects and race-identity contradictions prevent qualification. Explicit DNF/FELL/DISQ for an independently identified original runner can establish a losing starter with a null finishing position; no losing placing is invented. Output witnesses are written once and previous files cannot be overwritten.

**95 focused tests and four subtests pass**, including18 new parser tests. These are synthetic qualification tests, not claims about newly fetched source availability. The empty actual response manifest was executed through the CLI: all210 records are explicitly `NOT_ACQUIRED`, zero qualify. This new acquisition projection does not replace the nine older diagnostic joins.

To project future owner-admitted responses, use a manifest whose `records` are `{race_id,response_path,metadata_path}`. Metadata must contain `{requested_url,final_url,http_status,acquired_at,response_sha256}`. The exact authorized URL comes from the original reconciliation, not from the fetched page.

```bash
OPENBLAS_NUM_THREADS=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  scripts/project_acquired_live_results.py \
  --acquisition-manifest <response-manifest.json> \
  --benchmark-evidence /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --reconciliation /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results/reconciliation.json \
  --output <new-output-directory>
```

The existing [benchmark report](live_reconciliation_decision_20261011.md) remains the performance conclusion: zero qualified races, nine incomplete diagnostic joins,201 absent joins, and no later selector outcomes. No new performance conclusion follows from the acquisition approval itself.

Evidence root: `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence`. The approved grant, `acquisition/queue.INITIAL.json`, append-only `acquisition/events.jsonl`, `acquisition/owner-admission-inspection.json`, `acquisition/responses.NOT_STARTED.json`, all210 projected acquisition dispositions and independent review are retained. The next action is a verified owner-admitted execution route bound to this grant; until then the root owner makes no GET.
