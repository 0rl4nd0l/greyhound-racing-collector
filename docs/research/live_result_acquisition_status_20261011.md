# Approved July result acquisition — execution status

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
