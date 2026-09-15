# Default-off R3 journal continuation

Scope: the user-approved offline implementation following deployed PR #177
(`d8e4efe1ebeb44fff8193c3fcd7e7d8d55d7d182`). This document does not authorize
merge, deployment, restart, prediction, outcome access or experiment activation.
No existing journal, cohort, attempt, result or model is migrated or rewritten.

## Implementation contract

- The existing R3 process owns recurrence. It shares audited admission, JobStore,
  the fixed dispatcher, frozen scorer and sealed-bundle verifier with HTTP.
  The Race Collection Service still owns acquisition, index publication,
  receipts and official results. No extra service, timer or prediction database.
- Default-off means no journal thread, directory creation or result observation.
  The generator's existing `--enable` does **not** enable the journal.
- A separately reviewed activation manifest pins a canonical UUID, exact release
  commit, unchanged model/config hashes, governing protocol hash, future cutoff,
  admission end, explicit maximum jobs (1–64), and retained race exclusions.
  The protocol hash is a reference for human review, **not** proof of permission.
  Do not use a Betfair, October-successor or other restricted experiment protocol
  without establishing that this per-race result-access policy is authorized.
- Initial activation must occur before its declared cutoff. Its exact immutable
  manifest is retained under `operations/artifacts/research_journal/<UUID>`.
  Restart cannot move the cutoff, enlarge the allowance or change the contract.
- Every 60 seconds, observe at most 64 index rows; select by jump then race ID;
  admit at most one receipt-qualified future race. Use only `latest-research`,
  `manual-default`, `receipt`. Existing index/receipt and worker gates still apply.
- Reconcile all existing JobStore race IDs, across human and journal actors.
  Recorded races cannot receive another job. Missing receipts allocate nothing.
  A consumed attempt is never relaunched. Recover unclaimed queue entries and
  producer-completed verification through the same R3 operations.
- An unclaimed entry that is no longer admissible remains unchanged and is
  reported as `STOPPED_UNCLAIMED_ADMISSION`; it cannot prevent existing verified
  jobs closing or prevent the web process starting. No attempt is consumed.
- Any terminal unsuccessful job stops further admission for this activation.
  Unresolved attempts also prevent further admission. Existing successful jobs
  can still receive result closure. No replacement race after failure.
- Official-result reads occur only after jump and verified prediction completion,
  for that exact race. They require one terminal race, exact race/date/venue/URL,
  matching runner names/boxes, no conflicting native IDs, consistent timestamps,
  winner and complete unambiguous finish order. Ambiguity remains explicitly
  rejected evidence; it is not silently counted as a complete record.
- Closures are separate immutable files referencing the activation, JobInput,
  prediction and logical bundle hashes, plus retained official rows and their
  hash. The original prediction bundle never changes. Rechecking a closure
  re-verifies its bundle and compares its official rows with the independent
  canonical source; a self-rehashed local file is insufficient evidence.
- `CLOSED` means a scoreable research evidence join, **not a computed score**.
  There are no interim metrics, model comparisons, EV, staking or betting outputs.

## Operational limitations and denominators

### Result-acquisition prerequisite (follow-up to #178)

Before allocating a journal job, or dispatching an unclaimed journal job after
restart, require an outcome-blind collector precursor. Missing readiness reports
`RESULT_ACQUISITION_NOT_READY`: zero new jobs and zero consumed attempts. Existing
sealed predictions can still close; historical inputs and events are unchanged.

The supported proof is deliberately narrow: the verified index's run ID selects
`daily_race_ingest_shadow_<run_id>_daemon_autopilot` directly under the bound
collector evidence root. Its direct stage-2 predictions and feature rows must
identify the exact race, canonical URL, jump time and complete runner set, with
a matching source CSV inside that root. It reuses the collector's runner/CSV
completeness, race identity, time and result-field guards, without executing its
post-jump candidate loader or its network-capable "dry run". The current index
must still match the JobInput, including native runner IDs and runner digest.
The pinned full-service definition must still match its deployment digest and
configure the existing autonomous result-capture entry point and evidence root.

At most three source files plus the unit definition are read, each at most 2 MiB;
JSON row counts are bounded. Symlinks, non-regular files, changed files and
result-contaminated JSON fail closed. No directory scan, result-table lookup,
network access, enrollment, snapshot creation or alternative collector is added.
The source-read bounds are byte/row bounds, **not a wall-clock latency guarantee**.
Admission rechecks its future cutoff after all readiness inspection finishes.

This is a necessary source prerequisite, **not a reservation or proof of future
result delivery**. Snapshot-only, legacy-only and manifest/checkout-relative
fallback sources remain pending in this first implementation. Do not fabricate
a supported precursor to increase coverage. Timer activity, lock contention,
live-odds backlog eligibility, lookback/selection limits, official-source
availability and eventual canonical ingestion remain separate operational gates.
The preflight does not access outcomes to predict whether closure will succeed.

Release this follow-up only from its reviewed exact merged commit/tree, with a
regenerated default-off package. The deployed #178 release
`0869ada06fc7a16b716253c279d3fa9a1634de53` is the rollback source for this follow-up;
retain its package and all operations stores. Neither package installation nor
this documentation authorizes journal activation. Before the separately approved
future-only one-job acceptance, revalidate collector recurrence/backlog settings
and protocol authority. A pending precursor consumes no opportunity; a pending
official result after prediction is not a passing complete-record acceptance.

`jobs` in a tick report are durable opportunities in this activation, not every
race seen in the index. `admissions` describes that observation's preflight or
allocation decision, not a cumulative denominator. Count consumed attempts from
JobStore's claim evidence separately. A failed prediction remains a failure even
if its result is subsequently available. Historical private-journal records and
other forward-corpus populations are not added to these counts.

The official-row adapter has a one-second SQLite query budget and bounded row
counts/sizes. It uses `mode=ro&immutable=1` **only when no WAL, SHM or rollback
journal exists**, with file-identity checks before/after. Busy or changing source
evidence remains pending; no unsafe immutable read of live WAL is attempted.
Whether the live database supplies sufficiently frequent quiescent windows is
unverified. Persistent WAL would be a remaining availability blocker, requiring
a separately verified read-only snapshot/read path, not an identity relaxation.

Malformed/contradictory result evidence is `RESULT_REJECTED`, not a consumed
prediction failure and not a claim that the official source can never correct it.
Already-closed records are preserved if source rechecking becomes unavailable;
the observation then reports `CLOSURE_RECHECK_PENDING`, not a new closure.
Job reconciliation fails closed above 10,000 stored jobs; no silent truncation.
Initial reconciliation is one integrity-verified transactional snapshot, not
one complete integrity scan per historical job.

## Controlled rollout, only after approval

1. Review and merge the candidate, pin the resulting exact master commit/tree,
   and run current-head CI plus native R3/deployment checks. Do not deploy a
   reviewed branch SHA as though it were the eventual squash-merge SHA.
2. Generate the normal R3 package with fresh live authority and **without**
   `--journal-activation` first. Default-off rollout changes no research records.
3. Before any activation, review the governing protocol's population, outcome
   access and stop rules. Prepare a new UUID manifest using exact merged release
   and frozen hashes, a future cutoff comfortably after startup, and
   `maximum_jobs=1` for the bounded acceptance. This is not permission to resume
   the August private journal or enlarge a frozen cohort.
4. Add `--journal-activation /absolute/reviewed-activation.json` to the existing
   generator command. It adds a digest-pinned, read-only generated manifest;
   startup binds the continuation to the existing writable operations root.
   Verify all generated files, unit syntax, exact paths and read-only mounts.
5. Only with explicit installation/restart and one-job authority: install, verify
   health before cutoff, and observe a naturally arriving eligible receipt.
   Record index/receipt hashes and age, job ID, claim count, frozen identities,
   verified bundle, and eventual independently verified official-result closure.
   If no race qualifies, report zero opportunities/attempts. Do not backfill.
6. A consumed failure stops acceptance. Preserve its evidence; do not try another
   race. A successful prediction with no official result remains pending, not a
   passing end-to-end acceptance. Review recurring activation separately.

Rollback: regenerate without `--journal-activation` and, with the separately
approved operational action, restore the known-good #177 package/source
`d8e4efe1ebeb44fff8193c3fcd7e7d8d55d7d182`, or disable R3 through its generated
gate if required. Keep every operations directory, job, attempt, audit chain,
activation and closure. Never edit the retained cutoff to force another run.

## Offline validation

The agreed test seam is one coordinator cycle with real JobStore/audit stores,
native fixture-produced sealed bundles and collector-format result rows. The
fixture producer uses the existing predictor tests' deterministic feature/scorer
dependencies; this is not evidence of live scoring latency or model superiority.
Existing HTTP, bootstrap and generator tests also exercise the shared code.
No live prediction or canonical-result lookup is part of these tests.
