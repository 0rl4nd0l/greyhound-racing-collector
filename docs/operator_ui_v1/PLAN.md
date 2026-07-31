# Greyhound Operator UI V1 execution plan

## Authority and identity

This execution surface materializes the approved bundle
`/home/l4nd0/greyhound/GREYHOUND_OPERATOR_UI_SYSTEM_PLAN_BUNDLE.zip`,
plan `GREYHOUND_OPERATOR_UI_SYSTEM_PLAN.md`, and ticket schema
`greyhound-operator-ui-ticket-seed-v1`. The repository is
`0rl4nd0l/greyhound-racing-collector`; the accepted programme base is commit
`aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, tree
`b34eeefdff0ebf74cc4f38d99652bea848488abc`, on the canonical integration branch
`agent/operator-ui-programme-20260730`. A Codex X worktree branch is ephemeral
candidate provenance, not the programme branch. Its historical upstream
integration base was the actual merge parent
`51a5287d05c790e3855e5b74ce7117a29340135e`, merged by
`6f4fba42c45c73702efb017a21cbd284b44c1d04`. Later source drift is
`origin/master` `f38a125f6364b8a60d17ae9c971b0ce172874eea`, tree
`408a8adbfa2bd436132bc4d2c63e952aeb57c5a5`, parent
`51a5287d05c790e3855e5b74ce7117a29340135e`; it was incorporated by local merge
`0b08966b31c15d8b459b9c6b60a48b19030a9ce4`. Current accepted integration is
`51fe070ba2a0778bca0b0334c00cae9d75561952`, tree
`0c2b82a391b8dd0a1dcc525cf4203edc910843c1`.
`CONTRACTS.md` is the accepted source/evidence/authority contract. Live
repository evidence supersedes stale seed metadata without weakening either
the approved programme or that contract.

`GHU-000B` is an additive source-drift correction. It supersedes only source
details changed by `f38a125f` and does not reopen accepted `GHU-001`.
Parent-accepted prerequisite `GHU-000C1` closes the fixed-packet read-safety
finding before this documentation correction; its rejected predecessor remains
preserved and is not part of integration history.

The product is a private Flask/Jinja/static research console rooted at
`app.py`, `templates/`, and `static/`. It makes no verified market-edge claim.
It provides no training, promotion, EV, staking, betting, public exposure, or
outcome access before closure. FastAPI is secondary; stale frontend/TGR
surfaces are not product authority.

## Releases and gates

| Release | Entry | Outcome and exit gate |
|---|---|---|
| R0 | Verified programme base | `GHU-000`, `GHU-000A`, `GHU-001`, and `GHU-002` accepted and integrated. |
| R1 | R0 accepted | Responsive fixture golden flow with browser, accessibility, mobile, and desktop evidence; frozen UX review. Every fixture surface visibly says **PROTOTYPE DATA** and **RESEARCH ONLY — NOT FOR BETTING**. |
| R2 | R1 accepted | Truthful authenticated GET-only dashboard/API; every operational value carries source and freshness; auth/security/truth review accepted. Stale, unavailable, missing, malformed, or divergent evidence is never healthy, empty, success, or zero. |
| R3 | R2 accepted | Exact-race persisted/idempotent fixed worker, one click/one process, reconnect, evidence/audit, deterministic safety, and synthetic E2E accepted. `GHU-035` owns fixture/synthetic success and every terminal blocker. `GHU-036` produces repo-generated deployment behind a reversible default-off flag. `GHU-037` alone owns one natural cycle and one exact live UI job, and proves only its naturally reached path. |
| R4 | R2 accepted | Read-only corpus readiness and model lineage plus a non-executing experiment-spec builder. No training. |
| R5 | Deliberately deferred | Experiment execution, model persistence/registration, training, activation, and promotion require a future contract and separate authority. |

Each tranche ends with independent exact-delta review and parent acceptance.
Child implementation, review, parent integration, publication/PR, merge,
generated deployment, and runtime proof are separate events and claims.
Repository code, CI, or a deployment package cannot establish runtime
prediction proof; only valid `GHU-037` scoring can.

## Execution control

Legal states and transitions are:

- `planned -> ready` only when every prerequisite is accepted/integrated and
  current authority/source evidence exists.
- `ready -> active` only when assigned to one fresh bounded implementer.
- `active -> review` after a frozen diff and focused validation.
- `review -> accepted` only after independent review and parent exact-delta
  acceptance/integration.
- `review -> planned/ready` through the smallest correction ticket after
  rejection. A rejected candidate never regresses an accepted prerequisite and
  is preserved rather than silently overwritten.
- Any incomplete ticket becomes `blocked` only for an evidenced genuine stop;
  deliberately out-of-scope work becomes `deferred`.

The parent owns the graph, acceptance, commits, publication, merge, deployment,
and proof. Codex X may implement and validate, or independently review
read-only; it never approves itself. Closeout records exact base/head/tree/path/
diff identities, commands and exits, focused tests, reviewer identity/verdict,
parent decision, and commit/PR/merge/deploy/proof fields where applicable.

An accepted delivery may be bound atomically by an evidence pair: (A) an exact
frozen product/docs/test checkpoint and delta that has been independently
reviewed, and (B) a subsequent independently reviewed ledger-only closeout
delta, changing only programme ledgers, that records the product checkpoint
identities, validation, reviewer verdict, and parent acceptance/integration.
The parent must inspect and accept/integrate both members. The accepted product
checkpoint plus the accepted ledger-only closeout form the atomic evidence
binding; they need not be one mechanically self-referential commit.

Ordinary candidate state transitions remain in the product delta whenever the
required identities are knowable. A successor cannot leave `planned` until the
ledger-only closeout is independently reviewed and parent-integrated and
records each prerequisite's accepted evidence pair with exact commit/tree/
reviewer/parent-decision evidence. A successor `ready` state recorded in the
reviewed ledger-only closeout becomes durable and effective only when the
parent integrates that closeout; the unintegrated candidate neither preclaims
acceptance nor mutates durable programme state. Its assigned candidate then
records the legal `ready -> active -> review` transitions and focused
validation evidence before review. A ledger-only closeout must not change
product/test bytes, weaken evidence, preclaim its own parent acceptance, or
erase rejected history. A rejected candidate does not mutate the integrated
ledger. Append
`DECISIONS.md` only for an actual programme, architecture, or authority
decision; ordinary state changes belong in `TICKETS.md` and `STATUS.md`.

Specifically, the `GHU-010` bounded candidate must atomically record `GHU-002`
as accepted with its exact integration/review evidence and then record
`GHU-010`'s legal transition. `GHU-010` must not start or land while the
integrated ledger silently remains at the pre-integration snapshot. The same
handoff rule applies to every later ticket so the durable ledger cannot drift.
Current candidate statuses are not changed merely to preclaim a future parent
acceptance.

Validation is focused and classifier-selected. Broader suites run once only
when the repository classifier requires them or a shared/high-risk change
crosses a release gate; forecasting suites are not routine documentation
validation.

## Safety and terminal stops

Preserve one request, no retry, and stop at the first terminal state. There is
no second collector/browser, lock manipulation, race substitution, arbitrary
shell/path/URL/root/current-time input, direct canonical DB/history write from
the UI, manual installed-service edit, pre-closure outcome access, destructive
evidence cleanup, training, model persistence/registration/promotion, EV,
staking, betting, or public exposure. The future UI operations store is
separate from both the canonical racing DB and future Race Collection
operations DB; it does not exist until implemented.

Programme-level stops are limited to: inability to establish authoritative
source/UI/deployment identity; unresolved authority over protected control,
canonical writes, outcome access, training, promotion, or public exposure;
material provenance or temporal-leakage invalidity; a reachable duplicate
capture/arbitrary execution/protected-write/false-success path; or missing
separate authority for deployment, live proof, training, or promotion.
Ordinary review corrections are engineering work under the owner's unlimited
bounded correction-ticket authority, not owner blockers.
