# Greyhound Operator UI V1 decision log

Append-only: corrections add a superseding entry; they do not rewrite history.
Each entry records date, ID, context, decision, evidence, consequences, and
supersession.

## 2026-07-30 — DEC-GHU-000-AUTHORITY

- Context: R0 repository/UI/runtime audit.
- Decision: Authoritative UI is Flask/Jinja/static in `app.py`, `templates/`,
  and `static/`; FastAPI is secondary; `frontend/` and TGR are stale. No primary
  authn/authz/CSRF or generated UI service path is proven.
- Evidence: accepted `GHU-000`; exact source paths are in `CONTRACTS.md`.
- Consequences: extend the authoritative surface; fail closed on missing
  security/deployment capability.
- Supersession: none.

## 2026-07-30 — DEC-GHU-000A-CAPTURE-ONE

- Context: source delta after upstream PRs #79/#80.
- Decision: with no reusable receipt, the predictor synchronously starts zero
  or one collector-owned `capture-one` child. There is no interactive timer
  transport, retry, wait, lock steal, or race substitution; terminal handling
  is immediate. Contract budgets and ordering are authoritative.
- Evidence: accepted run `20260730T092421Z-6f4fba42c4-baf67b`, session
  `019fb257-217f-7d10-8409-b2a06a6bd20b`; `CONTRACTS.md`.
- Consequences: R3 tests bind synchronous records and preserve one-request/no-
  retry/stop-first-terminal.
- Supersession: refines DEC-GHU-000-AUTHORITY after #79/#80.

## 2026-07-30 — DEC-GHU-001-REJECTED-EVIDENCE

- Context: rejected contract candidates.
- Decision: rejected candidates remain provenance evidence and are superseded;
  they are not integrated product changes.
- Evidence: review history below.
- Consequences: corrections never silently overwrite or regress accepted work.
- Supersession: none.

## 2026-07-30 — DEC-GHU-001-ORIGINAL-REJECTED

- Context: original run `20260730T075157Z-9be52ecd58-7541d9`, implementer
  `019fb214-6373-7a32-8fae-3f67b1d4ab8c`.
- Decision: rejected for incorrect Level 1–4 authority mapping and vague
  freshness.
- Evidence: reviewer session `794f6fd7bead16cbbb06a94efa9cf4e5`;
  candidate/diff hashes `DATA_MISSING`.
- Consequences: no integration; smallest repair required.
- Supersession: superseded by the repaired successor.

## 2026-07-30 — DEC-GHU-001-REPAIR-REJECTED

- Context: repaired run `20260730T082719Z-9be52ecd58-5c31e1`.
- Decision: rejected for conflating the two collector lanes, omitting explicit
  R2 read-access audit, and leaving unused freshness term D.
- Evidence: implementer/reviewer sessions and hashes `DATA_MISSING`.
- Consequences: no integration; collector lanes and access audit became
  explicit contract requirements.
- Supersession: superseded by GHU-001D.

## 2026-07-30 — DEC-GHU-001D-SOURCE-STALE

- Context: GHU-001D was accepted before #79/#80.
- Decision: preserve it as accepted evidence, but supersede it as source-stale.
- Evidence: commit `6c19b1709e23b21c2b2b66e599e334745a6b1ff3`;
  other session/hash fields `DATA_MISSING`.
- Consequences: current integration must use the post-#79/#80 contract.
- Supersession: superseded by current GHU-001.

## 2026-07-30 — DEC-GHU-001E-LIVE-PROOF-REJECTED

- Context: run `20260730T093002Z-6f4fba42c4-dd989f`, implementer
  `019fb25c-66af-7d50-8b5f-3415a0dd5b42`.
- Decision: #79/#80 were correctly captured, but live proof wrongly required
  mutually exclusive branches; reviewer
  `019fb263-3127-7d03-b951-c55015acb7fb` rejected only that defect.
- Evidence: candidate SHA-256
  `baa3a2bff7e1d81930d9c76726f5455c556c23787415af0042a731c103a9970d`;
  diff SHA-256
  `39d98da03c067c552133aaaef92ed280b73c5e847495dc987a6710030f3e4e10`.
- Consequences: live proof must claim only its naturally reached branch.
- Supersession: superseded by GHU-001F.

## 2026-07-30 — DEC-GHU-001F-ACCEPTED

- Context: GHU-001F changed only the live-proof defect.
- Decision: independently accepted and integrated as current contract.
- Evidence: commit `aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, parent
  `6f4fba42c45c73702efb017a21cbd284b44c1d04`; file SHA-256
  `b2b4af016b24dafa4f121f0415c0d948a4c0699c73586a235c6547fc3002512b`;
  diff SHA-256
  `22df513812a6460fbae2f247a36bfca4f37536438d5eed6eabebda47b2774b30`;
  implementer `019fb267-ff74-71e3-bb91-2ed8d55be316`; reviewer
  `019fb26b-ae76-7411-91ae-b54d5514a137`.
- Consequences: this is current contract authority.
- Supersession: supersedes GHU-001D and GHU-001E.

## 2026-07-30 — DEC-GHU-PROGRAMME-AUTHORITY

- Context: execution governance.
- Decision: approved programme is authority; contracts, dependencies, and
  tickets may be refined autonomously inside it. Owner's unlimited bounded
  correction-ticket authority supersedes the seed's single-retry limit.
- Evidence: approved plan and current accepted contract.
- Consequences: ordinary review findings are engineering work, not owner
  escalation; parent retains acceptance/integration authority.
- Supersession: supersedes the seed's old procedural retry limit.

## 2026-07-30 — DEC-GHU-R3-PROOF-SPLIT

- Context: synthetic safety and live proof have different evidence scopes.
- Decision: GHU-035 proves fixture/synthetic success and every terminal blocker;
  GHU-037 runs one natural cycle and one exact live job and claims only the path
  naturally reached.
- Evidence: `CONTRACTS.md` release/proof gates.
- Consequences: live proof never induces mutually exclusive branches or retries.
- Supersession: incorporates the GHU-001E correction.

## 2026-07-30 — DEC-GHU-R5-DEFERRED

- Context: programme boundary.
- Decision: R5 training, experiment execution, model persistence/registration,
  activation, and promotion remain deliberately deferred.
- Evidence: approved programme and `CONTRACTS.md`.
- Consequences: R4 produces only non-executing specifications; future execution
  requires a new contract and separate authority.
- Supersession: none.
