# Review board

## Decision

`PROCEED` with one narrow read-only autonomous-capture handoff in the standalone
manual command. Do not modify the autonomous writer, PR #48, a unit, a timer,
the shared lock, the model, or the canonical database.

The selected design searches target-bearing finalized capture plans while the
writer lock is busy. It accepts only the unique exact-race, currently due
T-60/T-30/T-10/T-2 `APPENDED` attempt, reads its report, plan, form, and sidecar
once, compares every WIN and PLACE value against one query-only SQLite
snapshot, copies the accepted bytes into the command's private temporary
directory, regenerates features, and scores without acquiring or releasing the
writer lock.

## Options

1. **Automatic finalized-receipt polling — selected.** Preserves the one-command
   operator contract and lets a completed target append escape a longer daemon
   cycle without touching writer ownership.
2. **Explicit `--capture-report` only — safe but incomplete.** Smaller discovery
   surface, but requires the operator to locate an internal runtime path and
   does not fix the observed bounded-lock experience by itself.
3. **Synthesize a capture from `live_odds` — rejected.** Loses the capture-time
   validation and form/sidecar linkage and can violate scorer timestamp order.
4. **Modify PR #48 or the installed daemon — rejected.** Mixes this proof with
   legacy runtime, activation, deployment, and result-capable orchestration.

## Perspectives

### Architect

- Evidence: manual command lock/capture flow; autonomous final report writer;
  paired capture plan; frozen scorer single-read and hash checks.
- Finding: the existing unique report directory is already the narrow producer
  boundary; no new daemon API or persistence format is required.
- Uncertainty: runtime evidence files are writable local files, not externally
  authenticated records.
- Risk: broad recursive discovery or a post-capture refresh would introduce
  ambiguity and timestamp drift.
- Recommendation: plans first, exact target/window only, bounded roots and age,
  then read-once staging.

### Provenance/domain

- Evidence: exact Ballarat T-30/T-10 autonomous reports, plan packets, and
  `live_odds` provenance columns; `AGENTS.md` live capture claim rule.
- Finding: a trustworthy operational handoff needs both the capture report and
  the exact database group. WIN-only scorer validation is insufficient.
- Uncertainty: coordinated alteration of both local surfaces cannot be detected
  without a separate trust anchor.
- Risk: false claims of cryptographic immutability.
- Recommendation: describe the result as hash-sealed and DB-bound consistency
  at use time; compare both required markets and all provenance fields.

### Skeptic/red-team

- Evidence: scorer accepted-attempt selection; append report schema; fixed
  capture-window logic; PR #48 progress-file behavior.
- Finding: skip-only attempts, progress JSONL, report-only assertions, wrong
  window, wrong venue, mutated odds, or a newer invalid candidate must all fail
  closed.
- Uncertainty: filesystem mutation between discovery and scoring.
- Risk: TOCTOU if a source is reread.
- Recommendation: read selected inputs once and score private staged bytes;
  reject ambiguity rather than falling back to an older candidate.

### Product/value

- Evidence: issue #50 and the first manual Ballarat attempt.
- Finding: automatic target receipt reuse directly removes the observed operator
  dead end while preserving the supported one-command interface.
- Uncertainty: a fresh pre-jump race may not remain long enough for live proof.
- Risk: expanding into activation before the primitive is proven.
- Recommendation: implement and validate now; report `WAITING_FOR_FUTURE_RACE`
  if no safe live target remains.

### Validation/QA

- Evidence: existing manual-command tests, autonomous capture tests, scorer
  contract tests, and scout negative-test matrix.
- Finding: red tests must cover exact WIN/PLACE values, append metadata,
  plan/report binding, fixed-window timing, outcome/path escape, lock non-use,
  and deterministic replay.
- Uncertainty: the repository full suite has known unrelated environment and
  fixture failures.
- Risk: happy-path-only proof would miss altered receipt/DB mismatches.
- Recommendation: focused fixture integration plus full-suite comparison,
  Ruff, compile, hashes, and independent post-change review.

### Operations/runtime

- Evidence: installed PR #48 service/timer, active shared lock, unique autonomous
  report directories, and current read-only runtime state.
- Finding: the report is finalized after the append commits but before the
  broader daemon cycle releases the lock, so target scoring can safely remain a
  reader.
- Uncertainty: SQLite reader availability during unrelated writes.
- Risk: any acquire/release call on the receipt path could interfere with the
  daemon.
- Recommendation: SQLite URI `mode=ro`, `PRAGMA query_only=ON`, bounded polling,
  and explicit tests that no lock/refresh/capture callback runs on reuse.

### Repo hygiene / Git guard

- Evidence: clean sibling lane at baseline `5c235643`, validated V2 card,
  successful registry claim, guard permission, live origin/PR checks, and no
  matching duplicate.
- Finding: allowed product scope is three files; PR #45 ancestry and exact PR
  #46/#47 adoptions are already preserved by the baseline.
- Uncertainty: the generic guard could not resolve its configured canonical
  migration branch in this Greyhound repository.
- Risk: confusing that guard `DATA_MISSING` with origin drift.
- Recommendation: retain the guard warning and separately bind closeout to the
  freshly verified `origin/master` and GitHub heads.

### Chair

- Evidence: all perspectives above and the owner-approved V2 card.
- Finding: the automatic finalized-receipt design has the smallest operator
  blast radius that actually resolves the live lock outcome.
- Risk disposition: proceed only with every red-team condition encoded as a
  permanent test and with no stronger authenticity claim than the evidence can
  support.
- Final authority: root high-reasoning agent under the owner's exact task card;
  lower-tier workers supplied evidence but cannot authorize runtime/data
  boundary decisions.

## Minority objection

The architect and skeptic object that a report plus SQLite comparison is not a
cryptographic trust anchor: a coordinated local rewrite before use could make
both surfaces agree. The chair accepts this objection and narrows the claim.
This lane proves consistency at the moment of use for a fresh pre-jump
operation. Historical authentication is a separate later contract-design
problem and is not silently solved here.

## Runtime functionality proof

| Field | State before implementation |
|---|---|
| Intended output | One non-persisted full/half prediction from one exact finalized autonomous capture while the writer lock remains untouched |
| Live output location | stdout only; no prediction artifact |
| Pre-run count | query-only exact WIN/PLACE group count, to be recorded only for a future eligible race |
| Post-run count | must equal pre-run count |
| Rows/files inserted after run start | zero canonical rows; temporary files removed on exit |
| Gate status | implementation authorized; live proof pending |
| Result | `WAITING_FOR_FUTURE_RACE` until focused validation passes and a fresh receipt remains pre-jump |

## Zoom-out check

- Root problem: writer ownership and scoring consumption were coupled too
  tightly.
- Overfitting risk: avoid Ballarat-specific paths, names, counts, or timings.
- Report-loop risk: do not create another planning loop; encode the selected
  contract in code and tests.
- Broad progress: preserves the autonomous writer and makes its finalized
  evidence consumable by the supported manual tracer bullet.
- Class-based approach: exact producer receipt plus query-only consistency check
  can later support other readers, but no generic framework is added now.
- Production-readiness value: proves the standalone primitive; deployment and
  PR #48 disposition remain separately owner-bound.

## Post-implementation adjudication

The selected handoff is implemented and the board's code findings were closed:
producer finalization, strict source/venue/date/race identity, one-to-one
runner boxes, fetch and append windows, lossless odds, newer-invalid fallback,
read-once staging, query-only exact-row binding, lock non-use, and pre-seal/
pre-score window changes all have focused negative tests. The final relevant
suite passes 521 tests with one skip, and a real pre-jump receipt bound to 16
exact WIN/PLACE rows without a write.

Two independent reviewers found one remaining authority issue. The existing
fresh-feature builder queries historical `finish_position` and `placing`
columns, while this card authorizes only the exact receipt artifacts and exact
`live_odds` group. The chair therefore approves the implementation transition
but forbids the optional live score under this claim. Final operational stop:
`BLOCKED_TASK_CONTRACT`.
