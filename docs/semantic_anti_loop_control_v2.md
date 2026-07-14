# Semantic Anti-Loop Control V2 Pilot

Greyhound is the first and only pilot for Tenn Semantic Anti-Loop Control V2.
This policy governs agent work; it does not change models, databases, services,
timers, deployments, registry pointers, or production runtime.

## Run Boundary

Non-trivial diagnosis, reporting, implementation, and runtime proof require a
V2 task card, an active shared-registry claim, and portable preflight. Claim the
card first, then run task-card preflight without a topic so the newly created
card is evaluated as owned task metadata rather than as a topic-matched dirty
product file:

```bash
python3 "$HOME/tenn-semantic-anti-loop-v2-canonical/scripts/agent_job_registry.py" \
  claim docs/agent_tasks/<card>.md --repo-root .
python3 "$HOME/.codex/skills/tenn-git-guard/scripts/tenn_git_guard.py" preflight \
  --repo-root . --task-card docs/agent_tasks/<card>.md --json
```

The task card must declare only the capabilities the run may use. In
particular:

| Activity | Required capability | Does not authorize |
| --- | --- | --- |
| Read evidence | `READ` | report or data writes |
| Create report artifacts | `REPORT_WRITE` | research fitting |
| Ephemeral offline fitting | `RESEARCH_FIT` | model persistence or promotion |
| Materialize a bounded dataset | `DATASET_MATERIALIZE` | canonical DB writes |
| Change source code | `CODE_EDIT` | runtime activation |
| Persist a model artifact | `MODEL_PERSIST` | registry pointer changes |
| Write a disposable DB copy | `DB_COPY_WRITE` | canonical DB writes |
| Change the canonical DB | `CANONICAL_DB_WRITE` | service or timer changes |
| Change installed runtime state | `RUNTIME_CHANGE` | publication |
| Publish a branch or PR | `PUBLISH` | merge or deployment beyond task scope |

## Hook Enforcement

Greyhound sets `TENN_V2_REQUIRED=1` on both repo-local Codex hook paths. The
`PreToolUse` matcher `Bash|apply_patch|Edit|Write` dispatches `BeforeTool` to
the synced `$HOME/.codex` portable guard before a substantive tool can run.
The `Stop` hook dispatches to the same guard before terminal closeout. Both
commands bind `--repo-root` to `"$(git rev-parse --show-toplevel)"`, so a
subdirectory invocation cannot silently select a different repository.

When no explicit override is present, the guard selects exactly one non-stale
V2 registry record whose worktree is this repository. Before-tool admission
checks the requested operation against the task card's exact paths and
capabilities. For an active claim, Stop rechecks the full claimed diff,
declared capability use, `RUN_OUTCOME.json`, and its matching
`DECISION_ENTRY.json` candidate, then requires normal registry release. After
release, Stop validates the successful receipt; a report-free semantic stop is
accepted when preflight prevented a claim.
`TENN_AGENT_TASK_CARD` or `.tenn/active_agent_task` may be used as an optional
explicit override; the normal pilot path does not require or track an active
marker.

This is a declarative trust boundary, not an operating system sandbox. It
classifies the supplied tool payload and validates repository evidence; it
does not make an opaque executable, imported module, test plug-in, or shell
script trustworthy. A card holder must run only reviewed commands whose real
effects match the declared capabilities. The live daemon, lock, database,
model-persistence, promotion, timer, service, and publication boundaries in
`AGENTS.md` remain authoritative.

## Locked Closeout

A claimed V2 run writes two candidates in its task output directory:

- `RUN_OUTCOME.json`, describing the state and decision delta; and
- `DECISION_ENTRY.json`, describing the proposed durable decision.

The active Stop hook validates those candidates but does not publish the
decision. Registry `release` revalidates the claim identity, task-card hash,
all committed and uncommitted paths, outcome, and decision candidate. It then
reclassifies against the live shared ledger and, under the same registry lock,
appends the decision, writes a receipt, removes the active claim, and retains
the candidate as an auditable report artifact. A concurrent completion can
therefore turn a no-delta release into a terminal reuse or loop result instead
of creating another ledger row.

Direct `agent_decision_ledger.py append` is not a task-closeout mechanism. It
is reserved for an unclaimed, bounded seed import, requires
`--authorize-unclaimed-seed`, and rejects a matching active claim. An example
is intentionally labelled as seed-only:

```bash
python3 "$HOME/tenn-semantic-anti-loop-v2-canonical/scripts/agent_decision_ledger.py" \
  append --repo-root . --entry-file <reviewed-seed-entry.json> \
  --authorize-unclaimed-seed
```

## Stable Seed Boundary

The committed seed contains exactly four hash-bound historical decisions:

1. TheDogs published historical market source cleared its 300-race floor with
   663 complete races at the recorded snapshot.
2. The aggregate historical challenger remained `KEEP_BASELINE` on that
   recorded evaluation.
3. The strict Sportsbet same-floor comparison was `DATA_MISSING` for its
   recorded snapshot and blocks only its declared comparison/promotion lane.
4. The reviewed historical identity bridge was report-only ready while
   canonical copy repair remained blocked.

The seed deliberately excludes timer state, current capture counts,
prospective counts, service state, and other volatile runtime claims. Those
require fresh evidence. Build or verify the seed with:

```bash
python3 scripts/build_semantic_anti_loop_seed.py \
  --floor-summary <recorded-summary.json> \
  --evaluation-results <recorded-evaluation.json> \
  --strict-overlap <recorded-overlap.csv> \
  --bridge-proof <recorded-proof.json> \
  --output docs/agent_decisions/greyhound_semantic_anti_loop_seed_v1.jsonl
```

Use `--check` after generation to prove the tracked JSONL is unchanged. The
four approved seed decisions already exist once each in the shared ledger, so
the enforcement correction validates the ledger and searches those decision
IDs; it does not append or replay the seed:

```bash
python3 "$HOME/tenn-semantic-anti-loop-v2-canonical/scripts/agent_decision_ledger.py" \
  validate --repo-root .
python3 "$HOME/tenn-semantic-anti-loop-v2-canonical/scripts/agent_decision_ledger.py" \
  search --repo-root . --decision-id greyhound-thedogs-floor-663-20260709
```

Repeat the read-only search for the other three manifest decision IDs when
auditing seed identity. A second append is a defect, not an update.

## First-Five Review Result

The post-pilot review considered five distinct non-trivial scopes. Each had a
different scope fingerprint and materially different transition. None was
falsely stopped as a duplicate or loop:

| Distinct scope | Fingerprint prefix | False duplicate block |
| --- | --- | --- |
| Strict V4 database-copy canary | `6dfef6538738` | No |
| Strict V4 exact-commit lineage replay | `d7205b754f0e` | No |
| Strict V4 prospective runtime preparation | `d43f6fb76489` | No |
| Strict V4 effective database-state binding | `3805d972a3d9` | No |
| Strict V4 disabled replacement install | `62b000e01c68` | No |

Result: `FIRST_FIVE_REVIEW_PASSED` with five distinct scopes and zero false
duplicate blocks.

The append-only ledger retains two legacy duplicate chains: the copy-canary
scope has the original and `-closeout` decision rows, and the exact-commit
lineage scope has the original and `-reclaim` rows. They are not seed entries,
are not deleted or rewritten, and are not counted as extra distinct scopes.
They preserve the motivating defect: an older reclaim/closeout path could
publish the same semantic decision twice. Release-owned append under lock is
the forward fix.

After this correction merges, Greyhound's state is
`GREYHOUND_PILOT_ENFORCED`. Greyhound remains the first and only pilot;
adoption in any other repository is a separate reviewed decision, not an
automatic consequence of the first-five result.
