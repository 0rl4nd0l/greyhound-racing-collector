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
python3 "$HOME/.agents/skills/tenn-git-guard/scripts/tenn_git_guard.py" preflight \
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

The Stop hook dispatches through the installed Tenn V2 control plane. When no
explicit override is present, it selects exactly one non-stale V2 registry
record whose worktree is this repository, then validates the task-card diff,
`RUN_OUTCOME.json`, declared capability use, and matching append-only decision
entry. `TENN_AGENT_TASK_CARD` or `.tenn/active_agent_task` may be used as an
optional explicit override; the normal pilot path does not require either and
does not track an active marker.

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

Use `--check` after generation to prove the tracked JSONL is unchanged. Append
the four entries to the shared decision ledger only after the pilot PR is
merged and the manifest matches its reviewed artifacts.

## First-Five Review Gate

Broader adoption is not authorized until five post-merge non-trivial Greyhound
runs have been reviewed for false duplicate or loop blocks. Record each review
against the durable run outcome and decision search, not conversational claims.

| Run | State | Required review |
| --- | --- | --- |
| 1 | `PENDING` | classification, fingerprint, reused evidence, false-block check |
| 2 | `PENDING` | classification, fingerprint, reused evidence, false-block check |
| 3 | `PENDING` | classification, fingerprint, reused evidence, false-block check |
| 4 | `PENDING` | classification, fingerprint, reused evidence, false-block check |
| 5 | `PENDING` | classification, fingerprint, reused evidence, false-block check |

Until all five are reviewed, Greyhound remains the sole pilot and the gate is
`PILOT_PENDING_FIRST_FIVE`, not complete rollout.
