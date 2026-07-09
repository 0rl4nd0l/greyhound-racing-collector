# Greyhound Runtime Accuracy Review Board

Generated: 2026-06-22T10:44:19+10:00

## Scope

Continue the runtime and extraction-accuracy handoff from
`/tmp/greyhound_runtime_accuracy_handoff_20260622T1024+1000.md`.

Decision question: pick one bounded next implementation that improves the
accuracy investigation without training, promotion, betting, DB writes, daemon
control, registry mutation, or snapshot rewrites.

## Evidence Inspected

- Runtime worktree:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621`
- Branch: `codex/runtime-master-live-20260621`
- HEAD: `ab902f3ceee1bd18dcf9d00a16dafcf1ed6958b8`
- Upstream/base: `origin/master`
- Merge base: `fa58192d9692a15d7ebe77854b7c7b0ff78d4423`
- Remote `master` from `git ls-remote`: `fa58192d9692a15d7ebe77854b7c7b0ff78d4423`
- Worktree status before board output: clean, branch ahead of `origin/master`
  by 3 commits.
- Systemd odds-capture service workdir:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621`
- Systemd odds-capture runtime Python:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python`
- Service state at inspection: `ActiveState=activating`, `SubState=start`,
  `Result=success`, start timestamp `Mon 2026-06-22 10:41:49 AEST`.
- Timer state at inspection: `ActiveState=active`, `SubState=running`, last
  trigger `Mon 2026-06-22 10:41:49 AEST`.
- Latest complete odds report inspected:
  `autonomous_live_odds_capture_20260622T104038+1000_odds_capture_autopilot_autopilot/autonomous_live_odds_capture_report.json`
  with `status=READY`, `inserted_live_odds_rows=0`, `blocked_attempts=[]`.
- DB `live_odds` at inspection: `23052` rows, latest capture timestamp
  `2026-06-22T10:18:01.871635+10:00`.
- Race evidence inventory:
  `artifacts/full_evidence_orchestration_20260525/race_evidence_inventory_20260621T232643+1000_report_only/SUMMARY.md`

## Git Guard

`tenn-git-guard` registry support is unavailable in this repo:

- `scripts/agent_job_registry.py`: missing
- `scripts/agent_task_ledger.py`: missing

Fallback checks:

- GitHub PR search for runtime/accuracy/official-result/strict-odds terms:
  no matching PRs.
- GitHub issue search for runtime/accuracy/official-result/strict-odds terms:
  no matching issues.
- Worktree list shows many greyhound sibling worktrees, including older runtime
  and evidence lanes. None supersedes this exact handoff because the installed
  service currently points at this worktree.
- Docs/report search found related odds-capture daemon packet and race evidence
  inventory docs, but no active duplicate task card.

Guard decision: `warning`. Proceed only with a narrow report-only code/docs/test
change that does not touch daemon services, DB, registry, training, promotion, or
betting paths.

## Current Accuracy Denominator

From the race evidence inventory:

- Race union count: `1306`
- Shadow prediction races: `1126`
- Official-result artifact races: `530`
- Official-result evidence DB races: `526`
- Live odds races: `1225`
- Strict pre-jump odds races: `1176`
- Complete shadow/official/strict-odds scorecard races: `443`
- Model Top1 / Top3: `0.22799097065462753` / `0.5778781038374717`
- Market Top1 / Top3: `0.44469525959367945` / `0.7923250564334086`

Existing action counts:

```json
{
  "append_official_result_evidence_backlog": 4,
  "capture_official_result": 608,
  "collect_future_strict_prejump_odds": 16,
  "not_shadow_scored": 180,
  "ready_for_unified_evidence_evaluation": 443,
  "repair_official_result_runner_set_or_identity_join": 55
}
```

Existing scorecard skip counts:

```json
{
  "official_result_incomplete_for_shadow_boxes": 667,
  "shadow_predictions_missing": 180,
  "strict_odds_incomplete_for_shadow_boxes": 16
}
```

## Perspectives

### Architect

Evidence inspected: inventory builder, inventory report, AGENTS runtime guide,
systemd service command.

Finding: the real accuracy blocker is not one parser edge. The dominant
denominator loss is official-result coverage for shadow races, but the
scorecard currently compresses `608 + 55 + 4` official-result next actions into
one generic skip reason. That makes the next repair class less obvious than it
should be.

Recommended action: extend the report-only inventory to emit actionable
scorecard gap counts derived from the same row-level `recommended_next_action`
classification.

Uncertainty: this will not itself repair official-result capture or identity
joins; it only makes the next high-value repair class auditable.

### Skeptic

Evidence inspected: scorecard skip logic and action-count logic.

Finding: another report-only artifact can become a loop if it does not change
the next implementation choice. The value here is acceptable only if the output
is directly used to pick between official-result capture, backlog append,
runner-set identity repair, and strict-odds repair.

Recommended action: proceed only if the change is wired into the existing packet
and verified by a fresh full inventory run. Do not create a separate orphan
report format.

Risk: adding a second taxonomy could confuse operators if it diverges from
`recommended_next_action`.

### Product/Value

Evidence inspected: action counts and model-vs-market scorecard.

Finding: official-result capture and official-result runner-set repair have much
higher production-readiness value than odds collection, because strict odds are
missing for only `16` shadow races while official-result completeness blocks
`667` scorecard races.

Recommended action: make the inventory tell the user that ordering explicitly,
then use the next session for the highest-count class.

### Validation/Test

Evidence inspected: `tests/test_build_race_evidence_inventory_packet.py` and
`scripts/build_race_evidence_inventory_packet.py`.

Finding: there is a clean test seam. Existing fixtures already exercise
`ready_for_unified_evidence_evaluation` and
`append_official_result_evidence_backlog`. The test should add races for
`capture_official_result`, `repair_official_result_runner_set_or_identity_join`,
and `collect_future_strict_prejump_odds`, then assert the new gap taxonomy.

Recommended action: regression-test the metric shape before the final report
run.

### Repo Hygiene/Git Guard

Evidence inspected: git status, remote state, worktrees, GitHub PR/issue search,
missing Tenn registry scripts.

Finding: the branch is local and ahead of `origin/master` by three commits. That
is an owner/release risk, but not a blocker for a narrow local report-only
change in the installed runtime worktree. Registry/ledger evidence is
`DATA_MISSING`; fallback duplicate checks did not find a current duplicate.

Recommended action: keep the diff small, do not push without approval, and do
not touch unrelated worktrees or generated service files.

### Domain

Evidence inspected: AGENTS guide, inventory counts, scorecard metrics.

Finding: model accuracy cannot be claimed as improved while the model trails the
market on the 443-race complete slice. The safest path is to improve evidence
coverage and failure-class visibility before model work.

Recommended action: do not train, promote, emit EV, or place bets. Focus on
official-result denominator growth first.

## Chair Decision

Decision: `proceed`.

Bounded next goal:

Extend `scripts/build_race_evidence_inventory_packet.py` so the scorecard
metrics include an actionable gap taxonomy tied to each race's
`recommended_next_action`, then update tests/docs and run a fresh report-only
inventory packet.

Why this is the right next action:

- It addresses the root planning problem exposed by the handoff: repeated narrow
  fixes without a class-level denominator view.
- It is report-only and does not touch runtime service execution or the DB.
- It uses the existing inventory packet instead of introducing a disconnected
  artifact.
- It should make the next repair choice mechanically visible: official-result
  capture backlog versus runner-set/identity repair versus strict-odds repair.

Do not proceed beyond this bounded goal without a fresh board or owner
instruction.
