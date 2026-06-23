# Fresh Session Long-Running Goal

Use this `/goal` in a fresh session:

```text
/goal
Work in /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621.

Read these artifacts first:
- artifacts/full_evidence_orchestration_20260525/accurate_predictions_long_running_goal_review_board_20260623T184623+1000_report_only/BOARD.md
- artifacts/full_evidence_orchestration_20260525/accurate_predictions_long_running_goal_review_board_20260623T184623+1000_report_only/BOARD_DECISION.json
- artifacts/full_evidence_orchestration_20260525/accurate_predictions_long_running_goal_review_board_20260623T184623+1000_report_only/NEXT_GOAL.md
- /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/rolling_lineage_audit_20260623T183757+1000_report_only/SUMMARY.md
- /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/rolling_lineage_audit_20260623T183757+1000_report_only/rolling_lineage_audit_report.json

Act as an orchestrator of bounded implementation subagents. Start with
tenn-git-guard preflight. If ledger scripts/data are missing, record
DATA_MISSING and continue with fallback git/artifact checks. Treat existing
dirt as in-scope context; do not revert it.

Mission:
Remediate as many Greyhound prediction-readiness blockers as possible while
preserving source-safety. Push toward accurate predictions by restoring a
truthful 100+ race rolling comparison first, then using that denominator to
decide whether candidate model work can beat the market.

Hard boundaries:
- No DB mutation without explicit owner approval.
- No live runtime/service mutation without explicit owner approval.
- No training, promotion, EV, betting, snapshot rewrite, registry mutation, or
  production pointer update.
- Do not weaken identity, source, official-result, strict pre-jump timing,
  dual-baseline market-rank safety, or promotion PR gates.
- Validation may write report-only artifacts and tests.

Known current facts:
- Latest completed full daemon child from the lineage audit:
  20260623T181701+1000_daemon.
- Latest rolling packet in the audit:
  rolling_model_comparison_20260623T181701+1000_daemon_rejoin.
- Historical max rolling packet:
  rolling_model_comparison_20260614T000205+1000_daemon_rejoin.
- Safe joined races: 1033.
- DB live-odds distinct races: 1396.
- DB official-result evidence distinct races: 538.
- Safe joined races with eligible unified evidence somewhere: 986.
- Latest rolling sample races: 35.
- Latest rolling source unified reports: 29.
- Historical max rolling sample races: 178.
- Historical max rolling source unified reports: 508.
- Main lineage drop: 999 safe joined races are not requested by the latest
  rolling source set.

Milestones:
M0. Fresh preflight and latest evidence selection.
    Select the latest completed full daemon child. Ignore odds-only and
    lock-wait wrappers for accuracy report-only validation. Reconfirm the latest
    rolling/rejoin packet and historical 100+ packet.

M1. Rolling source inclusion root cause.
    Diagnose why current rolling source discovery uses 29 source reports while
    historical ready packets used 508. Identify the exact code path, filter,
    directory scope, manifest field, or wrapper-selection rule causing the
    narrowed denominator.

M2. Rolling source inclusion repair.
    Patch the rolling comparison source discovery so it can include cumulative
    eligible unified evidence from the retained evidence root, while preserving:
    dedupe by race ID, official-result provenance, strict pre-jump odds,
    Stage 2 prediction availability, and wrapper exclusions.

M3. 100+ eligible rolling validation.
    Run focused tests and a report-only rolling validation. Success is at least
    100 eligible rolling races. If still below 100, produce an exact race-level
    gap list classified by source-set missing, official-result missing,
    strict-prejump-odds missing, Stage 2 missing, identity mismatch, or other
    gate.

M4. Gate-contract completeness.
    Fix promotion-distance/high-accuracy diagnostics so gate contract audit and
    policy blockers are explicit and not ambiguous DATA_MISSING unless evidence
    is truly unavailable. Do not weaken gates.

M5. Model-quality decision on restored denominator.
    Once 100+ eligible races exist, compare candidates against market and
    baseline on the declared rank-first gates. If no candidate passes, produce a
    concrete model-improvement plan with feature/candidate hypotheses, tests,
    and no promotion.

M6. Evidence gap remediation plan.
    Only after M1-M3, prioritize any remaining official-result or strict
    pre-jump odds collection gaps from the rolling lineage, not raw DB counts.
    Keep collection/appends approval-gated.

M7. Final readiness gate.
    Stop with promotion still blocked unless all are true:
    - 100+ eligible rolling races
    - official-result and pre-jump timing gates pass
    - candidate beats market/baseline under declared gates
    - gate-contract audit/policy pass
    - promotion PR boundary allows owner-reviewed promotion

Subagent deployment guidance:
1. rolling-source-inclusion-worker owns rolling source discovery and tests.
2. gate-contract-worker owns promotion-distance/high-accuracy diagnostics and
   tests.
3. evidence-gap-prioritization-worker owns lineage/gap reporting improvements
   and tests.

Each worker must report touched files, tests run, blockers, and whether any
existing dirty file constrained their patch. Integrate only coherent, reviewed
changes.
```
