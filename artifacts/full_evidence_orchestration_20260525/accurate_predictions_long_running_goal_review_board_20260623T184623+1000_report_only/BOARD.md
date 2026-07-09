# Accurate Predictions Long-Running Goal Review Board

Generated: 2026-06-23T18:46:23+10:00

Mode: report-only board and next-goal packaging. No DB mutation, runtime
mutation, training, promotion, EV, betting, snapshot rewrite, registry mutation,
or weakening of identity/source/official-result/pre-jump timing gates was
performed by this board.

## Evidence Inspected

- Runtime checkout:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621`
- Evidence root:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`
- Latest lineage audit:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/rolling_lineage_audit_20260623T183757+1000_report_only`
- Latest completed full daemon child selected by lineage audit:
  `20260623T181701+1000_daemon`
- Latest available rolling packet in the audit:
  `rolling_model_comparison_20260623T181701+1000_daemon_rejoin`
- Historical max rolling packet in the audit:
  `rolling_model_comparison_20260614T000205+1000_daemon_rejoin`
- High accuracy packet:
  `high_accuracy_refinement_packet_20260623T181701+1000_daemon_autopilot_post_promotion_distance`

## Current Facts

- Safe joined races: `1033`
- DB live-odds distinct races: `1396`
- DB official-result evidence distinct races: `538`
- Safe joined races with eligible unified evidence somewhere: `986`
- Latest rolling sample races: `35`
- Latest rolling source unified reports: `29`
- Historical max rolling sample races: `178`
- Historical max rolling source unified reports: `508`
- Safe joined races not requested by latest rolling source set: `999`
- Promotion remains blocked by the current packet.

## Architect Perspective

Evidence inspected: lineage audit, rolling packet summary, high accuracy packet.

Finding: the system has enough historical/current evidence to make progress, but
the current rolling source-discovery path is narrowed. A long-running goal should
first restore broad cumulative rolling inclusion, then run the promotion-quality
gates on the restored denominator.

Uncertainty: the exact source-discovery regression is not proven in this board;
it must be diagnosed in code before patching.

Risk: fixing downstream model gates before fixing rolling inclusion can optimize
against a tiny, misleading sample.

Recommended action: proceed with bounded implementation, starting with rolling
source inclusion.

## Skeptic / Red-Team Perspective

Evidence inspected: promotion blockers and lineage drop buckets.

Finding: even after source inclusion is fixed, promotion is not automatically
safe. The best non-market candidate still fails rank-first safety and worsens
some market-relative metrics. Gate contracts also report `DATA_MISSING`.

Uncertainty: current candidate quality may change after the full eligible source
set is restored.

Risk: pressure to "push accuracy" could lead to gate weakening, promotion from
insufficient sample, or overfitting to recent races.

Recommended action: explicitly forbid gate weakening and require fresh
out-of-sample report-only validation before any promotion.

## Product / Value Perspective

Evidence inspected: user objective and current blockers.

Finding: the highest value is to get back to a truthful 100+ race rolling
evaluation, then identify whether the model is genuinely learning signal beyond
the market. More raw collection alone is lower value until the rolling source
set can use the evidence already present.

Uncertainty: if the restored 100+ packet still shows market dominance, the next
value step becomes model/feature work rather than collection.

Risk: repeated report-only loops without implementation will not advance the
prediction objective.

Recommended action: deploy implementation workers against source inclusion,
gate diagnostics, and evidence-gap prioritization.

## Validation / Test Perspective

Evidence inspected: latest sample counts, historical sample counts, promotion
blockers.

Finding: validation must prove raw/captured -> candidate -> accepted ->
evaluated -> reported lineage. Success is not "more rows exist"; success is
`100+` eligible races in rolling comparison plus all gate contracts passing.

Uncertainty: existing tests for rolling source discovery may be incomplete.

Risk: a patch that merely points at more directories may double count races,
include stale odds-only wrappers, or include races without strict pre-jump odds.

Recommended action: add regression tests for source discovery, dedupe, and
eligibility preservation before claiming readiness.

## Repo Hygiene / Git Guard Perspective

Evidence inspected: `tenn-git-guard` preflight.

Finding: guard support and registry pass, but live and committed ledgers are
`DATA_MISSING`. Existing dirt is extensive and must be treated as in-scope
context, not unrelated noise.

Uncertainty: ledger absence prevents definitive duplicate-work exclusion.

Risk: implementation workers can conflict if they edit the same runtime scripts.

Recommended action: split subagents by disjoint ownership and require each to
report touched files, tests, and unresolved conflicts.

## Domain Perspective

Evidence inspected: promotion metrics and gate contracts.

Finding: accurate greyhound prediction must beat the market on a declared,
source-safe slice, not merely improve Top1 on a tiny sample. Strict pre-jump
odds, official result provenance, runner identity, and timing gates are
non-negotiable because they prevent leakage and false confidence.

Uncertainty: the model may need feature/candidate work after the denominator is
fixed.

Risk: using post-jump odds, weak official results, or broad raw DB counts would
produce fake accuracy.

Recommended action: preserve all gates; use the restored sample to decide the
next model-improvement slice.

## Chair Decision

Decision: `proceed`.

Proceed with a long-running remediation goal, implemented by bounded workers.
The first production-readiness value is restoring broad rolling source inclusion
and proving the current eligible evidence can reach the review denominator. The
second value is making gate failures actionable without weakening them. The
third value is prioritizing remaining evidence gaps only after the current
source-set bug is understood.

Promotion remains blocked until `100+` eligible rolling races and all gate
contracts pass on fresh report-only evidence.
