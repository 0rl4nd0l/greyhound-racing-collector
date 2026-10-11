# Independent Spec and methodology review, 11 October 2026

Reviewed `git diff 27fe8945...b68b05c8`, then implementation and protocol amendments through `6c1aca4da9a9c28613b4867dfd84102cf49ceb77`. Specification: the user's research-orchestrator request and `dynamic_ability_plan_20261011.md`. Later documentation/publication commits require the root's final consistency check.

**Open findings: zero critical, high or medium findings. One P2 scope finding was repaired.** The user asked to investigate “replacing some historical averages”; the original `run_dynamic_ability_experiment.py:25` only added six summaries. Amendment `3e5ae134`, implemented at lines 26–34 and 148–155, records a fixed six-for-six replacement before its three fits. It preserves all additive negatives and stops without subset/parameter tuning. This is an explicitly disclosed post-review scope repair, not an untouched preregistration. Both experiments support rejection of these tested formulations, not dynamic ability generally.

Independent numerical checks:

- Verified 159 unique artifact hashes across both dynamic runs and three benchmark runs. Membership remains 720/656/975, with all evaluated fields matched. The 169 reserved allocations were inspected only as manifest metadata; no reserved labels were decoded.
- Independently reconstructed all 16,855 runner states across three arms: 303,390 field comparisons match exactly. Every date's forecasts precede its updates; opponent estimates are never revised backwards. Recomputed all six models' preprocessing from only 720 training races/5,204 rows: medians, means and standard deviations match exactly.
- Independently recalculated all reported development/later log losses, Brier scores and tied-top credit; maximum error `3.56e-15`. Replayed 77,182 unique held-out meta probabilities from saved coefficients; maximum error `3.34e-16`. All meta training dates precede prediction dates; development-fitted tree temperature never enters meta training. Six predictor fits and 49 scalar fits are retained, with no fresh-test relabelling.
- Replacement dynamic log loss is **1.810945**, versus recency **1.813093** and frozen baseline **1.750911**. Its SP correction scores **1.503206**, versus calibrated SP **1.503299**; this negligible exploratory difference does not establish added value. The negative decision is supported.
- Thirty focused tests pass. No research model was fitted during this review.

Source-reported SP remains a retrospective price proxy, with no executable cutoff; field/context reconstruction, sparse histories and ten later dates limit inference. Early-contest and movement hypotheses correctly remain insufficient evidence, not rejected experiments.

Independent scripts and JSON audits: `/mnt/tenn-nvme2/tenn/greyhound-dynamic-research-20261011-evidence/methodology-review-01/` and `methodology-review-replacement-01/` (`recompute.py`, `check_preprocessing.py`, `check_states.py` and corresponding audit JSON).
