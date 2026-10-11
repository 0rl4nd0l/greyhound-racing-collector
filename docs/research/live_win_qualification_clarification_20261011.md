# WIN qualification clarification before reconciliation scoring

The original PR #231 protocol and its hash remain unchanged. The current user explicitly requested an independent check of whether each incomplete-result requirement is necessary for the specified WIN benchmark.

The result worker and independent reviewer identified two retained complete eight-runner official fields with one unambiguous winner and tied **losing** positions. The original join implementation required each finishing position to equal its index in `box_order`. That rejects a seventh-place tie or fifth-place tie despite leaving the WIN target unambiguous.

For this continuation, admit an exact unchanged original runner field when every original runner has an authenticated official terminal disposition and exactly one runner is the winner. Preserve tied losing positions exactly as reported; do not invent a total order. A documented DNF/FELL/DISQ can establish nonwinner status without a fabricated rank. Contradictory winners, winning dead heats, unresolved runner identity/status, scratches and substitutions retain the original exclusions. Original model probabilities and complete decision-time market remain unchanged.

The original phrase “confirmed dead heats are diagnostic only” was broader than necessary if interpreted as excluding ties for losing places. Its fractional-winner-target context supports a WIN-specific interpretation, but this continuation records the clarification explicitly rather than claiming the old implementation passed these races. It is a repair of an overbroad finish-order check, approved within the current user's requested requirement review. No probability scores for the two newly joined races were inspected before this decision.

Membership, prediction selection, scores, calibration bins, uncertainty method, selector thresholds and selected/pass identities are unchanged. A new forecast or a reconstructed active field cannot satisfy this clarification.
