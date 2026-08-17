# Sportsbet + Betfair scheduled-off consensus freeze

Terminal state: `CONSENSUS_CANDIDATE_FROZEN`.
Prospective test: `READY_TO_FREEZE`.

## Population

- Strict joined surface: 1008 races / 7142 runners.
- Fit: 586 races / 4189 runners.
- Validation: 422 races / 2953 runners.
- August 2026 rows read or scored: 0.

## Frozen rule

- Betfair weight: 0.95.
- Sportsbet weight: 0.05.
- Both sources are normalized within race before the convex combination.
- Selection used fit log loss only; validation screened that one weight once.

## Validation

- Sportsbet log loss: 1.436572003.
- Betfair-only log loss: 1.398929678.
- Consensus log loss: 1.399038567.
- Consensus - Sportsbet log-loss delta: -0.037533436 (cluster bootstrap 95% CI -0.059840650 to -0.013857043).
- Consensus - Sportsbet Brier delta: -0.013432191 (95% CI -0.022933114 to -0.003357325).
- Consensus top-1/top-2/top-3: 0.478673 / 0.677725 / 0.810427.
- Consensus mean winner rank: 2.199052.

## Claim boundary

This is development screening evidence only. It does not confirm prospective edge, profitability,
betting value, promotion readiness, deployment readiness, or live scoring readiness. The fixed
2026-08-18 through 2026-09-30 cohort has not been ingested, labelled, or scored.
