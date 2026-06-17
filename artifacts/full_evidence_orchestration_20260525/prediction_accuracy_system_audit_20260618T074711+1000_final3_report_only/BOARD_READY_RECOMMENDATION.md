# Board-Ready Recommendation

Decision: `DATA_JOIN_REPAIR_NEXT`.

Runtime: `SHADOW_SCORER_RUNTIME_REPAIRED`.

Top blockers:
- `market_and_promotion_gate`: `PROMOTION_DISTANCE_BLOCKED` - run report-only model tournament on identical label/odds/prediction rows
- `aggregate_model_quality`: `PARTIAL_AGGREGATE_PENDING_MORE_RESULTS` - compare current model, market-only, blends, and candidate feature sets on one joined table
- `current_result_label_join`: `DATA_MISSING` - wait for official results or run safe result join when races have resulted
- `strict_odds_join`: `DATA_MISSING` - repair odds join only if rows exist but fail exact race/box/name matching
- `inactive_train_all_missing_features`: `BLOCKED` - backfill or retrain only after provenance-safe coverage and report-only ablation pass

Recommended next action: run a report-only model tournament/evaluation contract on identical rows before any activation, promotion, EV, betting, registry, DB, or label mutation.
