# Race Evidence Inventory

Final status: `RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION`
Recommended decision: `RUN_POST_BACKLOG_UNIFIED_EVALUATION`

- Race union count: `1306`
- Shadow prediction races: `1126`
- Official-result artifact races: `530`
- Official-result evidence DB races: `526`
- Live odds races: `1225`
- Strict pre-jump odds races: `1176`
- Shadow races with official-result evidence DB: `514`
- Shadow races with strict pre-jump odds: `1046`
- Shadow races complete for official result and strict odds: `443`
- Scorecard evaluation races: `443`
- Model Top1 / Top3: `0.22799097065462753` / `0.5778781038374717`
- Market Top1 / Top3: `0.44469525959367945` / `0.7923250564334086`
- Model mean winner rank: `3.4243792325056432`
- Market mean winner rank: `2.3769751693002257`

## Action Counts

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

## No-Write Guarantees

```json
{
  "active_model_replacement": false,
  "betting_or_ev_action": false,
  "daemon_control": false,
  "db_write": false,
  "label_write": false,
  "manifest_rewrite": false,
  "odds_write": false,
  "official_result_write": false,
  "production_pointer_update": false,
  "production_promotion": false,
  "registry_mutation": false,
  "snapshot_rewrite": false,
  "training": false
}
```
