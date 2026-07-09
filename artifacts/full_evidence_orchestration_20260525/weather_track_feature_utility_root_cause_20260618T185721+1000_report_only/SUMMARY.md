# Weather/Track Feature Utility Root Cause

- Final status: `KEEP_COLLECTING_ONLY_DATA_MISSING`
- Ablation status: `NOT_RUN_TRAIN_HOLDOUT_OR_UTILITY_EVIDENCE_MISSING`
- Reason: `source-safe weather/track evidence exists, but train/holdout coverage or non-flat utility evidence is not sufficient`
- Feature rows scanned: `53415`
- Sidecars scanned: `14769`
- Accepted both-weather-track races: `120`
- Accepted both-weather-track runner-row pct: `0.13795278852763104`
- Train/holdout split evidence available: `False`
- Protected paths unchanged: `True`

No production promotion, registry mutation, DB writes, label writes, schema mutation, EV output, betting output, daemon control, or model training was performed.
