# Same-Distance Feature Repair

## Executive Summary

This is a report-only packet repair. It preserves the clean official holdout and only repairs same-distance fields on historical rows when canonical DB metadata is safe.

Output directory: `artifacts/full_evidence_orchestration_20260525/bounded_same_distance_feature_repair_20260602`
Historical rows: `735`
Rolling rows: `208`
Leakage audit: `PASS`
Train/eval schema parity: `PASS`

## Target Resolution

- Resolution counts: `{'UNIQUE_DATE_VENUE': 5, 'AMBIGUOUS_OR_MISSING': 127}`
- Historical rows with resolved safe target metadata: `5`

## Coverage Delta

```json
{
  "avg_time_same_distance": {
    "historical": {
      "new_present_pct": 0.00816326530612245,
      "new_present_rows": 6,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.6201923076923077,
      "new_present_rows": 129,
      "old_present_pct": 0.6201923076923077,
      "old_present_rows": 129
    }
  },
  "best_time_same_distance": {
    "historical": {
      "new_present_pct": 0.00816326530612245,
      "new_present_rows": 6,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.6201923076923077,
      "new_present_rows": 129,
      "old_present_pct": 0.6201923076923077,
      "old_present_rows": 129
    }
  },
  "days_since_last_same_distance_start": {
    "historical": {
      "new_present_pct": 0.012244897959183673,
      "new_present_rows": 9,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "median_time_same_distance": {
    "historical": {
      "new_present_pct": 0.00816326530612245,
      "new_present_rows": 6,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "place_rate_same_distance": {
    "historical": {
      "new_present_pct": 0.047619047619047616,
      "new_present_rows": 35,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "prior_same_distance_start_count": {
    "historical": {
      "new_present_pct": 0.047619047619047616,
      "new_present_rows": 35,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "recent_avg_time_same_distance_5": {
    "historical": {
      "new_present_pct": 0.00816326530612245,
      "new_present_rows": 6,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "recent_best_time_same_distance_5": {
    "historical": {
      "new_present_pct": 0.00816326530612245,
      "new_present_rows": 6,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "same_distance_venue_best_time": {
    "historical": {
      "new_present_pct": 0.004081632653061225,
      "new_present_rows": 3,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "same_distance_venue_start_count": {
    "historical": {
      "new_present_pct": 0.047619047619047616,
      "new_present_rows": 35,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.0,
      "new_present_rows": 0,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    }
  },
  "starts_same_distance": {
    "historical": {
      "new_present_pct": 0.047619047619047616,
      "new_present_rows": 35,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.9230769230769231,
      "new_present_rows": 192,
      "old_present_pct": 0.9230769230769231,
      "old_present_rows": 192
    }
  },
  "win_rate_same_distance": {
    "historical": {
      "new_present_pct": 0.047619047619047616,
      "new_present_rows": 35,
      "old_present_pct": 0.0,
      "old_present_rows": 0
    },
    "rolling": {
      "new_present_pct": 0.625,
      "new_present_rows": 130,
      "old_present_pct": 0.625,
      "old_present_rows": 130
    }
  }
}
```

## Leakage Audit

```json
{
  "checks": {
    "embedded_form_history_dist_g_not_used_as_target_metadata": true,
    "historical_rows_use_canonical_db_history": true,
    "missing_history_remains_explicit": true,
    "no_future_rows_used": true,
    "no_snapshot_manifest_registry_mutation": true,
    "target_outcome_fields_excluded_from_history_query": true
  },
  "notes": {
    "target_resolution_counts": {
      "AMBIGUOUS_OR_MISSING": 127,
      "UNIQUE_DATE_VENUE": 5
    }
  },
  "status": "PASS"
}
```

## Schema Parity

```json
{
  "compatibility": {
    "avg_time_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "best_time_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "days_since_last_same_distance_start": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "median_time_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "place_rate_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "prior_same_distance_start_count": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "recent_avg_time_same_distance_5": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "recent_best_time_same_distance_5": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_distance_venue_best_time": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_distance_venue_start_count": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "starts_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "win_rate_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    }
  },
  "historical_present_fields": [
    "avg_time_same_distance",
    "best_time_same_distance",
    "days_since_last_same_distance_start",
    "median_time_same_distance",
    "place_rate_same_distance",
    "prior_same_distance_start_count",
    "recent_avg_time_same_distance_5",
    "recent_best_time_same_distance_5",
    "same_distance_venue_best_time",
    "same_distance_venue_start_count",
    "starts_same_distance",
    "win_rate_same_distance"
  ],
  "rolling_present_fields": [
    "avg_time_same_distance",
    "best_time_same_distance",
    "days_since_last_same_distance_start",
    "median_time_same_distance",
    "place_rate_same_distance",
    "prior_same_distance_start_count",
    "recent_avg_time_same_distance_5",
    "recent_best_time_same_distance_5",
    "same_distance_venue_best_time",
    "same_distance_venue_start_count",
    "starts_same_distance",
    "win_rate_same_distance"
  ],
  "status": "PASS"
}
```

## Optional Smoke Retest

- Output directory: `artifacts/full_evidence_orchestration_20260525/bounded_same_distance_feature_repair_20260602/optional_smoke_retest`
- Smoke retest status: `SUCCESS`
- Smoke retest recommendation: `HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS`
- Champion baseline: `top3=0.4828`, `brier=0.1221`, `log_loss=1.9948`, `box1_share=0.9310`
- `history_only_hgb`: `top3=0.3448`, `brier=0.1244`, `log_loss=2.0408`, `box1_share=0.1724`
- `no_box_history_hgb`: `top3=0.3103`, `brier=0.1247`, `log_loss=2.0548`, `box1_share=0.2069`
- `reduced_box_band_history_hgb`: `top3=0.3448`, `brier=0.1245`, `log_loss=2.0525`, `box1_share=0.2414`
- The repair restored same-distance coverage on historical rows, but the box-bias production gate remains red and the challenger still does not justify promotion or betting.

## No-Mutation Confirmation

- No production retrain, production model writes, promotion, betting, live result-ingest writes, result label writes, snapshot rewrites, manifest append, registry mutation, or fake EV/odds were performed.

## Known Gate

- The dedicated box-bias production-readiness gate remains red and was not weakened.

## Final Recommendation

`SAME_DISTANCE_REPAIR_NOT_SUFFICIENT_FOR_CHALLENGER`

## Closeout Addendum

- Latest relevant commit before closeout: `0385e2f7 docs(challenger): correct retest closeout commit hash`
- Helper status: `scripts/rebuild_same_distance_feature_packet.py` is untracked in git and was audited before staging.
- Test status: `tests/test_same_distance_feature_repair.py` is untracked in git and was audited before staging.
- Full-addition review: see `closeout_validation/same_distance_repair_full_addition.diff`.
- Validation status: focused same-distance repair tests, history retest tests, safety selector, `git diff --check`, and SQLite quick_check were all clean; the box-bias gate remained red as expected.
- No-mutation confirmation: no production retrain, model promotion, registry mutation, live result-ingest writes, result label writes, snapshot rewrites, or manifest append were performed.
- Endpoint health: `127.0.0.1:5002` remained refused/unavailable during this closeout.
- Active process check: no capture, ingest, promotion, or model-registry process was active.
- Protected-path check: `artifacts/prediction_snapshots/manifest.jsonl` and unrelated Tenn/extraction files were not staged.
- Final recommendation remains `SAME_DISTANCE_REPAIR_NOT_SUFFICIENT_FOR_CHALLENGER`.
- Next safe task: controlled feature-quality diagnosis of the repaired same-distance family or a separate target-grade context repair, not promotion.
