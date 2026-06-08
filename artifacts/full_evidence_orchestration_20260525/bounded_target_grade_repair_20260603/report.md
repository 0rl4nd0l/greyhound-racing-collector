# Target Grade Context Repair

## Executive Summary

This is a report-only packet repair. It preserves the clean official holdout, restores safe target-grade context where canonical metadata exists, and keeps unsafe historical grade tokens fail-closed.

Output directory: `/home/l4nd0/greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525/bounded_target_grade_repair_20260603`
Historical rows: `735`
Rolling rows: `208`
Safe target-grade rows: `227`
Normalized target-grade rows: `227`
Class-transition rows: `227`
Leakage audit: `PASS`
Train/eval schema parity: `PASS`

## Target Grade Resolution

- Same-distance resolution counts: `{'UNIQUE_DATE_VENUE': 5, 'AMBIGUOUS_OR_MISSING': 127}`
- Target-grade provenance counts: `{'SAFE_CANONICAL_DB': 35, 'AMBIGUOUS_OR_MISSING': 153, 'SAFE_CLEAN_OFFICIAL': 192, 'MISSING': 563}`
- Target-grade vocab counts: `{'CANONICAL': 219, 'MISSING': 716, 'LEGACY': 8}`

## Coverage Delta

```json
{
  "same_distance": {
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
    "same_distance_same_grade_avg_time": {
      "historical": {
        "new_present_pct": 0.0,
        "new_present_rows": 0,
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
    "same_distance_same_grade_best_time": {
      "historical": {
        "new_present_pct": 0.0,
        "new_present_rows": 0,
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
    "same_distance_same_grade_start_count": {
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
  },
  "target_grade": {
    "grade_change_direction": {
      "historical": {
        "new_present_pct": 0.036734693877551024,
        "new_present_rows": 27,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.32211538461538464,
        "new_present_rows": 67,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "grade_change_indicator": {
      "historical": {
        "new_present_pct": 0.036734693877551024,
        "new_present_rows": 27,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.32211538461538464,
        "new_present_rows": 67,
        "old_present_pct": 0.9230769230769231,
        "old_present_rows": 192
      }
    },
    "grade_strength_delta": {
      "historical": {
        "new_present_pct": 0.036734693877551024,
        "new_present_rows": 27,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.32211538461538464,
        "new_present_rows": 67,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "last_start_grade": {
      "historical": {
        "new_present_pct": 0.6217687074829932,
        "new_present_rows": 457,
        "old_present_pct": 0.6217687074829932,
        "old_present_rows": 457
      },
      "rolling": {
        "new_present_pct": 0.9567307692307693,
        "new_present_rows": 199,
        "old_present_pct": 0.9567307692307693,
        "old_present_rows": 199
      }
    },
    "last_start_grade_normalized": {
      "historical": {
        "new_present_pct": 0.6204081632653061,
        "new_present_rows": 456,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9471153846153846,
        "new_present_rows": 197,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "recent_grade_mode_5": {
      "historical": {
        "new_present_pct": 0.0,
        "new_present_rows": 0,
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
    "same_grade_place_rate": {
      "historical": {
        "new_present_pct": 0.047619047619047616,
        "new_present_rows": 35,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9230769230769231,
        "new_present_rows": 192,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "same_grade_start_count": {
      "historical": {
        "new_present_pct": 0.047619047619047616,
        "new_present_rows": 35,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9230769230769231,
        "new_present_rows": 192,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "same_grade_win_rate": {
      "historical": {
        "new_present_pct": 0.047619047619047616,
        "new_present_rows": 35,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9230769230769231,
        "new_present_rows": 192,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_normalized": {
      "historical": {
        "new_present_pct": 0.047619047619047616,
        "new_present_rows": 35,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9230769230769231,
        "new_present_rows": 192,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_provenance_reason": {
      "historical": {
        "new_present_pct": 1.0,
        "new_present_rows": 735,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 1.0,
        "new_present_rows": 208,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_provenance_status": {
      "historical": {
        "new_present_pct": 1.0,
        "new_present_rows": 735,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 1.0,
        "new_present_rows": 208,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_safe": {
      "historical": {
        "new_present_pct": 0.047619047619047616,
        "new_present_rows": 35,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 0.9230769230769231,
        "new_present_rows": 192,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_source": {
      "historical": {
        "new_present_pct": 0.23401360544217686,
        "new_present_rows": 172,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 1.0,
        "new_present_rows": 208,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    },
    "target_grade_vocab_status": {
      "historical": {
        "new_present_pct": 1.0,
        "new_present_rows": 735,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      },
      "rolling": {
        "new_present_pct": 1.0,
        "new_present_rows": 208,
        "old_present_pct": 0.0,
        "old_present_rows": 0
      }
    }
  }
}
```

## Class Transition Coverage

```json
{
  "grade_change_direction": {
    "historical": {
      "present_pct": 0.036734693877551024,
      "present_rows": 27,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.32211538461538464,
      "present_rows": 67,
      "rows": 208
    }
  },
  "grade_change_indicator": {
    "historical": {
      "present_pct": 0.036734693877551024,
      "present_rows": 27,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.32211538461538464,
      "present_rows": 67,
      "rows": 208
    }
  },
  "grade_strength_delta": {
    "historical": {
      "present_pct": 0.036734693877551024,
      "present_rows": 27,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.32211538461538464,
      "present_rows": 67,
      "rows": 208
    }
  },
  "last_start_grade": {
    "historical": {
      "present_pct": 0.6217687074829932,
      "present_rows": 457,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9567307692307693,
      "present_rows": 199,
      "rows": 208
    }
  },
  "last_start_grade_normalized": {
    "historical": {
      "present_pct": 0.6204081632653061,
      "present_rows": 456,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9471153846153846,
      "present_rows": 197,
      "rows": 208
    }
  },
  "recent_grade_mode_5": {
    "historical": {
      "present_pct": 0.0,
      "present_rows": 0,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.0,
      "present_rows": 0,
      "rows": 208
    }
  },
  "same_grade_place_rate": {
    "historical": {
      "present_pct": 0.047619047619047616,
      "present_rows": 35,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9230769230769231,
      "present_rows": 192,
      "rows": 208
    }
  },
  "same_grade_start_count": {
    "historical": {
      "present_pct": 0.047619047619047616,
      "present_rows": 35,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9230769230769231,
      "present_rows": 192,
      "rows": 208
    }
  },
  "same_grade_win_rate": {
    "historical": {
      "present_pct": 0.047619047619047616,
      "present_rows": 35,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9230769230769231,
      "present_rows": 192,
      "rows": 208
    }
  },
  "target_grade_normalized": {
    "historical": {
      "present_pct": 0.047619047619047616,
      "present_rows": 35,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9230769230769231,
      "present_rows": 192,
      "rows": 208
    }
  },
  "target_grade_provenance_reason": {
    "historical": {
      "present_pct": 1.0,
      "present_rows": 735,
      "rows": 735
    },
    "rolling": {
      "present_pct": 1.0,
      "present_rows": 208,
      "rows": 208
    }
  },
  "target_grade_provenance_status": {
    "historical": {
      "present_pct": 1.0,
      "present_rows": 735,
      "rows": 735
    },
    "rolling": {
      "present_pct": 1.0,
      "present_rows": 208,
      "rows": 208
    }
  },
  "target_grade_safe": {
    "historical": {
      "present_pct": 0.047619047619047616,
      "present_rows": 35,
      "rows": 735
    },
    "rolling": {
      "present_pct": 0.9230769230769231,
      "present_rows": 192,
      "rows": 208
    }
  },
  "target_grade_source": {
    "historical": {
      "present_pct": 0.23401360544217686,
      "present_rows": 172,
      "rows": 735
    },
    "rolling": {
      "present_pct": 1.0,
      "present_rows": 208,
      "rows": 208
    }
  },
  "target_grade_vocab_status": {
    "historical": {
      "present_pct": 1.0,
      "present_rows": 735,
      "rows": 735
    },
    "rolling": {
      "present_pct": 1.0,
      "present_rows": 208,
      "rows": 208
    }
  }
}
```

## Grade Vocabulary

```json
{
  "grade_context_fields": [
    "target_grade_safe",
    "target_grade_normalized",
    "target_grade_source",
    "target_grade_provenance_status",
    "target_grade_provenance_reason",
    "target_grade_vocab_status",
    "last_start_grade",
    "last_start_grade_normalized",
    "recent_grade_mode_5",
    "same_grade_start_count",
    "same_grade_win_rate",
    "same_grade_place_rate",
    "grade_change_indicator",
    "grade_change_direction",
    "grade_strength_delta"
  ],
  "target_grade_vocab_counts": {
    "CANONICAL": 219,
    "LEGACY": 8,
    "MISSING": 716
  }
}
```

## Leakage Audit

```json
{
  "checks": {
    "ambiguous_race_identity_remains_missing": true,
    "embedded_form_history_dist_g_not_used_as_target_metadata": true,
    "historical_rows_use_canonical_db_history": true,
    "missing_history_remains_explicit": true,
    "no_ev_synthesized": true,
    "no_future_rows_used": true,
    "no_labels_written": true,
    "no_manifest_entries_appended": true,
    "no_odds_synthesized": true,
    "no_snapshot_manifest_registry_mutation": true,
    "no_snapshot_rewrites": true,
    "raw_and_normalized_retained_separately": true,
    "target_grade_source_recorded": true,
    "target_outcome_fields_excluded_from_history_query": true,
    "unmapped_grade_values_not_guessed": true
  },
  "notes": {
    "target_grade_resolution_counts": {
      "AMBIGUOUS_OR_MISSING": 153,
      "MISSING": 563,
      "SAFE_CANONICAL_DB": 35,
      "SAFE_CLEAN_OFFICIAL": 192
    },
    "target_grade_vocab_counts": {
      "CANONICAL": 219,
      "LEGACY": 8,
      "MISSING": 716
    },
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
    "grade_change_direction": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "grade_change_indicator": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "grade_strength_delta": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "last_start_grade": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "last_start_grade_normalized": {
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
    "recent_grade_mode_5": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_distance_same_grade_avg_time": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_distance_same_grade_best_time": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_distance_same_grade_start_count": {
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
    "same_grade_place_rate": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_grade_start_count": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "same_grade_win_rate": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "starts_same_distance": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_normalized": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_provenance_reason": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_provenance_status": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_safe": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_source": {
      "compatible": true,
      "historical_present": true,
      "rolling_present": true
    },
    "target_grade_vocab_status": {
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
    "grade_change_direction",
    "grade_change_indicator",
    "grade_strength_delta",
    "last_start_grade",
    "last_start_grade_normalized",
    "median_time_same_distance",
    "place_rate_same_distance",
    "prior_same_distance_start_count",
    "recent_avg_time_same_distance_5",
    "recent_best_time_same_distance_5",
    "recent_grade_mode_5",
    "same_distance_same_grade_avg_time",
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_start_count",
    "same_distance_venue_best_time",
    "same_distance_venue_start_count",
    "same_grade_place_rate",
    "same_grade_start_count",
    "same_grade_win_rate",
    "starts_same_distance",
    "target_grade_normalized",
    "target_grade_provenance_reason",
    "target_grade_provenance_status",
    "target_grade_safe",
    "target_grade_source",
    "target_grade_vocab_status",
    "win_rate_same_distance"
  ],
  "rolling_present_fields": [
    "avg_time_same_distance",
    "best_time_same_distance",
    "days_since_last_same_distance_start",
    "grade_change_direction",
    "grade_change_indicator",
    "grade_strength_delta",
    "last_start_grade",
    "last_start_grade_normalized",
    "median_time_same_distance",
    "place_rate_same_distance",
    "prior_same_distance_start_count",
    "recent_avg_time_same_distance_5",
    "recent_best_time_same_distance_5",
    "recent_grade_mode_5",
    "same_distance_same_grade_avg_time",
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_start_count",
    "same_distance_venue_best_time",
    "same_distance_venue_start_count",
    "same_grade_place_rate",
    "same_grade_start_count",
    "same_grade_win_rate",
    "starts_same_distance",
    "target_grade_normalized",
    "target_grade_provenance_reason",
    "target_grade_provenance_status",
    "target_grade_safe",
    "target_grade_source",
    "target_grade_vocab_status",
    "win_rate_same_distance"
  ],
  "status": "PASS"
}
```

## Endpoint And SQLite Health

- Endpoint health: `127.0.0.1:8000 /api/health -> ok`; `127.0.0.1:5002 -> connection refused`
- SQLite quick_check: `ok`
- Active capture/ingest/promotion/model-registry processes: `none found`

## Optional Smoke Retest

- Status: `SUCCESS`
- Recommendation: `HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS`
- Packet variants compared: `champion_current_baseline`, `history_only_hgb`, `grade_context_hgb`, `no_box_history_hgb`, `reduced_box_band_history_hgb`, `reduced_box_band_grade_context_hgb`, and calibrated counterparts.
- Result: the new grade-context variants matched the corresponding history-only variants on the reported top-1/top-2/top-3 and box-1 share metrics. The repair did not improve the box-bias blocker or distinguish itself from the existing history-only challenger path.
- Key comparison: `history_only_hgb` and `grade_context_hgb` both reported top3 `0.3448` and box1 share `0.1724`; `reduced_box_band_history_hgb` and `reduced_box_band_grade_context_hgb` both reported top3 `0.3448` and box1 share `0.2414`.
- Interpretation: the bounded target-grade/context repair is leakage-safe and materially improves safe target-grade coverage, but it is not sufficient to clear a separate controlled challenger study on its own.

## No-Mutation Confirmation

- No production retrain, production model writes, promotion, betting, live result-ingest writes, result label writes, snapshot rewrites, manifest append, registry mutation, or fake EV/odds were performed.

## Known Gate

- The dedicated box-bias production-readiness gate remains red and was not weakened.

## Final Recommendation

`TARGET_GRADE_REPAIR_NOT_SUFFICIENT_FOR_CHALLENGER`

## Closeout Addendum

Latest commit before closeout:
`292bfc4f feat(features): add report-only same-distance repair packet`

Helper and test status:
- `scripts/rebuild_same_distance_feature_packet.py`: tracked and modified
- `scripts/run_history_feature_challenger_retest.py`: tracked and modified
- `tests/test_target_grade_context_repair.py`: untracked before staging, intentionally reviewed with intent-to-add
- `tests/test_same_distance_feature_repair.py`: tracked and modified

Full-diff audit result:
- The scoped diff was reviewed for only the five target files listed below.
- `git diff --check` for the scoped files was clean.
- No forbidden-path mutations were staged in this closeout pass.

Exact files proposed for staging:
- `scripts/rebuild_same_distance_feature_packet.py`
- `scripts/run_history_feature_challenger_retest.py`
- `tests/test_target_grade_context_repair.py`
- `tests/test_same_distance_feature_repair.py`
- `artifacts/full_evidence_orchestration_20260525/bounded_target_grade_repair_20260603/report.md`

Validation commands and outcomes:
- `git log -10 --oneline`: captured in `closeout_validation/git_log_10.txt`
- `git status --short --branch --untracked-files=all`: captured in `closeout_validation/git_status_before.txt`
- `git diff --cached --name-only`: captured in `closeout_validation/staged_before.txt`
- `git diff --check`: clean
- `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'`: `ok`
- `python3 -m py_compile ...`: clean
- `.venv/bin/python -m pytest -q tests/test_target_grade_context_repair.py --maxfail=1`: `3 passed`
- `.venv/bin/python -m pytest -q tests/test_same_distance_feature_repair.py --maxfail=1`: `5 passed, 1 skipped`
- `.venv/bin/python -m pytest -q tests/test_run_history_feature_challenger_retest.py --maxfail=1`: `8 passed`
- `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set or odds or ev or model_contract or calibration or same_distance or history_feature or target_grade or grade_context' --maxfail=1`: `247 passed, 1 skipped, 646 deselected`
- `.venv/bin/python -m pytest -q tests/test_box_bias_regression.py::test_favorite_box1_share_under_threshold --maxfail=1 -vv`: failed as expected with `Box 1 favorites share too high: 90.00% > 50% over 190 files.`

No-mutation confirmation:
- No production retrain.
- No production model writes.
- No model promotion.
- No model registry mutation.
- No betting.
- No live result ingestion writes.
- No result label writes.
- No snapshot writes or rewrites.
- No manifest append.
- No fake EV.
- No fake odds.
- `APPROVE_RESULT_LABEL_WRITE` remained unset.

Endpoint health:
- `127.0.0.1:8000 /api/health -> ok`
- `127.0.0.1:5002 -> connection refused`

Active-process check:
- No active `prejump_prediction_loop`, `capture_prediction_snapshot`, `ingest_results_for_date`, `promote`, or `model_registry` processes were found.

Protected-path checks:
- `artifacts/prediction_snapshots/manifest.jsonl` was not staged.
- Snapshot JSON files were not staged.
- `model_registry/` was not staged.
- `docs/model_registry/current_production.json` was not staged.
- `ml_models_v4/` was not staged.
- `advanced_models/` was not staged.
- Unrelated Tenn/extraction dirt was not staged.

Known gate:
- The dedicated box-bias production-readiness gate remains red and was not weakened.

Next safe task recommendation:
- Controlled feature-quality diagnosis for why repaired same-distance plus target-grade context still does not improve Top3 or calibration.
- Do not promote, bet, retrain production, or relax the box-bias gate.
