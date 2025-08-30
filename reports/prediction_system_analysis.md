# ML System V4 Prediction System Report

Executive summary
- Purpose: Document how V4 generates predictions from upcoming race CSVs, including feature engineering, model inference, calibration, normalization, confidence, place probabilities, and expected value (EV).
- Scope: V4 only (current leakage-safe, calibrated pipeline). Includes reproducible example with sample odds.
- Guarantees: Strict temporal leakage protection, per-race probability normalization, optional explainability metadata, schema contract checks.
- Key math:
  - Raw win probability: calibrated_pipeline.predict_proba(X)[:, 1]
  - Normalization (adaptive): simple vs softmax vs power, controlled by env vars
  - Confidence: weighted blend of own win prob, top-two margin, 1 - entropy, and feature completeness
  - EV (win): EV = p_win × (odds − 1) − (1 − p_win)

Architecture overview

Inference path
```mermaid path=null start=null
graph TD
  A[Upcoming race CSV (race data)] --> B[CsvIngestion.parse_csv]\nB --> C[_map_csv_to_v4_format + CSV historic enrich]\nC --> D[TemporalFeatureBuilder.build_features_for_race]\nD --> E[Sklearn pipeline (preprocessor + ExtraTrees)\nwrapped in CalibratedClassifierCV]\nE --> F[raw_win_prob = predict_proba[:,1]]\nF --> G[_group_normalize_probabilities (adaptive)]\nG --> H[place_prob heuristic]\nG --> I[confidence (margin + entropy + completeness)]\nG --> J[EV (if market_odds provided)]\nH --> K[Assemble outputs]\nI --> K\nJ --> K\nK --> L[Rank by win_prob_norm and return]
```

Training path
```mermaid path=null start=null
graph TD
  A[SQLite DB: dog_race_data, race_metadata, enhanced_expert_data] --> B[Query + Quality Filters]\nB --> C[TemporalFeatureBuilder.build_features_for_race]\nC --> D[Time-ordered split (80/20)]\nD --> E[Sklearn ColumnTransformer + ExtraTrees]\nE --> F[CalibratedClassifierCV (isotonic, cv=K)]\nF --> G[Metrics: Accuracy, AUC, Brier, Permutation Importance]\nG --> H[EV threshold learning (sim odds)]\nH --> I[Persist artifact + feature signature + contract]\nI --> J[Load at inference via registry or latest model]
```

Data semantics and compliance with repository rules
- Historical data = form guides and derivatives (used only for training/engineering historical features).
- Race data = upcoming race CSV participants/metadata (no outcomes; used for inference).
- Winner labels for training come from race webpages (ingested into DB), not from the form guide.
- CSV form guide format: 10 unique dogs; blank rows under each participant are historical entries and are parsed as CSV-derived historic stats.

Key components and file references
- Ingestion: src/parsers/csv_ingestion.py (CsvIngestion.parse_csv)
- V4 pipeline wrapper: prediction_pipeline_v4.py (PredictionPipelineV4)
- Leakage-safe features: temporal_feature_builder.py (TemporalFeatureBuilder)
- Model and inference: ml_system_v4.py (MLSystemV4)
- Optional feature store helpers: features/feature_store.py
- Docs: docs/ML_SYSTEM_V4_README.md, docs/data_dictionary/*.md

1) Ingestion and mapping
- CsvIngestion.parse_csv: detects delimiter, validates structure, and provides headers/records plus a validation report.
- prediction_pipeline_v4.py::_map_csv_to_v4_format:
  - Participants are detected (e.g., names like "2. Austrian Rose").
  - Maps columns to V4 schema (examples):
    - 'Dog Name' → dog_clean_name (cleans prefixes and punctuation, Title Case)
    - 'BOX' → box_number
    - 'WGT' → weight
    - 'DIST' → distance (numeric)
    - Venue and race_date parsed from filename
    - Defaults applied: track_condition='Good', weather='Fine', etc.
  - _enrich_with_csv_historical_data extracts embedded historical records (blank-name rows) to compute csv_historical_races, csv_avg_finish_position, csv_best_finish_position, csv_win_rate, csv_place_rate, csv_avg_time, csv_best_time.
- Alternatively, MLSystemV4.preprocess_upcoming_race_csv can be used to map a raw CSV DataFrame directly to the expected columns for ML inference.

2) Leakage-safe feature engineering
- temporal_feature_builder.py::TemporalFeatureBuilder
  - get_race_timestamp: builds a race timestamp (prefers race_time over race_date).
  - load_dog_historical_data: queries GREYHOUND_DB_PATH for past races for the dog, filtering to dates strictly before the target race timestamp and within a configurable lookback.
  - create_historical_features: builds weighted metrics using exponential decay (recent races emphasized):
    - Position metrics: historical_avg_position, historical_best_position, historical_win_rate, historical_place_rate, historical_form_trend
    - Time metrics (distance-adjusted if target distance present): historical_avg_time, historical_best_time, historical_time_consistency
    - Contextual metrics: venue_specific_*, grade_specific_*, best_distance_*; temporal: days_since_last_race, race_frequency
  - CSV-embedded stats (if present) override DB-derived equivalents for the target race row.
  - TGR integration (optional) augments features via DB-only lookup when enabled.
  - validate_temporal_integrity: asserts no post-race fields from the target race (e.g., finish_position, winner_name) and verifies row counts.

3) Model pipeline and calibration (V4)
- ml_system_v4.py::create_sklearn_pipeline
  - ColumnTransformer with passthrough numeric and OneHotEncoder for categorical features: ['venue', 'grade', 'track_condition', 'weather', 'trainer_name'].
  - Classifier: ExtraTreesClassifier; hyperparameters tunable via env (V4_TREES, V4_MAX_DEPTH, V4_MIN_SAMPLES_LEAF). class_weight='balanced', bootstrap=True, n_jobs=-1.
  - Calibration: CalibratedClassifierCV(method='isotonic', cv=V4_CALIB_FOLDS).
- Training data preparation
  - prepare_time_ordered_data: queries DB, applies strong quality filters (field size, single winner, valid positions, required metadata), time-orders races, and splits 80/20 by race timestamp.
  - Features for both train and test are built via the same builder used in inference.
  - Metrics: accuracy, ROC AUC, Brier score; optional permutation importance.
- Persistence and contracts
  - Model artifacts saved under ./ml_models_v4/*.joblib with feature_columns, categorical/numerical columns, and model_info.
  - Feature signature and a JSON feature contract are written under docs/model_contracts.
  - At inference, model loading prefers a Model Registry, then latest artifact, else creates a mock model.

4) Inference math and calculations
4.1 Raw win probability
- From calibrated pipeline output:
  - raw_win_prob_i = calibrated_pipeline.predict_proba(X_pred)[i, 1]

4.2 Per-race normalization (sums to ≈1)
- ml_system_v4.py::_group_normalize_probabilities
- Modes and controls (env overrides):
  - V4_NORMALIZATION_MODE ∈ {simple, softmax, power}; default adaptive selection by variance
  - V4_TEMP_SOFTMAX (default 2.0)
  - V4_POWER_EXP (default 1.8)
- Formulas:
  - Simple: p_i_norm = p_i / Σ_j p_j
  - Softmax (temperature T): p_i_norm = exp((p_i − max(p))/T) / Σ_j exp((p_j − max(p))/T)
  - Power (exponent α): p_i_norm = p_i^α / Σ_j p_j^α
- Adaptive selection:
  - Very low variance → power
  - Moderate variance → softmax
  - High variance → simple
- Validation: logs a warning if Σ p_i_norm ∉ [0.95, 1.05].

4.3 Place probability (heuristic)
- place_prob_norm_i = min(0.95, win_prob_norm_i × 2.8)
  - Note: heuristic, not separately calibrated.

4.4 Confidence score
- Components:
  - p1, p2: top-two win_prob_norm in the race
  - Entropy H = −Σ_i p_i log(max(p_i, ε)); normalized by log(N)
  - completeness_i: fraction of non-zero numeric features for runner i
- Weights (env):
  - V4_CONF_W1 (default 0.4), V4_CONF_W2 (0.4), V4_CONF_W3 (0.2)
- Formulas:
  - base_conf_i = w1 × win_prob_norm_i + w2 × max(p1 − p2, 0) + w3 × max(1 − H_norm, 0)
  - completeness_conf_i = min(0.95, completeness_i × 0.8 + 0.2)
  - confidence_i = 0.75 × base_conf_i + 0.25 × completeness_conf_i
  - Confidence level buckets: High (≥0.8), Medium (≥0.6), Low (≥0.4), Very Low (<0.4).

4.5 Expected Value (EV)
- If market odds supplied (decimal odds):
  - EV_win_i = win_prob_norm_i × (odds_i − 1) − (1 − win_prob_norm_i)
  - Flags ev_positive when EV_win_i > 0
  - Training-time threshold learning (_learn_ev_thresholds) uses simulated odds to optimize a simple EV threshold; inference accepts per-runner odds mapping for real EV.

5) Output schema, sorting, and ranking
- Per-runner prediction fields:
  - dog_name, dog_clean_name, box_number
  - win_prob_raw, win_prob_norm, place_prob_norm
  - win_probability (alias of win_prob_norm), calibration_applied
  - confidence, confidence_level
  - Optional EV: odds, ev_win, ev_positive (if odds provided)
- Race-level metadata:
  - calibration_meta { method, applied, normalization_sum }
  - explainability_meta: numeric means, summary of categorical features; a detailed record is logged to logs/
  - signature_meta { expected_signature, actual_signature, match }
- Sorting and rank:
  - Sorted by win_prob_norm descending
  - predicted_rank assigned as 1..N in sort order

6) Environment toggles (runtime)
- Data & environment
  - GREYHOUND_DB_PATH: path to SQLite DB for historical lookup (defaults to ./greyhound_racing_data.db if found)
  - UPCOMING_RACES_DIR: where upcoming CSVs live (UI/API)
  - TESTING: affects some behaviors in tests/CI
- Normalization and confidence
  - V4_NORMALIZATION_MODE ∈ {simple, softmax, power} (adaptive by default)
  - V4_TEMP_SOFTMAX: float, default 2.0
  - V4_POWER_EXP: float, default 1.8
  - V4_CONF_W1, V4_CONF_W2, V4_CONF_W3: floats, defaults 0.4, 0.4, 0.2
- Training hyperparameters
  - V4_TREES (default 1000), V4_MAX_DEPTH (20), V4_MIN_SAMPLES_LEAF (2), V4_CALIB_FOLDS (5)
- Feature contract
  - FEATURE_CONTRACT_CHECK_ON_LOAD: warning-only check
  - FEATURE_CONTRACT_ENFORCE: raise on mismatch
- TGR features
  - TGR_ENABLED: enables DB-only TGR integration (no scraping in prediction_only mode)
  - PREDICTION_IMPORT_MODE, ENABLE_RESULTS_SCRAPERS: control integrator behavior

7) Failures, fallbacks, and defenses
- Model loading order: Model Registry → latest ./ml_models_v4/*.joblib → create a lightweight/basic mock model.
- Feature signature drift: signature_meta reports mismatch; inference fails with a clear error to prevent silent misuse.
- Temporal leakage defenses:
  - validate_temporal_integrity for feature sets
  - Prediction-time assertion hook rejects future-dated race rows or presence of post-race fields/disabled odds-related fields.
- Missing database/historical rows: builder returns defaults; predictions still returned (confidence and completeness reflect missingness).

8) Reproducible local run (with EV)
- Prereqs (typical per WARP.md):
```bash path=null start=null
# One-time setup (do not execute here)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
export GREYHOUND_DB_PATH=./greyhound_racing_data.db
export UPCOMING_RACES_DIR=./upcoming_races_temp
# Optional tuning
export V4_NORMALIZATION_MODE=adaptive
export V4_CONF_W1=0.4 V4_CONF_W2=0.4 V4_CONF_W3=0.2
```

- Python example using MLSystemV4 directly (passes market_odds to compute EV):
```python path=null start=null
import os
from pathlib import Path
import pandas as pd
from ml_system_v4 import MLSystemV4

# Inputs
csv_path = "./upcoming_races_temp/Race 1 - ABCD - 2025-08-30.csv"  # replace with your file
race_id = Path(csv_path).stem

# Initialize V4
ml = MLSystemV4(db_path=os.getenv("GREYHOUND_DB_PATH", "./greyhound_racing_data.db"))

# Read raw CSV and preprocess to V4 race schema
raw_df = pd.read_csv(csv_path, encoding="utf-8")
race_df = ml.preprocess_upcoming_race_csv(raw_df, race_id)

# Provide sample decimal odds; keys must match dog_clean_name after preprocessing
# Tip: print(race_df[["dog_clean_name","box_number"]]) to get exact names
market_odds = {
    # "Dog Name" here must match the cleaned Title Case in race_df['dog_clean_name']
    # e.g., "Austrian Rose": 3.8,
}

# Predict (with EV)
result = ml.predict_race(race_df, race_id, market_odds=market_odds)

if result.get("success"):
    preds = result.get("predictions", [])
    for p in preds:
        print({
            "rank": p.get("predicted_rank"),
            "dog": p.get("dog_clean_name"),
            "win_prob_norm": round(p.get("win_prob_norm", 0.0), 4),
            "place_prob_norm": round(p.get("place_prob_norm", 0.0), 4),
            "confidence": round(p.get("confidence", 0.0), 3),
            "odds": p.get("odds"),
            "ev_win": round(p.get("ev_win", 0.0), 4) if p.get("ev_win") is not None else None,
            "ev_positive": p.get("ev_positive"),
        })
else:
    print("Prediction failed:", result)
```

- Notes:
  - If you prefer to let PredictionPipelineV4 do CSV mapping, you can still access its ml_system_v4 member and call predict_race with market_odds for EV.
  - Ensure dog names in market_odds match race_df['dog_clean_name'] exactly.

9) Assumptions and caveats
- CSV files are upcoming race data and contain no outcomes; embedded historical lines are used only as additional historical stats.
- Training labels (winners) originate from race webpages and arrive in the DB; V4 never uses target race outcomes at predict time.
- Place probability is a heuristic derived from win probability; calibrating place odds would require market place odds.
- EV thresholding is not automatically applied at inference; you can consume ev_win and ev_positive and apply your own thresholds.

10) References (files and key functions)
- Ingestion: src/parsers/csv_ingestion.py (CsvIngestion.parse_csv)
- Pipeline: prediction_pipeline_v4.py (PredictionPipelineV4.predict_race_file, _map_csv_to_v4_format, _enrich_with_csv_historical_data)
- Features: temporal_feature_builder.py (TemporalFeatureBuilder.build_features_for_race, load_dog_historical_data, create_historical_features, validate_temporal_integrity)
- Model & inference: ml_system_v4.py (create_sklearn_pipeline, prepare_time_ordered_data, train_model, predict_race, _group_normalize_probabilities, _learn_ev_thresholds)
- Feature store helpers: features/feature_store.py
- Docs: docs/ML_SYSTEM_V4_README.md, docs/data_dictionary/feature_engineering.md, docs/data_dictionary/predictions.md

Appendix A: Quick checklist (operational)
- DB present (GREYHOUND_DB_PATH) and reachable? If missing, historical features default and confidence may drop.
- Feature signature matches contract? If not, update contract or retrain to align schemas.
- Environment toggles tuned? Consider normalization mode and confidence weights for desired sharpness.
- Odds source available? Provide decimal odds for EV; ensure names match dog_clean_name.

