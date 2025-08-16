import json

# Paths to metadata files
metadata_files = [
    "model_registry/metadata/et_natural_migrated_model_20250727_202333_metadata.json",
    "model_registry/metadata/comprehensive_enhanced_balanced_gradient_boosting_20250731_192823_metadata.json",
    "model_registry/metadata/gradient_boosting_optimized_migrated_model_20250727_202331_metadata.json",
    "model_registry/metadata/balanced_random_forest_migrated_model_20250727_202334_metadata.json",
    "model_registry/metadata/logistic_regression_migrated_model_20250727_202335_metadata.json"
]

# Features from engineering pipeline
current_engineered_features = [
    "avg_position", "recent_form_avg", "market_confidence",
    "current_odds_log", "venue_experience", "place_rate", "current_weight",
    "time_consistency", "traditional_overall_score", "traditional_performance_score",
    "traditional_form_score", "traditional_consistency_score",
    "traditional_confidence_level", "win_rate", "long_term_form_trend",
    "position_consistency", "avg_time", "best_time", "time_improvement_trend",
    "avg_weight", "weight_consistency", "weight_vs_avg", "distance_specialization",
    "grade_experience", "fitness_score", "traditional_class_score",
    "traditional_fitness_score", "traditional_experience_score",
    "traditional_trainer_score", "traditional_track_condition_score",
    "traditional_distance_score", "traditional_key_factors_count",
    "traditional_risk_factors_count", "temperature", "humidity", "wind_speed",
    "pressure", "weather_adjustment_factor", "weather_clear", "weather_cloudy",
    "weather_rain", "weather_fog", "temp_cold", "temp_cool", "temp_optimal",
    "temp_warm", "temp_hot", "wind_calm", "wind_light", "wind_moderate",
    "wind_strong", "humidity_low", "humidity_normal", "humidity_high",
    "weather_experience_count", "weather_performance", "days_since_last",
    "competition_strength", "box_win_rate", "current_box", "field_size",
    "historical_races_count", "venue_encoded", "track_condition_encoded",
    "grade_encoded", "distance_numeric"
]

# Function to compare features
def compare_features(metadata_path, engineered_features):
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    model_features = set(metadata["feature_names"])
    engineered_features_set = set(engineered_features)

    missing_in_engineered = model_features - engineered_features_set
    extra_in_engineered = engineered_features_set - model_features

    return {
        "missing_features": missing_in_engineered,
        "extra_features": extra_in_engineered
    }

# Perform feature comparison for each metadata file
for metadata_file in metadata_files:
    drift_check = compare_features(metadata_file, current_engineered_features)
    print(f"Features missing in engineered list from {metadata_file}: {drift_check['missing_features']}")
    print(f"Extra features present in engineered list not in model from {metadata_file}: {drift_check['extra_features']}")

