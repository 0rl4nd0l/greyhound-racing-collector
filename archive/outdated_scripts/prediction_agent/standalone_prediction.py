#!/usr/bin/env python3
"""
Standalone prediction script using ultra_insights.json analysis
This version doesn't require OpenAI API and focuses on data-driven predictions
"""

import json
import os

import numpy as np
import pandas as pd


def get_weight_bracket(weight):
    """Determine weight bracket based on ultra_insights.json categories"""
    if weight < 27.5:
        return "Light"
    elif weight < 30.0:
        return "Medium_Light"
    elif weight < 32.0:
        return "Medium"
    elif weight < 34.0:
        return "Medium_Heavy"
    else:
        return "Heavy"


def get_weight_bracket_performance(ultra_insights, weight_bracket):
    """Get win rate for weight bracket from ultra_insights.json"""
    for bracket in ultra_insights["weight_impact"]["weight_bracket_performance"]:
        if bracket["weight_bracket"] == weight_bracket:
            return bracket["win_rate"]
    return 0.15  # Default win rate


def clean_race_data(df):
    """Clean and prepare the race data for prediction"""
    # Filter out rows with empty dog names (continuation rows)
    df = df[df["Dog Name"].str.contains(r"^\d+\.", na=False)]

    # Clean dog names - remove the number prefix
    df["Dog Name"] = df["Dog Name"].str.replace(r"^\d+\.\s*", "", regex=True)

    # Convert data types
    numeric_cols = ["WGT", "TIME", "DIST", "1 SEC", "BOX", "SP", "PLC"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calculate_advanced_metrics(dog_row, ultra_insights):
    """Calculate advanced metrics based on ultra_insights.json features"""

    # Extract basic values
    weight = dog_row["WGT"] if pd.notna(dog_row["WGT"]) else 30.0
    time = dog_row["TIME"] if pd.notna(dog_row["TIME"]) else 17.0
    dist = dog_row["DIST"] if pd.notna(dog_row["DIST"]) else 300
    sectional = dog_row["1 SEC"] if pd.notna(dog_row["1 SEC"]) else 6.0
    box = dog_row["BOX"] if pd.notna(dog_row["BOX"]) else 1

    # Calculate key metrics matching ultra_insights features
    metrics = {
        "relative_time": time / dist if dist > 0 else 0.057,
        "early_speed": dist / sectional if sectional > 0 else 17.0,
        "relative_weight": weight / 31.0,  # Normalized against typical weight
        "individual_time_numeric": time,
        "avg_speed": dist / time if time > 0 else 17.0,
        "box_number": box,
        "sectional_1_numeric": sectional,
        "sp_mean": dog_row["SP"] if pd.notna(dog_row["SP"]) else 10.0,
    }

    return metrics


def calculate_prediction_score(metrics, ultra_insights):
    """Calculate prediction score based on ultra_insights feature importance"""

    score = 0
    feature_contributions = {}

    # Apply feature importance weights
    for fi in ultra_insights["ultra_predictive_model"]["feature_importance"]:
        feature = fi["feature"]
        importance = fi["importance"]

        if feature in metrics and not np.isnan(metrics[feature]):
            contribution = importance * metrics[feature]
            score += contribution
            feature_contributions[feature] = contribution

    return score, feature_contributions


def run_comprehensive_prediction(race_file_path):
    """Run comprehensive prediction analysis"""

    # Load Ultimate Insights JSON
    with open("../ultimate_insights.json", "r") as f:
        ultra_insights = json.load(f)

    # Load race data
    race_data = pd.read_csv(race_file_path)
    race_data = clean_race_data(race_data)

    print("🏁 COMPREHENSIVE RACE ANALYSIS - Race 1 HEA - 11 July 2025")
    print("=" * 70)
    print(f"\nAnalyzing {len(race_data)} greyhounds using ultimate_insights.json")
    print(
        f"Model Accuracy: {ultra_insights['ultra_predictive_model']['model_accuracy']:.2%}"
    )
    print(
        f"Training Data: {ultra_insights['data_summary']['total_races']} races, {ultra_insights['data_summary']['total_entries']} entries"
    )
    print()

    # Calculate predictions for each greyhound
    predictions = []

    for _, dog_row in race_data.iterrows():
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(dog_row, ultra_insights)

        # Calculate base prediction score
        base_score, feature_contributions = calculate_prediction_score(
            metrics, ultra_insights
        )

        # Weight bracket adjustment
        weight_bracket = get_weight_bracket(metrics["relative_weight"] * 31.0)
        weight_performance = get_weight_bracket_performance(
            ultra_insights, weight_bracket
        )
        weight_adjustment = weight_performance * 0.15

        # Speed analysis bonus
        winners_avg_early_speed = ultra_insights["speed_analysis"][
            "early_speed_analysis"
        ]["winners_avg_early_speed"]
        speed_bonus = 0.1 if metrics["early_speed"] > winners_avg_early_speed else 0

        # Box position factor
        box_bonus = 0.05 if metrics["box_number"] <= 3 else 0

        # Calculate final score
        final_score = base_score + weight_adjustment + speed_bonus + box_bonus

        # Determine key strengths
        top_feature = max(feature_contributions.items(), key=lambda x: x[1])

        predictions.append(
            {
                "greyhound_name": dog_row["Dog Name"],
                "box": int(metrics["box_number"]),
                "weight": metrics["relative_weight"] * 31.0,
                "weight_bracket": weight_bracket,
                "early_speed": metrics["early_speed"],
                "relative_time": metrics["relative_time"],
                "avg_speed": metrics["avg_speed"],
                "sectional_time": metrics["sectional_1_numeric"],
                "starting_price": metrics["sp_mean"],
                "base_score": base_score,
                "weight_adjustment": weight_adjustment,
                "speed_bonus": speed_bonus,
                "box_bonus": box_bonus,
                "final_score": final_score,
                "top_feature": top_feature[0],
                "top_feature_value": top_feature[1],
                "feature_contributions": feature_contributions,
            }
        )

    # Normalize confidence scores
    total_score = sum(p["final_score"] for p in predictions)
    for p in predictions:
        p["confidence"] = (
            p["final_score"] / total_score if total_score > 0 else 1 / len(predictions)
        )

    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return predictions, ultra_insights


def print_detailed_analysis(predictions, ultra_insights):
    """Print comprehensive analysis results"""

    print("📊 DETAILED PREDICTIONS")
    print("=" * 70)

    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred["confidence"] * 100
        print(f"\n{i}. {pred['greyhound_name']} (Box {pred['box']})")
        print(f"   🎯 Confidence: {confidence_pct:.1f}%")
        print(f"   ⚖️  Weight: {pred['weight']:.1f}kg ({pred['weight_bracket']})")
        print(f"   🏃 Early Speed: {pred['early_speed']:.1f} m/s")
        print(f"   ⏱️  Average Speed: {pred['avg_speed']:.1f} m/s")
        print(f"   📈 Starting Price: ${pred['starting_price']:.1f}")
        print(
            f"   🔥 Top Feature: {pred['top_feature']} (contribution: {pred['top_feature_value']:.4f})"
        )
        print(f"   📋 Score Breakdown:")
        print(f"      • Base Score: {pred['base_score']:.4f}")
        print(f"      • Weight Adjustment: +{pred['weight_adjustment']:.4f}")
        print(f"      • Speed Bonus: +{pred['speed_bonus']:.4f}")
        print(f"      • Box Bonus: +{pred['box_bonus']:.4f}")
        print(f"      • Final Score: {pred['final_score']:.4f}")

    print(f"\n🎯 KEY INSIGHTS")
    print("=" * 70)

    # Top pick analysis
    top_pick = predictions[0]
    print(
        f"• TOP PICK: {top_pick['greyhound_name']} with {top_pick['confidence']*100:.1f}% confidence"
    )
    print(f"  - Primary advantage: {top_pick['top_feature']}")
    print(f"  - Weight category: {top_pick['weight_bracket']}")

    # Speed analysis
    fastest_early = max(predictions, key=lambda x: x["early_speed"])
    fastest_avg = max(predictions, key=lambda x: x["avg_speed"])
    print(f"• SPEED LEADERS:")
    print(
        f"  - Fastest early speed: {fastest_early['greyhound_name']} ({fastest_early['early_speed']:.1f} m/s)"
    )
    print(
        f"  - Fastest average speed: {fastest_avg['greyhound_name']} ({fastest_avg['avg_speed']:.1f} m/s)"
    )

    # Weight distribution
    weight_brackets = {}
    for p in predictions:
        bracket = p["weight_bracket"]
        if bracket not in weight_brackets:
            weight_brackets[bracket] = []
        weight_brackets[bracket].append(p["greyhound_name"])

    print(f"• WEIGHT DISTRIBUTION:")
    for bracket, dogs in weight_brackets.items():
        bracket_performance = get_weight_bracket_performance(ultra_insights, bracket)
        print(f"  - {bracket}: {len(dogs)} dogs (win rate: {bracket_performance:.1%})")

    # Market analysis
    market_leader = min(predictions, key=lambda x: x["starting_price"])
    print(f"• MARKET ANALYSIS:")
    print(
        f"  - Market leader: {market_leader['greyhound_name']} (${market_leader['starting_price']:.1f})"
    )
    print(f"  - Our top pick SP: ${top_pick['starting_price']:.1f}")

    # Feature importance insights
    print(
        f"• MODEL INSIGHTS (from {ultra_insights['ultra_predictive_model']['model_accuracy']:.1%} accuracy model):"
    )
    top_3_features = ultra_insights["ultra_predictive_model"]["feature_importance"][:3]
    for i, feature in enumerate(top_3_features, 1):
        print(f"  {i}. {feature['feature']}: {feature['importance']:.1%} importance")


if __name__ == "__main__":
    race_file = "form_guides/Race 2 - HEA - 11 July 2025.csv"

    if not os.path.exists(race_file):
        print(f"Error: {race_file} not found")
        exit(1)

    if not os.path.exists("../ultimate_insights.json"):
        print("Error: ultimate_insights.json not found in parent directory")
        exit(1)

    try:
        predictions, ultra_insights = run_comprehensive_prediction(race_file)
        print_detailed_analysis(predictions, ultra_insights)

        print(f"\n💡 BETTING RECOMMENDATIONS")
        print("=" * 70)

        # Top 3 picks
        print("🥇 WIN BETS:")
        for i, pred in enumerate(predictions[:3], 1):
            value_ratio = pred["confidence"] * 100 / pred["starting_price"]
            print(
                f"  {i}. {pred['greyhound_name']} - {pred['confidence']*100:.1f}% confidence @ ${pred['starting_price']:.1f} (Value: {value_ratio:.2f})"
            )

        # Quinella suggestions
        print("\n🎯 QUINELLA SUGGESTIONS:")
        print(
            f"  1-2: {predictions[0]['greyhound_name']} + {predictions[1]['greyhound_name']}"
        )
        print(
            f"  1-3: {predictions[0]['greyhound_name']} + {predictions[2]['greyhound_name']}"
        )

    except Exception as e:
        print(f"Error running prediction: {e}")
        import traceback

        traceback.print_exc()
