#!/usr/bin/env python3
"""
Weather-Enhanced Predictor Test Script
=====================================

This script demonstrates the weather-enhanced predictor capabilities
by running predictions on available upcoming race files and showing
the impact of weather conditions on race predictions.

Author: AI Assistant
Date: July 25, 2025
"""

import glob
import os
from pathlib import Path

from weather_enhanced_predictor import WeatherEnhancedPredictor


def find_upcoming_race_files():
    """Find all upcoming race CSV files"""
    upcoming_dir = Path("./upcoming_races")
    if not upcoming_dir.exists():
        print("⚠️ No upcoming_races directory found")
        return []

    race_files = list(upcoming_dir.glob("*.csv"))
    print(f"📁 Found {len(race_files)} upcoming race files")
    return race_files


def demonstrate_weather_predictions():
    """Demonstrate weather-enhanced predictions"""
    print("🚀 Weather-Enhanced Predictor Demo")
    print("=" * 50)

    # Initialize the weather-enhanced predictor
    predictor = WeatherEnhancedPredictor()

    # Find available race files
    race_files = find_upcoming_race_files()

    if not race_files:
        print("⚠️ No race files found for demonstration")
        return

    # Process up to 3 race files for demonstration
    demo_files = race_files[:3]

    results = []

    for i, race_file in enumerate(demo_files, 1):
        print(f"\n🏁 DEMO RACE {i}: {race_file.name}")
        print("-" * 40)

        try:
            # Make weather-enhanced prediction
            result = predictor.predict_race_file_with_weather(str(race_file))

            if result["success"]:
                # Extract key information
                race_info = result["summary"]["race_info"]
                weather_info = result["summary"]["weather_info"]
                predictions = result["predictions"]

                print(f"📍 Venue: {race_info['venue']}")
                print(f"📅 Date: {race_info['race_date']}")

                # Show weather information
                if weather_info and "note" not in weather_info:
                    print(f"🌤️ Weather: {weather_info['weather_condition']}")
                    print(f"🌡️ Temperature: {weather_info['temperature']}°C")
                    print(f"💧 Humidity: {weather_info.get('humidity', 'N/A')}%")
                    print(f"💨 Wind: {weather_info.get('wind_speed', 'N/A')} km/h")
                    weather_adj = result["summary"]["race_summary"][
                        "weather_adjustment_applied"
                    ]
                    print(
                        f"⚡ Weather adjustment: {weather_adj:.3f} ({((weather_adj - 1) * 100):+.1f}%)"
                    )
                else:
                    print("🌤️ Weather: No data available")

                print(f"\n🏆 TOP 3 PREDICTIONS:")
                for j, pred in enumerate(predictions[:3], 1):
                    base_score = pred.get(
                        "base_prediction_score", pred["prediction_score"]
                    )
                    final_score = pred["prediction_score"]
                    weather_exp = pred.get("weather_experience", 0)

                    print(f"  {j}. {pred['dog_name']} (Box {pred['box_number']})")

                    if "base_prediction_score" in pred:
                        print(f"     Base: {base_score:.3f} → Final: {final_score:.3f}")
                        print(f"     Weather Experience: {weather_exp} races")
                    else:
                        print(f"     Score: {final_score:.3f}")

                    print(f"     Confidence: {pred['confidence']:.2f}")

                # Store result for summary
                results.append(
                    {
                        "file": race_file.name,
                        "venue": race_info["venue"],
                        "weather_available": weather_info
                        and "note" not in weather_info,
                        "weather_condition": (
                            weather_info.get("weather_condition", "Unknown")
                            if weather_info
                            else "Unknown"
                        ),
                        "top_pick": (
                            predictions[0]["dog_name"] if predictions else "None"
                        ),
                        "weather_adjustment": result["summary"]["race_summary"][
                            "weather_adjustment_applied"
                        ],
                    }
                )

            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error processing {race_file.name}: {e}")

    # Summary
    if results:
        print(f"\n📊 DEMONSTRATION SUMMARY")
        print("=" * 50)

        weather_available_count = sum(1 for r in results if r["weather_available"])

        print(f"🏁 Races processed: {len(results)}")
        print(f"🌤️ Races with weather data: {weather_available_count}")
        print(
            f"📈 Weather data coverage: {(weather_available_count/len(results)*100):.1f}%"
        )

        print(f"\n🎯 RACE RESULTS:")
        for r in results:
            weather_status = "✅" if r["weather_available"] else "❌"
            adj_str = (
                f"{r['weather_adjustment']:.3f}"
                if r["weather_adjustment"] != 1.0
                else "No adj"
            )
            print(f"  {r['venue']}: Top pick: {r['top_pick']}")
            print(
                f"    Weather: {weather_status} {r['weather_condition']} (Adj: {adj_str})"
            )

        # Weather impact analysis
        weather_adjustments = [
            r["weather_adjustment"] for r in results if r["weather_available"]
        ]
        if weather_adjustments:
            avg_adjustment = sum(weather_adjustments) / len(weather_adjustments)
            print(f"\n⚡ WEATHER IMPACT ANALYSIS:")
            print(f"  Average weather adjustment: {avg_adjustment:.3f}")
            print(
                f"  Impact range: {min(weather_adjustments):.3f} to {max(weather_adjustments):.3f}"
            )

            positive_impact = sum(1 for adj in weather_adjustments if adj > 1.0)
            negative_impact = sum(1 for adj in weather_adjustments if adj < 1.0)

            print(f"  Positive weather conditions: {positive_impact}")
            print(f"  Negative weather conditions: {negative_impact}")


def compare_regular_vs_weather_enhanced():
    """Compare regular predictions vs weather-enhanced predictions"""
    print(f"\n🔄 COMPARISON: Regular vs Weather-Enhanced")
    print("=" * 50)

    # This would require implementing a comparison between the regular
    # comprehensive_ml_predictor and the weather_enhanced_predictor
    print("⚙️ This feature could be implemented to show side-by-side comparisons")
    print("   showing how weather conditions affect the final predictions.")


def main():
    """Main demonstration function"""
    print("🌤️ WEATHER-ENHANCED GREYHOUND RACE PREDICTOR DEMO")
    print("=" * 60)
    print("This demo shows how weather conditions are integrated into race predictions")
    print("=" * 60)

    # Run the main demonstration
    demonstrate_weather_predictions()

    # Optional: Compare regular vs enhanced
    compare_regular_vs_weather_enhanced()

    print(f"\n✅ Demo completed!")
    print("💡 Try running individual predictions with:")
    print("   python weather_enhanced_predictor.py <race_file.csv>")


if __name__ == "__main__":
    main()
