#!/usr/bin/env python3
"""
Example usage of the ROI-optimized ensemble system.

This script demonstrates:
1. How to optimize ensemble weights based on historical ROI
2. How to train an ensemble with ROI-optimized weights
3. How to use the ensemble for predictions
4. How to persist and load weights
"""

import logging

from advanced_ensemble_ml_system import AdvancedEnsembleMLSystem
from ensemble_roi_weighter import optimize_ensemble_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("🎯 ROI-Optimized Ensemble Example")
    logger.info("=" * 50)

    # Step 1: Optimize ensemble weights based on historical data
    logger.info("\n📊 Step 1: Computing ROI-optimized weights...")
    try:
        weights = optimize_ensemble_weights(
            db_path="greyhound_racing_data.db",
            output_path="models/ensemble_weights.json",
            limit_records=5000,  # Use recent 5000 records
        )
        logger.info(f"✅ Optimized weights computed: {weights}")
    except Exception as e:
        logger.error(f"❌ Weight optimization failed: {e}")
        return

    # Step 2: Train ensemble with ROI-optimized weights
    logger.info("\n🚀 Step 2: Training ensemble with ROI-optimized weights...")
    try:
        ensemble_system = AdvancedEnsembleMLSystem()
        success = ensemble_system.train_ensemble(["gradient_boosting", "random_forest"])

        if success:
            logger.info("✅ Ensemble training completed successfully!")

            # Display model info
            model_info = ensemble_system.get_model_info()
            logger.info(
                f"📈 Ensemble ROC AUC: {model_info.get('ensemble_roc_auc', 'N/A'):.4f}"
            )
            logger.info(f"🔧 Base models: {model_info.get('base_models', [])}")
            logger.info(f"⚖️ ROI weights: {ensemble_system.ensemble_weights}")

        else:
            logger.error("❌ Ensemble training failed")
            return

    except Exception as e:
        logger.error(f"❌ Ensemble training error: {e}")
        return

    # Step 3: Save the ensemble (weights are automatically saved)
    logger.info("\n💾 Step 3: Saving ensemble model...")
    try:
        model_path = ensemble_system.save_ensemble()
        logger.info(f"✅ Ensemble saved to: {model_path}")
        logger.info("✅ ROI weights saved to: models/ensemble_weights.json")
    except Exception as e:
        logger.error(f"❌ Error saving ensemble: {e}")
        return

    # Step 4: Make predictions with the ROI-optimized ensemble
    logger.info("\n🔮 Step 4: Making predictions...")
    try:
        # Example dog data
        test_dogs = [
            {
                "name": "Speed Champion",
                "box_number": 1,
                "weight": 31.5,
                "starting_price": 2.80,
                "individual_time": 29.30,
                "field_size": 8,
            },
            {
                "name": "Track Star",
                "box_number": 4,
                "weight": 33.0,
                "starting_price": 4.20,
                "individual_time": 30.10,
                "field_size": 8,
            },
        ]

        for i, dog in enumerate(test_dogs, 1):
            prediction = ensemble_system.predict(dog, market_odds=dog["starting_price"])

            logger.info(f"\n🐕 Dog {i}: {dog['name']}")
            logger.info(f"   Win Probability: {prediction['win_probability']:.3f}")
            logger.info(f"   Confidence: {prediction['confidence']:.3f}")
            logger.info(f"   Model: {prediction['model_info']}")

            if "betting_recommendation" in prediction:
                bet_rec = prediction["betting_recommendation"]
                logger.info(f"   Betting Recommendation: {bet_rec['bet_type']}")
                if bet_rec["has_value"]:
                    logger.info(f"   Expected Value: {bet_rec['expected_value']:.3f}")
                    logger.info(
                        f"   Recommended Stake: {bet_rec['recommended_stake']:.2%}"
                    )

        logger.info("\n🎉 ROI-optimized ensemble example completed successfully!")

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return


if __name__ == "__main__":
    main()
