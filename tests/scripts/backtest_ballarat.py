import logging
import pandas as pd
from step5_probability_converter import ProbabilityConverter
from ml_system_v4 import MLSystemV4
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackTestBallarat:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ml_system = MLSystemV4(db_path=db_path)
        self.converter = ProbabilityConverter()

    def run_backtest(self):
        # Load data
        logger.info("Loading test data...")
        _, test_data = self.ml_system.prepare_time_ordered_data()

        # Build features
        logger.info("Building features for test data...")
        test_features = self.ml_system.build_leakage_safe_features(test_data)
        X_test = test_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
        y_test = test_features['target']

        # Predict probabilities
        logger.info("Predicting probabilities...")
        predicted_proba = self.ml_system.calibrated_pipeline.predict_proba(X_test)[:, 1]

        # Compute log-loss and brier score
        log_loss_value = log_loss(y_test, predicted_proba)
        brier_score = brier_score_loss(y_test, predicted_proba)
        logger.info(f"Log Loss: {log_loss_value:.4f}")
        logger.info(f"Brier Score: {brier_score:.4f}")

        # Plot calibration curve
        logger.info("Generating calibration plot...")
        prob_true, prob_pred = calibration_curve(y_test, predicted_proba, n_bins=10)
        plt.figure(figsize=(10, 8))
        plt.plot(prob_pred, prob_true, marker=".", label="Ballarat Test")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Optionally apply isotonic regression
        logger.info("Applying isotonic regression...")
        self.ml_system.calibrated_pipeline = CalibratedClassifierCV(self.ml_system.calibrated_pipeline, method='isotonic', cv=5)
        self.ml_system.calibrated_pipeline.fit(X_test, y_test)

        # Re-evaluate after isotonic regression
        predicted_proba_adjusted = self.ml_system.calibrated_pipeline.predict_proba(X_test)[:, 1]
        log_loss_adj = log_loss(y_test, predicted_proba_adjusted)
        brier_score_adj = brier_score_loss(y_test, predicted_proba_adjusted)
        logger.info(f"Adjusted Log Loss: {log_loss_adj:.4f}")
        logger.info(f"Adjusted Brier Score: {brier_score_adj:.4f}")

if __name__ == "__main__":
    db_path = "greyhound_racing_data.db"
    backtester = BackTestBallarat(db_path=db_path)
    backtester.run_backtest()
