#!/usr/bin/env python3
"""
Run Calibration
================
This script performs calibration on model predictions and evaluates performance metrics including Brier score.
"""

import logging
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss
from probability_calibrator import ProbabilityCalibrator
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO)

# Define threshold for failing the workflow
BRIER_SCORE_THRESHOLD = 0.25


def run_calibration(model_path: str) -> None:
    """
    Run calibration process using the Probability Calibrator.

    :param model_path: Path to the model to calibrate.
    """
    logging.info(f"🎯 Running calibration verification for model: {model_path}")
    logging.info("=" * 60)

    # Initialize calibrator
    calibrator = ProbabilityCalibrator()
    
    # Load calibration data
    cal_data = calibrator._load_calibration_data()
    
    if cal_data.empty:
        logging.error("❌ No calibration data available")
        exit(1)
    
    logging.info(f"📊 Loaded {len(cal_data)} records for calibration evaluation")
    
    # Evaluate win probability calibration
    win_brier_score = None
    place_brier_score = None
    reliability_slope_win = None
    reliability_slope_place = None
    
    if ('raw_win_prob' in cal_data.columns and 
        'actual_win' in cal_data.columns):
        
        win_mask = ~(np.isnan(cal_data['raw_win_prob']) | np.isnan(cal_data['actual_win']))
        if win_mask.sum() > 10:
            raw_win_probs = cal_data.loc[win_mask, 'raw_win_prob'].values
            actual_wins = cal_data.loc[win_mask, 'actual_win'].values
            
            # Calculate raw Brier score
            win_brier_score = brier_score_loss(actual_wins, raw_win_probs)
            
            # Apply calibration if calibrator exists
            if calibrator.win_calibrator is not None:
                calibrated_win_probs = calibrator.win_calibrator.predict(raw_win_probs)
                calibrated_win_brier = brier_score_loss(actual_wins, calibrated_win_probs)
                
                logging.info(f"🎯 Win Probability Metrics:")
                logging.info(f"   Raw Brier Score: {win_brier_score:.4f}")
                logging.info(f"   Calibrated Brier Score: {calibrated_win_brier:.4f}")
                logging.info(f"   Improvement: {((win_brier_score - calibrated_win_brier) / win_brier_score * 100):.2f}%")
                
                win_brier_score = calibrated_win_brier
                
                # Calculate reliability slope (calibration curve)
                try:
                    fraction_pos, mean_pred = calibration_curve(actual_wins, calibrated_win_probs, n_bins=10)
                    # Calculate slope of the reliability diagram
                    if len(fraction_pos) > 1:
                        reliability_slope_win = np.polyfit(mean_pred, fraction_pos, 1)[0]
                        logging.info(f"   Reliability Slope: {reliability_slope_win:.4f}")
                        
                        # Plot calibration curve for win probabilities
                        plt.figure(figsize=(8, 6))
                        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                        plt.plot(mean_pred, fraction_pos, 'o-', label='Win probability calibration')
                        plt.xlabel('Mean Predicted Probability')
                        plt.ylabel('Fraction of Positives')
                        plt.title('Calibration Curve - Win Probabilities')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig('calibration_curve_win.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        logging.info("   📊 Win probability calibration curve saved to calibration_curve_win.png")
                except Exception as e:
                    logging.warning(f"Could not calculate reliability slope for win probabilities: {e}")
            else:
                logging.info(f"🎯 Win Probability Metrics (No Calibration):")
                logging.info(f"   Raw Brier Score: {win_brier_score:.4f}")
    
    # Evaluate place probability calibration
    if ('raw_place_prob' in cal_data.columns and 
        'actual_place' in cal_data.columns):
        
        place_mask = ~(np.isnan(cal_data['raw_place_prob']) | np.isnan(cal_data['actual_place']))
        if place_mask.sum() > 10:
            raw_place_probs = cal_data.loc[place_mask, 'raw_place_prob'].values
            actual_places = cal_data.loc[place_mask, 'actual_place'].values
            
            # Calculate raw Brier score
            place_brier_score = brier_score_loss(actual_places, raw_place_probs)
            
            # Apply calibration if calibrator exists
            if calibrator.place_calibrator is not None:
                calibrated_place_probs = calibrator.place_calibrator.predict(raw_place_probs)
                calibrated_place_brier = brier_score_loss(actual_places, calibrated_place_probs)
                
                logging.info(f"🎯 Place Probability Metrics:")
                logging.info(f"   Raw Brier Score: {place_brier_score:.4f}")
                logging.info(f"   Calibrated Brier Score: {calibrated_place_brier:.4f}")
                logging.info(f"   Improvement: {((place_brier_score - calibrated_place_brier) / place_brier_score * 100):.2f}%")
                
                place_brier_score = calibrated_place_brier
                
                # Calculate reliability slope (calibration curve)
                try:
                    fraction_pos, mean_pred = calibration_curve(actual_places, calibrated_place_probs, n_bins=10)
                    # Calculate slope of the reliability diagram
                    if len(fraction_pos) > 1:
                        reliability_slope_place = np.polyfit(mean_pred, fraction_pos, 1)[0]
                        logging.info(f"   Reliability Slope: {reliability_slope_place:.4f}")
                        
                        # Plot calibration curve for place probabilities
                        plt.figure(figsize=(8, 6))
                        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                        plt.plot(mean_pred, fraction_pos, 'o-', label='Place probability calibration')
                        plt.xlabel('Mean Predicted Probability')
                        plt.ylabel('Fraction of Positives')
                        plt.title('Calibration Curve - Place Probabilities')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig('calibration_curve_place.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        logging.info("   📊 Place probability calibration curve saved to calibration_curve_place.png")
                except Exception as e:
                    logging.warning(f"Could not calculate reliability slope for place probabilities: {e}")
            else:
                logging.info(f"🎯 Place Probability Metrics (No Calibration):")
                logging.info(f"   Raw Brier Score: {place_brier_score:.4f}")
    
    # Generate calibration artifacts
    artifacts = {
        "model_path": model_path,
        "win_brier_score": float(win_brier_score) if win_brier_score is not None else None,
        "place_brier_score": float(place_brier_score) if place_brier_score is not None else None,
        "reliability_slope_win": float(reliability_slope_win) if reliability_slope_win is not None else None,
        "reliability_slope_place": float(reliability_slope_place) if reliability_slope_place is not None else None,
        "threshold": BRIER_SCORE_THRESHOLD,
        "calibration_records": len(cal_data)
    }
    
    # Save artifacts to file
    with open("calibration_results.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    logging.info("📁 Calibration artifacts saved to calibration_results.json")
    
    # Check Brier score threshold
    overall_brier = win_brier_score if win_brier_score is not None else place_brier_score
    
    if overall_brier is None:
        logging.error("❌ No valid Brier score calculated. Cannot evaluate threshold.")
        exit(1)
    
    logging.info("=" * 60)
    logging.info(f"📊 CALIBRATION VERIFICATION RESULTS:")
    logging.info(f"   Overall Brier Score: {overall_brier:.4f}")
    logging.info(f"   Threshold: {BRIER_SCORE_THRESHOLD}")
    
    if overall_brier > BRIER_SCORE_THRESHOLD:
        logging.error(f"❌ WORKFLOW FAILED: Brier score {overall_brier:.4f} exceeds threshold {BRIER_SCORE_THRESHOLD}")
        exit(1)
    else:
        logging.info(f"✅ WORKFLOW PASSED: Brier score {overall_brier:.4f} is within acceptable range")
    
    logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run calibration on model predictions.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()

    run_calibration(args.model_path)


if __name__ == "__main__":
    main()
