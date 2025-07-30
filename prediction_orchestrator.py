#!/usr/bin/env python3
"""
Prediction Orchestrator - Intelligent Prediction and Betting Engine
===================================================================

This orchestrator is the central brain for greyhound race predictions.
It intelligently manages multiple prediction systems, provides detailed
analysis, and integrates professional-grade betting strategy optimization.

Key Features:
- Dynamic selection of the best available ML model
- Detailed prediction breakdowns (ensemble + individual models)
- Tunable, risk-aware betting strategies
- Graceful error handling and system resilience
- Comprehensive logging and performance monitoring
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import json

# Import all available prediction systems
from ml_system_v3 import MLSystemV3
from advanced_ensemble_ml_system import AdvancedEnsembleMLSystem, BettingStrategyOptimizer

logger = logging.getLogger(__name__)

class PredictionOrchestrator:
    """
    The central orchestrator for the greyhound prediction system.
    """
    
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.advanced_ensemble_system = None
        self.base_ml_system = None
        self.active_system = None
        
        self._initialize_systems()
        self._select_active_system()
        
        logger.info("🚀 Prediction Orchestrator Initialized")
        logger.info(f"   Active System: {self.get_active_system_name()}")
    
    def _initialize_systems(self):
        """Initialize all available prediction systems."""
        try:
            # Try to load the advanced ensemble model first
            advanced_model_dir = Path('./advanced_models')
            if advanced_model_dir.exists():
                model_files = list(advanced_model_dir.glob('*.joblib'))
                if model_files:
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    self.advanced_ensemble_system = AdvancedEnsembleMLSystem(self.db_path)
                    if self.advanced_ensemble_system.load_ensemble(latest_model):
                        logger.info(f"✅ Loaded advanced ensemble: {latest_model.name}")
                    else:
                        self.advanced_ensemble_system = None
        except Exception as e:
            logger.error(f"Error initializing advanced ensemble system: {e}")
        
        # Initialize the base ML system as a fallback
        try:
            self.base_ml_system = MLSystemV3(self.db_path)
            if not self.base_ml_system.pipeline:
                logger.warning("Base ML system has no trained model. Training one now...")
                self.base_ml_system.train_model()
            logger.info("✅ Initialized base ML system (v3)")
        except Exception as e:
            logger.error(f"Error initializing base ML system: {e}")

    def _select_active_system(self):
        """Select the best available system as the active one."""
        if self.advanced_ensemble_system:
            self.active_system = self.advanced_ensemble_system
            logger.info("🎯 Selected Advanced Ensemble as active system")
        elif self.base_ml_system:
            self.active_system = self.base_ml_system
            logger.info("🎯 Selected Base ML System as active system")
        else:
            raise RuntimeError("CRITICAL: No prediction systems could be initialized.")
    
    def get_active_system_name(self):
        """Get the name of the currently active prediction system."""
        if isinstance(self.active_system, AdvancedEnsembleMLSystem):
            return "AdvancedEnsembleMLSystem"
        elif isinstance(self.active_system, MLSystemV3):
            return "MLSystemV3"
        return "None"

    def predict_race(self, dog_data, market_odds=None, betting_strategy=None):
        """
        Make a prediction for a single dog, with detailed analysis and betting recommendation.
        
        Args:
            dog_data: Dictionary containing dog information
            market_odds: Market odds for the dog (optional)
            betting_strategy: Custom betting strategy parameters (optional)
        """
        if not self.active_system:
            return self._error_response("No active prediction system.")
            
        try:
            # Use the active system for prediction
            if isinstance(self.active_system, AdvancedEnsembleMLSystem):
                prediction = self.active_system.predict(dog_data, market_odds)
            else: # Fallback to MLSystemV3
                prediction = self.active_system.predict(dog_data)
                if market_odds:
                    # Add betting recommendation using default optimizer
                    optimizer = self._create_betting_optimizer(betting_strategy)
                    betting_rec = optimizer.calculate_betting_value(
                        prediction['win_probability'],
                        market_odds,
                        prediction['confidence']
                    )
                    prediction['betting_recommendation'] = betting_rec
            
            # Override betting recommendation if custom strategy provided
            if betting_strategy and market_odds:
                optimizer = self._create_betting_optimizer(betting_strategy)
                betting_rec = optimizer.calculate_betting_value(
                    prediction['win_probability'],
                    market_odds,
                    prediction['confidence']
                )
                prediction['betting_recommendation'] = betting_rec

            return {
                "success": True,
                "prediction_system": self.get_active_system_name(),
                "dog_name": dog_data.get('name', 'Unknown'),
                "market_odds": market_odds,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during race prediction: {e}", exc_info=True)
            return self._error_response(str(e))
    
    def _create_betting_optimizer(self, betting_strategy=None):
        """Create a betting optimizer with custom parameters."""
        optimizer = BettingStrategyOptimizer()
        
        if betting_strategy:
            if 'kelly_multiplier' in betting_strategy:
                optimizer.kelly_multiplier = betting_strategy['kelly_multiplier']
            if 'min_edge' in betting_strategy:
                optimizer.min_edge = betting_strategy['min_edge']
            if 'max_bet_size' in betting_strategy:
                optimizer.max_bet_size = betting_strategy['max_bet_size']
            if 'confidence_threshold' in betting_strategy:
                optimizer.confidence_threshold = betting_strategy['confidence_threshold']
        
        return optimizer
    
    def compare_betting_strategies(self, dog_data, market_odds, strategies):
        """
        Compare multiple betting strategies for the same dog/race.
        
        Args:
            dog_data: Dog information
            market_odds: Market odds
            strategies: Dictionary of strategy names and parameters
        """
        results = {}
        
        for strategy_name, strategy_params in strategies.items():
            result = self.predict_race(dog_data, market_odds, strategy_params)
            if result['success']:
                results[strategy_name] = result['prediction']['betting_recommendation']
        
        return results
    
    def get_system_status(self):
        """Get the status of all managed prediction systems."""
        return {
            "active_system": self.get_active_system_name(),
            "advanced_ensemble_status": {
                "initialized": self.advanced_ensemble_system is not None,
                "model_info": self.advanced_ensemble_system.get_model_info() if self.advanced_ensemble_system else None
            },
            "base_ml_system_status": {
                "initialized": self.base_ml_system is not None,
                "model_info": self.base_ml_system.get_model_info() if self.base_ml_system else None
            }
        }

    def _error_response(self, error_message):
        """Generate a standardized error response."""
        return {
            "success": False,
            "error": error_message,
            "prediction": {
                'win_probability': 0,
                'confidence': 0,
                'betting_recommendation': {
                    'has_value': False,
                    'bet_type': 'NO_BET'
                }
            }
        }

def run_comprehensive_prediction_analysis():
    """Run a comprehensive prediction analysis with multiple betting strategies."""
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = PredictionOrchestrator()
    
    print("\n🎯 COMPREHENSIVE PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Test with a sample dog
    test_dog = {
        'name': 'Lightning Bolt',
        'box_number': 1,
        'weight': 30.5,
        'starting_price': 2.80,
        'individual_time': 29.20,
        'field_size': 8,
        'temperature': 20.0,
        'humidity': 60.0,
        'wind_speed': 8.0
    }
    
    market_odds = 4.0  # Better value scenario
    
    print(f"\n🐕 Analyzing: {test_dog['name']}")
    print(f"📊 Market Odds: {market_odds}")
    
    # Define different betting strategies
    strategies = {
        'Conservative': {
            'kelly_multiplier': 0.1,
            'min_edge': 0.10,
            'confidence_threshold': 0.85,
            'max_bet_size': 0.05
        },
        'Moderate': {
            'kelly_multiplier': 0.25,
            'min_edge': 0.05,
            'confidence_threshold': 0.70,
            'max_bet_size': 0.10
        },
        'Aggressive': {
            'kelly_multiplier': 0.5,
            'min_edge': 0.02,
            'confidence_threshold': 0.60,
            'max_bet_size': 0.15
        }
    }
    
    # Run predictions with different strategies
    print(f"\n📈 PREDICTION RESULTS:")
    print("-" * 40)
    
    base_result = orchestrator.predict_race(test_dog, market_odds)
    if base_result['success']:
        pred = base_result['prediction']
        print(f"Win Probability: {pred['win_probability']:.2%}")
        print(f"Model Confidence: {pred['confidence']:.2%}")
        
        if 'individual_predictions' in pred:
            print(f"Individual Model Predictions:")
            for model, prob in pred['individual_predictions'].items():
                print(f"   {model}: {prob:.2%}")
    
    # Compare betting strategies
    print(f"\n💰 BETTING STRATEGY COMPARISON:")
    print("-" * 40)
    
    strategy_results = orchestrator.compare_betting_strategies(test_dog, market_odds, strategies)
    
    for strategy_name, betting_rec in strategy_results.items():
        print(f"\n{strategy_name} Strategy:")
        if betting_rec['has_value']:
            print(f"   ✅ BET RECOMMENDED")
            print(f"   Edge: {betting_rec['edge']:.2%}")
            print(f"   Stake: {betting_rec['recommended_stake']:.2%} of bankroll")
            print(f"   Expected Value: {betting_rec['expected_value']:.3f}")
            print(f"   Bet Type: {betting_rec['bet_type']}")
            print(f"   Risk Level: {betting_rec['risk_level']}")
        else:
            print(f"   ❌ NO BET")
    
    # System status
    print(f"\n🔧 SYSTEM STATUS:")
    print("-" * 40)
    status = orchestrator.get_system_status()
    print(f"Active System: {status['active_system']}")
    
    if status['advanced_ensemble_status']['initialized']:
        model_info = status['advanced_ensemble_status']['model_info']
        print(f"Ensemble ROC AUC: {model_info.get('ensemble_roc_auc', 'N/A')}")
        print(f"Base Models: {model_info.get('base_models', [])}")

if __name__ == "__main__":
    run_comprehensive_prediction_analysis()
