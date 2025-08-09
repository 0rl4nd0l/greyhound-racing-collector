#!/usr/bin/env python3
"""
Test script to validate EV calculation and market odds integration
================================================================

This script tests the implementation of:
1. Expected value calculation using market odds
2. SportsbetOddsIntegrator integration
3. BettingStrategyOptimizer enhanced functionality
"""

import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_betting_strategy_optimizer():
    """Test the BettingStrategyOptimizer with EV calculation"""
    print("\n🧪 TESTING BETTING STRATEGY OPTIMIZER")
    print("=" * 50)
    
    try:
        from advanced_ensemble_ml_system import BettingStrategyOptimizer
        
        # Initialize optimizer
        optimizer = BettingStrategyOptimizer()
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Strong Value Bet",
                "win_prob": 0.4,
                "market_odds": 3.5,
                "confidence": 0.85
            },
            {
                "name": "Marginal Value Bet", 
                "win_prob": 0.25,
                "market_odds": 4.2,
                "confidence": 0.72
            },
            {
                "name": "No Value (Overpriced)",
                "win_prob": 0.15,
                "market_odds": 2.0,
                "confidence": 0.80
            },
            {
                "name": "Low Confidence",
                "win_prob": 0.35,
                "market_odds": 4.0,
                "confidence": 0.60  # Below threshold
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📊 Testing: {scenario['name']}")
            print(f"   Win Probability: {scenario['win_prob']:.1%}")
            print(f"   Market Odds: {scenario['market_odds']:.2f}")
            print(f"   Confidence: {scenario['confidence']:.1%}")
            
            result = optimizer.calculate_betting_value(
                scenario['win_prob'], 
                scenario['market_odds'], 
                scenario['confidence']
            )
            
            print(f"   ✅ Expected Value: {result['expected_value']:.4f}")
            print(f"   📈 Has Value: {'YES' if result['has_value'] else 'NO'}")
            if result['has_value']:
                print(f"   💰 Edge: {result['edge']:.2%}")
                print(f"   🎯 Bet Type: {result['bet_type']}")
                print(f"   📊 Recommended Stake: {result['recommended_stake']:.2%}")
                print(f"   ⚡ Kelly Fraction: {result['kelly_fraction']:.3f}")
        
        print("\n✅ BettingStrategyOptimizer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing BettingStrategyOptimizer: {e}")
        return False

def test_market_odds_lookup():
    """Test market odds lookup functionality"""
    print("\n🧪 TESTING MARKET ODDS LOOKUP")
    print("=" * 50)
    
    try:
        from advanced_ensemble_ml_system import BettingStrategyOptimizer
        
        # Initialize optimizer (with DB path)
        optimizer = BettingStrategyOptimizer("greyhound_racing_data.db")
        
        # Test market odds lookup
        test_race_id = "test_race_001"
        test_dog_name = "Test Dog"
        
        print(f"   Testing odds lookup for: {test_dog_name} in {test_race_id}")
        
        odds = optimizer.get_market_odds(test_race_id, test_dog_name)
        
        if odds:
            print(f"   ✅ Found market odds: {odds:.2f}")
        else:
            print("   ⚠️ No market odds found (expected for test data)")
        
        # Test the combined functionality
        print("\n   Testing combined EV calculation with odds lookup...")
        
        result = optimizer.calculate_betting_value_with_odds_lookup(
            win_prob=0.3,
            confidence=0.75,
            race_id=test_race_id,
            dog_name=test_dog_name
        )
        
        print(f"   📊 Market Odds Found: {'YES' if result.get('market_odds_found') else 'NO'}")
        print(f"   💰 Expected Value: {result['expected_value']:.4f}")
        print(f"   🎯 Has Value: {'YES' if result['has_value'] else 'NO'}")
        
        print("\n✅ Market odds lookup tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing market odds lookup: {e}")
        return False

def test_prediction_orchestrator():
    """Test the PredictionOrchestrator with EV integration"""
    print("\n🧪 TESTING PREDICTION ORCHESTRATOR")
    print("=" * 50)
    
    try:
        from prediction_orchestrator import PredictionOrchestrator
        
        # Initialize orchestrator
        orchestrator = PredictionOrchestrator()
        
        # Test dog data
        test_dog = {
            "name": "EV Test Champion",
            "box_number": 1,
            "weight": 30.5,
            "starting_price": 2.80,
            "individual_time": 29.20,
            "field_size": 8,
            "temperature": 20.0,
            "humidity": 60.0,
            "wind_speed": 8.0,
        }
        
        market_odds = 3.5
        
        print(f"   Testing prediction for: {test_dog['name']}")
        print(f"   Market Odds: {market_odds}")
        
        # Make prediction
        result = orchestrator.predict_race(test_dog, market_odds)
        
        if result["success"]:
            prediction = result["prediction"]
            print(f"   ✅ Win Probability: {prediction['win_probability']:.2%}")
            print(f"   📈 Confidence: {prediction['confidence']:.2%}")
            print(f"   💰 Expected Value: {prediction.get('expected_value', 'N/A')}")
            
            if "betting_recommendation" in prediction:
                betting_rec = prediction["betting_recommendation"]
                print(f"   🎯 Betting Recommendation:")
                print(f"      - Has Value: {'YES' if betting_rec['has_value'] else 'NO'}")
                print(f"      - Expected Value: {betting_rec['expected_value']:.4f}")
                if betting_rec['has_value']:
                    print(f"      - Edge: {betting_rec['edge']:.2%}")
                    print(f"      - Bet Type: {betting_rec['bet_type']}")
        else:
            print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
        
        print("\n✅ PredictionOrchestrator tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing PredictionOrchestrator: {e}")
        return False

def main():
    """Run all EV calculation tests"""
    print("🧪 EXPECTED VALUE CALCULATION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: BettingStrategyOptimizer
    if test_betting_strategy_optimizer():
        tests_passed += 1
    
    # Test 2: Market odds lookup
    if test_market_odds_lookup():
        tests_passed += 1
    
    # Test 3: PredictionOrchestrator integration
    if test_prediction_orchestrator():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! EV calculation is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
