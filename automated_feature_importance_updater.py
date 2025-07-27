#!/usr/bin/env python3
"""
Automated Feature Importance Updater
====================================

This script automatically updates prediction models and analysis systems
with the latest feature importance insights each time improvements are run.

Features:
- Automatic feature importance analysis
- Dynamic weight adjustment in prediction pipeline
- Feature selection optimization
- Model parameter updates
- Configuration file updates
- Backup and rollback capabilities

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import json
import shutil
import re
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedFeatureImportanceUpdater:
    def __init__(self, config_file="feature_importance_config.json"):
        self.config_file = config_file
        self.backup_dir = Path("./feature_importance_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Key files to update
        self.target_files = {
            'prediction_pipeline': 'comprehensive_prediction_pipeline.py',
            'ml_system': 'comprehensive_enhanced_ml_system.py',
            'weather_predictor': 'weather_enhanced_predictor.py',
            'traditional_analysis': 'traditional_analysis.py'
        }
        
        # Feature importance thresholds
        self.importance_threshold = 0.05  # Features below this are deprioritized
        self.stability_threshold = 0.3    # CV above this is considered unstable
        
        print("🔄 Automated Feature Importance Updater Initialized")
    
    def run_feature_importance_analysis(self):
        """Run the feature importance analyzer and get latest results"""
        print("📊 Running feature importance analysis...")
        
        try:
            # Import and run the analyzer
            from feature_importance_analyzer import FeatureImportanceAnalyzer
            
            analyzer = FeatureImportanceAnalyzer()
            result = analyzer.run_comprehensive_analysis()
            
            if result:
                print("✅ Feature importance analysis completed")
                return self.load_latest_results()
            else:
                print("❌ Feature importance analysis failed")
                return None
                
        except Exception as e:
            print(f"❌ Error running feature importance analysis: {e}")
            return None
    
    def load_latest_results(self):
        """Load the most recent feature importance results"""
        try:
            results_dir = Path("./feature_analysis_results")
            
            # Find the latest report file
            report_files = list(results_dir.glob("feature_importance_report_*.json"))
            if not report_files:
                print("❌ No feature importance reports found")
                return None
            
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                results = json.load(f)
            
            print(f"📈 Loaded latest results from {latest_report.name}")
            return results
            
        except Exception as e:
            print(f"❌ Error loading feature importance results: {e}")
            return None
    
    def extract_feature_insights(self, results):
        """Extract actionable insights from feature importance results"""
        if not results or 'feature_stability_analysis' not in results:
            return None
        
        # Get Random Forest results (most reliable)
        rf_results = results['feature_stability_analysis'].get('random_forest', {})
        
        feature_names = rf_results.get('feature_names', [])
        mean_importance = rf_results.get('mean_importance', [])
        cv_importance = rf_results.get('cv_importance', [])
        
        if not all([feature_names, mean_importance, cv_importance]):
            return None
        
        # Create feature insights
        insights = {
            'high_impact_features': [],
            'stable_features': [],
            'unstable_features': [],
            'low_impact_features': [],
            'recommended_weights': {},
            'features_to_monitor': []
        }
        
        for i, (feature, importance, cv) in enumerate(zip(feature_names, mean_importance, cv_importance)):
            if importance is None or np.isnan(importance):
                continue
                
            if cv is None or np.isnan(cv):
                cv = 1.0  # Assume unstable if CV is unknown
            
            # Categorize features
            if importance > self.importance_threshold and cv < self.stability_threshold:
                insights['high_impact_features'].append(feature)
                insights['recommended_weights'][feature] = min(importance * 1.2, 1.0)
            elif cv < self.stability_threshold:
                insights['stable_features'].append(feature)
                insights['recommended_weights'][feature] = importance
            elif cv > self.stability_threshold:
                insights['unstable_features'].append(feature)
                insights['recommended_weights'][feature] = importance * 0.7  # Reduce weight
                insights['features_to_monitor'].append(feature)
            else:
                insights['low_impact_features'].append(feature)
                insights['recommended_weights'][feature] = importance * 0.5
        
        # Sort by importance
        insights['high_impact_features'] = sorted(
            insights['high_impact_features'], 
            key=lambda x: mean_importance[feature_names.index(x)], 
            reverse=True
        )
        
        return insights
    
    def backup_current_files(self):
        """Backup current configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        print(f"💾 Creating backup in {backup_subdir}")
        
        for file_type, filename in self.target_files.items():
            if os.path.exists(filename):
                backup_path = backup_subdir / filename
                shutil.copy2(filename, backup_path)
                print(f"  ✅ Backed up {filename}")
        
        return backup_subdir
    
    def update_prediction_pipeline_weights(self, insights):
        """Update the comprehensive prediction pipeline with new feature weights"""
        pipeline_file = self.target_files['prediction_pipeline']
        
        if not os.path.exists(pipeline_file):
            print(f"❌ Pipeline file {pipeline_file} not found")
            return False
        
        try:
            with open(pipeline_file, 'r') as f:
                content = f.read()
            
            # Find the weights section and update it
            weight_section_start = content.find("weights = {")
            if weight_section_start == -1:
                print("❌ Could not find weights section in pipeline")
                return False
            
            # Extract high-impact features for priority weighting
            high_impact = insights.get('high_impact_features', [])
            stable_features = insights.get('stable_features', [])
            
            # Create new weight configuration
            new_weights = self.generate_updated_weights(high_impact, stable_features)
            
            # Update the file content
            updated_content = self.update_weights_in_content(content, new_weights)
            
            with open(pipeline_file, 'w') as f:
                f.write(updated_content)
            
            print(f"✅ Updated {pipeline_file} with new feature weights")
            return True
            
        except Exception as e:
            print(f"❌ Error updating pipeline weights: {e}")
            return False
    
    def generate_updated_weights(self, high_impact_features, stable_features):
        """Generate updated weight configuration based on feature importance"""
        
        # Determine if market indicators are dominant
        market_features = ['current_odds_log', 'market_confidence']
        market_dominant = any(feat in high_impact_features[:3] for feat in market_features)
        
        # Determine if traditional features are strong
        traditional_features = ['avg_position', 'win_rate', 'place_rate', 'recent_form_avg']
        traditional_strong = any(feat in high_impact_features[:5] for feat in traditional_features)
        
        # Adjust weights based on findings
        if market_dominant:
            # Market data is most predictive
            weights = {
                'ml_system': 0.30,        # Slightly reduce ML
                'weather_enhanced': 0.20,  # Reduce weather
                'enhanced_data': 0.15,     # Reduce enhanced data
                'traditional': 0.35        # Increase traditional (includes market data)
            }
        elif traditional_strong:
            # Traditional analysis is strong
            weights = {
                'ml_system': 0.40,        # Increase ML
                'weather_enhanced': 0.20,  # Keep weather same
                'enhanced_data': 0.15,     # Reduce enhanced data
                'traditional': 0.25        # Balanced traditional
            }
        else:
            # Balanced approach
            weights = {
                'ml_system': 0.35,
                'weather_enhanced': 0.25,
                'enhanced_data': 0.20,
                'traditional': 0.20
            }
        
        return weights
    
    def update_weights_in_content(self, content, new_weights):
        """Update the weights section in file content"""
        
        # Find and replace the weights dictionary
        pattern = r'weights = \{[^}]+\}'
        
        new_weights_str = "weights = {\n"
        for method, weight in new_weights.items():
            new_weights_str += f"            '{method}': {weight}"
            if method != 'traditional':
                new_weights_str += " * data_quality,\n"
            else:
                new_weights_str += "  # Always available baseline\n"
        new_weights_str += "        }"
        
        updated_content = re.sub(pattern, new_weights_str, content)
        return updated_content
    
    def update_ml_system_features(self, insights):
        """Update ML system with feature selection based on importance"""
        ml_file = self.target_files['ml_system']
        
        if not os.path.exists(ml_file):
            print(f"❌ ML system file {ml_file} not found")
            return False
        
        try:
            with open(ml_file, 'r') as f:
                content = f.read()
            
            # Update high impact features list
            high_impact = insights.get('high_impact_features', [])[:8]  # Top 8
            stable_features = insights.get('stable_features', [])[:8]   # Top 8 stable
            
            # Find and update high_impact_features
            pattern = r'self\.high_impact_features = \[[^\]]+\]'
            new_features_str = f"self.high_impact_features = {high_impact}"
            content = re.sub(pattern, new_features_str, content)
            
            # Find and update stable_features
            pattern = r'self\.stable_features = \[[^\]]+\]'
            new_stable_str = f"self.stable_features = {stable_features}"
            content = re.sub(pattern, new_stable_str, content)
            
            with open(ml_file, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {ml_file} with new feature priorities")
            return True
            
        except Exception as e:
            print(f"❌ Error updating ML system features: {e}")
            return False
    
    def create_configuration_update_log(self, insights, backup_dir):
        """Create a log of configuration changes made"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(backup_dir),
            'feature_insights': insights,
            'changes_made': [],
            'files_updated': []
        }
        
        # Save the log
        log_file = self.backup_dir / f"update_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Update log saved to {log_file}")
        return log_file
    
    def run_automated_update(self):
        """Run the complete automated update process"""
        print("🚀 Starting automated feature importance update process...")
        print("=" * 60)
        
        # Step 1: Run feature importance analysis
        results = self.run_feature_importance_analysis()
        if not results:
            print("❌ Cannot proceed without feature importance results")
            return False
        
        # Step 2: Extract insights
        print("🔍 Extracting actionable insights...")
        insights = self.extract_feature_insights(results)
        if not insights:
            print("❌ Could not extract insights from results")
            return False
        
        print(f"📊 Found {len(insights['high_impact_features'])} high-impact features")
        print(f"📊 Found {len(insights['stable_features'])} stable features")
        print(f"⚠️  Found {len(insights['unstable_features'])} unstable features")
        
        # Step 3: Backup current files
        backup_dir = self.backup_current_files()
        
        # Step 4: Update prediction pipeline
        print("🔧 Updating prediction pipeline weights...")
        pipeline_updated = self.update_prediction_pipeline_weights(insights)
        
        # Step 5: Update ML system
        print("🔧 Updating ML system feature priorities...")
        ml_updated = self.update_ml_system_features(insights)
        
        # Step 6: Create update log
        log_file = self.create_configuration_update_log(insights, backup_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 UPDATE SUMMARY:")
        print(f"✅ Feature analysis: Completed")
        print(f"✅ Backup created: {backup_dir}")
        print(f"{'✅' if pipeline_updated else '❌'} Pipeline updated: {pipeline_updated}")
        print(f"{'✅' if ml_updated else '❌'} ML system updated: {ml_updated}")
        print(f"✅ Update log: {log_file}")
        
        if pipeline_updated and ml_updated:
            print("\n🎉 Automated update completed successfully!")
            print("Your prediction system has been optimized with the latest feature importance insights.")
            return True
        else:
            print("\n⚠️  Partial update completed. Some components may need manual attention.")
            return False

def main():
    """Main function to run the automated updater"""
    updater = AutomatedFeatureImportanceUpdater()
    success = updater.run_automated_update()
    
    if success:
        print("\n🔄 Your prediction system is now optimized!")
        print("Run your predictions to see the improvements.")
    else:
        print("\n❌ Update process encountered issues. Check the logs above.")
    
    return success

if __name__ == "__main__":
    main()
