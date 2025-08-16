#!/usr/bin/env python3
"""
Feature Quality & Leakage Assessment for Greyhound Analysis Predictor
====================================================================

Step 7: Feature Quality & Leakage Assessment

1. Catalogue all candidate features currently fed into ML pipelines.  
2. Evaluate each for:  
   • Cardinality / uniqueness  
   • Missingness rate  
   • Predictive utility proxy (e.g., mutual information with target).  
3. Perform leakage scan to ensure post-race information (e.g., final dividends) isn't present at prediction time.  
4. Recommend feature transformations or aggregations where beneficial.
"""

import logging
import sqlite3
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class FeatureQualityAssessment:
    """Comprehensive feature quality and leakage assessment for ML pipeline."""
    
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.assessment_results = {}
        self.leakage_results = {}
        self.feature_catalog = {}
        
        # Define potential leakage fields (post-race information)
        self.potential_leakage_fields = {
            'finish_position',
            'scraped_finish_position', 
            'winner_name',
            'winner_odds',
            'winner_margin',
            'margin',
            'beaten_margin',
            'race_time',
            'individual_time',  # Could be leakage if it's the actual race time
            'sectional_1st',
            'sectional_2nd', 
            'sectional_3rd',
            'prize_money',
            'final_dividends',
            'tote_win',
            'tote_place',
            'actual_odds',
            'final_odds'
        }
        
        # Define pre-race vs post-race feature categories
        self.pre_race_features = {
            'box_number',
            'weight', 
            'starting_price',
            'trainer_name',
            'dog_name',
            'dog_clean_name',
            'venue',
            'grade',
            'distance',
            'track_condition',
            'weather',
            'temperature',
            'humidity',
            'wind_speed',
            'field_size',
            'odds_decimal',
            'odds_fractional'
        }
        
    def load_comprehensive_data(self) -> pd.DataFrame:
        """Load comprehensive data from database for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Comprehensive query joining all relevant tables
            query = """
            SELECT 
                d.*,
                r.venue,
                r.grade,
                r.distance,
                r.track_condition,
                r.weather,
                r.temperature,
                r.humidity,
                r.wind_speed,
                r.field_size,
                r.race_date,
                r.winner_name,
                r.winner_odds,
                r.winner_margin,
                e.pir_rating,
                e.first_sectional,
                e.win_time,
                e.bonus_time
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            LEFT JOIN enhanced_expert_data e ON d.race_id = e.race_id 
                AND d.dog_clean_name = e.dog_clean_name
            WHERE d.race_id IS NOT NULL
            ORDER BY r.race_date DESC
            LIMIT 50000
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} comprehensive race records for assessment")
            logger.info(f"Available columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading comprehensive data: {e}")
            return pd.DataFrame()
    
    def catalogue_all_features(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Catalogue all candidate features from the ML pipeline."""
        logger.info("🔍 Cataloguing all candidate features...")
        
        feature_catalog = {}
        
        # Basic numerical features
        numerical_features = [
            'box_number', 'weight', 'starting_price', 'temperature', 
            'humidity', 'wind_speed', 'field_size', 'odds_decimal',
            'pir_rating', 'first_sectional', 'win_time', 'bonus_time'
        ]
        
        # Categorical features
        categorical_features = [
            'venue', 'grade', 'track_condition', 'weather', 'trainer_name',
            'running_style', 'data_source'
        ]
        
        # Time-based features (potential leakage risk)
        time_features = [
            'individual_time', 'sectional_1st', 'sectional_2nd', 'sectional_3rd',
            'race_time', 'winning_time'
        ]
        
        # Target and outcome features (definite leakage)
        outcome_features = [
            'finish_position', 'scraped_finish_position', 'winner_name',
            'winner_odds', 'winner_margin', 'margin', 'beaten_margin'
        ]
        
        # Derived features (created in ML pipeline)
        derived_features = [
            'price_rank', 'weight_rank', 'time_rank_in_race', 'box_advantage',
            'is_favorite', 'weight_to_field_ratio', 'price_to_field_ratio'
        ]
        
        # Process each feature category
        for feature in numerical_features:
            if feature in data.columns:
                feature_catalog[feature] = self.analyze_feature(data, feature, 'numerical')
        
        for feature in categorical_features:
            if feature in data.columns:
                feature_catalog[feature] = self.analyze_feature(data, feature, 'categorical')
        
        for feature in time_features:
            if feature in data.columns:
                feature_catalog[feature] = self.analyze_feature(data, feature, 'time_based')
        
        for feature in outcome_features:
            if feature in data.columns:
                feature_catalog[feature] = self.analyze_feature(data, feature, 'outcome')
        
        # Add traditional analysis features (from ml_system_v3.py)
        traditional_features = [
            'traditional_overall_score', 'traditional_performance_score',
            'traditional_form_score', 'traditional_class_score',
            'traditional_consistency_score', 'traditional_fitness_score',
            'traditional_experience_score', 'traditional_trainer_score',
            'traditional_track_condition_score', 'traditional_distance_score'
        ]
        
        for feature in traditional_features:
            # These are synthetic features, assign default analysis
            feature_catalog[feature] = {
                'type': 'synthetic_traditional',
                'cardinality': 'low',
                'missing_rate': 0.0,
                'data_type': 'float',
                'leakage_risk': 'low',
                'predictive_utility': 'medium'
            }
        
        self.feature_catalog = feature_catalog
        logger.info(f"✅ Catalogued {len(feature_catalog)} features across all categories")
        
        return feature_catalog
    
    def analyze_feature(self, data: pd.DataFrame, feature: str, category: str) -> Dict:
        """Analyze individual feature quality metrics."""
        if feature not in data.columns:
            return {'error': f'Feature {feature} not found in data'}
        
        feature_data = data[feature]
        analysis = {
            'feature_name': feature,
            'category': category,
            'data_type': str(feature_data.dtype),
            'total_samples': len(feature_data)
        }
        
        # Calculate missingness rate
        missing_count = feature_data.isnull().sum()
        analysis['missing_count'] = int(missing_count)
        analysis['missing_rate'] = float(missing_count / len(feature_data))
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(feature_data):
            analysis.update(self._analyze_numerical_feature(feature_data))
        else:
            analysis.update(self._analyze_categorical_feature(feature_data))
        
        # Leakage risk assessment
        analysis['leakage_risk'] = self._assess_leakage_risk(feature, category)
        
        return analysis
    
    def _analyze_numerical_feature(self, feature_data: pd.Series) -> Dict:
        """Analyze numerical feature characteristics."""
        valid_data = feature_data.dropna()
        
        if len(valid_data) == 0:
            return {'cardinality': 0, 'unique_values': 0}
        
        analysis = {
            'cardinality': 'high' if len(valid_data.unique()) > 100 else 'medium' if len(valid_data.unique()) > 10 else 'low',
            'unique_values': int(len(valid_data.unique())),
            'min_value': float(valid_data.min()),
            'max_value': float(valid_data.max()),
            'mean': float(valid_data.mean()),
            'median': float(valid_data.median()),
            'std': float(valid_data.std()),
            'skewness': float(stats.skew(valid_data)),
            'kurtosis': float(stats.kurtosis(valid_data))
        }
        
        # Detect outliers using IQR method
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = len(valid_data[(valid_data < Q1 - 1.5 * IQR) | (valid_data > Q3 + 1.5 * IQR)])
        analysis['outlier_count'] = int(outlier_count)
        analysis['outlier_rate'] = float(outlier_count / len(valid_data))
        
        return analysis
    
    def _analyze_categorical_feature(self, feature_data: pd.Series) -> Dict:
        """Analyze categorical feature characteristics."""
        valid_data = feature_data.dropna()
        
        if len(valid_data) == 0:
            return {'cardinality': 0, 'unique_values': 0}
        
        value_counts = valid_data.value_counts()
        
        analysis = {
            'cardinality': 'high' if len(value_counts) > 50 else 'medium' if len(value_counts) > 10 else 'low',
            'unique_values': int(len(value_counts)),
            'most_frequent_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
        }
        
        # Calculate entropy for diversity measure
        probabilities = value_counts / len(valid_data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        analysis['entropy'] = float(entropy)
        
        return analysis
    
    def _assess_leakage_risk(self, feature: str, category: str) -> str:
        """Assess data leakage risk for a feature."""
        if feature in self.potential_leakage_fields:
            return 'high'
        elif category == 'outcome':
            return 'high'
        elif category == 'time_based':
            return 'medium'  # Could be leakage if it's actual race performance
        elif feature in self.pre_race_features:
            return 'low'
        else:
            return 'medium'  # Unknown features get medium risk
    
    def calculate_predictive_utility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate predictive utility using mutual information with target."""
        logger.info("📊 Calculating predictive utility (mutual information with target)...")
        
        # Create target variable (did the dog win?)
        if 'finish_position' in data.columns:
            target = pd.to_numeric(data['finish_position'], errors='coerce')
            target = (target == 1).astype(int)
        else:
            logger.warning("No finish_position column found, using winner_name as target")
            target = (data['dog_clean_name'] == data['winner_name']).astype(int)
        
        mutual_info_scores = {}
        
        # Select features for analysis (exclude leakage features)
        analysis_features = []
        for feature in data.columns:
            if (feature not in self.potential_leakage_fields and 
                feature not in ['dog_name', 'dog_clean_name', 'race_id'] and
                not feature.startswith('scraped_')):
                analysis_features.append(feature)
        
        for feature in analysis_features:
            try:
                feature_data = data[feature].copy()
                
                # Handle different data types
                if pd.api.types.is_numeric_dtype(feature_data):
                    # Fill missing values with median
                    feature_data = feature_data.fillna(feature_data.median())
                    feature_data = feature_data.values.reshape(-1, 1)
                else:
                    # Encode categorical variables
                    feature_data = feature_data.fillna('unknown')
                    le = LabelEncoder()
                    feature_data = le.fit_transform(feature_data).reshape(-1, 1)
                
                # Calculate mutual information
                mi_score = mutual_info_classif(feature_data, target, random_state=42)[0]
                mutual_info_scores[feature] = float(mi_score)
                
            except Exception as e:
                logger.warning(f"Could not calculate MI for {feature}: {e}")
                mutual_info_scores[feature] = 0.0
        
        # Sort by predictive utility
        sorted_scores = dict(sorted(mutual_info_scores.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"✅ Calculated predictive utility for {len(sorted_scores)} features")
        return sorted_scores
    
    def perform_leakage_scan(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data leakage scan."""
        logger.info("🚨 Performing comprehensive data leakage scan...")
        
        leakage_results = {
            'high_risk_features': [],
            'medium_risk_features': [],
            'temporal_leakage': [],
            'target_leakage': [],
            'recommendations': []
        }
        
        # Check for obvious leakage features
        for feature in data.columns:
            risk = self._assess_leakage_risk(feature, 'unknown')
            
            if risk == 'high':
                leakage_results['high_risk_features'].append({
                    'feature': feature,
                    'reason': 'Post-race outcome information',
                    'action': 'REMOVE from prediction pipeline'
                })
            elif risk == 'medium':
                leakage_results['medium_risk_features'].append({
                    'feature': feature,
                    'reason': 'Potential temporal or indirect leakage',
                    'action': 'REVIEW carefully - may need transformation'
                })
        
        # Check for temporal leakage patterns
        if 'race_date' in data.columns and 'extraction_timestamp' in data.columns:
            data['race_date_parsed'] = pd.to_datetime(data['race_date'], errors='coerce')
            data['extraction_parsed'] = pd.to_datetime(data['extraction_timestamp'], errors='coerce')
            
            # Check if extraction happened after race
            future_extractions = data[data['extraction_parsed'] > data['race_date_parsed']]
            if len(future_extractions) > 0:
                leakage_results['temporal_leakage'].append({
                    'issue': 'Data extracted after race completion',
                    'count': len(future_extractions),
                    'percentage': len(future_extractions) / len(data) * 100,
                    'action': 'Verify data collection timing'
                })
        
        # Check for target leakage (perfect correlation with outcome)
        if 'finish_position' in data.columns:
            target = pd.to_numeric(data['finish_position'], errors='coerce')
            
            for feature in data.columns:
                if feature not in ['finish_position', 'race_id', 'dog_name', 'dog_clean_name']:
                    try:
                        if pd.api.types.is_numeric_dtype(data[feature]):
                            correlation = data[feature].corr(target)
                            if abs(correlation) > 0.9:  # Very high correlation
                                leakage_results['target_leakage'].append({
                                    'feature': feature,
                                    'correlation': float(correlation),
                                    'action': 'INVESTIGATE - potential perfect predictor'
                                })
                    except:
                        continue
        
        # Generate recommendations
        recommendations = self._generate_leakage_recommendations(leakage_results)
        leakage_results['recommendations'] = recommendations
        
        logger.info(f"✅ Leakage scan complete:")
        logger.info(f"   High-risk features: {len(leakage_results['high_risk_features'])}")
        logger.info(f"   Medium-risk features: {len(leakage_results['medium_risk_features'])}")
        logger.info(f"   Temporal issues: {len(leakage_results['temporal_leakage'])}")
        logger.info(f"   Target leakage: {len(leakage_results['target_leakage'])}")
        
        return leakage_results
    
    def _generate_leakage_recommendations(self, leakage_results: Dict) -> List[str]:
        """Generate actionable recommendations for addressing leakage."""
        recommendations = []
        
        if leakage_results['high_risk_features']:
            recommendations.append(
                f"CRITICAL: Remove {len(leakage_results['high_risk_features'])} high-risk features "
                "from prediction pipeline immediately"
            )
        
        if leakage_results['medium_risk_features']:
            recommendations.append(
                f"REVIEW: Carefully examine {len(leakage_results['medium_risk_features'])} medium-risk features "
                "for potential indirect leakage"
            )
        
        if leakage_results['temporal_leakage']:
            recommendations.append(
                "TEMPORAL: Implement strict data collection cutoffs to prevent future information leakage"
            )
        
        if leakage_results['target_leakage']:
            recommendations.append(
                "TARGET: Investigate highly correlated features for potential perfect predictors"
            )
        
        recommendations.extend([
            "Implement feature validation pipeline with leakage checks",
            "Add temporal validation to ensure prediction-time data availability",
            "Create separate train/validation splits with temporal boundaries",
            "Document feature lineage and collection timing"
        ])
        
        return recommendations
    
    def recommend_feature_transformations(self, data: pd.DataFrame, mutual_info_scores: Dict) -> Dict[str, List[str]]:
        """Recommend beneficial feature transformations and aggregations."""
        logger.info("💡 Generating feature transformation recommendations...")
        
        recommendations = {
            'normalization': [],
            'log_transform': [],
            'binning': [],
            'aggregation': [],
            'interaction': [],
            'dimensionality_reduction': [],
            'temporal_features': []
        }
        
        for feature in data.columns:
            if feature in mutual_info_scores and pd.api.types.is_numeric_dtype(data[feature]):
                feature_data = data[feature].dropna()
                
                if len(feature_data) == 0:
                    continue
                
                # Check for normalization needs
                if feature_data.std() > 100 * feature_data.mean():
                    recommendations['normalization'].append(
                        f"{feature}: High variance relative to mean (std={feature_data.std():.2f}, mean={feature_data.mean():.2f})"
                    )
                
                # Check for log transformation needs (right-skewed data)
                if stats.skew(feature_data) > 2 and feature_data.min() > 0:
                    recommendations['log_transform'].append(
                        f"{feature}: Highly right-skewed (skew={stats.skew(feature_data):.2f})"
                    )
                
                # Check for binning needs (high cardinality continuous)
                if len(feature_data.unique()) > 1000:
                    recommendations['binning'].append(
                        f"{feature}: Very high cardinality ({len(feature_data.unique())} unique values)"
                    )
        
        # Aggregation recommendations
        if 'venue' in data.columns and 'trainer_name' in data.columns:
            recommendations['aggregation'].append(
                "Create trainer performance by venue aggregations"
            )
        
        if 'dog_clean_name' in data.columns and 'distance' in data.columns:
            recommendations['aggregation'].append(
                "Create dog performance by distance aggregations"
            )
        
        # Interaction feature recommendations
        high_utility_features = [k for k, v in list(mutual_info_scores.items())[:10]]
        if len(high_utility_features) >= 2:
            recommendations['interaction'].append(
                f"Create interaction features between top predictors: {', '.join(high_utility_features[:5])}"
            )
        
        # Dimensionality reduction for categorical features
        categorical_features = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        high_cardinality_cats = []
        
        for feature in categorical_features:
            if feature in data.columns:
                unique_count = data[feature].nunique()
                if unique_count > 50:
                    high_cardinality_cats.append(f"{feature} ({unique_count} categories)")
        
        if high_cardinality_cats:
            recommendations['dimensionality_reduction'].append(
                f"Consider PCA or embedding for high-cardinality categoricals: {', '.join(high_cardinality_cats)}"
            )
        
        # Temporal feature recommendations
        if 'race_date' in data.columns:
            recommendations['temporal_features'].extend([
                "Extract day of week from race_date",
                "Create seasonal features (month, quarter)",
                "Add days since last race for each dog",
                "Create rolling averages for dog/trainer performance"
            ])
        
        logger.info(f"✅ Generated {sum(len(v) for v in recommendations.values())} transformation recommendations")
        return recommendations
    
    def generate_comprehensive_report(self, output_dir: str = "feature_assessment_report"):
        """Generate comprehensive assessment report."""
        logger.info("📋 Generating comprehensive feature assessment report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load data
        data = self.load_comprehensive_data()
        if data.empty:
            logger.error("No data available for assessment")
            return
        
        # Perform all assessments
        feature_catalog = self.catalogue_all_features(data)
        mutual_info_scores = self.calculate_predictive_utility(data)
        leakage_results = self.perform_leakage_scan(data)
        transformation_recommendations = self.recommend_feature_transformations(data, mutual_info_scores)
        
        # Generate summary statistics
        summary_stats = {
            'total_features_analyzed': len(feature_catalog),
            'high_risk_leakage_count': len(leakage_results['high_risk_features']),
            'medium_risk_leakage_count': len(leakage_results['medium_risk_features']),
            'features_with_high_missing_rate': len([f for f, info in feature_catalog.items() 
                                                  if isinstance(info, dict) and info.get('missing_rate', 0) > 0.3]),
            'top_predictive_features': list(mutual_info_scores.keys())[:10],
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        self._save_assessment_results(output_path, {
            'summary_stats': summary_stats,
            'feature_catalog': feature_catalog,
            'mutual_info_scores': mutual_info_scores,
            'leakage_results': leakage_results,
            'transformation_recommendations': transformation_recommendations
        })
        
        # Create visualizations
        self._create_assessment_visualizations(output_path, data, feature_catalog, mutual_info_scores)
        
        # Generate executive summary
        self._generate_executive_summary(output_path, summary_stats, leakage_results, transformation_recommendations)
        
        logger.info(f"✅ Comprehensive assessment report generated in {output_path}")
        return output_path
    
    def _save_assessment_results(self, output_path: Path, results: Dict):
        """Save assessment results to JSON files."""
        
        # Save main results
        with open(output_path / "feature_assessment_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save feature catalog separately for easy reference
        with open(output_path / "feature_catalog.json", "w") as f:
            json.dump(results['feature_catalog'], f, indent=2, default=str)
        
        # Save leakage scan results
        with open(output_path / "leakage_scan_results.json", "w") as f:
            json.dump(results['leakage_results'], f, indent=2, default=str)
    
    def _create_assessment_visualizations(self, output_path: Path, data: pd.DataFrame, 
                                        feature_catalog: Dict, mutual_info_scores: Dict):
        """Create visualization plots for assessment results."""
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Missing data heatmap
            plt.figure(figsize=(12, 8))
            missing_data = data.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                plt.bar(range(len(missing_data)), missing_data.values)
                plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
                plt.title('Missing Data Count by Feature')
                plt.ylabel('Missing Values Count')
                plt.tight_layout()
                plt.savefig(output_path / "missing_data_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Predictive utility plot
            plt.figure(figsize=(14, 8))
            top_features = dict(list(mutual_info_scores.items())[:20])
            
            plt.bar(range(len(top_features)), list(top_features.values()))
            plt.xticks(range(len(top_features)), list(top_features.keys()), rotation=45, ha='right')
            plt.title('Top 20 Features by Predictive Utility (Mutual Information)')
            plt.ylabel('Mutual Information Score')
            plt.tight_layout()
            plt.savefig(output_path / "predictive_utility_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Feature cardinality distribution
            cardinality_counts = {'low': 0, 'medium': 0, 'high': 0}
            for feature_info in feature_catalog.values():
                if isinstance(feature_info, dict) and 'cardinality' in feature_info:
                    cardinality = feature_info['cardinality']
                    if cardinality in cardinality_counts:
                        cardinality_counts[cardinality] += 1
            
            plt.figure(figsize=(8, 6))
            plt.pie(cardinality_counts.values(), labels=cardinality_counts.keys(), autopct='%1.1f%%')
            plt.title('Feature Cardinality Distribution')
            plt.savefig(output_path / "cardinality_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Assessment visualizations created")
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
    
    def _generate_executive_summary(self, output_path: Path, summary_stats: Dict, 
                                  leakage_results: Dict, transformation_recommendations: Dict):
        """Generate executive summary report."""
        
        summary_content = f"""
# Feature Quality & Leakage Assessment Report
## Greyhound Analysis Predictor - Step 7 Assessment

**Assessment Date:** {summary_stats['assessment_timestamp']}
**Total Features Analyzed:** {summary_stats['total_features_analyzed']}

## 🚨 CRITICAL FINDINGS

### Data Leakage Assessment
- **High-Risk Features:** {summary_stats['high_risk_leakage_count']} features identified
- **Medium-Risk Features:** {summary_stats['medium_risk_leakage_count']} features require review
- **Immediate Action Required:** {summary_stats['high_risk_leakage_count'] > 0}

### Data Quality Issues
- **High Missing Rate Features:** {summary_stats['features_with_high_missing_rate']} features (>30% missing)

## 📊 TOP PREDICTIVE FEATURES

The following features show highest predictive utility (mutual information with target):

"""
        
        for i, feature in enumerate(summary_stats['top_predictive_features'][:10], 1):
            summary_content += f"{i}. {feature}\n"
        
        summary_content += f"""

## 🔧 KEY RECOMMENDATIONS

### Immediate Actions (Critical)
"""
        
        for recommendation in leakage_results['recommendations'][:3]:
            summary_content += f"- {recommendation}\n"
        
        summary_content += f"""

### Feature Engineering Opportunities
"""
        
        # Add top transformation recommendations
        for category, recommendations in transformation_recommendations.items():
            if recommendations and category in ['normalization', 'log_transform', 'interaction']:
                summary_content += f"\n**{category.title()}:**\n"
                for rec in recommendations[:3]:
                    summary_content += f"- {rec}\n"
        
        summary_content += f"""

## 📈 IMPLEMENTATION PRIORITY

1. **CRITICAL:** Remove high-risk leakage features immediately
2. **HIGH:** Implement feature validation pipeline
3. **MEDIUM:** Apply recommended transformations for top predictive features
4. **LOW:** Optimize feature engineering for remaining features

## 📁 DETAILED REPORTS

- `feature_catalog.json` - Complete feature analysis
- `leakage_scan_results.json` - Detailed leakage assessment
- `feature_assessment_results.json` - Full technical results
- `*.png` - Visualization plots

---
*Generated by Feature Quality & Leakage Assessment Tool*
"""
        
        # Save executive summary
        with open(output_path / "EXECUTIVE_SUMMARY.md", "w") as f:
            f.write(summary_content)
        
        # Also create a simple text version for easy viewing
        with open(output_path / "executive_summary.txt", "w") as f:
            f.write(summary_content)


def main():
    """Main execution function."""
    logger.info("🚀 Starting Feature Quality & Leakage Assessment...")
    
    # Initialize assessment
    assessor = FeatureQualityAssessment()
    
    # Generate comprehensive report
    report_path = assessor.generate_comprehensive_report()
    
    # Print summary to console
    logger.info("📋 ASSESSMENT COMPLETE!")
    logger.info(f"📁 Full report available at: {report_path}")
    logger.info("🔍 Key files to review:")
    logger.info("   • EXECUTIVE_SUMMARY.md - Key findings and recommendations")
    logger.info("   • leakage_scan_results.json - Critical leakage issues")
    logger.info("   • feature_catalog.json - Complete feature analysis")
    
    print("\n" + "="*80)
    print("FEATURE QUALITY & LEAKAGE ASSESSMENT COMPLETE")
    print("="*80)
    print(f"📊 Report generated: {report_path}")
    print("🚨 NEXT STEPS:")
    print("   1. Review EXECUTIVE_SUMMARY.md for critical findings")
    print("   2. Address high-risk leakage features immediately")
    print("   3. Implement recommended feature transformations")
    print("   4. Validate ML pipeline with leakage-free features")
    print("="*80)


if __name__ == "__main__":
    main()
