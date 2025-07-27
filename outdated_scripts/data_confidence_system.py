#!/usr/bin/env python3
"""
Data Confidence Scoring System for Greyhound Racing Analysis
============================================================

This system assigns confidence scores to races based on data completeness,
ensuring that analysis results are weighted appropriately and incomplete
data doesn't skew the overall insights.

Confidence Scoring Methodology:
- Critical Data Fields: 60% of score
- Performance Data Fields: 25% of score  
- Supplementary Data Fields: 15% of score

Confidence Levels:
- A+ (90-100%): Complete data, highest confidence
- A  (80-89%):  Near complete, high confidence
- B+ (70-79%):  Good data, moderate-high confidence
- B  (60-69%):  Acceptable data, moderate confidence
- C+ (50-59%):  Limited data, low-moderate confidence
- C  (40-49%):  Poor data, low confidence
- D+ (30-39%):  Very poor data, very low confidence
- D  (20-29%):  Minimal data, minimal confidence
- F  (0-19%):   Insufficient data, exclude from analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataConfidenceScorer:
    """System for scoring data confidence based on completeness"""
    
    def __init__(self, db_path="comprehensive_greyhound_data.db"):
        self.db_path = db_path
        self.confidence_weights = {
            'critical': 0.60,      # Essential for core analysis
            'performance': 0.25,   # Important for advanced analysis
            'supplementary': 0.15  # Nice to have for enhanced insights
        }
        
        # Define field importance categories
        self.field_categories = {
            'critical': [
                'finish_position',     # Race result
                'trainer_name',        # Trainer info
                'box_number',          # Starting position
                'dog_clean_name',      # Dog identification
                'venue',               # Track location
                'race_date',           # When race occurred
                'distance'             # Race distance
            ],
            'performance': [
                'individual_time',     # Race time
                'weight',              # Dog weight
                'sectional_1st',       # First sectional
                'sectional_2nd',       # Second sectional
                'odds_decimal',        # Betting odds
                'margin',              # Winning margin
                'beaten_margin',       # Beaten distance
                'running_style'        # Running style
            ],
            'supplementary': [
                'weather',             # Weather conditions
                'track_condition',     # Track surface
                'grade',               # Race grade
                'prize_money_total',   # Prize money
                'sectional_3rd',       # Third sectional
                'form_guide_json',     # Form guide data
                'race_time',           # Overall race time
                'field_size'           # Number of runners
            ]
        }
    
    def load_data(self):
        """Load all data for confidence scoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load main datasets
                self.dog_data = pd.read_sql_query('SELECT * FROM dog_race_data', conn)
                self.race_data = pd.read_sql_query('SELECT * FROM race_metadata', conn)
                self.odds_data = pd.read_sql_query('SELECT * FROM odds_snapshots', conn)
                
                # Merge race metadata with dog data
                self.merged_data = pd.merge(
                    self.dog_data,
                    self.race_data[['race_id', 'venue', 'race_date', 'weather', 'distance', 
                                   'track_condition', 'grade', 'prize_money_total', 'race_time', 'field_size']],
                    on='race_id',
                    how='left'
                )
                
                # Add odds data
                if not self.odds_data.empty:
                    odds_agg = self.odds_data.groupby(['race_id', 'dog_clean_name'])['odds_decimal'].mean().reset_index()
                    self.merged_data = pd.merge(
                        self.merged_data,
                        odds_agg,
                        on=['race_id', 'dog_clean_name'],
                        how='left'
                    )
                
                print(f"✅ Data loaded: {len(self.merged_data)} records")
                return True
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_field_completeness(self, data):
        """Calculate completeness score for each field category"""
        scores = {}
        
        for category, fields in self.field_categories.items():
            available_fields = [f for f in fields if f in data.columns]
            
            if not available_fields:
                scores[category] = 0.0
                continue
            
            # Calculate completeness for each field
            field_scores = []
            for field in available_fields:
                if field in data.columns:
                    completeness = data[field].notna().mean()
                    field_scores.append(completeness)
                else:
                    field_scores.append(0.0)
            
            # Average completeness across fields in category
            scores[category] = np.mean(field_scores)
        
        return scores
    
    def calculate_confidence_score(self, completeness_scores):
        """Calculate overall confidence score using weighted categories"""
        weighted_score = 0.0
        
        for category, score in completeness_scores.items():
            weight = self.confidence_weights[category]
            weighted_score += score * weight
        
        return weighted_score * 100  # Convert to percentage
    
    def assign_confidence_grade(self, score):
        """Assign letter grade based on confidence score"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C+'
        elif score >= 40:
            return 'C'
        elif score >= 30:
            return 'D+'
        elif score >= 20:
            return 'D'
        else:
            return 'F'
    
    def get_confidence_description(self, grade):
        """Get description for confidence grade"""
        descriptions = {
            'A+': 'Complete data, highest confidence',
            'A': 'Near complete, high confidence',
            'B+': 'Good data, moderate-high confidence',
            'B': 'Acceptable data, moderate confidence',
            'C+': 'Limited data, low-moderate confidence',
            'C': 'Poor data, low confidence',
            'D+': 'Very poor data, very low confidence',
            'D': 'Minimal data, minimal confidence',
            'F': 'Insufficient data, exclude from analysis'
        }
        return descriptions.get(grade, 'Unknown confidence level')
    
    def score_individual_records(self):
        """Score confidence for each individual record"""
        if not hasattr(self, 'merged_data'):
            if not self.load_data():
                return None
        
        confidence_scores = []
        
        for idx, row in self.merged_data.iterrows():
            # Create single-row dataframe for scoring
            single_row = pd.DataFrame([row])
            
            # Calculate completeness scores
            completeness = self.calculate_field_completeness(single_row)
            
            # Calculate overall confidence score
            confidence_score = self.calculate_confidence_score(completeness)
            
            # Assign grade
            grade = self.assign_confidence_grade(confidence_score)
            
            confidence_scores.append({
                'record_id': idx,
                'race_id': row['race_id'],
                'dog_name': row['dog_name'],
                'confidence_score': confidence_score,
                'confidence_grade': grade,
                'confidence_description': self.get_confidence_description(grade),
                'critical_completeness': completeness['critical'],
                'performance_completeness': completeness['performance'],
                'supplementary_completeness': completeness['supplementary'],
                'include_in_analysis': grade != 'F'
            })
        
        return pd.DataFrame(confidence_scores)
    
    def score_race_level(self):
        """Score confidence at race level (aggregated)"""
        if not hasattr(self, 'merged_data'):
            if not self.load_data():
                return None
        
        race_confidence = []
        
        for race_id, race_group in self.merged_data.groupby('race_id'):
            # Calculate completeness for this race
            completeness = self.calculate_field_completeness(race_group)
            
            # Calculate overall confidence score
            confidence_score = self.calculate_confidence_score(completeness)
            
            # Assign grade
            grade = self.assign_confidence_grade(confidence_score)
            
            race_confidence.append({
                'race_id': race_id,
                'num_dogs': len(race_group),
                'confidence_score': confidence_score,
                'confidence_grade': grade,
                'confidence_description': self.get_confidence_description(grade),
                'critical_completeness': completeness['critical'],
                'performance_completeness': completeness['performance'],
                'supplementary_completeness': completeness['supplementary'],
                'include_in_analysis': grade != 'F'
            })
        
        return pd.DataFrame(race_confidence)
    
    def create_weighted_dataset(self, min_grade='C'):
        """Create dataset with confidence weights for analysis"""
        if not hasattr(self, 'merged_data'):
            if not self.load_data():
                return None
        
        # Get individual record confidence scores
        confidence_df = self.score_individual_records()
        
        # Merge with original data
        weighted_data = pd.merge(
            self.merged_data,
            confidence_df[['record_id', 'confidence_score', 'confidence_grade', 'include_in_analysis']],
            left_index=True,
            right_on='record_id',
            how='left'
        )
        
        # Filter by minimum grade
        grade_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
        min_grade_index = grade_order.index(min_grade)
        
        filtered_data = weighted_data[
            weighted_data['confidence_grade'].isin(grade_order[:min_grade_index + 1])
        ]
        
        # Add confidence weight (0-1 scale)
        filtered_data['confidence_weight'] = filtered_data['confidence_score'] / 100
        
        return filtered_data
    
    def generate_confidence_report(self):
        """Generate comprehensive confidence analysis report"""
        print("🎯 GENERATING DATA CONFIDENCE REPORT")
        print("=" * 50)
        
        if not self.load_data():
            return None
        
        # Get confidence scores
        record_confidence = self.score_individual_records()
        race_confidence = self.score_race_level()
        
        # Overall statistics
        print(f"📊 OVERALL CONFIDENCE STATISTICS:")
        print(f"   Total Records: {len(record_confidence):,}")
        print(f"   Total Races: {len(race_confidence):,}")
        print(f"   Average Record Confidence: {record_confidence['confidence_score'].mean():.1f}%")
        print(f"   Average Race Confidence: {race_confidence['confidence_score'].mean():.1f}%")
        
        print()
        
        # Confidence grade distribution
        print(f"📈 CONFIDENCE GRADE DISTRIBUTION:")
        record_grades = record_confidence['confidence_grade'].value_counts()
        race_grades = race_confidence['confidence_grade'].value_counts()
        
        print("   Records by Grade:")
        for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']:
            if grade in record_grades.index:
                count = record_grades[grade]
                pct = (count / len(record_confidence)) * 100
                print(f"     {grade}: {count:,} ({pct:.1f}%)")
        
        print()
        print("   Races by Grade:")
        for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']:
            if grade in race_grades.index:
                count = race_grades[grade]
                pct = (count / len(race_confidence)) * 100
                print(f"     {grade}: {count:,} ({pct:.1f}%)")
        
        print()
        
        # Usable data analysis
        usable_records = record_confidence[record_confidence['include_in_analysis']]
        usable_races = race_confidence[race_confidence['include_in_analysis']]
        
        print(f"✅ USABLE DATA ANALYSIS:")
        print(f"   Usable Records: {len(usable_records):,}/{len(record_confidence):,} ({(len(usable_records)/len(record_confidence))*100:.1f}%)")
        print(f"   Usable Races: {len(usable_races):,}/{len(race_confidence):,} ({(len(usable_races)/len(race_confidence))*100:.1f}%)")
        
        # High confidence data
        high_confidence = record_confidence[record_confidence['confidence_score'] >= 70]
        print(f"   High Confidence (B+ or better): {len(high_confidence):,} ({(len(high_confidence)/len(record_confidence))*100:.1f}%)")
        
        print()
        
        # Field completeness breakdown
        print(f"📊 FIELD COMPLETENESS BREAKDOWN:")
        avg_critical = record_confidence['critical_completeness'].mean()
        avg_performance = record_confidence['performance_completeness'].mean()
        avg_supplementary = record_confidence['supplementary_completeness'].mean()
        
        print(f"   Critical Fields: {avg_critical:.1%} average completeness")
        print(f"   Performance Fields: {avg_performance:.1%} average completeness")
        print(f"   Supplementary Fields: {avg_supplementary:.1%} average completeness")
        
        print()
        
        # Recommendations
        print(f"💡 RECOMMENDATIONS:")
        if avg_critical < 0.8:
            print(f"   ⚠️  Critical field completeness is low ({avg_critical:.1%}) - focus on basic race data")
        if avg_performance < 0.5:
            print(f"   ⚠️  Performance field completeness is low ({avg_performance:.1%}) - enhance scraping for times/weights")
        if len(high_confidence) < len(record_confidence) * 0.3:
            print(f"   ⚠️  Low proportion of high-confidence data - consider data enrichment")
        
        print(f"   ✅ Use minimum grade 'C' for general analysis")
        print(f"   ✅ Use minimum grade 'B' for performance-critical analysis")
        print(f"   ✅ Use minimum grade 'A' for high-precision predictions")
        
        # Save detailed results
        report_data = {
            'summary': {
                'total_records': len(record_confidence),
                'total_races': len(race_confidence),
                'avg_record_confidence': float(record_confidence['confidence_score'].mean()),
                'avg_race_confidence': float(race_confidence['confidence_score'].mean()),
                'usable_records': len(usable_records),
                'usable_races': len(usable_races),
                'high_confidence_records': len(high_confidence)
            },
            'grade_distribution': {
                'records': record_grades.to_dict(),
                'races': race_grades.to_dict()
            },
            'field_completeness': {
                'critical': float(avg_critical),
                'performance': float(avg_performance),
                'supplementary': float(avg_supplementary)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('data_confidence_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save detailed confidence scores
        record_confidence.to_csv('record_confidence_scores.csv', index=False)
        race_confidence.to_csv('race_confidence_scores.csv', index=False)
        
        print(f"📄 Detailed reports saved:")
        print(f"   - data_confidence_report.json")
        print(f"   - record_confidence_scores.csv")
        print(f"   - race_confidence_scores.csv")
        
        return {
            'record_confidence': record_confidence,
            'race_confidence': race_confidence,
            'summary': report_data
        }

def main():
    """Main execution"""
    scorer = DataConfidenceScorer()
    
    try:
        # Generate comprehensive confidence report
        results = scorer.generate_confidence_report()
        
        if results:
            print(f"\n✅ Confidence scoring complete!")
            print(f"📊 System now ready for weighted analysis")
            
            # Show sample of confidence-weighted data
            print(f"\n📈 SAMPLE CONFIDENCE-WEIGHTED DATA:")
            weighted_data = scorer.create_weighted_dataset(min_grade='C')
            if weighted_data is not None:
                print(f"   Weighted dataset: {len(weighted_data):,} records")
                print(f"   Average confidence weight: {weighted_data['confidence_weight'].mean():.3f}")
        else:
            print(f"❌ Error generating confidence report")
            
    except Exception as e:
        print(f"❌ Error in confidence scoring: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
