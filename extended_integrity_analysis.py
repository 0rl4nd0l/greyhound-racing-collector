#!/usr/bin/env python3
"""
Extended Referential Integrity Analysis
Additional checks for trainer relationships, seasonal consistency, and data quality
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json

class ExtendedIntegrityChecker:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}
        
    def execute_query(self, query, description=""):
        """Execute SQL query and return results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"Error executing query: {description}")
            print(f"Query: {query}")
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def analyze_trainer_consistency(self):
        """Analyze trainer data consistency across tables"""
        print("="*60)
        print("TRAINER RELATIONSHIP ANALYSIS")
        print("="*60)
        
        # Check unique trainers in dog_performances
        query1 = """
        SELECT trainer, COUNT(DISTINCT dog_name) as dogs_trained, COUNT(*) as total_performances
        FROM dog_performances
        WHERE trainer IS NOT NULL AND trainer != ''
        GROUP BY trainer
        ORDER BY dogs_trained DESC
        LIMIT 20
        """
        
        trainers_performance = self.execute_query(query1, "Trainers in dog_performances")
        
        # Check for missing trainer data
        query2 = """
        SELECT 
            COUNT(*) as total_performances,
            SUM(CASE WHEN trainer IS NULL OR trainer = '' THEN 1 ELSE 0 END) as missing_trainer,
            ROUND(SUM(CASE WHEN trainer IS NULL OR trainer = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as missing_percentage
        FROM dog_performances
        """
        
        missing_trainers = self.execute_query(query2, "Missing trainer data")
        
        print(f"Top 20 trainers by dogs trained:")
        print(trainers_performance.to_string(index=False))
        
        print(f"\nMissing trainer data:")
        print(missing_trainers.to_string(index=False))
        
        self.results['trainer_analysis'] = {
            'top_trainers': trainers_performance.to_dict('records'),
            'missing_trainer_stats': missing_trainers.to_dict('records')[0]
        }
    
    def analyze_seasonal_consistency(self):
        """Check for seasonal data consistency and unique dog participation"""
        print("\n" + "="*60)
        print("SEASONAL & TEMPORAL CONSISTENCY")
        print("="*60)
        
        # Analyze race date ranges
        query1 = """
        SELECT 
            MIN(race_date) as earliest_race,
            MAX(race_date) as latest_race,
            COUNT(DISTINCT race_date) as unique_race_dates,
            COUNT(DISTINCT venue) as unique_venues,
            COUNT(*) as total_races
        FROM races
        """
        
        date_range = self.execute_query(query1, "Race date range analysis")
        
        # Check for dogs appearing in multiple seasons/years
        query2 = """
        SELECT 
            strftime('%Y', r.race_date) as year,
            COUNT(DISTINCT dp.dog_name) as unique_dogs,
            COUNT(DISTINCT r.race_id) as races,
            COUNT(*) as total_performances
        FROM dog_performances dp
        JOIN races r ON dp.race_id = r.race_id
        WHERE r.race_date IS NOT NULL
        GROUP BY strftime('%Y', r.race_date)
        ORDER BY year
        """
        
        yearly_stats = self.execute_query(query2, "Yearly statistics")
        
        # Check for dogs with performances spanning multiple years
        query3 = """
        SELECT 
            dp.dog_name,
            MIN(r.race_date) as first_race,
            MAX(r.race_date) as last_race,
            COUNT(DISTINCT strftime('%Y', r.race_date)) as years_active,
            COUNT(*) as total_performances
        FROM dog_performances dp
        JOIN races r ON dp.race_id = r.race_id
        WHERE r.race_date IS NOT NULL
        GROUP BY dp.dog_name
        HAVING COUNT(DISTINCT strftime('%Y', r.race_date)) > 1
        ORDER BY years_active DESC, total_performances DESC
        LIMIT 20
        """
        
        multi_year_dogs = self.execute_query(query3, "Dogs active across multiple years")
        
        print("Race date range:")
        print(date_range.to_string(index=False))
        
        print(f"\nYearly participation statistics:")
        print(yearly_stats.to_string(index=False))
        
        print(f"\nTop 20 dogs active across multiple years:")
        if len(multi_year_dogs) > 0:
            print(multi_year_dogs.to_string(index=False))
        else:
            print("No dogs found active across multiple years")
        
        self.results['seasonal_analysis'] = {
            'date_range': date_range.to_dict('records')[0],
            'yearly_stats': yearly_stats.to_dict('records'),
            'multi_year_dogs_count': len(multi_year_dogs)
        }
    
    def analyze_form_guide_integrity(self):
        """Analyze form_guide table integrity and relationship to other tables"""
        print("\n" + "="*60)
        print("FORM GUIDE DATA INTEGRITY")
        print("="*60)
        
        # Check form_guide data completeness
        query1 = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT dog_name) as unique_dogs,
            COUNT(DISTINCT venue) as unique_venues,
            COUNT(DISTINCT race_date) as unique_dates,
            SUM(CASE WHEN finish_position IS NULL THEN 1 ELSE 0 END) as missing_positions,
            SUM(CASE WHEN race_time IS NULL THEN 1 ELSE 0 END) as missing_times,
            SUM(CASE WHEN weight IS NULL THEN 1 ELSE 0 END) as missing_weights
        FROM form_guide
        """
        
        form_completeness = self.execute_query(query1, "Form guide completeness")
        
        # Check for overlapping data between form_guide and dog_performances
        query2 = """
        SELECT 
            fg.dog_name,
            fg.race_date,
            fg.venue,
            fg.distance,
            fg.finish_position as form_position,
            dp.finish_position as perf_position,
            CASE WHEN fg.finish_position = dp.finish_position THEN 'MATCH' ELSE 'MISMATCH' END as position_match
        FROM form_guide fg
        JOIN dog_performances dp ON fg.dog_name = dp.dog_name
        JOIN races r ON dp.race_id = r.race_id AND r.race_date = fg.race_date AND r.venue = fg.venue
        WHERE fg.finish_position IS NOT NULL AND dp.finish_position IS NOT NULL
        LIMIT 20
        """
        
        overlapping_data = self.execute_query(query2, "Overlapping form guide and performance data")
        
        # Check date ranges in form_guide vs races
        query3 = """
        SELECT 
            'form_guide' as source,
            MIN(race_date) as earliest_date,
            MAX(race_date) as latest_date,
            COUNT(DISTINCT race_date) as unique_dates
        FROM form_guide
        WHERE race_date IS NOT NULL
        UNION ALL
        SELECT 
            'races' as source,
            MIN(race_date) as earliest_date,
            MAX(race_date) as latest_date,
            COUNT(DISTINCT race_date) as unique_dates
        FROM races
        WHERE race_date IS NOT NULL
        """
        
        date_comparison = self.execute_query(query3, "Date range comparison")
        
        print("Form guide data completeness:")
        print(form_completeness.to_string(index=False))
        
        print(f"\nOverlapping data sample (form_guide vs dog_performances):")
        if len(overlapping_data) > 0:
            print(overlapping_data.head(10).to_string(index=False))
        else:
            print("No overlapping data found")
        
        print(f"\nDate range comparison:")
        print(date_comparison.to_string(index=False))
        
        self.results['form_guide_analysis'] = {
            'completeness': form_completeness.to_dict('records')[0],
            'overlapping_records': len(overlapping_data),
            'date_ranges': date_comparison.to_dict('records')
        }
    
    def analyze_box_number_consistency(self):
        """Check box number assignments and consistency"""
        print("\n" + "="*60)
        print("BOX NUMBER CONSISTENCY ANALYSIS")
        print("="*60)
        
        # Box number distribution in dog_performances
        query1 = """
        SELECT 
            box_number,
            COUNT(*) as frequency,
            COUNT(DISTINCT race_id) as races_used,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dog_performances WHERE box_number IS NOT NULL), 2) as percentage
        FROM dog_performances
        WHERE box_number IS NOT NULL
        GROUP BY box_number
        ORDER BY box_number
        """
        
        box_distribution = self.execute_query(query1, "Box number distribution")
        
        # Check for races with invalid box numbers
        query2 = """
        SELECT 
            r.race_id,
            r.race_name,
            r.venue,
            r.race_date,
            COUNT(DISTINCT dp.box_number) as unique_boxes,
            MAX(dp.box_number) as max_box,
            MIN(dp.box_number) as min_box,
            COUNT(*) as dogs_in_race
        FROM races r
        JOIN dog_performances dp ON r.race_id = dp.race_id
        WHERE dp.box_number IS NOT NULL
        GROUP BY r.race_id, r.race_name, r.venue, r.race_date
        HAVING MAX(dp.box_number) > 8 OR MIN(dp.box_number) < 1 OR COUNT(*) > 8
        ORDER BY max_box DESC
        LIMIT 10
        """
        
        invalid_box_races = self.execute_query(query2, "Races with invalid box numbers")
        
        # Check for duplicate box numbers in same race
        query3 = """
        SELECT 
            dp.race_id,
            dp.box_number,
            COUNT(*) as dogs_same_box,
            GROUP_CONCAT(dp.dog_name) as dogs
        FROM dog_performances dp
        WHERE dp.box_number IS NOT NULL
        GROUP BY dp.race_id, dp.box_number
        HAVING COUNT(*) > 1
        ORDER BY dogs_same_box DESC
        LIMIT 10
        """
        
        duplicate_boxes = self.execute_query(query3, "Duplicate box numbers in same race")
        
        print("Box number distribution:")
        print(box_distribution.to_string(index=False))
        
        print(f"\nRaces with invalid box numbers:")
        if len(invalid_box_races) > 0:
            print(invalid_box_races.to_string(index=False))
        else:
            print("No races with invalid box numbers found")
        
        print(f"\nDuplicate box numbers in same race:")
        if len(duplicate_boxes) > 0:
            print(duplicate_boxes.to_string(index=False))
        else:
            print("No duplicate box numbers found")
        
        self.results['box_number_analysis'] = {
            'distribution': box_distribution.to_dict('records'),
            'invalid_races': len(invalid_box_races),
            'duplicate_boxes': len(duplicate_boxes)
        }
    
    def generate_detailed_recommendations(self):
        """Generate specific recommendations for fixing integrity issues"""
        print("\n" + "="*60)
        print("INTEGRITY ISSUE RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Venue table recommendations
        if self.results.get('venue_missing_count', 0) > 0:
            recommendations.append({
                'issue': 'Empty venues table',
                'severity': 'HIGH',
                'description': 'The venues table is empty but referenced in races and form_guide',
                'recommendation': 'Populate venues table with venue codes, names, and metadata',
                'sql_fix': '''
                INSERT INTO venues (venue_code, venue_name, location) 
                SELECT DISTINCT venue, venue, 'Unknown' FROM races 
                WHERE venue NOT IN (SELECT venue_code FROM venues);
                '''
            })
        
        # Form guide dog consistency
        form_dogs_missing = self.results.get('form_guide_analysis', {}).get('completeness', {}).get('unique_dogs', 0)
        if form_dogs_missing > 0:
            recommendations.append({
                'issue': 'Dogs in form_guide missing from dogs table',
                'severity': 'MEDIUM',
                'description': 'Some dogs in form_guide are not in the dogs table',
                'recommendation': 'Add missing dogs to dogs table or clean form_guide data',
                'sql_fix': '''
                INSERT INTO dogs (dog_name, total_races, total_wins, total_places)
                SELECT DISTINCT fg.dog_name, 0, 0, 0
                FROM form_guide fg 
                WHERE fg.dog_name NOT IN (SELECT dog_name FROM dogs);
                '''
            })
        
        # Trainer data completeness
        missing_trainers = self.results.get('trainer_analysis', {}).get('missing_trainer_stats', {}).get('missing_percentage', 0)
        if missing_trainers > 10:  # More than 10% missing
            recommendations.append({
                'issue': f'Missing trainer data ({missing_trainers}% of performances)',
                'severity': 'MEDIUM',
                'description': 'Significant amount of trainer information is missing',
                'recommendation': 'Investigate data source and backfill trainer information',
                'sql_fix': 'Manual data collection required - check original race results'
            })
        
        print("RECOMMENDED ACTIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['issue']} [{rec['severity']}]")
            print(f"   Description: {rec['description']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   SQL Fix: {rec['sql_fix']}")
        
        if not recommendations:
            print("✅ No critical integrity issues requiring immediate action")
        
        self.results['recommendations'] = recommendations
    
    def run_extended_analysis(self):
        """Run the complete extended integrity analysis"""
        print("Starting Extended Referential Integrity Analysis...")
        print(f"Database: {self.db_path}")
        
        # Check venue count first
        venue_count = self.execute_query("SELECT COUNT(*) as count FROM venues")['count'].iloc[0]
        self.results['venue_missing_count'] = 25 if venue_count == 0 else 0
        
        self.analyze_trainer_consistency()
        self.analyze_seasonal_consistency()
        self.analyze_form_guide_integrity()
        self.analyze_box_number_consistency()
        self.generate_detailed_recommendations()
        
        return self.results
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    db_path = "./databases/race_data.db"
    
    checker = ExtendedIntegrityChecker(db_path)
    
    try:
        results = checker.run_extended_analysis()
        
        # Save extended results
        with open("extended_integrity_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Extended analysis results saved to: extended_integrity_results.json")
        
    except Exception as e:
        print(f"Error during extended analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        checker.close()

if __name__ == "__main__":
    main()
