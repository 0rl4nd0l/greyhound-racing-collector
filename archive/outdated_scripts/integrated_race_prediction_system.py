#!/usr/bin/env python3
"""
Integrated Race Prediction System
=================================

Complete integration of:
1. Sportsbet race time scraper for accurate start times
2. Sportsbet odds integration for real-time market data
3. Unified predictor for comprehensive race analysis
4. Organized race display with chronological ordering

Usage: python3 integrated_race_prediction_system.py
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Import our integrated components
from sportsbet_race_time_scraper import SportsbetRaceTimeScraper, organize_races_with_sportsbet_times
from unified_predictor import UnifiedPredictor
from upcoming_race_browser import UpcomingRaceBrowser

class IntegratedRacePredictionSystem:
    """Complete integrated system for race prediction with live data"""
    
    def __init__(self):
        print("🚀 Initializing Integrated Race Prediction System...")
        
        # Initialize components
        self.race_time_scraper = SportsbetRaceTimeScraper()
        self.unified_predictor = UnifiedPredictor()
        self.race_browser = UpcomingRaceBrowser()
        
        # Directories
        self.upcoming_races_dir = Path('./upcoming_races')
        self.predictions_dir = Path('./predictions')
        self.predictions_dir.mkdir(exist_ok=True)
        
        print("✅ System initialized successfully!")
    
    def run_complete_analysis(self):
        """Run complete integrated analysis"""
        print("\n🏁 Starting Complete Integrated Race Analysis")
        print("=" * 60)
        
        # Step 1: Get organized race times from Sportsbet
        print("\n📊 Step 1: Fetching Race Times from Sportsbet")
        print("-" * 40)
        organized_races = organize_races_with_sportsbet_times()
        
        if not organized_races:
            print("⚠️ No races found from Sportsbet, checking local files...")
            organized_races = self._get_local_races()
        
        if not organized_races:
            print("❌ No races available for analysis")
            return
        
        print(f"✅ Found {len(organized_races)} races to analyze")
        
        # Step 2: Run predictions for each race
        print("\n🎯 Step 2: Generating Predictions for Each Race")
        print("-" * 40)
        
        prediction_results = []
        for i, race in enumerate(organized_races[:5], 1):  # Limit to first 5 races
            print(f"\n🏇 Analyzing Race {i}/{min(5, len(organized_races))}: {race.get('venue')} Race {race.get('race_number')}")
            
            # Find corresponding CSV file
            race_file = self._find_race_file(race)
            if not race_file:
                print(f"   ⚠️ No CSV file found for this race, skipping...")
                continue
            
            print(f"   📁 Using file: {race_file.name}")
            
            # Run unified prediction
            try:
                result = self.unified_predictor.predict_race_file(str(race_file))
                if result and result.get('success'):
                    # Add race timing info to results
                    result['race_timing'] = {
                        'race_time': race.get('race_time'),
                        'source': race.get('source', 'unknown'),
                        'sort_time': race.get('sort_time'),
                        'url': race.get('url')
                    }
                    prediction_results.append(result)
                    print(f"   ✅ Prediction completed successfully")
                    
                    # Show top 3 picks
                    if result.get('predictions'):
                        print(f"   🏆 Top 3 picks:")
                        for j, pick in enumerate(result['predictions'][:3], 1):
                            dog_name = pick.get('dog_name', 'Unknown')
                            box = pick.get('box_number', '?')
                            # Try different score field names
                            score = pick.get('final_score', pick.get('prediction_score', 0))
                            confidence = pick.get('confidence_level', 'UNKNOWN')
                            print(f"      {j}. {dog_name} (Box {box}) - Score: {score:.3f} ({confidence})")
                else:
                    print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Error running prediction: {e}")
        
        # Step 3: Generate comprehensive summary
        print("\n📈 Step 3: Generating Comprehensive Summary")
        print("-" * 40)
        
        if prediction_results:
            self._generate_comprehensive_summary(prediction_results, organized_races)
            self._save_integrated_results(prediction_results, organized_races)
        else:
            print("❌ No predictions were generated")
        
        print("\n✅ Complete Integrated Analysis Finished!")
        return prediction_results
    
    def _get_local_races(self):
        """Fallback: get races from local CSV files"""
        if not self.upcoming_races_dir.exists():
            return []
        
        races = []
        for csv_file in self.upcoming_races_dir.glob('*.csv'):
            if csv_file.name.lower().startswith('readme'):
                continue
            
            # Extract race info from filename
            import re
            pattern = r'Race (\d+) - ([A-Z-]+) - (\d{4}-\d{2}-\d{2})\.csv'
            match = re.match(pattern, csv_file.name)
            
            if match:
                race_number = int(match.group(1))
                venue_code = match.group(2)
                race_date = match.group(3)
                
                # Estimate time based on race number
                base_hour = 13  # 1 PM
                total_minutes = (race_number - 1) * 25
                hour = base_hour + (total_minutes // 60)
                minute = total_minutes % 60
                
                if hour > 12:
                    race_time = f'{hour - 12}:{minute:02d} PM'
                elif hour == 12:
                    race_time = f'12:{minute:02d} PM'
                else:
                    race_time = f'{hour}:{minute:02d} AM'
                
                races.append({
                    'venue': venue_code.replace('-', ' ').replace('_', ' '),
                    'race_number': race_number,
                    'race_time': race_time,
                    'sort_time': hour * 60 + minute,
                    'source': 'file_estimate',
                    'filename': csv_file.name
                })
        
        # Sort by time
        races.sort(key=lambda x: x['sort_time'])
        return races
    
    def _find_race_file(self, race_info):
        """Find the CSV file corresponding to a race"""
        venue = race_info.get('venue', '').upper().replace(' ', '-').replace('_', '-')
        race_number = race_info.get('race_number', 1)
        
        # Try different filename patterns
        today = datetime.now().strftime('%Y-%m-%d')
        possible_names = [
            f"Race {race_number} - {venue} - {today}.csv",
            f"Race_{race_number}_-_{venue}_-_{today}.csv",
            race_info.get('filename', '')
        ]
        
        for name in possible_names:
            if name:
                file_path = self.upcoming_races_dir / name
                if file_path.exists():
                    return file_path
        
        # Try partial matching
        for csv_file in self.upcoming_races_dir.glob('*.csv'):
            if csv_file.name.lower().startswith('readme'):
                continue
            
            if (str(race_number) in csv_file.name and 
                any(v_part in csv_file.name.upper() for v_part in venue.split('-'))):
                return csv_file
        
        return None
    
    def _generate_comprehensive_summary(self, prediction_results, organized_races):
        """Generate comprehensive summary of all predictions"""
        print("\n🏆 COMPREHENSIVE RACE ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        total_races = len(prediction_results)
        total_predictions = sum(len(r.get('predictions', [])) for r in prediction_results)
        
        print(f"📊 Analysis Overview:")
        print(f"   • Total races analyzed: {total_races}")
        print(f"   • Total predictions generated: {total_predictions}")
        print(f"   • Average predictions per race: {total_predictions/total_races:.1f}")
        
        # Show each race summary
        print(f"\n🏇 Individual Race Summaries:")
        for i, result in enumerate(prediction_results, 1):
            race_info = result.get('race_info', {})
            timing = result.get('race_timing', {})
            predictions = result.get('predictions', [])
            
            venue = race_info.get('venue', 'Unknown')
            race_num = race_info.get('race_number', '?')
            race_time = timing.get('race_time', 'TBA')
            time_source = '🌐' if timing.get('source') == 'sportsbet' else '📅'
            
            print(f"\n   Race {i}: {venue} Race {race_num} at {race_time} {time_source}")
            
            if predictions:
                print(f"      🥇 Winner Pick: {predictions[0].get('dog_name')} (Box {predictions[0].get('box_number')})")
                print(f"         Score: {predictions[0].get('final_score', 0):.3f} | Confidence: {predictions[0].get('confidence_level', 'UNKNOWN')}")
                
                if len(predictions) > 1:
                    print(f"      🥈 Place Pick: {predictions[1].get('dog_name')} (Box {predictions[1].get('box_number')})")
                    print(f"         Score: {predictions[1].get('final_score', 0):.3f}")
                
                # Show data quality
                data_quality = result.get('data_quality_summary', {})
                avg_quality = data_quality.get('average_quality', 0)
                dogs_with_data = data_quality.get('dogs_with_good_data', 0)
                total_dogs = data_quality.get('total_dogs', 0)
                
                print(f"      📈 Data Quality: {avg_quality:.2f} ({dogs_with_data}/{total_dogs} dogs with good data)")
            else:
                print(f"      ❌ No predictions generated")
        
        # Next races schedule
        print(f"\n⏰ Upcoming Race Schedule:")
        for i, race in enumerate(organized_races[:10], 1):
            race_time = race.get('race_time', 'TBA')
            venue = race.get('venue', 'Unknown')
            race_num = race.get('race_number', '?')
            source_icon = '🌐' if race.get('source') == 'sportsbet' else '📅'
            
            status = "✅ ANALYZED" if i <= len(prediction_results) else "⏳ PENDING"
            print(f"   {i:2d}. {race_time:>8} {source_icon} - Race {race_num} at {venue} - {status}")
        
        print(f"\nLegend: 🌐 = Live Sportsbet time | 📅 = Estimated time")
    
    def _save_integrated_results(self, prediction_results, organized_races):
        """Save integrated results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.predictions_dir / f"integrated_analysis_{timestamp}.json"
        
        integrated_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'system_version': 'Integrated Race Prediction System v1.0',
            'total_races_analyzed': len(prediction_results),
            'total_races_available': len(organized_races),
            'race_schedule': organized_races,
            'predictions': prediction_results,
            'summary': {
                'best_bets': [],
                'data_quality_overall': 0.0,
                'confidence_distribution': {}
            }
        }
        
        # Calculate summary statistics
        if prediction_results:
            # Best bets (highest confidence winners)
            for result in prediction_results:
                if result.get('predictions'):
                    top_pick = result['predictions'][0]
                    race_info = result.get('race_info', {})
                    timing = result.get('race_timing', {})
                    
                    integrated_data['summary']['best_bets'].append({
                        'venue': race_info.get('venue'),
                        'race_number': race_info.get('race_number'),
                        'race_time': timing.get('race_time'),
                        'dog_name': top_pick.get('dog_name'),
                        'box_number': top_pick.get('box_number'),
                        'score': top_pick.get('final_score'),
                        'confidence': top_pick.get('confidence_level')
                    })
            
            # Overall data quality
            all_qualities = []
            confidence_counts = {}
            
            for result in prediction_results:
                data_quality = result.get('data_quality_summary', {})
                if data_quality.get('average_quality'):
                    all_qualities.append(data_quality['average_quality'])
                
                for pred in result.get('predictions', []):
                    conf = pred.get('confidence_level', 'UNKNOWN')
                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            if all_qualities:
                integrated_data['summary']['data_quality_overall'] = sum(all_qualities) / len(all_qualities)
            
            integrated_data['summary']['confidence_distribution'] = confidence_counts
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(integrated_data, f, indent=2, default=str)
        
        print(f"\n💾 Integrated results saved to: {results_file}")
        
        # Also create a simplified summary file
        summary_file = self.predictions_dir / f"today_summary_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(summary_file, 'w') as f:
            f.write("TODAY'S RACING ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BEST BETS:\n")
            f.write("-" * 20 + "\n")
            for i, bet in enumerate(integrated_data['summary']['best_bets'][:5], 1):
                f.write(f"{i}. {bet['venue']} Race {bet['race_number']} at {bet['race_time']}\n")
                f.write(f"   {bet['dog_name']} (Box {bet['box_number']}) - Score: {bet['score']:.3f}\n")
                f.write(f"   Confidence: {bet['confidence']}\n\n")
            
            f.write(f"TOTAL RACES ANALYZED: {len(prediction_results)}\n")
            f.write(f"OVERALL DATA QUALITY: {integrated_data['summary']['data_quality_overall']:.2f}\n")
        
        print(f"📄 Summary saved to: {summary_file}")
    
    def update_race_times_only(self):
        """Update race times from Sportsbet without running full analysis"""
        print("🕐 Updating race times from Sportsbet...")
        organized_races = organize_races_with_sportsbet_times()
        
        if organized_races:
            print(f"✅ Updated times for {len(organized_races)} races")
            return organized_races
        else:
            print("❌ No race times updated")
            return []
    
    def predict_single_race(self, race_file_path):
        """Predict a single race file"""
        print(f"🎯 Predicting single race: {race_file_path}")
        
        if not os.path.exists(race_file_path):
            print(f"❌ Race file not found: {race_file_path}")
            return None
        
        try:
            # Use 'weather' enhancement level to avoid GPT issues with scoring display
            result = self.unified_predictor.predict_race_file(race_file_path, enhancement_level='weather')
            if result and result.get('success'):
                print("✅ Prediction completed successfully")
                
                # Show results
                predictions = result.get('predictions', [])
                if predictions:
                    print(f"\n🏆 TOP PICKS:")
                    for i, pick in enumerate(predictions[:5], 1):
                        dog_name = pick.get('dog_name', 'Unknown')
                        box = pick.get('box_number', '?')
                        # Try different score field names
                        score = pick.get('final_score', pick.get('prediction_score', 0))
                        confidence = pick.get('confidence_level', 'UNKNOWN')
                        print(f"   {i}. {dog_name} (Box {box}) - Score: {score:.3f} ({confidence})")
                
                return result
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def main():
    """Main entry point"""
    print("🏁 Integrated Race Prediction System")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Single race mode
        race_file = sys.argv[1]
        system = IntegratedRacePredictionSystem()
        system.predict_single_race(race_file)
    else:
        # Full analysis mode
        system = IntegratedRacePredictionSystem()
        system.run_complete_analysis()

if __name__ == "__main__":
    main()
