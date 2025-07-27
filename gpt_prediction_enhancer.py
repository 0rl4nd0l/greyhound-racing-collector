#!/usr/bin/env python3
"""
GPT Prediction Enhancer
========================

Integrates OpenAI GPT analysis with the existing greyhound racing prediction system.
This module acts as a bridge between the ML predictions and GPT insights.

Features:
- Enhances existing predictions with GPT insights
- Provides narrative analysis of race dynamics  
- Generates betting strategies with risk assessment
- Analyzes form patterns and market inefficiencies
- Creates comprehensive reports combining ML and AI analysis
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from openai_enhanced_analyzer import OpenAIEnhancedAnalyzer, RaceContext, DogProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTPredictionEnhancer:
    """Main class for enhancing predictions with GPT analysis"""
    
    def __init__(self, database_path: str = "greyhound_racing_data.db", 
                 predictions_dir: str = "./predictions",
                 enhanced_output_dir: str = "./gpt_enhanced_predictions"):
        
        self.database_path = database_path
        self.predictions_dir = Path(predictions_dir)
        self.enhanced_output_dir = Path(enhanced_output_dir)
        
        # Create output directory
        self.enhanced_output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI analyzer
        try:
            self.gpt_analyzer = OpenAIEnhancedAnalyzer(database_path=database_path)
            self.gpt_available = True
            logger.info("GPT analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"GPT analyzer initialization failed: {e}")
            self.gpt_available = False
    
    def enhance_race_prediction(self, race_file_path: str, 
                              include_betting_strategy: bool = True,
                              include_pattern_analysis: bool = True) -> Dict[str, Any]:
        """Enhance a single race prediction with GPT analysis"""
        
        if not self.gpt_available:
            return {"error": "GPT analyzer not available"}
        
        try:
            # Load existing prediction data
            race_data = self._load_race_data(race_file_path)
            if not race_data:
                return {"error": "Could not load race data"}
            
            # Load existing ML predictions if available
            existing_predictions = self._load_existing_predictions(race_data['race_id'])
            
            # Convert to GPT format
            race_context = self._create_race_context(race_data)
            dog_profiles = self._create_dog_profiles(race_data['dogs'])
            
            # Run GPT analysis
            logger.info(f"Running GPT analysis for race: {race_context.venue} Race {race_context.race_number}")
            
            # 1. Base race analysis
            race_analysis = self.gpt_analyzer.analyze_race_with_gpt(
                race_context, dog_profiles, 'race_analysis'
            )
            
            # 2. Enhance existing ML predictions if available
            enhanced_predictions = {}
            if existing_predictions:
                enhanced_predictions = self.gpt_analyzer.enhance_predictions_with_gpt(
                    existing_predictions['predictions'], race_context
                )
                # Store the base ML predictions for merging later
                enhanced_predictions['base_predictions'] = existing_predictions['predictions']
            
            # 3. Generate betting strategy
            betting_strategy = {}
            if include_betting_strategy and not race_analysis.get('error'):
                betting_strategy = self.gpt_analyzer.generate_betting_insights(
                    race_analysis, 
                    race_data.get('odds_data', []),
                    bankroll=1000, 
                    risk_tolerance="medium"
                )
            
            # 4. Historical pattern analysis
            pattern_analysis = {}
            if include_pattern_analysis:
                pattern_analysis = self.gpt_analyzer.analyze_historical_patterns(
                    race_context.venue, 
                    race_context.distance.replace('m', ''),
                    time_period_days=60
                )
            
            # 5. Compile comprehensive analysis with enhanced predictions that preserve ML scores
            # Create enhanced predictions that merge GPT insights with original ML scores
            final_enhanced_predictions = self._merge_gpt_with_ml_predictions(
                existing_predictions.get('predictions', []) if existing_predictions else [],
                race_analysis,
                enhanced_predictions
            )
            
            comprehensive_analysis = {
                "race_info": {
                    "venue": race_context.venue,
                    "race_number": race_context.race_number,
                    "race_date": race_context.race_date,
                    "distance": race_context.distance,
                    "grade": race_context.grade,
                    "track_condition": race_context.track_condition,
                    "weather": race_context.weather
                },
                "gpt_race_analysis": race_analysis,
                "enhanced_ml_predictions": enhanced_predictions,
                "betting_strategy": betting_strategy,
                "pattern_analysis": pattern_analysis,
                "merged_predictions": final_enhanced_predictions,  # The properly merged predictions
                "analysis_summary": self._create_analysis_summary(
                    race_analysis, enhanced_predictions, betting_strategy
                ),
                "timestamp": datetime.now().isoformat(),
                "tokens_used": self._calculate_total_tokens_used([
                    race_analysis, enhanced_predictions, betting_strategy, pattern_analysis
                ])
            }
            
            # Save enhanced analysis
            self._save_enhanced_analysis(comprehensive_analysis, race_data['race_id'])
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error enhancing race prediction: {e}")
            return {"error": str(e)}
    
    def enhance_multiple_races(self, race_files: List[str], 
                             max_races: int = 5) -> Dict[str, Any]:
        """Enhance multiple race predictions"""
        
        results = []
        total_tokens = 0
        successful_enhancements = 0
        
        for i, race_file in enumerate(race_files[:max_races]):
            logger.info(f"Enhancing race {i+1}/{min(len(race_files), max_races)}: {race_file}")
            
            enhancement = self.enhance_race_prediction(race_file)
            
            if 'error' not in enhancement:
                successful_enhancements += 1
                total_tokens += enhancement.get('tokens_used', 0)
            
            results.append({
                "race_file": race_file,
                "enhancement": enhancement
            })
        
        # Create batch summary
        batch_summary = {
            "total_races_processed": len(results),
            "successful_enhancements": successful_enhancements,
            "total_tokens_used": total_tokens,
            "average_tokens_per_race": total_tokens / successful_enhancements if successful_enhancements > 0 else 0,
            "estimated_cost_usd": self._estimate_cost(total_tokens),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "batch_summary": batch_summary,
            "race_enhancements": results
        }
    
    def generate_daily_insights(self, date_str: str = None) -> Dict[str, Any]:
        """Generate daily insights across all races"""
        
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Find all predictions for the given date
        prediction_files = self._find_predictions_by_date(date_str)
        
        if not prediction_files:
            return {"error": f"No predictions found for {date_str}"}
        
        # Load all race data
        daily_race_data = []
        for pred_file in prediction_files:
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    daily_race_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load prediction file {pred_file}: {e}")
        
        if not daily_race_data:
            return {"error": "Could not load any prediction data"}
        
        # Create comprehensive daily analysis prompt
        daily_prompt = f"""
        Analyze these greyhound racing predictions for {date_str} and provide comprehensive daily insights:

        DAILY RACE DATA:
        {json.dumps(daily_race_data, indent=2)[:8000]}  # Limit to prevent token overflow

        Provide:
        1. Overall market trends and patterns
        2. Best value opportunities across all races
        3. Risk management strategy for the day
        4. Venues showing strong form patterns
        5. Trainers/kennels in good form
        6. Weather/track condition impacts
        7. Recommended betting portfolio
        8. Races to avoid and why

        Focus on creating a profitable day-long strategy.
        """
        
        try:
            response = self.gpt_analyzer.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are creating a comprehensive daily greyhound racing strategy."},
                    {"role": "user", "content": daily_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            daily_insights = response.choices[0].message.content
            
            return {
                "date": date_str,
                "total_races_analyzed": len(daily_race_data),
                "daily_insights": daily_insights,
                "structured_recommendations": self._parse_daily_recommendations(daily_insights),
                "timestamp": datetime.now().isoformat(),
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating daily insights: {e}")
            return {"error": str(e)}
    
    def create_comprehensive_report(self, race_ids: List[str]) -> str:
        """Create a comprehensive report combining ML and GPT analysis"""
        
        report_sections = []
        
        # Header
        report_sections.append("=" * 80)
        report_sections.append("GREYHOUND RACING COMPREHENSIVE ANALYSIS REPORT")
        report_sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"Enhanced with OpenAI GPT-4 Analysis")
        report_sections.append("=" * 80)
        report_sections.append("")
        
        # Load enhanced analyses
        for race_id in race_ids:
            enhanced_file = self.enhanced_output_dir / f"gpt_enhanced_{race_id}.json"
            
            if enhanced_file.exists():
                try:
                    with open(enhanced_file, 'r') as f:
                        analysis = json.load(f)
                    
                    # Add race section
                    race_info = analysis.get('race_info', {})
                    report_sections.append(f"RACE: {race_info.get('venue')} Race {race_info.get('race_number')}")
                    report_sections.append(f"Date: {race_info.get('race_date')}")
                    report_sections.append(f"Distance: {race_info.get('distance')} | Grade: {race_info.get('grade')}")
                    report_sections.append(f"Track: {race_info.get('track_condition')}")
                    report_sections.append("-" * 50)
                    
                    # Add GPT insights
                    gpt_analysis = analysis.get('gpt_race_analysis', {})
                    if 'raw_analysis' in gpt_analysis:
                        report_sections.append("GPT ANALYSIS:")
                        report_sections.append(gpt_analysis['raw_analysis'][:1000] + "...")
                        report_sections.append("")
                    
                    # Add betting strategy
                    betting = analysis.get('betting_strategy', {})
                    if 'betting_strategy' in betting:
                        report_sections.append("BETTING STRATEGY:")
                        report_sections.append(betting['betting_strategy'][:800] + "...")
                        report_sections.append("")
                    
                    # Add summary
                    summary = analysis.get('analysis_summary', {})
                    if summary:
                        report_sections.append("KEY INSIGHTS:")
                        for key, value in summary.items():
                            if isinstance(value, str) and len(value) < 200:
                                report_sections.append(f"• {key}: {value}")
                        report_sections.append("")
                    
                    report_sections.append("=" * 50)
                    report_sections.append("")
                    
                except Exception as e:
                    logger.warning(f"Could not include race {race_id} in report: {e}")
        
        # Footer
        report_sections.append("Report generated by GPT-Enhanced Greyhound Racing Analysis System")
        report_sections.append(f"Total races analyzed: {len(race_ids)}")
        
        return "\n".join(report_sections)
    
    def _load_race_data(self, race_file_path: str) -> Optional[Dict]:
        """Load race data from CSV file"""
        try:
            race_path = Path(race_file_path)
            if not race_path.exists():
                logger.error(f"Race file not found: {race_file_path}")
                return None
            
            # Read CSV data
            df = pd.read_csv(race_file_path)
            
            # Extract race metadata from filename
            race_id = race_path.stem
            filename_parts = race_id.split(' - ')
            
            # Parse filename: "Race 1 - TAREE - 2025-07-26"
            if len(filename_parts) >= 3:
                race_number_part = filename_parts[0].replace('Race ', '')
                venue = filename_parts[1]
                race_date = filename_parts[2]
            else:
                race_number_part = '1'
                venue = 'Unknown'
                race_date = datetime.now().strftime('%Y-%m-%d')
            
            # Extract additional info from CSV data
            if not df.empty:
                # Get distance from first row
                first_row = df.iloc[0]
                distance = f"{first_row.get('DIST', 500)}m"
                
                # Get track from TRACK column (e.g., TARE -> TAREE)
                track_code = first_row.get('TRACK', venue)
                if track_code and venue == 'Unknown':
                    # Try to expand track code
                    track_mapping = {
                        'TARE': 'TAREE',
                        'MAIT': 'MAITLAND', 
                        'GRDN': 'GOSFORD',
                        'CASO': 'CASINO'
                    }
                    venue = track_mapping.get(track_code, track_code)
                
                # Get grade from G column
                grade = first_row.get('G', 'M')  # M for Maiden typically
            else:
                distance = '500m'
                grade = 'Unknown'
            
            # Convert to race data structure
            race_data = {
                "race_id": race_id,
                "venue": venue,
                "race_number": int(race_number_part) if race_number_part.isdigit() else 1,
                "distance": distance,
                "grade": f"Grade {grade}" if grade != 'Unknown' else 'Unknown',
                "track_condition": 'Good',  # Default since not in CSV
                "race_date": race_date,
                "dogs": []
            }
            
            # Extract dog data from actual CSV structure
            # Group rows by dog (consecutive rows with same number or empty string)
            current_dog = None
            for _, row in df.iterrows():
                dog_name = str(row.get('Dog Name', ''))
                
                # If this is a new dog (has a number at start)
                if dog_name and dog_name != 'nan' and not dog_name.startswith('"'):
                    # Parse dog name (e.g., "1. Keegan's Reports")
                    if '. ' in dog_name:
                        dog_number, actual_name = dog_name.split('. ', 1)
                    else:
                        dog_number = '0'
                        actual_name = dog_name
                    
                    current_dog = {
                        "name": actual_name,
                        "box_number": int(row.get('BOX', 0)) if pd.notna(row.get('BOX')) else 0,
                        "trainer": 'Unknown',  # Not in this CSV format
                        "weight": float(row.get('WGT', 0)) if pd.notna(row.get('WGT')) else 0,
                        "recent_form": [int(row.get('PLC', 0))] if pd.notna(row.get('PLC')) else [],
                        "best_time": float(row.get('TIME', 0)) if pd.notna(row.get('TIME')) else 0,
                        "last_start": str(row.get('DATE', '')),
                        "comments": f"Sex: {row.get('Sex', '')}, SP: {row.get('SP', '')}"
                    }
                    race_data["dogs"].append(current_dog)
                elif current_dog and dog_name.startswith('"'):
                    # This is additional form data for the current dog
                    if pd.notna(row.get('PLC')):
                        current_dog["recent_form"].append(int(row.get('PLC')))
                    # Update best time if this one is better
                    if pd.notna(row.get('TIME')) and float(row.get('TIME', 0)) < current_dog["best_time"]:
                        current_dog["best_time"] = float(row.get('TIME'))
            
            return race_data
            
        except Exception as e:
            logger.error(f"Error loading race data: {e}")
            return None
    
    def _load_existing_predictions(self, race_id: str) -> Optional[Dict]:
        """Load existing ML predictions for a race"""
        try:
            prediction_file = self.predictions_dir / f"prediction_{race_id}.json"
            
            if prediction_file.exists():
                with open(prediction_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not load existing predictions for {race_id}: {e}")
            return None
    
    def _create_race_context(self, race_data: Dict) -> RaceContext:
        """Convert race data to RaceContext object"""
        return RaceContext(
            venue=race_data.get('venue', 'Unknown'),
            race_number=race_data.get('race_number', 1),
            race_date=race_data.get('race_date', datetime.now().strftime('%Y-%m-%d')),
            distance=race_data.get('distance', '500m'),
            grade=race_data.get('grade', 'Unknown'),
            track_condition=race_data.get('track_condition', 'Good'),
            weather=race_data.get('weather'),
            field_size=len(race_data.get('dogs', []))
        )
    
    def _create_dog_profiles(self, dogs_data: List[Dict]) -> List[DogProfile]:
        """Convert dog data to DogProfile objects"""
        profiles = []
        
        for dog_data in dogs_data:
            profile = DogProfile(
                name=dog_data.get('name', 'Unknown'),
                box_number=dog_data.get('box_number', 0),
                recent_form=dog_data.get('recent_form', []),
                win_rate=self._calculate_win_rate(dog_data.get('recent_form', [])),
                place_rate=self._calculate_place_rate(dog_data.get('recent_form', [])),
                best_time=dog_data.get('best_time', 0),
                trainer=dog_data.get('trainer', 'Unknown'),
                weight=dog_data.get('weight', 0),
                historical_stats=self._create_historical_stats(dog_data)
            )
            profiles.append(profile)
        
        return profiles
    
    def _parse_form_string(self, form_str: str) -> List[int]:
        """Parse form string into list of positions"""
        if not form_str:
            return []
        
        form_positions = []
        # Handle various form formats
        for char in str(form_str):
            if char.isdigit():
                form_positions.append(int(char))
        
        return form_positions[-10:]  # Last 10 runs
    
    def _calculate_win_rate(self, recent_form: List[int]) -> float:
        """Calculate win rate from recent form"""
        if not recent_form:
            return 0.0
        wins = sum(1 for pos in recent_form if pos == 1)
        return (wins / len(recent_form)) * 100
    
    def _calculate_place_rate(self, recent_form: List[int]) -> float:
        """Calculate place rate from recent form"""
        if not recent_form:
            return 0.0
        places = sum(1 for pos in recent_form if pos <= 3)
        return (places / len(recent_form)) * 100
    
    def _create_historical_stats(self, dog_data: Dict) -> Dict:
        """Create historical stats dictionary"""
        form = dog_data.get('recent_form', [])
        return {
            "total_runs": len(form),
            "wins": sum(1 for pos in form if pos == 1),
            "places": sum(1 for pos in form if pos <= 3),
            "average_position": sum(form) / len(form) if form else 0,
            "best_recent_position": min(form) if form else 0,
            "last_start": dog_data.get('last_start', ''),
            "comments": dog_data.get('comments', '')
        }
    
    def _create_analysis_summary(self, race_analysis: Dict, 
                               enhanced_predictions: Dict, 
                               betting_strategy: Dict) -> Dict:
        """Create a summary of all analysis components"""
        summary = {
            "analysis_confidence": race_analysis.get('analysis_confidence', 0),
            "gpt_available": 'error' not in race_analysis,
            "prediction_enhancement": 'error' not in enhanced_predictions,
            "betting_strategy_available": 'error' not in betting_strategy,
            "key_insights": []
        }
        
        # Extract key insights from GPT analysis
        if 'structured_insights' in race_analysis:
            summary["key_insights"].extend(race_analysis['structured_insights'][:3])
        
        # Add prediction validation insights
        if 'enhanced_insights' in enhanced_predictions:
            insights = enhanced_predictions['enhanced_insights']
            if insights.get('key_factors'):
                summary["key_insights"].extend(insights['key_factors'][:2])
        
        return summary
    
    def _calculate_total_tokens_used(self, analyses: List[Dict]) -> int:
        """Calculate total tokens used across all analyses"""
        total_tokens = 0
        for analysis in analyses:
            if isinstance(analysis, dict):
                total_tokens += analysis.get('tokens_used', 0)
        return total_tokens
    
    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost in USD based on GPT-4 pricing"""
        # GPT-4 pricing (approximate): $0.03 per 1K tokens input, $0.06 per 1K tokens output
        # Assuming roughly 50/50 input/output split
        cost_per_1k = 0.045  # Average
        return (total_tokens / 1000) * cost_per_1k
    
    def _find_predictions_by_date(self, date_str: str) -> List[Path]:
        """Find all prediction files for a given date"""
        prediction_files = []
        
        if self.predictions_dir.exists():
            for file_path in self.predictions_dir.glob("prediction_*.json"):
                # Check if file contains the date (simple heuristic)
                if date_str.replace('-', '') in file_path.name:
                    prediction_files.append(file_path)
        
        return prediction_files
    
    def _parse_daily_recommendations(self, insights_text: str) -> Dict:
        """Parse daily recommendations from GPT insights"""
        recommendations = {
            "top_value_bets": [],
            "races_to_avoid": [],
            "portfolio_strategy": "",
            "risk_management": []
        }
        
        # Simple parsing based on keywords
        lines = insights_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'value' in line.lower() and ('bet' in line.lower() or 'opportunity' in line.lower()):
                current_section = 'top_value_bets'
            elif 'avoid' in line.lower():
                current_section = 'races_to_avoid'
            elif 'portfolio' in line.lower() or 'strategy' in line.lower():
                current_section = 'portfolio_strategy'
            elif 'risk' in line.lower():
                current_section = 'risk_management'
            
            if current_section and line and len(line) > 15:
                if current_section == 'portfolio_strategy':
                    recommendations[current_section] += line + " "
                else:
                    recommendations[current_section].append(line)
        
        return recommendations
    
    def _merge_gpt_with_ml_predictions(self, ml_predictions: List[Dict], 
                                     gpt_race_analysis: Dict, 
                                     gpt_enhanced_predictions: Dict) -> List[Dict]:
        """Merge GPT insights with ML predictions while preserving original scores"""
        if not ml_predictions:
            return []
        
        merged_predictions = []
        
        # Extract GPT finishing order if available
        gpt_order = []
        try:
            if 'raw_analysis' in gpt_race_analysis:
                import re
                # Try to extract predicted order from GPT analysis
                analysis_text = gpt_race_analysis['raw_analysis']
                # Look for order patterns in the JSON-like structure
                order_match = re.search(r'"order":\s*\[(.*?)\]', analysis_text, re.DOTALL)
                if order_match:
                    order_text = order_match.group(1)
                    # Extract dog names from the order
                    dog_names = re.findall(r'"([^"]+)"', order_text)
                    gpt_order = dog_names
        except Exception as e:
            logger.warning(f"Could not extract GPT order: {e}")
        
        # Create mapping of dog names to GPT order positions
        gpt_order_map = {}
        for i, dog_name in enumerate(gpt_order):
            gpt_order_map[dog_name.upper()] = i + 1
        
        # Process each ML prediction
        for ml_pred in ml_predictions:
            merged_pred = ml_pred.copy()  # Start with ML prediction
            
            dog_name = ml_pred.get('dog_name', '').upper()
            
            # Add GPT insights while preserving ML scores
            gpt_insights = {
                'gpt_predicted_position': gpt_order_map.get(dog_name),
                'gpt_analysis_available': 'error' not in gpt_race_analysis,
                'gpt_confidence': gpt_race_analysis.get('analysis_confidence', 0),
            }
            
            # Preserve original ML scores but add GPT context
            original_final_score = ml_pred.get('final_score', ml_pred.get('prediction_score', 0.5))
            original_confidence = ml_pred.get('confidence_level', 'LOW')
            
            # Adjust confidence based on GPT agreement
            enhanced_confidence = original_confidence
            if gpt_insights['gpt_predicted_position']:
                ml_rank = ml_pred.get('predicted_rank', 999)
                gpt_rank = gpt_insights['gpt_predicted_position']
                
                # If GPT and ML agree on top positions, increase confidence
                if ml_rank <= 3 and gpt_rank <= 3:
                    if original_confidence == 'LOW':
                        enhanced_confidence = 'MEDIUM'
                    elif original_confidence == 'MEDIUM':
                        enhanced_confidence = 'HIGH'
                elif abs(ml_rank - gpt_rank) > 3:
                    # Large disagreement - might lower confidence
                    if original_confidence == 'HIGH':
                        enhanced_confidence = 'MEDIUM'
            
            # Update the prediction with enhanced information
            merged_pred.update({
                'final_score': original_final_score,  # Preserve original ML score
                'prediction_score': original_final_score,  # Ensure consistency
                'confidence_level': enhanced_confidence,  # Enhanced confidence
                'gpt_insights': gpt_insights,
                'enhanced_with_gpt': True,
                'original_ml_confidence': original_confidence
            })
            
            # Add GPT reasoning to existing reasoning
            if 'reasoning' in merged_pred:
                if gpt_insights['gpt_predicted_position']:
                    merged_pred['reasoning'].append(
                        f"GPT Predicted Position: {gpt_insights['gpt_predicted_position']}"
                    )
            
            merged_predictions.append(merged_pred)
        
        # Sort by original final_score to maintain ML-based ranking
        merged_predictions.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Update predicted ranks
        for i, pred in enumerate(merged_predictions):
            pred['predicted_rank'] = i + 1
        
        return merged_predictions
    
    def _save_enhanced_analysis(self, analysis: Dict, race_id: str):
        """Save enhanced analysis to file"""
        try:
            output_file = self.enhanced_output_dir / f"gpt_enhanced_{race_id}.json"
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved enhanced analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced analysis: {e}")

def main():
    """Main function for testing the enhancer"""
    enhancer = GPTPredictionEnhancer()
    
    # Example: enhance a single race
    upcoming_dir = Path("./upcoming_races")
    if upcoming_dir.exists():
        csv_files = list(upcoming_dir.glob("*.csv"))
        if csv_files:
            print(f"Enhancing prediction for: {csv_files[0]}")
            result = enhancer.enhance_race_prediction(str(csv_files[0]))
            
            if 'error' not in result:
                print("Enhancement successful!")
                print(f"Tokens used: {result.get('tokens_used', 0)}")
                print(f"Analysis confidence: {result.get('gpt_race_analysis', {}).get('analysis_confidence', 0):.2f}")
            else:
                print(f"Enhancement failed: {result['error']}")
        else:
            print("No CSV files found in upcoming_races directory")
    else:
        print("upcoming_races directory not found")

if __name__ == "__main__":
    main()
