#!/usr/bin/env python3
"""
OpenAI Enhanced Greyhound Racing Analyzer
=========================================

Integrates OpenAI's GPT models to enhance greyhound racing analysis and predictions
with advanced natural language processing and reasoning capabilities.

Features:
- Race analysis with contextual insights
- Form guide interpretation
- Weather impact analysis
- Betting strategy recommendations
- Historical pattern recognition
- Risk assessment
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# import openai  # Legacy direct usage replaced by wrapper
import pandas as pd

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RaceContext:
    """Container for race context data"""

    venue: str
    race_number: int
    race_date: str
    distance: str
    grade: str
    track_condition: str
    weather: Optional[Dict] = None
    field_size: int = 0


@dataclass
class DogProfile:
    """Container for dog profile data"""

    name: str
    box_number: int
    recent_form: List[int]
    win_rate: float
    place_rate: float
    best_time: float
    trainer: str
    weight: float
    historical_stats: Dict


class OpenAIEnhancedAnalyzer:
    """Enhanced analyzer using OpenAI's GPT models for advanced race analysis"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        database_path: str = "greyhound_racing_data.db",
    ):
        # Initialize OpenAI API
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client via connectivity verifier wrapper
        from openai_connectivity_verifier import OpenAIConnectivityVerifier
        self._verifier = OpenAIConnectivityVerifier()
        # If api_key provided externally, set on verifier for consistency
        self._verifier.api_key = self.api_key
        if not self._verifier.client:
            # Attempt to initialize using available key (mock-safe)
            self._verifier.initialize_client(use_mock=(not bool(self.api_key)))
        self.client = self._verifier.get_enhanced_client()
        self.database_path = database_path

        # Configuration via centralized config
        from config.openai_config import get_openai_config
        cfg = get_openai_config()
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        # Check if we should use mock mode
        self.mock_mode = False
        if (
            not self.api_key
            or self.api_key.startswith("sk-test")
            or self.api_key == "your_openai_api_key_here"
        ):
            logger.warning("Using mock OpenAI mode - API key not valid")
            self.mock_mode = True

        # Analysis templates
        self.setup_analysis_templates()

    def setup_analysis_templates(self):
        """Setup analysis prompt templates for different types of analysis"""

        self.templates = {
            "race_analysis": """
You are an expert greyhound racing analyst with decades of experience. Analyze this race data and provide insights.

RACE CONTEXT:
- Venue: {venue}
- Race Number: {race_number}
- Date: {race_date}
- Distance: {distance}
- Grade: {grade}
- Track Condition: {track_condition}
- Weather: {weather}

DOGS IN RACE:
{dogs_data}

HISTORICAL VENUE DATA:
{venue_stats}

Please provide a comprehensive analysis including:
1. Race conditions impact on performance
2. Key contenders and their strengths/weaknesses
3. Value betting opportunities
4. Risk factors to consider
5. Predicted finishing order with confidence levels
6. Specific insights about track/distance specialists

Format your response as structured JSON with clear reasoning.
""",
            "form_guide_analysis": """
You are analyzing greyhound form guides. Interpret the racing patterns and provide insights.

DOG: {dog_name}
RECENT FORM: {recent_form}
STATISTICS:
- Win Rate: {win_rate}%
- Place Rate: {place_rate}%
- Best Time: {best_time}s
- Trainer: {trainer}
- Weight: {weight}kg

HISTORICAL PERFORMANCE:
{historical_data}

UPCOMING RACE CONDITIONS:
{race_conditions}

Analyze this dog's chances considering:
1. Current form trend (improving/declining)
2. Suitability to race conditions
3. Class/grade appropriateness  
4. Trainer/kennel patterns
5. Physical condition indicators
6. Historical performance at this venue/distance

Provide a detailed assessment with confidence rating (1-10).
""",
            "betting_strategy": """
You are a professional greyhound racing betting strategist. Given the race analysis and current odds, recommend betting strategies.

RACE ANALYSIS:
{race_analysis}

CURRENT ODDS:
{odds_data}

BANKROLL MANAGEMENT:
- Risk Tolerance: {risk_tolerance}
- Betting Bank: ${betting_bank}

Recommend:
1. Best value bets with reasoning
2. Exotic betting combinations (quinella, trifecta)
3. Lay betting opportunities
4. Risk management approach
5. Staking plan for this race
6. Expected value calculations

Focus on sustainable, profitable strategies rather than long-shot bets.
""",
            "weather_impact": """
You are analyzing how weather conditions affect greyhound racing performance.

CURRENT CONDITIONS:
{weather_data}

HISTORICAL WEATHER PERFORMANCE:
{historical_weather_data}

RACE DETAILS:
- Track Surface: {track_surface}
- Distance: {distance}
- Current Track Rating: {track_condition}

Analyze:
1. How current conditions favor different running styles
2. Dogs that perform well/poorly in these conditions
3. Expected pace scenarios
4. Track bias implications
5. Adjustments to standard form analysis
6. Confidence levels in predictions given conditions

Provide specific, actionable insights for this race.
""",
            "risk_assessment": """
You are conducting risk assessment for greyhound racing predictions.

PREDICTION DATA:
{predictions}

CONFIDENCE INDICATORS:
{confidence_data}

MARKET CONDITIONS:
{market_data}

Assess:
1. Prediction reliability factors
2. Potential upset scenarios
3. Model limitations for this race
4. Market efficiency indicators
5. Recommended confidence levels
6. Hedging strategies
7. When to avoid betting

Provide a comprehensive risk framework for this race.
""",
        }

    def analyze_race_with_gpt(
        self,
        race_context: RaceContext,
        dogs: List[DogProfile],
        analysis_type: str = "race_analysis",
    ) -> Dict[str, Any]:
        """Analyze race using GPT with comprehensive context"""

        try:
            # Prepare context data
            dogs_data = self._format_dogs_for_analysis(dogs)
            venue_stats = self._get_venue_historical_stats(race_context.venue)
            weather_data = race_context.weather or {"condition": "Unknown"}

            # Select appropriate template
            template = self.templates.get(
                analysis_type, self.templates["race_analysis"]
            )

            # Format prompt
            prompt = template.format(
                venue=race_context.venue,
                race_number=race_context.race_number,
                race_date=race_context.race_date,
                distance=race_context.distance,
                grade=race_context.grade,
                track_condition=race_context.track_condition,
                weather=json.dumps(weather_data, indent=2),
                dogs_data=dogs_data,
                venue_stats=venue_stats,
            )

            # Call OpenAI via wrapper
            from utils.openai_wrapper import OpenAIWrapper
            wrapper = OpenAIWrapper(self.client)
            from src.ai.prompts import system_prompt
            resp = wrapper.chat(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt("analyst"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model=self.model,
            )

            # Extract and parse response
            analysis_text = resp.text
            class Dummy:
                pass
            response = Dummy()
            response.usage = type("U", (), resp.usage) if isinstance(resp.usage, dict) else resp.usage

            # Try to parse as JSON, fallback to text analysis
            try:
                analysis_data = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_data = {
                    "analysis_type": analysis_type,
                    "raw_analysis": analysis_text,
                    "structured_insights": self._extract_insights_from_text(
                        analysis_text
                    ),
                }

            # Add metadata
            analysis_data.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": self.model,
                    "race_id": f"{race_context.venue}_{race_context.race_number}_{race_context.race_date}",
                    "analysis_confidence": self._calculate_analysis_confidence(
                        analysis_data
                    ),
                    "tokens_used": response.usage.total_tokens,
                }
            )

            return analysis_data

        except Exception as e:
            logger.error(f"GPT analysis failed: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
            }

    def enhance_predictions_with_gpt(
        self, base_predictions: List[Dict], race_context: RaceContext
    ) -> Dict[str, Any]:
        """Enhance existing ML predictions with GPT insights"""

        # Prepare prediction data for GPT analysis
        prediction_summary = {
            "top_3_picks": base_predictions[:3],
            "field_analysis": base_predictions,
            "prediction_confidence": self._calculate_prediction_confidence(
                base_predictions
            ),
            "model_consensus": self._analyze_model_consensus(base_predictions),
        }

        # Create enhanced analysis prompt
        enhancement_prompt = f"""
        Analyze these ML-generated greyhound racing predictions and provide enhanced insights:

        BASE PREDICTIONS:
        {json.dumps(prediction_summary, indent=2)}

        RACE CONTEXT:
        {json.dumps(race_context.__dict__, indent=2)}

        As an expert analyst, provide:
        1. Validation of ML predictions against racing logic
        2. Additional factors the ML model might have missed
        3. Contrarian viewpoints worth considering
        4. Confidence adjustments based on context
        5. Narrative explanation of likely race dynamics
        6. Specific betting recommendations

        Focus on adding human expertise that complements the quantitative analysis.
        """

        try:
            from utils.openai_wrapper import OpenAIWrapper
            wrapper = OpenAIWrapper(self.client)
            from src.ai.prompts import system_prompt
            resp = wrapper.chat(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt("analyst"),
                    },
                    {"role": "user", "content": enhancement_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
                model=self.model,
            )

            enhanced_analysis = resp.text

            return {
                "base_predictions": base_predictions,
                "gpt_enhancement": enhanced_analysis,
                "enhanced_insights": self._parse_enhancement_insights(
                    enhanced_analysis
                ),
                "final_recommendations": self._generate_final_recommendations(
                    base_predictions, enhanced_analysis
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction enhancement failed: {str(e)}")
            return {
                "base_predictions": base_predictions,
                "enhancement_error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def generate_betting_insights(
        self,
        race_analysis: Dict,
        odds_data: List[Dict],
        bankroll: float = 1000,
        risk_tolerance: str = "medium",
    ) -> Dict[str, Any]:
        """Generate betting strategy using GPT"""

        betting_prompt = self.templates["betting_strategy"].format(
            race_analysis=json.dumps(race_analysis, indent=2),
            odds_data=json.dumps(odds_data, indent=2),
            risk_tolerance=risk_tolerance,
            betting_bank=bankroll,
        )

        try:
            from utils.openai_wrapper import OpenAIWrapper
            wrapper = OpenAIWrapper(self.client)
            resp = wrapper.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional betting strategist focused on sustainable, profitable approaches.",
                    },
                    {"role": "user", "content": betting_prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
                model=self.model,
            )

            betting_strategy = resp.text

            return {
                "betting_strategy": betting_strategy,
                "structured_recommendations": self._parse_betting_recommendations(
                    betting_strategy
                ),
                "risk_assessment": self._assess_betting_risk(
                    betting_strategy, odds_data
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Betting insight generation failed: {str(e)}")
            return {"error": str(e)}

    def analyze_historical_patterns(
        self, venue: str, distance: str, time_period_days: int = 90
    ) -> Dict[str, Any]:
        """Use GPT to analyze historical patterns and trends"""

        # Get historical data
        historical_data = self._get_historical_race_data(
            venue, distance, time_period_days
        )

        pattern_prompt = f"""
        Analyze these historical greyhound racing patterns for {venue} at {distance}m:

        HISTORICAL DATA (Last {time_period_days} days):
        {json.dumps(historical_data, indent=2)}

        Identify:
        1. Seasonal/temporal trends
        2. Successful trainer/dog patterns
        3. Track bias indicators
        4. Class progression patterns
        5. Optimal betting spots
        6. Market inefficiencies
        7. Predictive indicators for future races

        Provide actionable insights for upcoming races at this venue/distance.
        """

        try:
            from utils.openai_wrapper import OpenAIWrapper
            wrapper = OpenAIWrapper(self.client)
            from src.ai.prompts import system_prompt
            resp = wrapper.chat(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt("analyst"),
                    },
                    {"role": "user", "content": pattern_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
                model=self.model,
            )

            pattern_analysis = resp.text

            return {
                "venue": venue,
                "distance": distance,
                "analysis_period": f"{time_period_days} days",
                "pattern_insights": pattern_analysis,
                "structured_patterns": self._extract_pattern_insights(pattern_analysis),
                "confidence_score": self._calculate_pattern_confidence(historical_data),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            return {"error": str(e)}

    def _format_dogs_for_analysis(self, dogs: List[DogProfile]) -> str:
        """Format dog data for GPT analysis"""
        formatted_dogs = []

        for dog in dogs:
            dog_info = f"""
            Dog: {dog.name} (Box {dog.box_number})
            Recent Form: {' '.join(map(str, dog.recent_form))}
            Win Rate: {dog.win_rate:.1f}%
            Place Rate: {dog.place_rate:.1f}%
            Best Time: {dog.best_time}s
            Trainer: {dog.trainer}
            Weight: {dog.weight}kg
            Historical Stats: {json.dumps(dog.historical_stats, indent=2)}
            """
            formatted_dogs.append(dog_info)

        return "\n".join(formatted_dogs)

    def _get_venue_historical_stats(self, venue: str) -> str:
        """Get historical statistics for a venue"""
        try:
            conn = sqlite3.connect(self.database_path)
            query = """
            SELECT 
                AVG(CAST(individual_time AS FLOAT)) as avg_time,
                COUNT(*) as total_races,
                AVG(field_size) as avg_field_size
            FROM dog_race_data d
            JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.venue = ? AND individual_time IS NOT NULL
            """

            df = pd.read_sql_query(query, conn, params=[venue])
            conn.close()

            return df.to_json(orient="records")

        except Exception as e:
            logger.error(f"Error getting venue stats: {e}")
            return "{}"

    def _get_historical_race_data(self, venue: str, distance: str, days: int) -> Dict:
        """Get historical race data for pattern analysis"""
        try:
            conn = sqlite3.connect(self.database_path)

            # Get recent races at this venue/distance
            query = """
            SELECT r.race_date, r.race_number, r.winner_name, r.winner_odds,
                   d.individual_time, d.finish_position
            FROM race_metadata r
            JOIN dog_race_data d ON r.race_id = d.race_id
            WHERE r.venue = ? AND r.distance = ?
            AND date(r.race_date) >= date('now', '-{} days')
            ORDER BY r.race_date DESC
            """.format(
                days
            )

            df = pd.read_sql_query(query, conn, params=[venue, distance])
            conn.close()

            # Summarize the data
            summary = {
                "total_races": len(df),
                "unique_winners": df["winner_name"].nunique(),
                "avg_winning_odds": (
                    df["winner_odds"].mean() if "winner_odds" in df.columns else 0
                ),
                "avg_winning_time": (
                    pd.to_numeric(df[df["finish_position"] == 1]["individual_time"], errors="coerce").mean()
                    if len(df) > 0
                    else 0
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {}

    def _extract_insights_from_text(self, text: str) -> List[str]:
        """Extract key insights from unstructured analysis text"""
        # Simple keyword-based insight extraction
        insights = []
        sentences = text.split(".")

        for sentence in sentences:
            sentence = sentence.strip()
            if any(
                keyword in sentence.lower()
                for keyword in [
                    "recommend",
                    "likely",
                    "expect",
                    "should",
                    "strong",
                    "weak",
                    "value",
                ]
            ):
                if len(sentence) > 20:  # Filter out very short sentences
                    insights.append(sentence)

        return insights[:10]  # Limit to top 10 insights

    def _calculate_analysis_confidence(self, analysis_data: Dict) -> float:
        """Calculate confidence score for GPT analysis"""
        # Simple confidence calculation based on data completeness and specificity
        base_confidence = 0.7

        # Adjust based on analysis detail
        if isinstance(analysis_data.get("raw_analysis"), str):
            text_length = len(analysis_data["raw_analysis"])
            if text_length > 1000:
                base_confidence += 0.1
            elif text_length < 500:
                base_confidence -= 0.1

        # Adjust based on structured insights
        insights_count = len(analysis_data.get("structured_insights", []))
        if insights_count > 5:
            base_confidence += 0.1
        elif insights_count < 3:
            base_confidence -= 0.1

        return max(0.1, min(0.95, base_confidence))

    def _calculate_prediction_confidence(self, predictions: List[Dict]) -> float:
        """Calculate overall confidence in predictions"""
        if not predictions:
            return 0.0

        scores = [p.get("prediction_score", 0) for p in predictions]
        if not scores:
            return 0.0

        # Confidence based on score distribution
        top_score = max(scores)
        score_spread = top_score - min(scores)

        # Higher spread indicates clearer favorites
        confidence = min(0.95, 0.5 + (score_spread * 0.5))
        return confidence

    def _analyze_model_consensus(self, predictions: List[Dict]) -> Dict:
        """Analyze consensus among different prediction factors"""
        consensus_data = {
            "strong_consensus": [],
            "weak_consensus": [],
            "conflicting_signals": [],
        }

        for pred in predictions[:5]:  # Top 5
            traditional_score = pred.get("traditional_score", 0)
            ml_score = pred.get("ml_score", 0)
            final_score = pred.get("prediction_score", 0)

            # Analyze consensus
            score_variance = abs(traditional_score - ml_score)
            if score_variance < 0.1:
                consensus_data["strong_consensus"].append(pred["dog_name"])
            elif score_variance > 0.3:
                consensus_data["conflicting_signals"].append(pred["dog_name"])
            else:
                consensus_data["weak_consensus"].append(pred["dog_name"])

        return consensus_data

    def _parse_enhancement_insights(self, enhancement_text: str) -> Dict:
        """Parse structured insights from enhancement text"""
        insights = {
            "key_factors": [],
            "risk_warnings": [],
            "value_opportunities": [],
            "race_narrative": "",
        }

        # Simple parsing based on common patterns
        lines = enhancement_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if "factor" in line.lower() or "key" in line.lower():
                current_section = "key_factors"
            elif "risk" in line.lower() or "warning" in line.lower():
                current_section = "risk_warnings"
            elif "value" in line.lower() or "opportunity" in line.lower():
                current_section = "value_opportunities"
            elif "narrative" in line.lower() or "story" in line.lower():
                current_section = "race_narrative"

            if (
                current_section
                and line
                and not any(
                    word in line.lower() for word in ["analyze", "provide", "focus"]
                )
            ):
                if current_section == "race_narrative":
                    insights[current_section] += line + " "
                else:
                    insights[current_section].append(line)

        return insights

    def _generate_final_recommendations(
        self, base_predictions: List[Dict], enhancement_text: str
    ) -> List[Dict]:
        """Generate final recommendations combining ML and GPT insights"""
        recommendations = []

        # Enhanced top 3 picks with GPT insights
        for i, pred in enumerate(base_predictions[:3]):
            recommendation = {
                "rank": i + 1,
                "dog_name": pred["dog_name"],
                "ml_score": pred.get("prediction_score", 0),
                "confidence": pred.get("confidence_level", "MEDIUM"),
                "gpt_validation": "See enhancement analysis",
                "betting_recommendation": self._extract_betting_advice(
                    pred["dog_name"], enhancement_text
                ),
            }
            recommendations.append(recommendation)

        return recommendations

    def _extract_betting_advice(self, dog_name: str, enhancement_text: str) -> str:
        """Extract betting advice for specific dog from enhancement text"""
        # Simple text search for dog-specific advice
        sentences = enhancement_text.split(".")
        for sentence in sentences:
            if dog_name.lower() in sentence.lower() and any(
                word in sentence.lower()
                for word in ["bet", "back", "value", "avoid", "lay"]
            ):
                return sentence.strip()
        return "Consider based on overall analysis"

    def _parse_betting_recommendations(self, betting_text: str) -> Dict:
        """Parse structured betting recommendations"""
        recommendations = {
            "win_bets": [],
            "place_bets": [],
            "exotic_bets": [],
            "lay_bets": [],
            "avoid": [],
        }

        # Simple pattern matching for betting recommendations
        lines = betting_text.split("\n")
        for line in lines:
            line = line.strip().lower()
            if "win" in line and ("bet" in line or "back" in line):
                recommendations["win_bets"].append(line)
            elif "place" in line and "bet" in line:
                recommendations["place_bets"].append(line)
            elif any(word in line for word in ["quinella", "trifecta", "exacta"]):
                recommendations["exotic_bets"].append(line)
            elif "lay" in line:
                recommendations["lay_bets"].append(line)
            elif "avoid" in line:
                recommendations["avoid"].append(line)

        return recommendations

    def _assess_betting_risk(
        self, betting_strategy: str, odds_data: List[Dict]
    ) -> Dict:
        """Assess risk level of betting strategy"""
        risk_assessment = {
            "overall_risk": "MEDIUM",
            "risk_factors": [],
            "risk_mitigation": [],
        }

        # Analyze risk indicators in strategy text
        strategy_lower = betting_strategy.lower()

        # High risk indicators
        if any(
            word in strategy_lower
            for word in ["long shot", "upset", "exotic", "high odds"]
        ):
            risk_assessment["overall_risk"] = "HIGH"
            risk_assessment["risk_factors"].append("High odds selections")

        # Low risk indicators
        if any(
            word in strategy_lower
            for word in ["favorite", "safe", "conservative", "place"]
        ):
            if risk_assessment["overall_risk"] != "HIGH":
                risk_assessment["overall_risk"] = "LOW"

        # Risk mitigation suggestions
        if "diversify" in strategy_lower or "hedge" in strategy_lower:
            risk_assessment["risk_mitigation"].append(
                "Portfolio diversification recommended"
            )

        return risk_assessment

    def _extract_pattern_insights(self, pattern_text: str) -> Dict:
        """Extract structured insights from pattern analysis"""
        patterns = {
            "seasonal_trends": [],
            "successful_patterns": [],
            "market_inefficiencies": [],
            "predictive_indicators": [],
        }

        # Parse patterns from text
        lines = pattern_text.split("\n")
        current_category = None

        for line in lines:
            line = line.strip()
            if "seasonal" in line.lower() or "temporal" in line.lower():
                current_category = "seasonal_trends"
            elif "successful" in line.lower() or "winning" in line.lower():
                current_category = "successful_patterns"
            elif "inefficien" in line.lower() or "value" in line.lower():
                current_category = "market_inefficiencies"
            elif "indicator" in line.lower() or "predicti" in line.lower():
                current_category = "predictive_indicators"
            elif current_category and line and len(line) > 10:
                patterns[current_category].append(line)

        return patterns

    def _calculate_pattern_confidence(self, historical_data: Dict) -> float:
        """Calculate confidence in pattern analysis based on data quality"""
        base_confidence = 0.6

        # Adjust based on data volume
        total_races = historical_data.get("total_races", 0)
        if total_races > 100:
            base_confidence += 0.2
        elif total_races < 20:
            base_confidence -= 0.2

        # Adjust based on data diversity
        unique_winners = historical_data.get("unique_winners", 0)
        if unique_winners > 10:
            base_confidence += 0.1
        elif unique_winners < 5:
            base_confidence -= 0.1

        return max(0.1, min(0.9, base_confidence))

    def save_analysis_to_database(self, analysis_data: Dict, race_id: str):
        """Save GPT analysis to database for future reference"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gpt_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    analysis_type TEXT,
                    analysis_data TEXT,
                    confidence_score REAL,
                    tokens_used INTEGER,
                    timestamp TEXT,
                    model_used TEXT
                )
            """
            )

            # Insert analysis
            cursor.execute(
                """
                INSERT INTO gpt_analysis 
                (race_id, analysis_type, analysis_data, confidence_score, tokens_used, timestamp, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_id,
                    analysis_data.get("analysis_type", "general"),
                    json.dumps(analysis_data),
                    analysis_data.get("analysis_confidence", 0.5),
                    analysis_data.get("tokens_used", 0),
                    analysis_data.get("timestamp"),
                    analysis_data.get("model_used", self.model),
                ),
            )

            conn.commit()
            conn.close()

            logger.info(f"Saved GPT analysis for race {race_id}")

        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        analyzer = OpenAIEnhancedAnalyzer()

        # Example race context
        race_context = RaceContext(
            venue="SANDOWN",
            race_number=5,
            race_date="2025-07-26",
            distance="515m",
            grade="Grade 5",
            track_condition="Good",
            weather={"temperature": 18, "condition": "Fine", "wind": "Light"},
        )

        # Example dogs (would normally come from database)
        dogs = [
            DogProfile(
                name="FAST_TRACKER",
                box_number=1,
                recent_form=[2, 1, 3, 1, 2],
                win_rate=25.0,
                place_rate=65.0,
                best_time=29.85,
                trainer="J. Smith",
                weight=32.5,
                historical_stats={"races": 20, "wins": 5, "places": 13},
            )
        ]

        # Run analysis
        print("Running GPT analysis...")
        analysis = analyzer.analyze_race_with_gpt(race_context, dogs)
        print(json.dumps(analysis, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")
