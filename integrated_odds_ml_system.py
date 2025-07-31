#!/usr/bin/env python3
"""
Integrated ML + Odds System for Greyhound Racing
===============================================

This system combines your ML predictions with live odds data to identify value betting opportunities.
It provides automated monitoring, alerts, and comprehensive analysis.

Features:
- Real-time odds integration with ML predictions
- Value betting opportunity detection
- Automated monitoring and alerts
- Historical tracking and analysis
- Multi-sportsbook support

Author: Orlando Lee
Date: July 27, 2025
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import schedule

# Import our scraping components
from event_scraper import EventScraper
from hybrid_odds_scraper import HybridOddsScraper
from sportsbook_factory import SportsbookFactory


class IntegratedOddsMLSystem:
    """Complete integrated system for ML predictions + live odds"""

    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.scraper = HybridOddsScraper(use_headless=True, timeout=30)
        self.predictions_cache = {}
        self.odds_cache = {}
        self.value_threshold = 0.05  # 5% edge minimum
        self.monitoring_active = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("integrated_system.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        self.setup_database()

    def setup_database(self):
        """Setup all necessary database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced live odds table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS live_odds_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_time TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                implied_probability REAL,
                market_type TEXT DEFAULT 'win',
                sportsbook TEXT,
                url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE
            )
        """
        )

        # Value betting opportunities
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS value_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                predicted_probability REAL,
                market_odds REAL,
                implied_probability REAL,
                value_percentage REAL,
                confidence_score REAL,
                kelly_fraction REAL,
                recommended_stake REAL,
                sportsbook TEXT,
                prediction_model TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """
        )

        # Odds movement tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                previous_odds REAL,
                current_odds REAL,
                odds_change_pct REAL,
                volume_indicator TEXT,
                movement_direction TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Performance tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                races_monitored INTEGER,
                value_bets_identified INTEGER,
                successful_predictions INTEGER,
                total_profit_loss REAL,
                roi_percentage REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()
        self.logger.info("Database setup completed")

    def load_ml_predictions(self, race_id: str = None) -> pd.DataFrame:
        """Load ML predictions from your existing system"""
        try:
            # Try to load from predictions directory
            predictions_dir = Path("predictions")
            if race_id:
                # Look for specific race prediction
                prediction_files = list(predictions_dir.glob(f"*{race_id}*.json"))
            else:
                # Load all recent predictions
                prediction_files = list(predictions_dir.glob("*.json"))
                # Sort by modification time, get most recent
                prediction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                prediction_files = prediction_files[:10]  # Last 10 predictions

            predictions_data = []

            for file_path in prediction_files:
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Extract predictions
                    if "predictions" in data:
                        for pred in data["predictions"]:
                            predictions_data.append(
                                {
                                    "race_id": data.get("race_id", file_path.stem),
                                    "venue": data.get("venue", "Unknown"),
                                    "race_number": data.get("race_number", 1),
                                    "dog_name": pred.get("dog_name", ""),
                                    "dog_clean_name": self.clean_dog_name(
                                        pred.get("dog_name", "")
                                    ),
                                    "predicted_probability": pred.get(
                                        "probability", pred.get("confidence", 0)
                                    ),
                                    "confidence_score": pred.get("confidence", 0),
                                    "model_features": pred.get("features", {}),
                                    "prediction_timestamp": data.get(
                                        "timestamp", datetime.now().isoformat()
                                    ),
                                }
                            )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load prediction file {file_path}: {e}"
                    )

            if predictions_data:
                df = pd.DataFrame(predictions_data)
                self.logger.info(f"Loaded {len(df)} ML predictions")
                return df
            else:
                self.logger.warning("No prediction data found")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error loading ML predictions: {e}")
            return pd.DataFrame()

    def get_live_odds(self, race_urls: List[str]) -> pd.DataFrame:
        """Get live odds from multiple race URLs"""
        all_odds_data = []

        for url in race_urls:
            try:
                self.logger.info(f"Scraping odds from: {url}")
                odds_df, metadata = self.scraper.scrape_odds(url)

                if metadata["success"] and odds_df is not None:
                    # Add metadata to odds data
                    odds_df["sportsbook"] = self.get_sportsbook_name(url)
                    odds_df["url"] = url
                    odds_df["scrape_timestamp"] = datetime.now()
                    odds_df["dog_clean_name"] = odds_df["selection_name"].apply(
                        self.clean_dog_name
                    )
                    odds_df["implied_probability"] = 1 / odds_df["odds"]

                    all_odds_data.append(odds_df)
                    self.logger.info(f"Successfully scraped {len(odds_df)} selections")
                else:
                    self.logger.warning(
                        f"Failed to scrape odds from {url}: {metadata.get('error_message', 'Unknown error')}"
                    )

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")

        if all_odds_data:
            combined_df = pd.concat(all_odds_data, ignore_index=True)
            self.store_live_odds(combined_df)
            return combined_df
        else:
            return pd.DataFrame()

    def identify_value_opportunities(
        self, predictions_df: pd.DataFrame, odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Identify value betting opportunities by comparing predictions with odds"""
        if predictions_df.empty or odds_df.empty:
            return pd.DataFrame()

        # Merge predictions with odds
        merged_df = pd.merge(
            predictions_df, odds_df, on=["dog_clean_name"], how="inner"
        )

        if merged_df.empty:
            self.logger.warning("No matches found between predictions and odds")
            return pd.DataFrame()

        # Calculate value
        merged_df["value_percentage"] = (
            merged_df["predicted_probability"] - merged_df["implied_probability"]
        ) * 100
        merged_df["kelly_fraction"] = np.maximum(
            (merged_df["predicted_probability"] * merged_df["odds"] - 1)
            / (merged_df["odds"] - 1),
            0,
        )
        merged_df["recommended_stake"] = np.minimum(
            merged_df["kelly_fraction"] * 0.25, 0.05
        )  # Max 5% stake

        # Filter for value opportunities
        value_bets = merged_df[
            (merged_df["value_percentage"] > self.value_threshold * 100)
            & (merged_df["confidence_score"] > 0.6)
            & (merged_df["predicted_probability"] > merged_df["implied_probability"])
        ].copy()

        if not value_bets.empty:
            # Sort by value percentage
            value_bets = value_bets.sort_values("value_percentage", ascending=False)
            self.store_value_opportunities(value_bets)
            self.logger.info(
                f"Identified {len(value_bets)} value betting opportunities"
            )

        return value_bets

    def store_live_odds(self, odds_df: pd.DataFrame):
        """Store live odds in database"""
        conn = sqlite3.connect(self.db_path)

        # Mark previous odds as not current
        cursor = conn.cursor()
        cursor.execute("UPDATE live_odds_enhanced SET is_current = FALSE")

        # Insert new odds
        for _, row in odds_df.iterrows():
            cursor.execute(
                """
                INSERT INTO live_odds_enhanced 
                (race_id, venue, dog_name, dog_clean_name, odds_decimal, implied_probability, 
                 market_type, sportsbook, url, timestamp, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row.get("race_id", "unknown"),
                    row.get("venue", "unknown"),
                    row.get("selection_name", ""),
                    row.get("dog_clean_name", ""),
                    row.get("odds", 0),
                    row.get("implied_probability", 0),
                    row.get("market_name", "win"),
                    row.get("sportsbook", "unknown"),
                    row.get("url", ""),
                    datetime.now(),
                    True,
                ),
            )

        conn.commit()
        conn.close()

    def store_value_opportunities(self, value_bets_df: pd.DataFrame):
        """Store value betting opportunities in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for _, row in value_bets_df.iterrows():
            cursor.execute(
                """
                INSERT INTO value_opportunities 
                (race_id, dog_clean_name, predicted_probability, market_odds, implied_probability,
                 value_percentage, confidence_score, kelly_fraction, recommended_stake, sportsbook, prediction_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row.get("race_id", "unknown"),
                    row.get("dog_clean_name", ""),
                    row.get("predicted_probability", 0),
                    row.get("odds", 0),
                    row.get("implied_probability", 0),
                    row.get("value_percentage", 0),
                    row.get("confidence_score", 0),
                    row.get("kelly_fraction", 0),
                    row.get("recommended_stake", 0),
                    row.get("sportsbook", "unknown"),
                    "ml_model",
                ),
            )

        conn.commit()
        conn.close()

    def run_complete_analysis(self, race_urls: List[str]) -> Dict:
        """Run complete analysis: load predictions, get odds, find value"""
        results = {
            "timestamp": datetime.now(),
            "races_analyzed": 0,
            "predictions_loaded": 0,
            "odds_scraped": 0,
            "value_opportunities": 0,
            "top_bets": [],
            "errors": [],
        }

        try:
            # Load ML predictions
            predictions_df = self.load_ml_predictions()
            results["predictions_loaded"] = len(predictions_df)

            if predictions_df.empty:
                results["errors"].append("No ML predictions available")
                return results

            # Get live odds
            odds_df = self.get_live_odds(race_urls)
            results["odds_scraped"] = len(odds_df)
            results["races_analyzed"] = len(race_urls)

            if odds_df.empty:
                results["errors"].append("No live odds available")
                return results

            # Find value opportunities
            value_bets = self.identify_value_opportunities(predictions_df, odds_df)
            results["value_opportunities"] = len(value_bets)

            if not value_bets.empty:
                # Get top 5 value bets
                top_bets = value_bets.head(5).to_dict("records")
                results["top_bets"] = top_bets

                # Send alerts for high-value opportunities
                self.send_value_alerts(value_bets)

            self.logger.info(
                f"Analysis complete: {results['value_opportunities']} opportunities found"
            )

        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            results["errors"].append(str(e))

        return results

    def send_value_alerts(self, value_bets_df: pd.DataFrame):
        """Send alerts for high-value betting opportunities"""
        high_value_bets = value_bets_df[
            value_bets_df["value_percentage"] > 10
        ]  # >10% edge

        if not high_value_bets.empty:
            self.logger.info("🚨 HIGH VALUE ALERTS 🚨")
            for _, bet in high_value_bets.iterrows():
                alert_msg = (
                    f"🎯 VALUE BET ALERT\n"
                    f"Dog: {bet['dog_name']}\n"
                    f"Venue: {bet.get('venue', 'Unknown')}\n"
                    f"Predicted Prob: {bet['predicted_probability']:.3f}\n"
                    f"Market Odds: {bet['odds']:.2f}\n"
                    f"Value: {bet['value_percentage']:.1f}%\n"
                    f"Recommended Stake: {bet['recommended_stake']:.1f}%\n"
                    f"Confidence: {bet['confidence_score']:.3f}"
                )
                self.logger.info(alert_msg)

                # Here you could add email, Slack, or other notification methods
                # self.send_email_alert(alert_msg)
                # self.send_slack_alert(alert_msg)

    def start_automated_monitoring(
        self, race_urls: List[str], interval_minutes: int = 15
    ):
        """Start automated monitoring of races"""
        self.monitoring_active = True

        def monitor_job():
            if self.monitoring_active:
                self.logger.info("Running automated monitoring cycle...")
                results = self.run_complete_analysis(race_urls)

                # Log summary
                summary = (
                    f"Monitoring Summary: "
                    f"{results['races_analyzed']} races, "
                    f"{results['value_opportunities']} opportunities"
                )
                self.logger.info(summary)

        # Schedule the monitoring
        schedule.every(interval_minutes).minutes.do(monitor_job)

        self.logger.info(
            f"Started automated monitoring every {interval_minutes} minutes"
        )

        # Run monitoring loop in separate thread
        def run_schedule():
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(10)

        monitor_thread = threading.Thread(target=run_schedule, daemon=True)
        monitor_thread.start()

        return monitor_thread

    def stop_automated_monitoring(self):
        """Stop automated monitoring"""
        self.monitoring_active = False
        schedule.clear()
        self.logger.info("Stopped automated monitoring")

    def get_performance_report(self, days: int = 7) -> Dict:
        """Generate performance report"""
        conn = sqlite3.connect(self.db_path)

        # Get value opportunities from last N days
        query = """
            SELECT * FROM value_opportunities 
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(
            days
        )

        opportunities_df = pd.read_sql_query(query, conn)

        # Get system performance metrics
        report = {
            "period_days": days,
            "total_opportunities": len(opportunities_df),
            "avg_value_percentage": (
                opportunities_df["value_percentage"].mean()
                if not opportunities_df.empty
                else 0
            ),
            "high_value_bets": len(
                opportunities_df[opportunities_df["value_percentage"] > 10]
            ),
            "sportsbooks_monitored": (
                opportunities_df["sportsbook"].nunique()
                if not opportunities_df.empty
                else 0
            ),
            "top_dogs": (
                opportunities_df["dog_clean_name"].value_counts().head(5).to_dict()
                if not opportunities_df.empty
                else {}
            ),
        }

        conn.close()
        return report

    @staticmethod
    def clean_dog_name(name: str) -> str:
        """Clean dog name for matching"""
        if not name:
            return ""
        # Remove common prefixes/suffixes and normalize
        cleaned = name.strip().upper()
        cleaned = cleaned.replace("'", "").replace("-", " ")
        return " ".join(cleaned.split())

    @staticmethod
    def get_sportsbook_name(url: str) -> str:
        """Extract sportsbook name from URL"""
        if "sportsbet.com.au" in url:
            return "Sportsbet"
        elif "tab.com.au" in url:
            return "TAB"
        elif "ladbrokes.com.au" in url:
            return "Ladbrokes"
        else:
            return "Unknown"


def demo_integrated_system():
    """Demonstrate the integrated system"""
    print("🎯 Integrated ML + Odds System Demo")
    print("=" * 50)

    # Initialize system
    system = IntegratedOddsMLSystem()

    # Test URLs - replace with current live races
    test_urls = [
        "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/sale/race-1-9443604",
        # Add more current URLs here
    ]

    # Run analysis
    print("\n📊 Running complete analysis...")
    results = system.run_complete_analysis(test_urls)

    # Display results
    print(f"\n✅ Analysis Results:")
    print(f"   Races analyzed: {results['races_analyzed']}")
    print(f"   Predictions loaded: {results['predictions_loaded']}")
    print(f"   Odds scraped: {results['odds_scraped']}")
    print(f"   Value opportunities: {results['value_opportunities']}")

    if results["top_bets"]:
        print(f"\n🎯 Top Value Bets:")
        for i, bet in enumerate(results["top_bets"][:3], 1):
            print(
                f"   {i}. {bet.get('dog_name', 'Unknown')}: {bet.get('value_percentage', 0):.1f}% value"
            )

    if results["errors"]:
        print(f"\n⚠️  Errors:")
        for error in results["errors"]:
            print(f"   - {error}")

    # Show performance report
    print(f"\n📈 Performance Report (Last 7 days):")
    report = system.get_performance_report()
    for key, value in report.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_integrated_system()
