#!/usr/bin/env python3
"""
Live Odds Dashboard
==================

Real-time dashboard displaying live odds from Sportsbet integrated with race predictions.
This creates a web interface to view current odds and value betting opportunities.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import threading
from sportsbet_odds_integrator import SportsbetOddsIntegrator

class LiveOddsDashboard:
    def __init__(self, db_path="greyhound_racing_data.db"):
        self.db_path = db_path
        self.integrator = None
        
    def get_current_odds(self) -> pd.DataFrame:
        """Get current live odds from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                race_id,
                venue,
                race_number,
                race_date,
                race_time,
                dog_name,
                dog_clean_name,
                box_number,
                odds_decimal,
                odds_fractional,
                timestamp
            FROM live_odds 
            WHERE is_current = TRUE 
            ORDER BY venue, race_number, box_number
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_value_bets(self) -> pd.DataFrame:
        """Get current value betting opportunities"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                race_id,
                dog_clean_name,
                predicted_probability,
                market_odds,
                implied_probability,
                value_percentage,
                confidence_level,
                bet_recommendation,
                timestamp
            FROM value_bets 
            WHERE timestamp > datetime('now', '-2 hours')
            ORDER BY value_percentage DESC
            LIMIT 20
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_odds_movements(self) -> pd.DataFrame:
        """Get recent odds movements"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                race_id,
                dog_clean_name,
                odds_decimal,
                odds_change,
                timestamp
            FROM odds_history 
            WHERE timestamp > datetime('now', '-1 hour')
            AND ABS(odds_change) > 0.5
            ORDER BY timestamp DESC
            LIMIT 50
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_race_summary(self) -> pd.DataFrame:
        """Get summary of current races with odds"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                venue,
                race_number,
                race_time,
                COUNT(*) as dog_count,
                MIN(odds_decimal) as favorite_odds,
                MAX(odds_decimal) as longest_odds,
                AVG(odds_decimal) as avg_odds,
                MAX(timestamp) as last_updated
            FROM live_odds 
            WHERE is_current = TRUE 
            GROUP BY race_id
            ORDER BY venue, race_number
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

def run_streamlit_dashboard():
    """Main Streamlit dashboard"""
    st.set_page_config(
        page_title="🐕 Live Greyhound Odds Dashboard",
        page_icon="🐕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    dashboard = LiveOddsDashboard()
    
    # Header
    st.title("🐕 Live Greyhound Racing Odds Dashboard")
    st.markdown("Real-time odds from Sportsbet with value betting opportunities")
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    # Get data
    current_odds = dashboard.get_current_odds()
    value_bets = dashboard.get_value_bets()
    race_summary = dashboard.get_race_summary()
    odds_movements = dashboard.get_odds_movements()
    
    # Summary metrics
    with col1:
        st.metric("🏁 Active Races", len(race_summary))
    
    with col2:
        st.metric("🐕 Dogs with Odds", len(current_odds))
    
    with col3:
        st.metric("💰 Value Opportunities", len(value_bets))
    
    # Race Summary Table
    st.subheader("📊 Current Races Summary")
    if not race_summary.empty:
        race_summary['last_updated'] = pd.to_datetime(race_summary['last_updated'])
        race_summary['minutes_ago'] = (datetime.now() - race_summary['last_updated']).dt.total_seconds() / 60
        
        st.dataframe(
            race_summary[['venue', 'race_number', 'race_time', 'dog_count', 'favorite_odds', 'longest_odds', 'minutes_ago']],
            column_config={
                "venue": "Venue",
                "race_number": "Race #",
                "race_time": "Time",
                "dog_count": "Dogs",
                "favorite_odds": st.column_config.NumberColumn("Favorite", format="$%.2f"),
                "longest_odds": st.column_config.NumberColumn("Longest", format="$%.2f"),
                "minutes_ago": st.column_config.NumberColumn("Updated (min ago)", format="%.1f")
            },
            use_container_width=True
        )
    else:
        st.info("No current race data available. Start the odds collector to begin gathering data.")
    
    # Two columns for detailed views
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Top Value Bets")
        if not value_bets.empty:
            for _, bet in value_bets.head(10).iterrows():
                with st.expander(f"🎯 {bet['dog_clean_name']} - {bet['value_percentage']:.1f}% Value"):
                    st.write(f"**Recommendation:** {bet['bet_recommendation']}")
                    st.write(f"**Market Odds:** ${bet['market_odds']:.2f}")
                    st.write(f"**Predicted Probability:** {bet['predicted_probability']:.1%}")
                    st.write(f"**Confidence:** {bet['confidence_level']}")
        else:
            st.info("No value bets available. Predictions needed to calculate value.")
    
    with col2:
        st.subheader("📈 Recent Odds Movements")
        if not odds_movements.empty:
            for _, move in odds_movements.head(10).iterrows():
                direction = "📈" if move['odds_change'] > 0 else "📉"
                change_text = f"{direction} ${abs(move['odds_change']):.2f}"
                with st.expander(f"{move['dog_clean_name']} - {change_text}"):
                    st.write(f"**Current Odds:** ${move['odds_decimal']:.2f}")
                    st.write(f"**Change:** {move['odds_change']:+.2f}")
                    st.write(f"**Time:** {move['timestamp']}")
        else:
            st.info("No significant odds movements detected.")
    
    # Detailed odds by race
    st.subheader("🐕 Detailed Odds by Race")
    if not current_odds.empty:
        # Group by race
        for race_id in current_odds['race_id'].unique()[:5]:  # Show first 5 races
            race_data = current_odds[current_odds['race_id'] == race_id]
            venue = race_data['venue'].iloc[0]
            race_num = race_data['race_number'].iloc[0]
            race_time = race_data['race_time'].iloc[0]
            
            with st.expander(f"🏁 {venue} Race {race_num} ({race_time})"):
                # Create odds chart
                fig = px.bar(
                    race_data,
                    x='dog_name',
                    y='odds_decimal',
                    title=f"{venue} Race {race_num} - Current Odds",
                    labels={'odds_decimal': 'Odds ($)', 'dog_name': 'Dog'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Odds table
                display_data = race_data[['box_number', 'dog_name', 'odds_decimal', 'odds_fractional']].copy()
                display_data.columns = ['Box', 'Dog Name', 'Decimal Odds', 'Fractional Odds']
                st.dataframe(display_data, use_container_width=True)
    
    # Footer with status
    st.markdown("---")
    if not current_odds.empty:
        last_update = current_odds['timestamp'].max()
        st.caption(f"Last updated: {last_update}")
    else:
        st.caption("⚠️ No live data available. Please start the odds collection system.")

def start_data_collection():
    """Start continuous data collection in background"""
    integrator = SportsbetOddsIntegrator()
    try:
        print("🚀 Starting continuous odds monitoring...")
        integrator.start_continuous_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Stopping odds collection...")
        integrator.close_driver()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # Run data collection
        start_data_collection()
    else:
        # Run dashboard
        run_streamlit_dashboard()
