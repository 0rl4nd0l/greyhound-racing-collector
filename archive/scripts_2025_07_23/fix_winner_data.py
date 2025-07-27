#!/usr/bin/env python3
"""
Fix Missing Race Winner Data
============================

This script fixes the missing race winner data by:
1. Extracting finish positions from dog names (e.g., "1. Miss Tumbleweed" -> position 1)
2. Cleaning dog names by removing position prefixes
3. Updating the finish_position field in dog_race_data
4. Updating the winner_name field in race_metadata for 1st place finishers

Author: AI Assistant
Date: July 14, 2025
"""

import sqlite3
import re
from datetime import datetime

class WinnerDataFixer:
    def __init__(self, db_path="./databases/comprehensive_greyhound_data.db"):
        self.db_path = db_path
        print("🔧 Winner Data Fixer initialized")
        print(f"📁 Database: {db_path}")
    
    def extract_position_from_name(self, dog_name):
        """Extract position number from dog name like '1. Miss Tumbleweed'"""
        if not dog_name:
            return None, dog_name
        
        # Pattern to match position number at start of name
        pattern = r'^(\d+)\.\s*(.+)$'
        match = re.match(pattern, dog_name.strip())
        
        if match:
            position = int(match.group(1))
            clean_name = match.group(2).strip()
            return position, clean_name
        
        return None, dog_name
    
    def fix_dog_data(self):
        """Fix dog names and extract finish positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all dog records
        cursor.execute("SELECT id, dog_name, finish_position FROM dog_race_data")
        dogs = cursor.fetchall()
        
        print(f"🐕 Processing {len(dogs)} dog records...")
        
        updated_count = 0
        position_extracted = 0
        
        for dog_id, dog_name, current_position in dogs:
            position, clean_name = self.extract_position_from_name(dog_name)
            
            # Update if we extracted a position or cleaned the name
            if position is not None or clean_name != dog_name:
                cursor.execute("""
                    UPDATE dog_race_data 
                    SET dog_name = ?, dog_clean_name = ?, finish_position = ?
                    WHERE id = ?
                """, (clean_name, clean_name, position, dog_id))
                
                updated_count += 1
                if position is not None:
                    position_extracted += 1
                    print(f"   ✅ {clean_name}: Position {position}")
        
        conn.commit()
        print(f"🔄 Updated {updated_count} dog records")
        print(f"📊 Extracted positions for {position_extracted} dogs")
        
        return conn, cursor
    
    def update_race_winners(self, conn, cursor):
        """Update race_metadata with winner information"""
        print("🏆 Updating race winners...")
        
        # Get all races with their winners (position 1)
        cursor.execute("""
            SELECT rm.race_id, rm.venue, rm.race_date, 
                   drd.dog_name, drd.starting_price, drd.individual_time
            FROM race_metadata rm
            LEFT JOIN dog_race_data drd ON rm.race_id = drd.race_id
            WHERE drd.finish_position = 1
        """)
        
        winners = cursor.fetchall()
        
        print(f"🏁 Found {len(winners)} race winners")
        
        updated_races = 0
        for race_id, venue, race_date, winner_name, winner_odds, winner_time in winners:
            cursor.execute("""
                UPDATE race_metadata
                SET winner_name = ?, winner_odds = ?, race_time = ?
                WHERE race_id = ?
            """, (winner_name, winner_odds, winner_time, race_id))
            
            updated_races += 1
            print(f"   🏆 {race_id}: {winner_name} (${winner_odds})")
        
        conn.commit()
        print(f"✅ Updated {updated_races} race winners")
    
    def update_race_margins(self, conn, cursor):
        """Calculate and update winner margins"""
        print("📏 Calculating winner margins...")
        
        # Get races with winners and runners-up
        cursor.execute("""
            SELECT rm.race_id, 
                   w.dog_name as winner_name,
                   r.dog_name as runner_up_name,
                   w.individual_time as winner_time,
                   r.individual_time as runner_up_time,
                   r.margin
            FROM race_metadata rm
            JOIN dog_race_data w ON rm.race_id = w.race_id AND w.finish_position = 1
            JOIN dog_race_data r ON rm.race_id = r.race_id AND r.finish_position = 2
            WHERE w.individual_time IS NOT NULL AND r.individual_time IS NOT NULL
        """)
        
        margin_data = cursor.fetchall()
        
        updated_margins = 0
        for race_id, winner_name, runner_up_name, winner_time, runner_up_time, margin in margin_data:
            # Use existing margin if available, otherwise calculate from times
            winner_margin = None
            
            if margin and margin.strip():
                # Try to extract numeric margin
                try:
                    margin_match = re.search(r'(\d+\.?\d*)', margin)
                    if margin_match:
                        winner_margin = float(margin_match.group(1))
                except:
                    pass
            
            # If no margin found, try to calculate from times
            if winner_margin is None and winner_time and runner_up_time:
                try:
                    # Parse times (assuming format like "30.12" for seconds)
                    w_time = float(winner_time)
                    r_time = float(runner_up_time)
                    time_diff = r_time - w_time
                    if time_diff > 0:
                        winner_margin = time_diff
                except:
                    pass
            
            if winner_margin is not None:
                cursor.execute("""
                    UPDATE race_metadata
                    SET winner_margin = ?
                    WHERE race_id = ?
                """, (winner_margin, race_id))
                
                updated_margins += 1
                print(f"   📏 {race_id}: {winner_name} won by {winner_margin}")
        
        conn.commit()
        print(f"✅ Updated {updated_margins} winner margins")
    
    def show_results(self, conn, cursor):
        """Show the results of the fix"""
        print("\n📊 RESULTS SUMMARY")
        print("=" * 50)
        
        # Count races with winners
        cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != ''")
        races_with_winners = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM race_metadata")
        total_races = cursor.fetchone()[0]
        
        print(f"🏆 Races with winners: {races_with_winners}/{total_races}")
        
        # Count dogs with positions
        cursor.execute("SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NOT NULL")
        dogs_with_positions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dog_race_data")
        total_dogs = cursor.fetchone()[0]
        
        print(f"📊 Dogs with positions: {dogs_with_positions}/{total_dogs}")
        
        # Show recent winners
        print("\n🏁 Recent Winners:")
        cursor.execute("""
            SELECT rm.race_date, rm.venue, rm.race_number, rm.winner_name, rm.winner_odds
            FROM race_metadata rm
            WHERE rm.winner_name IS NOT NULL AND rm.winner_name != ''
            ORDER BY rm.race_date DESC
            LIMIT 10
        """)
        
        winners = cursor.fetchall()
        for race_date, venue, race_number, winner_name, winner_odds in winners:
            odds_str = f"${winner_odds}" if winner_odds else "N/A"
            print(f"   🏆 {race_date} {venue} R{race_number}: {winner_name} ({odds_str})")
    
    def run_fix(self):
        """Run the complete fix process"""
        print("🚀 STARTING WINNER DATA FIX")
        print("=" * 50)
        
        try:
            # Step 1: Fix dog names and extract positions
            conn, cursor = self.fix_dog_data()
            
            # Step 2: Update race winners
            self.update_race_winners(conn, cursor)
            
            # Step 3: Update winner margins
            self.update_race_margins(conn, cursor)
            
            # Step 4: Show results
            self.show_results(conn, cursor)
            
            conn.close()
            print("\n✅ WINNER DATA FIX COMPLETE!")
            
        except Exception as e:
            print(f"❌ Error during fix: {e}")
            raise

def main():
    fixer = WinnerDataFixer()
    fixer.run_fix()

if __name__ == "__main__":
    main()
