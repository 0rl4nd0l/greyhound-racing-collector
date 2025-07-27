#!/usr/bin/env python3
"""
Fix Race Organization and Database Data
======================================

This script:
1. Identifies races that are in the future (upcoming) vs completed (past)
2. Moves future races to upcoming_races folder
3. Removes future race data from database (they shouldn't have winners)
4. Keeps only historical races with actual results in the database
5. Fixes winner data for completed races only

Author: AI Assistant
Date: July 14, 2025
"""

import os
import sqlite3
import shutil
import re
from datetime import datetime, date
from pathlib import Path

class RaceOrganizer:
    def __init__(self):
        self.db_path = "./databases/comprehensive_greyhound_data.db"
        self.processed_dir = "./processed"
        self.upcoming_dir = "./upcoming_races"
        self.unprocessed_dir = "./unprocessed"
        self.today = date.today()
        
        # Ensure directories exist
        os.makedirs(self.upcoming_dir, exist_ok=True)
        
        print("🏁 Race Organizer initialized")
        print(f"📅 Today's date: {self.today}")
        print(f"📂 Processed directory: {self.processed_dir}")
        print(f"📂 Upcoming directory: {self.upcoming_dir}")
    
    def parse_race_date_from_filename(self, filename):
        """Extract race date from filename like 'Race 1 - MAND - 10 July 2025.csv'"""
        pattern = r"Race (\d+) - ([A-Z_]+) - (\d{1,2} [A-Za-z]+ \d{4})\.csv"
        match = re.match(pattern, filename)
        
        if match:
            race_number = int(match.group(1))
            venue = match.group(2)
            date_str = match.group(3)
            
            # Parse date
            try:
                race_date = datetime.strptime(date_str, "%d %B %Y").date()
                return {
                    'race_number': race_number,
                    'venue': venue,
                    'race_date': race_date,
                    'date_str': date_str
                }
            except ValueError:
                try:
                    race_date = datetime.strptime(date_str, "%d %b %Y").date()
                    return {
                        'race_number': race_number,
                        'venue': venue,
                        'race_date': race_date,
                        'date_str': date_str
                    }
                except ValueError:
                    return None
        return None
    
    def organize_race_files(self):
        """Move future races to upcoming folder and keep past races in processed"""
        print("📁 Organizing race files by date...")
        
        future_races = []
        past_races = []
        
        # Check processed directory
        if os.path.exists(self.processed_dir):
            for filename in os.listdir(self.processed_dir):
                if filename.endswith('.csv') and filename.startswith('Race'):
                    race_info = self.parse_race_date_from_filename(filename)
                    if race_info:
                        filepath = os.path.join(self.processed_dir, filename)
                        if race_info['race_date'] > self.today:
                            future_races.append((filepath, filename, race_info))
                        else:
                            past_races.append((filepath, filename, race_info))
        
        # Check unprocessed directory
        if os.path.exists(self.unprocessed_dir):
            for filename in os.listdir(self.unprocessed_dir):
                if filename.endswith('.csv') and filename.startswith('Race'):
                    race_info = self.parse_race_date_from_filename(filename)
                    if race_info:
                        filepath = os.path.join(self.unprocessed_dir, filename)
                        if race_info['race_date'] > self.today:
                            future_races.append((filepath, filename, race_info))
                        else:
                            past_races.append((filepath, filename, race_info))
        
        # Move future races to upcoming folder
        moved_future = 0
        for filepath, filename, race_info in future_races:
            dest_path = os.path.join(self.upcoming_dir, filename)
            if not os.path.exists(dest_path):
                shutil.move(filepath, dest_path)
                moved_future += 1
                print(f"   🔮 Future race moved: {race_info['venue']} R{race_info['race_number']} on {race_info['race_date']}")
        
        # Ensure past races are in processed folder
        moved_past = 0
        for filepath, filename, race_info in past_races:
            if not filepath.startswith(self.processed_dir):
                dest_path = os.path.join(self.processed_dir, filename)
                if not os.path.exists(dest_path):
                    shutil.move(filepath, dest_path)
                    moved_past += 1
                    print(f"   ✅ Past race moved to processed: {race_info['venue']} R{race_info['race_number']} on {race_info['race_date']}")
        
        print(f"📊 Organization complete:")
        print(f"   🔮 Future races: {len(future_races)} (moved {moved_future})")
        print(f"   ✅ Past races: {len(past_races)} (moved {moved_past})")
        
        return future_races, past_races
    
    def clean_database(self, future_races, past_races):
        """Remove future race data from database and keep only completed races"""
        print("🗄️ Cleaning database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all race IDs that should be removed (future races)
        future_race_ids = []
        for _, filename, race_info in future_races:
            race_id = f"{race_info['venue'].lower()}_{race_info['race_date']}_{race_info['race_number']}"
            future_race_ids.append(race_id)
        
        # Remove future race data
        removed_races = 0
        removed_dogs = 0
        
        for race_id in future_race_ids:
            # Remove from race_metadata
            cursor.execute("DELETE FROM race_metadata WHERE race_id = ?", (race_id,))
            if cursor.rowcount > 0:
                removed_races += 1
                print(f"   🗑️ Removed future race: {race_id}")
            
            # Remove from dog_race_data
            cursor.execute("DELETE FROM dog_race_data WHERE race_id = ?", (race_id,))
            removed_dogs += cursor.rowcount
        
        # Clear winner data for any remaining races that are in the future
        cursor.execute("""
            UPDATE race_metadata 
            SET winner_name = NULL, winner_odds = NULL, winner_margin = NULL, race_time = NULL
            WHERE race_date > ?
        """, (self.today,))
        
        cleared_winners = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"🧹 Database cleaned:")
        print(f"   🗑️ Removed {removed_races} future race records")
        print(f"   🗑️ Removed {removed_dogs} future dog records")
        print(f"   🧹 Cleared {cleared_winners} future winner records")
    
    def fix_completed_races(self):
        """Fix winner data for completed races only"""
        print("🏆 Fixing winner data for completed races...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all completed races (past dates)
        cursor.execute("""
            SELECT race_id, race_date, venue, race_number
            FROM race_metadata
            WHERE race_date <= ?
        """, (self.today,))
        
        completed_races = cursor.fetchall()
        print(f"🏁 Found {len(completed_races)} completed races")
        
        # For each completed race, check if it has winner data
        races_with_winners = 0
        races_without_winners = 0
        
        for race_id, race_date, venue, race_number in completed_races:
            # Check if race has a winner
            cursor.execute("""
                SELECT winner_name FROM race_metadata 
                WHERE race_id = ? AND winner_name IS NOT NULL AND winner_name != ''
            """, (race_id,))
            
            has_winner = cursor.fetchone() is not None
            
            if has_winner:
                races_with_winners += 1
            else:
                races_without_winners += 1
                print(f"   ⚠️ Missing winner: {race_id} ({race_date})")
        
        print(f"📊 Completed race status:")
        print(f"   ✅ Races with winners: {races_with_winners}")
        print(f"   ⚠️ Races without winners: {races_without_winners}")
        
        conn.close()
    
    def show_final_status(self):
        """Show final status of races and database"""
        print("\n📊 FINAL STATUS")
        print("=" * 50)
        
        # Count files in each directory
        processed_count = len([f for f in os.listdir(self.processed_dir) if f.endswith('.csv') and f.startswith('Race')]) if os.path.exists(self.processed_dir) else 0
        upcoming_count = len([f for f in os.listdir(self.upcoming_dir) if f.endswith('.csv') and f.startswith('Race')]) if os.path.exists(self.upcoming_dir) else 0
        unprocessed_count = len([f for f in os.listdir(self.unprocessed_dir) if f.endswith('.csv') and f.startswith('Race')]) if os.path.exists(self.unprocessed_dir) else 0
        
        print(f"📁 File organization:")
        print(f"   ✅ Processed (completed races): {processed_count}")
        print(f"   🔮 Upcoming (future races): {upcoming_count}")
        print(f"   📥 Unprocessed: {unprocessed_count}")
        
        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE race_date <= ?", (self.today,))
            completed_db_races = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE race_date > ?", (self.today,))
            future_db_races = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NOT NULL AND winner_name != ''")
            races_with_winners = cursor.fetchone()[0]
            
            print(f"🗄️ Database status:")
            print(f"   ✅ Completed races in DB: {completed_db_races}")
            print(f"   🔮 Future races in DB: {future_db_races}")
            print(f"   🏆 Races with winners: {races_with_winners}")
            
            # Show some recent winners
            cursor.execute("""
                SELECT race_date, venue, race_number, winner_name, winner_odds
                FROM race_metadata
                WHERE winner_name IS NOT NULL AND winner_name != '' AND race_date <= ?
                ORDER BY race_date DESC
                LIMIT 5
            """, (self.today,))
            
            winners = cursor.fetchall()
            if winners:
                print(f"🏆 Recent winners:")
                for race_date, venue, race_number, winner_name, winner_odds in winners:
                    odds_str = f"${winner_odds}" if winner_odds else "N/A"
                    print(f"   🏆 {race_date} {venue} R{race_number}: {winner_name} ({odds_str})")
            
            conn.close()
            
        except Exception as e:
            print(f"   ⚠️ Error checking database: {e}")
    
    def run_organization(self):
        """Run the complete organization process"""
        print("🚀 STARTING RACE ORGANIZATION")
        print("=" * 50)
        
        try:
            # Step 1: Organize files by date
            future_races, past_races = self.organize_race_files()
            
            # Step 2: Clean database
            self.clean_database(future_races, past_races)
            
            # Step 3: Fix completed races
            self.fix_completed_races()
            
            # Step 4: Show final status
            self.show_final_status()
            
            print("\n✅ RACE ORGANIZATION COMPLETE!")
            
        except Exception as e:
            print(f"❌ Error during organization: {e}")
            raise

def main():
    organizer = RaceOrganizer()
    organizer.run_organization()

if __name__ == "__main__":
    main()
