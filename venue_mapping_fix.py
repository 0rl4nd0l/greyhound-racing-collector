#!/usr/bin/env python3
"""
Comprehensive Venue Mapping System
==================================

This module provides a comprehensive mapping between different venue naming conventions
used in Australian and New Zealand greyhound racing to fix historical data lookup issues.

Based on official venue codes from:
- The Greyhound Recorder
- Australian Racing Greyhound
- Punters.com.au
- Various racing authorities

Author: AI Assistant
Date: July 25, 2025
"""

import sqlite3
import json
from typing import Dict, List, Optional, Set

class GreyhoundVenueMapper:
    def __init__(self):
        """Initialize the comprehensive venue mapping system"""
        
        # Comprehensive venue mapping including all variations
        self.venue_mappings = {
            # New South Wales (NSW)
            'DAPT': {
                'official_name': 'Dapto',
                'codes': ['DAPT', 'dapt'],
                'track_codes': ['DAPT'],
                'location': 'NSW',
                'active': True
            },
            'GOSF': {
                'official_name': 'Gosford',
                'codes': ['GOSF', 'gosf'],
                'track_codes': ['GOSF'],
                'location': 'NSW',
                'active': True
            },
            'GOULBURN': {
                'official_name': 'Goulburn',
                'codes': ['GOULBURN', 'goulburn'],
                'track_codes': ['GOULBURN'],
                'location': 'NSW',
                'active': True
            },
            'GUNN': {
                'official_name': 'Gunnedah',
                'codes': ['GUNN', 'gunn', 'GUNNEDAH'],
                'track_codes': ['GUNN'],
                'location': 'NSW',
                'active': True
            },
            'RICH': {
                'official_name': 'Richmond',
                'codes': ['RICH', 'rich', 'RICHMOND'],
                'track_codes': ['RICH'],
                'location': 'NSW',
                'active': True
            },
            'TEMA': {
                'official_name': 'Temora',
                'codes': ['TEMA', 'tema', 'TEMORA'],
                'track_codes': ['TEMA'],
                'location': 'NSW',
                'active': True
            },
            'W_PK': {
                'official_name': 'Wentworth Park',
                'codes': ['W_PK', 'w_pk', 'WPK', 'WENTWORTH_PARK'],
                'track_codes': ['W_PK', 'WPK'],
                'location': 'NSW',
                'active': True
            },
            'WAGGA': {
                'official_name': 'Wagga Wagga',
                'codes': ['WAGGA', 'wagga', 'WAGGA_WAGGA'],
                'track_codes': ['WAGGA'],
                'location': 'NSW',
                'active': True
            },

            # Victoria (VIC)
            'HEA': {
                'official_name': 'Healesville',
                'codes': ['HEA', 'hea', 'HEALESVILLE'],
                'track_codes': ['HEA'],
                'location': 'VIC',
                'active': True
            },
            'SAN': {
                'official_name': 'Sandown Park',
                'codes': ['SAN', 'san', 'SANDOWN'],
                'track_codes': ['SAN'],
                'location': 'VIC',
                'active': True
            },
            'MEA': {
                'official_name': 'The Meadows',
                'codes': ['MEA', 'mea', 'MEADOWS'],
                'track_codes': ['MEA'],
                'location': 'VIC',
                'active': True
            },
            'BAL': {
                'official_name': 'Ballarat',
                'codes': ['BAL', 'bal', 'BALLARAT'],
                'track_codes': ['BAL'],
                'location': 'VIC',
                'active': True
            },
            'BEN': {
                'official_name': 'Bendigo',
                'codes': ['BEN', 'ben', 'BENDIGO'],
                'track_codes': ['BEN'],
                'location': 'VIC',
                'active': True
            },
            'GEE': {
                'official_name': 'Geelong',
                'codes': ['GEE', 'gee', 'GEELONG'],
                'track_codes': ['GEE'],
                'location': 'VIC',
                'active': True
            },
            'HOR': {
                'official_name': 'Horsham',
                'codes': ['HOR', 'hor', 'HORSHAM'],
                'track_codes': ['HOR'],
                'location': 'VIC',
                'active': True
            },
            'SAL': {
                'official_name': 'Sale',
                'codes': ['SAL', 'sal', 'SALE'],
                'track_codes': ['SAL'],
                'location': 'VIC',
                'active': True
            },
            'SHEP': {
                'official_name': 'Shepparton',
                'codes': ['SHEP', 'shep', 'SHEPPARTON'],
                'track_codes': ['SHEP'],
                'location': 'VIC',
                'active': True
            },
            'TRA': {
                'official_name': 'Traralgon',
                'codes': ['TRA', 'tra', 'TRARALGON'],
                'track_codes': ['TRA'],
                'location': 'VIC',
                'active': True
            },
            'WARR': {
                'official_name': 'Warragul',
                'codes': ['WARR', 'warr', 'WARRAGUL'],
                'track_codes': ['WARR'],
                'location': 'VIC',
                'active': True
            },
            'WAR': {
                'official_name': 'Warrnambool',
                'codes': ['WAR', 'war', 'WARRNAMBOOL'],
                'track_codes': ['WAR'],
                'location': 'VIC',
                'active': True
            },

            # Queensland (QLD) - THE KEY MISSING MAPPING!
            'QLD_STRAIGHT': {
                'official_name': 'Queensland Straight Track',
                'codes': ['LADBROKES-Q-STRAIGHT', 'Q-STRAIGHT', 'QLD-STRAIGHT', 'QUEENSLAND-STRAIGHT'],
                'track_codes': ['QST', 'QOT', 'QTT'],  # These are the historical track codes!
                'location': 'QLD',
                'active': True,
                'notes': 'Lakeside/Q1 Straight track - maps to QST/QOT/QTT historical codes'
            },
            'APTH': {
                'official_name': 'Albion Park',
                'codes': ['APTH', 'apth', 'ALBION_PARK'],
                'track_codes': ['APTH'],
                'location': 'QLD',
                'active': True
            },
            'APWE': {
                'official_name': 'Albion Park Wheel',
                'codes': ['APWE', 'apwe', 'ALBION_PARK_WHEEL'],
                'track_codes': ['APWE'],
                'location': 'QLD',
                'active': True
            },
            'CAPA': {
                'official_name': 'Capalaba',
                'codes': ['CAPA', 'capa', 'CAPALABA'],
                'track_codes': ['CAPA'],
                'location': 'QLD',
                'active': True
            },
            'IPSWICH': {
                'official_name': 'Ipswich',
                'codes': ['IPSWICH', 'ipswich', 'IPSU', 'IPSA', 'IPTU'],
                'track_codes': ['IPSU', 'IPSA', 'IPTU'],
                'location': 'QLD',
                'active': True
            },
            'ROCK': {
                'official_name': 'Rockhampton',
                'codes': ['ROCK', 'rock', 'ROCKHAMPTON'],
                'track_codes': ['ROCK'],
                'location': 'QLD',
                'active': True
            },

            # South Australia (SA)
            'AP_K': {
                'official_name': 'Angle Park',
                'codes': ['AP_K', 'ap_k', 'ANGLE_PARK'],
                'track_codes': ['AP_K'],
                'location': 'SA',
                'active': True
            },
            'GAWL': {
                'official_name': 'Gawler',
                'codes': ['GAWL', 'gawl', 'GAWLER'],
                'track_codes': ['GAWL'],
                'location': 'SA',
                'active': True
            },
            'MOUNT': {
                'official_name': 'Mount Gambier',
                'codes': ['MOUNT', 'mount', 'MOUNT_GAMBIER'],
                'track_codes': ['MOUNT'],
                'location': 'SA',
                'active': True
            },
            'MURR': {
                'official_name': 'Murray Bridge',
                'codes': ['MURR', 'murr', 'MURRAY_BRIDGE'],
                'track_codes': ['MURR'],
                'location': 'SA',
                'active': True
            },
            'GRDN': {
                'official_name': 'The Gardens',
                'codes': ['GRDN', 'grdn', 'GARDENS'],
                'track_codes': ['GRDN'],
                'location': 'SA',
                'active': True
            },

            # Western Australia (WA)
            'CANN': {
                'official_name': 'Cannington',
                'codes': ['CANN', 'cann', 'CANNINGTON'],
                'track_codes': ['CANN'],
                'location': 'WA',
                'active': True
            },
            'MAND': {
                'official_name': 'Mandurah',
                'codes': ['MAND', 'mand', 'MANDURAH'],
                'track_codes': ['MAND'],
                'location': 'WA',
                'active': True
            },
            'NOR': {
                'official_name': 'Northam',
                'codes': ['NOR', 'nor', 'NORTHAM'],
                'track_codes': ['NOR'],
                'location': 'WA',
                'active': True
            },

            # Tasmania (TAS)
            'HOBT': {
                'official_name': 'Hobart',
                'codes': ['HOBT', 'hobt', 'HOBART'],
                'track_codes': ['HOBT'],
                'location': 'TAS',
                'active': True
            },
            'DEVONPORT': {
                'official_name': 'Devonport',
                'codes': ['DEVONPORT', 'devonport'],
                'track_codes': ['DEVONPORT'],
                'location': 'TAS',
                'active': True
            },
            'LAUNCESTON': {
                'official_name': 'Launceston',
                'codes': ['LAUNCESTON', 'launceston'],
                'track_codes': ['LAUNCESTON'],
                'location': 'TAS',
                'active': True
            },

            # Northern Territory (NT)
            'DARW': {
                'official_name': 'Darwin',
                'codes': ['DARW', 'darw', 'DARWIN'],
                'track_codes': ['DARW'],
                'location': 'NT',
                'active': True
            }
        }

        # Create reverse lookup maps for fast searching
        self._build_reverse_maps()

    def _build_reverse_maps(self):
        """Build reverse lookup maps for efficient searching"""
        self.code_to_venue = {}
        self.track_code_to_venue = {}
        
        for venue_key, venue_info in self.venue_mappings.items():
            # Map all codes to this venue
            for code in venue_info['codes']:
                self.code_to_venue[code.upper()] = venue_key
                self.code_to_venue[code.lower()] = venue_key
            
            # Map all track codes to this venue
            for track_code in venue_info['track_codes']:
                self.track_code_to_venue[track_code.upper()] = venue_key
                self.track_code_to_venue[track_code.lower()] = venue_key

    def resolve_venue(self, venue_input: str) -> Optional[str]:
        """
        Resolve any venue input to standardized venue key
        
        Args:
            venue_input: Any venue name, code, or track code
            
        Returns:
            Standardized venue key or None if not found
        """
        if not venue_input:
            return None
            
        venue_clean = venue_input.strip()
        
        # Try direct code lookup
        if venue_clean.upper() in self.code_to_venue:
            return self.code_to_venue[venue_clean.upper()]
        
        # Try track code lookup
        if venue_clean.upper() in self.track_code_to_venue:
            return self.track_code_to_venue[venue_clean.upper()]
        
        # Try partial matching for complex names like "LADBROKES-Q-STRAIGHT"
        for venue_key, venue_info in self.venue_mappings.items():
            for code in venue_info['codes']:
                if code.upper() in venue_clean.upper() or venue_clean.upper() in code.upper():
                    return venue_key
        
        return None

    def get_track_codes_for_venue(self, venue_key: str) -> List[str]:
        """Get all historical track codes for a venue"""
        if venue_key in self.venue_mappings:
            return self.venue_mappings[venue_key]['track_codes']
        return []

    def get_venue_info(self, venue_key: str) -> Optional[Dict]:
        """Get complete venue information"""
        return self.venue_mappings.get(venue_key)

    def fix_database_venue_mappings(self, db_path: str = "greyhound_racing_data.db"):
        """
        Fix database venue mappings to use standardized codes
        """
        print("🔧 Fixing database venue mappings...")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check current venue distribution in race_metadata
            cursor.execute("SELECT DISTINCT venue, COUNT(*) FROM race_metadata GROUP BY venue")
            current_venues = cursor.fetchall()
            
            print(f"📊 Current venues in database:")
            for venue, count in current_venues:
                print(f"   {venue}: {count} races")
            
            # Check for QLD track codes in dog_race_data historical records
            cursor.execute("""
                SELECT DISTINCT historical_records 
                FROM dog_race_data 
                WHERE historical_records IS NOT NULL 
                AND historical_records LIKE '%QST%' 
                OR historical_records LIKE '%QOT%' 
                OR historical_records LIKE '%QTT%'
                LIMIT 10
            """)
            qld_records = cursor.fetchall()
            
            print(f"🔍 Found {len(qld_records)} records with QLD track codes")
            
            # Update venue mappings in a new table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS venue_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_key TEXT NOT NULL,
                    official_name TEXT NOT NULL,
                    venue_codes TEXT NOT NULL,
                    track_codes TEXT NOT NULL,
                    location TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing mappings
            cursor.execute("DELETE FROM venue_mappings")
            
            # Insert all venue mappings
            for venue_key, venue_info in self.venue_mappings.items():
                cursor.execute("""
                    INSERT INTO venue_mappings 
                    (venue_key, official_name, venue_codes, track_codes, location, active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    venue_key,
                    venue_info['official_name'],
                    json.dumps(venue_info['codes']),
                    json.dumps(venue_info['track_codes']),
                    venue_info['location'],
                    venue_info['active']
                ))
            
            conn.commit()
            print("✅ Venue mappings table created and populated")
            
            # Create function to help with historical data lookups
            self._create_venue_lookup_functions(cursor)
            
            conn.commit()
            conn.close()
            
            print("🎯 Database venue mapping fix completed!")
            
        except Exception as e:
            print(f"❌ Error fixing database venue mappings: {e}")

    def _create_venue_lookup_functions(self, cursor):
        """Create helper views for venue lookups"""
        
        # Create a view for easy venue resolution
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS venue_resolver AS
            SELECT 
                venue_key,
                official_name,
                venue_codes,
                track_codes,
                location
            FROM venue_mappings
            WHERE active = 1
        """)
        
        print("✅ Created venue_resolver view")

    def test_venue_resolution(self):
        """Test venue resolution with common inputs"""
        test_cases = [
            "LADBROKES-Q-STRAIGHT",
            "QST",
            "QOT", 
            "QTT",
            "AP_K",
            "RICH",
            "BAL",
            "CAPA",
            "DAPT",
            "Unknown_Venue"
        ]
        
        print("🧪 Testing venue resolution:")
        print("=" * 50)
        
        for test_input in test_cases:
            resolved = self.resolve_venue(test_input)
            if resolved:
                venue_info = self.get_venue_info(resolved)
                track_codes = ", ".join(venue_info['track_codes'])
                print(f"✅ {test_input:20} → {resolved:15} ({venue_info['official_name']}, tracks: {track_codes})")
            else:
                print(f"❌ {test_input:20} → Not found")

def main():
    """Main function to fix venue mapping issues"""
    print("🚀 Greyhound Venue Mapping Fix")
    print("=" * 50)
    
    # Initialize the mapper
    mapper = GreyhoundVenueMapper()
    
    # Test venue resolution
    mapper.test_venue_resolution()
    
    # Fix database mappings
    mapper.fix_database_venue_mappings()
    
    print("\n🎯 Key Fix: LADBROKES-Q-STRAIGHT now maps to QLD_STRAIGHT")
    print("   Historical track codes: QST, QOT, QTT")
    print("   This should resolve the 'No historical data available' issue!")

if __name__ == "__main__":
    main()
